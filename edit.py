import os
import torch.nn.functional as F
import torch
import sys
import cv2
import numpy as np
from PIL import Image
import json

import logging
class LoraWarningFilter(logging.Filter):
    def filter(self, record):
        return "No LoRA keys associated to" not in record.getMessage()

logging.getLogger("diffusers.loaders.peft").addFilter(LoraWarningFilter())
logging.getLogger("diffusers.loaders.lora_base").addFilter(LoraWarningFilter())

import tempfile
from safetensors.torch import load_file, save_file

# Make `src` and `train` importable regardless of the current working directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..', 'comfy_extras')))

# MagicQuill V2's own EasyControl engine (custom pipeline + transformer).
from src.pipeline_flux_kontext_control import FluxKontextControlPipeline
from src.transformer_flux import FluxTransformer2DModel
from diffusers import AutoencoderKL, GGUFQuantizationConfig

# Allow the GGUF single-file loader to recognise the custom transformer class
# (it shares the name "FluxTransformer2DModel" with the diffusers class).
import diffusers.loaders.single_file_model as sfm
_orig_get_single_file_loadable_mapping_class = sfm._get_single_file_loadable_mapping_class
def _patched_get_single_file_loadable_mapping_class(cls):
    if cls.__name__ == "FluxTransformer2DModel":
        return "FluxTransformer2DModel"
    return _orig_get_single_file_loadable_mapping_class(cls)
sfm._get_single_file_loadable_mapping_class = _patched_get_single_file_loadable_mapping_class
print("[OK] Monkey patch applied to diffusers._get_single_file_loadable_mapping_class")

# Auto-convert mixed-format LoRA checkpoints (PEFT -> diffusers) and add the
# `transformer.` prefix so the aux puzzle LoRA loads onto the custom pipeline.
_original_load_lora_weights = FluxKontextControlPipeline.load_lora_weights

def _patched_load_lora_weights(self, pretrained_model_name_or_path_or_dict, **kwargs):
    weight_name = kwargs.get("weight_name", "pytorch_lora_weights.safetensors")

    if isinstance(pretrained_model_name_or_path_or_dict, str):
        if os.path.isdir(pretrained_model_name_or_path_or_dict):
            lora_file = os.path.join(pretrained_model_name_or_path_or_dict, weight_name)
        else:
            lora_file = pretrained_model_name_or_path_or_dict

        if os.path.exists(lora_file):
            state_dict = load_file(lora_file)

            needs_format_conversion = any('lora_A.weight' in k or 'lora_B.weight' in k for k in state_dict.keys())
            needs_prefix = not any(k.startswith('transformer.') for k in state_dict.keys())

            if needs_format_conversion or needs_prefix:
                print(f"[LoRA] Processing LoRA: {lora_file}")
                converted_state = {}
                for key, value in state_dict.items():
                    new_key = key
                    if 'lora_A.weight' in new_key:
                        new_key = new_key.replace('lora_A.weight', 'lora.down.weight')
                    elif 'lora_B.weight' in new_key:
                        new_key = new_key.replace('lora_B.weight', 'lora.up.weight')
                    if not new_key.startswith('transformer.'):
                        new_key = f'transformer.{new_key}'
                    converted_state[new_key] = value

                print(f"   [OK] Total keys: {len(converted_state)}")
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_file = os.path.join(temp_dir, weight_name)
                    save_file(converted_state, temp_file)
                    return _original_load_lora_weights(self, temp_dir, **kwargs)
            else:
                print(f"[OK] LoRA already in correct format: {lora_file}")

    return _original_load_lora_weights(self, pretrained_model_name_or_path_or_dict, **kwargs)

FluxKontextControlPipeline.load_lora_weights = _patched_load_lora_weights
print("[OK] Monkey patch applied to FluxKontextControlPipeline.load_lora_weights")

from train.src.condition.edge_extraction import InformativeDetector, HEDDetector
from utils_node import BlendInpaint, JoinImageWithAlpha, GrowMask, InvertMask, ColorDetector
from segment_anything import sam_model_registry, SamPredictor

TEST_MODE = False


class KontextEditModel():
    """
    MagicQuill V2 editing engine, loaded for low (≈12 GB) VRAM:
      - transformer: GGUF-quantised FLUX.1-Kontext-dev (resident on GPU)
      - text_encoder (CLIP-L): GPU
      - text_encoder_2 (T5-XXL): FP8, kept on CPU
      - vae: offloaded to CPU between encode/decode by the pipeline
      - EasyControl task LoRAs (edge / color / local / removal) + aux puzzle LoRA
    """

    def __init__(self, base_model_path="HelloTestUser/FLUX.1-Kontext-dev", device="cuda",
                 aux_lora_dir=None, easycontrol_base_dir=None,
                 aux_lora_weight_name="puzzle_lora.safetensors",
                 aux_lora_weight=1.0):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if aux_lora_dir is None:
            aux_lora_dir = os.path.join(current_dir, "models", "v2_ckpt")
        if easycontrol_base_dir is None:
            easycontrol_base_dir = os.path.join(current_dir, "models", "v2_ckpt")

        # Preprocessors (used to build the spatial control conditions).
        self.mask_processor = GrowMask()
        self.scribble_processor = HEDDetector.from_pretrained()
        self.lineart_processor = InformativeDetector.from_pretrained()
        self.color_processor = ColorDetector()
        self.blender = BlendInpaint()

        self.device = device

        import gc
        from transformers import CLIPTextModel, CLIPTextConfig, T5EncoderModel, T5Config
        from optimum.quanto import quantize, qfloat8, freeze

        # 1. CLIP text encoder (~246MB) on GPU.
        print("Loading CLIP text encoder...")
        clip_config = CLIPTextConfig.from_pretrained(base_model_path, subfolder="text_encoder", token=False)
        text_encoder = CLIPTextModel(clip_config).to(device="cpu", dtype=torch.bfloat16)
        clip_path = os.path.join(current_dir, "models", "v2_ckpt", "split_files", "text_encoders", "clip_l.safetensors")
        clip_state_dict = load_file(clip_path)

        def load_text_encoder_state_dict(model, state_dict):
            model_keys = set(model.state_dict().keys())
            if set(state_dict.keys()) == model_keys:
                model.load_state_dict(state_dict)
                return
            new_sd = {}
            for k, v in state_dict.items():
                new_key = k
                if not k.startswith("text_model.") and "text_model." + k in model_keys:
                    new_key = "text_model." + k
                elif k.startswith("cond_stage_model.transformer."):
                    suffix = k.replace("cond_stage_model.transformer.", "")
                    new_key = suffix if suffix in model_keys else ("text_model." + suffix)
                elif k.startswith("transformer."):
                    suffix = k.replace("transformer.", "")
                    new_key = suffix if suffix in model_keys else ("text_model." + suffix)
                new_sd[new_key] = v
            model.load_state_dict(new_sd, strict=False)

        load_text_encoder_state_dict(text_encoder, clip_state_dict)
        text_encoder = text_encoder.to(device=self.device, dtype=torch.bfloat16)
        del clip_state_dict
        gc.collect()
        print("CLIP text encoder loaded on GPU.")

        # 2. T5 text encoder (~4.9GB) on CPU, quantised to FP8 to save RAM.
        print("Loading T5 text encoder...")
        t5_config = T5Config.from_pretrained(base_model_path, subfolder="text_encoder_2", token=False)
        text_encoder_2 = T5EncoderModel(t5_config).to(device="cpu", dtype=torch.bfloat16)
        t5_path = os.path.join(current_dir, "models", "v2_ckpt", "split_files", "text_encoders", "t5xxl_fp8_e4m3fn_scaled.safetensors")
        t5_state_dict = load_file(t5_path)

        def load_t5_state_dict(model, state_dict):
            model_keys = set(model.state_dict().keys())
            new_sd = {}
            for k, v in state_dict.items():
                if v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    v = v.to(torch.bfloat16)
                new_key = k
                if not k.startswith("encoder.") and "encoder." + k in model_keys:
                    new_key = "encoder." + k
                elif k.startswith("cond_stage_model.transformer."):
                    suffix = k.replace("cond_stage_model.transformer.", "")
                    new_key = suffix if suffix in model_keys else ("encoder." + suffix)
                elif k.startswith("transformer."):
                    suffix = k.replace("transformer.", "")
                    new_key = suffix if suffix in model_keys else ("encoder." + suffix)
                new_sd[new_key] = v
            model.load_state_dict(new_sd, strict=False)

        load_t5_state_dict(text_encoder_2, t5_state_dict)
        del t5_state_dict
        gc.collect()
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("T5 text encoder loaded (FP8) on CPU.")

        # 3. VAE (~335MB). Loaded on CPU; the pipeline moves it to GPU only for
        #    encode/decode and back, to keep VRAM free during denoising.
        vae_path = os.path.join(current_dir, "models", "v2_ckpt", "split_files", "vae", "ae.safetensors")
        vae = AutoencoderKL.from_single_file(
            vae_path, config=base_model_path, subfolder="vae", torch_dtype=torch.bfloat16, token=False
        ).to(device="cpu", dtype=torch.bfloat16)
        print("VAE loaded on CPU (offloaded between encode/decode).")

        # 4. GGUF transformer (~8.4GB) on GPU, loaded into the custom class so the
        #    EasyControl attention processors work.
        transformer_path = os.path.join(
            current_dir, "models", "v2_ckpt", "split_files", "diffusion_models", "flux1-kontext-dev-Q5_K_M.gguf"
        )
        print(f"Loading GGUF transformer from: {transformer_path}")
        transformer = FluxTransformer2DModel.from_single_file(
            transformer_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
            config=base_model_path,
            subfolder="transformer",
        ).to(device)
        print("GGUF transformer loaded on GPU.")

        # 5. Assemble the EasyControl pipeline from the components above
        #    (scheduler + tokenizers come from the base repo).
        self.pipe = FluxKontextControlPipeline.from_pretrained(
            base_model_path,
            transformer=transformer,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            vae=vae,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            token=False,
        )
        self.pipe.vae.to("cpu")
        try:
            self.pipe.vae.enable_tiling()
        except Exception as e:
            print(f"Failed to enable VAE tiling: {e}")

        # Cap the working resolution so the ~1MP Kontext double-sequence + GGUF
        # transformer fit in ~12GB VRAM (otherwise Windows spills into shared RAM
        # over PCIe -> hundreds of seconds per step). Override with the
        # MAGICQUILL_MAX_SIDE env var (e.g. 768 for tighter VRAM, 1280 for more).
        self.pipe.max_working_side = int(os.environ.get("MAGICQUILL_MAX_SIDE", "1024"))
        print(f"[Speed] max_working_side = {self.pipe.max_working_side} px")

        # Smaller control condition -> fewer cached cond tokens (less VRAM), still
        # enough to steer the brush. Override with MAGICQUILL_COND_SIZE.
        cond_size = int(os.environ.get("MAGICQUILL_COND_SIZE", "256"))

        # 6. EasyControl task LoRAs — this is what makes brush + prompt edits
        #    actually follow the strokes.
        control_lora_config = {
            "local":   {"path": os.path.join(easycontrol_base_dir, "local_lora.safetensors"),   "lora_weights": [1.0], "cond_size": cond_size},
            "removal": {"path": os.path.join(easycontrol_base_dir, "removal_lora.safetensors"), "lora_weights": [1.0], "cond_size": cond_size},
            "edge":    {"path": os.path.join(easycontrol_base_dir, "edge_lora.safetensors"),    "lora_weights": [1.0], "cond_size": cond_size},
            "color":   {"path": os.path.join(easycontrol_base_dir, "color_lora.safetensors"),   "lora_weights": [1.0], "cond_size": cond_size},
        }
        self.pipe.load_control_loras(control_lora_config)
        print("[OK] EasyControl task LoRAs loaded.")

        # 7. Aux puzzle LoRA for foreground mode (PEFT adapter, optional).
        self.aux_lora_weight_name = aux_lora_weight_name
        self.aux_lora_dir = aux_lora_dir
        self.aux_lora_weight = aux_lora_weight
        self.aux_adapter_name = "aux"
        self._aux_lora_available = False
        aux_path = os.path.join(self.aux_lora_dir, self.aux_lora_weight_name)
        if os.path.isfile(aux_path):
            try:
                self.pipe.load_lora_weights(aux_path, adapter_name=self.aux_adapter_name)
                self._aux_lora_available = True
                self._disable_aux_lora()
                print(f"Loaded aux LoRA: {aux_path}")
            except Exception as e:
                print(f"[WARN] Could not load aux LoRA ({e}); foreground mode will run without it.")
        else:
            print(f"Aux LoRA not found at {aux_path}, foreground mode will run without it.")

    def _tensor_to_pil(self, tensor_image):
        return Image.fromarray(np.clip(255. * tensor_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    def _pil_to_tensor(self, pil_image):
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

    def _debug_dump(self, name, tensor):
        # Save a tensor (mask or image) to ./debug_masks for inspection.
        # Enabled only when MAGICQUILL_DEBUG_MASKS is set, so normal runs are unaffected.
        if not os.environ.get("MAGICQUILL_DEBUG_MASKS"):
            return
        try:
            out_dir = os.environ.get("MAGICQUILL_DEBUG_DIR", "debug_masks")
            os.makedirs(out_dir, exist_ok=True)
            arr = tensor.detach().float().cpu().numpy()
            arr = np.squeeze(arr)
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(out_dir, f"{name}.png"))
            lo, hi = float(tensor.min()), float(tensor.max())
            frac_lt = float((tensor < 0.5).float().mean())
            print(f"[DEBUG] dumped {name}: shape={tuple(tensor.shape)} min={lo:.3f} max={hi:.3f} frac(<0.5)={frac_lt:.3f}")
        except Exception as e:
            print(f"[DEBUG] failed to dump {name}: {e}")

    def _ensure_channels_last(self, image_tensor: torch.Tensor, name: str = "image") -> torch.Tensor:
        # Normalize image tensors to (1, H, W, C) for all blend operations.
        if image_tensor.ndim != 4:
            raise ValueError(f"{name} must be 4D, got shape {tuple(image_tensor.shape)}")
        if image_tensor.shape[-1] in [1, 3, 4]:
            return image_tensor
        if image_tensor.shape[1] in [1, 3, 4]:
            return image_tensor.permute(0, 2, 3, 1).contiguous()
        raise ValueError(f"{name} has unsupported channel layout: {tuple(image_tensor.shape)}")

    def _mask_to_blend_tensor(self, mask_tensor: torch.Tensor, blur: bool = True) -> torch.Tensor:
        # Convert any mask layout to channels-last blend weights (1, H, W, 1).
        mask = mask_tensor.detach().float()
        if mask.ndim == 4:
            mask = mask[0, 0] if mask.shape[1] == 1 else mask[0].max(dim=0).values
        elif mask.ndim == 3:
            mask = mask[0]
        mask_np = (mask.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
        if blur:
            mask_np = cv2.GaussianBlur(mask_np, (11, 11), 3.0)
        return torch.from_numpy(mask_np / 255.0).float().unsqueeze(0).unsqueeze(-1)

    def _composite_preserve_mask(self, final_image, original_image, preserve_mask, label="regions"):
        if preserve_mask is None:
            return final_image
        if torch.sum(preserve_mask > 0.5).item() == 0:
            return final_image
        try:
            blend = self._mask_to_blend_tensor(preserve_mask)
            original_blend = self._ensure_channels_last(original_image, "original_image")
            final_image = self._ensure_channels_last(final_image, "final_image")
            return final_image * (1.0 - blend) + original_blend * blend
        except Exception as e:
            print(f"[WARN] Preserve composite failed ({label}): {e}")
            return final_image

    def clear_cache(self):
        for name, attn_processor in self.pipe.transformer.attn_processors.items():
            if hasattr(attn_processor, 'bank_kv'):
                attn_processor.bank_kv.clear()
            if hasattr(attn_processor, 'bank_attn'):
                attn_processor.bank_attn = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _enable_aux_lora(self):
        if not self._aux_lora_available:
            print("[Foreground] aux LoRA unavailable; running without it.")
            return
        self.pipe.enable_lora()
        self.pipe.set_adapters([self.aux_adapter_name], adapter_weights=[self.aux_lora_weight])
        print(f"Enabled aux LoRA '{self.aux_adapter_name}' with weight {self.aux_lora_weight}")

    def _disable_aux_lora(self):
        if not self._aux_lora_available:
            return
        self.pipe.disable_lora()
        print("Disabled aux LoRA")

    def _expand_mask(self, mask_tensor: torch.Tensor, expand: int = 0) -> torch.Tensor:
        if expand <= 0:
            return mask_tensor
        expanded = self.mask_processor.expand_mask(mask_tensor, expand=expand, tapered_corners=True)[0]
        return expanded

    def _tensor_mask_to_pil3(self, mask_tensor: torch.Tensor) -> Image.Image:
        mask_01 = torch.clamp(mask_tensor, 0.0, 1.0)
        if mask_01.ndim == 3 and mask_01.shape[-1] == 3:
            mask_01 = mask_01[..., 0]
        if mask_01.ndim == 3 and mask_01.shape[0] == 1:
            mask_01 = mask_01[0]
        pil = self._tensor_to_pil(mask_01.unsqueeze(-1).repeat(1, 1, 3))
        return pil

    def _apply_black_mask(self, image_tensor: torch.Tensor, binary_mask: torch.Tensor) -> Image.Image:
        # image_tensor: [1, H, W, 3] in [0,1]; binary_mask: [H, W] or [1, H, W], 1 = region to edit
        if binary_mask.ndim == 3:
            binary_mask = binary_mask[0]
        mask_bool = (binary_mask > 0.5)
        img = image_tensor.clone()
        img[0][mask_bool] = 0.0
        return self._tensor_to_pil(img)

    def edge_edit(self,
                image, colored_image, positive_prompt,
                base_mask, add_mask, remove_mask,
                fine_edge,
                edge_strength, color_strength,
                seed, steps, cfg):
        generator = torch.Generator(device=self.device).manual_seed(seed)

        original_image_tensor = image.clone()
        
        # CRITICAL FIX: Preserve matted objects when painting with brush
        # base_mask (total_mask) may contain BOTH matted objects AND brush strokes
        # add_mask, remove_mask are the paint brush stroke regions (LOW = stroke)
        # We need to:
        # 1. Extract ONLY the brush stroke regions (not entire base_mask)
        # 2. Preserve matted objects outside brush regions
        # 3. Regenerate only the brush stroke areas
        
        # Identify brush stroke regions (where add_mask or remove_mask have strokes)
        has_add_stroke = torch.sum(add_mask < 0.5).item() > 0
        has_remove_stroke = torch.sum(remove_mask < 0.5).item() > 0
        
        if has_add_stroke or has_remove_stroke:
            # Brush strokes exist: create a mask for ONLY brush stroke regions
            # Exclude the base_mask (matted object) regions to preserve them
            brush_only = torch.ones_like(base_mask)
            if has_add_stroke:
                brush_only = torch.minimum(brush_only, add_mask)
            if has_remove_stroke:
                brush_only = torch.minimum(brush_only, remove_mask)
            # brush_only now has LOW (<0.5) only where there are brush strokes
            
            # The edit region is the brush strokes, not the entire base_mask
            original_mask = self._expand_mask((brush_only < 0.5).float(), expand=25)
            
            # Preserve only matted/base regions that are NOT brush strokes.
            base_active = (base_mask < 0.5).float()
            brush_active = (brush_only < 0.5).float()
            matted_only = torch.clamp(base_active - brush_active, 0.0, 1.0)
            matted_object_mask = self._expand_mask(matted_only, expand=10)
        else:
            # No brush strokes, use base_mask as before
            original_mask = self._expand_mask((base_mask < 0.5).float(), expand=25)
            matted_object_mask = self._expand_mask((base_mask < 0.5).float(), expand=10)

        image_pil = self._tensor_to_pil(image)
        control_dict = {}
        lineart_output = None

        # Color brush -> color-block condition; otherwise edge/scribble condition.
        # Require a meaningful color-layer delta (not just merged-vs-background differences).
        color_delta = (image - colored_image).abs().max().item() if image.shape == colored_image.shape else 1.0
        use_color_control = color_delta > 1e-3 and not torch.equal(image, colored_image)
        if use_color_control:
            print("Apply color control")
            colored_image_pil = self._tensor_to_pil(colored_image)
            color_image_np = np.array(colored_image_pil)
            downsampled = cv2.resize(color_image_np, (32, 32), interpolation=cv2.INTER_AREA)
            upsampled = cv2.resize(downsampled, (256, 256), interpolation=cv2.INTER_NEAREST)
            color_block = Image.fromarray(upsampled)
            control_dict = {
                "type": "color",
                "spatial_images": [color_block],
                "gammas": [color_strength],
            }
        else:
            print("Apply edge control")
            if fine_edge == "enable":
                lineart_image = self.lineart_processor(np.array(self._tensor_to_pil(image.cpu().squeeze())), detect_resolution=1024, style="contour", output_type="pil")
                lineart_output = self._pil_to_tensor(lineart_image)
            else:
                scribble_image = self.scribble_processor(np.array(self._tensor_to_pil(image.cpu().squeeze())), safe=True, resolution=512, output_type="pil")
                lineart_output = self._pil_to_tensor(scribble_image)

            if lineart_output is None:
                raise ValueError("Preprocessor failed to generate lineart.")

            # Burn the user's brush strokes into the lineart (strokes are LOW).
            add_mask_resized = F.interpolate(add_mask.unsqueeze(0).float(), size=(lineart_output.shape[1], lineart_output.shape[2]), mode='nearest').squeeze(0)
            remove_mask_resized = F.interpolate(remove_mask.unsqueeze(0).float(), size=(lineart_output.shape[1], lineart_output.shape[2]), mode='nearest').squeeze(0)
            bool_add_stroke = (add_mask_resized < 0.5)
            bool_remove_stroke = (remove_mask_resized < 0.5)
            lineart_output[bool_remove_stroke] = 0.0
            lineart_output[bool_add_stroke] = 1.0

            control_dict = {
                "type": "edge",
                "spatial_images": [self._tensor_to_pil(lineart_output)],
                "gammas": [edge_strength],
            }

        colored_image_np = np.array(self._tensor_to_pil(colored_image))
        debug_image = lineart_output if lineart_output is not None else self.color_processor(colored_image_np, detect_resolution=1024, output_type="pil")

        result_pil = self.pipe(
            prompt=positive_prompt,
            image=image_pil,
            height=self.pipe.fit_kontext_resolution(image_pil)[1],
            width=self.pipe.fit_kontext_resolution(image_pil)[0],
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=generator,
            max_sequence_length=512,
            control_dict=control_dict,
        ).images[0]
        self.clear_cache()

        final_image = self._pil_to_tensor(result_pil)
        
        # CRITICAL FIX: Composite with matted objects to preserve them
        # If we have matted objects, blend the generated result back using the matted object mask
        # This ensures matted objects are restored where they should be preserved
        if matted_object_mask is not None and torch.sum(matted_object_mask > 0.5).item() > 0:
            final_image = self._composite_preserve_mask(
                final_image, original_image_tensor, matted_object_mask, "matted objects"
            )
            print("[OK] Matted objects preserved with smooth blending")
        
        return (final_image, debug_image, original_mask)

    def object_removal(self,
                       image, positive_prompt,
                       remove_mask,
                       local_strength,
                       seed, steps, cfg):
        generator = torch.Generator(device=self.device).manual_seed(seed)

        original_image_tensor = image.clone()
        original_mask = self._expand_mask((remove_mask < 0.5).float(), expand=10)

        image_pil = self._tensor_to_pil(image)
        spatial_pil = self._apply_black_mask(image, original_mask)
        control_dict = {
            "type": "removal",
            "spatial_images": [spatial_pil],
            "gammas": [local_strength],
        }

        result_pil = self.pipe(
            prompt=positive_prompt,
            image=image_pil,
            height=self.pipe.fit_kontext_resolution(image_pil)[1],
            width=self.pipe.fit_kontext_resolution(image_pil)[0],
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=generator,
            max_sequence_length=512,
            control_dict=control_dict,
        ).images[0]
        self.clear_cache()

        final_image = self._pil_to_tensor(result_pil)
        
        # Apply smooth blending to preserve boundary quality
        # Blur the mask for anti-aliasing at the boundary between generated and original
        try:
            original_mask_pil = self._tensor_to_pil(original_mask)
            original_mask_np = np.array(original_mask_pil)
            if original_mask_np.ndim == 3:  # Handle RGB/RGBA
                original_mask_np = original_mask_np[:, :, 0]
            blurred_mask = cv2.GaussianBlur(original_mask_np, (11, 11), 3.0)
            # _pil_to_tensor returns (1, H, W, C), so use (1, H, W, 1) for blend mask
            blend_mask_smooth = torch.from_numpy(blurred_mask / 255.0).float().unsqueeze(0).unsqueeze(-1)
            
            # Ensure original image is channels-last for blending
            original_blend = self._ensure_channels_last(original_image_tensor, "original_image_tensor")
            
            # original_mask marks the edited region (high inside edited area),
            # so keep generated pixels where mask is high, preserve original outside.
            final_image = final_image * blend_mask_smooth + original_blend * (1.0 - blend_mask_smooth)
            print("[OK] Object removal with smooth blending")
        except Exception as e:
            print(f"[WARN] Smooth blending failed ({e}), using generated result directly")
        
        return (final_image, self._pil_to_tensor(spatial_pil), original_mask)

    def local_edit(self,
                   image, positive_prompt, fill_mask, local_strength,
                   seed, steps, cfg, preserve_mask=None):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        original_image_tensor = image.clone()
        original_mask = self._expand_mask((fill_mask < 0.5).float(), expand=10)
        image_pil = self._tensor_to_pil(image)

        spatial_pil = self._apply_black_mask(image, original_mask)
        control_dict = {
            "type": "local",
            "spatial_images": [spatial_pil],
            "gammas": [local_strength],
        }

        result_pil = self.pipe(
            prompt=positive_prompt,
            image=image_pil,
            height=self.pipe.fit_kontext_resolution(image_pil)[1],
            width=self.pipe.fit_kontext_resolution(image_pil)[0],
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=generator,
            max_sequence_length=512,
            control_dict=control_dict,
        ).images[0]
        self.clear_cache()

        final_image = self._pil_to_tensor(result_pil)
        
        # Apply smooth blending to preserve boundary quality
        try:
            original_mask_pil = self._tensor_to_pil(original_mask)
            original_mask_np = np.array(original_mask_pil)
            if original_mask_np.ndim == 3:  # Handle RGB/RGBA
                original_mask_np = original_mask_np[:, :, 0]
            blurred_mask = cv2.GaussianBlur(original_mask_np, (11, 11), 3.0)
            # _pil_to_tensor returns (1, H, W, C), so use (1, H, W, 1) for blend mask
            blend_mask_smooth = torch.from_numpy(blurred_mask / 255.0).float().unsqueeze(0).unsqueeze(-1)
            
            # Ensure original image is channels-last for blending
            original_blend = self._ensure_channels_last(original_image_tensor, "original_image_tensor")
            
            # original_mask marks the edited region (high inside edited area),
            # so keep generated pixels where mask is high, preserve original outside.
            final_image = final_image * blend_mask_smooth + original_blend * (1.0 - blend_mask_smooth)
            print("[OK] Local edit with smooth blending")
        except Exception as e:
            print(f"[WARN] Smooth blending failed ({e}), using generated result directly")

        if preserve_mask is not None:
            final_image = self._composite_preserve_mask(
                final_image, original_image_tensor, preserve_mask, "matted objects"
            )
        
        return (final_image, self._pil_to_tensor(spatial_pil), original_mask)

    def foreground_edit(self,
                        merged_image, positive_prompt,
                        add_prop_mask, fill_mask, total_mask, fix_perspective, grow_size,
                        seed, steps, cfg):
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # app.py no longer prepends the foreground instruction, so build it here.
        positive_prompt = (
            "Fill in the white region naturally and adapt the foreground into the background. "
            "Preserve existing matted/pasted characters and objects outside the edited brush region. "
            + positive_prompt
        )
        if fix_perspective == "enable":
            positive_prompt = positive_prompt + " Fix the perspective if necessary."

        merged_original = self._ensure_channels_last(merged_image.clone(), "merged_image")

        # Convert brush masks to stroke-high so we can reuse the original logic.
        prop_hi = (add_prop_mask < 0.5).float()
        fill_hi = (fill_mask < 0.5).float()
        has_fill_stroke = torch.sum(fill_hi > 0.5).item() > 0

        self._debug_dump("01_add_prop_mask", add_prop_mask)
        self._debug_dump("02_fill_mask", fill_mask)
        self._debug_dump("03_total_mask", total_mask)
        self._debug_dump("04_prop_hi", prop_hi)
        self._debug_dump("05_merged_input", merged_original)

        if has_fill_stroke:
            # Magic-quill fill brush: only regenerate the brushed fill region.
            edit_mask = self._expand_mask(fill_hi, expand=25)
            white_region = fill_hi > 0.5
        else:
            # Prop integration: white halo around pasted props for backdrop fill.
            edit_mask = torch.clamp(self._expand_mask(prop_hi, expand=grow_size), 0.0, 1.0)
            white_region = ((prop_hi <= 0.5) & (edit_mask > 0.5)).squeeze(0)

        # Paint white only in regions that should be regenerated (never on matted props).
        img = merged_original.clone()
        if white_region.ndim == 3:
            white_region = white_region.squeeze(0)
        white_3 = white_region.unsqueeze(-1).expand(-1, -1, img.shape[-1])
        img[0] = torch.where(white_3, torch.ones_like(img[0]), img[0])

        self._debug_dump("06_edit_mask", edit_mask)
        self._debug_dump("07_white_region", white_region.float())
        self._debug_dump("08_whited_input", img)

        image_pil = self._tensor_to_pil(img)

        self._enable_aux_lora()
        try:
            result_pil = self.pipe(
                prompt=positive_prompt,
                image=image_pil,
                height=self.pipe.fit_kontext_resolution(image_pil)[1],
                width=self.pipe.fit_kontext_resolution(image_pil)[0],
                guidance_scale=cfg,
                num_inference_steps=steps,
                generator=generator,
                max_sequence_length=512,
                control_dict=None,
            ).images[0]
        finally:
            self._disable_aux_lora()
        self.clear_cache()

        final_image = self._pil_to_tensor(result_pil)
        
        # Blend generated content into edit region, then hard-lock matted props back in.
        try:
            edit_blend = self._mask_to_blend_tensor(edit_mask)
            final_image = self._ensure_channels_last(final_image, "final_image")
            final_image = final_image * edit_blend + merged_original * (1.0 - edit_blend)
            final_image = self._composite_preserve_mask(
                final_image, merged_original, self._expand_mask(prop_hi, expand=3), "matted props"
            )
            print("[OK] Foreground edit with matted preservation")
        except Exception as e:
            print(f"[WARN] Smooth blending failed ({e}), using generated result directly")
            final_image = self._composite_preserve_mask(
                final_image, merged_original, prop_hi, "matted props (fallback)"
            )
        
        return (final_image, self._pil_to_tensor(image_pil), edit_mask)

    def kontext_edit(self,
                     image, positive_prompt,
                     seed, steps, cfg):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        image_pil = self._tensor_to_pil(image)

        result_pil = self.pipe(
            prompt=positive_prompt,
            image=image_pil,
            height=self.pipe.fit_kontext_resolution(image_pil)[1],
            width=self.pipe.fit_kontext_resolution(image_pil)[0],
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=generator,
            max_sequence_length=512,
            control_dict=None,
        ).images[0]
        self.clear_cache()

        final_image = self._pil_to_tensor(result_pil)
        mask = torch.zeros((1, final_image.shape[1], final_image.shape[2]), dtype=torch.float32, device=final_image.device)
        return (final_image, image, mask)

    def process(self, image, colored_image,
                 merged_image, positive_prompt,
                total_mask, add_mask, remove_mask, add_prop_mask, fill_mask,
                fine_edge, fix_perspective, edge_strength, color_strength, local_strength, grow_size,
                seed, steps, cfg, flag="precise_edit"):
        # Force num_inference_steps to a hardcoded range of 17-18 steps
        steps = min(max(steps, 17), 18)
        if flag == "foreground":
            return self.foreground_edit(merged_image, positive_prompt, add_prop_mask, fill_mask, total_mask, fix_perspective, grow_size, seed, steps, cfg)
        elif flag == "local":
            preserve_mask = None
            if torch.sum(add_prop_mask < 0.5).item() > 0:
                preserve_mask = self._expand_mask((add_prop_mask < 0.5).float(), expand=3)
            return self.local_edit(
                image, positive_prompt, fill_mask, local_strength, seed, steps, cfg,
                preserve_mask=preserve_mask,
            )
        elif flag == "removal":
            return self.object_removal(image, positive_prompt, remove_mask, local_strength, seed, steps, cfg)
        elif flag == "precise_edit":
            return self.edge_edit(
                image, colored_image, positive_prompt,
                total_mask, add_mask, remove_mask,
                fine_edge,
                edge_strength, color_strength,
                seed, steps, cfg
            )
        elif flag == "kontext":
            return self.kontext_edit(image, positive_prompt, seed, steps, cfg)
        else:
            raise ValueError("Invalid Editing Type: {}".format(flag))


class SAM():
    def __init__(self):
        self.join_alpha = JoinImageWithAlpha()
        self.invert_mask = InvertMask()
        self.predictor = None
        # Initialize immediately with default or ask user to call load_model
        self.load_model()

    def load_model(self, model_type='vit_b', checkpoint_path=None, device='cpu'):
        if checkpoint_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_path = os.path.join(current_dir, 'models', 'sam', 'sam_vit_b_01ec64.pth')
            
        # You need to download the checkpoint manually: 
        # https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
        if not os.path.exists(checkpoint_path):
            print(f"Warning: SAM Checkpoint not found at {checkpoint_path}. Please download it.")
            return
            
        print(f"Loading SAM model: {model_type} from {checkpoint_path}")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)

    def morphological_operations(self, mask, kernel_size=11, iterations=1):
        mask_for_open_close = mask.clone()
        mask_for_close_open = mask.clone()
        
        for i in range(iterations):
            eroded = -F.max_pool2d(-mask_for_open_close, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            opened = F.max_pool2d(eroded, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            dilated = F.max_pool2d(opened, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            open_then_close = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            mask_for_open_close = open_then_close
        
        for i in range(iterations):
            dilated = F.max_pool2d(mask_for_close_open, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            closed = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            eroded = -F.max_pool2d(-closed, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            close_then_open = F.max_pool2d(eroded, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            mask_for_close_open = close_then_open
        
        final_mask = torch.min(open_then_close, close_then_open)
        return final_mask

    def process(self, image, keep_model_loaded=True, coordinates_positive=None, coordinates_negative=None, individual_objects=False, bboxes=None, mask=None):
        if self.predictor is None:
            self.load_model()
            if self.predictor is None:
                raise RuntimeError("SAM model not loaded.")

        # Prepare image for SAM (numpy uint8)
        # image tensor is [1, H, W, 3] float 0-1
        image_np = (image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        self.predictor.set_image(image_np)

        input_point = []
        input_label = []
        
        # Process points
        if coordinates_positive:
            coords = json.loads(coordinates_positive) if isinstance(coordinates_positive, str) else coordinates_positive
            for p in coords:
                input_point.append([p['x'], p['y']])
                input_label.append(1) # 1 = foreground
                
        if coordinates_negative:
            coords = json.loads(coordinates_negative) if isinstance(coordinates_negative, str) else coordinates_negative
            for p in coords:
                input_point.append([p['x'], p['y']])
                input_label.append(0) # 0 = background

        # Process bbox
        input_box = None
        if bboxes:
            
            box_list = []
            for box in bboxes:
                box_list.append(list(box))
            
            if len(box_list) > 0:
                input_box = np.array(box_list)

        if len(input_point) > 0:
            input_point = np.array(input_point)
            input_label = np.array(input_label)
        else:
            input_point = None
            input_label = None

        # Predict
        # We use multimask_output=False to get single best mask
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False,
        )
        
        # masks: [1, H, W]
        mask_np = masks[0]
        
        # Convert back to tensor [1, H, W]
        mask = torch.from_numpy(mask_np).float().unsqueeze(0)
        
        invert_mask = self.invert_mask.invert(mask)[0]
        image_with_alpha = self.join_alpha.join_image_with_alpha(image, invert_mask)[0]

        return (image_with_alpha, mask)
