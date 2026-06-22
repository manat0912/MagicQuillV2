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

# ── Pipeline + GGUF transformer imports ───────────────────────────────────────
from src.pipeline_flux_kontext_control import FluxKontextControlPipeline

# diffusers GGUF loader (requires diffusers >= 0.32 + gguf extra)
from diffusers import AutoencoderKL
from diffusers import FluxTransformer2DModel
from diffusers.quantizers.gguf.utils import GGUFQuantizationConfig
from safetensors.torch import load_file
import tempfile

# ── LoRA key-format monkey-patch (converts PEFT → diffusers keys on the fly) ──
_original_load_lora_weights = FluxKontextControlPipeline.load_lora_weights

def _patched_load_lora_weights(self, pretrained_model_name_or_path_or_dict, **kwargs):
    """Auto-convert mixed-format LoRAs and add transformer prefix."""
    weight_name = kwargs.get("weight_name", "pytorch_lora_weights.safetensors")

    if isinstance(pretrained_model_name_or_path_or_dict, str):
        if os.path.isdir(pretrained_model_name_or_path_or_dict):
            lora_file = os.path.join(pretrained_model_name_or_path_or_dict, weight_name)
        else:
            lora_file = pretrained_model_name_or_path_or_dict

        if os.path.exists(lora_file):
            state_dict = load_file(lora_file)

            needs_format_conversion = any(
                'lora_A.weight' in k or 'lora_B.weight' in k for k in state_dict.keys()
            )
            needs_prefix = not any(k.startswith('transformer.') for k in state_dict.keys())

            if needs_format_conversion or needs_prefix:
                print(f"Processing LoRA: {lora_file}")
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
                print(f"  Total keys: {len(converted_state)}")
                with tempfile.TemporaryDirectory() as temp_dir:
                    from safetensors.torch import save_file
                    temp_file = os.path.join(temp_dir, weight_name)
                    save_file(converted_state, temp_file)
                    return _original_load_lora_weights(self, temp_dir, **kwargs)
            else:
                print(f"LoRA already in correct format: {lora_file}")

    return _original_load_lora_weights(self, pretrained_model_name_or_path_or_dict, **kwargs)

FluxKontextControlPipeline.load_lora_weights = _patched_load_lora_weights
print("[OK] Monkey patch applied to FluxKontextControlPipeline.load_lora_weights")

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from train.src.condition.edge_extraction import InformativeDetector, HEDDetector
from utils_node import BlendInpaint, JoinImageWithAlpha, GrowMask, InvertMask, ColorDetector
from segment_anything import sam_model_registry, SamPredictor

TEST_MODE = False


def to_active_high_mask(mask_tensor: torch.Tensor) -> torch.Tensor:
    """Convert active-low frontend mask (0=stroke, 1=bg) → active-high (1=stroke)."""
    if mask_tensor is None:
        return torch.zeros((1, 512, 512), dtype=torch.float32)
    flat_mask = mask_tensor.clone().detach()
    # If no pixels are below 0.1, the layer is completely empty.
    if torch.sum(flat_mask < 0.1).item() == 0:
        return torch.zeros_like(mask_tensor)
    return (flat_mask < 0.5).float()


class KontextEditModel():
    """
    MagicQuill V2 editing engine — GGUF fast-load variant.

    Components loaded:
      - CLIP-L text encoder          (GPU, bf16)
      - T5-XXL text encoder          (CPU, FP8-quantised)
      - VAE ae.safetensors           (CPU, offloaded between encode/decode)
      - flux1-kontext-dev Q5_K_M     (GPU, GGUF bf16 compute)
      - EasyControl task LoRAs       (edge / color / local / removal)
      - Aux puzzle LoRA              (optional, for prop/foreground mode)
    """

    def __init__(self,
                 # HF repo used ONLY for scheduler + tokenizer configs (no weights downloaded).
                 base_model_path="HelloTestUser/FLUX.1-Kontext-dev",
                 device="cuda",
                 aux_lora_dir=None,
                 easycontrol_base_dir=None,
                 aux_lora_weight_name="puzzle_lora.safetensors",
                 aux_lora_weight=1.0):

        if aux_lora_dir is None:
            aux_lora_dir = os.path.join(current_dir, "models", "v2_ckpt")
        if easycontrol_base_dir is None:
            easycontrol_base_dir = os.path.join(current_dir, "models", "v2_ckpt")

        self.mask_processor  = GrowMask()
        self.scribble_processor = HEDDetector.from_pretrained()
        self.lineart_processor  = InformativeDetector.from_pretrained()
        self.color_processor    = ColorDetector()
        self.blender            = BlendInpaint()
        self.device = device

        import gc
        from transformers import CLIPTextModel, CLIPTextConfig, T5EncoderModel, T5Config
        from optimum.quanto import quantize, qfloat8, freeze

        # ── 1. CLIP text encoder (~246 MB) ────────────────────────────────────
        print("Loading CLIP text encoder...")
        clip_config = CLIPTextConfig.from_pretrained(
            base_model_path, subfolder="text_encoder", token=False
        )
        text_encoder = CLIPTextModel(clip_config).to(device="cpu", dtype=torch.bfloat16)
        clip_path = os.path.join(
            current_dir, "models", "v2_ckpt", "split_files",
            "text_encoders", "clip_l.safetensors"
        )
        clip_state_dict = load_file(clip_path)

        def _load_clip(model, sd):
            model_keys = set(model.state_dict().keys())
            if set(sd.keys()) == model_keys:
                model.load_state_dict(sd)
                return
            new_sd = {}
            for k, v in sd.items():
                nk = k
                if not k.startswith("text_model.") and "text_model." + k in model_keys:
                    nk = "text_model." + k
                elif k.startswith("transformer."):
                    sfx = k.replace("transformer.", "")
                    nk = sfx if sfx in model_keys else "text_model." + sfx
                new_sd[nk] = v
            model.load_state_dict(new_sd, strict=False)

        _load_clip(text_encoder, clip_state_dict)
        text_encoder = text_encoder.to(device=self.device, dtype=torch.bfloat16)
        del clip_state_dict
        gc.collect()
        print("CLIP text encoder loaded.")

        # ── 2. T5 text encoder (~4.9 GB, FP8 on CPU) ─────────────────────────
        print("Loading T5 text encoder...")
        t5_config = T5Config.from_pretrained(
            base_model_path, subfolder="text_encoder_2", token=False
        )
        text_encoder_2 = T5EncoderModel(t5_config).to(device="cpu", dtype=torch.bfloat16)
        t5_path = os.path.join(
            current_dir, "models", "v2_ckpt", "split_files",
            "text_encoders", "t5xxl_fp8_e4m3fn_scaled.safetensors"
        )
        t5_state_dict = load_file(t5_path)

        def _load_t5(model, sd):
            model_keys = set(model.state_dict().keys())
            new_sd = {}
            for k, v in sd.items():
                if v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    v = v.to(torch.bfloat16)
                nk = k
                if not k.startswith("encoder.") and "encoder." + k in model_keys:
                    nk = "encoder." + k
                elif k.startswith("transformer."):
                    sfx = k.replace("transformer.", "")
                    nk = sfx if sfx in model_keys else "encoder." + sfx
                new_sd[nk] = v
            model.load_state_dict(new_sd, strict=False)

        _load_t5(text_encoder_2, t5_state_dict)
        del t5_state_dict
        gc.collect()
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("T5 text encoder loaded (FP8, CPU).")

        # ── 3. VAE (~335 MB, CPU-offloaded) ───────────────────────────────────
        vae_path = os.path.join(
            current_dir, "models", "v2_ckpt", "split_files", "vae", "ae.safetensors"
        )
        vae = AutoencoderKL.from_single_file(
            vae_path,
            config=base_model_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            token=False,
        ).to(device="cpu", dtype=torch.bfloat16)
        print("VAE loaded (CPU offloaded).")

        # ── 4. Flux Kontext GGUF transformer (~8.4 GB, GPU) ───────────────────
        transformer_path = os.path.join(
            current_dir, "models", "v2_ckpt", "split_files",
            "diffusion_models", "flux1-kontext-dev-Q5_K_M.gguf"
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

        # ── 5. Assemble pipeline (scheduler + tokenizers from HF config only) ─
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
            print(f"VAE tiling unavailable: {e}")

        # Clamp working resolution to keep within VRAM budget.
        self.pipe.max_working_side = int(os.environ.get("MAGICQUILL_MAX_SIDE", "1024"))
        print(f"[Speed] max_working_side = {self.pipe.max_working_side} px")

        cond_size = int(os.environ.get("MAGICQUILL_COND_SIZE", "256"))

        # ── 6. EasyControl task LoRAs ─────────────────────────────────────────
        control_lora_config = {
            "local":   {"path": os.path.join(easycontrol_base_dir, "local_lora.safetensors"),   "lora_weights": [1.0], "cond_size": cond_size},
            "removal": {"path": os.path.join(easycontrol_base_dir, "removal_lora.safetensors"), "lora_weights": [1.0], "cond_size": cond_size},
            "edge":    {"path": os.path.join(easycontrol_base_dir, "edge_lora.safetensors"),    "lora_weights": [1.0], "cond_size": cond_size},
            "color":   {"path": os.path.join(easycontrol_base_dir, "color_lora.safetensors"),   "lora_weights": [1.0], "cond_size": cond_size},
        }
        self.pipe.load_control_loras(control_lora_config)
        print("[OK] EasyControl task LoRAs loaded.")

        # ── 7. Aux puzzle LoRA (optional) ─────────────────────────────────────
        self.aux_lora_weight_name = aux_lora_weight_name
        self.aux_lora_dir         = aux_lora_dir
        self.aux_lora_weight      = aux_lora_weight
        self.aux_adapter_name     = "aux"
        self._aux_lora_available  = False
        aux_path = os.path.join(self.aux_lora_dir, self.aux_lora_weight_name)
        if os.path.isfile(aux_path):
            try:
                self.pipe.load_lora_weights(aux_path, adapter_name=self.aux_adapter_name)
                self._aux_lora_available = True
                self._disable_aux_lora()
                print(f"Loaded aux LoRA: {aux_path}")
            except Exception as e:
                print(f"[WARN] Could not load aux LoRA ({e}); foreground mode runs without it.")
        else:
            print(f"Aux LoRA not found at {aux_path}, foreground mode runs without it.")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _tensor_to_pil(self, tensor_image):
        return Image.fromarray(
            np.clip(255. * tensor_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )

    def _pil_to_tensor(self, pil_image):
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

    def _ensure_channels_last(self, tensor, name="tensor"):
        """Ensure tensor is [H, W, C] for blending math."""
        t = tensor.squeeze(0) if tensor.ndim == 4 else tensor
        if t.ndim == 3 and t.shape[0] in (1, 3, 4) and t.shape[-1] not in (1, 3, 4):
            t = t.permute(1, 2, 0)
        return t

    def _expand_mask(self, mask, expand=10):
        """Morphologically dilate a single-channel active-high mask."""
        if expand > 0:
            mask = self.mask_processor.expand_mask({"mask": mask.unsqueeze(0)}, expand, tapered_corners=True)["mask"].squeeze(0)
        return mask.clamp(0.0, 1.0)

    def _apply_black_mask(self, image, mask):
        """Zero-out pixels where mask > 0.5; return PIL."""
        img = self._ensure_channels_last(image.clone(), "image")
        m   = mask.squeeze().unsqueeze(-1) if mask.ndim < 3 else mask.squeeze(0).permute(1, 2, 0) if mask.shape[0] == 1 else mask
        blacked = img * (1.0 - m.clamp(0, 1))
        return self._tensor_to_pil(blacked.unsqueeze(0).permute(0, 3, 1, 2) if blacked.ndim == 3 else blacked)

    def _composite_preserve_mask(self, generated, original, preserve_mask, label="region"):
        """Paste original pixels back wherever preserve_mask > 0.5."""
        gen  = self._ensure_channels_last(generated, "generated")
        orig = self._ensure_channels_last(original,  "original")
        m    = preserve_mask.squeeze()
        if m.ndim == 3:
            m = m[0]
        m = m.unsqueeze(-1).to(gen.device)
        blended = gen * (1.0 - m) + orig * m
        print(f"[OK] Preserved {label} ({int(m.sum().item())} px)")
        return blended

    def _enable_aux_lora(self):
        if self._aux_lora_available:
            self.pipe.set_adapters([self.aux_adapter_name], adapter_weights=[self.aux_lora_weight])

    def _disable_aux_lora(self):
        if self._aux_lora_available:
            self.pipe.disable_adapters()

    def clear_cache(self):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Editing methods ────────────────────────────────────────────────────────

    def local_edit(self,
                   image, positive_prompt, fill_mask, local_strength,
                   seed, steps, cfg, preserve_mask=None):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        original_image_tensor = image.clone()
        # Hard cap: local fill paths must never exceed 18 steps.
        steps = min(steps, 18)

        fill_active = to_active_high_mask(fill_mask)
        has_fill_stroke = torch.sum(fill_active > 0.5).item() > 0
        if has_fill_stroke:
            steps = 18

        original_mask = self._expand_mask(fill_active, expand=10)
        generation_mask = original_mask.clone()

        img = image.clone()

        # Alpha-composite guard: strip any RGBA transparency before VAE encoding.
        if img.shape[0] == 4:
            alpha_ch = img[3:4].clamp(0.0, 1.0)
            img = img[:3] * alpha_ch + original_image_tensor[:3] * (1.0 - alpha_ch)
        elif img.shape[-1] == 4:
            alpha_ch = img[..., 3:4].clamp(0.0, 1.0)
            img = img[..., :3] * alpha_ch + original_image_tensor[..., :3] * (1.0 - alpha_ch)

        # Seed the fill region with Gaussian noise so Flux flow-matching
        # generates new content (flat white is treated as a preserve boundary).
        fill_hi = fill_active.squeeze(0) if fill_active.ndim == 3 else fill_active
        brush_mask_3 = (fill_hi > 0.5).unsqueeze(0).expand(img.shape[0], -1, -1)
        noise_pixels = torch.randn_like(img, generator=torch.Generator(device=img.device).manual_seed(seed))
        img = torch.where(brush_mask_3, noise_pixels, img)
        image_pil = self._tensor_to_pil(img)

        bg_mask   = 1.0 - original_mask
        spatial_pil = self._apply_black_mask(image, bg_mask)

        control_dict = {
            "type": "local",
            "spatial_images": [spatial_pil],
            "gammas": [local_strength],
        }
        result_pil = self.pipe(
            prompt=positive_prompt,
            image=image_pil,
            mask_image=generation_mask,
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

        # Smooth-blend back into original along mask border.
        try:
            mask_np = np.array(self._tensor_to_pil(original_mask))
            if mask_np.ndim == 3:
                mask_np = mask_np[:, :, 0]
            blurred = cv2.GaussianBlur(mask_np, (11, 11), 3.0)
            blend_w = torch.from_numpy(blurred / 255.0).float().unsqueeze(0).unsqueeze(-1)
            orig_cl = self._ensure_channels_last(original_image_tensor, "original")
            final_image = final_image * blend_w + orig_cl * (1.0 - blend_w)
            print("[OK] Local edit with smooth blending")
        except Exception as e:
            print(f"[WARN] Smooth blending failed ({e}), using raw result")

        if preserve_mask is not None:
            final_image = self._composite_preserve_mask(
                final_image, original_image_tensor, preserve_mask, "preserved zone"
            )
        return (final_image, self._tensor_to_pil(spatial_pil), original_mask)

    def edge_edit(self,
                  image, colored_image, positive_prompt,
                  base_mask, add_mask, remove_mask,
                  fine_edge,
                  edge_strength, color_strength,
                  seed, steps, cfg):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        original_image_tensor = image.clone()

        add_active    = to_active_high_mask(add_mask)
        remove_active = to_active_high_mask(remove_mask)
        base_active   = to_active_high_mask(base_mask)

        has_add_stroke    = torch.sum(add_active).item() > 0
        has_remove_stroke = torch.sum(remove_active).item() > 0

        if has_add_stroke or has_remove_stroke:
            brush_only         = torch.clamp(add_active + remove_active, 0.0, 1.0)
            original_mask      = self._expand_mask(brush_only, expand=25)
            matted_only        = torch.clamp(base_active - brush_only, 0.0, 1.0)
            matted_object_mask = self._expand_mask(matted_only, expand=10)
        else:
            original_mask      = self._expand_mask(base_active, expand=25)
            matted_object_mask = self._expand_mask(base_active, expand=10)

        image_pil     = self._tensor_to_pil(image)
        control_dict  = {}
        lineart_output = None

        color_delta = (image - colored_image).abs().max().item() if image.shape == colored_image.shape else 1.0
        use_color_control = color_delta > 1e-3 and not torch.equal(image, colored_image)

        if use_color_control:
            print("Apply color control")
            colored_image_pil = self._tensor_to_pil(colored_image)
            color_image_np    = np.array(colored_image_pil)
            downsampled = cv2.resize(color_image_np, (32, 32), interpolation=cv2.INTER_AREA)
            upsampled   = cv2.resize(downsampled, (256, 256), interpolation=cv2.INTER_NEAREST)
            color_block = Image.fromarray(upsampled)
            control_dict = {
                "type": "color",
                "spatial_images": [color_block],
                "gammas": [color_strength],
            }
        else:
            print("Apply edge control")
            if fine_edge == "enable":
                lineart_image  = self.lineart_processor(
                    np.array(self._tensor_to_pil(image.cpu().squeeze())),
                    detect_resolution=1024, style="contour", output_type="pil"
                )
                lineart_output = self._pil_to_tensor(lineart_image)
            else:
                scribble_image = self.scribble_processor(
                    np.array(self._tensor_to_pil(image.cpu().squeeze())),
                    safe=True, resolution=512, output_type="pil"
                )
                lineart_output = self._pil_to_tensor(scribble_image)

            if lineart_output is None:
                raise ValueError("Preprocessor failed to generate lineart.")

            add_mask_resized    = F.interpolate(add_active.unsqueeze(0).float(),    size=(lineart_output.shape[1], lineart_output.shape[2]), mode='nearest').squeeze(0)
            remove_mask_resized = F.interpolate(remove_active.unsqueeze(0).float(), size=(lineart_output.shape[1], lineart_output.shape[2]), mode='nearest').squeeze(0)
            lineart_output[remove_mask_resized > 0.5] = 0.0
            lineart_output[add_mask_resized    > 0.5] = 1.0

            control_dict = {
                "type": "edge",
                "spatial_images": [self._tensor_to_pil(lineart_output)],
                "gammas": [edge_strength],
            }

        colored_image_np = np.array(self._tensor_to_pil(colored_image))
        debug_image = (
            lineart_output if lineart_output is not None
            else self.color_processor(colored_image_np, detect_resolution=1024, output_type="pil")
        )

        result_pil = self.pipe(
            prompt=positive_prompt,
            image=image_pil,
            mask_image=original_mask,
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
        remove_active  = to_active_high_mask(remove_mask)
        original_mask  = self._expand_mask(remove_active, expand=10)

        image_pil   = self._tensor_to_pil(image)
        spatial_pil = self._apply_black_mask(image, original_mask)
        control_dict = {
            "type": "removal",
            "spatial_images": [spatial_pil],
            "gammas": [local_strength],
        }

        result_pil = self.pipe(
            prompt=positive_prompt,
            image=image_pil,
            mask_image=original_mask,
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

        try:
            mask_np = np.array(self._tensor_to_pil(original_mask))
            if mask_np.ndim == 3:
                mask_np = mask_np[:, :, 0]
            blurred = cv2.GaussianBlur(mask_np, (11, 11), 3.0)
            blend_w = torch.from_numpy(blurred / 255.0).float().unsqueeze(0).unsqueeze(-1)
            orig_cl = self._ensure_channels_last(original_image_tensor, "original")
            final_image = final_image * blend_w + orig_cl * (1.0 - blend_w)
            print("[OK] Object removal with smooth blending")
        except Exception as e:
            print(f"[WARN] Smooth blending failed ({e}), using raw result")

        return (final_image, self._tensor_to_pil(spatial_pil), original_mask)

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
        mask = torch.zeros(
            (1, final_image.shape[1], final_image.shape[2]),
            dtype=torch.float32, device=final_image.device
        )
        return (final_image, image, mask)

    def process(self, image, colored_image,
                merged_image, positive_prompt,
                total_mask, add_mask, remove_mask, add_prop_mask, fill_mask,
                fine_edge, fix_perspective, edge_strength, color_strength, local_strength, grow_size,
                seed, steps, cfg, flag="precise_edit"):

        if flag in ("foreground", "local"):
            steps = min(steps, 18)

            if flag == "foreground":
                # Use fill_mask (user's actual brush strokes) as the generation
                # target. total_mask covers the whole prop bounding box and causes
                # local_edit to black-out the entire prop zone.
                # Fall back to total_mask only when no fill strokes are present.
                fill_active_check = to_active_high_mask(fill_mask)
                fill_has_pixels   = torch.sum(fill_active_check > 0.5).item() > 0
                if fill_has_pixels:
                    effective_fill_mask = fill_mask
                    print(f"[ROUTING] foreground -> local_edit | fill_mask ({int(torch.sum(fill_active_check>0.5).item())} px), merged canvas")
                else:
                    effective_fill_mask = total_mask
                    print(f"[ROUTING] foreground -> local_edit | total_mask (no fill strokes), merged canvas")
                canvas_image = merged_image
            else:
                effective_fill_mask = fill_mask
                canvas_image        = image
                print(f"[ROUTING] local -> local_edit | fill_mask, original image")

            fill_active_check2 = to_active_high_mask(effective_fill_mask)
            print(f"[DIAGNOSTIC] effective_fill_mask sum: {torch.sum(fill_active_check2 > 0.5).item()}")

            return self.local_edit(
                canvas_image, positive_prompt, effective_fill_mask, local_strength,
                seed, steps, cfg,
                preserve_mask=None,
            )

        elif flag == "removal":
            return self.object_removal(
                image, positive_prompt, remove_mask, local_strength, seed, steps, cfg
            )
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
            raise ValueError(f"Invalid Editing Type: {flag}")


# ── SAM (Segment Anything) ─────────────────────────────────────────────────────

class SAM():
    def __init__(self):
        self.join_alpha  = JoinImageWithAlpha()
        self.invert_mask = InvertMask()
        self.predictor   = None
        self.load_model()

    def load_model(self, model_type='vit_b', checkpoint_path=None, device='cpu'):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'models', 'sam', 'sam_vit_b_01ec64.pth'
            )
        if not os.path.exists(checkpoint_path):
            print(f"Warning: SAM checkpoint not found at {checkpoint_path}.")
            return
        print(f"Loading SAM model: {model_type} from {checkpoint_path}")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)

    def morphological_operations(self, mask, kernel_size=11, iterations=1):
        mask_for_open_close = mask.clone()
        mask_for_close_open = mask.clone()

        for _ in range(iterations):
            eroded           = -F.max_pool2d(-mask_for_open_close, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            opened           = F.max_pool2d(eroded, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            dilated          = F.max_pool2d(opened, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            open_then_close  = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            mask_for_open_close = open_then_close

        for _ in range(iterations):
            dilated          = F.max_pool2d(mask_for_close_open, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            closed           = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            eroded           = -F.max_pool2d(-closed, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            close_then_open  = F.max_pool2d(eroded, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            mask_for_close_open = close_then_open

        return torch.min(open_then_close, close_then_open)

    def process(self, image, keep_model_loaded=True,
                coordinates_positive=None, coordinates_negative=None,
                individual_objects=False, bboxes=None, mask=None):
        if self.predictor is None:
            self.load_model()
            if self.predictor is None:
                raise RuntimeError("SAM model not loaded.")

        image_np = (image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        self.predictor.set_image(image_np)

        input_point, input_label = [], []

        if coordinates_positive:
            coords = json.loads(coordinates_positive) if isinstance(coordinates_positive, str) else coordinates_positive
            for p in coords:
                input_point.append([p['x'], p['y']])
                input_label.append(1)

        if coordinates_negative:
            coords = json.loads(coordinates_negative) if isinstance(coordinates_negative, str) else coordinates_negative
            for p in coords:
                input_point.append([p['x'], p['y']])
                input_label.append(0)

        input_box = None
        if bboxes:
            box_list = [list(box) for box in bboxes]
            if box_list:
                input_box = np.array(box_list)

        if input_point:
            input_point = np.array(input_point)
            input_label = np.array(input_label)
        else:
            input_point = input_label = None

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False,
        )

        mask_np          = masks[0]
        mask             = torch.from_numpy(mask_np).float().unsqueeze(0)
        invert_mask      = self.invert_mask.invert(mask)[0]
        image_with_alpha = self.join_alpha.join_image_with_alpha(image, invert_mask)[0]

        return (image_with_alpha, mask)
