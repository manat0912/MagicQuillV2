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

# New imports for the diffuser pipeline
from diffusers import FluxControlNetInpaintPipeline, FluxControlNetModel, FluxTransformer2DModel


import tempfile
from safetensors.torch import load_file, save_file

_original_load_lora_weights = FluxControlNetInpaintPipeline.load_lora_weights

def _repair_after_lora_load(pipe):
    """Repair transformer state after LoRA loading."""
    print("[LoRA] Re-applying layerwise casting to transformer...")
    try:
        pipe.transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16
        )
        print("[LoRA] Re-applied layerwise casting successfully.")
    except Exception as e:
        print(f"[LoRA] Error during enable_layerwise_casting: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    # Fix skipped modules whose weights were corrupted from BF16 to FP8.
    repaired_count = 0
    print("[LoRA] Repairing skipped modules...")
    try:
        modules_list = list(pipe.transformer.named_modules())
        print(f"[LoRA] Total modules to scan: {len(modules_list)}")
        for idx, (name, module) in enumerate(modules_list):
            targets = [(name, module)]
            if hasattr(module, 'base_layer'):
                targets.append((name + ".base_layer", module.base_layer))
            
            for target_name, target in targets:
                # Check if this module has its own parameters (not inherited) in FP8
                has_fp8_params = any(
                    p.dtype == torch.float8_e4m3fn
                    for p in target.parameters(recurse=False)
                )
                if not has_fp8_params:
                    continue
                
                has_hook = (
                    hasattr(target, '_diffusers_hook')
                    and target._diffusers_hook.get_hook("layerwise_casting") is not None
                )
                if not has_hook:
                    target.to(torch.bfloat16)
                    repaired_count += 1
            
            if (idx + 1) % 200 == 0:
                print(f"[LoRA] Scanned {idx + 1}/{len(modules_list)} modules. Repaired so far: {repaired_count}")
    except Exception as e:
        print(f"[LoRA] Error during module repair loop: {e}")
        import traceback
        traceback.print_exc()
        raise e
        
    if repaired_count > 0:
        print(f"[LoRA] Repaired {repaired_count} skipped modules back to bfloat16")
    
    # Cast LoRA adapter parameters to BF16
    casted_count = 0
    print("[LoRA] Casting LoRA adapter parameters...")
    try:
        for name, param in pipe.transformer.named_parameters():
            if "lora" in name and param.dtype != torch.bfloat16:
                param.data = param.data.to(torch.bfloat16)
                casted_count += 1
    except Exception as e:
        print(f"[LoRA] Error during LoRA parameter casting: {e}")
        import traceback
        traceback.print_exc()
        raise e
        
    if casted_count > 0:
        print(f"[LoRA] Casted {casted_count} LoRA parameters to bfloat16")
    print("[LoRA] Finished _repair_after_lora_load.")

def _patched_load_lora_weights(self, pretrained_model_name_or_path_or_dict, **kwargs):
    """自动转换混合格式的 LoRA 并添加 transformer 前缀"""
    if "prefix" not in kwargs:
        kwargs["prefix"] = None
    weight_name = kwargs.get("weight_name", "pytorch_lora_weights.safetensors")
    
    if isinstance(pretrained_model_name_or_path_or_dict, str):
        if os.path.isdir(pretrained_model_name_or_path_or_dict):
            lora_file = os.path.join(pretrained_model_name_or_path_or_dict, weight_name)
        else:
            lora_file = pretrained_model_name_or_path_or_dict
        
        if os.path.exists(lora_file):
            state_dict = load_file(lora_file)
            
            # 检查是否需要转换格式或添加前缀
            needs_format_conversion = any('lora_A.weight' in k or 'lora_B.weight' in k for k in state_dict.keys())
            needs_prefix = not any(k.startswith('transformer.') for k in state_dict.keys())
            
            if needs_format_conversion or needs_prefix:
                print(f"[LoRA] Processing LoRA: {lora_file}")
                if needs_format_conversion:
                    print(f"   - Converting PEFT format to diffusers format")
                if needs_prefix:
                    print(f"   - Adding 'transformer.' prefix to keys")
                
                converted_state = {}
                converted_count = 0
                
                for key, value in state_dict.items():
                    new_key = key
                    
                    # 步骤 1: 转换 PEFT 格式到 diffusers 格式
                    if 'lora_A.weight' in new_key:
                        new_key = new_key.replace('lora_A.weight', 'lora.down.weight')
                        converted_count += 1
                    elif 'lora_B.weight' in new_key:
                        new_key = new_key.replace('lora_B.weight', 'lora.up.weight')
                        converted_count += 1
                    
                    # 步骤 2: 添加 transformer 前缀（如果还没有的话）
                    if not new_key.startswith('transformer.'):
                        new_key = f'transformer.{new_key}'
                    
                    converted_state[new_key] = value
                
                if needs_format_conversion:
                    print(f"   [OK] Converted {converted_count} PEFT keys")
                print(f"   [OK] Total keys: {len(converted_state)}")
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_file = os.path.join(temp_dir, weight_name)
                    save_file(converted_state, temp_file)
                    res = _original_load_lora_weights(self, temp_dir, **kwargs)
                    _repair_after_lora_load(self)
                    return res
            else:
                print(f"[OK] LoRA already in correct format: {lora_file}")
    
    # 不需要转换，使用原始方法
    res = _original_load_lora_weights(self, pretrained_model_name_or_path_or_dict, **kwargs)
    _repair_after_lora_load(self)
    return res

# 应用 monkey patch
FluxControlNetInpaintPipeline.load_lora_weights = _patched_load_lora_weights
print("[OK] Monkey patch applied to FluxControlNetInpaintPipeline.load_lora_weights")

import diffusers.loaders.single_file_model as sfm
_orig_get_single_file_loadable_mapping_class = sfm._get_single_file_loadable_mapping_class
def _patched_get_single_file_loadable_mapping_class(cls):
    # Support custom FluxTransformer2DModel class for single file loading
    if cls.__name__ == "FluxTransformer2DModel":
        return "FluxTransformer2DModel"
    return _orig_get_single_file_loadable_mapping_class(cls)
sfm._get_single_file_loadable_mapping_class = _patched_get_single_file_loadable_mapping_class
print("[OK] Monkey patch applied to diffusers._get_single_file_loadable_mapping_class")

# Monkeypatch _get_t5_prompt_embeds to prevent device mismatch when text_encoder_2 is on CPU
def patched_get_t5_prompt_embeds(
    self,
    prompt=None,
    num_images_per_prompt=1,
    max_sequence_length=512,
    device=None,
    dtype=None,
):
    import logging
    logger = logging.getLogger("diffusers")
    from diffusers.loaders import TextualInversionLoaderMixin

    device = device or self._execution_device
    dtype = dtype or self.text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if isinstance(self, TextualInversionLoaderMixin):
        prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

    text_inputs = self.tokenizer_2(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because `max_sequence_length` is set to "
            f" {max_sequence_length} tokens: {removed_text}"
        )

    t5_device = next(self.text_encoder_2.parameters()).device
    prompt_embeds = self.text_encoder_2(text_input_ids.to(t5_device), output_hidden_states=False)[0]

    dtype = self.text_encoder_2.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds

FluxControlNetInpaintPipeline._get_t5_prompt_embeds = patched_get_t5_prompt_embeds
print("[OK] Monkey patch applied to FluxControlNetInpaintPipeline._get_t5_prompt_embeds")

# Monkeypatch _get_clip_prompt_embeds to prevent device mismatch when text_encoder is on CPU
def patched_get_clip_prompt_embeds(
    self,
    prompt: str | list[str],
    num_images_per_prompt: int = 1,
    device: torch.device | None = None,
):
    import logging
    logger = logging.getLogger("diffusers")
    from diffusers.loaders import TextualInversionLoaderMixin

    device = device or self._execution_device

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if isinstance(self, TextualInversionLoaderMixin):
        prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

    text_inputs = self.tokenizer(
        prompt,
        padding="max_length",
        max_length=self.tokenizer_max_length,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
        logger.warning(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {self.tokenizer_max_length} tokens: {removed_text}"
        )
    
    clip_device = next(self.text_encoder.parameters()).device
    prompt_embeds = self.text_encoder(text_input_ids.to(clip_device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds

FluxControlNetInpaintPipeline._get_clip_prompt_embeds = patched_get_clip_prompt_embeds
print("[OK] Monkey patch applied to FluxControlNetInpaintPipeline._get_clip_prompt_embeds")

# Monkeypatch prepare_mask_latents to save the mask condition for the ControlNet wrapper
_orig_prepare_mask_latents = FluxControlNetInpaintPipeline.prepare_mask_latents
def patched_prepare_mask_latents(self, *args, **kwargs):
    mask = args[0] if len(args) > 0 else kwargs.get('mask')
    self._current_mask_condition = mask
    return _orig_prepare_mask_latents(self, *args, **kwargs)
FluxControlNetInpaintPipeline.prepare_mask_latents = patched_prepare_mask_latents
print("[OK] Monkey patch applied to FluxControlNetInpaintPipeline.prepare_mask_latents")

# Monkeypatch FluxControlNetModel.__init__ to support 68 channels for controlnet_x_embedder and 64 for x_embedder
from diffusers.configuration_utils import register_to_config
original_flux_controlnet_init = FluxControlNetModel.__init__

@register_to_config
def patched_flux_controlnet_init(
    self,
    patch_size: int = 1,
    in_channels: int = 64,
    num_layers: int = 19,
    num_single_layers: int = 38,
    attention_head_dim: int = 128,
    num_attention_heads: int = 24,
    joint_attention_dim: int = 4096,
    pooled_projection_dim: int = 768,
    guidance_embeds: bool = False,
    axes_dims_rope: list[int] = [16, 56, 56],
    num_mode: int = None,
    conditioning_embedding_channels: int = None,
):
    original_flux_controlnet_init(
        self,
        patch_size=patch_size,
        in_channels=in_channels,
        num_layers=num_layers,
        num_single_layers=num_single_layers,
        attention_head_dim=attention_head_dim,
        num_attention_heads=num_attention_heads,
        joint_attention_dim=joint_attention_dim,
        pooled_projection_dim=pooled_projection_dim,
        guidance_embeds=guidance_embeds,
        axes_dims_rope=axes_dims_rope,
        num_mode=num_mode,
        conditioning_embedding_channels=conditioning_embedding_channels,
    )
    import torch.nn as nn
    from diffusers.models.controlnets.controlnet_flux import zero_module
    in_channels = self.out_channels
    extra_channels = 4
    self.controlnet_x_embedder = zero_module(nn.Linear(in_channels + extra_channels, self.inner_dim))

FluxControlNetModel.__init__ = patched_flux_controlnet_init
print("[OK] Monkey patch applied to FluxControlNetModel.__init__")

# Wrapper to concatenate packed mask to controlnet_cond (64 + 4 = 68 channels)
class FluxControlNetWrapper(torch.nn.Module):
    def __init__(self, controlnet, pipe):
        super().__init__()
        self.controlnet = controlnet
        self.pipe = pipe
        self.config = controlnet.config

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in ('_modules', 'controlnet'):
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            return getattr(self.controlnet, name)

    @property
    def __class__(self):
        return FluxControlNetModel

    def forward(self, hidden_states, controlnet_cond, **kwargs):
        mask_cond = getattr(self.pipe, '_current_mask_condition', None)
        if mask_cond is not None and controlnet_cond.shape[-1] == 64:
            batch_size = controlnet_cond.shape[0]
            height, width = mask_cond.shape[-2:]
            latent_h = 2 * (int(height) // 16)
            latent_w = 2 * (int(width) // 16)
            
            mask_resized = torch.nn.functional.interpolate(mask_cond, size=(latent_h, latent_w))
            mask_resized = mask_resized.to(device=controlnet_cond.device, dtype=controlnet_cond.dtype)
            
            packed_mask = self.pipe._pack_latents(
                mask_resized,
                batch_size,
                1, # 1 input channel
                latent_h,
                latent_w
            )
            controlnet_cond = torch.cat([controlnet_cond, packed_mask], dim=-1)
            
        return self.controlnet(hidden_states=hidden_states, controlnet_cond=controlnet_cond, **kwargs)


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..', 'comfy_extras')))

from train.src.condition.edge_extraction import InformativeDetector, HEDDetector
from utils_node import BlendInpaint, JoinImageWithAlpha, GrowMask, InvertMask, ColorDetector
from segment_anything import sam_model_registry, SamPredictor

TEST_MODE = False

class KontextEditModel():
    def __init__(self, base_model_path="HelloTestUser/FLUX.1-Kontext-dev", device="cuda",
                 aux_lora_dir=None, easycontrol_base_dir=None,
                 aux_lora_weight_name="puzzle_lora.safetensors",
                 aux_lora_weight=1.0):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if aux_lora_dir is None:
            aux_lora_dir = os.path.join(current_dir, "models", "v2_ckpt")
        if easycontrol_base_dir is None:
            easycontrol_base_dir = os.path.join(current_dir, "models", "v2_ckpt")

        # Keep necessary preprocessors
        self.mask_processor = GrowMask()
        self.scribble_processor = HEDDetector.from_pretrained()
        self.lineart_processor = InformativeDetector.from_pretrained()
        self.color_processor = ColorDetector()
        self.blender = BlendInpaint()

        # Initialize the new pipeline (Kontext version)
        self.device = device
        
        # 1. Load CLIP text encoder (~246MB) on CUDA
        from transformers import CLIPTextModel, CLIPTextConfig
        from safetensors.torch import load_file
        import gc

        print("Loading CLIP text encoder config...")
        clip_config = CLIPTextConfig.from_pretrained(
            base_model_path,
            subfolder="text_encoder",
            token=False
        )
        print("Instantiating CLIP text encoder model...")
        text_encoder = CLIPTextModel(clip_config).to(device="cpu", dtype=torch.bfloat16)

        clip_path = os.path.join(
            current_dir, "models", "v2_ckpt", "split_files", "text_encoders", "clip_l.safetensors"
        )
        print(f"Loading CLIP weights from: {clip_path}")
        clip_state_dict = load_file(clip_path)

        def load_text_encoder_state_dict(model, state_dict):
            model_keys = set(model.state_dict().keys())
            sd_keys = set(state_dict.keys())
            if model_keys == sd_keys:
                model.load_state_dict(state_dict)
                return
            new_sd = {}
            for k, v in state_dict.items():
                new_key = k
                if not k.startswith("text_model.") and "text_model." + k in model_keys:
                    new_key = "text_model." + k
                elif k.startswith("cond_stage_model.transformer."):
                    suffix = k.replace("cond_stage_model.transformer.", "")
                    if suffix in model_keys:
                        new_key = suffix
                    elif "text_model." + suffix in model_keys:
                        new_key = "text_model." + suffix
                elif k.startswith("transformer."):
                    suffix = k.replace("transformer.", "")
                    if suffix in model_keys:
                        new_key = suffix
                    elif "text_model." + suffix in model_keys:
                        new_key = "text_model." + suffix
                new_sd[new_key] = v
            model.load_state_dict(new_sd, strict=False)

        load_text_encoder_state_dict(text_encoder, clip_state_dict)
        text_encoder = text_encoder.to(device=self.device, dtype=torch.bfloat16)
        print("CLIP text encoder loaded successfully and moved to GPU.")
        del clip_state_dict
        gc.collect()

        # 2. Load T5 text encoder (~4.9GB) on CPU
        from transformers import T5EncoderModel, T5Config
        from optimum.quanto import quantize, qfloat8, freeze

        print("Loading T5 text encoder config...")
        t5_config = T5Config.from_pretrained(
            base_model_path,
            subfolder="text_encoder_2",
            token=False
        )
        print("Instantiating T5 text encoder model...")
        text_encoder_2 = T5EncoderModel(t5_config).to(device="cpu", dtype=torch.bfloat16)

        t5_path = os.path.join(
            current_dir, "models", "v2_ckpt", "split_files", "text_encoders", "t5xxl_fp8_e4m3fn_scaled.safetensors"
        )
        print(f"Loading T5 weights from: {t5_path}")
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
                    if suffix in model_keys:
                        new_key = suffix
                    elif "encoder." + suffix in model_keys:
                        new_key = "encoder." + suffix
                elif k.startswith("transformer."):
                    suffix = k.replace("transformer.", "")
                    if suffix in model_keys:
                        new_key = suffix
                    elif "encoder." + suffix in model_keys:
                        new_key = "encoder." + suffix
                new_sd[new_key] = v
            model.load_state_dict(new_sd, strict=False)

        load_t5_state_dict(text_encoder_2, t5_state_dict)
        print("T5 text encoder loaded successfully. Quantizing to FP8 (qfloat8) on CPU...")
        del t5_state_dict
        gc.collect()

        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)
        print("T5 text encoder quantized successfully. Cleaning up memory...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. Load VAE model (~335MB) on CUDA
        from diffusers import AutoencoderKL
        vae_path = os.path.join(
            current_dir, "models", "v2_ckpt", "split_files", "vae", "ae.safetensors"
        )
        print(f"Loading VAE from: {vae_path}")
        vae = AutoencoderKL.from_single_file(
            vae_path,
            config=base_model_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            token=False
        )
        vae = vae.to(device=self.device, dtype=torch.bfloat16)
        print("VAE loaded and moved to GPU.")

        # 4. Load GGUF transformer model (~8.42GB) on CUDA
        from diffusers import GGUFQuantizationConfig
        transformer_path = os.path.join(
            current_dir, "models", "v2_ckpt", "split_files", "diffusion_models", "flux1-kontext-dev-Q5_K_M.gguf"
        )
        print(f"Loading GGUF transformer from: {transformer_path}")
        
        transformer = FluxTransformer2DModel.from_single_file(
            transformer_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
            config=base_model_path,
            subfolder="transformer"
        ).to(device)

        # 5. Load Alimama Inpaint ControlNet on CUDA
        controlnet_path = os.path.join(
            os.path.abspath(os.path.join(current_dir, "..")),
            "cache", "HF_HOME", "hub",
            "models--alimama-creative--FLUX.1-dev-Controlnet-Inpainting-Beta",
            "snapshots", "4c71e88b32ab247b3c2518803224c7c6473dbeb9"
        )
        print(f"Loading ControlNet from: {controlnet_path}")
        controlnet = FluxControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            local_files_only=True
        )
        print("Quantizing ControlNet to FP8 (qfloat8) on GPU...")
        quantize(controlnet, weights=qfloat8)
        freeze(controlnet)
        controlnet.to(self.device)
        print("ControlNet loaded and quantized successfully.")

        # 6. Assemble pipeline
        self.pipe = FluxControlNetInpaintPipeline.from_pretrained(
            base_model_path,
            transformer=transformer,
            controlnet=controlnet,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            vae=vae,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            token=False
        )
        self.pipe.controlnet = FluxControlNetWrapper(self.pipe.controlnet, self.pipe)
        
        try:
            vae.enable_tiling()
        except Exception as e:
            print(f"Failed to enable VAE tiling: {e}")

        # EasyControl LoRAs and aux LoRA are bypassed since Alimama ControlNet handles inpainting natively.
        print("[Inpaint] EasyControl and aux LoRA loading bypassed for ControlNet inpainting.")
        self.aux_lora_weight_name = aux_lora_weight_name
        self.aux_lora_dir = aux_lora_dir
        self.aux_lora_weight = aux_lora_weight
        self.aux_adapter_name = "aux"


    # gamma is now applied inside the pipeline based on control_dict

    def _tensor_to_pil(self, tensor_image):
        # Converts a ComfyUI-style tensor [1, H, W, 3] to a PIL Image
        return Image.fromarray(np.clip(255. * tensor_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    def _pil_to_tensor(self, pil_image):
        # Converts a PIL image to a ComfyUI-style tensor [1, H, W, 3]
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

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
        print("[Fill] _enable_aux_lora is a no-op for FLUX.1 Fill")

    def _disable_aux_lora(self):
        print("[Fill] _disable_aux_lora is a no-op for FLUX.1 Fill")

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
        # image_tensor: [1, H, W, 3] in [0,1]
        # binary_mask: [H, W] or [1, H, W], 1=mask area (white)
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
        
        # Prepare mask and original image
        original_image_tensor = image.clone()
        
        # Invert the brush stroke masks (since create_alpha_mask returns 1.0 on background, 0.0 on stroke)
        add_stroke = (add_mask < 0.5).float()
        remove_stroke = (remove_mask < 0.5).float()
        stroke_mask = torch.clamp(add_stroke + remove_stroke, 0.0, 1.0)
        
        # Detect color brush strokes by comparing image and colored_image
        if not torch.equal(image, colored_image):
            color_diff = torch.mean(torch.abs(image - colored_image), dim=-1, keepdim=True)
            color_mask = (color_diff > 0.01).float().permute(0, 3, 1, 2).squeeze(1) # [1, H, W]
            stroke_mask = torch.clamp(stroke_mask + color_mask.to(stroke_mask.device), 0.0, 1.0)
            
        original_mask = self._expand_mask(stroke_mask, expand=10)
        
        # Pass the colored image containing sketches/strokes as input image
        image_pil = self._tensor_to_pil(colored_image)
        mask_pil = self._tensor_mask_to_pil3(original_mask)
        
        lineart_output = None

        # Determine control type: color or edge for debug visualization
        if not torch.equal(image, colored_image):
            print("Apply color control (debug visualization)")
            colored_image_pil = self._tensor_to_pil(colored_image)
            color_image_np = np.array(colored_image_pil)
            downsampled = cv2.resize(color_image_np, (32, 32), interpolation=cv2.INTER_AREA)
            upsampled = cv2.resize(downsampled, (256, 256), interpolation=cv2.INTER_NEAREST)
            color_block = Image.fromarray(upsampled)
            user_val = color_strength
        else:
            print("Apply edge control (debug visualization)")
            if fine_edge == "enable":
                lineart_image = self.lineart_processor(np.array(self._tensor_to_pil(image.cpu().squeeze())), detect_resolution=1024, style="contour", output_type="pil")
                lineart_output = self._pil_to_tensor(lineart_image)
            else:
                scribble_image = self.scribble_processor(np.array(self._tensor_to_pil(image.cpu().squeeze())), safe=True, resolution=512, output_type="pil")
                lineart_output = self._pil_to_tensor(scribble_image)
            
            if lineart_output is not None:
                # Apply user sketches to the lineart (since masks are inverted, stroke is < 0.5)
                add_mask_resized = F.interpolate(add_mask.unsqueeze(0).float(), size=(lineart_output.shape[1], lineart_output.shape[2]), mode='nearest').squeeze(0)
                remove_mask_resized = F.interpolate(remove_mask.unsqueeze(0).float(), size=(lineart_output.shape[1], lineart_output.shape[2]), mode='nearest').squeeze(0)
                bool_add_stroke = (add_mask_resized < 0.5)
                bool_remove_stroke = (remove_mask_resized < 0.5)
                lineart_output[bool_remove_stroke] = 0.0
                lineart_output[bool_add_stroke] = 1.0
            
            user_val = edge_strength

        # Prepare debug/output images
        colored_image_np = np.array(self._tensor_to_pil(colored_image))
        debug_image = lineart_output if lineart_output is not None else self.color_processor(colored_image_np, detect_resolution=1024, output_type="pil")

        # Map user strength slider (0.0 to 5.0) to denoising strength (0.4 to 1.0)
        # Higher user slider value -> lower denoising strength -> preserves more of user's sketch/color
        denoise_strength = max(0.4, min(1.0, 1.0 - 0.15 * user_val))
        print(f"[Precise Edit] User value: {user_val}, mapped to denoising strength: {denoise_strength}")

        # Run inference
        result_pil = self.pipe(
            prompt=positive_prompt,
            image=image_pil,
            mask_image=mask_pil,
            control_image=image_pil,
            height=image_pil.height,
            width=image_pil.width,
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=generator,
            max_sequence_length=512,
            strength=denoise_strength
        ).images[0]
        self.clear_cache()

        result_tensor = self._pil_to_tensor(result_pil)

        # Blender blend_inpaint over original_mask
        final_image = self.blender.blend_inpaint(
            result_tensor, original_image_tensor, original_mask, kernel=31, sigma=15
        )[0]

        return (final_image, debug_image, original_mask)

    def object_removal(self,
                       image, positive_prompt, 
                       remove_mask, 
                       local_strength,
                       seed, steps, cfg):
        
        generator = torch.Generator(device=self.device).manual_seed(seed)

        original_image_tensor = image.clone()
        # Convert inverted remove_mask
        target_mask = (remove_mask < 0.5).float()
        original_mask = self._expand_mask(target_mask, expand=10)
        
        image_pil = self._tensor_to_pil(image)
        mask_pil = self._tensor_mask_to_pil3(original_mask)
        spatial_pil = self._apply_black_mask(image, original_mask)

        result_pil = self.pipe(
            prompt=positive_prompt,
            image=image_pil,
            mask_image=mask_pil,
            control_image=image_pil,
            height=image_pil.height,
            width=image_pil.width,
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=generator,
            max_sequence_length=512,
        ).images[0]
        self.clear_cache()

        result_tensor = self._pil_to_tensor(result_pil)  # [1, H', W', 3]

        # Blender blend_inpaint: smooth feathered composite back into the original image
        final_image = self.blender.blend_inpaint(
            result_tensor, original_image_tensor, original_mask, kernel=31, sigma=15
        )[0]
        return (final_image, self._pil_to_tensor(spatial_pil), original_mask)

    def local_edit(self,
                   image, positive_prompt, fill_mask, local_strength,
                   seed, steps, cfg):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        original_image_tensor = image.clone()
        # Convert inverted fill_mask
        target_mask = (fill_mask < 0.5).float()
        original_mask = self._expand_mask(target_mask, expand=10)
        image_pil = self._tensor_to_pil(image)
        mask_pil = self._tensor_mask_to_pil3(original_mask)
        spatial_pil = self._apply_black_mask(image, original_mask)

        result_pil = self.pipe(
            prompt=positive_prompt,
            image=image_pil,
            mask_image=mask_pil,
            control_image=image_pil,
            height=image_pil.height,
            width=image_pil.width,
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=generator,
            max_sequence_length=512,
        ).images[0]
        self.clear_cache()

        result_tensor = self._pil_to_tensor(result_pil)  # [1, H', W', 3]

        # Blender blend_inpaint: smooth feathered composite back into the original image
        final_image = self.blender.blend_inpaint(
            result_tensor, original_image_tensor, original_mask, kernel=31, sigma=15
        )[0]
        return (final_image, self._pil_to_tensor(spatial_pil), original_mask)

    def foreground_edit(self,
                        merged_image, positive_prompt,
                        add_prop_mask, fill_mask, total_mask, fix_perspective, grow_size,
                        seed, steps, cfg):
        generator = torch.Generator(device=self.device).manual_seed(seed)

        if fix_perspective == "enable":
            positive_prompt = positive_prompt + " Fix the perspective if necessary."

        # Convert inverted prop mask: prop_mask has 1.0 on character, 0.0 on background
        prop_mask = (add_prop_mask < 0.5).float()
        
        # Boundary/border of the character
        expanded_prop = self._expand_mask(prop_mask, expand=grow_size)
        border_mask = torch.clamp(expanded_prop - prop_mask, 0.0, 1.0)
        
        # Manual brush stroke region (fill_mask has 0.0 on stroke, 1.0 on background)
        brush_mask = (fill_mask < 0.5).float()
        
        # edit_mask is the target region to generate (background + border + brush stroke)
        edit_mask = torch.clamp(total_mask + border_mask + brush_mask, 0.0, 1.0)
        
        # final_mask is expanded by 25 pixels to provide a smooth blending transition
        final_mask = self._expand_mask(edit_mask, expand=25)

        image_pil = self._tensor_to_pil(merged_image)
        mask_pil = self._tensor_mask_to_pil3(edit_mask)

        # Scene-aware prompt adapted for native inpainting
        scene_prompt = "adapt the foreground object naturally into the background. " + positive_prompt

        # Enable aux LoRA only for foreground (no-op for FLUX.1 Fill)
        self._enable_aux_lora()

        result_pil = self.pipe(
            prompt=scene_prompt,
            image=image_pil,
            mask_image=mask_pil,
            control_image=image_pil,
            height=image_pil.height,
            width=image_pil.width,
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=generator,
            max_sequence_length=512,
        ).images[0]

        # Disable aux LoRA afterwards
        self._disable_aux_lora()

        result_tensor = self._pil_to_tensor(result_pil)  # [1, H', W', 3]

        # Blender blend_inpaint: smooth feathered composite back into the original image
        final_image = self.blender.blend_inpaint(
            result_tensor, merged_image, final_mask, kernel=31, sigma=15
        )[0]

        return (final_image, self._pil_to_tensor(image_pil), edit_mask)

    def kontext_edit(self,
                     image, positive_prompt,
                     seed, steps, cfg):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        image_pil = self._tensor_to_pil(image)
        mask_pil = Image.new("L", image_pil.size, 255)

        result_pil = self.pipe(
            prompt=positive_prompt,
            image=image_pil,
            mask_image=mask_pil,
            control_image=image_pil,
            height=image_pil.height,
            width=image_pil.width,
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=generator,
            max_sequence_length=512,
            strength=1.0
        ).images[0]

        final_image = self._pil_to_tensor(result_pil)
        mask = torch.zeros((1, final_image.shape[1], final_image.shape[2]), dtype=torch.float32, device=final_image.device)
        return (final_image, image, mask)

    def process(self, image, colored_image, 
                 merged_image, positive_prompt,
                total_mask, add_mask, remove_mask, add_prop_mask, fill_mask, 
                fine_edge, fix_perspective, edge_strength, color_strength, local_strength, grow_size,
                seed, steps, cfg, flag="precise_edit"):
        if flag == "foreground":
            return self.foreground_edit(merged_image, positive_prompt, add_prop_mask, fill_mask, total_mask, fix_perspective, grow_size, seed, steps, cfg)
        elif flag == "local":
            return self.local_edit(image, positive_prompt, fill_mask, local_strength, seed, steps, cfg)
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
