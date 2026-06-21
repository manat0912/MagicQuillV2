import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxIPAdapterMixin, FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils  import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from torchvision.transforms.functional import pad
from diffusers.models.attention_processor import FluxAttnProcessor2_0
from .lora_helper import prepare_lora_processors, load_checkpoint
from .layers_cache import MultiDoubleStreamBlockLoraProcessor, MultiSingleStreamBlockLoraProcessor
import re


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def prepare_latent_image_ids_(height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height, device=device)[:, None]  # y
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width, device=device)[None, :]  # x
    return latent_image_ids


def prepare_latent_subject_ids(height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height, device=device)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width, device=device)[None, :]
    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )
    return latent_image_ids.to(device=device, dtype=dtype)


def resize_position_encoding(
    batch_size, original_height, original_width, target_height, target_width, device, dtype
):
    latent_image_ids = prepare_latent_image_ids_(original_height // 2, original_width // 2, device, dtype)
    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    scale_h = original_height / target_height
    scale_w = original_width / target_width
    latent_image_ids_resized = torch.zeros(target_height // 2, target_width // 2, 3, device=device, dtype=dtype)
    latent_image_ids_resized[..., 1] = (
        latent_image_ids_resized[..., 1] + torch.arange(target_height // 2, device=device)[:, None] * scale_h
    )
    latent_image_ids_resized[..., 2] = (
        latent_image_ids_resized[..., 2] + torch.arange(target_width // 2, device=device)[None, :] * scale_w
    )

    cond_latent_image_id_height, cond_latent_image_id_width, cond_latent_image_id_channels = (
        latent_image_ids_resized.shape
    )
    cond_latent_image_ids = latent_image_ids_resized.reshape(
        cond_latent_image_id_height * cond_latent_image_id_width, cond_latent_image_id_channels
    )
    return latent_image_ids, cond_latent_image_ids


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class FluxKontextControlPipeline(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
):
    r"""
    The Flux Kontext pipeline for image-to-image and text-to-image generation with control module.

    Reference: https://bfl.ai/announcements/flux-1-kontext-dev

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    @property
    def _execution_device(self):
        if hasattr(self, "transformer") and self.transformer is not None:
            return self.transformer.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def device(self):
        if hasattr(self, "transformer") and self.transformer is not None:
            return self.transformer.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=None,
            feature_extractor=None,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        # Flux latents are packed into 2x2 patches, so use VAE factor multiplied by patch size for image processing
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 128
        self.latent_channels = self.vae.config.latent_channels if getattr(self, "vae", None) else 16
        self.control_lora_processors: Dict[str, Dict[str, Any]] = {}
        self.control_lora_cond_sizes: Dict[str, Any] = {}
        self.control_lora_weights: Dict[str, Any] = {}
        self.current_control_type: Optional[Union[str, List[str]]] = None

    def load_control_loras(self, lora_config: Dict[str, Dict[str, Any]]):
        """
        Loads and prepares LoRA attention processors for different control types.
        Args:
            lora_config: A dict where keys are control types (e.g., 'edge') and values are dicts
                containing 'path', 'lora_weights', and 'cond_size'.
        """
        for control_type, config in lora_config.items():
            print(f"Loading LoRA for control type: {control_type}")
            checkpoint = load_checkpoint(config["path"])
            processors = prepare_lora_processors(
                checkpoint=checkpoint,
                lora_weights=config["lora_weights"],
                transformer=self.transformer,
                cond_size=config["cond_size"],
                number=len(config["lora_weights"]) if config.get("lora_weights") is not None else None,
            )
            self.control_lora_processors[control_type] = processors
            self.control_lora_cond_sizes[control_type] = config["cond_size"]
            self.control_lora_weights[control_type] = config["lora_weights"]
        print("All control LoRAs loaded and prepared.")

    def _combine_control_loras(self, control_types: List[str]):
        """
        Combines multiple control LoRAs into a single set of attention processors.
        """
        if not control_types:
            return FluxAttnProcessor2_0()

        try:
            first_param = next(self.transformer.parameters())
            target_device = first_param.device
            target_dtype = first_param.dtype
            # The base transformer may be quantized (GGUF -> uint8, or fp8); LoRA
            # layers are trainable nn.Linear params and must be a real floating
            # point dtype, so fall back to bf16 like prepare_lora_processors does.
            if (
                str(target_dtype).endswith("float8_e4m3fn")
                or str(target_dtype).endswith("float8_e5m2")
                or not target_dtype.is_floating_point
            ):
                target_dtype = torch.bfloat16
        except StopIteration:
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            target_dtype = torch.float32

        combined_procs = {}
        all_lora_weights = []
        
        total_loras = 0
        all_ranks = []
        all_cond_sizes = []

        for control_type in control_types:
            procs = self.control_lora_processors.get(control_type)
            if not procs:
                raise ValueError(f"Control type '{control_type}' not loaded.")
            conf_weights = self.control_lora_weights.get(control_type)
            if conf_weights is None:
                raise ValueError(f"Control type '{control_type}' has no configured lora_weights.")
            all_lora_weights.extend(conf_weights)
            
            first_proc = next(iter(procs.values()))
            n_loras_in_control = first_proc.n_loras
            total_loras += n_loras_in_control
            
            proc_ranks = [lora.down.weight.shape[0] for lora in first_proc.q_loras]
            all_ranks.extend(proc_ranks)

            cond_size = self.control_lora_cond_sizes[control_type]
            cond_sizes = [cond_size] * n_loras_in_control if not isinstance(cond_size, list) else cond_size
            all_cond_sizes.extend(cond_sizes)

        for name in self.transformer.attn_processors.keys():
            match = re.search(r'\.(\d+)\.', name)
            if not match:
                continue
            layer_index = int(match.group(1))

            if name.startswith("transformer_blocks"):
                new_proc = MultiDoubleStreamBlockLoraProcessor(
                    dim=3072, ranks=all_ranks, network_alphas=all_ranks, lora_weights=all_lora_weights,
                    device=target_device, dtype=target_dtype, 
                    cond_widths=all_cond_sizes, cond_heights=all_cond_sizes, n_loras=total_loras
                )
            elif name.startswith("single_transformer_blocks"):
                new_proc = MultiSingleStreamBlockLoraProcessor(
                    dim=3072, ranks=all_ranks, network_alphas=all_ranks, lora_weights=all_lora_weights,
                    device=target_device, dtype=target_dtype,
                    cond_widths=all_cond_sizes, cond_heights=all_cond_sizes, n_loras=total_loras
                )
            else:
                continue
            
            lora_idx_offset = 0
            for control_type in control_types:
                source_proc = self.control_lora_processors[control_type][name]
                for i in range(source_proc.n_loras):
                    current_lora_idx = lora_idx_offset + i
                    new_proc.q_loras[current_lora_idx].load_state_dict(source_proc.q_loras[i].state_dict())
                    new_proc.k_loras[current_lora_idx].load_state_dict(source_proc.k_loras[i].state_dict())
                    new_proc.v_loras[current_lora_idx].load_state_dict(source_proc.v_loras[i].state_dict())
                    if hasattr(new_proc, 'proj_loras'):
                        new_proc.proj_loras[current_lora_idx].load_state_dict(source_proc.proj_loras[i].state_dict())

                lora_idx_offset += source_proc.n_loras

            combined_procs[name] = new_proc.to(device=target_device, dtype=target_dtype)
            
        return combined_procs

    def set_gamma_values(self, gammas: List[float]):
        """
        Set gamma values for bias control modulation on current attention processors and attention modules.
        """
        print(f"Setting gamma values to: {gammas}")
        try:
            first_param = next(self.transformer.parameters())
            device = first_param.device
            dtype = first_param.dtype
            if (
                str(dtype).endswith("float8_e4m3fn")
                or str(dtype).endswith("float8_e5m2")
                or not dtype.is_floating_point
            ):
                dtype = torch.bfloat16
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float32
        gamma_tensor = torch.tensor(gammas, device=device, dtype=dtype)
        for name, attn_processor in self.transformer.attn_processors.items():
            if hasattr(attn_processor, 'q_loras'):
                setattr(attn_processor, 'c_factor', gamma_tensor)

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
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
        
        import logging
        logger_token = logging.getLogger("transformers.tokenization_utils_base")
        old_level = logger_token.level
        logger_token.setLevel(logging.ERROR)
        try:
            untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids
        finally:
            logger_token.setLevel(old_level)

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])

        prompt_embeds = self.text_encoder_2(text_input_ids.to(self.text_encoder_2.device), output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._get_clip_prompt_embeds
    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
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
        
        import logging
        logger_token = logging.getLogger("transformers.tokenization_utils_base")
        old_level = logger_token.level
        logger_token.setLevel(logging.ERROR)
        try:
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        finally:
            logger_token.setLevel(old_level)
            
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
        prompt_embeds = self.text_encoder(text_input_ids.to(self.text_encoder.device), output_hidden_states=False)

        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        device = device or self._execution_device

        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    # Adapted from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._prepare_latent_image_ids
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._pack_latents
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._unpack_latents
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        self.vae.to(self.device)
        image = image.to(dtype=self.vae.dtype, device=self.vae.device)
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        
        self.vae.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return image_latents

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        image,
        subject_images,
        spatial_images,
        latents=None,
        cond_size=512,
        num_subject_images: int = 0,
        num_spatial_images: int = 0,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        height_cond = 2 * (cond_size // (self.vae_scale_factor * 2))
        width_cond = 2 * (cond_size // (self.vae_scale_factor * 2))

        image_latents = image_ids = None
        image_latent_h = 0

        # Prepare noise latents
        shape = (batch_size, num_channels_latents, height, width)
        if latents is None:
            noise_latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            noise_latents = latents.to(device=device, dtype=dtype)

        noise_latents = self._pack_latents(noise_latents, batch_size, num_channels_latents, height, width)
        noise_latent_image_ids, cond_latent_image_ids_resized = resize_position_encoding(
            batch_size, height, width, height_cond, width_cond, device, dtype
        )
        noise_latent_image_ids[..., 0] = 0

        cond_latents_to_concat = []
        latents_ids_to_concat = [noise_latent_image_ids]

        # 1. Prepare `image` (Kontext) latents
        if image is not None:
            image = image.to(device=device, dtype=dtype)
            if image.shape[1] != self.latent_channels:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            else:
                image_latents = image

            image_latent_h, image_latent_w = image_latents.shape[2:]
            image_latents = self._pack_latents(
                image_latents, batch_size, num_channels_latents, image_latent_h, image_latent_w
            )
            image_ids = self._prepare_latent_image_ids(
                batch_size, image_latent_h // 2, image_latent_w // 2, device, dtype
            )
            image_ids[..., 0] = 1  # Mark as condition
            latents_ids_to_concat.append(image_ids)

        # 2. Prepare `subject_images` latents
        if subject_images is not None and num_subject_images > 0:
            subject_images = subject_images.to(device=device, dtype=dtype)
            subject_image_latents = self._encode_vae_image(image=subject_images, generator=generator)
            subject_latent_h, subject_latent_w = subject_image_latents.shape[2:]
            subject_latents = self._pack_latents(
                subject_image_latents, batch_size, num_channels_latents, subject_latent_h, subject_latent_w
            )

            latent_subject_ids = prepare_latent_subject_ids(height_cond // 2, width_cond // 2, device, dtype)
            latent_subject_ids[..., 0] = 1
            latent_subject_ids[:, 1] += image_latent_h // 2
            subject_latent_image_ids = torch.cat([latent_subject_ids for _ in range(num_subject_images)], dim=0)

            cond_latents_to_concat.append(subject_latents)
            latents_ids_to_concat.append(subject_latent_image_ids)

        # 3. Prepare `spatial_images` latents
        if spatial_images is not None and num_spatial_images > 0:
            spatial_images = spatial_images.to(device=device, dtype=dtype)
            spatial_image_latents = self._encode_vae_image(image=spatial_images, generator=generator)
            spatial_latent_h, spatial_latent_w = spatial_image_latents.shape[2:]
            cond_latents = self._pack_latents(
                spatial_image_latents, batch_size, num_channels_latents, spatial_latent_h, spatial_latent_w
            )
            cond_latent_image_ids_resized[..., 0] = 2  # Mark as condition
            cond_latent_image_ids = torch.cat(
                [cond_latent_image_ids_resized for _ in range(num_spatial_images)], dim=0
            )

            cond_latents_to_concat.append(cond_latents)
            latents_ids_to_concat.append(cond_latent_image_ids)

        cond_latents = torch.cat(cond_latents_to_concat, dim=1) if cond_latents_to_concat else None
        latent_image_ids = torch.cat(latents_ids_to_concat, dim=0)

        return noise_latents, image_latents, cond_latents, latent_image_ids

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def fit_kontext_resolution(self, image):
        """
        Snap an image to the nearest Kontext-friendly resolution, optionally
        capped to `self.max_working_side` (long side, in px) to keep the token
        count / activation memory within a small (e.g. 12 GB) VRAM budget.
        Returns (width, height), both multiples of vae_scale_factor * 2.
        """
        img = image[0] if isinstance(image, list) else image
        image_height, image_width = self.image_processor.get_default_height_width(img)
        aspect_ratio = image_width / image_height
        _, image_width, image_height = min(
            (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
        )
        multiple_of = self.vae_scale_factor * 2
        image_width = image_width // multiple_of * multiple_of
        image_height = image_height // multiple_of * multiple_of

        max_side = getattr(self, "max_working_side", None)
        if max_side:
            longest = max(image_width, image_height)
            if longest > max_side:
                scale = max_side / longest
                image_width = max(multiple_of, int(image_width * scale) // multiple_of * multiple_of)
                image_height = max(multiple_of, int(image_height * scale) // multiple_of * multiple_of)
        return image_width, image_height

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        mask_image: Optional[Union[torch.Tensor, Image.Image]] = None,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        cond_size: int = 512,
        control_dict: Optional[Dict[str, Any]] = None,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        control_dict = control_dict or {}

        spatial_images = control_dict.get("spatial_images", [])
        num_spatial_images = len(spatial_images)
        subject_images = control_dict.get("subject_images", [])
        num_subject_images = len(subject_images)

        requested_control_type = control_dict.get("type") or None

        if requested_control_type and isinstance(requested_control_type, str):
            requested_control_type = [requested_control_type]

        if not requested_control_type and self.current_control_type:
            print("Reverting to default attention processors.")
            self.transformer.set_attn_processor(FluxAttnProcessor2_0())
            self.current_control_type = None
        elif requested_control_type != self.current_control_type:
            if requested_control_type:
                print(f"Switching to LoRA control type(s): {requested_control_type}")
                processors = self._combine_control_loras(requested_control_type)
                self.transformer.set_attn_processor(processors)
                self.cond_size = self.control_lora_cond_sizes[requested_control_type[0]]
                self.current_control_type = requested_control_type

        if hasattr(self, "cond_size"):
            selected_cond_size = self.cond_size
            if isinstance(selected_cond_size, list) and len(selected_cond_size) > 0:
                cond_size = int(selected_cond_size[0])
            elif isinstance(selected_cond_size, int):
                cond_size = selected_cond_size

        if requested_control_type:
            raw_gammas = control_dict.get("gammas", [])
            if not isinstance(raw_gammas, list):
                raw_gammas = [raw_gammas]
            flattened_gammas: List[float] = []
            for g in raw_gammas:
                if isinstance(g, (list, tuple)):
                    flattened_gammas.extend([float(x) for x in g])
                else:
                    flattened_gammas.append(float(g))
            if len(flattened_gammas) > 0:
                self.set_gamma_values(flattened_gammas)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 3. Preprocess images
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            img = image[0] if isinstance(image, list) else image
            image_width, image_height = self.fit_kontext_resolution(img)
            image = self.image_processor.resize(image, image_height, image_width)
            image = self.image_processor.preprocess(image, image_height, image_width)

        if len(subject_images) > 0:
            subject_image_ls = []
            for subject_image in subject_images:
                w, h = subject_image.size[:2]
                scale = cond_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                subject_image = self.image_processor.preprocess(subject_image, height=new_h, width=new_w)
                subject_image = subject_image.to(dtype=self.vae.dtype)
                pad_h = cond_size - subject_image.shape[-2]
                pad_w = cond_size - subject_image.shape[-1]
                subject_image = pad(
                    subject_image, padding=(int(pad_w / 2), int(pad_h / 2), int(pad_w / 2), int(pad_h / 2)), fill=0
                )
                subject_image_ls.append(subject_image)
            subject_images = torch.cat(subject_image_ls, dim=-2)
        else:
            subject_images = None

        if len(spatial_images) > 0:
            condition_image_ls = []
            for img in spatial_images:
                condition_image = self.image_processor.preprocess(img, height=cond_size, width=cond_size)
                condition_image = condition_image.to(dtype=self.vae.dtype)
                condition_image_ls.append(condition_image)
            spatial_images = torch.cat(condition_image_ls, dim=-2)
        else:
            spatial_images = None

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents, cond_latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            image,
            subject_images,
            spatial_images,
            latents,
            cond_size,
            num_subject_images=num_subject_images,
            num_spatial_images=num_spatial_images,
        )

        # 4.5. Prepare mask packed tensor if mask_image is provided
        mask_packed = None
        initial_noise = None
        if mask_image is not None and image is not None:
            img = image[0] if isinstance(image, list) else image
            image_width, image_height = self.fit_kontext_resolution(img)
            
            if isinstance(mask_image, Image.Image):
                mask_tensor = np.array(mask_image.convert("L")).astype(np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask_tensor).to(device=device, dtype=prompt_embeds.dtype)
            elif isinstance(mask_image, np.ndarray):
                mask_tensor = mask_image.astype(np.float32)
                if mask_tensor.max() > 1.0:
                    mask_tensor /= 255.0
                mask_tensor = torch.from_numpy(mask_tensor).to(device=device, dtype=prompt_embeds.dtype)
            else:
                mask_tensor = mask_image.to(device=device, dtype=prompt_embeds.dtype)
            
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            elif mask_tensor.ndim == 3:
                if mask_tensor.shape[0] == 1:
                    mask_tensor = mask_tensor.unsqueeze(0)
                else:
                    mask_tensor = mask_tensor.unsqueeze(1)
            
            h_latent = image_height // 16
            w_latent = image_width // 16
            mask_resized = F.interpolate(mask_tensor, size=(h_latent, w_latent), mode="nearest")
            mask_packed = mask_resized.view(batch_size * num_images_per_prompt, h_latent * w_latent, 1)
            initial_noise = latents.clone()

        # 5. Prepare timesteps
        #
        # IMPORTANT: When num_inference_steps is 18 or fewer (fill-brush / local-edit
        # paths), we SKIP the custom sigma linspace and use the scheduler's native
        # num_inference_steps path.  The sigma linspace -> mu-shift path was causing
        # the FlowMatchEulerDiscrete scheduler to silently re-expand the schedule
        # into 100+ denoising steps regardless of the value we passed in.
        # For larger step counts (user-set on other modes) we keep the existing path.
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        if sigmas is None and num_inference_steps <= 18:
            # Fast-path: let the scheduler produce exactly num_inference_steps timesteps.
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                mu=mu,
            )
        else:
            # Standard path: caller supplied custom sigmas, or step count is large.
            if sigmas is None:
                sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                sigmas=sigmas,
                mu=mu,
            )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        print(f"[INFO] num_inference_steps={num_inference_steps}, actual timesteps={len(timesteps)}")

        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        for name, attn_processor in self.transformer.attn_processors.items():
            if hasattr(attn_processor, "bank_kv"):
                attn_processor.bank_kv.clear()
            if hasattr(attn_processor, "bank_attn"):
                attn_processor.bank_attn = None

        if cond_latents is not None:
            latent_model_input = latents
            if image_latents is not None:
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=1)
            print(latent_model_input.shape)
            warmup_latents = latent_model_input
            warmup_latent_ids = latent_image_ids
            t = torch.tensor([timesteps[0]], device=device)
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            _ = self.transformer(
                hidden_states=warmup_latents,
                cond_hidden_states=cond_latents,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=warmup_latent_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]

        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latent_model_input, image_latents], dim=1)
                
                self._current_timestep = t
                dtype = getattr(self.transformer, "dtype", torch.bfloat16)
                timestep = t.expand(latents.shape[0]).to(dtype)
                guidance_input = guidance.to(dtype) if isinstance(guidance, torch.Tensor) else guidance
                
                noise_pred = self.transformer(
                    hidden_states=latent_model_input.to(dtype),
                    cond_hidden_states=cond_latents.to(dtype) if cond_latents is not None else None,
                    timestep=(timestep / 1000).to(dtype),
                    guidance=guidance_input,
                    pooled_projections=pooled_prompt_embeds.to(dtype),
                    encoder_hidden_states=prompt_embeds.to(dtype),
                    txt_ids=text_ids.to(dtype),
                    img_ids=latent_image_ids.to(dtype),
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                noise_pred = noise_pred[:, : latents.size(1)]

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                # Step-by-step latent blending for inpainting
                if mask_packed is not None and image_latents is not None:
                    # Cast image_latents, initial_noise, and mask_packed to match latents' device and dtype
                    image_latents = image_latents.to(device=latents.device, dtype=latents.dtype)
                    initial_noise = initial_noise.to(device=latents.device, dtype=latents.dtype)
                    mask_packed = mask_packed.to(device=latents.device, dtype=latents.dtype)
                    
                    # --- ADD THIS SHAPE PROTECTION FOR PACKED FLUX LATENTS ---
                    if image_latents.shape != latents.shape:
                        image_latents = image_latents.expand_as(latents)
                    if initial_noise.shape != latents.shape:
                        initial_noise = initial_noise.expand_as(latents)
                    # --------------------------------------------------------
                    
                    if i < len(timesteps) - 1:
                        next_t = timesteps[i + 1]
                        normalized_t = (next_t / 1000.0).to(device=latents.device, dtype=latents.dtype)
                        noisy_image_latents = (1.0 - normalized_t) * image_latents + normalized_t * initial_noise
                    else:
                        noisy_image_latents = image_latents
                        
                    # Handle dimension matching for channel-packed Flux latents
                    if mask_packed.shape != latents.shape:
                        mask_packed = mask_packed.expand_as(latents)
                        
                    latents = torch.where(mask_packed > 0.5, latents, noisy_image_latents)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            self.vae.to(self.device)
            latents = latents.to(dtype=self.vae.dtype, device=self.vae.device)
            image = self.vae.decode(latents, return_dict=False)[0]
            self.vae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
