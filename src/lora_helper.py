from diffusers.models.attention_processor import FluxAttnProcessor2_0
from safetensors.torch import load_file
import re
import torch
from .layers_cache import MultiDoubleStreamBlockLoraProcessor, MultiSingleStreamBlockLoraProcessor

device = "cuda"

def load_safetensors(path):
    """Safely loads tensors from a file and maps them to the CPU."""
    return load_file(path, device="cpu")

def get_lora_count_from_checkpoint(checkpoint):
    """
    Infers the number of LoRA modules stored in a checkpoint by inspecting its keys.
    Also prints a sample of keys for debugging.
    """
    lora_indices = set()
    # Regex to find '..._loras.X.' where X is a number.
    indexed_pattern = re.compile(r'._loras\.(\d+)\.')
    found_keys = []

    for key in checkpoint.keys():
        match = indexed_pattern.search(key)
        if match:
            lora_indices.add(int(match.group(1)))
            if len(found_keys) < 5 and key not in found_keys:
                found_keys.append(key)

    if lora_indices:
        lora_count = max(lora_indices) + 1
        print("INFO: Auto-detected indexed LoRA keys in checkpoint.")
        print(f"      Found {lora_count} LoRA module(s).")
        print("      Sample keys:", found_keys)
        return lora_count

    # Fallback for legacy, non-indexed checkpoints.
    legacy_found = False
    legacy_key_sample = ""
    for key in checkpoint.keys():
        if '.q_lora.' in key:
            legacy_found = True
            legacy_key_sample = key
            break

    if legacy_found:
        print("INFO: Auto-detected legacy (non-indexed) LoRA keys in checkpoint.")
        print("      Assuming 1 LoRA module.")
        print("      Sample key:", legacy_key_sample)
        return 1

    print("WARNING: No LoRA keys found in the checkpoint.")
    return 0

def get_lora_ranks(checkpoint, num_loras):
    """
    Determines the rank for each LoRA module from the checkpoint.
    It supports both indexed (e.g., 'loras.0') and legacy non-indexed formats.
    """
    ranks = {}
    
    # First, try to find ranks for all indexed LoRA modules.
    for i in range(num_loras):
        # Find a key that uniquely identifies the i-th LoRA's down projection.
        rank_pattern = re.compile(f'._loras\.({i})\.down\.weight')
        for k, v in checkpoint.items():
            if rank_pattern.search(k):
                ranks[i] = v.shape[0]
                break
    
    # If not all ranks were found, there might be legacy keys or a mismatch.
    if len(ranks) != num_loras:
        # Fallback for single, non-indexed LoRA checkpoints.
        if num_loras == 1:
            for k, v in checkpoint.items():
                if ".q_lora.down.weight" in k:
                    return [v.shape[0]]

        # If still unresolved, use the rank of the very first LoRA found as a default for all.
        first_found_rank = next((v.shape[0] for k, v in checkpoint.items() if k.endswith(".down.weight")), None)
        
        if first_found_rank is None:
            raise ValueError("Could not determine any LoRA rank from the provided checkpoint.")

        # Return a list where missing ranks are filled with the first one found.
        return [ranks.get(i, first_found_rank) for i in range(num_loras)]

    # Return the list of ranks sorted by LoRA index.
    return [ranks[i] for i in range(num_loras)]


def load_checkpoint(local_path):
    if local_path is not None:
        if '.safetensors' in local_path:
            print(f"Loading .safetensors checkpoint from {local_path}")
            checkpoint = load_safetensors(local_path)
        else:
            print(f"Loading checkpoint from {local_path}")
            checkpoint = torch.load(local_path, map_location='cpu')
    return checkpoint


def prepare_lora_processors(checkpoint, lora_weights, transformer, cond_size, number=None):
    # Ensure processors match the transformer's dtype, but load on CPU to save VRAM
    target_device = "cpu"
    try:
        first_param = next(transformer.parameters())
        target_dtype = first_param.dtype
        # Force bfloat16 for LoRA if base model is in FP8 or quantized (non-floating point) format
        if str(target_dtype).endswith('float8_e4m3fn') or str(target_dtype).endswith('float8_e5m2') or not target_dtype.is_floating_point:
            target_dtype = torch.bfloat16
    except StopIteration:
        target_dtype = torch.bfloat16

    if number is None:
        number = get_lora_count_from_checkpoint(checkpoint)
        if number == 0:
            return {} 

        if lora_weights and len(lora_weights) != number:
            print(f"WARNING: Provided `lora_weights` length ({len(lora_weights)}) differs from detected LoRA count ({number}).")
            final_weights = (lora_weights + [1.0] * number)[:number]
            print(f"         Adjusting weights to: {final_weights}")
            lora_weights = final_weights
        elif not lora_weights:
            print(f"INFO: No `lora_weights` provided. Defaulting to weights of 1.0 for all {number} LoRAs.")
            lora_weights = [1.0] * number
    
    ranks = get_lora_ranks(checkpoint, number)
    print("INFO: Determined ranks for LoRA modules:", ranks)
    
    cond_widths = cond_size if isinstance(cond_size, list) else [cond_size] * number
    cond_heights = cond_size if isinstance(cond_size, list) else [cond_size] * number
    
    lora_attn_procs = {}
    double_blocks_idx = list(range(19))
    single_blocks_idx = list(range(38))
    
    # Get all attention processor names from the transformer to iterate over
    for name in transformer.attn_processors.keys():
        match = re.search(r'\.(\d+)\.', name)
        if not match:
            continue
        layer_index = int(match.group(1))

        if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
            lora_state_dicts = {
                key: value for key, value in checkpoint.items() 
                if f"transformer_blocks.{layer_index}." in key
            }

            lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
                dim=3072, ranks=ranks, network_alphas=ranks, lora_weights=lora_weights, 
                device=target_device, dtype=target_dtype, cond_widths=cond_widths, cond_heights=cond_heights, n_loras=number
            )

            for n in range(number):
                lora_prefix_q = f"{name}.q_loras.{n}"
                lora_prefix_k = f"{name}.k_loras.{n}"
                lora_prefix_v = f"{name}.v_loras.{n}"
                lora_prefix_proj = f"{name}.proj_loras.{n}"
                
                lora_attn_procs[name].q_loras[n].down.weight.data = lora_state_dicts.get(f'{lora_prefix_q}.down.weight')
                lora_attn_procs[name].q_loras[n].up.weight.data = lora_state_dicts.get(f'{lora_prefix_q}.up.weight')
                lora_attn_procs[name].k_loras[n].down.weight.data = lora_state_dicts.get(f'{lora_prefix_k}.down.weight')
                lora_attn_procs[name].k_loras[n].up.weight.data = lora_state_dicts.get(f'{lora_prefix_k}.up.weight')
                lora_attn_procs[name].v_loras[n].down.weight.data = lora_state_dicts.get(f'{lora_prefix_v}.down.weight')
                lora_attn_procs[name].v_loras[n].up.weight.data = lora_state_dicts.get(f'{lora_prefix_v}.up.weight')
                lora_attn_procs[name].proj_loras[n].down.weight.data = lora_state_dicts.get(f'{lora_prefix_proj}.down.weight')
                lora_attn_procs[name].proj_loras[n].up.weight.data = lora_state_dicts.get(f'{lora_prefix_proj}.up.weight')
                lora_attn_procs[name].to(device=target_device, dtype=target_dtype)
        
        elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
            lora_state_dicts = {
                key: value for key, value in checkpoint.items() 
                if f"single_transformer_blocks.{layer_index}." in key
            }
            
            lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
                dim=3072, ranks=ranks, network_alphas=ranks, lora_weights=lora_weights, 
                device=target_device, dtype=target_dtype, cond_widths=cond_widths, cond_heights=cond_heights, n_loras=number
            )

            for n in range(number):
                lora_prefix_q = f"{name}.q_loras.{n}"
                lora_prefix_k = f"{name}.k_loras.{n}"
                lora_prefix_v = f"{name}.v_loras.{n}"
                
                lora_attn_procs[name].q_loras[n].down.weight.data = lora_state_dicts.get(f'{lora_prefix_q}.down.weight')
                lora_attn_procs[name].q_loras[n].up.weight.data = lora_state_dicts.get(f'{lora_prefix_q}.up.weight')
                lora_attn_procs[name].k_loras[n].down.weight.data = lora_state_dicts.get(f'{lora_prefix_k}.down.weight')
                lora_attn_procs[name].k_loras[n].up.weight.data = lora_state_dicts.get(f'{lora_prefix_k}.up.weight')
                lora_attn_procs[name].v_loras[n].down.weight.data = lora_state_dicts.get(f'{lora_prefix_v}.down.weight')
                lora_attn_procs[name].v_loras[n].up.weight.data = lora_state_dicts.get(f'{lora_prefix_v}.up.weight')
                lora_attn_procs[name].to(device=target_device, dtype=target_dtype)
    return lora_attn_procs