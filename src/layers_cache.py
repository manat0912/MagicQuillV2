import inspect
import math
from typing import Callable, List, Optional, Tuple, Union, Any, Dict
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from diffusers.models.attention_processor import Attention

TXTLEN = 128
KONTEXT = False
    
class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        cond_widths: Optional[List[int]] = None,
        cond_heights: Optional[List[int]] = None,
        lora_index: int = 0,
        n_loras: int = 1,
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)
        
        self.cond_heights = cond_heights if cond_heights is not None else [512]
        self.cond_widths = cond_widths if cond_widths is not None else [512]
        self.lora_index = lora_index
        self.n_loras = n_loras

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        batch_size = hidden_states.shape[0]
        
        cond_sizes = [(w // 8 * h // 8 * 16 // 64) for w, h in zip(self.cond_widths, self.cond_heights)]
        total_cond_size = sum(cond_sizes)
        block_size = hidden_states.shape[1] - total_cond_size
        
        offset = sum(cond_sizes[:self.lora_index])
        current_cond_size = cond_sizes[self.lora_index]

        shape = (batch_size, hidden_states.shape[1], 3072)
        mask = torch.ones(shape, device=hidden_states.device, dtype=dtype) 
        
        mask[:, :block_size + offset, :] = 0
        mask[:, block_size + offset + current_cond_size:, :] = 0
        
        hidden_states = mask * hidden_states
        
        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)
    

class MultiSingleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, ranks: List[int], lora_weights: List[float], network_alphas: List[float], device=None, dtype=None, cond_widths: Optional[List[int]] = None, cond_heights: Optional[List[int]] = None, n_loras=1):
        super().__init__()
        self.n_loras = n_loras
        self.cond_widths = cond_widths if cond_widths is not None else [512]
        self.cond_heights = cond_heights if cond_heights is not None else [512]
        
        self.q_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i], network_alphas[i], device=device, dtype=dtype, cond_widths=self.cond_widths, cond_heights=self.cond_heights, lora_index=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.k_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i], network_alphas[i], device=device, dtype=dtype, cond_widths=self.cond_widths, cond_heights=self.cond_heights, lora_index=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.v_loras = nn.ModuleList([
            LoRALinearLayer(dim, dim, ranks[i], network_alphas[i], device=device, dtype=dtype, cond_widths=self.cond_widths, cond_heights=self.cond_heights, lora_index=i, n_loras=n_loras)
            for i in range(n_loras)
        ])
        self.lora_weights = lora_weights
        self.bank_attn = None
        self.bank_kv: List[torch.Tensor] = []
        

    def __call__(self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        use_cond = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        scaled_seq_len = hidden_states.shape[1]
        
        cond_sizes = [(w // 8 * h // 8 * 16 // 64) for w, h in zip(self.cond_widths, self.cond_heights)]
        total_cond_size = sum(cond_sizes)
        block_size = scaled_seq_len - total_cond_size
        
        scaled_cond_sizes = cond_sizes
        scaled_block_size = block_size
        
        global TXTLEN
        global KONTEXT
        if KONTEXT:
            img_start, img_end = TXTLEN, (TXTLEN + block_size) // 2
        else:
            img_start, img_end = TXTLEN, block_size
        cond_start, cond_end = block_size, scaled_seq_len

        cache = len(self.bank_kv) == 0
        
        if cache:
            query = attn.to_q(hidden_states) 
            key = attn.to_k(hidden_states) 
            value = attn.to_v(hidden_states) 
            for i in range(self.n_loras):
                query = query + self.lora_weights[i] * self.q_loras[i](hidden_states)
                key = key + self.lora_weights[i] * self.k_loras[i](hidden_states)
                value = value + self.lora_weights[i] * self.v_loras[i](hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            self.bank_kv.extend([key[:, :, scaled_block_size:, :], value[:, :, scaled_block_size:, :]])
            
            if attn.norm_q is not None: query = attn.norm_q(query)
            if attn.norm_k is not None: key = attn.norm_k(key)

            if image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb
                query, key = apply_rotary_emb(query, image_rotary_emb), apply_rotary_emb(key, image_rotary_emb)
        
            mask = torch.ones((scaled_seq_len, scaled_seq_len), device=hidden_states.device)
            mask[ :scaled_block_size, :] = 0
            
            current_offset = 0
            for i in range(self.n_loras):
                start, end = scaled_block_size + current_offset, scaled_block_size + current_offset + scaled_cond_sizes[i]
                mask[start:end, start:end] = 0
                current_offset += scaled_cond_sizes[i]

            mask *= -1e20
            
            c_factor = getattr(self, "c_factor", None)
            if c_factor is not None:
                # print(f"Using c_factor: {c_factor}")
                current_offset = 0
                for i in range(self.n_loras):
                    bias = torch.log(c_factor[i])
                    cond_i_start, cond_i_end = cond_start + current_offset, cond_start + current_offset + scaled_cond_sizes[i]
                    mask[img_start:img_end, cond_i_start:cond_i_end] = bias
                    current_offset += scaled_cond_sizes[i]

            # c_factor_kontext = getattr(self, "c_factor_kontext", None)
            # if c_factor_kontext is not None:
            #     bias = torch.log(c_factor_kontext)
            #     kontext_start, kontext_end = img_end, block_size
            #     mask[img_start:img_end, kontext_start:kontext_end] = bias
            #     mask[kontext_start:kontext_end, img_start:img_end] = bias

            # mask[kontext_start:kontext_end, kontext_end:] = -1e20
            
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=mask.to(query.dtype))
            self.bank_attn = hidden_states[:, :, scaled_block_size:, :]
            
        else:
            query, key, value = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)

            inner_dim = query.shape[-1]
            head_dim = inner_dim // attn.heads
            
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            key = torch.cat([key[:, :, :scaled_block_size, :], self.bank_kv[0]], dim=-2)
            value = torch.cat([value[:, :, :scaled_block_size, :], self.bank_kv[1]], dim=-2)

            if attn.norm_q is not None: query = attn.norm_q(query)
            if attn.norm_k is not None: key = attn.norm_k(key)

            if image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb
                query, key = apply_rotary_emb(query, image_rotary_emb), apply_rotary_emb(key, image_rotary_emb)
            
            query = query[:, :, :scaled_block_size, :]
            
            attn_mask = None
            c_factor = getattr(self, "c_factor", None)
            if c_factor is not None:
                # print(f"Using c_factor: {c_factor}")
                attn_mask = torch.zeros((query.shape[2], key.shape[2]), device=query.device, dtype=query.dtype)
                current_offset = 0
                for i in range(self.n_loras):
                    bias = torch.log(c_factor[i])
                    cond_i_start, cond_i_end = cond_start + current_offset, cond_start + current_offset + scaled_cond_sizes[i]
                    attn_mask[img_start:img_end, cond_i_start:cond_i_end] = bias
                    current_offset += scaled_cond_sizes[i]

            # c_factor_kontext = getattr(self, "c_factor_kontext", None)
            # if c_factor_kontext is not None:
            #     if attn_mask is None:
            #         attn_mask = torch.zeros((query.shape[2], key.shape[2]), device=query.device, dtype=query.dtype)
            #     bias = torch.log(c_factor_kontext)
            #     kontext_start, kontext_end = img_end, block_size
            #     attn_mask[img_start:img_end, kontext_start:kontext_end] = bias
            #     attn_mask[kontext_start:kontext_end, img_start:img_end] = bias

            # attn_mask[kontext_start:kontext_end, kontext_end:] = -1e20

            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attn_mask)
            if self.bank_attn is not None: hidden_states = torch.cat([hidden_states, self.bank_attn], dim=-2)
            
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        cond_hidden_states = hidden_states[:, block_size:,:]
        hidden_states = hidden_states[:, : block_size,:]

        return (hidden_states, cond_hidden_states) if use_cond else hidden_states


class MultiDoubleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, ranks: List[int], lora_weights: List[float], network_alphas: List[float], device=None, dtype=None, cond_widths: Optional[List[int]] = None, cond_heights: Optional[List[int]] = None, n_loras=1):
        super().__init__()
        
        self.n_loras = n_loras
        self.cond_widths = cond_widths if cond_widths is not None else [512]
        self.cond_heights = cond_heights if cond_heights is not None else [512]
        self.q_loras = nn.ModuleList([LoRALinearLayer(dim, dim, ranks[i], network_alphas[i], device=device, dtype=dtype, cond_widths=self.cond_widths, cond_heights=self.cond_heights, lora_index=i, n_loras=n_loras) for i in range(n_loras)])
        self.k_loras = nn.ModuleList([LoRALinearLayer(dim, dim, ranks[i], network_alphas[i], device=device, dtype=dtype, cond_widths=self.cond_widths, cond_heights=self.cond_heights, lora_index=i, n_loras=n_loras) for i in range(n_loras)])
        self.v_loras = nn.ModuleList([LoRALinearLayer(dim, dim, ranks[i], network_alphas[i], device=device, dtype=dtype, cond_widths=self.cond_widths, cond_heights=self.cond_heights, lora_index=i, n_loras=n_loras) for i in range(n_loras)])
        self.proj_loras = nn.ModuleList([LoRALinearLayer(dim, dim, ranks[i], network_alphas[i], device=device, dtype=dtype, cond_widths=self.cond_widths, cond_heights=self.cond_heights, lora_index=i, n_loras=n_loras) for i in range(n_loras)])
        self.lora_weights = lora_weights
        self.bank_attn = None
        self.bank_kv: List[torch.Tensor] = []


    def __call__(self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        use_cond=False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        
        global TXTLEN
        global KONTEXT
        TXTLEN = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 128

        batch_size, _, _ = hidden_states.shape
        
        cond_sizes = [(w // 8 * h // 8 * 16 // 64) for w, h in zip(self.cond_widths, self.cond_heights)]
        block_size = hidden_states.shape[1] - sum(cond_sizes)
        
        scaled_seq_len = encoder_hidden_states.shape[1] + hidden_states.shape[1]
        scaled_cond_sizes = cond_sizes
        scaled_block_size = scaled_seq_len - sum(scaled_cond_sizes)
        
        if KONTEXT:
            img_start, img_end = TXTLEN, (TXTLEN + block_size) // 2
        else:
            img_start, img_end = TXTLEN, block_size
        cond_start, cond_end = scaled_block_size, scaled_seq_len

        inner_dim, head_dim = 3072, 3072 // attn.heads
        
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_added_q is not None: encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None: encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
        
        cache = len(self.bank_kv) == 0
        
        if cache:
            query, key, value = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
            for i in range(self.n_loras):
                query, key, value = query + self.lora_weights[i] * self.q_loras[i](hidden_states), key + self.lora_weights[i] * self.k_loras[i](hidden_states), value + self.lora_weights[i] * self.v_loras[i](hidden_states)

            query, key, value = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2), key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2), value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            self.bank_kv.extend([key[:, :, block_size:, :], value[:, :, block_size:, :]])

            if attn.norm_q is not None: query = attn.norm_q(query)
            if attn.norm_k is not None: key = attn.norm_k(key)
            
            query, key, value = torch.cat([encoder_hidden_states_query_proj, query], dim=2), torch.cat([encoder_hidden_states_key_proj, key], dim=2), torch.cat([encoder_hidden_states_value_proj, value], dim=2)

            if image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb              
                query, key = apply_rotary_emb(query, image_rotary_emb), apply_rotary_emb(key, image_rotary_emb)
            
            mask = torch.ones((scaled_seq_len, scaled_seq_len), device=hidden_states.device)
            mask[:scaled_block_size, :] = 0
            
            current_offset = 0
            for i in range(self.n_loras):
                start, end = scaled_block_size + current_offset, scaled_block_size + current_offset + scaled_cond_sizes[i]
                mask[start:end, start:end] = 0
                current_offset += scaled_cond_sizes[i]
            
            mask *= -1e20
            
            c_factor = getattr(self, "c_factor", None)
            if c_factor is not None:
                # print(f"Using c_factor: {c_factor}")
                current_offset = 0
                for i in range(self.n_loras):
                    bias = torch.log(c_factor[i])
                    cond_i_start, cond_i_end = cond_start + current_offset, cond_start + current_offset + scaled_cond_sizes[i]
                    mask[img_start:img_end, cond_i_start:cond_i_end] = bias
                    current_offset += scaled_cond_sizes[i]
            
            # c_factor_kontext = getattr(self, "c_factor_kontext", None)
            # if c_factor_kontext is not None:
            #     bias = torch.log(c_factor_kontext)
            #     kontext_start, kontext_end = img_end, block_size
            #     mask[img_start:img_end, kontext_start:kontext_end] = bias
            #     mask[kontext_start:kontext_end, img_start:img_end] = bias

            # mask[kontext_start:kontext_end, kontext_end:] = -1e20

            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=mask.to(query.dtype))
            self.bank_attn = hidden_states[:, :, scaled_block_size:, :]
        
        else:
            query, key, value = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
    
            query, key, value = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2), key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2), value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            key, value = torch.cat([key[:, :, :block_size, :], self.bank_kv[0]], dim=-2), torch.cat([value[:, :, :block_size, :], self.bank_kv[1]], dim=-2)
            
            if attn.norm_q is not None: query = attn.norm_q(query)
            if attn.norm_k is not None: key = attn.norm_k(key)

            query, key, value = torch.cat([encoder_hidden_states_query_proj, query], dim=2), torch.cat([encoder_hidden_states_key_proj, key], dim=2), torch.cat([encoder_hidden_states_value_proj, value], dim=2)

            if image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb
                query, key = apply_rotary_emb(query, image_rotary_emb), apply_rotary_emb(key, image_rotary_emb)
            
            query = query[:, :, :scaled_block_size, :]

            attn_mask = None
            c_factor = getattr(self, "c_factor", None)
            if c_factor is not None:
                # print(f"Using c_factor: {c_factor}")
                attn_mask = torch.zeros((query.shape[2], key.shape[2]), device=query.device, dtype=query.dtype)
                current_offset = 0
                for i in range(self.n_loras):
                    bias = torch.log(c_factor[i])
                    cond_i_start, cond_i_end = cond_start + current_offset, cond_start + current_offset + scaled_cond_sizes[i]
                    attn_mask[img_start:img_end, cond_i_start:cond_i_end] = bias
                    current_offset += scaled_cond_sizes[i]
            
            # c_factor_kontext = getattr(self, "c_factor_kontext", None)
            # if c_factor_kontext is not None:
            #     if attn_mask is None:
            #         attn_mask = torch.zeros((query.shape[2], key.shape[2]), device=query.device, dtype=query.dtype)
            #     bias = torch.log(c_factor_kontext)
            #     kontext_start, kontext_end = img_end, block_size
            #     attn_mask[img_start:img_end, kontext_start:kontext_end] = bias
            #     attn_mask[kontext_start:kontext_end, img_start:img_end] = bias
   
            # attn_mask[kontext_start:kontext_end, kontext_end:] = -1e20

            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attn_mask)
            if self.bank_attn is not None: hidden_states = torch.cat([hidden_states, self.bank_attn], dim=-2)
            
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        encoder_hidden_states, hidden_states = hidden_states[:, :encoder_hidden_states.shape[1]], hidden_states[:, encoder_hidden_states.shape[1]:]

        hidden_states = attn.to_out[0](hidden_states)
        for i in range(self.n_loras):
             hidden_states = hidden_states + self.lora_weights[i] * self.proj_loras[i](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        
        cond_hidden_states = hidden_states[:, block_size:,:]
        hidden_states = hidden_states[:, :block_size,:]
        
        return (hidden_states, encoder_hidden_states, cond_hidden_states) if use_cond else (encoder_hidden_states, hidden_states)