from PIL import Image
from datasets import Dataset
from torchvision import transforms
import random
import torch
import os
from .pipeline_flux_kontext_control import PREFERRED_KONTEXT_RESOLUTIONS
from .jsonl_datasets_kontext import make_train_dataset_inpaint_mask
import numpy as np
import json
from .generate_diff_mask import generate_final_difference_mask, align_images

Image.MAX_IMAGE_PIXELS = None
BLEND_PIXEL_VALUES = True

def multiple_16(num: float):
    return int(round(num / 16) * 16)
    
def choose_kontext_resolution_from_wh(width: int, height: int):
    aspect_ratio = width / max(1, height)
    _, best_w, best_h = min(
        (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
    )
    return best_w, best_h

def collate_fn(examples):
    if examples[0].get("cond_pixel_values") is not None:
        cond_pixel_values = torch.stack([example["cond_pixel_values"] for example in examples])
        cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
    else:
        cond_pixel_values = None
    if examples[0].get("source_pixel_values") is not None: 
        source_pixel_values = torch.stack([example["source_pixel_values"] for example in examples])
        source_pixel_values = source_pixel_values.to(memory_format=torch.contiguous_format).float()
    else:
        source_pixel_values = None

    target_pixel_values = torch.stack([example["pixel_values"] for example in examples])
    target_pixel_values = target_pixel_values.to(memory_format=torch.contiguous_format).float()
    token_ids_clip = torch.stack([example["token_ids_clip"] for example in examples])
    token_ids_t5 = torch.stack([example["token_ids_t5"] for example in examples])

    mask_values = None
    if examples[0].get("mask_values") is not None:
        mask_values = torch.stack([example["mask_values"] for example in examples])
        mask_values = mask_values.to(memory_format=torch.contiguous_format).float()

    return {
        "cond_pixel_values": cond_pixel_values,
        "source_pixel_values": source_pixel_values,
        "pixel_values": target_pixel_values,
        "text_ids_1": token_ids_clip,
        "text_ids_2": token_ids_t5,
        "mask_values": mask_values,
    }


# New dataset for local_edits JSON mapping with on-the-fly diff-mask generation
def make_train_dataset_local_edits(args, tokenizers, accelerator=None):
    # Read JSON entries
    with open(args.local_edits_json, "r", encoding="utf-8") as f:
        entries = json.load(f)

    samples = []
    for item in entries:
        rel_path = item.get("path", "")
        local_edits = item.get("local_edits", []) or []
        if not rel_path or not local_edits:
            continue

        base_name = os.path.basename(rel_path)
        prefix = os.path.splitext(base_name)[0]
        group_dir = os.path.basename(os.path.dirname(rel_path))
        gid_int = None
        try:
            gid_int = int(group_dir)
        except Exception:
            try:
                digits = "".join([ch for ch in group_dir if ch.isdigit()])
                gid_int = int(digits) if digits else None
            except Exception:
                gid_int = None

        group_str = group_dir  # e.g., "0139" from the JSON path segment

        # Resolve source/target directories strictly as base/<0139>
        src_dir_candidates = [os.path.join(args.source_frames_dir, group_str)]
        tgt_dir_candidates = [os.path.join(args.target_frames_dir, group_str)]
        src_dir = next((d for d in src_dir_candidates if d and os.path.isdir(d)), None)
        tgt_dir = next((d for d in tgt_dir_candidates if d and os.path.isdir(d)), None)
        if src_dir is None or tgt_dir is None:
            continue

        src_path = os.path.join(src_dir, f"{prefix}.png")
        for idx, prompt in enumerate(local_edits, start=1):
            tgt_path = os.path.join(tgt_dir, f"{prefix}_{idx}.png")
            mask_path = os.path.join(args.masks_dir, group_str, f"{prefix}_{idx}.png")
            if not (os.path.exists(src_path) and os.path.exists(tgt_path) and os.path.exists(mask_path)):
                continue
            samples.append({
                "source_image": src_path,
                "target_image": tgt_path,
                "mask_image": mask_path,
                "prompt": prompt,
            })

    size = args.cond_size

    to_tensor_and_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    cond_train_transforms = transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    tokenizer_clip = tokenizers[0]
    tokenizer_t5 = tokenizers[1]

    def tokenize_prompt_single(caption: str):
        text_inputs_clip = tokenizer_clip(
            [caption],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_1 = text_inputs_clip.input_ids[0]

        text_inputs_t5 = tokenizer_t5(
            [caption],
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_t5.input_ids[0]
        return text_input_ids_1, text_input_ids_2

    class LocalEditsDataset(torch.utils.data.Dataset):
        def __init__(self, samples_ls):
            self.samples = samples_ls
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            sample = self.samples[idx]
            s_p = sample["source_image"]
            t_p = sample["target_image"]
            m_p = sample["mask_image"]
            cap = sample["prompt"]

            rr = random.randint(10, 20)
            ri = random.randint(3, 5)
            import cv2
            mask_loaded = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE)
            if mask_loaded is None:
                raise ValueError("mask load failed")
            mask = mask_loaded.copy()

            # Pre-expand mask by a fixed number of pixels before any random expansion
            # Uses a cross-shaped kernel when tapered_corners is True to emulate "tapered" growth
            pre_expand_px = int(getattr(args, "pre_expand_mask_px", 50))
            pre_expand_tapered = bool(getattr(args, "pre_expand_tapered_corners", True))
            if pre_expand_px != 0:
                c = 0 if pre_expand_tapered else 1
                pre_kernel = np.array([[c, 1, c],
                                       [1, 1, 1],
                                       [c, 1, c]], dtype=np.uint8)
                if pre_expand_px > 0:
                    mask = cv2.dilate(mask, pre_kernel, iterations=pre_expand_px)
                else:
                    mask = cv2.erode(mask, pre_kernel, iterations=abs(pre_expand_px))
            if rr > 0 and ri > 0:
                ksize = max(1, 2 * int(rr) + 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                for _ in range(max(1, ri)):
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            src_aligned, tgt_aligned = align_images(s_p, t_p)

            best_w, best_h = choose_kontext_resolution_from_wh(tgt_aligned.width, tgt_aligned.height)
            final_img_rs = tgt_aligned.resize((best_w, best_h), resample=Image.BILINEAR)
            raw_img_rs = src_aligned.resize((best_w, best_h), resample=Image.BILINEAR)

            target_tensor = to_tensor_and_norm(final_img_rs)
            source_tensor = to_tensor_and_norm(raw_img_rs)

            mask_img = Image.fromarray(mask.astype(np.uint8)).convert("L")
            if mask_img.size != src_aligned.size:
                mask_img = mask_img.resize(src_aligned.size, Image.NEAREST)
            mask_np = np.array(mask_img)

            mask_bin = (mask_np > 127).astype(np.uint8)
            inv_mask = (1 - mask_bin).astype(np.uint8)
            src_np = np.array(src_aligned)
            masked_raw_np = src_np * inv_mask[..., None]
            masked_raw_img = Image.fromarray(masked_raw_np.astype(np.uint8))
            cond_tensor = cond_train_transforms(masked_raw_img)

            # Prepare mask_values tensor at training resolution (best_w, best_h)
            mask_img_rs = mask_img.resize((best_w, best_h), Image.NEAREST)
            mask_np_rs = np.array(mask_img_rs)
            mask_bin_rs = (mask_np_rs > 127).astype(np.float32)
            mask_tensor = torch.from_numpy(mask_bin_rs).unsqueeze(0)  # [1, H, W]

            ids1, ids2 = tokenize_prompt_single(cap if isinstance(cap, str) else "")

            # Optionally blend target and source using a blurred mask, controlled by args
            if getattr(args, "blend_pixel_values", BLEND_PIXEL_VALUES):
                blend_kernel = int(getattr(args, "blend_kernel", 21))
                if blend_kernel % 2 == 0:
                    blend_kernel += 1
                blend_sigma = float(getattr(args, "blend_sigma", 10.0))
                gb = transforms.GaussianBlur(kernel_size=(blend_kernel, blend_kernel), sigma=(blend_sigma, blend_sigma))
                # mask_tensor: [1, H, W] in [0,1]
                blurred_mask = gb(mask_tensor)  # [1, H, W]
                # Expand to 3 channels to match image tensors
                blurred_mask_3c = blurred_mask.expand(target_tensor.shape[0], -1, -1)  # [3, H, W]
                # Blend in normalized space (both tensors already normalized to [-1, 1])
                target_tensor = (source_tensor * (1.0 - blurred_mask_3c)) + (target_tensor * blurred_mask_3c)
                target_tensor = target_tensor.clamp(-1.0, 1.0)

            return {
                "source_pixel_values": source_tensor,
                "pixel_values": target_tensor,
                "cond_pixel_values": cond_tensor,
                "token_ids_clip": ids1,
                "token_ids_t5": ids2,
                "mask_values": mask_tensor,
            }

    return LocalEditsDataset(samples)


class BalancedMixDataset(torch.utils.data.Dataset):
    """
    A wrapper dataset that mixes two datasets with a configurable ratio.

    ratio_b_per_a defines how many samples from dataset_b for each sample from dataset_a:
      - 0   => only dataset_a (local edits)
      - 1   => 1:1 mix (default)
      - 2   => 1:2 mix (A:B)
      - any float supported (e.g., 0.5 => 2:1 mix)
    """
    def __init__(self, dataset_a, dataset_b, ratio_b_per_a: float = 1.0):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.ratio_b_per_a = max(0.0, float(ratio_b_per_a))

        len_a = len(dataset_a)
        len_b = len(dataset_b)

        # If ratio is 0, use all of dataset_a only
        if self.ratio_b_per_a == 0 or len_b == 0:
            a_indices = list(range(len_a))
            random.shuffle(a_indices)
            self.mapping = [(0, i) for i in a_indices]
            return

        # Determine how many we can draw without replacement
        # n_a limited by A size and B availability according to ratio
        n_a_by_ratio = int(len_b / self.ratio_b_per_a)
        n_a = min(len_a, max(1, n_a_by_ratio))
        n_b = min(len_b, max(1, int(round(n_a * self.ratio_b_per_a))))

        a_indices = list(range(len_a))
        b_indices = list(range(len_b))
        random.shuffle(a_indices)
        random.shuffle(b_indices)
        a_indices = a_indices[: n_a]
        b_indices = b_indices[: n_b]

        mixed = [(0, i) for i in a_indices] + [(1, i) for i in b_indices]
        random.shuffle(mixed)
        self.mapping = mixed

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        which, real_idx = self.mapping[idx]
        if which == 0:
            return self.dataset_a[real_idx]
        else:
            return self.dataset_b[real_idx]


def make_train_dataset_mixed(args, tokenizers, accelerator=None):
    """
    Create a mixed dataset from:
      - Local edits dataset (this file)
      - Inpaint-mask JSONL dataset (jsonl_datasets_kontext.make_train_dataset_inpaint_mask)

    Ratio control via args.mix_ratio (float):
      - 0   => only local edits dataset
      - 1   => 1:1 mix (local:inpaint)
      - 2   => 1:2 mix, etc.

    Requirements:
      - args.local_edits_json and related dirs must be set for local edits
      - args.train_data_dir must be set for the JSONL inpaint dataset
    """
    ds_local = make_train_dataset_local_edits(args, tokenizers, accelerator)
    ds_inpaint = make_train_dataset_inpaint_mask(args, tokenizers, accelerator)
    mix_ratio = getattr(args, "mix_ratio", 1.0)
    return BalancedMixDataset(ds_local, ds_inpaint, ratio_b_per_a=mix_ratio)