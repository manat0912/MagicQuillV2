from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import torch
import os
from datasets import load_dataset
import numpy as np
import json

Image.MAX_IMAGE_PIXELS = None


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


def _resolve_jsonl(path_str: str):
    if path_str is None or str(path_str).strip() == "":
        raise ValueError("train_data_jsonl is empty. Please set --train_data_jsonl to a JSON/JSONL file or a folder.")
    if os.path.isdir(path_str):
        files = [
            os.path.join(path_str, f)
            for f in os.listdir(path_str)
            if f.lower().endswith((".jsonl", ".json"))
        ]
        if not files:
            raise ValueError(f"No .json or .jsonl files found under directory: {path_str}")
        return {"train": sorted(files)}
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"train_data_jsonl not found: {path_str}")
    return {"train": [path_str]}


def _tokenize(tokenizers, caption: str):
    tokenizer_clip = tokenizers[0]
    tokenizer_t5 = tokenizers[1]
    text_inputs_clip = tokenizer_clip(
        [caption], padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    text_inputs_t5 = tokenizer_t5(
        [caption], padding="max_length", max_length=128, truncation=True, return_tensors="pt"
    )
    return text_inputs_clip.input_ids[0], text_inputs_t5.input_ids[0]


def _prepend_caption(caption: str) -> str:
    """Prepend instruction and keep only instruction with 20% prob."""
    instruction = "Fill in the white region naturally and adapt the foreground into the background. Fix the perspective of the foreground object if necessary."
    if random.random() < 0.2:
        return instruction
    caption = caption or ""
    if caption.strip():
        return f"{instruction} {caption.strip()}"
    return instruction


def _color_augment(pil_img: Image.Image) -> Image.Image:
    brightness = random.uniform(0.8, 1.2)
    contrast = random.uniform(0.8, 1.2)
    saturation = random.uniform(0.8, 1.2)
    hue = random.uniform(-0.05, 0.05)
    img = TF.adjust_brightness(pil_img, brightness)
    img = TF.adjust_contrast(img, contrast)
    img = TF.adjust_saturation(img, saturation)
    img = TF.adjust_hue(img, hue)
    return img


def _dilate_mask(mask_bin: np.ndarray, min_px: int = 5, max_px: int = 100) -> np.ndarray:
    """Grow binary mask by a random radius in [min_px, max_px]. Expects values {0,1}."""
    import cv2
    radius = int(max(min_px, min(max_px, random.randint(min_px, max_px))))
    if radius <= 0:
        return mask_bin.astype(np.uint8)
    ksize = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    grown = cv2.dilate(mask_bin.astype(np.uint8), kernel, iterations=1)
    return (grown > 0).astype(np.uint8)


def _random_point_inside_mask(mask_bin: np.ndarray) -> tuple:
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        h, w = mask_bin.shape
        return w // 2, h // 2
    idx = random.randrange(len(xs))
    return int(xs[idx]), int(ys[idx])


def _bbox_containing_mask(mask_bin: np.ndarray, img_w: int, img_h: int) -> tuple:
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return 0, 0, img_w - 1, img_h - 1
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    # Random padding
    max_pad = int(0.25 * min(img_w, img_h))
    pad_x1 = random.randint(0, max_pad)
    pad_x2 = random.randint(0, max_pad)
    pad_y1 = random.randint(0, max_pad)
    pad_y2 = random.randint(0, max_pad)
    x1 = max(0, x1 - pad_x1)
    y1 = max(0, y1 - pad_y1)
    x2 = min(img_w - 1, x2 + pad_x2)
    y2 = min(img_h - 1, y2 + pad_y2)
    return x1, y1, x2, y2


def _constrained_random_mask(mask_bin: np.ndarray, image_h: int, image_w: int, aug_prob: float = 0.7) -> np.ndarray:
    """Generate random mask whose box contains or starts in m_p, and brush strokes start inside m_p.
    Returns binary 0/1 array of shape (H,W).
    """
    import cv2
    if random.random() >= aug_prob:
        return np.zeros((image_h, image_w), dtype=np.uint8)

    # Scale similar to reference
    ref_size = 1024
    scale_factor = max(1.0, min(image_h, image_w) / float(ref_size))

    out = np.zeros((image_h, image_w), dtype=np.uint8)

    # Choose exactly one augmentation: bbox OR stroke
    if random.random() < 0.2:
        # BBox augmentation: draw N boxes (randomized), first box often contains mask
        num_boxes = random.randint(1, 6)
        for b in range(num_boxes):
            if b == 0 and random.random() < 0.5:
                x1, y1, x2, y2 = _bbox_containing_mask(mask_bin, image_w, image_h)
            else:
                sx, sy = _random_point_inside_mask(mask_bin)
                max_w = int(500 * scale_factor)
                min_w = int(100 * scale_factor)
                bw = random.randint(max(1, min_w), max(2, max_w))
                bh = random.randint(max(1, min_w), max(2, max_w))
                x1 = max(0, sx - random.randint(0, bw))
                y1 = max(0, sy - random.randint(0, bh))
                x2 = min(image_w - 1, x1 + bw)
                y2 = min(image_h - 1, y1 + bh)
            out[y1:y2 + 1, x1:x2 + 1] = 1
    else:
        # Stroke augmentation: draw N strokes starting inside mask
        num_strokes = random.randint(1, 6)
        for _ in range(num_strokes):
            num_points = random.randint(10, 30)
            stroke_width = random.randint(max(1, int(100 * scale_factor)), max(2, int(400 * scale_factor)))
            max_offset = max(1, int(100 * scale_factor))
            start_x, start_y = _random_point_inside_mask(mask_bin)
            px, py = start_x, start_y
            for _ in range(num_points):
                dx = random.randint(-max_offset, max_offset)
                dy = random.randint(-max_offset, max_offset)
                nx = int(np.clip(px + dx, 0, image_w - 1))
                ny = int(np.clip(py + dy, 0, image_h - 1))
                cv2.line(out, (px, py), (nx, ny), 1, stroke_width)
                px, py = nx, ny

    return (out > 0).astype(np.uint8)


def make_placement_dataset_subjects(args, tokenizers, accelerator=None, base_dir=None):
    """
    Dataset for JSONL with fields:
      - generated_image_path: relative to base_dir (target image with object)
      - mask_path: relative to base_dir (mask of object)
      - generated_width, generated_height: image dimensions
      - final_prompt: caption
      - relight_images: list of {mode, path} for relighted versions

    source image construction:
      - background is target_image with a hole punched by grown mask
      - foreground is randomly selected from relight_images with weights
      - includes perspective transformation (moved from interactive dataset)
    
    Args:
        base_dir: Base directory for resolving relative paths. If None, uses args.placement_base_dir.
    """
    if base_dir is None:
        base_dir = getattr(args, "placement_base_dir")
    
    data_files = _resolve_jsonl(getattr(args, "placement_data_jsonl", None))
    file_paths = data_files.get("train", [])
    records = []
    for p in file_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    try:
                        obj = json.loads(line.rstrip(","))
                    except Exception:
                        continue
                # Keep only fields we need
                pruned = {
                    "generated_image_path": obj.get("generated_image_path"),
                    "mask_path": obj.get("mask_path"),
                    "generated_width": obj.get("generated_width"),
                    "generated_height": obj.get("generated_height"),
                    "final_prompt": obj.get("final_prompt"),
                    "relight_images": obj.get("relight_images"),
                }
                records.append(pruned)

    size = int(getattr(args, "cond_size", 512))

    to_tensor_and_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    class PlacementDataset(torch.utils.data.Dataset):
        def __init__(self, hf_ds, base_dir):
            self.ds = hf_ds
            self.base_dir = base_dir
        def __len__(self):
            # Triplicate sampling per record
            return len(self.ds)
        def __getitem__(self, idx):
            rec = self.ds[idx % len(self.ds)]

            t_rel = rec.get("generated_image_path", "")
            m_rel = rec.get("mask_path", "")
            
            # Both are relative paths
            t_p = os.path.join(self.base_dir, t_rel)
            m_p = os.path.join(self.base_dir, m_rel)

            import cv2
            mask_loaded = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE)
            if mask_loaded is None:
                raise ValueError(f"Failed to read mask: {m_p}")

            tgt_img = Image.open(t_p).convert("RGB")

            fw = int(rec.get("generated_width", tgt_img.width))
            fh = int(rec.get("generated_height", tgt_img.height))
            tgt_img = tgt_img.resize((fw, fh), resample=Image.BILINEAR)
            mask_img = Image.fromarray(mask_loaded.astype(np.uint8)).convert("L").resize((fw, fh), Image.NEAREST)

            target_tensor = to_tensor_and_norm(tgt_img)

            # Binary mask at final_size
            mask_np = np.array(mask_img)
            mask_bin = (mask_np > 127).astype(np.uint8)

            # 1) Grow mask by random 50-100 pixels
            grown_mask = _dilate_mask(mask_bin, 50, 200)

            # 2) Optional random augmentation mask constrained by mask
            rand_mask = _constrained_random_mask(mask_bin, fh, fw, 7)

            # 3) Final union mask
            union_mask = np.clip(grown_mask | rand_mask, 0, 1).astype(np.uint8)
            tgt_np = np.array(tgt_img)

            # Helper: choose relighted image from relight_images with weights
            def _choose_relight_image(rec, width, height):
                relight_list = rec.get("relight_images") or []
                # Build map mode -> path
                mode_to_path = {}
                for it in relight_list:
                    try:
                        mode = str(it.get("mode", "")).strip().lower()
                        path = it.get("path")
                    except Exception:
                        continue
                    if not mode or not path:
                        continue
                    mode_to_path[mode] = path

                weighted_order = [
                    ("grayscale", 0.5),
                    ("low", 0.3),
                    ("high", 0.2),
                ]

                # Filter to available
                available = [(m, w) for (m, w) in weighted_order if m in mode_to_path]
                chosen_path = None
                if available:
                    rnd = random.random()
                    cum = 0.0
                    total_w = sum(w for _, w in available)
                    for m, w in available:
                        cum += w / total_w
                        if rnd <= cum:
                            chosen_path = mode_to_path.get(m)
                            break
                    if chosen_path is None:
                        chosen_path = mode_to_path.get(available[-1][0])
                else:
                    # Fallback to any provided path
                    if mode_to_path:
                        chosen_path = next(iter(mode_to_path.values()))

                # Open chosen image
                if chosen_path is not None:
                    try:
                        # Paths are relative to base_dir
                        open_path = os.path.join(self.base_dir, chosen_path)
                        img = Image.open(open_path).convert("RGB").resize((width, height), resample=Image.BILINEAR)
                        return img
                    except Exception:
                        pass

                # Fallback: return target image
                return Image.open(t_p).convert("RGB").resize((width, height), resample=Image.BILINEAR)

            # Choose base image with probabilities:
            # 20%: original target, 20%: color augment(target), 60%: relight augment
            rsel = random.random()
            if rsel < 0.2:
                base_img = tgt_img
            elif rsel < 0.4:
                base_img = _color_augment(tgt_img)
            else:
                base_img = _choose_relight_image(rec, fw, fh)
            base_np = np.array(base_img)
            fore_np = base_np.copy()

            # Random perspective augmentation (50%): apply to foreground ROI (mask bbox) and its mask only
            perspective_applied = False
            roi_update = None
            paste_mask_bool = mask_bin.astype(bool)
            if random.random() < 0.5:
                try:
                    import cv2
                    ys, xs = np.where(mask_bin > 0)
                    if len(xs) > 0 and len(ys) > 0:
                        x1, x2 = int(xs.min()), int(xs.max())
                        y1, y2 = int(ys.min()), int(ys.max())
                        if x2 > x1 and y2 > y1:
                            roi = base_np[y1:y2 + 1, x1:x2 + 1]
                            roi_mask = mask_bin[y1:y2 + 1, x1:x2 + 1]
                            bh, bw = roi.shape[:2]
                            # Random perturbation relative to ROI size
                            max_ratio = random.uniform(0.1, 0.3)
                            dx = bw * max_ratio
                            dy = bh * max_ratio
                            src = np.float32([[0, 0], [bw - 1, 0], [bw - 1, bh - 1], [0, bh - 1]])
                            dst = np.float32([
                                [np.clip(random.uniform(-dx, dx), 0, bw - 1), np.clip(random.uniform(-dy, dy), 0, bh - 1)],
                                [np.clip(bw - 1 + random.uniform(-dx, dx), 0, bw - 1), np.clip(random.uniform(-dy, dy), 0, bh - 1)],
                                [np.clip(bw - 1 + random.uniform(-dx, dx), 0, bw - 1), np.clip(bh - 1 + random.uniform(-dy, dy), 0, bh - 1)],
                                [np.clip(random.uniform(-dx, dx), 0, bw - 1), np.clip(bh - 1 + random.uniform(-dy, dy), 0, bh - 1)],
                            ])
                            M = cv2.getPerspectiveTransform(src, dst)
                            warped_roi = cv2.warpPerspective(roi, M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
                            warped_mask_roi = cv2.warpPerspective((roi_mask.astype(np.uint8) * 255), M, (bw, bh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127
                            # Build a fresh foreground canvas
                            fore_np = np.zeros_like(base_np)
                            h_warp, w_warp = warped_mask_roi.shape
                            y2c = y1 + h_warp
                            x2c = x1 + w_warp
                            fore_np[y1:y2c, x1:x2c][warped_mask_roi] = warped_roi[warped_mask_roi]
                            paste_mask_bool = np.zeros_like(mask_bin, dtype=bool)
                            paste_mask_bool[y1:y2c, x1:x2c] = warped_mask_roi
                            roi_update = (x1, y1, h_warp, w_warp, warped_mask_roi)
                            perspective_applied = True
                except Exception:
                    perspective_applied = False
                    paste_mask_bool = mask_bin.astype(bool)
                    fore_np = base_np

            # Optional: simulate resolution artifacts
            if random.random() < 0.7:
                ys, xs = np.where(paste_mask_bool)
                if len(xs) > 0 and len(ys) > 0:
                    x1, x2 = int(xs.min()), int(xs.max())
                    y1, y2 = int(ys.min()), int(ys.max())
                    if x2 > x1 and y2 > y1:
                        crop = fore_np[y1:y2 + 1, x1:x2 + 1]
                        ch, cw = crop.shape[:2]
                        scale = random.uniform(0.15, 0.9)
                        dw = max(1, int(cw * scale))
                        dh = max(1, int(ch * scale))
                        try:
                            small = Image.fromarray(crop.astype(np.uint8)).resize((dw, dh), Image.BICUBIC)
                            back = small.resize((cw, ch), Image.BICUBIC)
                            crop_blurred = np.array(back).astype(np.uint8)
                            fore_np[y1:y2 + 1, x1:x2 + 1] = crop_blurred
                        except Exception:
                            pass

            # Build masked target and compose
            union_mask_for_target = union_mask.copy()
            if roi_update is not None:
                rx, ry, rh, rw, warped_mask_roi = roi_update
                um_roi = union_mask_for_target[ry:ry + rh, rx:rx + rw]
                union_mask_for_target[ry:ry + rh, rx:rx + rw] = np.clip(um_roi | warped_mask_roi.astype(np.uint8), 0, 1)
            masked_t_np = tgt_np.copy()
            masked_t_np[union_mask_for_target.astype(bool)] = 255
            composed_np = masked_t_np.copy()
            m_fore = paste_mask_bool
            composed_np[m_fore] = fore_np[m_fore]

            # Build tensors
            source_tensor = to_tensor_and_norm(Image.fromarray(composed_np.astype(np.uint8)))
            mask_tensor = torch.from_numpy(union_mask.astype(np.float32)).unsqueeze(0)

            # Caption: prepend instruction
            cap_orig = rec.get("final_prompt", "") or ""
            # Handle list format in final_prompt
            if isinstance(cap_orig, list) and len(cap_orig) > 0:
                cap_orig = cap_orig[0] if isinstance(cap_orig[0], str) else str(cap_orig[0])
            cap = _prepend_caption(cap_orig)
            if perspective_applied:
                cap = f"{cap} Fix the perspective if necessary."
            ids1, ids2 = _tokenize(tokenizers, cap)

            return {
                "source_pixel_values": source_tensor,
                "pixel_values": target_tensor,
                "token_ids_clip": ids1,
                "token_ids_t5": ids2,
                "mask_values": mask_tensor,
            }

    return PlacementDataset(records, base_dir)


def make_interactive_dataset_subjects(args, tokenizers, accelerator=None, base_dir=None):
    """
    Dataset for JSONL with fields:
      - input_path: relative to base_dir (target image)
      - output_path: absolute path to image with foreground
      - mask_after_completion: absolute path to mask
      - img_width, img_height: resize dimensions
      - prompt: caption

    source image construction:
      - background is target_image with a hole punched by grown `mask_after_completion`
      - foreground is from `output_path` image, pasted using original `mask_after_completion`
      - 50% chance to color augment the foreground source
      - NO perspective transform (moved to placement dataset)
    
    Args:
        base_dir: Base directory for resolving relative paths. If None, uses args.interactive_base_dir.
    """
    if base_dir is None:
        base_dir = getattr(args, "interactive_base_dir")
    
    data_files = _resolve_jsonl(getattr(args, "train_data_jsonl", None))
    file_paths = data_files.get("train", [])
    records = []
    for p in file_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    # Best-effort: strip any trailing commas and retry
                    try:
                        obj = json.loads(line.rstrip(","))
                    except Exception:
                        continue
                # Keep only fields we actually need to avoid schema issues
                pruned = {
                    "input_path": obj.get("input_path"),
                    "output_path": obj.get("output_path"),
                    "mask_after_completion": obj.get("mask_after_completion"),
                    "img_width": obj.get("img_width"),
                    "img_height": obj.get("img_height"),
                    "prompt": obj.get("prompt"),
                    # New optional fields
                    "generated_images": obj.get("generated_images"),
                    "positive_prompt_used": obj.get("positive_prompt_used"),
                    "negative_caption_used": obj.get("negative_caption_used"),
                }
                records.append(pruned)

    size = int(getattr(args, "cond_size", 512))

    to_tensor_and_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    class SubjectsDataset(torch.utils.data.Dataset):
        def __init__(self, hf_ds, base_dir):
            self.ds = hf_ds
            self.base_dir = base_dir
        def __len__(self):
            # Triplicate sampling per record
            return len(self.ds)
        def __getitem__(self, idx):
            rec = self.ds[idx % len(self.ds)]

            t_rel = rec.get("input_path", "")
            foreground_p = rec.get("output_path", "")
            m_abs = rec.get("mask_after_completion", "")

            if not os.path.isabs(m_abs):
                raise ValueError("mask_after_completion must be absolute")
            if not os.path.isabs(foreground_p):
                raise ValueError("output_path must be absolute")

            t_p = os.path.join(self.base_dir, t_rel)
            m_p = m_abs

            import cv2
            mask_loaded = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE)
            if mask_loaded is None:
                raise ValueError(f"Failed to read mask: {m_p}")

            tgt_img = Image.open(t_p).convert("RGB")
            foreground_source_img = Image.open(foreground_p).convert("RGB")

            fw = int(rec.get("img_width", tgt_img.width))
            fh = int(rec.get("img_height", tgt_img.height))
            tgt_img = tgt_img.resize((fw, fh), resample=Image.BILINEAR)
            foreground_source_img = foreground_source_img.resize((fw, fh), resample=Image.BILINEAR)
            mask_img = Image.fromarray(mask_loaded.astype(np.uint8)).convert("L").resize((fw, fh), Image.NEAREST)

            # Ensure PIL images to tensors for outputs based on new logic later
            target_tensor = to_tensor_and_norm(tgt_img)

            # Binary mask at final_size
            mask_np = np.array(mask_img)
            mask_bin = (mask_np > 127).astype(np.uint8)

            # 1) Grow m_p by random 50-100 pixels
            grown_mask = _dilate_mask(mask_bin, 50, 200)

            # 2) Optional random augmentation mask constrained by m_p
            rand_mask = _constrained_random_mask(mask_bin, fh, fw, aug_prob=0.7)

            # 3) Final union mask
            union_mask = np.clip(grown_mask | rand_mask, 0, 1).astype(np.uint8)
            tgt_np = np.array(tgt_img)

            # Helper: choose relighted image from generated_images with weights
            def _choose_relight_image(rec, default_img, width, height):
                gen_list = rec.get("generated_images") or []
                # Build map mode -> path
                mode_to_path = {}
                for it in gen_list:
                    try:
                        mode = str(it.get("mode", "")).strip().lower()
                        path = it.get("path")
                    except Exception:
                        continue
                    if not mode or not path:
                        continue
                    mode_to_path[mode] = path

                # Weighted selection among available modes
                weighted_order = [
                    ("grayscale", 0.5),
                    ("low", 0.3),
                    ("high", 0.2),
                ]

                # Filter to available
                available = [(m, w) for (m, w) in weighted_order if m in mode_to_path]
                chosen_path = None
                if available:
                    rnd = random.random()
                    cum = 0.0
                    total_w = sum(w for _, w in available)
                    for m, w in available:
                        cum += w / total_w
                        if rnd <= cum:
                            chosen_path = mode_to_path.get(m)
                            break
                    if chosen_path is None:
                        chosen_path = mode_to_path.get(available[-1][0])
                else:
                    # Fallback to any provided path
                    if mode_to_path:
                        chosen_path = next(iter(mode_to_path.values()))

                # Open chosen image
                if chosen_path is not None:
                    try:
                        open_path = chosen_path
                        # generated paths are typically absolute; if not, use as-is
                        img = Image.open(open_path).convert("RGB").resize((width, height), resample=Image.BILINEAR)
                        return img
                    except Exception:
                        pass

                return default_img

            # 5) Choose base image with probabilities:
            # 20%: original, 20%: color augment(original), 60%: relight augment
            rsel = random.random()
            if rsel < 0.2:
                base_img = foreground_source_img
            elif rsel < 0.4:
                base_img = _color_augment(foreground_source_img)
            else:
                base_img = _choose_relight_image(rec, foreground_source_img, fw, fh)
            base_np = np.array(base_img)

            # 5.1) Random perspective augmentation (20%): apply to foreground ROI (mask bbox) and its mask only
            perspective_applied = False
            roi_update = None
            paste_mask_bool = mask_bin.astype(bool)
            if random.random() < 0.5:
                try:
                    import cv2
                    ys, xs = np.where(mask_bin > 0)
                    if len(xs) > 0 and len(ys) > 0:
                        x1, x2 = int(xs.min()), int(xs.max())
                        y1, y2 = int(ys.min()), int(ys.max())
                        if x2 > x1 and y2 > y1:
                            roi = base_np[y1:y2 + 1, x1:x2 + 1]
                            roi_mask = mask_bin[y1:y2 + 1, x1:x2 + 1]
                            bh, bw = roi.shape[:2]
                            # Random perturbation relative to ROI size
                            max_ratio = random.uniform(0.1, 0.3)
                            dx = bw * max_ratio
                            dy = bh * max_ratio
                            src = np.float32([[0, 0], [bw - 1, 0], [bw - 1, bh - 1], [0, bh - 1]])
                            dst = np.float32([
                                [np.clip(random.uniform(-dx, dx), 0, bw - 1), np.clip(random.uniform(-dy, dy), 0, bh - 1)],
                                [np.clip(bw - 1 + random.uniform(-dx, dx), 0, bw - 1), np.clip(random.uniform(-dy, dy), 0, bh - 1)],
                                [np.clip(bw - 1 + random.uniform(-dx, dx), 0, bw - 1), np.clip(bh - 1 + random.uniform(-dy, dy), 0, bh - 1)],
                                [np.clip(random.uniform(-dx, dx), 0, bw - 1), np.clip(bh - 1 + random.uniform(-dy, dy), 0, bh - 1)],
                            ])
                            M = cv2.getPerspectiveTransform(src, dst)
                            warped_roi = cv2.warpPerspective(roi, M, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
                            warped_mask_roi = cv2.warpPerspective((roi_mask.astype(np.uint8) * 255), M, (bw, bh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127
                            # Build a fresh foreground canvas
                            fore_np = np.zeros_like(base_np)
                            h_warp, w_warp = warped_mask_roi.shape
                            y2c = y1 + h_warp
                            x2c = x1 + w_warp
                            fore_np[y1:y2c, x1:x2c][warped_mask_roi] = warped_roi[warped_mask_roi]
                            paste_mask_bool = np.zeros_like(mask_bin, dtype=bool)
                            paste_mask_bool[y1:y2c, x1:x2c] = warped_mask_roi
                            roi_update = (x1, y1, h_warp, w_warp, warped_mask_roi)
                            perspective_applied = True
                            base_np = fore_np
                except Exception:
                    perspective_applied = False
                    paste_mask_bool = mask_bin.astype(bool)

            # Optional: simulate cut-out foregrounds coming from different resolutions by
            # downscaling the masked foreground region and upscaling back to original size.
            # This introduces realistic blur/aliasing seen in real inpaint workflows.
            if random.random() < 0.7:
                ys, xs = np.where(mask_bin > 0)
                if len(xs) > 0 and len(ys) > 0:
                    x1, x2 = int(xs.min()), int(xs.max())
                    y1, y2 = int(ys.min()), int(ys.max())
                    # Ensure valid box
                    if x2 > x1 and y2 > y1:
                        crop = base_np[y1:y2 + 1, x1:x2 + 1]
                        ch, cw = crop.shape[:2]
                        scale = random.uniform(0.2, 0.9)
                        dw = max(1, int(cw * scale))
                        dh = max(1, int(ch * scale))
                        try:
                            small = Image.fromarray(crop.astype(np.uint8)).resize((dw, dh), Image.BICUBIC)
                            back = small.resize((cw, ch), Image.BICUBIC)
                            crop_blurred = np.array(back).astype(np.uint8)
                            base_np[y1:y2 + 1, x1:x2 + 1] = crop_blurred
                        except Exception:
                            # Fallback: skip if resize fails
                            pass

            # 6) Build masked target using (possibly) updated union mask; then paste
            union_mask_for_target = union_mask.copy()
            if roi_update is not None:
                rx, ry, rh, rw, warped_mask_roi = roi_update
                # Ensure union covers the warped foreground area inside ROI using warped shape
                um_roi = union_mask_for_target[ry:ry + rh, rx:rx + rw]
                union_mask_for_target[ry:ry + rh, rx:rx + rw] = np.clip(um_roi | warped_mask_roi.astype(np.uint8), 0, 1)
            masked_t_np = tgt_np.copy()
            masked_t_np[union_mask_for_target.astype(bool)] = 255
            composed_np = masked_t_np.copy()
            m_fore = paste_mask_bool
            composed_np[m_fore] = base_np[m_fore]

            # 7) Build tensors
            source_tensor = to_tensor_and_norm(Image.fromarray(composed_np.astype(np.uint8)))
            mask_tensor = torch.from_numpy(union_mask.astype(np.float32)).unsqueeze(0)

            # 8) Caption: prepend instruction, 20% keep only instruction
            cap_orig = rec.get("prompt", "") or ""
            cap = _prepend_caption(cap_orig)
            if perspective_applied:
                cap = f"{cap} Fix the perspective if necessary."
            ids1, ids2 = _tokenize(tokenizers, cap)

            return {
                "source_pixel_values": source_tensor,
                "pixel_values": target_tensor,
                "token_ids_clip": ids1,
                "token_ids_t5": ids2,
                "mask_values": mask_tensor,
            }

    return SubjectsDataset(records, base_dir)


def make_pexels_dataset_subjects(args, tokenizers, accelerator=None, base_dir=None):
    """
    Dataset for JSONL with fields:
      - input_path: relative to base_dir (target image)
      - output_path: relative to relight_base_dir (relighted image)
      - final_size: {width, height} resize applied
      - caption: text caption
    
    Modified to use segmentation maps instead of raw_mask_path.
    Randomly selects 2-5 segments from segmentation map, applies augmentation to each, and takes union.
    This simulates multiple foreground objects being placed like a puzzle.
    
    Each segment independently uses: 20% original, 20% color_augment, 60% relighted image.
    
    Args:
        base_dir: Base directory for resolving relative paths. If None, uses args.pexels_base_dir.
    """
    if base_dir is None:
        base_dir = getattr(args, "pexels_base_dir", "/mnt/robby-b1/common/datasets")
    
    relight_base_dir = getattr(args, "pexels_relight_base_dir", "/robby/share/Editing/lzc/data/relight_outputs")
    seg_base_dir = getattr(args, "seg_base_dir", "/mnt/robby-b1/common/datasets/pexels-mask/20190515093182")
    
    data_files = _resolve_jsonl(getattr(args, "pexels_data_jsonl", None))
    file_paths = data_files.get("train", [])
    records = []
    for p in file_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    try:
                        obj = json.loads(line.rstrip(","))
                    except Exception:
                        continue
                pruned = {
                    "input_path": obj.get("input_path"),
                    "output_path": obj.get("output_path"),
                    "final_size": obj.get("final_size"),
                    "caption": obj.get("caption"),
                }
                records.append(pruned)

    to_tensor_and_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    class PexelsDataset(torch.utils.data.Dataset):
        def __init__(self, hf_ds, base_dir, relight_base_dir, seg_base_dir):
            self.ds = hf_ds
            self.base_dir = base_dir
            self.relight_base_dir = relight_base_dir
            self.seg_base_dir = seg_base_dir
        
        def __len__(self):
            return len(self.ds)
        
        def _extract_hash_from_filename(self, filename: str) -> str:
            """Extract hash from input filename for segmentation map lookup."""
            stem = os.path.splitext(os.path.basename(filename))[0]
            if '_' in stem:
                parts = stem.split('_')
                return parts[-1]
            return stem
        
        def _build_segmap_path(self, input_filename: str) -> str:
            """Build path to segmentation map from input filename."""
            hash_id = self._extract_hash_from_filename(input_filename)
            return os.path.join(self.seg_base_dir, f"{hash_id}_map.png")
        
        def _load_segmap_uint32(self, seg_path: str):
            """Load segmentation map as uint32 array."""
            import cv2
            try:
                with Image.open(seg_path) as im:
                    if im.mode == 'P':
                        seg = np.array(im)
                    elif im.mode in ('I;16', 'I', 'L'):
                        seg = np.array(im)
                    else:
                        seg = np.array(im.convert('L'))
            except Exception:
                return None

            if seg.ndim == 3:
                seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
            return seg.astype(np.uint32)
        
        def _extract_multiple_segments(
            self,
            image_h: int,
            image_w: int,
            seg_path: str,
            min_area_ratio: float = 0.02,
            max_area_ratio: float = 0.4,
        ):
            """Extract 2-5 individual segment masks from segmentation map."""
            import cv2
            seg = self._load_segmap_uint32(seg_path)
            if seg is None:
                return []

            if seg.shape != (image_h, image_w):
                seg = cv2.resize(seg.astype(np.uint16), (image_w, image_h), interpolation=cv2.INTER_NEAREST).astype(np.uint32)

            labels, counts = np.unique(seg, return_counts=True)
            if labels.size == 0:
                return []

            # Exclude background label 0
            bg_mask = labels == 0
            labels = labels[~bg_mask]
            counts = counts[~bg_mask]
            if labels.size == 0:
                return []

            area = image_h * image_w
            min_px = int(round(min_area_ratio * area))
            max_px = int(round(max_area_ratio * area))
            keep = (counts >= min_px) & (counts <= max_px)
            cand_labels = labels[keep]
            if cand_labels.size == 0:
                return []

            # Select 2-5 labels randomly
            max_sel = min(5, cand_labels.size)
            min_sel = min(2, cand_labels.size)
            num_to_select = random.randint(min_sel, max_sel)
            chosen = np.random.choice(cand_labels, size=num_to_select, replace=False)

            # Create individual masks for each chosen label
            individual_masks = []
            for lab in chosen:
                binm = (seg == int(lab)).astype(np.uint8)
                # Apply opening operation to clean up mask
                k = max(3, int(round(max(image_h, image_w) * 0.01)))
                if k % 2 == 0:
                    k += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                eroded = cv2.erode(binm, kernel, iterations=1)
                opened = cv2.dilate(eroded, kernel, iterations=1)
                individual_masks.append(opened)
            
            return individual_masks
        
        def __getitem__(self, idx):
            rec = self.ds[idx % len(self.ds)]

            t_rel = rec.get("input_path", "")
            r_rel = rec.get("output_path", "")
            
            t_p = os.path.join(self.base_dir, t_rel)
            relight_p = os.path.join(self.relight_base_dir, r_rel)

            import cv2
            tgt_img = Image.open(t_p).convert("RGB")
            
            # Load relighted image, fallback to target if not available
            try:
                relighted_img = Image.open(relight_p).convert("RGB")
            except Exception:
                relighted_img = tgt_img.copy()

            final_size = rec.get("final_size", {}) or {}
            fw = int(final_size.get("width", tgt_img.width))
            fh = int(final_size.get("height", tgt_img.height))
            tgt_img = tgt_img.resize((fw, fh), resample=Image.BILINEAR)
            relighted_img = relighted_img.resize((fw, fh), resample=Image.BILINEAR)

            target_tensor = to_tensor_and_norm(tgt_img)

            # Build segmentation map path and extract multiple segments
            input_filename = os.path.basename(t_rel)
            seg_path = self._build_segmap_path(input_filename)
            individual_masks = self._extract_multiple_segments(fh, fw, seg_path)
            
            if not individual_masks:
                # Fallback: create empty mask (will be handled gracefully)
                union_mask = np.zeros((fh, fw), dtype=np.uint8)
                individual_masks = []
            else:
                # Apply augmentation to each segment mask and take union
                augmented_masks = []
                for seg_mask in individual_masks:
                    # 1) Grow mask by random 50-200 pixels
                    grown = _dilate_mask(seg_mask, 50, 200)
                    # 2) Optional random augmentation mask constrained by this segment
                    rand_mask = _constrained_random_mask(seg_mask, fh, fw, aug_prob=0.7)
                    # 3) Union for this segment
                    seg_union = np.clip(grown | rand_mask, 0, 1).astype(np.uint8)
                    augmented_masks.append(seg_union)
                
                # Take union of all augmented masks
                union_mask = np.zeros((fh, fw), dtype=np.uint8)
                for m in augmented_masks:
                    union_mask = np.clip(union_mask | m, 0, 1).astype(np.uint8)
            
            tgt_np = np.array(tgt_img)

            # Build masked target first
            masked_t_np = tgt_np.copy()
            masked_t_np[union_mask.astype(bool)] = 255
            composed_np = masked_t_np.copy()

            # Process each segment independently with different augmentations
            # This simulates multiple foreground objects from different sources
            for seg_mask in individual_masks:
                # 1) Choose source for this segment: 20% original, 20% color_augment, 60% relighted
                r = random.random()
                if r < 0.2:
                    # Original image
                    seg_source_img = tgt_img
                else:
                    seg_source_img = _color_augment(tgt_img)
                # elif r < 0.4:
                #     # Color augmentation
                #     seg_source_img = _color_augment(tgt_img)
                # else:
                #     # Relighted image
                #     seg_source_img = relighted_img
                
                seg_source_np = np.array(seg_source_img)

                # 2) Apply resolution augmentation to this segment's region
                if random.random() < 0.7:
                    ys, xs = np.where(seg_mask > 0)
                    if len(xs) > 0 and len(ys) > 0:
                        x1, x2 = int(xs.min()), int(xs.max())
                        y1, y2 = int(ys.min()), int(ys.max())
                        if x2 > x1 and y2 > y1:
                            crop = seg_source_np[y1:y2 + 1, x1:x2 + 1]
                            ch, cw = crop.shape[:2]
                            scale = random.uniform(0.2, 0.9)
                            dw = max(1, int(cw * scale))
                            dh = max(1, int(ch * scale))
                            try:
                                small = Image.fromarray(crop.astype(np.uint8)).resize((dw, dh), Image.BICUBIC)
                                back = small.resize((cw, ch), Image.BICUBIC)
                                crop_blurred = np.array(back).astype(np.uint8)
                                seg_source_np[y1:y2 + 1, x1:x2 + 1] = crop_blurred
                            except Exception:
                                pass

                # 3) Paste this segment onto composed image
                m_fore = seg_mask.astype(bool)
                composed_np[m_fore] = seg_source_np[m_fore]

            # Build tensors
            source_tensor = to_tensor_and_norm(Image.fromarray(composed_np.astype(np.uint8)))
            mask_tensor = torch.from_numpy(union_mask.astype(np.float32)).unsqueeze(0)

            # Caption: prepend instruction
            cap_orig = rec.get("caption", "") or ""
            cap = _prepend_caption(cap_orig)
            ids1, ids2 = _tokenize(tokenizers, cap)

            return {
                "source_pixel_values": source_tensor,
                "pixel_values": target_tensor,
                "token_ids_clip": ids1,
                "token_ids_t5": ids2,
                "mask_values": mask_tensor,
            }

    return PexelsDataset(records, base_dir, relight_base_dir, seg_base_dir)


def make_mixed_dataset(args, tokenizers, interactive_jsonl_path=None, placement_jsonl_path=None, 
                       pexels_jsonl_path=None, interactive_base_dir=None, placement_base_dir=None,
                       pexels_base_dir=None, interactive_weight=1.0, placement_weight=1.0, 
                       pexels_weight=1.0, accelerator=None):
    """
    Create a mixed dataset combining interactive, placement, and pexels datasets.
    
    Args:
        args: Arguments object with dataset configuration
        tokenizers: Tuple of tokenizers for text encoding
        interactive_jsonl_path: Path to interactive dataset JSONL (optional)
        placement_jsonl_path: Path to placement dataset JSONL (optional)
        pexels_jsonl_path: Path to pexels dataset JSONL (optional)
        interactive_base_dir: Base directory for interactive dataset paths (optional)
        placement_base_dir: Base directory for placement dataset paths (optional)
        pexels_base_dir: Base directory for pexels dataset paths (optional)
        interactive_weight: Sampling weight for interactive dataset (default: 1.0)
        placement_weight: Sampling weight for placement dataset (default: 1.0)
        pexels_weight: Sampling weight for pexels dataset (default: 1.0)
        accelerator: Optional accelerator object
    
    Returns:
        Mixed dataset that samples from all provided datasets with specified weights
    """
    datasets = []
    dataset_names = []
    dataset_weights = []
    
    # Create interactive dataset if path provided
    if interactive_jsonl_path:
        interactive_args = type('Args', (), {})()
        for k, v in vars(args).items():
            setattr(interactive_args, k, v)
        interactive_args.train_data_jsonl = interactive_jsonl_path
        if interactive_base_dir:
            interactive_args.interactive_base_dir = interactive_base_dir
        interactive_ds = make_interactive_dataset_subjects(interactive_args, tokenizers, accelerator)
        datasets.append(interactive_ds)
        dataset_names.append("interactive")
        dataset_weights.append(interactive_weight)
    
    # Create placement dataset if path provided
    if placement_jsonl_path:
        placement_args = type('Args', (), {})()
        for k, v in vars(args).items():
            setattr(placement_args, k, v)
        placement_args.placement_data_jsonl = placement_jsonl_path
        if placement_base_dir:
            placement_args.placement_base_dir = placement_base_dir
        placement_ds = make_placement_dataset_subjects(placement_args, tokenizers, accelerator)
        datasets.append(placement_ds)
        dataset_names.append("placement")
        dataset_weights.append(placement_weight)
    
    # Create pexels dataset if path provided
    if pexels_jsonl_path:
        pexels_args = type('Args', (), {})()
        for k, v in vars(args).items():
            setattr(pexels_args, k, v)
        pexels_args.pexels_data_jsonl = pexels_jsonl_path
        if pexels_base_dir:
            pexels_args.pexels_base_dir = pexels_base_dir
        pexels_ds = make_pexels_dataset_subjects(pexels_args, tokenizers, accelerator)
        datasets.append(pexels_ds)
        dataset_names.append("pexels")
        dataset_weights.append(pexels_weight)
    
    if not datasets:
        raise ValueError("At least one dataset path must be provided")
    
    if len(datasets) == 1:
        return datasets[0]
    
    # Mixed dataset class with balanced sampling (based on smallest dataset)
    class MixedDataset(torch.utils.data.Dataset):
        def __init__(self, datasets, dataset_names, dataset_weights):
            self.datasets = datasets
            self.dataset_names = dataset_names
            self.lengths = [len(ds) for ds in datasets]
            
            # Normalize weights
            total_weight = sum(dataset_weights)
            self.weights = [w / total_weight for w in dataset_weights]
            
            # Calculate samples per dataset based on smallest dataset and weights
            # Find the minimum weighted size
            min_weighted_size = min(length / weight for length, weight in zip(self.lengths, dataset_weights))
            
            # Each dataset contributes samples proportional to its weight, scaled by min_weighted_size
            self.samples_per_dataset = [int(min_weighted_size * w) for w in dataset_weights]
            self.total_length = sum(self.samples_per_dataset)
            
            # Build cumulative sample counts for indexing
            self.cumsum_samples = [0]
            for count in self.samples_per_dataset:
                self.cumsum_samples.append(self.cumsum_samples[-1] + count)
            
            print(f"Balanced mixed dataset created:")
            for i, name in enumerate(dataset_names):
                print(f"  {name}: {self.lengths[i]} total, {self.samples_per_dataset[i]} per epoch")
            print(f"  Total samples per epoch: {self.total_length}")
        
        def __len__(self):
            return self.total_length
        
        def __getitem__(self, idx):
            # Determine which dataset this idx belongs to
            dataset_idx = 0
            for i in range(len(self.cumsum_samples) - 1):
                if self.cumsum_samples[i] <= idx < self.cumsum_samples[i + 1]:
                    dataset_idx = i
                    break
            
            # Randomly sample from the selected dataset (enables different samples each epoch)
            local_idx = random.randint(0, self.lengths[dataset_idx] - 1)
            sample = self.datasets[dataset_idx][local_idx]
            # Add dataset source information
            sample["dataset_source"] = self.dataset_names[dataset_idx]
            return sample
    
    return MixedDataset(datasets, dataset_names, dataset_weights)


def _run_test_mode(
    interactive_jsonl: str = None,
    placement_jsonl: str = None,
    pexels_jsonl: str = None,
    interactive_base_dir: str = None,
    placement_base_dir: str = None,
    pexels_base_dir: str = None,
    pexels_relight_base_dir: str = None,
    seg_base_dir: str = None,
    interactive_weight: float = 1.0,
    placement_weight: float = 1.0,
    pexels_weight: float = 1.0,
    output_dir: str = "test_output",
    num_samples: int = 100
):
    """Test dataset by saving samples with source labels.
    
    Args:
        interactive_jsonl: Path to interactive dataset JSONL (optional)
        placement_jsonl: Path to placement dataset JSONL (optional)
        pexels_jsonl: Path to pexels dataset JSONL (optional)
        interactive_base_dir: Base directory for interactive dataset
        placement_base_dir: Base directory for placement dataset
        pexels_base_dir: Base directory for pexels dataset
        pexels_relight_base_dir: Base directory for pexels relighted images
        seg_base_dir: Directory containing segmentation maps for pexels dataset
        interactive_weight: Sampling weight for interactive dataset (default: 1.0)
        placement_weight: Sampling weight for placement dataset (default: 1.0)
        pexels_weight: Sampling weight for pexels dataset (default: 1.0)
        output_dir: Output directory for test images
        num_samples: Number of samples to save
    """
    if not interactive_jsonl and not placement_jsonl and not pexels_jsonl:
        raise ValueError("At least one dataset path must be provided")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dummy tokenizers for testing
    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            class Result:
                input_ids = torch.zeros(1, 77, dtype=torch.long)
            return Result()
    
    tokenizers = (DummyTokenizer(), DummyTokenizer())
    
    # Create args object
    class Args:
        cond_size = 512
    
    args = Args()
    args.train_data_jsonl = interactive_jsonl
    args.placement_data_jsonl = placement_jsonl
    args.pexels_data_jsonl = pexels_jsonl
    args.interactive_base_dir = interactive_base_dir
    args.placement_base_dir = placement_base_dir
    args.pexels_base_dir = pexels_base_dir
    args.pexels_relight_base_dir = pexels_relight_base_dir if pexels_relight_base_dir else "/robby/share/Editing/lzc/data/relight_outputs"
    args.seg_base_dir = seg_base_dir if seg_base_dir else "/mnt/robby-b1/common/datasets/pexels-mask/20190515093182"
    
    # Create dataset (single or mixed)
    try:
        # Count how many datasets are provided
        num_datasets = sum([bool(interactive_jsonl), bool(placement_jsonl), bool(pexels_jsonl)])
        
        if num_datasets > 1:
            dataset = make_mixed_dataset(
                args, tokenizers,
                interactive_jsonl_path=interactive_jsonl,
                placement_jsonl_path=placement_jsonl,
                pexels_jsonl_path=pexels_jsonl,
                interactive_base_dir=args.interactive_base_dir,
                placement_base_dir=args.placement_base_dir,
                pexels_base_dir=args.pexels_base_dir,
                interactive_weight=interactive_weight,
                placement_weight=placement_weight,
                pexels_weight=pexels_weight
            )
            print(f"Created mixed dataset with {len(dataset)} samples")
            weights_str = []
            if interactive_jsonl:
                weights_str.append(f"Interactive: {interactive_weight:.2f}")
            if placement_jsonl:
                weights_str.append(f"Placement: {placement_weight:.2f}")
            if pexels_jsonl:
                weights_str.append(f"Pexels: {pexels_weight:.2f}")
            print(f"Sampling weights - {', '.join(weights_str)}")
        elif pexels_jsonl:
            dataset = make_pexels_dataset_subjects(args, tokenizers, base_dir=pexels_base_dir)
            print(f"Created pexels dataset with {len(dataset)} samples")
        elif placement_jsonl:
            dataset = make_placement_dataset_subjects(args, tokenizers, base_dir=args.placement_base_dir)
            print(f"Created placement dataset with {len(dataset)} samples")
        else:
            dataset = make_interactive_dataset_subjects(args, tokenizers, base_dir=args.interactive_base_dir)
            print(f"Created interactive dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Sample and save
    saved = 0
    counts = {}
    
    for attempt in range(min(num_samples * 3, len(dataset))):
        try:
            idx = random.randint(0, len(dataset) - 1)
            sample = dataset[idx]
            
            source_name = sample.get("dataset_source", "single")
            counts[source_name] = counts.get(source_name, 0) + 1
            
            # Denormalize tensors from [-1, 1] to [0, 255]
            source_np = ((sample["source_pixel_values"].permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            target_np = ((sample["pixel_values"].permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

            # Save images
            idx_str = f"{saved:05d}"
            Image.fromarray(source_np).save(os.path.join(output_dir, f"{idx_str}_{source_name}_source.jpg"))
            Image.fromarray(target_np).save(os.path.join(output_dir, f"{idx_str}_{source_name}_target.jpg"))
            
            saved += 1
            if saved % 10 == 0:
                print(f"Saved {saved}/{num_samples} samples - {counts}")
            if saved >= num_samples:
                break
        
        except Exception as e:
            print(f"Failed to process sample: {e}")
            continue
    
    print(f"\nTest complete. Saved {saved} samples to {output_dir}")
    print(f"Distribution: {counts}")


def _parse_test_args():
    import argparse
    parser = argparse.ArgumentParser(description="Test visualization for Kontext datasets")
    parser.add_argument("--interactive_jsonl", type=str, default="/robby/share/Editing/lzc/HOI_v1/final_metadata.jsonl", 
                        help="Path to interactive dataset JSONL")
    parser.add_argument("--placement_jsonl", type=str, default="/robby/share/Editing/lzc/subject_placement/metadata_relight.jsonl", 
                        help="Path to placement dataset JSONL")
    parser.add_argument("--pexels_jsonl", type=str, default=None, 
                        help="Path to pexels dataset JSONL")
    parser.add_argument("--interactive_base_dir", type=str, default="/robby/share/Editing/lzc/HOI_v1", 
                        help="Base directory for interactive dataset")
    parser.add_argument("--placement_base_dir", type=str, default=None, 
                        help="Base directory for placement dataset")
    parser.add_argument("--pexels_base_dir", type=str, default=None, 
                        help="Base directory for pexels dataset")
    parser.add_argument("--pexels_relight_base_dir", type=str, default="/robby/share/Editing/lzc/data/relight_outputs", 
                        help="Base directory for pexels relighted images")
    parser.add_argument("--seg_base_dir", type=str, default=None, 
                        help="Directory containing segmentation maps for pexels dataset")
    parser.add_argument("--interactive_weight", type=float, default=1.0, 
                        help="Sampling weight for interactive dataset (default: 1.0)")
    parser.add_argument("--placement_weight", type=float, default=1.0, 
                        help="Sampling weight for placement dataset (default: 1.0)")
    parser.add_argument("--pexels_weight", type=float, default=0, 
                        help="Sampling weight for pexels dataset (default: 1.0)")
    parser.add_argument("--output_dir", type=str, default="visualize_output", 
                        help="Output directory to save pairs")
    parser.add_argument("--num_samples", type=int, default=100, 
                        help="Number of pairs to save")
    
    # Legacy arguments
    parser.add_argument("--test_jsonl", type=str, default=None, 
                        help="Legacy: Path to JSONL (uses as interactive_jsonl)")
    parser.add_argument("--base_dir", type=str, default=None, 
                        help="Legacy: Base directory (uses as interactive_base_dir)")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = _parse_test_args()
        
        # Handle legacy args
        interactive_jsonl = args.interactive_jsonl or args.test_jsonl
        interactive_base_dir = args.interactive_base_dir or args.base_dir
        
        _run_test_mode(
            interactive_jsonl=interactive_jsonl,
            placement_jsonl=args.placement_jsonl,
            pexels_jsonl=args.pexels_jsonl,
            interactive_base_dir=interactive_base_dir,
            placement_base_dir=args.placement_base_dir,
            pexels_base_dir=args.pexels_base_dir,
            pexels_relight_base_dir=args.pexels_relight_base_dir,
            seg_base_dir=args.seg_base_dir,
            interactive_weight=args.interactive_weight,
            placement_weight=args.placement_weight,
            pexels_weight=args.pexels_weight,
            output_dir=args.output_dir,
            num_samples=args.num_samples
        )
    except SystemExit:
        # Allow import usage without triggering test mode
        pass

