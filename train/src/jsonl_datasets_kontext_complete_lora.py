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

def _prepend_caption(description: str, obj_name: str) -> str:
    """Build instruction with stochastic OBJECT choice and keep only instruction with 20% prob.

    OBJECT choice (equal probability):
      - literal string "object"
      - JSON field `object` with '_' replaced by space
      - JSON field `description`
    """
    # Prepare options for OBJECT slot
    cleaned_obj = (obj_name or "object").replace("_", " ").strip() or "object"
    desc_opt = (description or "object").strip() or "object"
    object_slot = random.choice(["object", cleaned_obj, desc_opt])

    instruction = f"Complete the {object_slot}'s missing parts if necessary. White Background;"

    return instruction

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

def _apply_white_brushstrokes(image_np: np.ndarray, mask_bin: np.ndarray = None) -> np.ndarray:
    """Draw random white brushstrokes on the RGB image array and return modified array.
    Strokes preferentially start within mask_bin if provided.
    """
    import cv2
    h, w = image_np.shape[:2]
    rng = random.Random()

    # Determine stroke counts and sizes based on image size
    ref = max(1, min(h, w))
    num_strokes = rng.randint(1, 5)
    max_offset = max(5, ref // 40)
    min_th = max(2, ref // 40)
    max_th = max(min_th + 1, ref // 5)

    out = image_np.copy()
    prefer_mask_p = 0.33 if mask_bin is not None and mask_bin.any() else 0.0

    def rand_point_inside_mask():
        ys, xs = np.where(mask_bin > 0)
        if len(xs) == 0:
            return rng.randrange(w), rng.randrange(h)
        i = rng.randrange(len(xs))
        return int(xs[i]), int(ys[i])

    def rand_point_any():
        return rng.randrange(w), rng.randrange(h)

    for _ in range(num_strokes):
        if rng.random() < prefer_mask_p:
            px, py = rand_point_inside_mask()
        else:
            px, py = rand_point_any()
        px, py = rand_point_any()

        # Polyline with several jittered segments
        segments = rng.randint(40, 80)
        thickness = rng.randint(min_th, max_th)
        for _ in range(segments):
            dx = rng.randint(-max_offset, max_offset)
            dy = rng.randint(-max_offset, max_offset)
            nx = int(np.clip(px + dx, 0, w - 1))
            ny = int(np.clip(py + dy, 0, h - 1))
            cv2.line(out, (px, py), (nx, ny), (255, 255, 255), thickness)
            px, py = nx, ny

    return out


def make_train_dataset_subjects(args, tokenizers, accelerator=None):
    """
    Dataset for JSONL with fields (one JSON object per line):
      - white_image_path: absolute path to base image used for both pixel_values and source_pixel_values
      - mask_path: absolute path to mask image (grayscale)
      - img_width: target width
      - img_height: target height
      - description: caption text

    Behavior:
      - pixel_values = white_image_path resized to (img_width, img_height)
      - source_pixel_values = same image but with random white brushstrokes overlaid
      - mask_values = binarized mask from mask_path resized with nearest neighbor
      - captions tokenized from description
    """
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
                # Keep only fields we need for this dataset schema
                pruned = {
                    "white_image_path": obj.get("white_image_path"),
                    "mask_path": obj.get("mask_path"),
                    "img_width": obj.get("img_width"),
                    "img_height": obj.get("img_height"),
                    "description": obj.get("description"),
                    "object": obj.get("object"),
                }
                records.append(pruned)

    size = int(getattr(args, "cond_size", 512))

    to_tensor_and_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Repeat each record with independent random brushstrokes
    REPEATS_PER_IMAGE = 5

    class SubjectsDataset(torch.utils.data.Dataset):
        def __init__(self, hf_ds):
            self.ds = hf_ds
            self.repeats = REPEATS_PER_IMAGE
        def __len__(self):
            if self.repeats and self.repeats > 1:
                return len(self.ds) * self.repeats
            return len(self.ds)
        def __getitem__(self, idx):
            if self.repeats and self.repeats > 1:
                base_idx = idx % len(self.ds)
            else:
                base_idx = idx
            rec = self.ds[base_idx]

            white_p = rec.get("white_image_path", "") or ""
            mask_p = rec.get("mask_path", "") or ""

            if not os.path.isabs(white_p):
                # Allow absolute path only to avoid ambiguity
                raise ValueError("white_image_path must be absolute")
            if not os.path.isabs(mask_p):
                raise ValueError("mask_path must be absolute")

            import cv2
            mask_loaded = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
            if mask_loaded is None:
                raise ValueError(f"Failed to read mask: {mask_p}")

            base_img = Image.open(white_p).convert("RGB")

            # Desired output size
            fw = int(rec.get("img_width") or base_img.width)
            fh = int(rec.get("img_height") or base_img.height)
            base_img = base_img.resize((fw, fh), resample=Image.BILINEAR)
            mask_img = Image.fromarray(mask_loaded.astype(np.uint8)).convert("L").resize((fw, fh), Image.NEAREST)

            # Tensors: target is the clean white image
            target_tensor = to_tensor_and_norm(base_img)

            # Binary mask at final_size
            mask_np = np.array(mask_img)
            mask_bin = (mask_np > 127).astype(np.uint8)

            # Build source by drawing random white brushstrokes on top of the white image
            base_np = np.array(base_img).astype(np.uint8)
            stroked_np = _apply_white_brushstrokes(base_np, mask_bin)

            # Build tensors
            source_tensor = to_tensor_and_norm(Image.fromarray(stroked_np.astype(np.uint8)))
            mask_tensor = torch.from_numpy(mask_bin.astype(np.float32)).unsqueeze(0)

            # Caption: build instruction using description and object
            description = rec.get("description", "")
            obj_name = rec.get("object", "")
            cap = _prepend_caption(description, obj_name)
            ids1, ids2 = _tokenize(tokenizers, cap)

            return {
                "source_pixel_values": source_tensor,
                "pixel_values": target_tensor,
                "token_ids_clip": ids1,
                "token_ids_t5": ids2,
                "mask_values": mask_tensor,
            }

    return SubjectsDataset(records)




def _run_test_mode(test_jsonl: str, output_dir: str, num_samples: int = 50):
    """Utility to visualize augmentation: saves pairs of (target, source) images.
    Reads the JSONL directly, applies the same logic as dataset to produce
    pixel_values (target) and source_pixel_values (with white strokes),
    then writes them to output_dir for manual inspection.
    """
    os.makedirs(output_dir, exist_ok=True)
    to_tensor_and_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Minimal tokenizers shim to reuse dataset tokenization pipeline
    class _NoOpTokenizer:
        def __call__(self, texts, padding=None, max_length=None, truncation=None, return_tensors=None):
            return type("T", (), {"input_ids": torch.zeros((1, 1), dtype=torch.long)})()

    tokenizers = [_NoOpTokenizer(), _NoOpTokenizer()]

    saved = 0
    line_idx = 0
    import cv2
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for raw in f:
            if saved >= num_samples:
                break
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                try:
                    obj = json.loads(raw.rstrip(","))
                except Exception:
                    continue

            rec = {
                "white_image_path": obj.get("white_image_path"),
                "mask_path": obj.get("mask_path"),
                "img_width": obj.get("img_width"),
                "img_height": obj.get("img_height"),
                "description": obj.get("description"),
            }

            white_p = rec.get("white_image_path", "") or ""
            mask_p = rec.get("mask_path", "") or ""
            if not white_p or not mask_p:
                continue
            if not (os.path.isabs(white_p) and os.path.isabs(mask_p)):
                continue

            mask_loaded = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
            if mask_loaded is None:
                continue

            try:
                base_img = Image.open(white_p).convert("RGB")
            except Exception:
                continue

            fw = int(rec.get("img_width") or base_img.width)
            fh = int(rec.get("img_height") or base_img.height)
            base_img = base_img.resize((fw, fh), resample=Image.BILINEAR)
            mask_img = Image.fromarray(mask_loaded.astype(np.uint8)).convert("L").resize((fw, fh), Image.NEAREST)

            mask_np = np.array(mask_img)
            mask_bin = (mask_np > 127).astype(np.uint8)

            base_np = np.array(base_img).astype(np.uint8)
            stroked_np = _apply_white_brushstrokes(base_np, mask_bin)

            # Save images
            idx_str = f"{line_idx:05d}"
            try:
                Image.fromarray(base_np).save(os.path.join(output_dir, f"{idx_str}_target.jpg"))
                Image.fromarray(stroked_np).save(os.path.join(output_dir, f"{idx_str}_source.jpg"))
                Image.fromarray((mask_bin * 255).astype(np.uint8)).save(os.path.join(output_dir, f"{idx_str}_mask.png"))
                saved += 1
            except Exception:
                pass
            line_idx += 1


def _parse_test_args():
    import argparse
    parser = argparse.ArgumentParser(description="Test visualization for Kontext complete dataset")
    parser.add_argument("--test_jsonl", type=str, default="/robby/share/Editing/lzc/subject_completion/white_bg_picked/results_picked_filtered.jsonl", help="Path to JSONL to preview")
    parser.add_argument("--output_dir", type=str, default="/robby/share/Editing/lzc/subject_completion/train_test", help="Output directory to save pairs")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of pairs to save")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = _parse_test_args()
        _run_test_mode(args.test_jsonl, args.output_dir, args.num_samples)
    except SystemExit:
        # Allow import usage without triggering test mode
        pass