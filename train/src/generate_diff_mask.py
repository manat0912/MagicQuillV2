#!/usr/bin/env python3
"""
Standalone script: Given two images, generate a final difference mask using the
same pipeline as visualize_mask_diff (without any visualization output).

Pipeline:
1) Align images to a preferred resolution/crop so they share the same size.
2) Pixel-diff screening across parameter combinations; skip if any hull ratio is
   outside [hull_min_allowed, hull_max_allowed].
3) Color-diff to produce the final mask; remove small areas and re-check hull
   ratio. Save final mask to output path.
"""

import os
import json
import argparse
from typing import Tuple, Optional

import numpy as np
from PIL import Image
import cv2


PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328),
    (832, 1248), (880, 1184), (944, 1104), (1024, 1024), (1104, 944),
    (1184, 880), (1248, 832), (1328, 800), (1392, 752), (1456, 720),
    (1504, 688), (1568, 672),
]


def choose_preferred_resolution(image_width: int, image_height: int) -> Tuple[int, int]:
    aspect_ratio = image_width / max(1, image_height)
    best = min(((abs(aspect_ratio - (w / h)), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS), key=lambda x: x[0])
    _, w_best, h_best = best
    return int(w_best), int(h_best)


def align_images(source_path: str, target_path: str) -> Tuple[Image.Image, Image.Image]:
    source_img = Image.open(source_path).convert("RGB")
    target_img = Image.open(target_path).convert("RGB")

    pref_w, pref_h = choose_preferred_resolution(source_img.width, source_img.height)
    source_resized = source_img.resize((pref_w, pref_h), Image.Resampling.LANCZOS)

    tgt_w, tgt_h = target_img.width, target_img.height
    crop_w = min(source_resized.width, tgt_w)
    crop_h = min(source_resized.height, tgt_h)

    source_aligned = source_resized.crop((0, 0, crop_w, crop_h))
    target_aligned = target_img.crop((0, 0, crop_w, crop_h))
    return source_aligned, target_aligned


def pil_to_cv_gray(img: Image.Image) -> np.ndarray:
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray


def generate_pixel_diff_mask(img1: Image.Image, img2: Image.Image, threshold: Optional[int] = None, clean_kernel_size: Optional[int] = 11) -> np.ndarray:
    img1_gray = pil_to_cv_gray(img1)
    img2_gray = pil_to_cv_gray(img2)
    diff = cv2.absdiff(img1_gray, img2_gray)
    if threshold is None:
        mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    else:
        mask = cv2.threshold(diff, int(threshold), 255, cv2.THRESH_BINARY)[1]
    if clean_kernel_size and clean_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_kernel_size, clean_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def generate_color_diff_mask(img1: Image.Image, img2: Image.Image, threshold: Optional[int] = None, clean_kernel_size: Optional[int] = 21) -> np.ndarray:
    bgr1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    bgr2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    lab1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2LAB).astype("float32")
    lab2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2LAB).astype("float32")
    diff = lab1 - lab2
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    if threshold is None:
        mask = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    else:
        mask = cv2.threshold(dist_u8, int(threshold), 255, cv2.THRESH_BINARY)[1]
    if clean_kernel_size and clean_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_kernel_size, clean_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def compute_unified_contour(mask_bin: np.ndarray, contours: list, min_area: int = 40, method: str = "morph", morph_kernel: int = 15, morph_iters: int = 1, approx_epsilon_ratio: float = 0.01):
    valid_cnts = []
    for c in contours:
        if cv2.contourArea(c) >= max(1, min_area):
            valid_cnts.append(c)
    if not valid_cnts:
        return None
    if method == "convex_hull":
        all_points = np.vstack(valid_cnts)
        hull = cv2.convexHull(all_points)
        epsilon = approx_epsilon_ratio * cv2.arcLength(hull, True)
        unified = cv2.approxPolyDP(hull, epsilon, True)
        return unified
    union = np.zeros_like(mask_bin)
    cv2.drawContours(union, valid_cnts, -1, 255, thickness=-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    union_closed = union.copy()
    for _ in range(max(1, morph_iters)):
        union_closed = cv2.morphologyEx(union_closed, cv2.MORPH_CLOSE, kernel)
    ext = cv2.findContours(union_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ext = ext[0] if len(ext) == 2 else ext[1]
    if not ext:
        return None
    largest = max(ext, key=cv2.contourArea)
    epsilon = approx_epsilon_ratio * cv2.arcLength(largest, True)
    unified = cv2.approxPolyDP(largest, epsilon, True)
    return unified


def compute_hull_area_ratio(mask: np.ndarray, min_area: int = 40) -> float:
    mask_bin = (mask > 0).astype("uint8") * 255
    cnts = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if not cnts:
        return 0.0
    hull_cnt = compute_unified_contour(mask_bin, cnts, min_area=min_area, method="convex_hull", morph_kernel=15, morph_iters=1)
    if hull_cnt is None or len(hull_cnt) < 3:
        return 0.0
    hull_area = float(cv2.contourArea(hull_cnt))
    img_area = float(mask_bin.shape[0] * mask_bin.shape[1])
    return hull_area / max(1.0, img_area)


def clean_and_fill_mask(mask: np.ndarray, min_area: int = 40) -> np.ndarray:
    mask_bin = (mask > 0).astype("uint8") * 255
    cnts = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cleaned = np.zeros_like(mask_bin)
    for c in cnts:
        if cv2.contourArea(c) >= max(1, min_area):
            cv2.drawContours(cleaned, [c], 0, 255, -1)
    return cleaned


def generate_final_difference_mask(source_path: str,
                                   target_path: str,
                                   hull_min_allowed: float = 0.001,
                                   hull_max_allowed: float = 0.75,
                                   pixel_parameters: Optional[list] = None,
                                   pixel_clean_kernel_default: int = 11,
                                   color_clean_kernel: int = 3,
                                   roll_radius: int = 0,
                                   roll_iters: int = 1) -> Optional[np.ndarray]:
    if pixel_parameters is None:
        # Mirrors the tuned combinations used in visualization script
        pixel_parameters = [(None, 5), (None, 11), (50, 5)]

    src_img, tgt_img = align_images(source_path, target_path)

    # Pixel screening across parameter combinations
    violation = False
    for thr, ksize in pixel_parameters:
        pm = generate_pixel_diff_mask(src_img, tgt_img, threshold=thr, clean_kernel_size=ksize)
        r = compute_hull_area_ratio(pm, min_area=40)
        if r < hull_min_allowed or r > hull_max_allowed:
            violation = True
            break
    if violation:
        # Failure: do not produce any mask
        return None

    # Color-based final mask → cleaned small areas
    color_mask = generate_color_diff_mask(src_img, tgt_img, threshold=None, clean_kernel_size=color_clean_kernel)
    cleaned = clean_and_fill_mask(color_mask, min_area=40)

    # Produce binary mask from the convex hull contour of the cleaned mask
    mask_bin = (cleaned > 0).astype("uint8") * 255
    cnts = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    hull_cnt = compute_unified_contour(mask_bin, cnts, min_area=40, method="convex_hull", morph_kernel=15, morph_iters=1)
    if hull_cnt is None or len(hull_cnt) < 3:
        return None

    h_mask = np.zeros_like(mask_bin)
    cv2.drawContours(h_mask, [hull_cnt], -1, 255, thickness=-1)

    # Rolling-circle smoothing: closing then opening with a disk of radius R
    if roll_radius and roll_radius > 0 and roll_iters and roll_iters > 0:
        ksize = max(1, 2 * int(roll_radius) + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        for _ in range(max(1, roll_iters)):
            h_mask = cv2.morphologyEx(h_mask, cv2.MORPH_CLOSE, kernel)
            h_mask = cv2.morphologyEx(h_mask, cv2.MORPH_OPEN, kernel)

    # Final hull ratio check on the hull-filled binary mask
    r_final = compute_hull_area_ratio(h_mask, min_area=40)
    if r_final > hull_max_allowed or r_final < hull_min_allowed:
        return None

    return h_mask


def main():
    parser = argparse.ArgumentParser(description="Generate final difference mask (single pair or whole dataset)")
    # Single-pair mode (optional): if provided, runs single pair; otherwise runs dataset mode
    parser.add_argument("--source", help="Path to source image")
    parser.add_argument("--target", help="Path to target image")
    parser.add_argument("--output", help="Path to write the final mask (PNG)")
    # Dataset mode (defaults to user's dataset paths)
    parser.add_argument("--dataset_dir", default="/home/lzc/KontextFill/InstructV2V/extracted_dataset", help="Base dataset dir with source_images/ and target_images/")
    parser.add_argument("--dataset_output_dir", default="/home/lzc/KontextFill/visualizations_masks/inference_masks_smoothing", help="Output directory for batch masks")
    parser.add_argument("--json_path", default="/home/lzc/KontextFill/InstructV2V/extracted_dataset/extracted_data.json", help="Dataset JSON mapping with fields 'source_image' and 'target_image'")
    # Common params
    parser.add_argument("--hull_min_allowed", type=float, default=0.001)
    parser.add_argument("--hull_max_allowed", type=float, default=0.75)
    parser.add_argument("--color_clean_kernel", type=int, default=3)
    parser.add_argument("--roll_radius", type=int, default=15, help="Rolling-circle smoothing radius (pixels); 0 disables")
    parser.add_argument("--roll_iters", type=int, default=5, help="Rolling smoothing iterations")

    args = parser.parse_args()

    pixel_parameters = [(None, 5), (None, 11), (50, 5)]

    # Decide mode: single or dataset
    if args.source and args.target and args.output:
        mask = generate_final_difference_mask(
            source_path=args.source,
            target_path=args.target,
            hull_min_allowed=args.hull_min_allowed,
            hull_max_allowed=args.hull_max_allowed,
            pixel_parameters=pixel_parameters,
            color_clean_kernel=args.color_clean_kernel,
            roll_radius=args.roll_radius,
            roll_iters=args.roll_iters,
        )
        if mask is None:
            print("Single-pair inference failed; no output saved.")
            return
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        cv2.imwrite(args.output, mask)
        return

    # Dataset mode using JSON mapping
    out_dir = args.dataset_output_dir
    os.makedirs(out_dir, exist_ok=True)

    processed = 0
    skipped = 0
    failed = 0
    missing_files = 0
    try:
        with open(args.json_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except Exception as e:
        print(f"Failed to read JSON mapping at {args.json_path}: {e}")
        entries = []

    for item in entries:
        try:
            src_rel = item.get("source_image")
            tgt_rel = item.get("target_image")
            edit_id = item.get("id")
            if not src_rel or not tgt_rel:
                skipped += 1
                continue
            s = os.path.join(args.dataset_dir, src_rel)
            t = os.path.join(args.dataset_dir, tgt_rel)
            if not (os.path.exists(s) and os.path.exists(t)):
                missing_files += 1
                continue
            mask = generate_final_difference_mask(
                source_path=s,
                target_path=t,
                hull_min_allowed=args.hull_min_allowed,
                hull_max_allowed=args.hull_max_allowed,
                pixel_parameters=pixel_parameters,
                color_clean_kernel=args.color_clean_kernel,
                roll_radius=args.roll_radius,
                roll_iters=args.roll_iters,
            )
            if mask is None:
                failed += 1
                continue
            name = f"edit_{int(edit_id):04d}" if isinstance(edit_id, int) or (isinstance(edit_id, str) and edit_id.isdigit()) else os.path.splitext(os.path.basename(src_rel))[0]
            out_path = os.path.join(out_dir, f"{name}.png")
            cv2.imwrite(out_path, mask)
            processed += 1
        except Exception as e:
            skipped += 1
            continue
    print(f"Batch done. Processed={processed}, Failed={failed}, Skipped={skipped}, MissingFiles={missing_files}, OutputDir={out_dir}")


if __name__ == "__main__":
    main()


