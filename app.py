import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import numpy as np
import logging
class LoraWarningFilter(logging.Filter):
    def filter(self, record):
        return "No LoRA keys associated to" not in record.getMessage()

logging.getLogger("diffusers.loaders.peft").addFilter(LoraWarningFilter())
logging.getLogger("diffusers.loaders.lora_base").addFilter(LoraWarningFilter())
import gradio as gr
from gradio_magicquillv2 import MagicQuillV2
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
import base64
from PIL import Image, ImageOps
import io
import random
import time
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import json

# Try importing as a package (recommended)
from edit import KontextEditModel, SAM
from util import (
    load_and_preprocess_image,
    read_base64_image as read_base64_image_utils,
    create_alpha_mask,
    tensor_to_base64,
    get_mask_bbox
)

# Initialize models
print("Initializing models...")
kontext_model = KontextEditModel()
sam_model = SAM()
print("Models initialized.")

css = """
.ms {
    width: 60%;
    margin: auto
}
"""

url = "http://localhost:7860"

def generate(merged_image, total_mask, original_image, add_color_image, add_edge_mask, remove_edge_mask, fill_mask, add_prop_image, positive_prompt, negative_prompt, fine_edge, fix_perspective, grow_size, edge_strength, color_strength, local_strength, seed, steps, cfg):
    print("prompt is:", positive_prompt)
    print("other parameters:", negative_prompt, fine_edge, fix_perspective, grow_size, edge_strength, color_strength, local_strength, seed, steps, cfg)
    
    if kontext_model is None:
        raise RuntimeError("KontextEditModel not initialized")

    # Preprocess inputs
    # utils.read_base64_image returns BytesIO, which create_alpha_mask accepts (via Image.open)
    # load_and_preprocess_image accepts path, so we might need to check if it accepts file-like object.
    # utils.load_and_preprocess_image uses Image.open(image_path), so BytesIO works.
    
    merged_image_tensor = load_and_preprocess_image(read_base64_image_utils(merged_image))
    total_mask_tensor = create_alpha_mask(read_base64_image_utils(total_mask))
    original_image_tensor = load_and_preprocess_image(read_base64_image_utils(original_image))
    
    if add_color_image:
        add_color_image_tensor = load_and_preprocess_image(read_base64_image_utils(add_color_image))
    else:
        add_color_image_tensor = original_image_tensor
        
    add_mask = create_alpha_mask(read_base64_image_utils(add_edge_mask)) if add_edge_mask else torch.ones_like(total_mask_tensor)
    remove_mask = create_alpha_mask(read_base64_image_utils(remove_edge_mask)) if remove_edge_mask else torch.ones_like(total_mask_tensor)
    fill_mask_tensor = create_alpha_mask(read_base64_image_utils(fill_mask)) if fill_mask else torch.ones_like(total_mask_tensor)

    # Build add_prop_mask from the prop image alpha channel.
    # create_alpha_mask gives active-low convention: 0.0 = inside prop, 1.0 = outside.
    # When add_prop_image is a solid (non-transparent) image, alpha=255 everywhere,
    # so create_alpha_mask returns all-zeros ("entire image is prop").
    # We use total_mask_tensor as a proxy in that case, which reflects what the
    # frontend actually painted.
    if add_prop_image:
        raw_prop_mask = create_alpha_mask(read_base64_image_utils(add_prop_image))
        # If the mask is entirely zero (solid prop image), fall back to total_mask
        if torch.sum(raw_prop_mask > 0.5).item() == 0:
            add_prop_mask = total_mask_tensor
            print("[INFO] Prop image has no transparent areas — using total_mask as prop_mask")
        else:
            add_prop_mask = raw_prop_mask
    else:
        add_prop_mask = torch.ones_like(total_mask_tensor)

    # Clean the checkered brush pattern in the merged canvas
    # by reconstructing the canvas (combining original background with clean pasted props)
    # to avoid having checkers in the VAE input or erasing props.
    if add_prop_image:
        try:
            prop_pil = Image.open(read_base64_image_utils(add_prop_image)).convert("RGBA")
            prop_alpha_np = np.array(prop_pil.getchannel('A')).astype(np.float32) / 255.0
            prop_alpha = torch.from_numpy(prop_alpha_np).to(device=original_image_tensor.device, dtype=original_image_tensor.dtype)[None, ..., None]
            # Load prop RGB without compositing onto white — preserve the clean pixel values.
            prop_rgb_np = np.array(prop_pil.convert("RGB")).astype(np.float32) / 255.0
            prop_rgb = torch.from_numpy(prop_rgb_np)[None, ...].to(device=original_image_tensor.device, dtype=original_image_tensor.dtype)
            
            merged_image_tensor = prop_rgb * prop_alpha + original_image_tensor * (1.0 - prop_alpha)
            print("[OK] Reconstructed clean merged canvas with props (no checkers)")
        except Exception as e:
            print(f"[WARN] Failed to reconstruct clean merged canvas ({e}), using default merged canvas")

    # Some UI flows record magic-quill strokes only on total_mask (not add_edge_mask).
    extra_active = torch.clamp((total_mask_tensor < 0.5).float() - (add_prop_mask < 0.5).float(), 0.0, 1.0)
    if torch.sum(extra_active > 0.5).item() > 0:
        add_mask = torch.minimum(add_mask, 1.0 - extra_active)

    # ── Detect which brushes the user actually drew ──────────────────────────
    # create_alpha_mask uses active-LOW convention:
    #   0.0 = stroke drawn here (inside the brush mark)
    #   1.0 = empty / background
    # A completely un-stroked mask layer is ALL 1.0 from the frontend.
    # "sum < 0.5" on an all-1.0 tensor = 0 → correctly says "no strokes".
    # "sum < 0.5" on an all-0.0 tensor = total pixels → WRONG (create_alpha_mask
    # returns all-zeros when the prop PNG has no transparent pixels, or when
    # the edge mask is all-white). We must check for values actually below 0.5
    # only when there are meaningful strokes, not when the whole layer is zeroed.
    #
    # Robust check: a mask has real strokes only if it has BOTH pixels < 0.5
    # (stroke) AND pixels >= 0.5 (non-stroke) — i.e. it is not uniform.
    def _has_strokes(mask_tensor: torch.Tensor) -> bool:
        """Return True only if the mask contains real user-drawn strokes.
        An all-zeros OR all-ones mask means the layer is empty."""
        low  = torch.sum(mask_tensor < 0.1).item()   # stroke pixels
        high = torch.sum(mask_tensor > 0.9).item()   # background pixels
        total = mask_tensor.numel()
        # Require at least 0.5% stroke coverage AND the mask is not uniform
        return low > (total * 0.005) and high > 0

    has_prop  = _has_strokes(add_prop_mask)
    has_brush = _has_strokes(add_mask) or _has_strokes(remove_mask)
    has_fill  = _has_strokes(fill_mask_tensor)

    print(
        "[ROUTING] stroke detection:",
        f"has_prop={has_prop}",
        f"has_brush={has_brush}",
        f"has_fill={has_fill}",
        f"add_mask_low={torch.sum(add_mask < 0.1).item()}",
        f"remove_mask_low={torch.sum(remove_mask < 0.1).item()}",
        f"fill_mask_low={torch.sum(fill_mask_tensor < 0.1).item()}",
        f"prop_mask_low={torch.sum(add_prop_mask < 0.1).item()}",
    )

    edit_image_tensor = original_image_tensor

    # Determine flag and modify prompt
    flag = "kontext"
    if has_brush:
        # Magic-quill edge brush: use precise edit with edge/color controls.
        flag = "precise_edit"
        if has_prop:
            edit_image_tensor = merged_image_tensor
            # Keep edge control for matte+brush edits; background-only color layer confuses color LoRA.
            add_color_image_tensor = merged_image_tensor
    elif has_fill:
        # Fill brush: both fill-only and fill+prop go through local_edit().
        # For fill+prop, pass merged_image (clean composite) as the canvas;
        # process() will use total_mask so the transparent zone is included.
        if has_prop:
            flag = "foreground"   # process() routes both to local_edit
            edit_image_tensor = merged_image_tensor
        else:
            flag = "local"
    elif has_prop:
        flag = "foreground"   # process() routes to local_edit with total_mask
        edit_image_tensor = merged_image_tensor
    elif (torch.sum(remove_mask < 0.5).item() > 0 and torch.sum(add_mask < 0.5).item() == 0):
        positive_prompt = "remove the instance"
        flag = "removal"
    elif (torch.sum(add_mask < 0.5).item() > 0 or torch.sum(remove_mask < 0.5).item() > 0 or (not torch.equal(original_image_tensor, add_color_image_tensor))):
        flag = "precise_edit"
    
    print("positive prompt: ", positive_prompt)
    print("current flag: ", flag)
    print(
        "mask stats:",
        f"prop={torch.sum(add_prop_mask < 0.5).item()}",
        f"brush={torch.sum(add_mask < 0.5).item()}",
        f"fill={torch.sum(fill_mask_tensor < 0.5).item()}",
        f"extra={torch.sum(extra_active > 0.5).item()}",
    )

    final_image, condition, mask = kontext_model.process(
        edit_image_tensor,
        add_color_image_tensor,
        merged_image_tensor,
        positive_prompt,
        total_mask_tensor,
        add_mask,
        remove_mask,
        add_prop_mask,
        fill_mask_tensor,
        fine_edge,
        fix_perspective,
        edge_strength,
        color_strength,
        local_strength,
        grow_size,
        seed,
        steps,
        cfg,
        flag,
    )

    # tensor_to_base64 returns pure base64 string
    res_base64 = tensor_to_base64(final_image)
    return res_base64

def generate_image_handler(x, negative_prompt, fine_edge, fix_perspective, grow_size, edge_strength, color_strength, local_strength, seed, steps, cfg):
    merged_image = x['from_frontend']['img']
    total_mask = x['from_frontend']['total_mask']
    original_image = x['from_frontend']['original_image']
    add_color_image = x['from_frontend']['add_color_image']
    add_edge_mask = x['from_frontend']['add_edge_mask']
    remove_edge_mask = x['from_frontend']['remove_edge_mask']
    fill_mask = x['from_frontend']['fill_mask']
    add_prop_image = x['from_frontend']['add_prop_image']
    positive_prompt = x['from_backend']['prompt']

    try:
        res_base64 = generate(
            merged_image,
            total_mask,
            original_image,
            add_color_image,
            add_edge_mask,
            remove_edge_mask,
            fill_mask,
            add_prop_image,
            positive_prompt,
            negative_prompt,
            fine_edge,
            fix_perspective,
            grow_size,
            edge_strength,
            color_strength,
            local_strength,
            seed,
            steps,
            cfg
        )
        x["from_backend"]["generated_image"] = res_base64
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in generation: {e}")
        x["from_backend"]["generated_image"] = None
    
    return x


with gr.Blocks(title="MagicQuill V2") as demo:
    with gr.Row():
        ms = MagicQuillV2()

    with gr.Row():
        with gr.Column():
            btn = gr.Button("Run", variant="primary")
        with gr.Column():
            with gr.Accordion("parameters", open=False):
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="",
                    interactive=True
                )
                fine_edge = gr.Radio(
                    label="Fine Edge",
                    choices=['enable', 'disable'],
                    value='disable',
                    interactive=True
                )
                fix_perspective = gr.Radio(
                    label="Fix Perspective",
                    choices=['enable', 'disable'],
                    value='disable',
                    interactive=True
                )
                grow_size = gr.Slider(
                    label="Grow Size",
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=1,
                    interactive=True
                )
                edge_strength = gr.Slider(
                    label="Edge Strength",
                    minimum=0.0,
                    maximum=5.0,
                    value=0.6,
                    step=0.01,
                    interactive=True
                )
                color_strength = gr.Slider(
                    label="Color Strength",
                    minimum=0.0,
                    maximum=5.0,
                    value=1.5,
                    step=0.01,
                    interactive=True
                )
                local_strength = gr.Slider(
                    label="Local Strength",
                    minimum=0.0,
                    maximum=5.0,
                    value=1.0,
                    step=0.01,
                    interactive=True
                )
                seed = gr.Number(
                    label="Seed",
                    value=-1,
                    precision=0,
                    interactive=True
                )
                steps = gr.Slider(
                    label="Steps",
                    minimum=1,
                    maximum=50,
                    value=18,
                    interactive=True
                )
                cfg = gr.Slider(
                    label="CFG",
                    minimum=0.0,
                    maximum=20.0,
                    value=3.5,
                    step=0.1,
                    interactive=True
                )

        btn.click(generate_image_handler, inputs=[ms, negative_prompt, fine_edge, fix_perspective, grow_size, edge_strength, color_strength, local_strength, seed, steps, cfg], outputs=ms)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_root_url(request: Request, route_path: str, root_path: str | None):
    print(root_path)
    return root_path
gr.route_utils.get_root_url = get_root_url

@app.post("/magic_quill/process_background_img")
async def process_background_img(request: Request):
    img = await request.json()
    from util import process_background
    # process_background returns tensor [1, H, W, 3] in uint8 or float
    resized_img_tensor = process_background(img)
    
    # tensor_to_base64 from util expects tensor
    resized_img_base64 = "data:image/webp;base64," + tensor_to_base64(
        resized_img_tensor, 
        quality=80,
        method=6
    )
    return resized_img_base64

@app.post("/magic_quill/segmentation")
async def segmentation(request: Request):
    json_data = await request.json()
    image_base64 = json_data.get("image", None)
    coordinates_positive = json_data.get("coordinates_positive", None)
    coordinates_negative = json_data.get("coordinates_negative", None)
    bboxes = json_data.get("bboxes", None)

    if sam_model is None:
        return {"error": "sam model not initialized"}

    # Process image
    image_tensor = load_and_preprocess_image(read_base64_image_utils(image_base64))

    # Process coordinates and bboxes
    pos_coordinates = None
    if coordinates_positive and len(coordinates_positive) > 0:
        pos_coordinates = []
        for coord in coordinates_positive:
            coord['x'] = int(round(coord['x']))
            coord['y'] = int(round(coord['y']))
            pos_coordinates.append({'x': coord['x'], 'y': coord['y']})
        pos_coordinates = json.dumps(pos_coordinates)
        
    neg_coordinates = None
    if coordinates_negative and len(coordinates_negative) > 0:
        neg_coordinates = []
        for coord in coordinates_negative:
            coord['x'] = int(round(coord['x']))
            coord['y'] = int(round(coord['y']))
            neg_coordinates.append({'x': coord['x'], 'y': coord['y']})
        neg_coordinates = json.dumps(neg_coordinates)
        
    bboxes_xyxy = None
    if bboxes and len(bboxes) > 0:
        valid_bboxes = []
        for bbox in bboxes:
            if (bbox.get("startX") is None or
                bbox.get("startY") is None or
                bbox.get("endX") is None or
                bbox.get("endY") is None):
                continue
            else:
                x_min = max(min(int(bbox["startX"]), int(bbox["endX"])), 0)
                y_min = max(min(int(bbox["startY"]), int(bbox["endY"])), 0)
                x_max = min(max(int(bbox["startX"]), int(bbox["endX"])), image_tensor.shape[2])
                y_max = min(max(int(bbox["startY"]), int(bbox["endY"])), image_tensor.shape[1])
                valid_bboxes.append((x_min, y_min, x_max, y_max))
        
        bboxes_xyxy = []
        for bbox in valid_bboxes:
            x_min, y_min, x_max, y_max = bbox
            bboxes_xyxy.append((x_min, y_min, x_max, y_max))

    print(f"Segmentation request: pos={pos_coordinates}, neg={neg_coordinates}, bboxes={bboxes_xyxy}")

    # Execute segmentation
    segmentation_image, segmentation_mask = sam_model.process(
        image_tensor, 
        coordinates_positive=pos_coordinates, 
        coordinates_negative=neg_coordinates, 
        bboxes=bboxes_xyxy
    )
    
    # Get bbox of the mask
    mask_bbox = get_mask_bbox(segmentation_mask)
    if mask_bbox:
        x_min, y_min, x_max, y_max = mask_bbox
        seg_bbox = {'startX': x_min, 'startY': y_min, 'endX': x_max, 'endY': y_max}
    else:
        seg_bbox = {'startX': 0, 'startY': 0, 'endX': 0, 'endY': 0}

    print(seg_bbox)
    
    # Convert result to base64
    image_base64_res = tensor_to_base64(segmentation_image)
    
    return {
        "error": False,
        "segmentation_image": "data:image/webp;base64," + image_base64_res, 
        "segmentation_bbox": seg_bbox
    }

app = gr.mount_gradio_app(app, demo, "/")

if __name__ == "__main__":
    import logging
    import sys
    
    class LogFilter(logging.Filter):
        def filter(self, record):
            log_msg = record.getMessage()
            if "FOO /" in log_msg or "ui-sans-serif-Regular.woff2" in log_msg or "system-ui-Regular.woff2" in log_msg:
                return False
            return True

    sys.modules['__main__'].LogFilter = LogFilter

    from uvicorn.config import LOGGING_CONFIG
    LOGGING_CONFIG["filters"] = LOGGING_CONFIG.get("filters", {})
    LOGGING_CONFIG["filters"]["access_filter"] = {
        "()": "__main__.LogFilter"
    }
    if "filters" not in LOGGING_CONFIG["loggers"]["uvicorn.access"]:
        LOGGING_CONFIG["loggers"]["uvicorn.access"]["filters"] = []
    if "access_filter" not in LOGGING_CONFIG["loggers"]["uvicorn.access"]["filters"]:
        LOGGING_CONFIG["loggers"]["uvicorn.access"]["filters"].append("access_filter")

    uvicorn.run(app, host="127.0.0.1", port=7860)
    # demo.launch()
