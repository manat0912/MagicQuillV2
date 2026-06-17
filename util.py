import random
from collections import Counter
import numpy as np
from torchvision import transforms
import cv2  # OpenCV
import torch
import re
import io
import base64
from PIL import Image, ImageOps

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

def get_bounding_box_from_mask(mask, padded=False):
    mask = mask.squeeze()
    rows, cols = torch.where(mask > 0.5)
    if len(rows) == 0 or len(cols) == 0:
        return (0, 0, 0, 0)
    height, width = mask.shape
    if padded:
        padded_size = max(width, height)
        if width < height:
            offset_x = (padded_size - width) / 2
            offset_y = 0
        else:
            offset_y = (padded_size - height) / 2
            offset_x = 0
        top_left_x = round(float((torch.min(cols).item() + offset_x) / padded_size), 3)
        bottom_right_x = round(float((torch.max(cols).item() + offset_x) / padded_size), 3)
        top_left_y = round(float((torch.min(rows).item() + offset_y) / padded_size), 3)
        bottom_right_y = round(float((torch.max(rows).item() + offset_y) / padded_size), 3)
    else:
        offset_x = 0
        offset_y = 0

        top_left_x = round(float(torch.min(cols).item() / width), 3)
        bottom_right_x = round(float(torch.max(cols).item() / width), 3)
        top_left_y = round(float(torch.min(rows).item() / height), 3)
        bottom_right_y = round(float(torch.max(rows).item() / height), 3)

    
    return (top_left_x, top_left_y, bottom_right_x, bottom_right_y)

def extract_bbox(text):
    pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
    match = re.search(pattern, text)
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4)))

def resize_bbox(bbox, width_ratio, height_ratio):
    x1, y1, x2, y2 = bbox
    new_x1 = int(x1 * width_ratio)
    new_y1 = int(y1 * height_ratio)
    new_x2 = int(x2 * width_ratio)
    new_y2 = int(y2 * height_ratio)

    return (new_x1, new_y1, new_x2, new_y2)


def tensor_to_base64(tensor, quality=80, method=6):
    tensor = tensor.squeeze(0).clone().detach().cpu()
    
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float64 or tensor.dtype == torch.float16:
        tensor *= 255
    tensor = tensor.to(torch.uint8)
    
    if tensor.ndim == 2:  # 灰度图像
        pil_image = Image.fromarray(tensor.numpy(), 'L')
        pil_image = pil_image.convert('RGB')
    elif tensor.ndim == 3:
        if tensor.shape[2] == 1:  # 单通道
            pil_image = Image.fromarray(tensor.numpy().squeeze(2), 'L')
            pil_image = pil_image.convert('RGB')
        elif tensor.shape[2] == 3:  # RGB
            pil_image = Image.fromarray(tensor.numpy(), 'RGB')
        elif tensor.shape[2] == 4:  # RGBA
            pil_image = Image.fromarray(tensor.numpy(), 'RGBA')
        else:
            raise ValueError(f"Unsupported number of channels: {tensor.shape[2]}")
    else:
        raise ValueError(f"Unsupported tensor dimensions: {tensor.ndim}")
    
    buffered = io.BytesIO()
    pil_image.save(buffered, format="WEBP", quality=quality, method=method, lossless=False)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def load_and_preprocess_image(image_path, convert_to='RGB', has_alpha=False):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)

    if image.mode == 'RGBA':
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    image = image.convert(convert_to)
    image_array = np.array(image).astype(np.float32) / 255.0

    if has_alpha and convert_to == 'RGBA':
        image_tensor = torch.from_numpy(image_array)[None,]
    else:
        if len(image_array.shape) == 3 and image_array.shape[2] > 3:
            image_array = image_array[:, :, :3]
        image_tensor = torch.from_numpy(image_array)[None,]
    
    return image_tensor

def process_background(base64_image, convert_to='RGB', size=None):
    image_data = read_base64_image(base64_image)
    image = Image.open(image_data)
    image = ImageOps.exif_transpose(image)
    image = image.convert(convert_to)

    # Select preferred size by closest aspect ratio, then snap to multiple_of
    w0, h0 = image.size
    aspect_ratio = (w0 / h0) if h0 != 0 else 1.0
    # Choose the (w, h) whose aspect ratio is closest to the input
    _, tw, th = min((abs(aspect_ratio - w / h), w, h) for (w, h) in PREFERRED_KONTEXT_RESOLUTIONS)
    multiple_of = 16  # default: vae_scale_factor (8) * 2
    tw = (tw // multiple_of) * multiple_of
    th = (th // multiple_of) * multiple_of

    if (w0, h0) != (tw, th):
        image = image.resize((tw, th), resample=Image.BICUBIC)

    image_array = np.array(image).astype(np.uint8)
    image_tensor = torch.from_numpy(image_array)[None,]
    return image_tensor

def read_base64_image(base64_image):
    if base64_image.startswith("data:image/png;base64,"):
        base64_image = base64_image.split(",")[1]
    elif base64_image.startswith("data:image/jpeg;base64,"):
        base64_image = base64_image.split(",")[1]
    elif base64_image.startswith("data:image/webp;base64,"):
        base64_image = base64_image.split(",")[1]
    else:
        raise ValueError("Unsupported image format.")
    image_data = base64.b64decode(base64_image)
    return io.BytesIO(image_data)

def create_alpha_mask(image_path):
    """Create an alpha mask from the alpha channel of an image."""
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    mask = torch.zeros((1, image.height, image.width), dtype=torch.float32)
    if 'A' in image.getbands():
        alpha_channel = np.array(image.getchannel('A')).astype(np.float32) / 255.0
        mask[0] = 1.0 - torch.from_numpy(alpha_channel)
    return mask

def get_mask_bbox(mask_tensor, padding=10):
    assert len(mask_tensor.shape) == 3 and mask_tensor.shape[0] == 1
    _, H, W = mask_tensor.shape
    mask_2d = mask_tensor.squeeze(0)
    
    y_coords, x_coords = torch.where(mask_2d > 0)
    
    if len(y_coords) == 0:
        return None
    
    x_min = int(torch.min(x_coords))
    y_min = int(torch.min(y_coords))
    x_max = int(torch.max(x_coords))
    y_max = int(torch.max(y_coords))
    
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(W - 1, x_max + padding)
    y_max = min(H - 1, y_max + padding)
    
    return x_min, y_min, x_max, y_max

def tensor_to_pil(tensor):
    tensor = tensor.squeeze(0).clone().detach().cpu()
    if tensor.dtype in [torch.float32, torch.float64, torch.float16]:
        if tensor.max() <= 1.0:
            tensor *= 255
        tensor = tensor.to(torch.uint8)
    
    if tensor.ndim == 2:  # 灰度图像 [H, W]
        return Image.fromarray(tensor.numpy(), 'L')
    elif tensor.ndim == 3:
        if tensor.shape[2] == 1:  # 单通道 [H, W, 1]
            return Image.fromarray(tensor.numpy().squeeze(2), 'L')
        elif tensor.shape[2] >= 3:  # RGB [H, W, 3]
            return Image.fromarray(tensor.numpy(), 'RGB')
        else:
            raise ValueError(f"不支持的通道数: {tensor.shape[2]}")
    else:
        raise ValueError(f"不支持的tensor维度: {tensor.ndim}")