from PIL import Image
from datasets import load_dataset
from torchvision import transforms
import random
import torch
import os
from .pipeline_flux_kontext_control import PREFERRED_KONTEXT_RESOLUTIONS
import numpy as np
from src.condition.edge_extraction import (
    CannyDetector, PidiNetDetector, TEDDetector, LineartStandardDetector, HEDdetector,
    AnyLinePreprocessor, LineartDetector, InformativeDetector
)

Image.MAX_IMAGE_PIXELS = None

def multiple_16(num: float):
    return int(round(num / 16) * 16)

def load_image_safely(image_path, size, root="/mnt/robby-b1/common/datasets/"):
    image_path = os.path.join(root, image_path)
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print("file error: "+image_path)
        with open("failed_images.txt", "a") as f:
            f.write(f"{image_path}\n")
        return Image.new("RGB", (size, size), (255, 255, 255))
    
def choose_kontext_resolution_from_wh(width: int, height: int):
    aspect_ratio = width / max(1, height)
    _, best_w, best_h = min(
        (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
    )
    return best_w, best_h

class EdgeExtractorManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EdgeExtractorManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.edge_extractors = None
            self.device = None
            self._initialized = True

    def set_device(self, device):
        self.device = device

    def get_edge_extractors(self, device=None):
        # 强制在CPU上初始化，避免DataLoader子进程中触发CUDA初始化
        current_device = "cpu"
        if device is not None:
            self.set_device(current_device)

        if self.edge_extractors is None or len(self.edge_extractors) == 0:
            self.edge_extractors = [
                ("canny", CannyDetector()),
                ("pidinet", PidiNetDetector.from_pretrained().to(current_device)),
                ("ted", TEDDetector.from_pretrained().to(current_device)),
                # ("lineart_standard", LineartStandardDetector()),
                ("hed",  HEDdetector.from_pretrained().to(current_device)),
                ("anyline", AnyLinePreprocessor.from_pretrained().to(current_device)),
                # ("lineart", LineartDetector.from_pretrained().to(current_device)),
                ("informative", InformativeDetector.from_pretrained().to(current_device)),
            ]

        return self.edge_extractors

edge_extractor_manager = EdgeExtractorManager()

def collate_fn(examples):
    if examples[0].get("cond_pixel_values") is not None:
        cond_pixel_values = torch.stack([example["cond_pixel_values"] for example in examples])
        cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
    else:
        cond_pixel_values = None
    source_pixel_values = None

    target_pixel_values = torch.stack([example["pixel_values"] for example in examples])
    target_pixel_values = target_pixel_values.to(memory_format=torch.contiguous_format).float()
    token_ids_clip = torch.stack([example["token_ids_clip"] for example in examples])
    token_ids_t5 = torch.stack([example["token_ids_t5"] for example in examples])

    return {
        "cond_pixel_values": cond_pixel_values,
        "source_pixel_values": source_pixel_values,
        "pixel_values": target_pixel_values,
        "text_ids_1": token_ids_clip,
        "text_ids_2": token_ids_t5,
    }


def make_train_dataset_inpaint_mask(args, tokenizers, accelerator=None):
    # 加载CSV数据集：三列，第一列为图片相对路径，第三列为caption
    if args.train_data_dir is not None:
        dataset = load_dataset('csv', data_files=args.train_data_dir)

    # 列名兼容处理：使用第 0 列作为图片路径，第 2 列作为caption
    column_names = dataset["train"].column_names
    image_col = column_names[0]
    caption_col = column_names[2] if len(column_names) >= 3 else column_names[-1]

    size = args.cond_size

    # 设备设置（用于分布式时将部分检测器放到对应GPU）
    if accelerator is not None:
        device = accelerator.device
        edge_extractor_manager.set_device(device)
    else:
        device = "cpu"

    # Transforms
    to_tensor_and_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # 与 jsonl_datasets_edge.py 保持一致：Resize -> CenterCrop -> ToTensor -> Normalize
    cond_train_transforms = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    tokenizer_clip = tokenizers[0]
    tokenizer_t5 = tokenizers[1]

    def tokenize_prompt_clip_t5(examples):
        captions_raw = examples[caption_col]
        captions = []
        for c in captions_raw:
            if isinstance(c, str):
                if random.random() < 0.25:
                    captions.append("")
                else:
                    captions.append(c)
            else:
                captions.append("")

        text_inputs_clip = tokenizer_clip(
            captions,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids_1 = text_inputs_clip.input_ids

        text_inputs_t5 = tokenizer_t5(
            captions,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_t5.input_ids
        return text_input_ids_1, text_input_ids_2

    def preprocess_train(examples):
        batch = {}

        img_paths = examples[image_col]

        target_tensors = []
        cond_tensors = []

        for p in img_paths:
            # Load image by joining with root in load_image_safely
            img = load_image_safely(p, size)
            img = img.convert("RGB")

            # Resize to Kontext preferred resolution for target
            w, h = img.size
            best_w, best_h = choose_kontext_resolution_from_wh(w, h)
            img_rs = img.resize((best_w, best_h), resample=Image.BILINEAR)
            target_tensor = to_tensor_and_norm(img_rs)

            # Build edge condition
            extractor_name, extractor = random.choice(edge_extractor_manager.get_edge_extractors())
            img_np = np.array(img)
            if extractor_name == "informative":
                edge = extractor(img_np, style="contour")
            else:
                edge = extractor(img_np)

            if extractor_name == "ted":
                th = 128
            else:
                th = 32

            edge_np = np.array(edge) if isinstance(edge, Image.Image) else edge
            if edge_np.ndim == 3:
                edge_np = edge_np[..., 0]
            edge_bin = (edge_np > th).astype(np.float32)
            edge_pil = Image.fromarray((edge_bin * 255).astype(np.uint8))
            edge_tensor = cond_train_transforms(edge_pil)
            edge_tensor = edge_tensor.repeat(3, 1, 1)

            target_tensors.append(target_tensor)
            cond_tensors.append(edge_tensor)

        batch["pixel_values"] = target_tensors
        batch["cond_pixel_values"] = cond_tensors

        batch["token_ids_clip"], batch["token_ids_t5"] = tokenize_prompt_clip_t5(examples)
        return batch

    if accelerator is not None:
        with accelerator.main_process_first():
            train_dataset = dataset["train"].with_transform(preprocess_train)
    else:
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset