import warnings
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
import os

from einops import rearrange

from .util import HWC3, nms, safe_step, resize_image_with_pad, common_input_validate, get_intensity_mask, combine_layers

from .pidi import pidinet
from .ted import TED
from .lineart import Generator as LineartGenerator
from .informative_drawing import Generator
from .hed import ControlNetHED_Apache2

from pathlib import Path

from skimage import morphology
import argparse
from tqdm import tqdm


PREPROCESSORS_ROOT = os.getenv("PREPROCESSORS_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "models/preprocessors"))


class HEDDetector:
    def __init__(self, netNetwork):
        self.netNetwork = netNetwork
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, filename="ControlNetHED.pth"):
        model_path = os.path.join(PREPROCESSORS_ROOT, filename)

        netNetwork = ControlNetHED_Apache2()
        netNetwork.load_state_dict(torch.load(model_path, map_location='cpu'))
        netNetwork.float().eval()

        return cls(netNetwork)
    
    def to(self, device):
        self.netNetwork.to(device)
        self.device = device
        return self
    

    def __call__(self, input_image, detect_resolution=512, safe=False, output_type=None, scribble=True, upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)

        assert input_image.ndim == 3
        H, W, C = input_image.shape
        with torch.no_grad():
            image_hed = torch.from_numpy(input_image).float().to(self.device)
            image_hed = rearrange(image_hed, 'h w c -> 1 c h w')
            edges = self.netNetwork(image_hed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
            if safe:
                edge = safe_step(edge)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = edge
        
        if scribble:
            detected_map = nms(detected_map, 127, 3.0)
            detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
            detected_map[detected_map > 4] = 255
            detected_map[detected_map < 255] = 0

        detected_map = HWC3(remove_pad(detected_map))

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map


class CannyDetector:
    def __call__(self, input_image=None, low_threshold=100, high_threshold=200, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        detected_map, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        detected_map = cv2.Canny(detected_map, low_threshold, high_threshold)
        detected_map = HWC3(remove_pad(detected_map))
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map

class PidiNetDetector:
    def __init__(self, netNetwork):
        self.netNetwork = netNetwork
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, filename="table5_pidinet.pth"):
        model_path = os.path.join(PREPROCESSORS_ROOT, filename)

        netNetwork = pidinet()
        netNetwork.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path)['state_dict'].items()})
        netNetwork.eval()

        return cls(netNetwork)

    def to(self, device):
        self.netNetwork.to(device)
        self.device = device
        return self
    
    def __call__(self, input_image, detect_resolution=512, safe=False, output_type=None, scribble=True, apply_filter=False, upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        detected_map, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        
        detected_map = detected_map[:, :, ::-1].copy()
        with torch.no_grad():
            image_pidi = torch.from_numpy(detected_map).float().to(self.device)
            image_pidi = image_pidi / 255.0
            image_pidi = rearrange(image_pidi, 'h w c -> 1 c h w')
            edge = self.netNetwork(image_pidi)[-1]
            edge = edge.cpu().numpy()
            if apply_filter:
                edge = edge > 0.5 
            if safe:
                edge = safe_step(edge)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = edge[0, 0]

        if scribble:
            detected_map = nms(detected_map, 127, 3.0)
            detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
            detected_map[detected_map > 4] = 255
            detected_map[detected_map < 255] = 0

        detected_map = HWC3(remove_pad(detected_map))

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map

class TEDDetector:
    def __init__(self, model):
        self.model = model
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, filename="7_model.pth"):
        model_path = os.path.join(PREPROCESSORS_ROOT, filename)
        model = TED()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return cls(model)
    
    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def __call__(self, input_image, detect_resolution=512, safe_steps=2, upscale_method="INTER_CUBIC", output_type=None, **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)

        H, W, _ = input_image.shape
        with torch.no_grad():
            image_teed = torch.from_numpy(input_image.copy()).float().to(self.device)
            image_teed = rearrange(image_teed, 'h w c -> 1 c h w')
            edges = self.model(image_teed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
            if safe_steps != 0:
                edge = safe_step(edge, safe_steps)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
    
        detected_map = remove_pad(HWC3(edge))
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map[..., :3])
        
        return detected_map
    
class LineartStandardDetector:
    def __call__(self, input_image=None, guassian_sigma=6.0, intensity_threshold=8, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        
        x = input_image.astype(np.float32)
        g = cv2.GaussianBlur(x, (0, 0), guassian_sigma)
        intensity = np.min(g - x, axis=2).clip(0, 255)
        intensity /= max(16, np.median(intensity[intensity > intensity_threshold]))
        intensity *= 127
        detected_map = intensity.clip(0, 255).astype(np.uint8)
        
        detected_map = HWC3(remove_pad(detected_map))
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
        return detected_map
    
class AnyLinePreprocessor:
    def __init__(self, mteed_model, lineart_standard_detector):
        self.device = "cpu"
        self.mteed_model = mteed_model
        self.lineart_standard_detector = lineart_standard_detector

    @classmethod
    def from_pretrained(cls, mteed_filename="MTEED.pth"):
        mteed_model = TEDDetector.from_pretrained(filename=mteed_filename)
        lineart_standard_detector = LineartStandardDetector()
        return cls(mteed_model, lineart_standard_detector)

    def to(self, device):
        self.mteed_model.to(device)
        self.device = device
        return self

    def __call__(self, image, resolution=512, lineart_lower_bound=0, lineart_upper_bound=1, object_min_size=36, object_connectivity=1):
        # Process the image with MTEED model
        mteed_result = self.mteed_model(image, detect_resolution=resolution)

        # Process the image with the lineart standard preprocessor
        lineart_result = self.lineart_standard_detector(image, guassian_sigma=2, intensity_threshold=3, resolution=resolution)

        _lineart_result  = get_intensity_mask(lineart_result, lower_bound=lineart_lower_bound, upper_bound=lineart_upper_bound)
        _cleaned = morphology.remove_small_objects(_lineart_result.astype(bool), min_size=object_min_size, connectivity=object_connectivity)
        _lineart_result = _lineart_result * _cleaned
        _mteed_result = mteed_result

        result = combine_layers(_mteed_result, _lineart_result)
        # print(result.shape)
        return result

class LineartDetector:
    def __init__(self, model, coarse_model):
        self.model = model
        self.model_coarse = coarse_model
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, filename="sk_model.pth", coarse_filename="sk_model2.pth"):
        model_path = os.path.join(PREPROCESSORS_ROOT, filename)
        coarse_model_path = os.path.join(PREPROCESSORS_ROOT, coarse_filename)

        model = LineartGenerator(3, 1, 3)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        coarse_model = LineartGenerator(3, 1, 3)
        coarse_model.load_state_dict(torch.load(coarse_model_path, map_location="cpu"))
        coarse_model.eval()

        return cls(model, coarse_model)
    
    def to(self, device):
        self.model.to(device)
        self.model_coarse.to(device)
        self.device = device
        return self
    
    def __call__(self, input_image, coarse=False, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        detected_map, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)

        model = self.model_coarse if coarse else self.model
        assert detected_map.ndim == 3
        with torch.no_grad():
            image = torch.from_numpy(detected_map).float().to(self.device)
            image = image / 255.0
            image = rearrange(image, 'h w c -> 1 c h w')
            line = model(image)[0][0]

            line = line.cpu().numpy()
            line = (line * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = HWC3(line)
        detected_map = remove_pad(255 - detected_map)
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map


class InformativeDetector:
    def __init__(self, anime_model, contour_model):
        self.anime_model = anime_model
        self.contour_model = contour_model
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, anime_filename="anime_style.pth", contour_filename="contour_style.pth"):
        anime_model_path = os.path.join(PREPROCESSORS_ROOT, anime_filename)
        contour_model_path = os.path.join(PREPROCESSORS_ROOT, contour_filename)

        # 创建两个Generator模型
        anime_model = Generator(3, 1, 3)  # input_nc=3, output_nc=1, n_blocks=3
        anime_model.load_state_dict(torch.load(anime_model_path, map_location="cpu"))
        anime_model.eval()

        contour_model = Generator(3, 1, 3)  # input_nc=3, output_nc=1, n_blocks=3
        contour_model.load_state_dict(torch.load(contour_model_path, map_location="cpu"))
        contour_model.eval()

        return cls(anime_model, contour_model)
    
    def to(self, device):
        self.anime_model.to(device)
        self.contour_model.to(device)
        self.device = device
        return self
    
    def __call__(self, input_image, style="anime", detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        """
        提取sketch
        
        Args:
            input_image: 输入图像
            style: "anime" 或 "contour"
            detect_resolution: 检测分辨率
            output_type: 输出类型
            upscale_method: 上采样方法
        """
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        detected_map, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)

        # 选择模型
        model = self.anime_model if style == "anime" else self.contour_model
        
        assert detected_map.ndim == 3
        with torch.no_grad():
            image = torch.from_numpy(detected_map).float().to(self.device)
            image = image / 255.0
            # 转换维度 (h, w, c) -> (1, c, h, w)
            image = image.permute(2, 0, 1).unsqueeze(0)
            
            # 生成sketch
            sketch = model(image)
            sketch = sketch[0][0]  # 取出第一个batch的第一个通道

            sketch = sketch.cpu().numpy()
            sketch = (sketch * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = HWC3(sketch)
        detected_map = remove_pad(255 - detected_map)  # 反转颜色
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map