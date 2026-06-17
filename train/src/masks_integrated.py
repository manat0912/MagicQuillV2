import math
import random
import logging
from enum import Enum

import cv2
import numpy as np
import random

LOGGER = logging.getLogger(__name__)

class LinearRamp:
    def __init__(self, start_value=0, end_value=1, start_iter=-1, end_iter=0):
        self.start_value = start_value
        self.end_value = end_value
        self.start_iter = start_iter
        self.end_iter = end_iter

    def __call__(self, i):
        if i < self.start_iter:
            return self.start_value
        if i >= self.end_iter:
            return self.end_value
        part = (i - self.start_iter) / (self.end_iter - self.start_iter)
        return self.start_value * (1 - part) + self.end_value * part

class DrawMethod(Enum):
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'

def make_random_irregular_mask(shape, max_angle=4, max_len=60, max_width=20, min_times=0, max_times=10,
                               draw_method=DrawMethod.LINE):
    """生成不规则mask - 基于角度和长度的线条"""
    draw_method = DrawMethod(draw_method)

    height, width = shape
    mask = np.zeros((height, width), np.float32)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(max_len)
            brush_w = 5 + np.random.randint(max_width)
            end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, width)
            end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, height)
            if draw_method == DrawMethod.LINE:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
            elif draw_method == DrawMethod.CIRCLE:
                cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1., thickness=-1)
            elif draw_method == DrawMethod.SQUARE:
                radius = brush_w // 2
                mask[start_y - radius:start_y + radius, start_x - radius:start_x + radius] = 1
            start_x, start_y = end_x, end_y
    return mask[None, ...]


def make_random_rectangle_mask(shape, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3):
    """生成随机矩形mask"""
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    bbox_max_size = min(bbox_max_size, height - margin * 2, width - margin * 2)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        box_width = np.random.randint(bbox_min_size, bbox_max_size)
        box_height = np.random.randint(bbox_min_size, bbox_max_size)
        start_x = np.random.randint(margin, width - margin - box_width + 1)
        start_y = np.random.randint(margin, height - margin - box_height + 1)
        mask[start_y:start_y + box_height, start_x:start_x + box_width] = 1
    return mask[None, ...]


def make_random_superres_mask(shape, min_step=2, max_step=4, min_width=1, max_width=3):
    """生成超分辨率风格的规则网格mask"""
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    step_x = np.random.randint(min_step, max_step + 1)
    width_x = np.random.randint(min_width, min(step_x, max_width + 1))
    offset_x = np.random.randint(0, step_x)

    step_y = np.random.randint(min_step, max_step + 1)
    width_y = np.random.randint(min_width, min(step_y, max_width + 1))
    offset_y = np.random.randint(0, step_y)

    for dy in range(width_y):
        mask[offset_y + dy::step_y] = 1
    for dx in range(width_x):
        mask[:, offset_x + dx::step_x] = 1
    return mask[None, ...]


def make_brush_stroke_mask(shape, num_strokes_range=(1, 5), stroke_width_range=(5, 30), 
                          max_offset=50, num_points_range=(5, 15)):
    """生成笔刷描边样式的mask - 基于随机偏移的连续线条"""
    num_strokes = random.randint(*num_strokes_range)
    height, width = shape
    mask = np.zeros((height, width), dtype=np.float32)
    
    for _ in range(num_strokes):
        # 随机起点
        start_x = random.randint(0, width)
        start_y = random.randint(0, height)
        
        # 随机描边参数
        num_points = random.randint(*num_points_range)
        stroke_width = random.randint(*stroke_width_range)
        
        points = [(start_x, start_y)]
        
        # 生成连续的点
        for i in range(num_points):
            prev_x, prev_y = points[-1]
            # 添加随机偏移
            dx = random.randint(-max_offset, max_offset)
            dy = random.randint(-max_offset, max_offset)
            new_x = max(0, min(width, prev_x + dx))
            new_y = max(0, min(height, prev_y + dy))
            points.append((new_x, new_y))
        
        # 绘制描边
        for i in range(len(points) - 1):
            cv2.line(mask, points[i], points[i+1], 1.0, stroke_width)
    
    return mask[None, ...]


class RandomIrregularMaskGenerator:
    """不规则mask生成器"""
    def __init__(self, max_angle=4, max_len=60, max_width=20, min_times=0, max_times=10, ramp_kwargs=None,
                 draw_method=DrawMethod.LINE):
        self.max_angle = max_angle
        self.max_len = max_len
        self.max_width = max_width
        self.min_times = min_times
        self.max_times = max_times
        self.draw_method = draw_method
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, img, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_max_len = int(max(1, self.max_len * coef))
        cur_max_width = int(max(1, self.max_width * coef))
        cur_max_times = int(self.min_times + 1 + (self.max_times - self.min_times) * coef)
        return make_random_irregular_mask(img.shape[1:], max_angle=self.max_angle, max_len=cur_max_len,
                                          max_width=cur_max_width, min_times=self.min_times, max_times=cur_max_times,
                                          draw_method=self.draw_method)


class RandomRectangleMaskGenerator:
    """矩形mask生成器"""
    def __init__(self, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3, ramp_kwargs=None):
        self.margin = margin
        self.bbox_min_size = bbox_min_size
        self.bbox_max_size = bbox_max_size
        self.min_times = min_times
        self.max_times = max_times
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, img, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_bbox_max_size = int(self.bbox_min_size + 1 + (self.bbox_max_size - self.bbox_min_size) * coef)
        cur_max_times = int(self.min_times + (self.max_times - self.min_times) * coef)
        return make_random_rectangle_mask(img.shape[1:], margin=self.margin, bbox_min_size=self.bbox_min_size,
                                          bbox_max_size=cur_bbox_max_size, min_times=self.min_times,
                                          max_times=cur_max_times)


class RandomSuperresMaskGenerator:
    """超分辨率mask生成器"""
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, img, iter_i=None):
        return make_random_superres_mask(img.shape[1:], **self.kwargs)


class BrushStrokeMaskGenerator:
    """笔刷描边mask生成器"""
    def __init__(self, num_strokes_range=(1, 5), stroke_width_range=(5, 30), 
                 max_offset=50, num_points_range=(5, 15), ramp_kwargs=None):
        self.num_strokes_range = num_strokes_range
        self.stroke_width_range = stroke_width_range
        self.max_offset = max_offset
        self.num_points_range = num_points_range
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, img, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_num_strokes = int(max(1, self.num_strokes_range[1] * coef))
        cur_stroke_width_range = (
            int(max(1, self.stroke_width_range[0] * coef)),
            int(max(1, self.stroke_width_range[1] * coef))
        )
        return make_brush_stroke_mask(
            img.shape[1:], 
            num_strokes_range=(cur_num_strokes, cur_num_strokes),
            stroke_width_range=cur_stroke_width_range,
            max_offset=self.max_offset,
            num_points_range=self.num_points_range
        )


class DumbAreaMaskGenerator:
    """简单区域mask生成器"""
    min_ratio = 0.1
    max_ratio = 0.35
    default_ratio = 0.225

    def __init__(self, is_training):
        #Parameters:
        #    is_training(bool): If true - random rectangular mask, if false - central square mask
        self.is_training = is_training

    def _random_vector(self, dimension):
        if self.is_training:
            lower_limit = math.sqrt(self.min_ratio)
            upper_limit = math.sqrt(self.max_ratio)
            mask_side = round((random.random() * (upper_limit - lower_limit) + lower_limit) * dimension)
            u = random.randint(0, dimension-mask_side-1)
            v = u+mask_side 
        else:
            margin = (math.sqrt(self.default_ratio) / 2) * dimension
            u = round(dimension/2 - margin)
            v = round(dimension/2 + margin)
        return u, v

    def __call__(self, img, iter_i=None, raw_image=None):
        c, height, width = img.shape
        mask = np.zeros((height, width), np.float32)
        x1, x2 = self._random_vector(width)
        y1, y2 = self._random_vector(height)
        mask[x1:x2, y1:y2] = 1
        return mask[None, ...]


class IntegratedMaskGenerator:
    """集成的mask生成器 - 支持多种mask类型混合"""
    def __init__(self, irregular_proba=1/4, irregular_kwargs=None,
                 box_proba=1/4, box_kwargs=None,
                 segm_proba=1/4, segm_kwargs=None,
                 brush_stroke_proba=1/4, brush_stroke_kwargs=None,
                 superres_proba=0, superres_kwargs=None,
                 squares_proba=0, squares_kwargs=None,
                 invert_proba=0):
        self.probas = []
        self.gens = []

        if irregular_proba > 0:
            self.probas.append(irregular_proba)
            if irregular_kwargs is None:
                irregular_kwargs = {}
            else:
                irregular_kwargs = dict(irregular_kwargs)
            irregular_kwargs['draw_method'] = DrawMethod.LINE
            self.gens.append(RandomIrregularMaskGenerator(**irregular_kwargs))

        if box_proba > 0:
            self.probas.append(box_proba)
            if box_kwargs is None:
                box_kwargs = {}
            self.gens.append(RandomRectangleMaskGenerator(**box_kwargs))

        if brush_stroke_proba > 0:
            self.probas.append(brush_stroke_proba)
            if brush_stroke_kwargs is None:
                brush_stroke_kwargs = {}
            self.gens.append(BrushStrokeMaskGenerator(**brush_stroke_kwargs))

        if superres_proba > 0:
            self.probas.append(superres_proba)
            if superres_kwargs is None:
                superres_kwargs = {}
            self.gens.append(RandomSuperresMaskGenerator(**superres_kwargs))

        if squares_proba > 0:
            self.probas.append(squares_proba)
            if squares_kwargs is None:
                squares_kwargs = {}
            else:
                squares_kwargs = dict(squares_kwargs)
            squares_kwargs['draw_method'] = DrawMethod.SQUARE
            self.gens.append(RandomIrregularMaskGenerator(**squares_kwargs))

        self.probas = np.array(self.probas, dtype='float32')
        self.probas /= self.probas.sum()
        self.invert_proba = invert_proba

    def __call__(self, img, iter_i=None, raw_image=None):
        kind = np.random.choice(len(self.probas), p=self.probas)
        gen = self.gens[kind]
        result = gen(img, iter_i=iter_i, raw_image=raw_image)
        if self.invert_proba > 0 and random.random() < self.invert_proba:
            result = 1 - result
        return result


def get_mask_generator(kind, kwargs):
    """获取mask生成器的工厂函数"""
    if kind is None:
        kind = "integrated"
    if kwargs is None:
        kwargs = {}

    if kind == "integrated":
        cl = IntegratedMaskGenerator
    elif kind == "irregular":
        cl = RandomIrregularMaskGenerator
    elif kind == "rectangle":
        cl = RandomRectangleMaskGenerator
    elif kind == "brush_stroke":
        cl = BrushStrokeMaskGenerator
    elif kind == "superres":
        cl = RandomSuperresMaskGenerator
    elif kind == "dumb":
        cl = DumbAreaMaskGenerator
    else:
        raise NotImplementedError(f"No such generator kind = {kind}")
    return cl(**kwargs)