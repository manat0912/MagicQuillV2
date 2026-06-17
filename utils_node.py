import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import trange
import torchvision.transforms as T
import torch.nn.functional as F
from typing import Tuple
import scipy.ndimage
import cv2
from train.src.condition.util import HWC3, common_input_validate

def check_image_mask(image, mask, name):
    if len(image.shape) < 4:
        # image tensor shape should be [B, H, W, C], but batch somehow is missing
        image = image[None,:,:,:]
    
    if len(mask.shape) > 3:
        # mask tensor shape should be [B, H, W] but we get [B, H, W, C], image may be?
        # take first mask, red channel
        mask = (mask[:,:,:,0])[:,:,:]
    elif len(mask.shape) < 3:
        # mask tensor shape should be [B, H, W] but batch somehow is missing
        mask = mask[None,:,:]

    if image.shape[0] > mask.shape[0]:
        print(name, "gets batch of images (%d) but only %d masks" % (image.shape[0], mask.shape[0]))
        if mask.shape[0] == 1: 
            print(name, "will copy the mask to fill batch")
            mask = torch.cat([mask] * image.shape[0], dim=0)
        else:
            print(name, "will add empty masks to fill batch")
            empty_mask = torch.zeros([image.shape[0] - mask.shape[0], mask.shape[1], mask.shape[2]])
            mask = torch.cat([mask, empty_mask], dim=0)
    elif image.shape[0] < mask.shape[0]:
        print(name, "gets batch of images (%d) but too many (%d) masks" % (image.shape[0], mask.shape[0]))
        mask = mask[:image.shape[0],:,:]

    return (image, mask)


def cv2_resize_shortest_edge(image, size):
    h, w = image.shape[:2]
    if h < w:
        new_h = size
        new_w = int(round(w / h * size))
    else:
        new_w = size
        new_h = int(round(h / w * size))
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image

def apply_color(img, res=512):
    img = cv2_resize_shortest_edge(img, res)
    h, w = img.shape[:2]

    input_img_color = cv2.resize(img, (w//64, h//64), interpolation=cv2.INTER_CUBIC)  
    input_img_color = cv2.resize(input_img_color, (w, h), interpolation=cv2.INTER_NEAREST)
    return input_img_color

#Color T2I like multiples-of-64, upscale methods are fixed.
class ColorDetector:
    def __call__(self, input_image=None, detect_resolution=512, output_type=None, **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image = HWC3(input_image)
        detected_map = HWC3(apply_color(input_image, detect_resolution))
        
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
            
        return detected_map


class InpaintPreprocessor:
    def preprocess(self, image, mask, black_pixel_for_xinsir_cn=False):
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(image.shape[1], image.shape[2]), mode="bilinear")
        mask = mask.movedim(1,-1).expand((-1,-1,-1,3))
        image = image.clone()
        if black_pixel_for_xinsir_cn:
            masked_pixel = 0.0
        else:
            masked_pixel = -1.0
        image[mask > 0.5] = masked_pixel
        return (image,)


class BlendInpaint:
    def blend_inpaint(self, inpaint: torch.Tensor, original: torch.Tensor, mask, kernel: int, sigma:int, origin=None) -> Tuple[torch.Tensor]:

        original, mask = check_image_mask(original, mask, 'Blend Inpaint')

        if len(inpaint.shape) < 4:
            # image tensor shape should be [B, H, W, C], but batch somehow is missing
            inpaint = inpaint[None,:,:,:]

        if inpaint.shape[0] < original.shape[0]:
            print("Blend Inpaint gets batch of original images (%d) but only (%d) inpaint images" % (original.shape[0], inpaint.shape[0]))
            original= original[:inpaint.shape[0],:,:]
            mask = mask[:inpaint.shape[0],:,:]

        if inpaint.shape[0] > original.shape[0]:
            # batch over inpaint
            count = 0
            original_list = []
            mask_list = []
            origin_list = []
            while (count < inpaint.shape[0]):
                for i in range(original.shape[0]):
                    original_list.append(original[i][None,:,:,:])
                    mask_list.append(mask[i][None,:,:])
                    if origin is not None:
                        origin_list.append(origin[i][None,:])
                    count += 1
                    if count >= inpaint.shape[0]:
                        break
            original = torch.concat(original_list, dim=0)
            mask = torch.concat(mask_list, dim=0)
            if origin is not None:
                origin = torch.concat(origin_list, dim=0)

        if kernel % 2 == 0:
            kernel += 1
        transform = T.GaussianBlur(kernel_size=(kernel, kernel), sigma=(sigma, sigma))

        ret = []
        blurred = []
        for i in range(inpaint.shape[0]):
            if origin is None:
                blurred_mask = transform(mask[i][None,None,:,:]).to(original.device).to(original.dtype)
                blurred.append(blurred_mask[0])

                result = torch.nn.functional.interpolate(
                    inpaint[i][None,:,:,:].permute(0, 3, 1, 2), 
                    size=(
                        original[i].shape[0], 
                        original[i].shape[1],
                    )
                ).permute(0, 2, 3, 1).to(original.device).to(original.dtype)
            else:
                # got mask from CutForInpaint
                height, width, _ = original[i].shape
                x0 = origin[i][0].item()
                y0 = origin[i][1].item()

                if mask[i].shape[0] < height or mask[i].shape[1] < width:
                    padded_mask = F.pad(input=mask[i], pad=(x0, width-x0-mask[i].shape[1], 
                                                            y0, height-y0-mask[i].shape[0]), mode='constant', value=0)
                else:
                    padded_mask = mask[i]
                blurred_mask = transform(padded_mask[None,None,:,:]).to(original.device).to(original.dtype)
                blurred.append(blurred_mask[0][0])

                result = F.pad(input=inpaint[i], pad=(0, 0, x0, width-x0-inpaint[i].shape[1], 
                                                      y0, height-y0-inpaint[i].shape[0]), mode='constant', value=0)
                result = result[None,:,:,:].to(original.device).to(original.dtype)

            ret.append(original[i] * (1.0 - blurred_mask[0][0][:,:,None]) + result[0] * blurred_mask[0][0][:,:,None])

        return (torch.stack(ret), torch.stack(blurred), )


def resize_mask(mask, shape):
    return torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(shape[0], shape[1]), mode="bilinear").squeeze(1)

class JoinImageWithAlpha:
    def join_image_with_alpha(self, image: torch.Tensor, alpha: torch.Tensor):
        batch_size = min(len(image), len(alpha))
        out_images = []

        alpha = 1.0 - resize_mask(alpha, image.shape[1:])
        for i in range(batch_size):
           out_images.append(torch.cat((image[i][:,:,:3], alpha[i].unsqueeze(2)), dim=2))

        result = (torch.stack(out_images),)
        return result

class GrowMask:
    def expand_mask(self, mask, expand, tapered_corners):
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c],
                           [1, 1, 1],
                           [c, 1, c]])
        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        out = []
        for m in mask:
            output = m.numpy()
            for _ in range(abs(expand)):
                if expand < 0:
                    output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                else:
                    output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            output = torch.from_numpy(output)
            out.append(output)
        return (torch.stack(out, dim=0),)

class InvertMask:
    def invert(self, mask):
        out = 1.0 - mask
        return (out,)