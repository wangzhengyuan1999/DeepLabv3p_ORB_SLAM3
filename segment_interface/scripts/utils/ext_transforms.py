from collections.abc import Sequence
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import transforms
import numbers
import numpy as np


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class ExtCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lbl):
        for t in self.transforms:
            img, lbl = t(img, lbl)
        return img, lbl

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class ExtRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        if torch.rand(1) < self.p:
            return F.hflip(img), F.hflip(lbl)
        return img, lbl

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class ExtRandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        if torch.rand(1) < self.p:
            return F.vflip(img), F.vflip(lbl)
        return img, lbl

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class ExtCenterCrop(object):
    def __init__(self, size):
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

    def __call__(self, img, lbl):
        return F.center_crop(img, self.size), F.center_crop(lbl, self.size)

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"


class ExtRandomScale(object):
    def __init__(self, scale_range, interpolation=F.InterpolationMode.BILINEAR):
        self.scale_range = scale_range
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        scale = torch.empty(1).uniform_(scale[0], scale[1]).item()
        return ExtScale(scale, self.interpolation)(img, lbl)

    def __repr__(self):
        detail = f"(scale_range={self.scale_range}, interpolation={self.interpolation.value}"
        return f"{self.__class__.__name__}{detail}"


class ExtScale(object):
    def __init__(self, scale, interpolation=F.InterpolationMode.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        _, height, width = F.get_dimensions(img)
        target_size = (int(height * self.scale), int(width * self.scale))
        return F.resize(img, target_size, self.interpolation), F.resize(lbl, target_size, F.InterpolationMode.NEAREST)

    def __repr__(self):
        detail = f"(scale={self.scale}, interpolation={self.interpolation.value}"
        return f"{self.__class__.__name__}{detail}"


class ExtRandomRotation(object):
    def __init__(self, degrees, interpolation=F.InterpolationMode.NEAREST, expand=False, center=None, fill=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.fill = fill

    @staticmethod
    def get_params(degrees):
        return transforms.RandomRotation.get_params(degrees)

    def __call__(self, img, lbl):
        fill = self.fill
        channels, _, _ = F.get_dimensions(img)
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]
        angle = self.get_params(self.degrees)
        return (F.rotate(img, angle, self.interpolation, self.expand, self.center, self.fill), 
                F.rotate(lbl, angle, self.interpolation, self.expand, self.center, self.fill))

    def __repr__(self):
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(degrees={self.degrees}"
        format_string += f", interpolation={interpolate_str}"
        format_string += f", expand={self.expand}"
        if self.center is not None:
            format_string += f", center={self.center}"
        if self.fill is not None:
            format_string += f", fill={self.fill}"
        format_string += ")"
        return format_string


class ExtToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor. This transform does not support torchscript.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.

    .. note::
        Because the input image is scaled to [0.0, 1.0], this transformation should not be used when
        transforming target image masks. See the `references`_ for implementing the transforms for image masks.

    .. _references: https://github.com/pytorch/vision/tree/main/references/segmentation
    """
    
    def __call__(self, img, lbl):
        return F.to_tensor(img), torch.from_numpy(np.array(lbl))

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ExtNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, lbl):
        return F.normalize(img, self.mean, self.std), lbl

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class ExtRandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()

        self.size = tuple(_setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        return transforms.RandomCrop.get_params(img, output_size)

    def __call__(self, img, lbl):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            lbl = F.pad(lbl, self.padding, self.fill, self.padding_mode)

        _, height, width = F.get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            lbl = F.pad(lbl, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            lbl = F.pad(lbl, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(lbl, i, j, h, w)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, padding={self.padding}, pad_if_needed={self.pad_if_needed}, fill={self.fill}, padding_mode={self.padding_mode})"


class ExtResize(object):
    def __init__(self, size, interpolation=F.InterpolationMode.BILINEAR, max_size=None):
        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size

    def __call__(self, img, lbl):
        return (F.resize(img, self.size, self.interpolation, self.max_size), 
                F.resize(lbl, self.size, F.InterpolationMode.NEAREST, self.max_size))

    def __repr__(self):
        detail = f"(size={self.size}, interpolation={self.interpolation.value}, max_size={self.max_size})"
        return f"{self.__class__.__name__}{detail}"


if __name__ == '__main__':
    # train_transform = ExtCompose([
    #     ExtRandomCrop((224, 224), pad_if_needed=True),
    #     ExtToTensor(),
    #     ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ExtResize((224, 224))
    # ])
    # print(train_transform)
    pass
