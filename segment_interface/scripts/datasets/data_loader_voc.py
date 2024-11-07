import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as F


def _voc_colormap(N=256):
    bitget = lambda val, idx: (val & (1 << idx)) != 0
 
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (bitget(c, 0) << 7 - j)
            g |= (bitget(c, 1) << 7 - j)
            b |= (bitget(c, 2) << 7 - j)
            c >>= 3
        cmap[i] = [r, g, b]
    return cmap


def _voc_colors_to_indices(voc_colormap):
    colors_to_indices = torch.zeros(256 ** 3)
    for i, color in enumerate(voc_colormap):
        colors_to_indices[(color[0] * 256 + color[1]) * 256 + color[2]] = i
    return colors_to_indices


VOC_COLORMAP = _voc_colormap(21)


VOC_COLORS_TO_INDICES = _voc_colors_to_indices(VOC_COLORMAP)


VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


def voc_color_labels_to_indices(color_labels, voc_colors_to_indices):
    if isinstance(color_labels, Image.Image) and color_labels.mode in ('P', 'RGB'):
        color_labels = F.pil_to_tensor(color_labels.convert('RGB'))

    color_labels_channels = F.get_image_num_channels(color_labels)
    if color_labels_channels != 3:
        raise ValueError(f"Input image permitted channel values are 3, but found {color_labels_channels}")

    color_labels = color_labels.long()
    idx = (color_labels[..., 0, :, :] * 256 + color_labels[..., 1, :, :]) * 256 + color_labels[..., 2, :, :]
    return voc_colors_to_indices[idx]


def voc_indices_to_color_labels(indices, voc_colormap):
    return voc_colormap[indices.long()]


class VOCSegDataset(torch.utils.data.Dataset):
    def __init__(self, voc_root, is_train, transform):
        self.transform = transform

        imgs_dir = os.path.join(voc_root, 'JPEGImages')
        masks_dir = os.path.join(voc_root, 'SegmentationClass')
        splits_dir = os.path.join(voc_root, 'ImageSets', 'Segmentation')
        
        split_file = os.path.join(splits_dir, ('train' if is_train else 'val') + '.txt')
        with open(split_file, 'r') as f:
            file_names = f.read().split()

        self.features = [os.path.join(imgs_dir, file_name + ".jpg") for file_name in file_names]
        self.labels = [os.path.join(masks_dir, file_name + ".png") for file_name in file_names]

    def __getitem__(self, idx):
        feature = Image.open(self.features[idx])
        label = Image.open(self.labels[idx])
        if self.transform:
            feature, label = self.transform(feature, label)
        return feature, label

    def __len__(self):
        return len(self.features)
