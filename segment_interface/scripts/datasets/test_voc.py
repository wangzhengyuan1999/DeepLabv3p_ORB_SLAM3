import os
import torch
import torchvision
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


img_path = '~/data/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg'
lbl_path = '~/data/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png'


# torchvision.io.read_image RGB 读取
img = torchvision.io.read_image(lbl_path, torchvision.io.image.ImageReadMode.RGB)
print(img.shape) # torch.Size([3, 281, 500])
print(F.to_pil_image(img).size, F.to_pil_image(img).mode, len(F.to_pil_image(img).split())) # (500, 281) RGB 3
print(np.array(img).shape) # (3, 281, 500)

# torchvision.io.read_image UNCHANGED 读取
img = torchvision.io.read_image(lbl_path)
print(img.shape) # torch.Size([1, 281, 500])
print(F.to_pil_image(img).size, F.to_pil_image(img).mode, len(F.to_pil_image(img).split())) # (500, 281) L 1
print(np.array(img).shape) # (1, 281, 500)


# PIL.Image 
img = Image.open(lbl_path)
print(img.size, img.mode, len(img.split())) # (500, 281) P 1
print(F.to_tensor(img).shape) # torch.Size([1, 281, 500])
print(np.array(img).shape) # (281, 500)
print(torch.from_numpy(np.array(img)).shape) # (281, 500)
# print(np.array(img.getpalette()).reshape(-1, 3))
img_1 = np.array(img)
print(np.unique(np.array(img))) # [  0   1  15 255]
img_1[img_1 == 255] = 0
print(np.unique(img_1)) # [ 0  1 15]
img_1 = F.to_tensor(img)
print(img_1.unique()) # tensor([0.0000, 0.0039, 0.0588, 1.0000])
img_1[img_1 == 1] = 0
print(img_1.unique()) # tensor([0.0000, 0.0039, 0.0588])

img = Image.open(img_path)
print(img.size, img.mode, len(img.split())) # (500, 281) RGB 3
print(F.to_tensor(img).shape) # torch.Size([3, 281, 500])
print(np.array(img).shape) # (281, 500, 3)

# F.to_tensor 和 F.pil_to_tensor
img = Image.open(lbl_path)
print(F.to_tensor(img).dtype)
print(F.to_tensor(img).unique())
print(F.pil_to_tensor(img).dtype)
print(F.pil_to_tensor(img).unique())



# plt.imshow()
# plt.show()

# def voc_colormap(N=256):
#     bitget = lambda val, idx: (val & (1 << idx)) != 0
 
#     cmap = np.zeros((N, 3), dtype=np.uint8)
#     for i in range(N):
#         r = g = b = 0
#         c = i
#         for j in range(8):
#             r |= (bitget(c, 0) << 7 - j)
#             g |= (bitget(c, 1) << 7 - j)
#             b |= (bitget(c, 2) << 7 - j)
#             c >>= 3
#         cmap[i] = [r, g, b]
#     return cmap

# print(voc_colormap(21))
