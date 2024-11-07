from torch import nn
from torchvision import models


"""models.resnet18()
Conv2d:             torch.Size([1, 64, 112, 112])   conv1
BatchNorm2d:        torch.Size([1, 64, 112, 112])   bn1
ReLU:               torch.Size([1, 64, 112, 112])   relu 
MaxPool2d:          torch.Size([1, 64, 56, 56])     maxpool
Sequential:         torch.Size([1, 64, 56, 56])     layer1  <-  low_level_features
Sequential:         torch.Size([1, 128, 28, 28])    layer2
Sequential:         torch.Size([1, 256, 14, 14])    layer3  <-  output_stride == 8
Sequential:         torch.Size([1, 512, 7, 7])      layer4  <-  output_stride == 8 or output_stride == 16
AdaptiveAvgPool2d:  torch.Size([1, 512, 1, 1])      avgpool <-  del
Linear:             torch.Size([1, 1000])           fc      <-  del
"""
class ResNet18(nn.Module):
    def __init__(self, output_stride, weights=None):
        super().__init__()

        resnet18 = models.resnet18(weights=weights)

        if output_stride == 16:
            for name, layer in resnet18.named_children():
                if name == 'layer4':
                    layer[0].conv1.stride = (1, 1)
                    layer[0].conv2.dilation = (2, 2)
                    layer[0].conv2.padding = (2, 2)
                    layer[0].downsample[0].stride = (1, 1)
                    layer[1].conv1.dilation = (2, 2)
                    layer[1].conv1.padding = (2, 2)
                    layer[1].conv2.dilation = (2, 2)
                    layer[1].conv2.padding = (2, 2)
                self.add_module(name, layer)
        else: # output_stride == 8
            for name, layer in resnet18.named_children():
                if name == 'layer3':
                    layer[0].conv1.stride = (1, 1)
                    layer[0].conv2.dilation = (2, 2)
                    layer[0].conv2.padding = (2, 2)
                    layer[0].downsample[0].stride = (1, 1)
                    layer[1].conv1.dilation = (2, 2)
                    layer[1].conv1.padding = (2, 2)
                    layer[1].conv2.dilation = (2, 2)
                    layer[1].conv2.padding = (2, 2)
                elif name == 'layer4':
                    layer[0].conv1.stride = (1, 1)
                    layer[0].conv1.dilation = (2, 2)
                    layer[0].conv1.padding = (2, 2)
                    layer[0].conv2.dilation = (4, 4)
                    layer[0].conv2.padding = (4, 4)
                    layer[0].downsample[0].stride = (1, 1)
                    layer[1].conv1.dilation = (4, 4)
                    layer[1].conv1.padding = (4, 4)
                    layer[1].conv2.dilation = (4, 4)
                    layer[1].conv2.padding = (4, 4)
                self.add_module(name, layer)


class ResNet34(nn.Module):
    def __init__(self, output_stride, weights=None):
        super().__init__()

        resnet34 = models.resnet34(weights=weights)

        if output_stride == 16:
            for name, layer in resnet34.named_children():
                if name == 'layer4':
                    for i in range(len(layer)):
                        if i == 0:
                            layer[0].conv1.stride = (1, 1)
                            layer[0].conv2.dilation = (2, 2)
                            layer[0].conv2.padding = (2, 2)
                            layer[0].downsample[0].stride = (1, 1)
                        else:
                            layer[i].conv1.dilation = (2, 2)
                            layer[i].conv1.padding = (2, 2)
                            layer[i].conv2.dilation = (2, 2)
                            layer[i].conv2.padding = (2, 2)
                self.add_module(name, layer)
        else: # output_stride == 8
            for name, layer in resnet34.named_children():
                if name == 'layer3':
                    for i in range(len(layer)):
                        if i == 0:
                            layer[0].conv1.stride = (1, 1)
                            layer[0].conv2.dilation = (2, 2)
                            layer[0].conv2.padding = (2, 2)
                            layer[0].downsample[0].stride = (1, 1)
                        else:
                            layer[i].conv1.dilation = (2, 2)
                            layer[i].conv1.padding = (2, 2)
                            layer[i].conv2.dilation = (2, 2)
                            layer[i].conv2.padding = (2, 2)
                elif name == 'layer4':
                    for i in range(len(layer)):
                        if i == 0:
                            layer[0].conv1.stride = (1, 1)
                            layer[0].conv2.dilation = (4, 4)
                            layer[0].conv2.padding = (4, 4)
                            layer[0].downsample[0].stride = (1, 1)
                        else:
                            layer[i].conv1.dilation = (4, 4)
                            layer[i].conv1.padding = (4, 4)
                            layer[i].conv2.dilation = (4, 4)
                            layer[i].conv2.padding = (4, 4)
                self.add_module(name, layer)


if __name__ == '__main__':
    # print(models.segmentation.deeplabv3_resnet50().backbone)
    # print(models.resnet34())
    pass