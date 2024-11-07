from torch import nn
from torchvision import models
# from torchsummary import summary


"""
Conv2dNormActivation:   torch.Size([4, 16, 112, 112])   0
InvertedResidual:       torch.Size([4, 16, 112, 112])   1
InvertedResidual:       torch.Size([4, 24, 56, 56])     2
InvertedResidual:       torch.Size([4, 24, 56, 56])     3   <-  low_level_features
InvertedResidual:       torch.Size([4, 40, 28, 28])     4
InvertedResidual:       torch.Size([4, 40, 28, 28])     5
InvertedResidual:       torch.Size([4, 40, 28, 28])     6
InvertedResidual:       torch.Size([4, 80, 14, 14])     7   <-  output_stride == 8
InvertedResidual:       torch.Size([4, 80, 14, 14])     8
InvertedResidual:       torch.Size([4, 80, 14, 14])     9
InvertedResidual:       torch.Size([4, 80, 14, 14])     10
InvertedResidual:       torch.Size([4, 112, 14, 14])    11
InvertedResidual:       torch.Size([4, 112, 14, 14])    12
InvertedResidual:       torch.Size([4, 160, 7, 7])      13  <-  output_stride == 8 or output_stride == 16
InvertedResidual:       torch.Size([4, 160, 7, 7])      14 
InvertedResidual:       torch.Size([4, 160, 7, 7])      15
Conv2dNormActivation:   torch.Size([4, 960, 7, 7])      16
"""
class MobileNetV3Large(nn.Module):
    def __init__(self, output_stride=16, weights=None):
        super().__init__()

        self.features = nn.Sequential()
        mobilenet_v3_large_features = models.mobilenet_v3_large(weights=weights).features

        if output_stride == 16:
            for name, layer in mobilenet_v3_large_features.named_children():
                if name == '13':
                    layer.block[1][0].stride = (1, 1)
                if name in ('13', '14', '15'):
                    layer.block[1][0].dilation = (2, 2)
                    layer.block[1][0].padding = (4, 4)
                self.features.add_module(name, layer)
        else: # output_stride == 8
            for name, layer in mobilenet_v3_large_features.named_children():
                if name in ('7', '13'):
                    layer.block[1][0].stride = (1, 1)
                if name in ('7', '8', '9', '10', '11', '12'):
                    layer.block[1][0].dilation = (2, 2)
                    layer.block[1][0].padding = (2, 2)
                elif name in ('13', '14', '15'):
                    layer.block[1][0].dilation = (4, 4)
                    layer.block[1][0].padding = (8, 8)
                self.features.add_module(name, layer)

    def forward(self, x):
        out = self.features(x)
        return out
    

"""
Conv2dNormActivation:   torch.Size([4, 32, 112, 112])   0
InvertedResidual:       torch.Size([4, 16, 112, 112])   1
InvertedResidual:       torch.Size([4, 24, 56, 56])     2
InvertedResidual:       torch.Size([4, 24, 56, 56])     3   <-  low_level_features
InvertedResidual:       torch.Size([4, 32, 28, 28])     4
InvertedResidual:       torch.Size([4, 32, 28, 28])     5
InvertedResidual:       torch.Size([4, 32, 28, 28])     6
InvertedResidual:       torch.Size([4, 64, 14, 14])     7   <-  output_stride == 8
InvertedResidual:       torch.Size([4, 64, 14, 14])     8
InvertedResidual:       torch.Size([4, 64, 14, 14])     9
InvertedResidual:       torch.Size([4, 64, 14, 14])     10
InvertedResidual:       torch.Size([4, 96, 14, 14])     11
InvertedResidual:       torch.Size([4, 96, 14, 14])     12
InvertedResidual:       torch.Size([4, 96, 14, 14])     13
InvertedResidual:       torch.Size([4, 160, 7, 7])      14  <-  output_stride == 8 or output_stride == 16
InvertedResidual:       torch.Size([4, 160, 7, 7])      15
InvertedResidual:       torch.Size([4, 160, 7, 7])      16
InvertedResidual:       torch.Size([4, 320, 7, 7])      17
Conv2dNormActivation:   torch.Size([4, 1280, 7, 7])     18
"""
class MobileNetV2(nn.Module):
    def __init__(self, output_stride=16, weights=None):
        super().__init__()

        self.features = nn.Sequential()
        mobilenet_v2_features = models.mobilenet_v2(weights=weights).features

        if output_stride == 16:
            for name, layer in mobilenet_v2_features.named_children():
                if name == '14':
                    layer.conv[1][0].stride = (1, 1)
                elif name in ('15', '16', '17'):
                    layer.conv[1][0].dilation = (2, 2)
                    layer.conv[1][0].padding = (2, 2)
                self.features.add_module(name, layer)
        else: # output_stride == 8
            for name, layer in mobilenet_v2_features.named_children():
                if name in ('7', '14'):
                    layer.conv[1][0].stride = (1, 1)
                if name in ('8', '9', '10', '11', '12', '13', '14'):
                    layer.conv[1][0].dilation = (2, 2)
                    layer.conv[1][0].padding = (2, 2)
                elif name in ('15', '16', '17'):
                    layer.conv[1][0].dilation = (4, 4)
                    layer.conv[1][0].padding = (4, 4)
                self.features.add_module(name, layer)
    
    def forward(self, x):
        out = self.features(x)
        return out


if __name__ == '__main__':
    # print(models.segmentation.deeplabv3_resnet50().backbone)
    print(MobileNetV3Large(8))
