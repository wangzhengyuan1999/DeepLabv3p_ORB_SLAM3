from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.fcn import FCNHead
from resnest.torch import resnest

if __name__ == '__main__':
    from _aspp import *
    from backbone import mobilenet, resnet
else:
    from ._aspp import *
    from .backbone import mobilenet, resnet


# deeplabv3
def deeplabv3_mobilenet_v3_large(num_classes=21, output_stride=16, pretrained_backbone=False, weights_backbone=None, aux_loss=None, weights=None):
    if not weights_backbone and pretrained_backbone:
        weights_backbone = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1

    return models.segmentation.deeplabv3_mobilenet_v3_large(num_classes=num_classes, weights_backbone=weights_backbone, aux_loss=aux_loss, weights=weights)


def deeplabv3_mobilenet_v3_large_correct(num_classes=21, output_stride=16, pretrained_backbone=False, weights_backbone=None, aux_loss=None, weights=None):
    if not weights_backbone and pretrained_backbone:
        weights_backbone = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1

    backbone = mobilenet.MobileNetV3Large(output_stride, weights=weights_backbone)
    backbone = backbone.features

    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    out_pos = stage_indices[-1]
    out_inplanes = backbone[out_pos].out_channels
    aux_pos = stage_indices[-4]
    aux_inplanes = backbone[aux_pos].out_channels
    return_layers = {str(out_pos): "out"}
    if aux_loss:
        return_layers[str(aux_pos)] = "aux"

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    aux_classifier = FCNHead(aux_inplanes, num_classes) if aux_loss else None
    classifier = DeepLabv3Head(out_inplanes, num_classes, output_stride)

    return DeepLabv3(backbone, classifier, aux_classifier)


def deeplabv3_resnet50(num_classes=21, output_stride=16, pretrained_backbone=False, weights_backbone=None, aux_loss=None, weights=None):
    if not weights_backbone and pretrained_backbone:
        weights_backbone = models.ResNet50_Weights.IMAGENET1K_V1

    if output_stride == 8:
        return models.segmentation.deeplabv3_resnet50(num_classes=num_classes, weights_backbone=weights_backbone, aux_loss=aux_loss, weights=weights)
    else:
        backbone = models.resnet50(weights=weights_backbone, replace_stride_with_dilation=[False, False, True])

        return_layers = {"layer4": "out"}
        if aux_loss:
            return_layers["layer3"] = "aux"

        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        aux_classifier = FCNHead(1024, num_classes) if aux_loss else None
        classifier = DeepLabv3Head(2048, num_classes, 16)

        return DeepLabv3plus(backbone, classifier, aux_classifier)


def deeplabv3_resnet101(num_classes=21, output_stride=16, pretrained_backbone=False, weights_backbone=None, aux_loss=None, weights=None):
    if not weights_backbone and pretrained_backbone:
        weights_backbone = models.ResNet101_Weights.IMAGENET1K_V1

    if output_stride == 8:
        return models.segmentation.deeplabv3_resnet101(num_classes=num_classes, weights_backbone=weights_backbone, aux_loss=aux_loss, weights=weights)
    else:
        backbone = models.resnet101(weights=weights_backbone, replace_stride_with_dilation=[False, False, True])

        return_layers = {"layer4": "out"}
        if aux_loss:
            return_layers["layer3"] = "aux"

        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        aux_classifier = FCNHead(1024, num_classes) if aux_loss else None
        classifier = DeepLabv3Head(2048, num_classes, 16)

        return DeepLabv3plus(backbone, classifier, aux_classifier)


# deeplabv3plus_mobilenet_v3
def deeplabv3plus_mobilenet_v3_large(num_classes=21, output_stride=16, pretrained_backbone=False, weights_backbone=None, aux_loss=None, weights=None):
    if not weights_backbone and pretrained_backbone:
        weights_backbone = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1

    backbone = mobilenet.MobileNetV3Large(output_stride, weights=weights_backbone)
    backbone = backbone.features

    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    out_pos = stage_indices[-1]
    out_inplanes = backbone[out_pos].out_channels
    aux_pos = stage_indices[-4]
    aux_inplanes = backbone[aux_pos].out_channels
    low_level_pos = 3
    low_level_inplanes = backbone[low_level_pos].out_channels
   
    return_layers = {str(out_pos): "out"}
    return_layers[str(low_level_pos)] = "low_level"
    if aux_loss:
        return_layers[str(aux_pos)] = "aux"

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    aux_classifier = FCNHead(aux_inplanes, num_classes) if aux_loss else None
    classifier = DeepLabv3plusHead(out_inplanes, low_level_inplanes, num_classes, output_stride)

    return DeepLabv3plus(backbone, classifier, aux_classifier)


# deeplabv3plus_mobilenet_v2
def deeplabv3plus_mobilenet_v2(num_classes=21, output_stride=16, pretrained_backbone=False, weights_backbone=None, aux_loss=None, weights=None):
    if not weights_backbone and pretrained_backbone:
        weights_backbone = models.MobileNet_V2_Weights.IMAGENET1K_V1

    backbone = mobilenet.MobileNetV2(output_stride, weights=weights_backbone)
    backbone = backbone.features

    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    out_pos = stage_indices[-1]
    out_inplanes = backbone[out_pos].out_channels
    aux_pos = stage_indices[-4]
    aux_inplanes = backbone[aux_pos].out_channels
    low_level_pos = 3
    low_level_inplanes = backbone[low_level_pos].out_channels
   
    return_layers = {str(out_pos): "out"}
    return_layers[str(low_level_pos)] = "low_level"
    if aux_loss:
        return_layers[str(aux_pos)] = "aux"

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    aux_classifier = FCNHead(aux_inplanes, num_classes) if aux_loss else None
    classifier = DeepLabv3plusHead(out_inplanes, low_level_inplanes, num_classes, output_stride)

    return DeepLabv3plus(backbone, classifier, aux_classifier)


# deeplabv3plus_resnet18 34
def deeplabv3plus_resnet18(num_classes=21, output_stride=16, pretrained_backbone=False, weights_backbone=None, aux_loss=None, weights=None):
    if not weights_backbone and pretrained_backbone:
        weights_backbone = models.ResNet18_Weights.IMAGENET1K_V1

    backbone = resnet.ResNet18(output_stride, weights=weights_backbone)
    
    return_layers = {"layer4": "out"}
    return_layers["layer1"] = "low_level"
    if aux_loss:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(256, num_classes) if aux_loss else None
    classifier = DeepLabv3plusHead(512, 64, num_classes, output_stride)
    
    return DeepLabv3plus(backbone, classifier, aux_classifier)


def deeplabv3plus_resnet34(num_classes=21, output_stride=16, pretrained_backbone=False, weights_backbone=None, aux_loss=None, weights=None):
    if not weights_backbone and pretrained_backbone:
        weights_backbone = models.ResNet34_Weights.IMAGENET1K_V1

    backbone = resnet.ResNet34(output_stride, weights=weights_backbone)
    
    return_layers = {"layer4": "out"}
    return_layers["layer1"] = "low_level"
    if aux_loss:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(256, num_classes) if aux_loss else None
    classifier = DeepLabv3plusHead(512, 64, num_classes, output_stride)
    
    return DeepLabv3plus(backbone, classifier, aux_classifier)


# deeplabv3plus_resnet50 101 152
def deeplabv3plus_resnet50(num_classes=21, output_stride=16, pretrained_backbone=False, weights_backbone=None, aux_loss=None, weights=None):
    if not weights_backbone and pretrained_backbone:
        weights_backbone = models.ResNet50_Weights.IMAGENET1K_V1

    replace_stride_with_dilation = [False, False, True] if output_stride == 16 else [False, True, True]
    backbone = models.resnet50(weights=weights_backbone, replace_stride_with_dilation=replace_stride_with_dilation)
    
    return_layers = {"layer4": "out"}
    return_layers["layer1"] = "low_level"
    if aux_loss:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(1024, num_classes) if aux_loss else None
    classifier = DeepLabv3plusHead(2048, 256, num_classes, output_stride)
    
    return DeepLabv3plus(backbone, classifier, aux_classifier)


def deeplabv3plus_resnet101(num_classes=21, output_stride=16, pretrained_backbone=False, weights_backbone=None, aux_loss=None, weights=None):
    if not weights_backbone and pretrained_backbone:
        weights_backbone = models.ResNet101_Weights.IMAGENET1K_V1

    replace_stride_with_dilation = [False, False, True] if output_stride == 16 else [False, True, True]
    backbone = models.resnet101(weights=weights_backbone, replace_stride_with_dilation=replace_stride_with_dilation)
    
    return_layers = {"layer4": "out"}
    return_layers["layer1"] = "low_level"
    if aux_loss:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(1024, num_classes) if aux_loss else None
    classifier = DeepLabv3plusHead(2048, 256, num_classes, output_stride)
    
    return DeepLabv3plus(backbone, classifier, aux_classifier)


def deeplabv3plus_resnet152(num_classes=21, output_stride=16, pretrained_backbone=False, weights_backbone=None, aux_loss=None, weights=None):
    if not weights_backbone and pretrained_backbone:
        weights_backbone = models.ResNet152_Weights.IMAGENET1K_V1

    replace_stride_with_dilation = [False, False, True] if output_stride == 16 else [False, True, True]
    backbone = models.resnet152(weights=weights_backbone, replace_stride_with_dilation=replace_stride_with_dilation)
    
    return_layers = {"layer4": "out"}
    return_layers["layer1"] = "low_level"
    if aux_loss:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(1024, num_classes) if aux_loss else None
    classifier = DeepLabv3plusHead(2048, 256, num_classes, output_stride)
    
    return DeepLabv3plus(backbone, classifier, aux_classifier)


# deeplabv3plussc_resnet50
def deeplabv3plussc_resnet50(num_classes=21, output_stride=16, pretrained_backbone=False, weights_backbone=None, aux_loss=None, weights=None):
    if not weights_backbone and pretrained_backbone:
        weights_backbone = models.ResNet50_Weights.IMAGENET1K_V1

    replace_stride_with_dilation = [False, False, True] if output_stride == 16 else [False, True, True]
    backbone = models.resnet50(weights=weights_backbone, replace_stride_with_dilation=replace_stride_with_dilation)
    
    return_layers = {"layer4": "out"}
    return_layers["layer1"] = "low_level"
    if aux_loss:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(1024, num_classes) if aux_loss else None
    classifier = DeepLabv3plusHeadSC(2048, 256, num_classes, output_stride)
    
    return DeepLabv3plus(backbone, classifier, aux_classifier)


# deeplabv3plus_resnest50
def deeplabv3plus_resnest50(num_classes=21, output_stride=16, pretrained_backbone=False, weights_backbone=None, aux_loss=None, weights=None):
    # if not weights_backbone and pretrained_backbone:
    #     weights_backbone = models.ResNet50_Weights.IMAGENET1K_V1

    # replace_stride_with_dilation = [False, False, True] if output_stride == 16 else [False, True, True]
    # backbone = models.resnet50(weights=weights_backbone, replace_stride_with_dilation=replace_stride_with_dilation)
    dilation = 2 if output_stride == 16 else 4
    backbone = resnest.resnest50(pretrained=pretrained_backbone, dilation=dilation)
    
    return_layers = {"layer4": "out"}
    return_layers["layer1"] = "low_level"
    if aux_loss:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(1024, num_classes) if aux_loss else None
    classifier = DeepLabv3plusHead(2048, 256, num_classes, output_stride)
    
    return DeepLabv3plus(backbone, classifier, aux_classifier)


# deeplabv3plussc_resnest50
def deeplabv3plussc_resnest50(num_classes=21, output_stride=16, pretrained_backbone=False, weights_backbone=None, aux_loss=None, weights=None):
    # if not weights_backbone and pretrained_backbone:
    #     weights_backbone = models.ResNet50_Weights.IMAGENET1K_V1

    # replace_stride_with_dilation = [False, False, True] if output_stride == 16 else [False, True, True]
    # backbone = models.resnet50(weights=weights_backbone, replace_stride_with_dilation=replace_stride_with_dilation)
    dilation = 2 if output_stride == 16 else 4
    backbone = resnest.resnest50(pretrained=pretrained_backbone, dilation=dilation)
    
    return_layers = {"layer4": "out"}
    return_layers["layer1"] = "low_level"
    if aux_loss:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(1024, num_classes) if aux_loss else None
    classifier = DeepLabv3plusHeadSC(2048, 256, num_classes, output_stride)
    
    return DeepLabv3plus(backbone, classifier, aux_classifier)


# DeepLabv3 and DeepLabv3plus
class DeepLabv3(nn.Module):
    def __init__(self, backbone, classifier, aux_classifier=None):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        result = OrderedDict()
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = x

        return result
    

class DeepLabv3plus(DeepLabv3):
    pass


class DeepLabv3Head(nn.Sequential):
    def __init__(self, in_channels, num_classes, output_stride):
        super().__init__(
            ASPP(in_channels, [6, 12, 18] if output_stride == 16 else [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, feature):
        return super().forward(feature["out"])


class DeepLabv3plusHead(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, output_stride):
        super().__init__()

        self.low_level_cov = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False), nn.BatchNorm2d(48),
            nn.ReLU()
        )
        
        self.aspp = ASPP(in_channels, [6, 12, 18] if output_stride == 16 else [12, 24, 36])

        self.project = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, feature):
        low_level_feature = self.low_level_cov(feature["low_level"])
        output_feature = F.interpolate(self.aspp(feature["out"]), 
                                       size=low_level_feature.shape[-2:], mode="bilinear", 
                                       align_corners=False)
        
        return self.project(torch.cat((low_level_feature, output_feature), 1))


class DeepLabv3plusHeadSC(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, output_stride):
        super().__init__()

        self.low_level_cov = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False), nn.BatchNorm2d(48),
            nn.ReLU()
        )
        
        self.aspp = ASPPSC(in_channels, [6, 12, 18] if output_stride == 16 else [12, 24, 36])

        self.project = nn.Sequential(
            # nn.Conv2d(48 + 256, 256, 3, padding=1, bias=False),
            # DW
            nn.Conv2d(48 + 256, 48 + 256, 3, padding=1, groups=48 + 256, bias=False),
            # PW
            nn.Conv2d(48 + 256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Conv2d(256, 256, 3, padding=1, bias=False),
            # DW
            nn.Conv2d(256, 256, 3, padding=1, groups=256, bias=False),
            # PW
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, feature):
        low_level_feature = self.low_level_cov(feature["low_level"])
        output_feature = F.interpolate(self.aspp(feature["out"]), 
                                       size=low_level_feature.shape[-2:], mode="bilinear", 
                                       align_corners=False)
        
        return self.project(torch.cat((low_level_feature, output_feature), 1))
