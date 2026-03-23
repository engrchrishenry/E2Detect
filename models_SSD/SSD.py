from torch import nn
import torch

from .vgg import VGG, VGG_event
from .box_head import BoxHead


class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = VGG(input_size=cfg.MODEL.INPUT_SIZE)
        self.box_head = BoxHead(cfg)

        if cfg.MODEL.BACKBONE.PRETRAINED:
            self.backbone.init_from_pretrain(torch.load(cfg.MODEL.BACKBONE.WEIGHTS_PATH, map_location=cfg.MODEL.BACKBONE.MAP_LOCATION))
    
    def forward(self, images, targets=None):
        features = self.backbone(images)
        temp_list = list(features)
        backbone_feature_raw = temp_list.pop(0)
        backbone_feature_normed = temp_list[0]
        features = tuple(temp_list)
        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses
        return detections, (backbone_feature_raw, backbone_feature_normed)


class SSDDetector_event(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = VGG_event(input_size=cfg.MODEL.INPUT_SIZE)
        self.box_head = BoxHead(cfg)

        if cfg.MODEL.BACKBONE.PRETRAINED:
            self.backbone.init_from_pretrain(torch.load(cfg.MODEL.BACKBONE.WEIGHTS_PATH, map_location=cfg.MODEL.BACKBONE.MAP_LOCATION))

    def forward(self, images, targets=None):
        features = self.backbone(images)
        backbone_feature = features[0]
        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses
        return detections, backbone_feature

