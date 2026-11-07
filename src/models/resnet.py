"""ResNet backbone with DPE-compatible state-dict keys (0.*, 2.*)."""

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNetClassifier(nn.Sequential):
    def __init__(self, weights="IMAGENET1K_V2", freeze_bn=False):
        if weights not in (None, "IMAGENET1K_V1", "IMAGENET1K_V2"):
            raise ValueError(f"Unsupported ResNet-50 weights: {weights}")
        network = resnet50(weights=None if weights is None else ResNet50_Weights[weights])
        super().__init__(nn.Sequential(*list(network.children())[:-1]),
                         nn.Flatten(1), nn.Linear(network.fc.in_features, 2))
        self.freeze_bn = freeze_bn

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_bn:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def extract_features(self, inputs):
        return self[1](self[0](inputs))


def load_resnet_checkpoint(path, *, freeze_bn=False, device="cpu"):
    # Never download ImageNet weights when loading a fine-tuned model.
    model = ResNetClassifier(weights=None, freeze_bn=freeze_bn)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device)
