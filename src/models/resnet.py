"""ResNet backbone with DPE-compatible state-dict keys (0.*, 2.*)."""

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNetClassifier(nn.Sequential):
    def __init__(self, weights="IMAGENET1K_V2", freeze_bn=False, num_classes=2):
        if weights not in (None, "IMAGENET1K_V1", "IMAGENET1K_V2"):
            raise ValueError(f"Unsupported ResNet-50 weights: {weights}")
        network = resnet50(weights=None if weights is None else ResNet50_Weights[weights])
        super().__init__(nn.Sequential(*list(network.children())[:-1]),
                         nn.Flatten(1), nn.Linear(network.fc.in_features, num_classes))
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

    def extract_stage_maps(self, inputs):
        """Return intermediate stage feature maps and pooled 2048-D embedding without altering state_dict."""
        stem = self[0]
        x = stem[3](stem[2](stem[1](stem[0](inputs))))  # conv1 -> bn1 -> relu -> maxpool
        c1 = stem[4](x)   # layer1 [B, 256, 56, 56]
        c2 = stem[5](c1)  # layer2 [B, 512, 28, 28]
        c3 = stem[6](c2)  # layer3 [B, 1024, 14, 14]
        c4 = stem[7](c3)  # layer4 [B, 2048, 7, 7]
        pooled = self[1](stem[8](c4))  # avgpool -> flatten [B, 2048]
        return {"layer1": c1, "layer2": c2, "layer3": c3, "layer4_map": c4, "pooled": pooled}


def load_resnet_checkpoint(path, *, freeze_bn=False, device="cpu", num_classes=2):
    # Never download ImageNet weights when loading a fine-tuned model.
    model = ResNetClassifier(weights=None, freeze_bn=freeze_bn, num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device)
