import torch.nn as nn
import torchvision.models as models


class Detector(nn.Module):
    def __init__(self, config, pre_trained=True):
        super(Detector, self).__init__()
        self.config = config
        self.net = models.efficientnet_b0(
            weights=(
                models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pre_trained else None
            )
        )

        out_dim = config.detector_patch_resolution**2
        self.net.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(1280, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.net(x)
        x = x.view(
            batch_size,
            self.config.detector_patch_resolution,
            self.config.detector_patch_resolution,
        )
        return x
