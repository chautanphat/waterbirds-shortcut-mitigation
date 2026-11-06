import torch.nn as nn
import torchvision.models as models


class Classifier(nn.Module):
    def __init__(self, config, pre_trained=True):
        super(Classifier, self).__init__()

        if "waterbird" in config.dataset_type:
            out_dim = 2
        elif "spawrious" in config.dataset_type:
            out_dim = 4
        elif "spurious_vehicles" in config.dataset_type:
            out_dim = 4
        else:
            raise ValueError("invalid dataset_type: ", config.dataset_type)

        self.config = config
        self.out_dim = out_dim
        if config.model_class == "resnet":
            self.net = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1 if pre_trained else None
            )
            self.head = nn.Linear(self.net.fc.in_features, out_dim)
            self.net.fc = nn.Identity()
        elif config.model_class == "vit":
            self.net = models.vit_b_16(
                weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pre_trained else None
            )
            self.head = nn.Linear(self.net.heads.head.in_features, out_dim)
            self.net.heads.head = nn.Identity()
        else:
            raise ValueError("invalid model_class:", config.model_class)

    def forward(self, x):
        x = self.net(x)
        x = self.head(x)
        return x

    def embeddings(self, x):
        return self.net(x)


if __name__ == "__main__":

    import torch
    from utils import Config

    config = Config("./methods/erm/config.yaml")
    model = Classifier(config)

    x = torch.rand((4, 3, 224, 224))
    y = model(x)
    e = model.embeddings(x)

    print(x.shape)
    print(y.shape)
    print(e.shape)
