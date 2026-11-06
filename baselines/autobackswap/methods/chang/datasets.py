import os

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image

from methods.common.waterbird import WaterBird
from methods.common.spawrious import Spawrious
from methods.common.vehicles import SpuriousVehicles

# segmentation thresholds for factual inpainting (used to replace the background with noise)
WB_F_SEG_THR = 0.5
SPAWRIOUS_F_SEG_THR = 0.0
SPURIOUS_VEHICLES_F_SEG_THR = 0.0

# factual inpainting gaussian noise std
INFILL_SIGMA = 0.2


def set_rng(dataset, seed):
    rng = torch.Generator().manual_seed(seed)

    def attach(d):
        if hasattr(d, "datasets"):  # ConcatDataset
            for sub in d.datasets:
                attach(sub)
        else:
            d.rng = rng

    attach(dataset)


def generate_factual_image(image, segment, thr, rng):
    C, H, W = image.shape
    fg_mask = segment > thr
    fg_mask = fg_mask.repeat(C, 1, 1)
    f_image = torch.rand(C, 1, 1, generator=rng).repeat(1, H, W)
    f_image += torch.randn(C, H, W, generator=rng) * INFILL_SIGMA
    f_image = torch.clamp(f_image, 0, 1)
    f_image[fg_mask] = image[fg_mask]
    return f_image


class WaterbirdsCounterfactual(WaterBird):

    def __getitem__(self, index):

        to_tensor = T.ToTensor()
        folder, file = self.metadata.img_filename.iloc[index].split("/")

        # extract label
        label = int(self.metadata.y.iloc[index])

        # load image
        image_path = os.path.join(self.root, self.folder_name, "images", folder, file)
        image = to_tensor(Image.open(image_path).convert("RGB"))

        # load counterfactual image
        cf_path = os.path.join(
            self.root, self.folder_name, "counterfactuals", folder, file
        )
        cf_image = to_tensor(Image.open(cf_path).convert("RGB"))

        # load segmentation image
        segment_path = os.path.join(
            self.root,
            self.folder_name,
            "segmentations",
            folder,
            file.replace(".jpg", ".png"),
        )
        segmentation = to_tensor(Image.open(segment_path).convert("L"))

        # create factual image (fg + random noise bg)
        f_image = generate_factual_image(image, segmentation, WB_F_SEG_THR, self.rng)

        return (
            self.transform(image),
            self.transform(f_image),
            self.transform(cf_image),
            label,
        )


class SpawriousCounterfactual(Spawrious):

    def __getitem__(self, index):

        if self.load_images:
            to_tensor = T.ToTensor()

            # default image
            image_path = self.image_paths[index]
            image = to_tensor(Image.open(image_path).convert("RGB"))

            # counterfactual image
            cf_image_path = image_path.replace(
                "spawrious224", "spawrious224_counterfactuals"
            )
            cf_image = to_tensor(Image.open(cf_image_path).convert("RGB"))

            # factual image
            segment_path = image_path.replace(
                "spawrious224", "spawrious224_segmentation_masks"
            ).replace(".png", ".tiff")
            segment = to_tensor(Image.open(segment_path).convert("L"))
            f_image = generate_factual_image(
                image, segment, SPAWRIOUS_F_SEG_THR, self.rng
            )

        if self.yield_groups:
            location = self.locations[index]
            return None, self.label, location

        return (
            self.transform(image),
            self.transform(f_image),
            self.transform(cf_image),
            self.label,
        )


class SpuriousVehiclesCounterfactual(SpuriousVehicles):

    def __getitem__(self, index):

        if self.load_images:
            to_tensor = T.ToTensor()

            # default image
            image_path = self.image_paths[index]
            image = to_tensor(Image.open(image_path).convert("RGB"))

            # counterfactual image
            cf_image_path = image_path.replace(
                "spurious_vehicles", "spurious_vehicles_counterfactuals"
            )
            cf_image = to_tensor(Image.open(cf_image_path).convert("RGB"))

            # factual image
            segment_path = (
                image_path.replace(
                    "spurious_vehicles", "spurious_vehicles_segmentation_masks"
                )[:-4]
                + ".tiff"
            )
            segment = to_tensor(Image.open(segment_path).convert("L"))

            assert (
                image.shape[1:] == segment.shape[1:]
            ), f"image and segmentation size mismatch: \n {image_path} \n {cf_image_path} \n {segment_path}"
            f_image = generate_factual_image(
                image, segment, SPURIOUS_VEHICLES_F_SEG_THR, self.rng
            )

        if self.yield_groups:
            return None, self.label, self.spur_att

        return (
            self.transform(image),
            self.transform(f_image),
            self.transform(cf_image),
            self.label,
        )


if __name__ == "__main__":

    from torchvision.utils import save_image
    from utils import Config, get_train_transform, seed_everything

    # config = Config("./methods/chang/config_wb.yaml")
    # config = Config("./methods/chang/config_spawrious.yaml")
    config = Config("./methods/chang/config_spurious_vehicles.yaml")

    seed_everything(config.random_seed)

    train_transform = get_train_transform(config, to_tensor=False)

    # dataset = WaterbirdsCounterfactual(
    #     root="./data",
    #     split="valid",
    #     groups=(1, 2, 3, 4),
    #     transform=train_transform,
    # )

    # variant = config.dataset_type.split("/")[-1]
    # dataset = SpawriousCounterfactual.joint(root="./data",
    #     variant=variant,
    #     split="train",
    #     transform=train_transform,
    #     m2m_include_generic_bg=config.spawrious_m2m_include_generic
    # )

    setting = config.dataset_type.split("_")[-1]
    dataset = SpuriousVehiclesCounterfactual.groups(
        root="./data",
        split="train",
        transform=train_transform,
        concat=True,
        setting=setting,
    )

    set_rng(dataset, config.random_seed)
    print(len(dataset))

    dataloder = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    @torch.no_grad()
    def denorm(x, config):
        mean = torch.tensor(config.mean, dtype=torch.float32, device=x.device).view(
            1, 3, 1, 1
        )
        std = torch.tensor(config.std, dtype=torch.float32, device=x.device).view(
            1, 3, 1, 1
        )
        return x * std + mean

    for idx, (image, f_image, cf_image, label) in enumerate(dataloder):

        print(image.shape, image.dtype)
        print(f_image.shape, f_image.dtype)
        print(cf_image.shape, cf_image.dtype)
        print(label.shape, label.dtype)

        save_image(
            denorm(image, config), f"{dataset.__class__.__name__}_{idx}_image.png"
        )
        save_image(
            denorm(f_image, config), f"{dataset.__class__.__name__}_{idx}_f_image.png"
        )
        save_image(
            denorm(cf_image, config), f"{dataset.__class__.__name__}_{idx}_cf_image.png"
        )

        break
