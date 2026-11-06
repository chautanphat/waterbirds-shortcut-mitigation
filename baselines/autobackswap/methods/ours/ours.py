from torch.utils.data import DataLoader

from methods.ours.datasets import (
    WaterbirdsAugmented,
    SpawriousAugmented,
    SpuriousVehiclesAugmented,
)
from methods.common.common_utils import load_data
from methods.erm import erm
from utils import get_train_transform, seed_everything


def load_train_data(config):

    train_transform = get_train_transform(config)
    aug_config = {
        "infilling": config.infilling,
        "fg_transform_scale": config.fg_transform_scale,
        "fg_transform_rotation": config.fg_transform_rotation,
        "fg_transform_translation": config.fg_transform_translation,
    }

    if "waterbird" in config.dataset_type:
        groups = (1, 2, 3, 4) if config.waterbirds_use_minority else (1, 4)
        train_dataset = WaterbirdsAugmented(
            root="./data",
            split="train",
            groups=groups,
            transform=train_transform,
            seed=config.random_seed,
            aug_config=aug_config,
            use_gt_segmentations=config.use_gt_segmentations,
        )
        weight = train_dataset.weight
    elif "spawrious" in config.dataset_type:
        variant = config.dataset_type.split("/")[-1]
        train_dataset = SpawriousAugmented(
            root="./data",
            split="train",
            variant=variant,
            transform=train_transform,
            seed=config.random_seed,
            m2m_include_generic_bg=config.spawrious_m2m_include_generic,
            aug_config=aug_config,
            dataset_type=config.dataset_type.replace("/", "_"),
            use_gt_segmentations=config.use_gt_segmentations,
        )
        weight = None
    elif "spurious_vehicles" in config.dataset_type:
        setting = config.dataset_type.split("_")[-1]
        train_dataset = SpuriousVehiclesAugmented(
            root="./data",
            split="train",
            transform=train_transform,
            seed=config.random_seed,
            aug_config=aug_config,
            setting=setting,
            use_gt_segmentations=config.use_gt_segmentations,
        )
        weight = None
    else:
        raise ValueError(f"unknown dataset: '{config.dataset_type}'")

    # return dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    return train_dataloader, weight


def run(config):
    seed_everything(config.random_seed)
    data = load_data(config, skip_train=True)
    data["train"] = load_train_data(config)
    erm.run_erm(config, data)
