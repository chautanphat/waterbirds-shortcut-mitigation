import os

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms as T

from torch.utils.data import Dataset
from PIL import Image

from methods.common.spawrious import load_metadata, CLASS_NAMES as SPAWRIOUS_CLASS_NAMES
from methods.common.vehicles import SpuriousVehicles

WATERBIRDS_SEGMENT_THR = 0.25
SPAWRIOUS_SEGMENT_THR = 0.2
VEHICLES_SEGMENT_THR = 0.4


def decode_mask(mask_str):
    size = int(len(mask_str) ** 0.5)
    mask = list(map(int, list(mask_str)))
    mask = torch.tensor(mask, dtype=torch.float32)
    mask = mask.reshape(size, size)
    return mask


def offset(image, dx, dy):
    assert image.ndim == 3, "image must be C x H x W"

    C, H, W = image.shape

    # make output tensor of zeros
    out = torch.zeros_like(image)

    # compute source & destination slice indices
    # x: horizontal axis (W), y: vertical axis (H)
    src_x0 = max(0, -dx)
    src_x1 = min(W, W - dx)
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    src_y0 = max(0, -dy)
    src_y1 = min(H, H - dy)
    dst_y0 = max(0, dy)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    # copy the chunk
    out[:, dst_y0:dst_y1, dst_x0:dst_x1] = image[:, src_y0:src_y1, src_x0:src_x1]

    return out


# --------- Augmentations ---------


def identity(image, mask):
    return image, mask


def flip(image, mask):
    return image.flip(dims=(2,)), mask.flip(dims=(1,))


def rot90_k1(image, mask):
    return (
        image.rot90(k=1, dims=(1, 2)),
        mask.rot90(k=1, dims=(0, 1)),
    )


def rot90_k3(image, mask):
    return (
        image.rot90(k=3, dims=(1, 2)),
        mask.rot90(k=3, dims=(0, 1)),
    )


# --------------------------------


def split_data(data, train_ratio, valid_ratio, test_ratio):

    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-8

    if isinstance(data, pd.DataFrame):
        data = data.sample(frac=1, random_state=0).reset_index(
            drop=True
        )  # randomize order
        n_total = len(data)
        data = data.iloc
    else:
        data = list(data)  # already randomized
        n_total = len(data)

    n_train = int(round(train_ratio * n_total))
    n_valid = int(round(valid_ratio * n_total))
    n_test = n_total - n_train - n_valid  # remainder

    # enforce zero-sized test split if requested
    if test_ratio == 0:
        n_test = 0
        n_valid = n_total - n_train  # absorb remainder into validation

    train = data[:n_train]
    valid = data[n_train : n_train + n_valid]
    test = data[n_train + n_valid : n_train + n_valid + n_test]

    return {
        "train": train,
        "valid": valid,
        "test": test,
    }


class DetectorDataset(Dataset):

    def __init__(self, config, rotation_augmentation, flip_augmentation, random_seed=0):
        self.config = config

        self.augmentations = [identity]
        if rotation_augmentation:
            self.augmentations += [rot90_k1, rot90_k3]
        if flip_augmentation:
            self.augmentations += [flip]

        self.rng = np.random.default_rng(random_seed)
        self.masks = self._load_masks()
        self.n_masks = len(self.masks)
        self.n_data = self.n_masks * len(self.augmentations)
        self.patch_size = config.image_size // config.detector_patch_resolution
        if config.image_size % config.detector_patch_resolution != 0:
            raise ValueError(
                f"image_size ({config.image_size}) must be divisible by "
                f"patch_resolution ({config.detector_patch_resolution})"
            )

    def __len__(self):
        return self.n_data

    def _load_masks(self):
        raise NotImplementedError

    def _get_sample(self, index):
        raise NotImplementedError

    def _rand_delta(self):
        max_shift = self.config.detector_max_mask_offset
        return int(self.rng.integers(-max_shift, max_shift + 1))

    def __getitem__(self, index):
        aug_index, index = divmod(index, self.n_masks)
        image, mask = self._get_sample(index)
        image, mask = self.augmentations[aug_index](image, mask)
        dx = self._rand_delta()
        dy = self._rand_delta()
        mask = offset(mask.unsqueeze(0), dx, dy).squeeze(0)
        image = offset(image, dx * self.patch_size, dy * self.patch_size)
        mask = 1 - mask
        return image, mask


class WaterbirdsHandLabeled(DetectorDataset):
    """Waterbirds detector dataset from hand-labeled data."""

    def __init__(self, config, root, transform, split):

        self.root = root
        self.transform = transform
        self.split = split
        self.folder_name = "waterbird_complete95_forest2water2"

        metadata = pd.read_csv(os.path.join(root, self.folder_name, "metadata.csv"))
        self.file_names = dict(zip(metadata.img_id, metadata.img_filename))

        super().__init__(config, rotation_augmentation=True, flip_augmentation=True)

    def _load_masks(self):
        masks_df = pd.read_csv(os.path.join(self.root, self.folder_name, "masks.csv"))
        return split_data(masks_df, *self.config.detector_data_split)[self.split]

    def _get_sample(self, index):
        img_id = self.masks["info"].iloc[index]
        file_path = os.path.join(
            self.root, self.folder_name, "images", self.file_names[img_id]
        )
        image = self.transform(Image.open(file_path).convert("RGB"))
        mask = decode_mask(str(self.masks["mask"].iloc[index]))
        return image, mask


class WaterbirdsFromSegmentations(DetectorDataset):
    """Waterbirds detector dataset derived from segmentation data."""

    def __init__(self, config, root, transform, split, n_samples):

        self.root = root
        self.transform = transform

        self.folder_name = "waterbird_complete95_forest2water2"
        metadata = pd.read_csv(os.path.join(root, self.folder_name, "metadata.csv"))
        metadata = metadata[
            (metadata["y"] == metadata["place"]) & (metadata["split"] == 0)
        ]  # filters samples from majority groups which are in the training split
        metadata = metadata.sample(n=n_samples, random_state=0).reset_index(
            drop=True
        )  # randomly sample a subset
        metadata = split_data(metadata, *config.detector_data_split)[split]
        self.rel_paths = metadata.img_filename.to_list()
        self.segment_transform = T.Compose(
            [T.Resize((config.image_size, config.image_size)), T.ToTensor()]
        )
        # get aux label type from config, default to "optimal" if not specified
        self.aux_label_type = getattr(config, "aux_label_type", "optimal")

        super().__init__(config, rotation_augmentation=True, flip_augmentation=True)

    def _load_masks(self):
        class DummyMasks:
            def __len__(_self):
                return len(self.rel_paths)

        return DummyMasks()

    def _get_sample(self, index):
        rel_path = self.rel_paths[index]

        # load image
        image_path = os.path.join(self.root, self.folder_name, "images", rel_path)
        image = self.transform(Image.open(image_path).convert("RGB"))

        # compute mask from segmentation
        segment_path = os.path.join(
            self.root,
            self.folder_name,
            "segmentations",
            rel_path.replace(".jpg", ".png"),
        )
        segment = self.segment_transform(Image.open(segment_path).convert("L"))
        segments_pool = F.avg_pool2d(segment, self.patch_size)
        optimal_mask = (segments_pool > WATERBIRDS_SEGMENT_THR).float().squeeze()

        if self.aux_label_type == "optimal":
            return image, optimal_mask

        if "undersample" in self.aux_label_type:
            biased_mask = (segments_pool >= 1.0).float().squeeze()
        elif "oversample" in self.aux_label_type:
            biased_mask = (segments_pool > 0.0).float().squeeze()

        if "optimal" in self.aux_label_type:
            # combine optimal and biased masks 50-50 elementwise
            rng = np.random.default_rng(
                seed=index
            )  # must be deterministic per-sample thus not using self.rng
            rng_mask = torch.tensor(
                rng.integers(0, 2, size=optimal_mask.shape), dtype=torch.float32
            )
            mask = rng_mask * optimal_mask + (1 - rng_mask) * biased_mask
        else:
            mask = biased_mask

        return image, mask


def spawrious_collect_rel_paths(root, variant, exclude_empty_segmentations=False):
    # Collect relative file paths for the specified variant across environments and classes
    type, difficulty = variant.split("_")
    metadata = load_metadata(
        os.path.join(root, f"spawrious224/{type}/{difficulty}.json")
    )
    sort_key = lambda fname: int(fname.split(".")[0].split("_")[-1])
    n_images = 3168
    rel_paths = []
    for env in ["0", "1"]:
        for cls in SPAWRIOUS_CLASS_NAMES:
            if variant.startswith("o2o"):
                loc = metadata[cls][0]
            elif variant.startswith("m2m"):
                loc = metadata[cls][0][int(env)]
            else:
                raise ValueError(f"invalid variant: '{variant}'")
            filenames = sorted(
                os.listdir(os.path.join(root, "spawrious224", "m2m", env, loc, cls)),
                key=sort_key,
            )
            # Add generic background fot o2o-variants
            if variant.startswith("o2o"):
                mu = 0.97 if env == "0" else 0.87
                n_generic = n_images - int((n_images * mu))
                filenames = filenames[: n_images - n_generic]
                loc_generic = metadata["generic"]
                filenames_generic = sorted(
                    os.listdir(
                        os.path.join(root, "spawrious224", "m2m", env, loc_generic, cls)
                    ),
                    key=sort_key,
                )[:n_generic]
                rel_paths += [
                    f"{env}/{loc_generic}/{cls}/{filename.replace('.png', '')}"
                    for filename in filenames_generic
                ]
            rel_paths += [
                f"{env}/{loc}/{cls}/{filename.replace('.png', '')}"
                for filename in filenames
            ]
    # exclude empty segmentation images
    if exclude_empty_segmentations:
        raise NotImplementedError
    return rel_paths


class SpawriousFromSegmentations(DetectorDataset):
    """Spawrious detector dataset derived from segmentation data."""

    def __init__(self, config, root, transform, split, variant, n_samples):

        self.root = root
        self.transform = transform
        self.segment_transform = T.Compose(
            [T.Resize((config.image_size, config.image_size)), T.ToTensor()]
        )

        # Randomly select a subset from the files and split data into train/valid/test
        rng = np.random.default_rng(0)
        rel_paths = spawrious_collect_rel_paths(
            root, variant, exclude_empty_segmentations=False
        )
        rel_paths = rng.choice(rel_paths, n_samples, replace=False)
        rel_paths = split_data(rel_paths, *config.detector_data_split)[split]
        self.rel_paths = rel_paths

        super().__init__(config, rotation_augmentation=False, flip_augmentation=True)

    def _load_masks(self):
        class DummyMasks:
            def __len__(_self):
                return len(self.rel_paths)

        return DummyMasks()

    def _get_sample(self, index):

        rel_path = self.rel_paths[index]

        # load image
        image_path = os.path.join(self.root, "spawrious224", "m2m", rel_path + ".png")
        image = self.transform(Image.open(image_path).convert("RGB"))

        # compute mask from segmentation
        segment_path = os.path.join(
            self.root, "spawrious224_segmentation_masks", "m2m", rel_path + ".tiff"
        )
        segment = self.segment_transform(
            Image.open(segment_path).convert("L")
        )  # segmentation with {0, 1} floats
        segments_pool = F.avg_pool2d(segment, self.patch_size)
        mask = (segments_pool > SPAWRIOUS_SEGMENT_THR).float().squeeze()

        # if mask is empty, select a different random sample
        if mask.sum() == 0:
            new_index = self.rng.integers(0, len(self.rel_paths))
            return self._get_sample(new_index)

        return image, mask


def vehicles_collect_paths(root, setting, exclude_empty_segmentations=False):
    dataset = SpuriousVehicles.groups(
        root=root,
        split="train",
        transform=None,
        yield_groups=False,
        load_images=False,
        exclude_empty_segmentations=exclude_empty_segmentations,
        concat=True,
        setting=setting,
    )
    paths = [path for path, _ in dataset]
    return paths


class SpuriousVehiclesFromSegmentations(DetectorDataset):
    """SpuriousVehicles detector dataset derived from segmentation data."""

    def __init__(self, config, root, transform, split, setting, n_samples):

        self.transform = transform
        self.segment_transform = T.Compose(
            [T.Resize((config.image_size, config.image_size)), T.ToTensor()]
        )
        # Collect image paths from the training split of the classifier Dataset
        paths = vehicles_collect_paths(root, setting, exclude_empty_segmentations=False)

        # Randomly select a subset from the files and split data into train/valid/test
        rng = np.random.default_rng(0)
        paths = rng.choice(paths, n_samples, replace=False)
        paths = split_data(paths, *config.detector_data_split)[split]
        self.paths = paths

        super().__init__(config, rotation_augmentation=False, flip_augmentation=True)

    def _load_masks(self):
        class DummyMasks:
            def __len__(_self):
                return len(self.paths)

        return DummyMasks()

    def _get_sample(self, index):

        # load image
        image_path = self.paths[index]
        image = self.transform(Image.open(image_path).convert("RGB"))

        # compute mask from segmentation
        segment_path = image_path.replace(
            "spurious_vehicles", "spurious_vehicles_segmentation_masks"
        )
        segment_path = segment_path[:-4] + ".tiff"
        segment = self.segment_transform(
            Image.open(segment_path).convert("L")
        )  # segmentation with {0, 1} floats
        segment_pool = F.avg_pool2d(segment, self.patch_size)
        mask = (segment_pool > VEHICLES_SEGMENT_THR).float().squeeze()

        # if mask is empty, select a different random sample
        if mask.sum() == 0:
            new_index = self.rng.integers(0, len(self.paths))
            return self._get_sample(new_index)

        return image, mask


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from utils import Config

    config = Config("methods/ours/config_spurious_vehicles.yaml")

    test_transform = T.Compose(
        [
            T.Resize((config.image_size, config.image_size)),
            T.ToTensor(),
        ]
    )

    setting = config.dataset_type.split("_")[-1]
    dataset = SpuriousVehiclesFromSegmentations(
        config,
        "data",
        transform=test_transform,
        split="train",
        setting=setting,
        n_samples=8,
    )
    # image: [3, 224, 224]
    # mask: [16, 16]

    n = len(dataset)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i in range(n):
        image, mask = dataset[i]

        # Upsample mask (repeat pixels)
        mask_up = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(), size=(224, 224), mode="nearest"
        ).squeeze()

        img_np = image.permute(1, 2, 0).cpu().numpy()
        mask_np = mask_up.cpu().numpy()

        ax = axes[i]
        ax.imshow(img_np)
        ax.imshow(mask_np, alpha=0.5, cmap="Reds")
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
