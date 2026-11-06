import os
import random

from torch.utils.data import Dataset, ConcatDataset
from PIL import Image

SPUR_ATT_NAMES = ["urban", "highway", "rural", "off-road", "parked"]
CLASS_NAMES = ["sedan", "minivan", "SUV", "pickup truck"]
O2O_GENERIC_ATT = "parked"
O2O_GENERIC_FRAC = 0.25
VALID_RATIO = 0.2


def get_setting(setting):
    # Group IDs:
    # 0: unused
    # 1: train
    # 2: test
    # 3: train neutral
    # 4: test shifted
    if setting.lower() == "m2m":
        return [
            [1, 1, 2, 2, 0],
            [1, 1, 2, 2, 0],
            [2, 2, 1, 1, 0],
            [2, 2, 1, 1, 0],
        ]
    if setting.lower() == "o2o":
        return [
            [1, 2, 0, 0, 1],
            [0, 1, 2, 0, 1],
            [0, 0, 1, 2, 1],
            [2, 0, 0, 1, 1],
        ]
    raise ValueError(f"invalid setting: '{setting}'")


def get_group(label, spur_att):
    return label * len(SPUR_ATT_NAMES) + spur_att + 1


def get_group_id(split):
    if split in ["train", "valid"]:
        return 1
    elif split == "test":
        return 2
    elif split == "test_shifted":
        return 4
    else:
        raise ValueError(f"invalid split: {split}")


def split_train_valid(image_paths, split, valid_ratio):
    rng = random.Random(0)  # fixed seed, do not change
    rng.shuffle(image_paths)
    split_idx = int(len(image_paths) * valid_ratio)

    if split == "valid":
        selected = image_paths[:split_idx]
    elif split == "train":
        selected = image_paths[split_idx:]
    else:
        raise ValueError(f"invalid split: {split}")

    return selected


class SpuriousVehicles(Dataset):

    def __init__(
        self,
        root,
        split,
        transform,
        condition,
        vehicle,
        yield_groups=False,
        load_images=True,
        exclude_empty_segmentations=False,
        setting=None,
    ):

        self.transform = transform
        self.yield_groups = yield_groups
        self.load_images = load_images

        vehicle_index = CLASS_NAMES.index(vehicle)
        condition_index = SPUR_ATT_NAMES.index(condition)
        setting_matrix = get_setting(setting)

        # safety checks
        group_id = setting_matrix[vehicle_index][condition_index]
        group_id_should = get_group_id(split)
        assert (
            group_id == group_id_should
        ), "Mismatch between selected condition/vehicle and group split setting"
        # load image paths
        data_path = os.path.join(root, "spurious_vehicles", condition, vehicle)
        sort_key = lambda fname: int(fname.split(".")[0])
        image_paths = [
            os.path.join(data_path, img)
            for img in sorted(os.listdir(data_path), key=sort_key)
        ]

        # o2o generic
        if setting == "o2o" and condition == O2O_GENERIC_ATT:
            n_frac = int(len(image_paths) * O2O_GENERIC_FRAC)
            image_paths = image_paths[:n_frac]

        # split training and validation
        if split in ["train", "valid"]:
            image_paths = split_train_valid(image_paths, split, VALID_RATIO)

        # exclude empty segmentation images
        if exclude_empty_segmentations:
            raise NotImplementedError

        self.image_paths = image_paths
        self.label = vehicle_index
        self.spur_att = condition_index

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        if self.load_images:
            image = self.transform(Image.open(image_path).convert("RGB"))
        else:
            image = image_path

        if self.yield_groups:
            return image, self.label, self.spur_att

        return image, self.label

    @classmethod
    def groups(
        cls,
        root,
        split,
        transform,
        yield_groups=False,
        load_images=True,
        exclude_empty_segmentations=False,
        ret_group_info=False,
        concat=False,
        setting=None,
    ):

        group_id = get_group_id(split)
        setting_matrix = get_setting(setting)
        datasets = []
        group_info = []

        for class_index, class_name in enumerate(CLASS_NAMES):
            for spur_att_index, spu_att_name in enumerate(SPUR_ATT_NAMES):
                group_id_cur = setting_matrix[class_index][spur_att_index]
                if group_id_cur == group_id:
                    dataset = cls(
                        root,
                        split,
                        transform,
                        spu_att_name,
                        class_name,
                        yield_groups,
                        load_images,
                        exclude_empty_segmentations,
                        setting,
                    )
                    datasets.append(dataset)
                    group_info.append((class_index, spur_att_index))

        if concat:
            datasets = ConcatDataset(datasets)

        if ret_group_info:
            return group_info, datasets

        return datasets


if __name__ == "__main__":

    for setting in ["m2m", "o2o"]:
        print("=" * 10, setting, "=" * 10)
        for split in ["train", "valid", "test"]:
            print("-" * 10, split, "-" * 10)
            datasets = SpuriousVehicles.groups(
                "./data", split, None, False, False, False, False, False, setting
            )
            print(len(datasets))
            for dataset in datasets:
                print(
                    len(dataset),
                    CLASS_NAMES[dataset.label],
                    SPUR_ATT_NAMES[dataset.spur_att],
                )
