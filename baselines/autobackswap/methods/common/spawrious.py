import os
import json
import random

from torch.utils.data import Dataset, ConcatDataset, Subset
from PIL import Image

CLASS_NAMES = ("bulldog", "corgi", "dachshund", "labrador")
LOCATION_NAMES = ("desert", "jungle", "dirt", "mountain", "snow", "beach")
VALID_RATIO = 0.2


def get_group(label, location):
    return label * len(LOCATION_NAMES) + location + 1


def load_metadata(path):
    with open(path) as f:
        return json.load(f)


def get_unused_locations_m2m(
    metadata, splits=(0, 1)
):  # splits=(0, 1) <=> (train, test)
    unused = list(LOCATION_NAMES)
    for cls in CLASS_NAMES:
        for split in splits:
            for env in [0, 1]:  # env0, env1
                loc = metadata[cls][split][env]
                if loc in unused:
                    unused.remove(loc)
    return unused


def get_unused_locations_o2o(metadata, splits=(0, 1)):
    unused = list(LOCATION_NAMES)
    for cls in CLASS_NAMES:
        for split in splits:
            loc = metadata[cls][split]
            if loc in unused:
                unused.remove(loc)
    unused.remove(metadata["generic"])
    return unused


def add_generic_location(
    root, partial_path, image_paths, cls, loc, loc_generic, mu, sort_key
):
    n_generic = len(image_paths) - int((len(image_paths) * mu))
    data_path_generic = os.path.join(root, partial_path, loc_generic, cls)
    image_paths_generic = [
        os.path.join(data_path_generic, img)
        for img in sorted(os.listdir(data_path_generic), key=sort_key)
    ]
    image_paths = image_paths[:-n_generic] + image_paths_generic[:n_generic]
    locations = [LOCATION_NAMES.index(loc)] * (len(image_paths) - n_generic) + [
        LOCATION_NAMES.index(loc_generic)
    ] * n_generic
    return image_paths, locations


def split_train_valid(image_paths, locations, split, valid_ratio):
    assert len(image_paths) == len(locations)

    pairs = list(zip(image_paths, locations))
    rng = random.Random(0)  # fixed seed, do not change
    rng.shuffle(pairs)
    split_idx = int(len(pairs) * valid_ratio)

    if split == "valid":
        selected = pairs[:split_idx]
    elif split == "train":
        selected = pairs[split_idx:]
    else:
        raise ValueError(f"invalid split: {split}")

    image_paths_out, locations_out = zip(*selected)
    return list(image_paths_out), list(locations_out)


class Spawrious(Dataset):

    def __init__(
        self,
        root,
        variant,
        env,
        cls,
        transform,
        split,
        yield_groups=False,
        load_images=True,
        m2m_include_generic_bg=False,
        exclude_empty_segmentations=False,
    ):

        assert cls in CLASS_NAMES, f"Invalid cls: {cls}. Must be one of {CLASS_NAMES}"
        assert env in {0, 1}, f"Invalid env: {env}. Must be one of {0, 1}"

        partial_path = f"spawrious224/m2m/{env}"  # o2o folders contain duplicate of m2m data, so only the m2m path is required
        type, difficulty = variant.split("_")
        metadata = load_metadata(
            os.path.join(root, f"spawrious224/{type}/{difficulty}.json")
        )
        train = not split.startswith("test")

        if variant.startswith("o2o"):
            if split == "test_shifted":
                unused_locations = get_unused_locations_o2o(
                    metadata, splits=(0,)
                )  # unused locations in the training set
                if len(unused_locations) > 0:
                    loc = unused_locations[
                        0
                    ]  # pick 1st unused location for shifted test dataset
                else:
                    raise ValueError(
                        f"Cannot create shifted test dataset for variant '{variant}': no unused locations available."
                    )
            else:
                loc = metadata[cls][0 if train else 1]
        elif variant.startswith("m2m"):
            if split == "test_shifted":
                unused_locations = get_unused_locations_m2m(
                    metadata
                )  # unused locations in all splits
                loc = unused_locations[
                    1
                ]  # pick 2nd unused location for shifted test dataset
            else:
                loc = metadata[cls][0 if train else 1][env]
        else:
            raise ValueError(f"invalid variant: '{variant}")

        data_path = os.path.join(root, partial_path, loc, cls)
        sort_key = lambda fname: int(fname.split(".")[0].split("_")[-1])
        image_paths = [
            os.path.join(data_path, img)
            for img in sorted(os.listdir(data_path), key=sort_key)
        ]

        # replace part of the training data with generic locations according to mu
        if train and variant.startswith("o2o"):
            mu = 0.97 if env == 0 else 0.87
            loc_generic = metadata["generic"]
            image_paths, locations = add_generic_location(
                root, partial_path, image_paths, cls, loc, loc_generic, mu, sort_key
            )
        elif train and variant.startswith("m2m") and m2m_include_generic_bg:
            mu = 0.92  # average of 0.97 and 0.87
            unused_locations = get_unused_locations_m2m(
                metadata
            )  # unused locations in all splits
            loc_generic = unused_locations[0]  # pick 1st unused location as generic one
            image_paths, locations = add_generic_location(
                root, partial_path, image_paths, cls, loc, loc_generic, mu, sort_key
            )
        else:
            locations = [LOCATION_NAMES.index(loc)] * len(image_paths)

        # split training and validation
        if train:
            image_paths, locations = split_train_valid(
                image_paths, locations, split, VALID_RATIO
            )

        # exclude empty segmentation images
        if exclude_empty_segmentations:
            raise NotImplementedError

        self.image_paths = image_paths
        self.locations = locations

        self.label = CLASS_NAMES.index(cls)
        self.transform = transform
        self.yield_groups = yield_groups
        self.load_images = load_images

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        if self.load_images:
            image = self.transform(Image.open(image_path).convert("RGB"))
        else:
            image = image_path

        if self.yield_groups:
            return image, self.label, self.locations[index]

        return image, self.label

    @classmethod
    def joint(
        _class,
        variant,
        split,
        transform,
        root,
        yield_groups=False,
        load_images=True,
        m2m_include_generic_bg=False,
        exclude_empty_segmentations=False,
    ):
        """Returns a dataset containing all classes across both environments."""
        dataset = ConcatDataset(
            [
                ConcatDataset(
                    [
                        _class(
                            root=root,
                            variant=variant,
                            env=env,
                            cls=cls,
                            transform=transform,
                            split=split,
                            yield_groups=yield_groups,
                            load_images=load_images,
                            m2m_include_generic_bg=m2m_include_generic_bg,
                            exclude_empty_segmentations=exclude_empty_segmentations,
                        )
                        for env in [0, 1]
                    ]
                )
                for cls in CLASS_NAMES
            ]
        )
        return dataset

    @classmethod
    def groups(
        _class,
        variant,
        split,
        transform,
        root,
        yield_groups=False,
        ret_group_info=False,
        m2m_include_generic_bg=False,
        exclude_empty_segmentations=False,
    ):
        """Return the dataset split into groups, where each group corresponds to a unique (class, location) pair."""

        # lightweight helper dataset to extract group information
        dataset_helper = _class.joint(
            variant=variant,
            split=split,
            transform=transform,
            root=root,
            yield_groups=True,
            load_images=False,
            m2m_include_generic_bg=m2m_include_generic_bg,
            exclude_empty_segmentations=exclude_empty_segmentations,
        )
        group_indices = dict()
        for idx, (_, cls, loc) in enumerate(dataset_helper):
            group = (cls, loc)
            if group in group_indices:
                group_indices[group].append(idx)
            else:
                group_indices[group] = [idx]

        # split dataset based on obtained group information
        dataset = _class.joint(
            variant=variant,
            split=split,
            transform=transform,
            root=root,
            yield_groups=yield_groups,
            m2m_include_generic_bg=m2m_include_generic_bg,
            exclude_empty_segmentations=exclude_empty_segmentations,
        )
        dataset_groups = {
            group: Subset(dataset, indices=indices)
            for group, indices in group_indices.items()
        }

        if ret_group_info:
            return dataset_groups.keys(), dataset_groups.values()

        return dataset_groups.values()
