import os
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image

CLASS_NAMES = ("landbird", "waterbird")


def get_group(label, place):
    return label * 2 + place + 1


class WaterBird(Dataset):

    def __init__(
        self, root, split, groups, transform, yield_groups=False, load_images=True
    ):

        self.root = root
        self.transform = transform
        self.yield_groups = yield_groups
        self.load_images = load_images
        self.folder_name = "waterbird_complete95_forest2water2"

        # load metadata
        metadata = pd.read_csv(os.path.join(root, self.folder_name, "metadata.csv"))

        # filter metadata according to splits
        split_id = {"train": 0, "valid": 1, "test": 2}
        desired_ids = [split_id[split]]
        metadata = metadata[metadata.split.isin(desired_ids)]

        # filter samples according to groups
        metadata = metadata.copy()
        metadata["group"] = metadata.y * 2 + metadata.place + 1
        metadata = metadata[metadata.group.isin(groups)]
        self.metadata = metadata
        self.n_data = len(metadata)

        # compute loss function weight
        if split == "train":
            num_neg = len(metadata[metadata.y == 0])
            num_pos = len(metadata[metadata.y == 1])
            pos_weight = num_neg / num_pos
            self.weight = [1.0, pos_weight]

    def __len__(self):
        return self.n_data

    def __getitem__(self, index):

        folder, file = self.metadata.img_filename.iloc[index].split("/")

        # Extract label
        label = int(self.metadata.y.iloc[index])

        # Load image
        image_path = os.path.join(self.root, self.folder_name, "images", folder, file)
        if self.load_images:
            image = self.transform(Image.open(image_path).convert("RGB"))
        else:
            image = image_path

        if self.yield_groups:
            place = int(self.metadata.place.iloc[index])
            return image, label, place

        return image, label
