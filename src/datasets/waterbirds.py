"""Waterbirds metadata and deterministic SubpopBench preprocessing.

Preprocessing follows YyzHarry/SubpopBench (MIT, Yuzhe Yang 2023).
See src/methods/erm/README.md and LICENSE for attribution.
"""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


SPLITS = {"train": 0, "val": 1, "test": 2}


def waterbirds_transform():
    # Square resize, rather than aspect-ratio-preserving Resize(256).
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class Waterbirds(Dataset):
    """Return (split-local index, image, target, background).

    ``data_dir`` is the Waterbirds directory, not its parent. Supports both
    project-prepared SubpopBench metadata and the original Stanford archive.
    Rows retain CSV order, including for feature extraction consumed by DPE.
    """

    def __init__(self, data_dir, split, transform=None):
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; choose from {tuple(SPLITS)}")
        root = Path(data_dir)
        archive = root / "waterbird_complete95_forest2water2"
        candidates = [root / "metadata_waterbirds.csv", root / "metadata.csv",
                      archive / "metadata.csv"]
        self.metadata_path = next((p for p in candidates if p.is_file()), None)
        if self.metadata_path is None:
            raise FileNotFoundError(f"No Waterbirds metadata found in {root}")
        frame = pd.read_csv(self.metadata_path)
        for standard, original in (("filename", "img_filename"), ("a", "place")):
            if standard not in frame and original in frame:
                frame[standard] = frame[original]
        required = {"filename", "y", "a", "split"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Metadata missing columns: {sorted(required - set(frame.columns))}")
        if frame[list(required)].isna().any().any():
            raise ValueError("Waterbirds metadata contains missing values")
        for column, allowed in (("y", {0, 1}), ("a", {0, 1}), ("split", {0, 1, 2})):
            if not set(frame[column]).issubset(allowed):
                raise ValueError(f"Invalid Waterbirds {column} values")
        frame = frame.loc[frame.split == SPLITS[split]].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"Waterbirds split {split!r} is empty")
        self.paths = []
        for filename in frame.filename:
            options = [root / str(filename), archive / str(filename)]
            path = next((p for p in options if p.is_file()), None)
            if path is None:
                raise FileNotFoundError(f"Image {filename!r} missing under {root}")
            self.paths.append(path)
        self.filenames = frame.filename.astype(str).tolist()
        self.y = frame.y.to_numpy(dtype="int64")
        self.a = frame.a.to_numpy(dtype="int64")
        self.g = 2 * self.y + self.a
        self.split = split
        self.transform = transform if transform is not None else waterbirds_transform()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            inputs = self.transform(image.convert("RGB"))
        return index, inputs, int(self.y[index]), int(self.a[index])
