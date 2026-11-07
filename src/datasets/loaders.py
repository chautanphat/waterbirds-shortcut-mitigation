"""Sampling policies shared by backbone training and downstream methods."""

import torch
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler


def training_loader(dataset, batch_size, workers=0, sampling="uniform", *, steps=None):
    if sampling == "class_balanced":
        labels = torch.as_tensor(dataset.y)
        _, inverse, counts = labels.unique(return_inverse=True, return_counts=True)
        weights = counts.float().reciprocal()[inverse]
        sampler = WeightedRandomSampler(
            weights, num_samples=steps * batch_size if steps else len(dataset), replacement=True)
    elif sampling == "uniform":
        # SubpopBench samples uniformly WITH replacement, including across steps.
        sampler = RandomSampler(
            dataset, replacement=True, num_samples=steps * batch_size if steps else len(dataset))
    else:
        raise ValueError(f"Unknown sampling policy: {sampling}")
    if steps is None and len(dataset) < batch_size:
        raise ValueError("Training dataset is smaller than batch size with drop_last=True")
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      num_workers=workers, drop_last=True)


def evaluation_loader(dataset, batch_size=512, workers=0):
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                      shuffle=False, drop_last=False)
