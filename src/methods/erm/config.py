"""Explicit, validated ERM settings; no dependencies on external baselines."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class DatasetConfig:
    data_dir: str = "data/waterbirds"
    batch_size: int = 128
    batch_size_eval: int = 256
    workers: int = 4


@dataclass
class ModelConfig:
    weights: str | None = "IMAGENET1K_V2"
    freeze_bn: bool = False


@dataclass
class TrainingConfig:
    steps: int | None = None
    epochs: int | None = 100
    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0
    sampling: str = "class_balanced"
    eval_every: int = 300  # used only for step-based runs; epoch runs validate each epoch
    seed: int = 0
    device: str = "cuda"


@dataclass
class ERMConfig:
    profile: str = "dpe_standard_erm"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "checkpoints/waterbirds_erm"

    def validate(self):
        d, m, t = self.dataset, self.model, self.training
        for name, value in (("batch_size", d.batch_size), ("batch_size_eval", d.batch_size_eval),
                            ("eval_every", t.eval_every)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(d.workers) is not int or d.workers < 0:
            raise ValueError("workers must be a nonnegative integer")
        if (t.steps is None) == (t.epochs is None):
            raise ValueError("Set exactly one of training.steps or training.epochs")
        for name, value in (("steps", t.steps), ("epochs", t.epochs)):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or null")
        if not t.lr > 0 or not t.weight_decay >= 0 or not 0 <= t.momentum < 1:
            raise ValueError("Invalid SGD learning rate, weight decay or momentum")
        if t.sampling not in ("uniform", "class_balanced"):
            raise ValueError("sampling must be uniform or class_balanced")
        if m.weights not in (None, "IMAGENET1K_V1", "IMAGENET1K_V2"):
            raise ValueError("Unsupported ImageNet weights")
        if type(m.freeze_bn) is not bool:
            raise ValueError("freeze_bn must be a boolean")
        if type(t.seed) is not int or not 0 <= t.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not isinstance(t.device, str) or not t.device:
            raise ValueError("device must be a nonempty string")
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise ValueError("output_dir must be a nonempty string")
        return self

    def as_dict(self):
        return asdict(self)


def load_config(path):
    with Path(path).open() as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping")
    values = dict(raw)
    for name, cls in (("dataset", DatasetConfig), ("model", ModelConfig), ("training", TrainingConfig)):
        section = values.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f"{name} must be a mapping")
        unknown = set(section) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")
        values[name] = cls(**section)
    unknown = set(values) - {f.name for f in fields(ERMConfig)}
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    return ERMConfig(**values).validate()
