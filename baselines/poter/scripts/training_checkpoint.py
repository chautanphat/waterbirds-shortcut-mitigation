"""Portable epoch-boundary checkpoints for the Kaggle Waterbirds runner."""

import os
import random
from pathlib import Path

import numpy as np
import torch


def capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        for device, rng in enumerate(state["cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(rng.cpu(), device=device)


def save_checkpoint(path, model, optimizer, epoch, config, best_wga, loss_computer):
    path = Path(path)
    loss_state = {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in vars(loss_computer).items()
        if key not in {"criterion", "group_str"}
    }
    state = {
        "format_version": 1,
        "epoch": epoch,
        "config": config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_wga": best_wga,
        "loss_computer": loss_state,
        "rng": capture_rng(),
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, config):
    # Only load checkpoints created by this runner, from your own saved outputs.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 1 or state["config"] != config:
        raise ValueError("Checkpoint does not match the code, environment, dataset or settings")
    return state


def restore_training(state, model, optimizer, loss_computer):
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    device = next(model.parameters()).device
    for key, value in state["loss_computer"].items():
        setattr(loss_computer, key, value.to(device) if torch.is_tensor(value) else value)
    # Restore last, after model/loader/optimizer construction has consumed RNG.
    restore_rng(state["rng"])
