# CF(GAN) + F(Random) implementation according to https://arxiv.org/pdf/2106.01127
# AOT-GAN (https://arxiv.org/pdf/2104.01431) model used instead of CAGAN (https://arxiv.org/pdf/1801.07892)

import contextlib

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

# from torch.optim.lr_scheduler import ReduceLROnPlateau

from methods.common.model import Classifier
from methods.chang.datasets import (
    WaterbirdsCounterfactual,
    SpawriousCounterfactual,
    SpuriousVehiclesCounterfactual,
    set_rng,
)
from methods.common.common_utils import load_data, training_loop, get_vit_scheduler
from methods.common.common_utils import run_epoch as common_run_epoch
from utils import *


def eval_func(**kwargs):

    config = kwargs["config"]
    model = kwargs["model"]
    data = kwargs["data"]
    split_name = kwargs["split_name"]

    dataloaders, counts, group_info = data[split_name]
    accuracies = [
        common_run_epoch(config, model, dataloader, compute_acc=True)[1]
        for dataloader in dataloaders
    ]
    return accuracies, counts, group_info


def load_train_data(config):

    train_transform = get_train_transform(config, to_tensor=False)

    if "waterbird" in config.dataset_type:
        groups = (1, 2, 3, 4) if config.waterbirds_use_minority else (1, 4)
        train_dataset = WaterbirdsCounterfactual(
            root="./data",
            split="train",
            transform=train_transform,
            groups=groups,
        )
        weight = train_dataset.weight
    elif "spawrious" in config.dataset_type:
        variant = config.dataset_type.split("/")[-1]
        train_dataset = SpawriousCounterfactual.joint(
            root="./data",
            variant=variant,
            split="train",
            transform=train_transform,
            m2m_include_generic_bg=config.spawrious_m2m_include_generic,
            exclude_empty_segmentations=False,
        )
        weight = None
    elif "spurious_vehicles" in config.dataset_type:
        setting = config.dataset_type.split("_")[-1]
        train_dataset = SpuriousVehiclesCounterfactual.groups(
            root="./data",
            split="train",
            transform=train_transform,
            concat=True,
            setting=setting,
        )
        weight = None
    else:
        raise ValueError(f"unknown dataset: '{config.dataset_type}'")

    set_rng(train_dataset, config.random_seed)

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


def counterfactual_loss(logits, neg_labels):
    """Counterfactual loss function according to original paper."""
    # neg_labels: indices of the "forbidden class" per batch
    # binary-class special case
    if logits.shape[1] == 2:
        return F.cross_entropy(logits, 1 - neg_labels, reduction="none")
    # multi-class
    log_probs = F.log_softmax(logits, dim=1)
    log_p = log_probs[torch.arange(logits.shape[0]), neg_labels]
    return -torch.log1p(
        -torch.exp(log_p)
    )  # for numerical stability: log(1 - p_c) = log(1 - exp(log_p)) = log1p(-exp(..))


def run_epoch(config, model, dataloader, optim, weight):

    if config.use_weight and weight is not None:
        weight = torch.tensor(weight, device=config.device)
    else:
        weight = None

    model.train()
    all_losses = []

    for images, f_images, cf_images, labels in dataloader:

        with (
            torch.autocast(device_type=config.device, dtype=torch.bfloat16)
            if config.cuda_optimizations
            else contextlib.nullcontext()
        ):

            batch_size = images.shape[0]
            labels = labels.to(config.device)

            # forward pass
            images_trip = torch.cat([images, f_images, cf_images], dim=0).to(
                config.device
            )
            logits_trip = model(images_trip)
            logits, f_logits, cf_logits = torch.split(logits_trip, batch_size, dim=0)

            ce_losses = F.cross_entropy(
                logits,
                labels,
                reduction="none",
                weight=weight,
                label_smoothing=config.target_smoothing,
            )
            f_losses = F.cross_entropy(f_logits, labels, reduction="none")
            cf_losses = counterfactual_loss(cf_logits, labels)
            losses = ce_losses + f_losses + cf_losses
            loss = losses.mean()

            all_losses += losses.detach().cpu().tolist()

        # optimization
        optim.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

    return all_losses


def run(config):

    if config.cuda_optimizations:
        torch.set_float32_matmul_precision(
            "high"
        )  # enables float32 tensor cores for matrix multiplication for better performance

    seed_everything(config.random_seed)
    data = load_data(config, skip_train=True)
    data["train"] = load_train_data(config)

    model = Classifier(config)
    model = model.to(config.device)
    if config.cuda_optimizations:
        model = torch.compile(model)

    if config.model_class == "resnet":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
        )
        scheduler = None
    else:  # vit
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate / 100,
            weight_decay=config.weight_decay * 10,
        )
        scheduler = get_vit_scheduler(optim, config)

    # training
    run_path, loss_logs = training_loop(
        config=config,
        model=model,
        optim=optim,
        data=data,
        num_epochs=config.num_epochs,
        run_epoch_func=run_epoch,
        scheduler=scheduler,
    )

    # testing
    finalize_results(
        config=config,
        data=data,
        model=model,
        run_path=run_path,
        loss_logs=loss_logs,
        eval_func=eval_func,
    )
