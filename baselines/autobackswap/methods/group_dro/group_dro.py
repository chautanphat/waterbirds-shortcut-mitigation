import contextlib

import torch
from torch.nn.utils import clip_grad_norm_

from methods.common.model import Classifier
from methods.common.common_utils import load_data, training_loop, get_vit_scheduler
from methods.erm.erm import eval_func
from utils import *


def run_epoch(config, model, dataloader, optim, weight):

    model.train()
    losses = []
    loss_fun = torch.nn.CrossEntropyLoss(
        label_smoothing=config.target_smoothing, reduction="none"
    )

    if "waterbird" in config.dataset_type.lower():
        from methods.common.waterbird import get_group

        n_groups = 4 if config.waterbirds_use_minority else 2
    elif "spawrious" in config.dataset_type.lower():
        from methods.common.spawrious import get_group

        variant = config.dataset_type.split("/")[-1]
        if variant.startswith("m2m") and config.spawrious_m2m_include_generic:
            n_groups = (
                12  # n_groups is 12 for the m2m variants when adding generic background
            )
        else:
            n_groups = 8
    elif "spurious_vehicles" in config.dataset_type.lower():
        from methods.common.vehicles import get_group

        setting = config.dataset_type.split("_")[-1]
        if setting == "m2m":
            n_groups = 8
        elif setting == "o2o":
            n_groups = 8
        else:
            raise ValueError(f"invalid setting: '{setting}'")
    else:
        raise NotImplementedError(
            "Group DRO is only implemented for Waterbirds, Spawrious and Spurious-Vehicles datasets"
        )

    # load group weights and dict from run file
    _, run_path = get_run_name_and_path(config)
    group_weights_path, group_dict_path = (
        f"{run_path}/group_weights.pt",
        f"{run_path}/group_dict.pt",
    )
    if os.path.exists(group_weights_path):
        group_weights = torch.load(
            group_weights_path, map_location=config.device, weights_only=True
        )
    else:
        group_weights = torch.ones(n_groups, device=config.device) / n_groups
    if os.path.exists(group_dict_path):
        group_losses = torch.load(
            group_dict_path, map_location=config.device, weights_only=True
        )
    else:
        group_losses = dict()
        # populate group_losses dict with keys
        for images, labels, locations in dataloader:
            groups = get_group(labels, locations)
            for g in groups.unique():
                if g.item() not in group_losses.keys():
                    group_losses[g.item()] = None
            if len(group_losses.keys()) >= n_groups:
                break

    old_group_losses = group_losses.copy()

    for images, labels, locations in dataloader:

        # reset group losses each batch
        group_losses = {g: None for g in group_losses.keys()}

        # send data to device
        images = images.to(config.device)
        labels = labels.to(config.device)
        locations = locations.to(config.device)

        # forward pass & loss
        with (
            torch.autocast(device_type=config.device, dtype=torch.bfloat16)
            if config.cuda_optimizations
            else contextlib.nullcontext()
        ):

            logits = model(images)
            loss = loss_fun(logits, labels)

            groups = get_group(labels, locations)
            # compute avg group losses
            for g in group_losses.keys():
                group_mask = groups == g
                if group_mask.sum() > 0:
                    group_loss = loss[group_mask].mean()
                    group_losses[g] = group_loss

            # update group weights
            with torch.no_grad():
                for n, g in enumerate(old_group_losses.keys()):
                    if old_group_losses[g] is not None:
                        group_weights[n] *= torch.exp(
                            config.alpha * old_group_losses[g]
                        )
                group_weights /= group_weights.sum()
            # compute weighted loss
            present_weights = group_weights.clone()
            for n, g in enumerate(group_losses.keys()):
                if group_losses[g] is None:
                    present_weights[n] = 0.0
            present_weights /= present_weights.sum()
            loss = sum(
                [
                    present_weights[n] * group_losses[g]
                    for n, g in enumerate(group_losses.keys())
                    if group_losses[g] is not None
                ]
            )

            losses.append(loss.item())

        # optimization
        optim.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

    # save group weights to file
    torch.save(group_weights, group_weights_path)
    # need to store so indexing is consistent across epochs
    group_losses = {g: None for g in group_losses.keys()}
    torch.save(group_losses, group_dict_path)

    avg_loss = sum(losses) / len(losses)
    return avg_loss


def run(config):

    # # remove group_weights from previous runs - for debug mode
    # group_weights_path = f"{run_path}/group_weights.pt"
    # group_dict_path = f"{run_path}/group_dict.pt"
    # if os.path.exists(group_weights_path):
    #     os.remove(group_weights_path)
    # if os.path.exists(group_dict_path):
    #     os.remove(group_dict_path)

    if config.cuda_optimizations:
        torch.set_float32_matmul_precision(
            "high"
        )  # enables float32 tensor cores for matrix multiplication for better performance

    seed_everything(config.random_seed)
    data = load_data(config, yield_groups=True)

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
