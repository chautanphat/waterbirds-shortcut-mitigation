import contextlib

import torch
from torch.nn.utils import clip_grad_norm_

from methods.common.model import Classifier
from methods.common.common_utils import load_data, training_loop, get_vit_scheduler
from methods.erm.erm import eval_func
from utils import *


def run_epoch(config, model, dataloader, optim, weight):

    if config.use_weight and weight is not None:
        weight = torch.tensor(weight, device=config.device)
    else:
        weight = None

    loss_fun = torch.nn.CrossEntropyLoss(
        label_smoothing=config.target_smoothing, weight=weight
    )

    model.train()
    losses = []

    for images, labels, locations in dataloader:

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
            embeddings = model.net(images)
            logits = model.head(embeddings)
            loss = loss_fun(logits, labels)

            # calculate covariance matrices per location for CORAL alignment
            covariances = []
            for loc in locations.unique():
                loc_mask = locations == loc
                if loc_mask.sum() > 1:  # need at least 2 samples to compute covariance
                    loc_embeddings = embeddings[loc_mask]
                    # center the features
                    loc_embeddings_centered = loc_embeddings - loc_embeddings.mean(
                        dim=0, keepdim=True
                    )
                    # compute covariance matrix: C = (1/n) * X^T * X
                    n = loc_embeddings.size(0)
                    loc_cov = (loc_embeddings_centered.T @ loc_embeddings_centered) / n
                    covariances.append(loc_cov)

            # calculate barycenter (mean) of covariance matrices
            if len(covariances) > 0:
                cov_barycenter = torch.stack(covariances, dim=0).mean(dim=0)

                # Deep CORAL loss: sum of squared Frobenius norms between each domain and barycenter
                coral_loss = 0.0
                d = embeddings.size(1)  # feature dimension
                for cov in covariances:
                    # Frobenius norm squared: ||C_i - C_bary||_F^2
                    coral_loss += torch.sum((cov - cov_barycenter) ** 2)

                # normalize: divide by number of domains and 4*d^2 (standard Deep CORAL normalization)
                coral_loss /= len(covariances) * 4 * d * d

                loss += config.coral_lambda * coral_loss

            losses.append(loss.item())

        # optimization
        optim.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

    avg_loss = sum(losses) / len(losses)

    return avg_loss


def run(config):

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
