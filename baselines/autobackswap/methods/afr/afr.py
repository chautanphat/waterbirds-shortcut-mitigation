import copy

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from methods.common.model import Classifier
from methods.common.waterbird import WaterBird
from methods.common.spawrious import Spawrious
from methods.common.vehicles import SpuriousVehicles
from methods.common.common_utils import (
    run_epoch,
    load_data,
    training_loop,
    get_vit_scheduler,
)
from methods.erm.erm import eval_func
from utils import *


def load_train_reweight_data(config):

    train_transform = get_train_transform(config)
    test_transform = get_test_transform(config)

    if "waterbird" in config.dataset_type:
        groups = (1, 2, 3, 4) if config.waterbirds_use_minority else (1, 4)
        # training data
        full_train_dataset = WaterBird(
            root="./data", split="train", groups=groups, transform=train_transform
        )
        train_dataset = fixed_split(
            full_train_dataset, config.afr_split, config.random_seed
        )[0]
        weight = full_train_dataset.weight

        # reweighting data
        full_train_dataset_test_transform = WaterBird(
            root="./data", split="train", groups=groups, transform=test_transform
        )
        reweight_dataset = fixed_split(
            full_train_dataset_test_transform, config.afr_split, config.random_seed
        )[1]

    elif "spawrious" in config.dataset_type:
        variant = config.dataset_type.split("/")[-1]
        # training data
        full_train_dataset = Spawrious.joint(
            variant,
            "train",
            train_transform,
            "./data",
            m2m_include_generic_bg=config.spawrious_m2m_include_generic,
        )
        train_dataset = fixed_split(
            full_train_dataset, config.afr_split, config.random_seed
        )[0]
        weight = None

        # reweighting data
        full_train_dataset_test_transform = Spawrious.joint(
            variant,
            "train",
            test_transform,
            "./data",
            m2m_include_generic_bg=config.spawrious_m2m_include_generic,
        )
        reweight_dataset = fixed_split(
            full_train_dataset_test_transform, config.afr_split, config.random_seed
        )[1]

    elif "spurious_vehicles" in config.dataset_type:
        setting = config.dataset_type.split("_")[-1]

        # training data
        full_train_dataset = SpuriousVehicles.groups(
            "./data", "train", train_transform, concat=True, setting=setting
        )
        train_dataset = fixed_split(
            full_train_dataset, config.afr_split, config.random_seed
        )[0]
        weight = None

        # reweighting data
        full_train_dataset_test_transform = SpuriousVehicles.groups(
            "./data", "train", test_transform, concat=True, setting=setting
        )
        reweight_dataset = fixed_split(
            full_train_dataset_test_transform, config.afr_split, config.random_seed
        )[1]

    else:
        raise NotImplementedError(f"unknown dataset: '{config.dataset_type}'")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    reweight_dataloader = DataLoader(
        reweight_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # no shuffling as full-batch is used during reweighting
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    return {
        "train": (train_dataloader, weight),
        "reweight": reweight_dataloader,
    }


@torch.no_grad()
def get_embeddings_and_weights(config, model, dataloader, compute_weights=True):

    model.to(config.device)
    model.eval()
    num_classes = model.out_dim

    if compute_weights:
        # compute betas
        counts = torch.zeros(num_classes)
        for _, batch_labels in dataloader:
            counts += torch.bincount(batch_labels, minlength=num_classes)
        # division by zero check
        if torch.any(counts == 0):
            raise ValueError(f"detected a class with zero samples")
        betas = 1 / counts

        # prepare gamma values
        gamma_values = np.linspace(
            *config.afr_gamma_linspace, dtype=np.float32
        ).tolist()
        weights_dict = {g: [] for g in gamma_values}

    # compute weights
    embeddings = []
    labels = []

    for batch_images, batch_labels in dataloader:
        # extract and store embeddings
        batch_embeddings = model.embeddings(batch_images.to(config.device))
        embeddings.append(batch_embeddings.cpu())
        labels.append(batch_labels)

        # compute (unnormalized) weights from logits for each gamma value
        if compute_weights:
            logits = model.head(batch_embeddings).cpu()
            probs = torch.nn.functional.softmax(logits, dim=1)
            probs_correct = probs[torch.arange(probs.size(0)), batch_labels]
            for gamma, weights_list in weights_dict.items():
                weights = betas[batch_labels] * torch.exp(-gamma * probs_correct)
                weights_list.append(weights)

    embeddings = torch.vstack(embeddings)
    labels = torch.cat(labels)

    # weight normalization
    if compute_weights:
        for gamma, weights_list in weights_dict.items():
            weights_tensor = torch.cat(weights_list)
            weights_tensor /= weights_tensor.sum(dim=0)
            weights_dict[gamma] = weights_tensor

        return embeddings, labels, weights_dict

    return embeddings, labels


@torch.no_grad()
def compute_group_accuracies(config, head, embeddings):
    group_embeddings, group_labels = embeddings["valid_groups"]
    head.to(config.device)
    group_predictions = [
        head(embed.to(config.device)).cpu().argmax(dim=1) for embed in group_embeddings
    ]
    accuracies = [
        float((preds == labels).sum() / labels.shape[0])
        for preds, labels in zip(group_predictions, group_labels)
    ]
    return accuracies


def reweight_model_head(config, erm_head, embeddings, weights_dict):
    """Full-batch head re-training."""

    # store original erm head parameters
    extract_params = lambda layer: torch.cat((layer.weight.flatten(), layer.bias))
    erm_params = extract_params(erm_head).detach().to(config.device)

    # data
    reweight_embeddings, reweight_labels = embeddings["reweight"]
    reweight_embeddings = reweight_embeddings.to(config.device)
    reweight_labels = reweight_labels.to(config.device)

    # loss function
    loss_fun = torch.nn.CrossEntropyLoss(reduction="none")

    # progressbar
    lambda_values = np.linspace(*config.afr_lambda_linspace, dtype=np.float32).tolist()
    total_iterations = (
        len(weights_dict)
        * len(lambda_values)
        * len(config.afr_learning_rates)
        * config.afr_num_epochs
    )
    pbar = tqdm(total=total_iterations)
    loss_dict = dict()

    # print(f"lambda values: {lambda_values}")
    # print(f"weight values: {list(weights_dict.keys())}")

    # tune hyperparameters
    best_wga = 0.0  # best worst-group-accuracy
    best_head = None

    erm_head.eval()
    best_params = {"gamma": None, "lambda": None, "lr": None}
    # wga = min(compute_group_accuracies(config, erm_head, embeddings))
    # print(f"erm wga: {wga*100:.2f} %")

    for gamma, weights in weights_dict.items():
        weights = weights.detach().to(config.device)

        for lambda_ in lambda_values:

            for lr in config.afr_learning_rates:

                # initialize head to erm weights and setup optimizer
                head = copy.deepcopy(erm_head).to(config.device)
                optim = torch.optim.SGD(
                    head.parameters(),
                    lr=lr,
                    weight_decay=config.afr_weight_decay,
                    momentum=config.afr_momentum,
                )
                losses = []

                for _ in range(config.afr_num_epochs):

                    # forward pass
                    head.train()
                    params = extract_params(head)
                    reweight_logits = head(reweight_embeddings)

                    # loss
                    ce_loss = torch.sum(
                        weights * loss_fun(reweight_logits, reweight_labels)
                    )
                    l2_penalty = lambda_ * torch.sum((params - erm_params) ** 2)
                    loss = ce_loss + l2_penalty

                    # logging
                    losses.append(loss.item())

                    # optimization
                    optim.zero_grad()
                    loss.backward()
                    clip_grad_norm_(head.parameters(), max_norm=config.afr_max_norm)
                    optim.step()

                    # validation & early stopping
                    head.eval()
                    wga = min(
                        compute_group_accuracies(config, head, embeddings)
                    )  # worst group accuracy

                    if wga > best_wga:
                        best_wga = wga
                        best_head = copy.deepcopy(head)  # checkpoint
                        # store best params
                        best_params["gamma"] = gamma
                        best_params["lambda"] = lambda_
                        best_params["lr"] = lr

                    pbar.set_description(
                        f"STAGE-2 | Validation-WGA: {best_wga*100:.2f} %"
                    )
                    pbar.update(1)

                loss_dict[f"γ={gamma:.1f}|λ={lambda_:.1f}"] = [
                    (losses, [])
                ]  # [(training, validation)] 1x run

    pbar.close()

    return best_head, loss_dict, best_params, best_wga


def run(config):

    if config.cuda_optimizations:
        torch.set_float32_matmul_precision(
            "high"
        )  # enables float32 tensor cores for matrix multiplication for better performance

    # load data
    seed_everything(config.random_seed)
    data = load_data(config, skip_train=True) | load_train_reweight_data(
        config
    )  # merge data dictionaries
    reweight_data = data["reweight"]
    valid_dataloaders, _ = data["valid"]

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
        scheduler = CosineAnnealingLR(
            optim, T_max=config.num_epochs, eta_min=config.learning_rate / 100.0
        )
    else:  # vit
        optim = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate / 100,
            weight_decay=config.weight_decay * 10,
        )
        scheduler = get_vit_scheduler(optim, config)

    # 1st stage
    run_path, loss_logs = training_loop(
        config=config,
        model=model,
        optim=optim,
        data=data,
        num_epochs=config.num_epochs,
        run_epoch_func=run_epoch,
        info="STAGE-1 | ",
        scheduler=scheduler,
    )

    # 2nd stage
    reweight_embeddings, reweight_labels, weights_dict = get_embeddings_and_weights(
        config, model, reweight_data
    )
    valid_group_embeddings, valid_group_labels = zip(
        *[
            get_embeddings_and_weights(config, model, dataloader, compute_weights=False)
            for dataloader in valid_dataloaders
        ]
    )
    embeddings = {
        # embeddings from stage 1
        "reweight": (reweight_embeddings, reweight_labels),
        "valid_groups": (valid_group_embeddings, valid_group_labels),
    }
    best_head, loss_dict, params, wga = reweight_model_head(
        config, model.head, embeddings, weights_dict
    )
    model.head = best_head  # overwrite head with retrained one

    # testing
    finalize_results(
        config=config,
        data=data,
        model=model,
        run_path=run_path,
        loss_logs=loss_logs,
        eval_func=eval_func,
    )

    # write tuning results
    with open(f"{run_path}/tuning.txt", "w") as f:
        f.write("2nd Stage parameters:\n")
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"Validation WGA: {wga*100:02.3f} %\n")
