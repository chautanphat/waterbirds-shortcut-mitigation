import torch
from joblib import Parallel, delayed

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from methods.common.spawrious import Spawrious
from methods.common.waterbird import WaterBird
from methods.common.vehicles import SpuriousVehicles
from methods.common.model import Classifier
from methods.common.common_utils import (
    load_data,
    run_epoch,
    training_loop,
    get_vit_scheduler,
)
from utils import *

# DFR_Tr^Val implementation according to https://arxiv.org/pdf/2204.02937

C_VALUES = [1.0, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]


@torch.no_grad()
def extract_embeddings(config, model, dataloader):
    """Extracts feature vectors from dataset."""

    embeddings = []
    labels_ = []

    for images, labels in dataloader:
        embed = model.embeddings(images.to(config.device))
        embeddings.append(embed)
        labels_.append(labels)

    embeddings = torch.vstack(embeddings).cpu().numpy()
    labels_ = torch.cat(labels_).numpy()

    return embeddings, labels_


def balanced_groups_subsets(groups_dataset, seed):
    """
    Return a list of balanced random subsets, where each subset contains the same
    number of samples equal to the size of the smallest group.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    min_group_size = min(map(len, groups_dataset))
    balanced_groups = [
        Subset(
            dataset,
            indices=torch.randperm(len(dataset), generator=rng).tolist()[
                :min_group_size
            ],
        )
        for dataset in groups_dataset
    ]
    # print("DFR balanced group size: ", min_group_size)
    return balanced_groups


def load_reweight_data(config, seed=0):

    transform = get_test_transform(config)

    if "waterbird" in config.dataset_type:
        group_numbers = (1, 2, 3, 4) if config.waterbirds_use_minority else (1, 4)
        groups = [
            WaterBird(root="./data", split="valid", groups=(g,), transform=transform)
            for g in group_numbers
        ]  # individual groups
    elif "spawrious" in config.dataset_type:
        variant = config.dataset_type.split("/")[-1]
        groups = Spawrious.groups(
            variant=variant,
            split="valid",
            transform=transform,
            root="./data",
            m2m_include_generic_bg=config.spawrious_m2m_include_generic,
        )
    elif "spurious_vehicles" in config.dataset_type:
        setting = config.dataset_type.split("_")[-1]
        groups = SpuriousVehicles.groups(
            root="./data", split="valid", transform=transform, setting=setting
        )
    else:
        raise NotImplementedError(f"unknown dataset: '{config.dataset_type}'")

    balanced_groups = balanced_groups_subsets(groups, seed)
    return [
        DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,  # no shuffling as full-batch is used during reweighting
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )
        for dataset in balanced_groups
    ]


def tune_regularization_run(seed, reweight_embeddings, reweight_labels, logreg_params):
    """
    Tunes the logistic-regression regularization parameter 'C' by evaluating a
    set of candidate values on a random held-out portion (1/2) of the reweighting data.
    """

    n_data = len(reweight_embeddings[0])
    n_half = n_data // 2

    rng = torch.Generator()
    rng.manual_seed(seed)
    indices = torch.randperm(n_data, generator=rng)
    split1, split2 = indices[:n_half], indices[n_half:]
    x_train_groups, x_val_groups = zip(
        *[
            (embeddings[split1], embeddings[split2])
            for embeddings in reweight_embeddings
        ]
    )
    y_train_groups, y_val_groups = zip(
        *[(labels[split1], labels[split2]) for labels in reweight_labels]
    )
    x_train = np.concatenate(x_train_groups)
    y_train = np.concatenate(y_train_groups)

    contributions = np.zeros(len(C_VALUES), dtype=float)

    for i, c in enumerate(C_VALUES):
        logreg = LogisticRegression(
            penalty="l1", C=c, random_state=int(seed), **logreg_params
        )

        logreg.fit(x_train, y_train)
        y_pred_groups = [logreg.predict(x) for x in x_val_groups]
        group_accuracies = [
            (y_pred == y_val).mean()
            for y_pred, y_val in zip(y_pred_groups, y_val_groups)
        ]
        contributions[i] = float(min(group_accuracies))

    return contributions


def tune_regularization_parallel(
    reweight_embeddings, reweight_labels, n_runs, n_classes, n_jobs=8, verbose=True
):

    # total_num_runs = n_runs * len(C_VALUES)
    print(f"Tuning regularizing parameter ...")

    if n_classes > 2:
        logreg_params = dict(solver="saga", max_iter=1000, tol=1e-3)
    else:
        logreg_params = dict(solver="liblinear", max_iter=100, tol=1e-4)

    results = Parallel(n_jobs=n_jobs, verbose=verbose, backend="loky")(
        delayed(tune_regularization_run)(
            seed, reweight_embeddings, reweight_labels, logreg_params
        )
        for seed in range(n_runs)
    )

    summed = np.sum(np.stack(results, axis=0), axis=0)
    optimal_idx = int(np.argmax(summed))
    optimal_c = C_VALUES[optimal_idx]
    return optimal_c


def train_logreg(reweight_embeddings, reweight_labels, c, seed, n_classes):
    """Train a Logistic Regression model to be used as final prediction layer (model head)."""

    x_train = np.concatenate(reweight_embeddings)
    y_train = np.concatenate(reweight_labels)

    if n_classes > 2:
        logreg_params = dict(solver="saga", max_iter=1000, tol=1e-3)
    else:
        logreg_params = dict(solver="liblinear", max_iter=100, tol=1e-4)

    logreg = LogisticRegression(penalty="l1", C=c, random_state=seed, **logreg_params)

    logreg.fit(x_train, y_train)

    return logreg


def process_groups(config, model, data):
    if isinstance(data, list):
        embeddings, labels = zip(
            *[extract_embeddings(config, model, dataloader) for dataloader in data]
        )
    else:
        embeddings, labels = extract_embeddings(config, model, data)
    return embeddings, labels


def normalize(embeddings, mean=None, std=None, return_stats=False):
    # normalize embeddings according to reweighting data
    if mean is None or std is None:
        all_embeddings = np.concatenate(embeddings)
        mean = all_embeddings.mean(axis=0)
        std = all_embeddings.std(axis=0)
    embeddings_norm = [(embeddings - mean) / std for embeddings in embeddings]
    if return_stats:
        return embeddings_norm, (mean, std)
    return embeddings_norm


def eval_func(**kwargs):

    config = kwargs["config"]
    model = kwargs["model"]
    data = kwargs["data"]
    split_name = kwargs["split_name"]
    logreg = kwargs["logreg"]
    norm_stats = kwargs["norm_stats"]

    dataloaders, counts, group_info = data[split_name]
    embeddings, labels = process_groups(config, model, dataloaders)
    embeddings = normalize(embeddings, *norm_stats)
    preds = [logreg.predict(x) for x in embeddings]
    accuracies = [
        float((y_pred == y_test).mean()) for y_pred, y_test in zip(preds, labels)
    ]

    return accuracies, counts, group_info


def run(config):

    if config.cuda_optimizations:
        torch.set_float32_matmul_precision(
            "high"
        )  # enables float32 tensor cores for matrix multiplication for better performance

    # load training and testing data
    seed_everything(config.random_seed)
    data = load_data(config)
    train_data, _ = data["train"]

    # train erm base model
    model = Classifier(config)
    model = model.to(config.device)
    if config.cuda_optimizations:
        model = torch.compile(model)
    n_classes = model.out_dim

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

    # 2nd stage ------------------------------------

    # extract training embeddings and compute normalization statistics
    train_embeddings, _ = process_groups(config, model, train_data)
    _, norm_stats = normalize(train_embeddings, return_stats=True)

    # tune regularization parameter
    reweight_embeddings, reweight_labels = process_groups(
        config, model, load_reweight_data(config)
    )
    reweight_embeddings = normalize(reweight_embeddings, *norm_stats)
    c_optimal = tune_regularization_parallel(
        reweight_embeddings, reweight_labels, config.dfr_tune_runs, n_classes
    )

    # train heads
    coefs, intercepts = [], []
    logreg = None

    for seed in tqdm(range(config.dfr_train_runs), desc="STAGE 2", unit="run"):
        reweight_embeddings, reweight_labels = process_groups(
            config, model, load_reweight_data(config, seed)
        )  # current balanced subset from validation data
        reweight_embeddings = normalize(reweight_embeddings, *norm_stats)
        logreg = train_logreg(
            reweight_embeddings, reweight_labels, c_optimal, seed, n_classes
        )
        coefs.append(logreg.coef_)
        intercepts.append(logreg.intercept_)

    # overwrite logreg weights with average weights
    logreg.coef_ = np.mean(coefs, axis=0)
    logreg.intercept_ = np.mean(intercepts, axis=0)

    # ---------------------------------------------------

    # testing
    finalize_results(
        config=config,
        data=data,
        model=model,
        run_path=run_path,
        loss_logs=loss_logs,
        eval_func=eval_func,
        logreg=logreg,
        norm_stats=norm_stats,
    )
