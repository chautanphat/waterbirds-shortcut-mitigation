import torch

from methods.common.model import Classifier
from methods.common.common_utils import (
    load_data,
    run_epoch,
    training_loop,
    get_vit_scheduler,
)
from utils import *


def eval_func(**kwargs):

    config = kwargs["config"]
    model = kwargs["model"]
    data = kwargs["data"]
    split_name = kwargs["split_name"]

    dataloaders, counts, group_info = data[split_name]
    accuracies = [
        run_epoch(config, model, dataloader, compute_acc=True)[1]
        for dataloader in dataloaders
    ]
    return accuracies, counts, group_info


def run_erm(config, data):

    if config.cuda_optimizations:
        torch.set_float32_matmul_precision(
            "high"
        )  # enables float32 tensor cores for matrix multiplication for better performance

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


def run(config):
    seed_everything(config.random_seed)
    data = load_data(config)
    run_erm(config, data)
