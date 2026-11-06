import gc
import contextlib
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import v2
from torch.optim import lr_scheduler


from methods.common.spawrious import Spawrious
from methods.common.waterbird import WaterBird
from methods.common.vehicles import SpuriousVehicles
from utils import get_test_transform, get_train_transform, create_output_dir


def group_counts(datasets):
    return [len(dataset) for dataset in datasets]


def load_data(config, yield_groups=False, skip_train=False):

    train_transform = get_train_transform(config)
    test_transform = get_test_transform(config)

    # --- Waterbirds dataset --------------------------------------------------

    if "waterbird" in config.dataset_type:
        groups = (1, 2, 3, 4) if config.waterbirds_use_minority else (1, 4)

        # training data
        if not skip_train:
            train_dataset = WaterBird(
                root="./data",
                split="train",
                groups=groups,
                transform=train_transform,
                yield_groups=yield_groups,
            )
            weight = train_dataset.weight

        # validation data (individual groups)
        valid_datasets = [
            WaterBird(
                root="./data", split="valid", groups=(g + 1,), transform=test_transform
            )
            for g in range(4)
            if g + 1 in groups
        ]

        # test data (individual groups)
        test_group_info, test_datasets = None, [
            WaterBird(
                root="./data", split="test", groups=(g + 1,), transform=test_transform
            )
            for g in range(4)
        ]

    # --- Spawrious dataset ------------------------------------------------------

    elif "spawrious" in config.dataset_type:
        variant = config.dataset_type.split("/")[-1]

        # training data
        if not skip_train:
            train_dataset = Spawrious.joint(
                variant,
                "train",
                train_transform,
                "./data",
                yield_groups,
                m2m_include_generic_bg=config.spawrious_m2m_include_generic,
            )
            weight = None

        # validation data(individual groups)
        valid_datasets = Spawrious.groups(
            variant,
            "valid",
            test_transform,
            "./data",
            yield_groups=False,
            m2m_include_generic_bg=config.spawrious_m2m_include_generic,
        )

        # test data (individual groups)
        test_group_info, test_datasets = Spawrious.groups(
            variant,
            "test",
            test_transform,
            "./data",
            yield_groups=False,
            ret_group_info=True,
        )
        test_shifted_group_info, test_shifted_datasets = Spawrious.groups(
            variant,
            "test_shifted",
            test_transform,
            "./data",
            yield_groups=False,
            ret_group_info=True,
        )

    # --- Spurious Vehicles dataset ------------------------------------------------------

    elif "spurious_vehicles" in config.dataset_type:
        setting = config.dataset_type.split("_")[-1]
        # training data
        if not skip_train:
            train_dataset = SpuriousVehicles.groups(
                "./data",
                "train",
                train_transform,
                yield_groups,
                concat=True,
                setting=setting,
            )
            weight = None

        # validation data(individual groups)
        valid_datasets = SpuriousVehicles.groups(
            "./data", "valid", test_transform, yield_groups=False, setting=setting
        )

        # test data (individual groups)
        test_group_info, test_datasets = SpuriousVehicles.groups(
            "./data",
            "test",
            test_transform,
            yield_groups=False,
            ret_group_info=True,
            setting=setting,
        )
        # test_shifted_group_info, test_shifted_datasets  = SpuriousVehicles.groups("./data", "test_shifted", test_transform, yield_groups=False, ret_group_info=True, setting=setting)

    # --- Dataloaders ---------------------------------------------------------------

    results = dict()
    # training dataloader
    if not skip_train:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )
        results["train"] = train_dataloader, weight

    # validation dataloaders
    valid_dataloaders = [
        DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )
        for dataset in valid_datasets
    ]
    valid_counts = group_counts(valid_datasets)
    results["valid"] = valid_dataloaders, valid_counts

    # test dataloaders
    test_dataloaders = [
        DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )
        for dataset in test_datasets
    ]
    test_counts = group_counts(test_datasets)
    results["test"] = test_dataloaders, test_counts, test_group_info

    # shifted test dataloaders
    if (
        "spawrious"
        in config.dataset_type
        # "spurious_vehicles" in config.dataset_type
    ):
        test_shifted_dataloaders = [
            DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                drop_last=False,
            )
            for dataset in test_shifted_datasets
        ]
        test_shifted_counts = group_counts(test_shifted_datasets)
        results["test_shifted"] = (
            test_shifted_dataloaders,
            test_shifted_counts,
            test_shifted_group_info,
        )

    # final data
    return results


def run_epoch(config, model, dataloader, optim=None, weight=None, compute_acc=False):

    if optim:
        is_training = True
        model.train()
    else:
        is_training = False
        model.eval()

    all_losses = []
    if compute_acc:
        correct = 0
        total = 0

    use_weight = getattr(config, "use_weight", False)
    heavy_aug = getattr(config, "heavy_augmentations", False)
    smoothing = getattr(config, "target_smoothing", 0.0)

    if use_weight and weight is not None:
        weight = torch.tensor(weight, device=config.device)
    else:
        weight = None

    loss_fun = torch.nn.CrossEntropyLoss(
        weight=weight, label_smoothing=smoothing, reduction="none"
    )

    if heavy_aug and is_training:
        cutmix_or_mixup_or_jitter = v2.RandomChoice(
            [
                v2.CutMix(num_classes=model.out_dim, alpha=config.cut_alpha),
                v2.MixUp(num_classes=model.out_dim, alpha=config.mix_alpha),
                v2.ColorJitter(brightness=0.5),
            ]
        )

    for images, labels in dataloader:

        with contextlib.nullcontext() if is_training else torch.no_grad():

            # cutmix & mixup
            labels_org = labels
            if heavy_aug and is_training:
                images, labels = cutmix_or_mixup_or_jitter(images, labels)

            # send data to device
            images = images.to(config.device)
            labels = labels.to(config.device)

            # forward pass & loss
            with (
                torch.autocast(device_type=config.device, dtype=torch.bfloat16)
                if config.cuda_optimizations
                else contextlib.nullcontext()
            ):
                logits = model(images)
                losses = loss_fun(logits, labels)
                loss = losses.mean(dim=0)
                all_losses += losses.detach().cpu().tolist()

            # optimization
            if is_training:
                optim.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()

            # accuracy
            if compute_acc:
                with torch.no_grad():
                    preds = logits.argmax(dim=1).cpu()
                    correct += (preds == labels_org).sum().item()
                    total += labels_org.shape[0]

    if compute_acc:
        accuracy = correct / total
        return all_losses, accuracy

    return all_losses


def training_loop(
    config,
    model,
    optim,
    data,
    num_epochs,
    run_epoch_func,
    info="",
    scheduler=None,
):

    # training loss logging
    train_losses = []

    # validation loss logging
    val_losses = []
    best_val_loss = float("inf")
    val_loss = None

    # validation accuracy logging
    val_accuracies = []
    best_val_accuracy = 0
    val_accuracy = None

    pbar = tqdm(range(num_epochs), unit="epoch", desc=info)

    _, run_path = create_output_dir(config)
    checkpoint_path = f"{run_path}/checkpoint.pt"
    final_path = f"{run_path}/final.pt"

    for _ in pbar:

        # Training
        train_dataloader, weight = data["train"]
        train_loss = run_epoch_func(config, model, train_dataloader, optim, weight)
        if isinstance(train_loss, list):
            # compute average training loss, if reduction was set to none.
            train_loss = sum(train_loss) / len(train_loss)
        train_losses.append(train_loss)

        # Validation & early stopping
        if config.early_stopping == "none":
            pbar.set_description(f"{info}train_loss: {train_loss:.4f}")
        else:
            valid_dataloaders, valid_counts = data["valid"]
            group_val_losses, group_val_accuracies = zip(
                *[
                    run_epoch(config, model, dataloader, compute_acc=True)
                    for dataloader in valid_dataloaders
                ]
            )

            # Compute average validation loss
            losses = []
            for group_losses in group_val_losses:
                losses += group_losses
            val_loss = sum(losses) / len(losses)
            val_losses.append(val_loss)

            # Accuracy-based early stopping
            if "acc" in config.early_stopping:
                # Convert results to numpy arrays
                # Compute desired validation accuracy metric
                if config.early_stopping == "val_acc":
                    group_val_accuracies = np.array(group_val_accuracies)
                    valid_counts = np.array(valid_counts)
                    val_accuracy = float(
                        (group_val_accuracies * valid_counts).sum() / valid_counts.sum()
                    )  # overall validation accuracy
                elif config.early_stopping == "worst_group_val_acc":
                    val_accuracy = min(
                        group_val_accuracies
                    )  # worst-group validation accuracy
                else:
                    raise ValueError(
                        f"invalid early stopping setting: '{config.early_stopping}'"
                    )
                val_accuracies.append(val_accuracy)
                # Create early stopping checkpoint
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    torch.save(model.state_dict(), checkpoint_path)
                    pbar.set_description(
                        f"{info}best {config.early_stopping}: {best_val_accuracy*100:.2f} %"
                    )
            # Loss-based early stopping
            elif config.early_stopping == "val_loss":
                # Create early stopping checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), checkpoint_path)
                    pbar.set_description(
                        f"{info}best {config.early_stopping}: {best_val_loss:.4f}"
                    )
            else:
                raise ValueError(
                    f"invalid early stopping setting: '{config.early_stopping}'"
                )

        # lr scheduler
        if isinstance(scheduler, lr_scheduler.CosineAnnealingLR) or isinstance(
            scheduler, lr_scheduler.SequentialLR
        ):
            scheduler.step()
        elif isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

    # create final checkpoint
    torch.save(model.state_dict(), final_path)

    if config.early_stopping == "none":
        model.load_state_dict(
            torch.load(
                final_path, map_location=torch.device(config.device), weights_only=True
            )
        )
    else:
        model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=torch.device(config.device),
                weights_only=True,
            )
        )

    torch.cuda.empty_cache()
    gc.collect()
    model.eval()

    loss_logs = train_losses, val_losses, val_accuracies

    return run_path, loss_logs


def load_aotgan(submodule_path):
    """Loads AOT-GAN model from git submodule 'AOT-GAN-for-Inpainting'."""
    import importlib.util
    import pathlib
    import sys

    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        # avoid argparse interference
        old_argv = sys.argv
        try:
            sys.argv = [old_argv[0]]
            spec.loader.exec_module(module)
        finally:
            sys.argv = old_argv
        return module

    submodule_path = pathlib.Path(submodule_path)

    option = load_module(
        "aot_utils_option", pathlib.Path(submodule_path) / "src" / "utils" / "option.py"
    )

    common = load_module(
        "model.common", pathlib.Path(submodule_path) / "src" / "model" / "common.py"
    )

    aotgan = load_module(
        "model.aotgan", pathlib.Path(submodule_path) / "src" / "model" / "aotgan.py"
    )

    return aotgan.InpaintGenerator(option.args)


def get_vit_scheduler(optim, config):
    warmup_epochs = 5

    warmup = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=0.1, total_iters=warmup_epochs
    )

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=config.num_epochs - warmup_epochs,
        eta_min=config.learning_rate / 100.0,
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optim, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )
