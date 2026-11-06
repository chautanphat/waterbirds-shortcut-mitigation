import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as T
from torch.utils.data import random_split
from methods.common.spawrious import (
    CLASS_NAMES as SPAWRIOUS_CLASS_NAMES,
    LOCATION_NAMES as SPAWRIOUS_LOCATION_NAMES,
)
from methods.common.vehicles import (
    CLASS_NAMES as VEHICLES_CLASS_NAMES,
    SPUR_ATT_NAMES as VEHICLES_SPUR_ATT_NAMES,
)

TRANSFORMERS_CACHE_DIR = None  # set custom directory or use None for default directory


class Config:
    def __init__(self, path):
        with open(path) as f:
            d = yaml.safe_load(f)
        for key, value in d.items():
            self.__dict__[key] = value


def number_params(model):
    return sum(p.numel() for p in model.parameters())


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fixed_split(dataset, split, seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return random_split(dataset, split, generator=generator)


def get_test_transform(config, to_tensor=True):
    scale = 256 / 224
    return T.Compose(
        [
            T.Resize((int(config.image_size * scale), int(config.image_size * scale))),
            T.CenterCrop(config.image_size),
            *([T.ToTensor()] if to_tensor else []),
            T.Normalize(mean=config.mean, std=config.std),
        ]
    )


def get_train_transform(config, to_tensor=True):
    rand_augment = (
        config.heavy_augmentations if hasattr(config, "heavy_augmentations") else False
    )
    tranform = T.Compose(
        [
            T.RandomResizedCrop(
                config.image_size,
                scale=(0.7, 1.0),
                ratio=(0.75, 4 / 3),
                interpolation=2,
            ),
            T.RandomHorizontalFlip(),
            *([T.RandAugment()] if rand_augment else []),
            *([T.ToTensor()] if to_tensor else []),
            T.Normalize(mean=config.mean, std=config.std),
        ]
    )
    return tranform


def plot_results(
    loss_logs, fname="training.png", show_labels=True, early_stopping=None
):
    """
    Plot training/validation loss and validation accuracy stacked vertically.

    loss_logs: dict[label] -> list of runs
               each run is (train_losses, val_losses, val_accuracies)
               train_losses, val_losses, val_accuracies are lists (may be empty or None)
    """

    # create figure with two rows
    fig, (ax_train_loss, ax_val_loss, ax_acc) = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
        figsize=(12, 12),
    )

    for label, runs in loss_logs.items():
        for i, (train_losses, val_losses, val_accuracies) in enumerate(runs):

            # training loss
            if train_losses:
                label_train = f"{label} train_loss" if i <= 0 and show_labels else None
                epochs_train = range(1, len(train_losses) + 1)
                ax_train_loss.plot(
                    epochs_train,
                    train_losses,
                    color="#1f77b4",
                    label=label_train,
                    zorder=1,
                )
                ax_train_loss.set_xticks(range(0, len(train_losses) + 1, 10))

            # validation loss
            if val_losses:
                label_val_loss = f"{label} val_loss" if i <= 0 and show_labels else None
                epochs_val_loss = range(1, len(val_losses) + 1)
                ax_val_loss.plot(
                    epochs_val_loss,
                    val_losses,
                    color="#ff7f0e",
                    label=label_val_loss,
                    zorder=1,
                )
                ax_val_loss.set_xticks(range(0, len(val_losses) + 1, 10))
                if early_stopping == "val_loss":
                    # mark best loss for this run
                    min_val_loss = min(val_losses)
                    min_val_epoch = val_losses.index(min_val_loss) + 1
                    ax_val_loss.scatter(
                        [min_val_epoch],
                        [min_val_loss],
                        s=48,
                        c="#ff7f0e",
                        marker="o",
                        zorder=2,
                        label="checkpoint",
                    )
                    fig.suptitle(
                        f"Training Progress | best val_loss: {min_val_loss:.4f} @ epoch {min_val_epoch}"
                    )

            # validation accuracy
            if val_accuracies:
                label_val_acc = (
                    f"{label} {early_stopping}" if i <= 0 and show_labels else None
                )
                epochs_val_acc = range(1, len(val_accuracies) + 1)
                ax_acc.plot(
                    epochs_val_acc,
                    val_accuracies,
                    color="#2ca02c",
                    label=label_val_acc,
                    zorder=1,
                )
                ax_acc.set_xticks(range(0, len(val_accuracies) + 1, 10))
                # mark best accuracy for this run
                max_val_acc = max(val_accuracies)
                max_val_epoch = val_accuracies.index(max_val_acc) + 1
                ax_acc.scatter(
                    [max_val_epoch],
                    [max_val_acc],
                    s=48,
                    c="#2ca02c",
                    marker="o",
                    zorder=2,
                    label="checkpoint",
                )
                fig.suptitle(
                    f"Training Progress | best {early_stopping}: {max_val_acc*100:.2f}% @ epoch {max_val_epoch}"
                )

    # labels, limits, grid, title
    ax_train_loss.set_ylabel("training loss")
    ax_train_loss.set_yscale("log")
    ax_train_loss.grid(True, which="both")

    ax_val_loss.set_ylabel("validation loss")
    ax_val_loss.set_yscale("log")
    ax_val_loss.grid(True, which="both")

    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_yticks(np.linspace(0.0, 1.0, 11))
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.grid(True, which="both")

    # combine legends from both axes and avoid duplicates
    handles_train_loss, labels_train_loss = ax_train_loss.get_legend_handles_labels()
    handles_val_loss, labels_val_loss = ax_val_loss.get_legend_handles_labels()
    handles_acc, labels_acc = ax_acc.get_legend_handles_labels()
    handles, labels = [], []
    for h, l in (
        list(zip(handles_train_loss, labels_train_loss))
        + list(zip(handles_val_loss, labels_val_loss))
        + list(zip(handles_acc, labels_acc))
    ):
        if l not in labels:
            labels.append(l)
            handles.append(h)

    if labels:
        ax_acc.legend(handles, labels, loc="lower right", fontsize="small")

    # title and layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave room for suptitle
    plt.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def get_run_name_and_path(config):
    dataset = config.dataset_type.replace("/", "_")
    if "waterbird" in config.dataset_type:
        dataset += (
            "_wm" if config.waterbirds_use_minority else "_wom"
        )  # with or without minority
    run_name = (
        f"{config.run_label}_{dataset}_{config.model_class}_seed{config.random_seed}"
    )
    run_path = os.path.join(config.results_dir, run_name)
    return run_name, run_path


def create_output_dir(config):
    # create output directory
    run_name, run_path = get_run_name_and_path(config)
    exists_ok = True if "debug" in config.run_label else False
    try:
        os.makedirs(run_path, exist_ok=exists_ok)
    except FileExistsError:
        raise FileExistsError(
            f"Output directory {run_path} already exists. Please choose a different run_name or delete the existing directory."
        )
    # dump config
    yaml.dump(config, open(f"{run_path}/config.yaml", "w"))
    return run_name, run_path


def write_results(config, run_path, loss_logs, accuracy_dict):
    # write losses to file
    if loss_logs is not None:
        with open(f"{run_path}/losses.txt", "w") as f:
            for epoch in range(len(loss_logs[0])):
                train_loss = loss_logs[0][epoch]
                if loss_logs[1]:
                    valid_loss = loss_logs[1][epoch]
                    if loss_logs[2]:
                        valid_acc = loss_logs[2][epoch]
                        f.write(
                            f"Epoch {epoch+1:03d}: Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}\n"
                        )
                    else:
                        f.write(
                            f"Epoch {epoch+1:03d}: Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}\n"
                        )
                else:
                    f.write(f"Epoch {epoch+1:03d}: Train Loss: {train_loss:.4f}\n")

    # log final test accuracies to file

    for split_name, (accuracies, group_info) in accuracy_dict.items():

        with open(f"{run_path}/{split_name}_accuracies.txt", "w") as f:

            if "waterbird" in config.dataset_type:
                f.write(f"Landbird on Land: {accuracies[0]*100:02.3f} %\n")
                f.write(f"Waterbird on Land: {accuracies[1]*100:02.3f} %\n")
                f.write(f"Landbird on Water: {accuracies[2]*100:02.3f} %\n")
                f.write(f"Waterbird on Water: {accuracies[3]*100:02.3f} %\n")
                f.write("-" * 30 + "\n")
                f.write(f"Majority Accuracy: {accuracies[4]*100:02.3f} %\n")
                f.write(f"Minority Accuracy: {accuracies[5]*100:02.3f} %\n")
                f.write(f"Overall Accuracy: {accuracies[6]*100:02.3f} %\n")
                f.write(f"Worst Group Accuracy: {accuracies[7]*100:02.3f} %\n")

            elif "spawrious" in config.dataset_type:
                # f.write(f"--- {split_name.upper()} ---\n")

                group_accuracies = accuracies[:-2]
                group_info = list(group_info)

                for cls_idx, cls in enumerate(SPAWRIOUS_CLASS_NAMES):
                    for loc_idx, loc in enumerate(SPAWRIOUS_LOCATION_NAMES):
                        f.write(f"{cls.capitalize()} on {loc.capitalize()}: ")
                        group = (cls_idx, loc_idx)
                        if group in group_info:
                            acc = group_accuracies[group_info.index(group)]
                            f.write(f"{acc*100:02.3f} %")
                        else:
                            f.write(f"N/A")
                        f.write("\n")

                f.write("-" * 30 + "\n")
                f.write(f"Overall Accuracy: {accuracies[-2]*100:02.3f} %\n")
                f.write(f"Worst Group Accuracy: {accuracies[-1]*100:02.3f} %\n")

            elif "spurious_vehicles" in config.dataset_type:
                # f.write(f"--- {split_name.upper()} ---\n")

                group_accuracies = accuracies[:-2]
                group_info = list(group_info)

                for cls_idx, cls in enumerate(VEHICLES_CLASS_NAMES):
                    for loc_idx, loc in enumerate(VEHICLES_SPUR_ATT_NAMES):
                        f.write(f"{cls.capitalize()}; {loc.capitalize()}: ")
                        group = (cls_idx, loc_idx)
                        if group in group_info:
                            acc = group_accuracies[group_info.index(group)]
                            f.write(f"{acc*100:02.3f} %")
                        else:
                            f.write(f"N/A")
                        f.write("\n")

                f.write("-" * 30 + "\n")
                f.write(f"Overall Accuracy: {accuracies[-2]*100:02.3f} %\n")
                f.write(f"Worst Group Accuracy: {accuracies[-1]*100:02.3f} %\n")


def compute_additional_accuracies(config, accuracies, counts):

    accuracies = np.array(accuracies)
    counts = np.array(counts)

    correct = accuracies * counts
    overall_acc = float(correct.sum() / counts.sum())
    worst_acc = float(accuracies.min())

    if "waterbird" in config.dataset_type:
        majority_acc = float((correct[0] + correct[3]) / (counts[0] + counts[3]))
        minority_acc = float((correct[1] + correct[2]) / (counts[1] + counts[2]))
        additional_accuracies = [majority_acc, minority_acc, overall_acc, worst_acc]
    elif (
        "spawrious" in config.dataset_type or "spurious_vehicles" in config.dataset_type
    ):
        additional_accuracies = [overall_acc, worst_acc]

    return additional_accuracies


def get_eval_split_names(config):
    if (
        "spawrious"
        in config.dataset_type
        # "spurious_vehicles" in config.dataset_type
    ):
        return ["test", "test_shifted"]
    return ["test"]


def evaluate_model(**kwargs):

    config = kwargs["config"]
    eval_func = kwargs["eval_func"]

    accuracy_dict = dict()
    for split_name in get_eval_split_names(config):
        accuracies, counts, group_info = eval_func(split_name=split_name, **kwargs)
        accuracies += compute_additional_accuracies(config, accuracies, counts)
        accuracy_dict[split_name] = accuracies, group_info

    return accuracy_dict


def finalize_results(**kwargs):

    config = kwargs["config"]
    run_path = kwargs["run_path"]
    loss_logs = kwargs["loss_logs"] if "loss_logs" in kwargs else None

    accuracy_dict = evaluate_model(**kwargs)
    write_results(config, run_path, loss_logs, accuracy_dict)
    if loss_logs:
        plot_results(
            {"classifier": [loss_logs]},
            fname=f"{run_path}/loss.png",
            early_stopping=config.early_stopping,
        )
