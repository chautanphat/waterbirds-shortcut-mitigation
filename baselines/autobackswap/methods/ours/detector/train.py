import os
import gc
import yaml
import contextlib

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from methods.ours.detector.model import Detector
from methods.ours.detector.dataset import *
from utils import seed_everything, plot_results

N_SAMPLES_SPAWRIOUS = 400
N_SAMPLES_VEHICLES = 400


def load_data(config):

    train_transform = T.Compose(
        [
            # Do not apply augmentations (e.g. flipping, rotation etc.) here.
            # These are handled internally to ensure images and masks stay aligned.
            T.Resize((config.image_size, config.image_size)),
            T.ColorJitter(0.2, 0.2, 0.2, 0.02),
            T.RandomGrayscale(0.2),
            T.ToTensor(),
            T.Normalize(mean=config.mean, std=config.std),
        ]
    )

    test_transform = T.Compose(
        [
            T.Resize((config.image_size, config.image_size)),
            T.ToTensor(),
            T.Normalize(mean=config.mean, std=config.std),
        ]
    )

    # custom override for ablation study
    if hasattr(config, "detector_n_samples"):
        N_SAMPLES_WATERBIRDS = config.detector_n_samples

    data = {}
    print("total number of samples (including additional augmentation samples):")
    for split in ["train", "valid", "test"]:
        transform = train_transform if split == "train" else test_transform
        shuffle = split == "train"

        if config.detector_dataset == "waterbirds_hand_labeled":
            dataset = WaterbirdsHandLabeled(
                config=config, root="./data", transform=transform, split=split
            )
        elif config.detector_dataset == "waterbirds_from_segmentation":
            dataset = WaterbirdsFromSegmentations(
                config=config,
                root="./data",
                transform=transform,
                split=split,
                n_samples=N_SAMPLES_WATERBIRDS,
            )
        elif config.detector_dataset == "spawrious_from_segmentation":
            variant = config.dataset_type.split("/")[-1]
            dataset = SpawriousFromSegmentations(
                config=config,
                root="./data",
                transform=transform,
                split=split,
                variant=variant,
                n_samples=N_SAMPLES_SPAWRIOUS,
            )
        elif config.detector_dataset == "spurious_vehicles_from_segmentation":
            setting = config.dataset_type.split("_")[-1]
            dataset = SpuriousVehiclesFromSegmentations(
                config=config,
                root="./data",
                transform=transform,
                split=split,
                setting=setting,
                n_samples=N_SAMPLES_VEHICLES,
            )
        else:
            raise ValueError(f"invalid dataset_name: '{config.detector_dataset}'")

        print(f" - {split}: {len(dataset)}")

        dataloader = DataLoader(
            dataset,
            batch_size=config.detector_batch_size,
            shuffle=shuffle,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )
        data[split] = dataloader

    return data


def run_epoch(config, model, data_loader, optim=None, compute_acc=False):

    if optim:
        model.train()
    else:
        model.eval()

    if compute_acc:
        actual_labels = []
        pred_labels = []

    total_loss = 0
    total_elems = 0

    loss_fun = torch.nn.BCEWithLogitsLoss(reduction="none")

    for images, masks in data_loader:

        images = images.to(config.device)
        masks = masks.to(config.device)

        with torch.no_grad() if optim is None else contextlib.nullcontext():

            # target smoothing
            masks = (
                masks * (1 - config.detector_target_smoothing)
                + 0.5 * config.detector_target_smoothing
            )

            # forward pass & loss
            with (
                torch.autocast(device_type=config.device, dtype=torch.bfloat16)
                if config.cuda_optimizations
                else contextlib.nullcontext()
            ):

                logits = model(images)
                logits = logits.flatten(1)
                masks = masks.flatten(1)
                losses = loss_fun(logits, masks)
                loss = losses.mean()

                # loss logging
                total_loss += losses.sum().item()
                total_elems += losses.numel()

            # optimization
            if optim:
                optim.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()

            # accuracy
            if compute_acc:
                with torch.no_grad():
                    pred_labels.append(F.sigmoid(logits.cpu().flatten()) > 0.5)
                    actual_labels.append(masks.cpu().flatten() > 0.5)

    avg_loss = total_loss / total_elems

    if compute_acc:
        acc = balanced_accuracy_score(torch.cat(actual_labels), torch.cat(pred_labels))
        return avg_loss, acc

    return avg_loss


def training_loop(config, model, optim, data, num_epochs, scheduler=None):

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

    scheduler_metric = None

    pbar = tqdm(range(num_epochs), unit="epoch")

    results_dir = f"./methods/ours/detector/results_{config.dataset_type}_{config.detector_dataset}_seed{config.random_seed}"
    os.makedirs(results_dir, exist_ok=False)
    yaml.dump(config, open(f"{results_dir}/config.yaml", "w"))
    checkpoint_path = f"{results_dir}/checkpoint.pt"
    final_path = f"{results_dir}/final.pt"

    for _ in pbar:

        # training & lr scheduling
        train_loss = run_epoch(config, model, data["train"], optim)
        train_losses.append(train_loss)

        # validation
        val_loss, val_accuracy = run_epoch(
            config, model, data["valid"], compute_acc=True
        )
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        current_lr = optim.param_groups[0]["lr"]

        # early stopping
        if config.detector_early_stopping == "val_acc":
            scheduler_metric = val_accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), checkpoint_path)
                pbar.set_description(
                    f"best {config.detector_early_stopping}: {best_val_accuracy*100:.2f} % | lr: {current_lr:.2e}"
                )
        elif config.detector_early_stopping == "val_loss":
            scheduler_metric = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), checkpoint_path)
                pbar.set_description(
                    f"best {config.detector_early_stopping}: {best_val_loss:.4f} | lr: {current_lr:.2e}"
                )
        else:
            raise ValueError(
                f"invalid early stopping setting: '{config.detector_early_stopping}'"
            )

        # lr scheduler
        if scheduler:
            scheduler.step(scheduler_metric)

    # create final checkpoint
    torch.save(model.state_dict(), final_path)

    # load best checkpoint
    model.load_state_dict(
        torch.load(
            checkpoint_path, map_location=torch.device(config.device), weights_only=True
        )
    )

    torch.cuda.empty_cache()
    gc.collect()

    # plot loss
    loss_logs = train_losses, val_losses, val_accuracies
    plot_results(
        {"detector": [loss_logs]},
        fname=f"{results_dir}/losses.png",
        early_stopping=config.detector_early_stopping,
    )

    # write results
    with open(f"{results_dir}/losses.txt", "w") as f:
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            f.write(
                f"Epoch {epoch+1:03d}: Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}\n"
            )
    with open(f"{results_dir}/accuracies.txt", "w") as f:
        for split in ["train", "valid", "test"]:
            if data[split]:
                accuracy = run_epoch(config, model, data[split], compute_acc=True)[1]
                f.write(f"balanced accuracy ({split}): {accuracy*100:02.3f} %\n")


def train(config):

    config.cuda_optimizations = False  # train without mixed precision and torch.compile
    seed_everything(config.random_seed)
    data = load_data(config)

    if config.cuda_optimizations:
        torch.set_float32_matmul_precision(
            "high"
        )  # enables float32 tensor cores for matrix multiplication for better performance

    model = Detector(config)
    model = model.to(config.device)
    if config.cuda_optimizations:
        model = torch.compile(model)

    if config.detector_optim_type == "adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=config.detector_learning_rate,
            weight_decay=config.detector_weight_decay,
        )
        scheduler = None
    elif config.detector_optim_type == "sgd":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=config.detector_learning_rate,
            weight_decay=config.detector_weight_decay,
            momentum=config.detector_momentum,
        )
        if config.detector_early_stopping == "val_acc":
            mode = "max"
            threshold_mode = "abs"
            threshold = 0.0001
            patience = 20
        elif config.detector_early_stopping == "val_loss":
            mode = "min"
            threshold_mode = "rel"
            threshold = 0.0001
            patience = 10
        else:
            raise ValueError(
                f"invalid early stopping setting: '{config.detector_early_stopping}'"
            )
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=optim,
            mode=mode,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
        )
    else:
        raise ValueError(f"invalid optimizer: '{config.optimizer}'")

    # training
    training_loop(
        config=config,
        model=model,
        optim=optim,
        data=data,
        num_epochs=config.detector_num_epochs,
        scheduler=scheduler,
    )


# def run_experiment():
#     config = Config("./detector/config.yaml")

#     config.detector_dataset = "waterbirds_segmentation"
#     patch_resolutions = [2, 4, 8, 16, 32]

#     for res in reversed(patch_resolutions):
#         config.detector_patch_resolution = res
#         config.detector_max_mask_offset = res // 4

#         print(f"patch resolution: {res}x{res} " + "-" * 50)
#         seed_everything(config.random_seed)
#         run(config, plot_fname=f"training_losses_{res}x{res}", checkpoint_fname=f"detector_{res}")
