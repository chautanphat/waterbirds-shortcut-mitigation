import os
from tqdm import tqdm

import torch

from methods.common.waterbird import CLASS_NAMES as WATERBIRD_CLASS_NAMES
from methods.common.spawrious import CLASS_NAMES as SPAWRIOUS_CLASS_NAMES
from methods.common.common_utils import load_data
from utils import *

import open_clip

CLIP_MODEL = "ViT-bigG-14"
CLIP_DATASET = "laion2b_s39b_b160k"
# Smaller model for debugging
# CLIP_MODEL = "RN101"
# CLIP_DATASET = "openai"


def evaluate(config, model, tokenizer, data):

    if "waterbird" in config.dataset_type:
        class_names = WATERBIRD_CLASS_NAMES
    elif "spawrious" in config.dataset_type:
        class_names = SPAWRIOUS_CLASS_NAMES
    else:
        raise ValueError(f"unknown dataset '{config.dataset_type}'")

    cosims = torch.zeros((len(data.dataset), len(class_names)))
    labels = []
    idx = 0

    for images, batch_labels in tqdm(data):
        for class_idx, class_name in enumerate(class_names):

            with torch.no_grad():
                text_features = model.encode_text(
                    tokenizer(class_name).to(config.device)
                )
            text_features = torch.nn.functional.normalize(text_features, dim=-1).cpu()

            with torch.no_grad():
                image_features = model.encode_image(images.to(config.device))

            image_features = torch.nn.functional.normalize(image_features, dim=-1).cpu()

            for i in range(len(image_features)):
                cosims[idx + i, class_idx] = torch.dot(
                    image_features[i], text_features[0]
                ).item()

        idx += len(images)
        labels.append(batch_labels)

    labels = torch.cat(labels, dim=0).to(torch.int32)

    # compute accuracy
    preds = cosims.argmax(dim=1)
    correct = (preds == labels).sum().item()
    accuracy = correct / len(data.dataset)

    print(f"Accuracy: {accuracy:.4f}")

    return accuracy


def eval_func(**kwargs):

    config = kwargs["config"]
    model = kwargs["model"]
    data = kwargs["data"]
    split_name = kwargs["split_name"]
    tokenizer = kwargs["tokenizer"]

    dataloaders, counts, group_info = data[split_name]
    accuracies = [
        evaluate(config, model, tokenizer, dataloader) for dataloader in dataloaders
    ]
    return accuracies, counts, group_info


def run(config):

    _, run_path = create_output_dir(config)

    seed_everything(config.random_seed)
    data = load_data(config)

    model, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_DATASET, cache_dir=TRANSFORMERS_CACHE_DIR
    )
    model = model.to(config.device).eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)

    finalize_results(
        config=config,
        data=data,
        model=model,
        run_path=run_path,
        eval_func=eval_func,
        tokenizer=tokenizer,
    )
