from tqdm import tqdm

import torch

from methods.common.waterbird import CLASS_NAMES as WATERBIRD_CLASS_NAMES
from methods.common.spawrious import CLASS_NAMES as SPAWRIOUS_CLASS_NAMES
from methods.common.common_utils import load_data
from utils import *

from transformers import Sam3Processor, Sam3Model


def evaluate(config, model, processor, data):

    mean = torch.tensor(config.mean).view(1, -1, 1, 1)
    std = torch.tensor(config.std).view(1, -1, 1, 1)

    if "waterbird" in config.dataset_type:
        class_names = WATERBIRD_CLASS_NAMES
    elif "spawrious" in config.dataset_type:
        class_names = SPAWRIOUS_CLASS_NAMES
    else:
        raise ValueError(f"unknown dataset '{config.dataset_type}'")

    counts = torch.zeros((len(data.dataset), len(class_names)), dtype=torch.int32)
    labels = []
    idx = 0

    for images, batch_labels in tqdm(data):

        # get uint8 images
        images = images * std + mean  # B x C x H x W
        images *= 255.0  # scale to [0, 255]
        images = images.to(torch.uint8)

        for image in images:
            for class_idx, class_name in enumerate(class_names):
                inputs = processor(
                    images=image, text=class_name, return_tensors="pt"
                ).to(config.device)

                with torch.no_grad():
                    outputs = model(**inputs)

                results = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.5,
                    mask_threshold=0.5,
                    target_sizes=inputs.get("original_sizes").tolist(),
                )[0]

                counts[idx, class_idx] = sum(
                    [mask.sum().item() for mask in results["masks"]]
                )
            idx += 1
        labels.append(batch_labels)

    labels = torch.cat(labels, dim=0).to(torch.int32)

    # compute accuracy
    preds = counts.argmax(dim=1)
    correct = (preds == labels).sum().item()
    accuracy = correct / len(data.dataset)

    print(f"Accuracy: {accuracy:.4f}")

    return accuracy


def eval_func(**kwargs):

    config = kwargs["config"]
    model = kwargs["model"]
    data = kwargs["data"]
    split_name = kwargs["split_name"]
    processor = kwargs["processor"]

    dataloaders, counts, group_info = data[split_name]
    accuracies = [
        evaluate(config, model, processor, dataloader) for dataloader in dataloaders
    ]
    return accuracies, counts, group_info


def run(config):

    _, run_path = create_output_dir(config)

    seed_everything(config.random_seed)
    data = load_data(config)

    model = Sam3Model.from_pretrained(
        "facebook/sam3", cache_dir=TRANSFORMERS_CACHE_DIR
    ).to(config.device)
    processor = Sam3Processor.from_pretrained(
        "facebook/sam3", cache_dir=TRANSFORMERS_CACHE_DIR
    )

    finalize_results(
        config=config,
        data=data,
        model=model,
        run_path=run_path,
        eval_func=eval_func,
        processor=processor,
    )
