from tqdm import tqdm
import base64
from io import BytesIO
from PIL import Image

import torch

from methods.common.waterbird import CLASS_NAMES as WATERBIRD_CLASS_NAMES
from methods.common.spawrious import CLASS_NAMES as SPAWRIOUS_CLASS_NAMES
from methods.common.common_utils import load_data
from utils import *

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def tensor_to_base64(tensor):
    """
    Convert a torch tensor image to a base64 string.
    Assumes C x H x W format and values in [0, 255].
    """
    img = Image.fromarray((tensor.permute(1, 2, 0).cpu().numpy()))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


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


def evaluate(config, model, processor, data):

    mean = torch.tensor(config.mean).view(1, -1, 1, 1)
    std = torch.tensor(config.std).view(1, -1, 1, 1)

    if "waterbird" in config.dataset_type:
        class_names = WATERBIRD_CLASS_NAMES
        pretext = "Does this picture show a landbird or a waterbird species?"
    elif "spawrious" in config.dataset_type:
        class_names = SPAWRIOUS_CLASS_NAMES
        pretext = "What type of dog breed is shown in this picture?"
    else:
        raise ValueError(f"unknown dataset '{config.dataset_type}'")

    if config.ignore_background:
        pretext += " Ignore the background for your answer."

    preds = torch.zeros((len(data.dataset),), dtype=torch.int32)
    labels = []
    idx = 0

    for images, batch_labels in tqdm(data):

        # get uint8 images
        images = images * std + mean  # B x C x H x W
        images *= 255.0  # scale to [0, 255]
        images = images.byte()

        for image in images:

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": tensor_to_base64(image),
                        },
                        {
                            "type": "text",
                            "text": f"{pretext} Provide as answer a single word from this list: {', '.join(class_names)}. Answer:",
                        },
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=2048)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            # print(output_text)
            answer = output_text[0].split("</think>\n\n")[-1].strip().lower()
            # print(answer)

            # map answer to class index
            for class_idx, class_name in enumerate(class_names):
                if class_name.lower() in answer:
                    preds[idx] = class_idx
                    break

            idx += 1
        labels.append(batch_labels)

    labels = torch.cat(labels, dim=0).to(torch.int32)

    # compute accuracy
    correct = (preds == labels).sum().item()
    accuracy = correct / len(data.dataset)

    print(f"Accuracy: {accuracy:.4f}")

    return accuracy


def run(config):

    _, run_path = create_output_dir(config)

    seed_everything(config.random_seed)
    data = load_data(config)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Thinking",
        dtype="auto",
        device_map="auto",
        cache_dir=TRANSFORMERS_CACHE_DIR,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Thinking")

    finalize_results(
        config=config,
        data=data,
        model=model,
        run_path=run_path,
        eval_func=eval_func,
        processor=processor,
    )
