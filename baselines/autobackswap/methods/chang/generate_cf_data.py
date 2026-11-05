import os
import contextlib

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from methods.common.common_utils import load_aotgan

# segmentation thresholds for counterfactual inpainting (used derive bounding box and inpaint the foreground)
WB_CF_SEG_THR = 0.1
SPAWRIOUS_CF_SEG_THR = 0.0
SPURIOUS_VEHICLES_CF_SEG_THR = 0.0
AOT_DIM = 512


@torch.no_grad()
def bounding_box_mask(segmentation, threshold):
    mask = segmentation > threshold
    ys, xs = mask.nonzero(as_tuple=True)
    if len(xs) == 0:
        return torch.zeros_like(mask, dtype=torch.bool)
    # bounding box coordinates
    x1, x2 = xs.min().item(), xs.max().item()
    y1, y2 = ys.min().item(), ys.max().item()
    bb = torch.zeros_like(mask, dtype=torch.bool)
    bb[y1 : y2 + 1, x1 : x2 + 1] = True
    return bb


@torch.no_grad()
def aot_postprocess(image):
    image = torch.clamp(image, -1.0, 1.0)
    image = (image + 1) / 2.0
    return image


def get_aot_transform():
    return T.Compose([T.Resize((AOT_DIM, AOT_DIM)), T.ToTensor()])


@torch.no_grad()
def generate_counterfactuals_waterbirds(config, data_path, generator):

    aot_transform = get_aot_transform()

    out_path = os.path.join(data_path, "counterfactuals")
    if os.path.isdir(out_path):
        raise Exception(
            f"counterfactuals folder already exists, please remove it before running the script:\n{out_path}"
        )
    os.makedirs(out_path)

    generator.to(config.device)

    for folder_name in tqdm(
        os.listdir(os.path.join(data_path, "images")),
        desc="Generating CF data for Waterbirds",
    ):

        image_names = os.listdir(os.path.join(data_path, "images", folder_name))
        images = []
        masks = []
        for image_name in image_names:
            # load image and segmentation
            image_path = os.path.join(data_path, "images", folder_name, image_name)
            segment_path = os.path.join(
                data_path,
                "segmentations",
                folder_name,
                image_name.replace(".jpg", ".png"),
            )
            image = aot_transform(Image.open(image_path).convert("RGB"))
            segment = aot_transform(Image.open(segment_path).convert("L"))
            # prepare image and bounding box mask for inpainting
            mask = (
                bounding_box_mask(segment.squeeze(), WB_CF_SEG_THR).unsqueeze(0).float()
            )
            image = (image * 2.0 - 1.0) * (1 - mask) + mask
            images.append(image)
            masks.append(mask)

        # perform inpainting
        images = torch.stack(images, dim=0).to(config.device)
        masks = torch.stack(masks, dim=0).to(config.device)
        with (
            torch.autocast(device_type=config.device, dtype=torch.bfloat16)
            if config.cuda_optimizations
            else contextlib.nullcontext()
        ):
            images_out = generator(images, masks).cpu()
        images_out = aot_postprocess(images_out.float())
        images = aot_postprocess(images)
        masks = masks.cpu().bool().expand(-1, 3, -1, -1)
        images_inp = images.cpu()
        images_inp[masks] = images_out[masks]

        # save inpainted images
        for i in range(images_inp.shape[0]):
            image_path = os.path.join(out_path, folder_name)
            os.makedirs(image_path, exist_ok=True)
            image = images_inp[i]
            image = (image.clamp(0, 1) * 255).permute(1, 2, 0).byte().numpy()
            Image.fromarray(image, "RGB").save(os.path.join(image_path, image_names[i]))


@torch.no_grad()
def generate_counterfactuals_spawrious(config, data_root, generator, batch_size):

    aot_transform = get_aot_transform()
    output_transform = T.Resize((224, 224))

    images_root = os.path.join(data_root, "spawrious224")
    segments_root = os.path.join(data_root, "spawrious224_segmentation_masks")
    inpaints_root = os.path.join(data_root, "spawrious224_counterfactuals")

    if os.path.isdir(inpaints_root):
        raise Exception(
            f"counterfactuals folder already exists, please remove it before running the script:\n{inpaints_root}"
        )
    os.makedirs(inpaints_root)

    # collect paths to all images
    file_paths = []
    for root, dirs, files in os.walk(images_root):
        if "m2m" in root:
            for file in files:
                if file.endswith(".png"):
                    rel_path = os.path.relpath(root, images_root)
                    file_name = os.path.splitext(file)[0]
                    file_paths.append((rel_path, file_name))
    file_paths_batches = [
        file_paths[i : i + batch_size] for i in range(0, len(file_paths), batch_size)
    ]

    # create output folder
    for batch in tqdm(file_paths_batches, desc="Generating CF data for Spawrious"):

        images = []
        masks = []

        for rel_path, file_name in batch:

            # load image and segmentation
            image_path = os.path.join(images_root, rel_path, file_name + ".png")
            segment_path = os.path.join(segments_root, rel_path, file_name + ".tiff")
            image = aot_transform(Image.open(image_path).convert("RGB"))
            segment = aot_transform(Image.open(segment_path).convert("L"))

            # prepare image and bounding box mask for inpainting
            mask = (
                bounding_box_mask(segment.squeeze(), SPAWRIOUS_CF_SEG_THR)
                .unsqueeze(0)
                .float()
            )
            image = (image * 2.0 - 1.0) * (1 - mask) + mask
            images.append(image)
            masks.append(mask)

        # perform inpainting
        images = torch.stack(images, dim=0).to(config.device)
        masks = torch.stack(masks, dim=0).to(config.device)
        with (
            torch.autocast(device_type=config.device, dtype=torch.bfloat16)
            if config.cuda_optimizations
            else contextlib.nullcontext()
        ):
            images_inp = generator(images, masks).cpu()
        images_inp = output_transform(aot_postprocess(images_inp))
        # save inpainted images
        for i, (rel_path, file_name) in enumerate(batch):
            os.makedirs(os.path.join(inpaints_root, rel_path), exist_ok=True)
            inpaint_path = os.path.join(inpaints_root, rel_path, file_name + ".png")
            image = images_inp[i]
            image = (image.clamp(0, 1) * 255).permute(1, 2, 0).byte().numpy()
            Image.fromarray(image, "RGB").save(inpaint_path)


@torch.no_grad()
def generate_counterfactuals_spurious_vehicles(
    config, data_root, generator, batch_size
):

    aot_transform = get_aot_transform()

    images_root = os.path.join(data_root, "spurious_vehicles")
    inpaints_root = os.path.join(data_root, "spurious_vehicles_counterfactuals")
    if os.path.isdir(inpaints_root):
        raise Exception(
            f"counterfactuals folder already exists, please remove it before running the script:\n{inpaints_root}"
        )

    # collect paths to all images
    file_paths = []
    for root, dirs, files in os.walk(images_root):
        for file in files:
            if file.endswith(".png"):
                full_path = os.path.join(root, file)
                file_paths.append(full_path)
    file_paths = [file_paths[:-4] for file_paths in file_paths]
    file_paths_batches = [
        file_paths[i : i + batch_size] for i in range(0, len(file_paths), batch_size)
    ]

    # create output folder
    for batch in tqdm(
        file_paths_batches, desc="Generating CF data for Vehicles dataset"
    ):

        # load images and segmentations
        images = []
        masks = []
        for file_path in batch:
            image_path = file_path + ".png"
            segment_path = (
                file_path.replace(
                    "spurious_vehicles", "spurious_vehicles_segmentation_masks"
                )
                + ".tiff"
            )
            image = aot_transform(Image.open(image_path).convert("RGB"))
            segment = aot_transform(Image.open(segment_path).convert("L"))
            # prepare image and bounding box mask for inpainting
            mask = (
                bounding_box_mask(segment.squeeze(), SPURIOUS_VEHICLES_CF_SEG_THR)
                .unsqueeze(0)
                .float()
            )
            image = (image * 2.0 - 1.0) * (1 - mask) + mask
            images.append(image)
            masks.append(mask)

        # perform inpainting
        images = torch.stack(images, dim=0).to(config.device)
        masks = torch.stack(masks, dim=0).to(config.device)
        with (
            torch.autocast(device_type=config.device, dtype=torch.bfloat16)
            if config.cuda_optimizations
            else contextlib.nullcontext()
        ):
            images_inp = generator(images, masks).cpu()
        images_inp = aot_postprocess(images_inp)
        # save inpainted images
        for i, file_path in enumerate(batch):
            inpaint_path = (
                file_path.replace(
                    "spurious_vehicles", "spurious_vehicles_counterfactuals"
                )
                + ".png"
            )
            image = images_inp[i]
            image = (image.clamp(0, 1) * 255).permute(1, 2, 0).byte().numpy()
            os.makedirs(os.path.dirname(inpaint_path) + os.sep, exist_ok=True)
            Image.fromarray(image, "RGB").save(inpaint_path)


def run(config):

    generator = load_aotgan(config.cf_aot_gan_path)
    generator.load_state_dict(torch.load(config.cf_generator_ckpt, map_location="cpu"))
    generator.eval()
    generator.to(config.device)
    if config.cuda_optimizations:
        generator = torch.compile(generator)

    if "waterbird" in config.dataset_type:
        generate_counterfactuals_waterbirds(
            config=config,
            data_path="./data/waterbird_complete95_forest2water2",
            generator=generator,
        )
    elif "spawrious" in config.dataset_type:
        generate_counterfactuals_spawrious(
            config=config, data_root="./data", generator=generator, batch_size=64
        )
    elif "spurious_vehicles" in config.dataset_type:
        generate_counterfactuals_spurious_vehicles(
            config=config, data_root="./data", generator=generator, batch_size=64
        )
    else:
        raise ValueError(f"unknown dataset type: '{config.dataset_type}'")
