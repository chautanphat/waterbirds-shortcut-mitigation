import os
import math
import contextlib

import torch
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from methods.ours.detector.model import Detector
from methods.ours.detector.dataset import (
    spawrious_collect_rel_paths,
    vehicles_collect_paths,
)
from methods.common.common_utils import load_aotgan

AOT_DIM = 512


class ImageSeparator:

    def __init__(self, config, generator, detector):

        self.config = config
        self.generator = generator.to(config.device)
        self.detector = detector.to(config.device)

        self.transform_g = T.Compose([T.Resize((AOT_DIM, AOT_DIM)), T.ToTensor()])

        self.transform_d = T.Compose(
            [
                T.Resize((config.image_size, config.image_size)),
                T.ToTensor(),
                T.Normalize(mean=config.mean, std=config.std),
            ]
        )

    def _prob_to_masks(self, probs, thr, upsample=False):
        mask = (probs < thr).float()
        if upsample:
            assert mask.dim() == 3  # B, H, W

            mask = torch.nn.functional.interpolate(
                mask[:, None].float(),  # B,1,H,W
                size=(AOT_DIM, AOT_DIM),
                mode="nearest",
            )[
                :, 0
            ]  # B,H,W

            mask = mask.to(mask.dtype)
        return mask.unsqueeze(1)

    def _tensors_to_images(self, tensors, resize=False):
        images = []
        for i in range(tensors.shape[0]):
            image = tensors[i]
            # BG images
            if tensors.shape[1] == 3:
                if resize:
                    image = T.Resize((self.config.image_size, self.config.image_size))(
                        image
                    )
                image = (image.clamp(0, 1) * 255).permute(1, 2, 0).byte().numpy()
                image = Image.fromarray(image, "RGB")
            # FG masks
            elif tensors.shape[1] == 1:
                image = (image.clamp(0, 1) * 255).squeeze().byte().numpy()
                image = Image.fromarray(image, "L").convert("1")
            else:
                raise ValueError("invalid shape")
            images.append(image)
        return images

    @torch.no_grad()
    def separate(self, images, resize):

        images_d = torch.stack([self.transform_d(image) for image in images], dim=0).to(
            self.config.device
        )
        images_g = torch.stack([self.transform_g(image) for image in images], dim=0).to(
            self.config.device
        )

        # --- Predict FG/BG masks ----

        with (
            torch.autocast(device_type=self.config.device, dtype=torch.bfloat16)
            if self.config.cuda_optimizations
            else contextlib.nullcontext()
        ):
            logits = self.detector(images_d)
        probs = F.sigmoid(logits.float())

        masks_tight = self._prob_to_masks(probs, self.config.augment_tight_threshold)
        masks_loose = self._prob_to_masks(probs, self.config.augment_loose_threshold)
        masks_g = self._prob_to_masks(
            probs, self.config.augment_loose_threshold, upsample=True
        )

        # --- Inpaint BGs ----
        images_g_input = (images_g * 2.0 - 1.0) * (1 - masks_g) + masks_g

        with (
            torch.autocast(device_type=self.config.device, dtype=torch.bfloat16)
            if self.config.cuda_optimizations
            else contextlib.nullcontext()
        ):
            images_g_output = self.generator(images_g_input, masks_g)
        images_g_output = images_g_output.float()
        images_g_output = torch.clamp(images_g_output, -1.0, 1.0)
        images_g_output = (images_g_output + 1) / 2.0
        masks_g = masks_g.bool().expand(-1, 3, -1, -1)
        images_inp = images_g
        images_inp[masks_g] = images_g_output[masks_g]

        # --- Save results ----

        masks_tight = self._tensors_to_images(masks_tight.cpu())
        masks_loose = self._tensors_to_images(masks_loose.cpu())
        images_inp = self._tensors_to_images(images_inp.cpu(), resize=resize)

        return masks_tight, masks_loose, images_inp


def generate_waterbirds(data_path, separator, seed):

    img_path = os.path.join(data_path, "images")
    masks_tight_path = os.path.join(data_path, f"abs_seed{seed}", "masks_tight")
    masks_loose_path = os.path.join(data_path, f"abs_seed{seed}", "masks_loose")
    inpaint_path = os.path.join(data_path, f"abs_seed{seed}", "inpainted")

    if (
        os.path.isdir(masks_tight_path)
        or os.path.isdir(masks_loose_path)
        or os.path.isdir(inpaint_path)
    ):
        raise Exception(
            f"augmentation folders already exist, please remove it before running the script:\n{masks_tight_path}\n{masks_loose_path}\n{inpaint_path}"
        )

    os.makedirs(masks_tight_path)
    os.makedirs(masks_loose_path)
    os.makedirs(inpaint_path)

    for folder in tqdm(
        sorted(os.listdir(img_path)), desc="BG/FG Separation Waterbirds"
    ):
        filenames = sorted(os.listdir(os.path.join(img_path, folder)))

        # Load images
        images = []
        for filename in filenames:
            image_path = os.path.join(img_path, folder, filename)
            image = Image.open(image_path).convert("RGB")
            images.append(image)

        # Separate image into FG+BG
        masks_tight, masks_loose, images_inp = separator.separate(images, resize=False)

        # Save separated images
        for i in range(len(images)):
            for images, path in zip(
                [masks_tight, masks_loose, images_inp],
                [masks_tight_path, masks_loose_path, inpaint_path],
            ):
                rel_path = os.path.join(path, folder)
                os.makedirs(rel_path, exist_ok=True)

                filename = filenames[i]
                image = images[i]
                if image.mode == "1":
                    filename = filename.replace(".jpg", ".tiff")

                image.save(os.path.join(rel_path, filename), compression_level=0)


def generate_spawrious(data_path, separator, seed, batch_size, dataset_type):

    img_path = os.path.join(data_path, "spawrious224")
    masks_tight_path = os.path.join(
        data_path, f"{dataset_type}_abs_seed{seed}", "masks_tight"
    )
    masks_loose_path = os.path.join(
        data_path, f"{dataset_type}_abs_seed{seed}", "masks_loose"
    )
    inpaint_path = os.path.join(
        data_path, f"{dataset_type}_abs_seed{seed}", "inpainted"
    )

    if (
        os.path.isdir(masks_tight_path)
        or os.path.isdir(masks_loose_path)
        or os.path.isdir(inpaint_path)
    ):
        raise Exception(
            f"augmentation folders already exist, please remove it before running the script:\n{masks_tight_path}\n{masks_loose_path}\n{inpaint_path}"
        )

    os.makedirs(masks_tight_path, exist_ok=True)
    os.makedirs(masks_loose_path, exist_ok=True)
    os.makedirs(inpaint_path, exist_ok=True)

    # collect the file paths to all images in the specified variant
    _, type, difficulty = dataset_type.split("_")
    rel_paths = spawrious_collect_rel_paths(
        root=data_path, variant=f"{type}_{difficulty}"
    )
    file_paths = []
    for rel_path in rel_paths:
        env, loc, cls, fname = rel_path.split("/")
        file_path = (f"m2m/{env}/{loc}/{cls}", fname)
        file_paths.append(file_path)
    file_paths_batches = [
        file_paths[i : i + batch_size] for i in range(0, len(file_paths), batch_size)
    ]  # split into batches

    # create output folder
    for batch in tqdm(file_paths_batches, desc=f"BG/FG Separation {dataset_type}"):

        images = []
        for rel_path, filename in batch:
            image_path = os.path.join(img_path, rel_path, filename) + ".png"
            image = Image.open(image_path).convert("RGB")
            images.append(image)

        # Separate image into FG+BG
        masks_tight, masks_loose, images_inp = separator.separate(images, resize=True)

        # Save separated images
        for images, root_path in zip(
            [masks_tight, masks_loose, images_inp],
            [masks_tight_path, masks_loose_path, inpaint_path],
        ):
            for i in range(len(images)):
                rel_path, filename = batch[i]
                image = images[i]
                if image.mode == "1":
                    filename += ".tiff"
                elif image.mode == "RGB":
                    filename += ".png"
                else:
                    raise ValueError
                os.makedirs(os.path.join(root_path, rel_path), exist_ok=True)
                image.save(
                    os.path.join(root_path, rel_path, filename), compression_level=0
                )


def generate_spurious_vehicles(data_path, separator, seed, batch_size, dataset_type):

    setting = dataset_type.split("_")[-1]

    out_path = os.path.join(data_path, f"spurious_vehicles_{setting}_abs_seed{seed}")
    if os.path.isdir(out_path):
        raise Exception(
            f"augmentation folders already exist, please remove it before running the script:\n{out_path}"
        )

    # Collect image paths
    paths = vehicles_collect_paths(data_path, setting)
    paths = [path[:-4] for path in paths]
    paths_batches = [
        paths[i : i + batch_size] for i in range(0, len(paths), batch_size)
    ]  # split into batches

    # create output folder
    for batch in tqdm(paths_batches, desc=f"BG/FG Separation spurious"):

        images = []
        for path in batch:
            image = Image.open(os.path.join(path + ".png")).convert("RGB")
            images.append(image)

        # Separate image into FG+BG
        masks_tight, masks_loose, images_inp = separator.separate(images, resize=False)

        # Save separated images
        for images, foldername in zip(
            [masks_tight, masks_loose, images_inp],
            ["masks_tight", "masks_loose", "inpainted"],
        ):
            for i in range(len(images)):
                image = images[i]
                path = batch[i]
                path = path.replace(
                    "spurious_vehicles",
                    f"spurious_vehicles_{setting}_abs_seed{seed}/{foldername}",
                )
                if image.mode == "1":
                    path += ".tiff"
                elif image.mode == "RGB":
                    path += ".png"
                else:
                    raise ValueError
                os.makedirs(os.path.dirname(path) + os.sep, exist_ok=True)
                image.save(path, compression_level=0)


def run(config):

    # load generator model
    generator = load_aotgan(config.augment_aot_gan_path)
    generator.load_state_dict(
        torch.load(config.augment_generator_ckpt, map_location="cpu")
    )
    if config.cuda_optimizations:
        generator = torch.compile(generator)
    generator.eval()

    # load detector model
    detector_path = f"./methods/ours/detector/results_{config.dataset_type}_{config.detector_dataset}_seed{config.random_seed}/checkpoint.pt"
    detector = Detector(config)
    detector.load_state_dict(
        torch.load(
            detector_path,
            map_location=torch.device(config.device),
        )
    )
    if config.cuda_optimizations:
        detector = torch.compile(detector)
    detector.eval()

    separator = ImageSeparator(config, generator, detector)

    if "waterbird" in config.dataset_type:
        generate_waterbirds(
            data_path="./data/waterbird_complete95_forest2water2",
            separator=separator,
            seed=config.random_seed,
        )
    elif "spawrious" in config.dataset_type:
        generate_spawrious(
            data_path="./data",
            separator=separator,
            seed=config.random_seed,
            batch_size=config.augment_batch_size,
            dataset_type=config.dataset_type.replace("/", "_"),
        )
    elif "spurious_vehicles" in config.dataset_type:
        generate_spurious_vehicles(
            data_path="./data",
            separator=separator,
            seed=config.random_seed,
            batch_size=config.augment_batch_size,
            dataset_type=config.dataset_type,
        )
    else:
        raise ValueError(f"unknown dataset type: '{config.dataset_type}'")
