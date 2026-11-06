import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from methods.common.waterbird import WaterBird
from methods.common.spawrious import Spawrious
from methods.common.vehicles import SpuriousVehicles

GAUSS_NOISE_STD = 0.25
N_CHANNELS = 3


def bbox_from_mask(mask):
    """Returns bounding box given mask."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, (x2 - x1 + 1), (y2 - y1 + 1)


class AugmentedDataset(Dataset):

    def __init__(
        self, transform, seed, aug_config, path_replace_spec, use_gt_segmentations
    ):

        self.transform = transform
        self.seed = seed
        self.path_replace_spec = path_replace_spec
        self.rng = np.random.default_rng(seed)

        # FG transformation configuration
        assert aug_config["infilling"] in [
            "noise",
            "gray",
            "average",
            "scramble_pixel",
            "scramble_patch",
            "gan",
        ], f"unknown infilling setting: '{aug_config['infilling']}'"
        if aug_config["infilling"] == "scramble_patch" and use_gt_segmentations:
            raise ValueError(
                "'scramble_patch' not available when using ground truth segmentations"
            )
        self.use_gt_segmentations = use_gt_segmentations
        self.infilling = aug_config["infilling"]
        self.scale_range = aug_config["fg_transform_scale"]
        self.rotation_range = aug_config["fg_transform_rotation"]
        self.translation_range = aug_config["fg_transform_translation"]

        self.dataset = None
        self.bg_paths = None

    def _collect_backgrounds_paths(self, n_classes):
        # Collect background image paths grouped by class label
        bg_paths = [[] for _ in range(n_classes)]
        for path, label in self.dataset:
            src, dst, _ = self.path_replace_spec
            if self.use_gt_segmentations:
                path = path.replace(src, f"{src}_counterfactuals")
            else:
                path = path.replace(src, f"{dst}abs_seed{self.seed}/inpainted")
            bg_paths[label].append(path)
        return bg_paths

    def _load_mask(self, image_path, image_size, mask_type, resize=True):
        src, dst, ext = self.path_replace_spec
        if mask_type == "masks_tight":
            mask_path = (
                image_path.replace(src, f"{dst}abs_seed{self.seed}/{mask_type}")[
                    : -len(ext)
                ]
                + ".tiff"
            )
        elif mask_type == "masks_loose":
            mask_path = (
                image_path.replace("inpainted", "masks_loose")[: -len(ext)] + ".tiff"
            )
        mask = Image.open(mask_path).convert("L")
        if resize:
            mask = mask.resize(image_size, Image.Resampling.NEAREST)
        return mask

    def _load_mask_gt(self, image_path):
        src, dst, ext = self.path_replace_spec
        image_path = image_path.replace("_counterfactuals", "")
        image_path = image_path.replace(src, f"{src}_segmentation_masks")
        mask_path = image_path[: -len(ext)] + ".tiff"
        mask = Image.open(mask_path).convert("L")
        return mask

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        # Sample a background-only image (inpainted using AOT-GAN) with class-balanced sampling
        # by first selecting a class then an image within that class
        rand_class = self.rng.integers(0, len(self.bg_paths))
        rand_index = self.rng.integers(0, len(self.bg_paths[rand_class]))
        bg_path = self.bg_paths[rand_class][rand_index]
        bg_image = Image.open(bg_path).convert("RGB")
        size = bg_image.size

        # Load standard image and class label
        img_path, label = self.dataset[index]
        image = Image.open(img_path).convert("RGB")
        image = image.resize(size, Image.Resampling.NEAREST)

        # Apply non-GAN infilling techniques
        if self.infilling != "gan":

            if self.use_gt_segmentations:
                mask = self._load_mask_gt(bg_path)
                mask = np.array(mask)
                loose_mask = np.zeros_like(mask)
                xywh = bbox_from_mask(mask)
                if xywh is not None:
                    x, y, w, h = xywh
                    loose_mask[y : y + h, x : x + w] = 1
                loose_mask = loose_mask.astype(bool)
                num_fg_pixels = loose_mask.sum()
            else:
                # Load the loose inpainting mask (True = region to replace)
                if "patch" in self.infilling:
                    loose_mask = self._load_mask(
                        bg_path, size, "masks_loose", resize=False
                    )
                else:
                    loose_mask = self._load_mask(bg_path, size, "masks_loose")

                loose_mask = np.array(loose_mask).astype(bool)
                num_fg_pixels = loose_mask.sum()

            # Check if mask is non-empty
            if num_fg_pixels > 0:
                bg_image = np.array(bg_image)

                # Fill masked region with a constant gray value
                if self.infilling == "gray":
                    bg_image[loose_mask] = 128

                # Fill masked region with random noise
                elif self.infilling == "noise":
                    uniform = self.rng.random(size=(1, N_CHANNELS))
                    gauss = (
                        self.rng.standard_normal(size=(num_fg_pixels, N_CHANNELS))
                        * GAUSS_NOISE_STD
                    )
                    noise = uniform + gauss
                    noise = np.round(np.clip(noise, 0, 1) * 255).astype(bg_image.dtype)
                    bg_image[loose_mask] = noise

                # Fill masked region with the average color of the unmasked area
                elif self.infilling == "average":
                    outside = bg_image[~loose_mask]
                    if outside.size > 0:
                        outside_avg = np.round(outside.mean(axis=0)).astype(
                            bg_image.dtype
                        )
                        bg_image[loose_mask] = outside_avg
                    else:
                        bg_image[loose_mask] = (
                            128  # fallback to gray if outside area is empty
                        )

                # Fill masked region by randomly sampling pixels from the unmasked area
                elif self.infilling == "scramble_pixel":
                    outside = bg_image[~loose_mask]
                    if outside.size > 0:
                        scramble_idx = self.rng.integers(
                            0, outside.shape[0], size=num_fg_pixels
                        )
                        bg_image[loose_mask] = outside[scramble_idx]
                    else:
                        bg_image[loose_mask] = (
                            128  # fallback to gray if outside area is empty
                        )

                # Same as above, but using whole patches instead of individual pixels
                elif self.infilling == "scramble_patch":
                    outside = np.column_stack(np.nonzero(~loose_mask))
                    if outside.size > 0:
                        inside = np.column_stack(np.nonzero(loose_mask))
                        patch_size = size[0] // loose_mask.shape[0]

                        for idx in range(inside.shape[0]):
                            iy, ix = inside[idx]  # current patch from inside area
                            oy, ox = outside[
                                self.rng.integers(0, outside.shape[0])
                            ]  # sample random patch from outside area
                            bg_image[
                                iy * patch_size : (iy + 1) * patch_size,
                                ix * patch_size : (ix + 1) * patch_size,
                                :,
                            ] = bg_image[
                                oy * patch_size : (oy + 1) * patch_size,
                                ox * patch_size : (ox + 1) * patch_size,
                                :,
                            ]  # replace current patch with random patch from outside
                    else:
                        loose_mask = self._load_mask(bg_path, size, "masks_loose")
                        bg_image[loose_mask] = (
                            128  # fallback to gray if outside area is empty
                        )

                # Convert array back to an image object
                bg_image = Image.fromarray(bg_image)

        # Create foreground-only by using the tight mask
        if self.use_gt_segmentations:
            tight_mask = self._load_mask_gt(img_path)
        else:
            tight_mask = self._load_mask(img_path, size, "masks_tight")
        fg_image = Image.fromarray(np.dstack([image, tight_mask])).convert("RGBA")

        # Apply random transformations to the foreground

        # Resize
        if self.scale_range is not None:
            scale = self.rng.uniform(*self.scale_range)
            new_size = (int(fg_image.width * scale), int(fg_image.height * scale))
            fg_image = fg_image.resize(new_size, Image.Resampling.NEAREST)

        # Rotate
        if self.rotation_range is not None:
            angle = self.rng.integers(*self.rotation_range)
            fg_image = fg_image.rotate(angle, Image.Resampling.NEAREST)

        # Translation
        if self.translation_range is not None:
            xywh = bbox_from_mask(tight_mask)
            if xywh is None:
                # no fg-object
                tx, ty = 0, 0
            else:
                # clip translation interval so fg-object stays inside
                img_w, img_h = size
                x, y, w, h = xywh
                tx, ty = self.rng.integers(*self.translation_range, 2)
                tx_min, tx_max = -x, img_w - (x + w)
                ty_min, ty_max = -y, img_h - (y + h)
                tx = int(np.clip(tx, tx_min, tx_max))
                ty = int(np.clip(ty, ty_min, ty_max))
        else:
            tx, ty = 0, 0

        # Composite the transformed foreground onto the background using its alpha mask
        bg_image.paste(fg_image, (tx, ty), fg_image)

        return (self.transform(bg_image), label)


class WaterbirdsAugmented(AugmentedDataset):

    def __init__(self, transform, seed, aug_config, **kwargs):
        super().__init__(
            transform,
            seed,
            aug_config,
            path_replace_spec=("images", "", ".jpg"),
            use_gt_segmentations=kwargs["use_gt_segmentations"],
        )

        self.dataset = WaterBird(
            root=kwargs["root"],
            split=kwargs["split"],
            groups=kwargs["groups"],
            transform=transform,
            yield_groups=False,
            load_images=False,
        )

        self.bg_paths = self._collect_backgrounds_paths(n_classes=2)
        self.weight = self.dataset.weight


class SpawriousAugmented(AugmentedDataset):

    def __init__(self, transform, seed, aug_config, **kwargs):
        super().__init__(
            transform,
            seed,
            aug_config,
            path_replace_spec=("spawrious224", kwargs["dataset_type"] + "_", ".png"),
            use_gt_segmentations=kwargs["use_gt_segmentations"],
        )

        self.dataset = Spawrious.joint(
            variant=kwargs["variant"],
            split=kwargs["split"],
            transform=transform,
            root=kwargs["root"],
            yield_groups=False,
            load_images=False,
            m2m_include_generic_bg=kwargs["m2m_include_generic_bg"],
        )
        self.bg_paths = self._collect_backgrounds_paths(n_classes=4)


class SpuriousVehiclesAugmented(AugmentedDataset):

    def __init__(self, transform, seed, aug_config, **kwargs):
        super().__init__(
            transform,
            seed,
            aug_config,
            path_replace_spec=(
                "spurious_vehicles",
                f"spurious_vehicles_{kwargs['setting']}_",
                ".png",
            ),
            use_gt_segmentations=kwargs["use_gt_segmentations"],
        )
        self.dataset = SpuriousVehicles.groups(
            root=kwargs["root"],
            split=kwargs["split"],
            transform=transform,
            yield_groups=False,
            load_images=False,
            concat=True,
            setting=kwargs["setting"],
        )

        self.bg_paths = self._collect_backgrounds_paths(n_classes=4)


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    from utils import Config, get_train_transform, get_test_transform, seed_everything

    # config = Config("./methods/ours/config_wb.yaml")
    # config = Config("./methods/ours/config_spawrious.yaml")
    # config.dataset_type = "spawrious/m2m_medium"
    config = Config("./methods/ours/config_spurious_vehicles.yaml")
    # config.random_seed = 123
    config.random_seed = 0
    config.use_gt_segmentations = True

    for infilling in [
        "noise",
        "gray",
        "average",
        "scramble_pixel",
        "scramble_patch",
        "gan",
    ]:
        config.infilling = infilling
        # transform = get_train_transform(config)
        transform = get_test_transform(config)
        seed_everything(config.random_seed)

        aug_config = {
            "infilling": config.infilling,
            "fg_transform_scale": config.fg_transform_scale,
            "fg_transform_rotation": config.fg_transform_rotation,
            "fg_transform_translation": config.fg_transform_translation,
            # "fg_transform_scale": None,
            # "fg_transform_rotation": None,
            # "fg_transform_translation": None,
        }

        try:
            if "waterbird" in config.dataset_type:
                dataset = WaterbirdsAugmented(
                    root="./data",
                    split="train",
                    groups=(1, 2, 3, 4),
                    transform=transform,
                    seed=config.random_seed,
                    aug_config=aug_config,
                    use_gt_segmentations=config.use_gt_segmentations,
                )
            elif "spawrious" in config.dataset_type:
                variant = config.dataset_type.split("/")[-1]
                dataset = SpawriousAugmented(
                    variant=variant,
                    split="train",
                    transform=transform,
                    root="./data",
                    m2m_include_generic_bg=False,
                    seed=config.random_seed,
                    aug_config=aug_config,
                    dataset_type=config.dataset_type.replace("/", "_"),
                    use_gt_segmentations=config.use_gt_segmentations,
                )
            elif "spurious_vehicles" in config.dataset_type:
                setting = config.dataset_type.split("_")[-1]
                dataset = SpuriousVehiclesAugmented(
                    root="./data",
                    split="train",
                    transform=transform,
                    seed=config.random_seed,
                    aug_config=aug_config,
                    setting=setting,
                    use_gt_segmentations=config.use_gt_segmentations,
                )
        except:
            continue

        # print("number of samples:", len(dataset))
        image, label = dataset[0]

        dataloder = DataLoader(
            dataset=dataset,
            batch_size=32,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            drop_last=False,
        )

        @torch.no_grad()
        def denorm(x, config):
            mean = torch.tensor(config.mean, dtype=torch.float32, device=x.device).view(
                1, 3, 1, 1
            )
            std = torch.tensor(config.std, dtype=torch.float32, device=x.device).view(
                1, 3, 1, 1
            )
            return x * std + mean

        for idx, (image, label) in enumerate(dataloder):
            print(infilling)
            # print(image.shape, image.dtype)
            # print(label.shape, label.dtype)
            save_image(
                denorm(image, config),
                f"{dataset.__class__.__name__}_{idx}_image_{infilling}.png",
            )
            break
