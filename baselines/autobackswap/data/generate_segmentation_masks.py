import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from transformers import Sam3Processor, Sam3Model

##### Configuration #####

TRANSFORMERS_CACHE_DIR = None  # set custom directory or use None for default directory
DEVICE = "cuda:0"

BATCH_SIZE = 8
MODE = ["spawrious", "spurious_vehicles"][1]
print("Running in mode:", MODE)

#########################

if MODE == "spawrious":
    INPUT_DIR = "spawrious224"
    OUT_DIR = "spawrious224_segmentation_masks"
    ROOT_DIRS = ["m2m"]
    KEYWORD = "dog"
    FILETYPE = ".png"
    RESUME_PROCESSING = True
    COMPILE = False
elif MODE == "spurious_vehicles":
    INPUT_DIR = "spurious_vehicles"
    OUT_DIR = "spurious_vehicles_segmentation_masks"
    ROOT_DIRS = []
    FILETYPE = ".png"
    RESUME_PROCESSING = True
    COMPILE = True


def sort_key(fname):
    try:
        if MODE == "spawrious":
            return int(fname.split(".")[0].split("_")[-1])
        elif MODE == "spurious_vehicles":
            return int(fname.split(".")[0])
    except:
        return -1


def get_image_paths(root_dir, masks=False):
    file_ext = ".tiff" if masks else FILETYPE
    image_paths = [] 
    for root, dirs, files in os.walk(root_dir):
        if any(rd in root for rd in ROOT_DIRS) or MODE == "spurious_vehicles":
            for file in sorted(files, key=sort_key):
                if file.endswith(file_ext):
                    if masks:
                        root = root.replace(OUT_DIR, INPUT_DIR)
                        file = file.replace(".tiff", FILETYPE)
                    image_paths.append((root, file))
    return image_paths

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def main():

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = Sam3Model.from_pretrained("facebook/sam3", cache_dir=TRANSFORMERS_CACHE_DIR).to(DEVICE)
    if COMPILE:
        model = torch.compile(model, mode="reduce-overhead")
    processor = Sam3Processor.from_pretrained("facebook/sam3", cache_dir=TRANSFORMERS_CACHE_DIR)

    # loop over all images in input folder, also in subfolders and collect all paths
    image_paths = get_image_paths(INPUT_DIR)
    print(f"Found {len(image_paths):,} images.")

    if RESUME_PROCESSING:
        # skip images that have already been processed
        image_paths_done = get_image_paths(OUT_DIR, masks=True)
        image_paths_todo = list(set(image_paths) - set(image_paths_done))
        print(f"{len(image_paths_done):,} images have already been processed.")
        print(f"{len(image_paths_todo):,} images remain to be processed.")
    else:
        image_paths_todo = image_paths

    # split files into batches
    image_paths_batches = [image_paths_todo[i:i + BATCH_SIZE] for i in range(0, len(image_paths_todo), BATCH_SIZE)]

    # create output folder
    os.makedirs(OUT_DIR, exist_ok=True)

    for batch in tqdm(image_paths_batches, desc="Generating segmentations with SAM3", unit="batch"):

        seed_everything(0)

        images = [Image.open(os.path.join(root, file)).convert("RGB") for root, file in batch]

        if MODE == "spurious_vehicles":
            texts = [root.split("/")[2] for (root, file) in batch]  # class name instead of static keyword
        else:
            texts = [KEYWORD] * BATCH_SIZE

        # process image
        inputs = processor(
            images=images, 
            text=texts[:len(images)],
            return_tensors="pt",
        ).to(DEVICE)

        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            outputs = model(**inputs)

        # Post-process results
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )

        assert len(results) == len(batch), f"Results length {len(results)} does not match batch length {len(batch)}"

        for i, ((root, file), result) in enumerate(zip(batch, results)):

            H, W = images[i].size[1], images[i].size[0]

            # superimpose all masks
            seg = torch.zeros((H, W), dtype=torch.uint8).to(DEVICE)
            for mask in result["masks"]:
                seg = torch.logical_or(seg, mask.byte())

            assert seg.shape == (H, W), f"Segmentation shape {seg.shape} does not match image shape {(H, W)}"

            # get segmentation mask
            seg = seg.cpu().numpy().astype(np.uint8) * 255

            # create output path
            relative_path = os.path.relpath(root, INPUT_DIR)
            output_dir = os.path.join(OUT_DIR, relative_path)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".tiff")

            # save segmentation mask as tiff
            seg_image = Image.fromarray(seg, mode="L").convert("1")
            seg_image.save(output_path)

        for img in images:
            img.close()


if __name__ == "__main__":
    main()
