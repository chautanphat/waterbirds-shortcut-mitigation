import os
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
from tqdm import tqdm

# ------------------------
# CONFIG
# ------------------------

USE_LOCAL_WEIGHTS = False  # set to False to download weights from huggingface
WIDTH, HEIGHT = (512, 512)
NUM_IMAGES_PER_GROUP = 256
BATCH_SIZE = 8
BASE_DIR = "spurious_vehicles"
GGUF_URL = "https://huggingface.co/city96/FLUX.1-schnell-gguf/blob/main/flux1-schnell-Q8_0.gguf"

transformer = FluxTransformer2DModel.from_single_file(
    GGUF_URL,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
    local_files_only=USE_LOCAL_WEIGHTS
)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    local_files_only=USE_LOCAL_WEIGHTS
)

pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(disable=True)

# ------------------------
# SETTINGS
# ------------------------

settings = dict(
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    height=HEIGHT,
    width=WIDTH,
)

prompt_suffix = "captured from a camera, fully visible, centered in the frame, with no occlusion, rear view, no CGI, no render, no illustration"

objects = {
    "sedan": "photo of a single sedan car",
    "minivan": "photo of a single minivan / MPV",
    "SUV": "photo of a single, typical and non-off-road SUV",
    "pickup truck": "photo of a single pickup truck",
}

contexts = {
    "urban": "urban city street, buildings, sidewalks, street signs",
    "highway": "open highway, multiple lanes, guardrails, distant horizon",
    "rural": "rural countryside, fields, small roads, farms, vegetation",
    "off-road": "off-road terrain, dirt paths, mud, rocks, uneven ground",
    "parked": "stationary in a parking lot, single car",
}

# ------------------------
# GENERATION
# ------------------------

# progressbar
existing_total = 0
for ctx_name in contexts:
    for obj_name in objects:
        folder = os.path.join(BASE_DIR, ctx_name, obj_name)
        if os.path.exists(folder):
            existing_total += len([
                f for f in os.listdir(folder)
                if f.endswith(".png")
            ])

total = len(objects) * len(contexts) * NUM_IMAGES_PER_GROUP
pbar = tqdm(total=total, initial=existing_total, unit="image", desc="Generating images")


# main loop
for j, (ctx_name, ctx_prompt) in enumerate(contexts.items()):
    for i, (obj_name, obj_prompt) in enumerate(objects.items()):

        folder = os.path.join(BASE_DIR, ctx_name, obj_name)
        os.makedirs(folder, exist_ok=True)

        # count existing images
        existing_files = sorted([
            f for f in os.listdir(folder)
            if f.endswith(".png")
        ])

        generated = len(existing_files)
        img_idx = generated

        if generated >= NUM_IMAGES_PER_GROUP:
            continue

        prompt = f"{obj_prompt}, {ctx_prompt}, {prompt_suffix}"

        while generated < NUM_IMAGES_PER_GROUP:

            seed = (
                i * len(contexts) * NUM_IMAGES_PER_GROUP +
                j * NUM_IMAGES_PER_GROUP +
                img_idx
            )
            generator = torch.Generator("cpu").manual_seed(seed)

            remaining = NUM_IMAGES_PER_GROUP - generated
            current_batch = min(BATCH_SIZE, remaining)

            output = pipe(
                prompt=prompt,
                generator=generator,
                num_images_per_prompt=current_batch,
                **settings
            )

            for k, img in enumerate(output.images):
                if generated >= NUM_IMAGES_PER_GROUP:
                    break

                path = os.path.join(folder, f"{img_idx}.png")

                # safety
                if os.path.exists(path):
                    img_idx += 1
                    continue

                img.save(path)

                img_idx += 1
                generated += 1
            
            pbar.update(current_batch)

pbar.close()
print("Done!")
