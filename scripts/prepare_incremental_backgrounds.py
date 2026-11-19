"""Find reviewable bird-free Places crops in canonical Waterbirds train images."""
import argparse
from collections import Counter
import hashlib
import io
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import maximum_filter

from audit_vulnerability_ot_incremental import DATA, MASK_SHA, OUTPUT, TRAIN_DATA
from src.counterfactual.common import pixel_hash, read_csv, sha256, write_csv

SCENES = {"bamboo_forest": "land", "forest/broadleaf": "land", "lake/natural": "water", "ocean": "water"}


def find_crop(mask, size=256, margin=16):
    support = np.asarray(mask) > 0
    h, w = support.shape
    if min(h, w) < size or not support.any() or support.all():
        return None
    blocked = maximum_filter(support, size=2 * margin + 1, mode="constant")
    integral = np.pad(blocked, ((1, 0), (1, 0))).cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)
    sums = integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]
    ys, xs = np.nonzero(sums == 0)
    if not len(xs):
        return None
    center_distance = (2 * xs + size - w) ** 2 + (2 * ys + size - h) ** 2
    best = int(center_distance.argmin())
    x, y = int(xs[best]), int(ys[best])
    return x, y, x + size, y + size


def run(archive_path, cohort_path, output, per_scene, *, data_root=DATA, crop_size=256):
    if sha256(archive_path) != MASK_SHA:
        raise ValueError("Caltech segmentation archive SHA-256 differs")
    cohort = read_csv(cohort_path)
    reserve = {r["sample_id"] for r in cohort if r["planning_role"] == "reserve"}
    metadata = read_csv(data_root / "metadata.csv")
    place_counts = Counter(r["place_filename"] for r in metadata)
    candidates = {}
    for r in metadata:
        scene = "/".join(r["place_filename"].strip("/").split("/")[1:-1])
        if r["split"] == "0" and r["img_id"] in reserve and place_counts[r["place_filename"]] == 1 and scene in SCENES:
            candidates[Path(r["img_filename"]).with_suffix(".png").as_posix()] = (r, scene)
    records = []
    by_scene = Counter()
    seen_pixels = set()
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            canonical_mask = "/".join(PurePosixPath(member.name).parts[-2:])
            if not member.isfile() or canonical_mask not in candidates:
                continue
            source, scene = candidates[canonical_mask]
            if by_scene[scene] >= per_scene:
                continue
            with archive.extractfile(member) as stream:
                raw_mask = stream.read()
            with Image.open(io.BytesIO(raw_mask)) as image:
                mask = image.convert("L")
            # Native archive geometry, with neither image nor mask resized.
            source_path = TRAIN_DATA / source["img_filename"]
            with Image.open(source_path) as image:
                if image.size != mask.size or image.getexif().get(274, 1) != 1:
                    continue
                bounds = find_crop(mask, size=crop_size)
                if bounds is None:
                    continue
                crop = image.convert("RGB").crop(bounds)
            pixels = pixel_hash(crop)
            if pixels in seen_pixels:
                continue
            seen_pixels.add(pixels)
            candidate_id = f"train-bg-{source['img_id']}"
            path = output / f"{candidate_id}.png"
            crop.save(path)
            records.append(dict(background_id=candidate_id, source_bird_id=source["img_id"], source_split="train",
                                place_filename=source["place_filename"], scene=scene, background_type=SCENES[scene],
                                crop_xyxy=":".join(map(str, bounds)), exclusion_margin_pixels=16,
                                source_image_sha256=sha256(source_path), source_image_path=str(source_path),
                                mask_archive_member=member.name, source_mask_sha256=hashlib.sha256(raw_mask).hexdigest(),
                                path=str(path.resolve()), sha256=sha256(path), pixel_sha256=pixels))
            by_scene[scene] += 1
            if all(by_scene[s] == per_scene for s in SCENES):
                break
    if not all(by_scene[s] >= 2 for s in SCENES):
        raise ValueError(f"Insufficient disjoint bird-free crops: {dict(by_scene)}")
    grouped = {scene: [r for r in records if r["scene"] == scene] for scene in SCENES}
    width, height = 160, 186
    per_scene_rows = (per_scene + 5) // 6
    sheet = Image.new("RGB", (6 * width, len(SCENES) * per_scene_rows * height), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (scene, images) in enumerate(grouped.items()):
        for j, r in enumerate(images):
            x, y = j % 6 * width, (i * per_scene_rows + j // 6) * height
            with Image.open(r["path"]) as img:
                sheet.paste(img.resize((width, width), Image.Resampling.BILINEAR), (x, y))
            draw.text((x + 3, y + width + 3), f"{r['source_bird_id']} | {scene}",
                      font=ImageFont.load_default(size=10), fill="black")
    sheet.save(output / "background_candidates.png")
    write_csv(output / "background_candidates.csv", records)
    print(f"Prepared {len(records)} background candidates for visual review")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=OUTPUT / "assets/segmentations.tgz")
    parser.add_argument("--cohort", type=Path, default=OUTPUT / "cohort_manifest.csv")
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=OUTPUT / "assets/background_review")
    parser.add_argument("--per-scene", type=int, default=12)
    args = parser.parse_args()
    if args.per_scene < 2 or args.crop_size < 1:
        parser.error("--per-scene must be at least two and --crop-size must be positive")
    run(args.archive, args.cohort, args.output, args.per_scene, data_root=args.data_root, crop_size=args.crop_size)
