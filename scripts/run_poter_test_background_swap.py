"""Score paired Land/Water composites for the label-filtered Waterbirds test set."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.counterfactual.common import KNOWN_WATERBIRDS_LABEL_ERROR_SPECIES, contained, pixel_hash, read_csv, sha256, write_json
from src.counterfactual.compositor import composite, foreground_layer
from src.counterfactual.data import ForegroundRepository
from src.counterfactual.failure_statistics import validate_predictions
from src.counterfactual.model import ModelAdapter

SCENE_PAIRS = (("train-bg-7127", "train-bg-5143"), ("train-bg-6661", "train-bg-1324"))
SEED = 20261119
LABELS = ("Landbird", "Waterbird")
MASK_MIN_COVERAGE = 204


def load_inputs(source):
    results = source / "results/poter_failure"
    integrity = json.loads((results / "integrity.json").read_text(encoding="utf-8"))["sha256"]
    for name in ("original_test_predictions.csv", "checkpoint_audit.json"):
        if sha256(results / name) != integrity[name]:
            raise ValueError(f"Saved source differs: {name}")
    checkpoint = json.loads((results / "checkpoint_audit.json").read_text(encoding="utf-8"))["POTER"]
    foregrounds = ForegroundRepository(source / "data/waterbirds")
    metadata = read_csv(foregrounds.root / "metadata.csv")
    source_by_id = {int(row["img_id"]): row for row in metadata if row["split"] == "2"}
    predictions = pd.read_csv(results / "original_test_predictions.csv", float_precision="round_trip")
    test_fg = pd.DataFrame([row for row in foregrounds.rows if row["original_split"] == "test"])
    test_fg["bird_source_id"] = test_fg.bird_source_id.astype(int)
    validate_predictions(predictions, test_fg, original=True)
    cohort = predictions.loc[predictions.model_name.eq("POTER")].sort_values("source_image_id").reset_index(drop=True)
    if len(cohort) != len(source_by_id) or set(cohort.source_image_id) != set(source_by_id) or set(cohort.model_seed) != {1}:
        raise ValueError("Expected exactly the canonical POTER seed 1 test predictions")
    for row in cohort.itertuples(index=False):
        meta = source_by_id[row.source_image_id]
        if (row.true_y, row.original_group, row.species_id) != (int(meta["y"]), 2*int(meta["y"])+int(meta["place"]), meta["img_filename"].split("/")[0]):
            raise ValueError(f"Official metadata differs for {row.source_image_id}")
    cohort = cohort[~cohort.species_id.isin(KNOWN_WATERBIRDS_LABEL_ERROR_SPECIES)].reset_index(drop=True)
    if len(cohort) != 5704 or int((cohort.pred_y == cohort.true_y).sum()) != 5311:
        raise ValueError("Filtered cohort differs from the audited benchmark")
    # Shuffle only IDs and assign equal numbers of birds to each fixed scene pair.
    assignment = np.empty(len(cohort), dtype=int)
    assignment[np.random.default_rng(SEED).permutation(len(cohort))] = np.arange(len(cohort)) % len(SCENE_PAIRS)
    return cohort.assign(scene_pair=assignment), foregrounds, checkpoint, integrity["original_test_predictions.csv"]


def load_backgrounds(source, data_root):
    metadata = read_csv(data_root / "metadata.csv")
    test_places = {r["place_filename"] for r in metadata if r["split"] == "2"}
    bank = {row["background_id"]: row for row in read_csv(source / "outputs/vulnerability_ot_incremental/background_banks.csv")}
    backgrounds, rows = {}, []
    for pair in SCENE_PAIRS:
        for name, kind in zip(pair, ("land", "water")):
            row = bank[name]
            path = contained(source, Path(row["path"]).relative_to(source))
            if (row["source_split"], row["background_type"], row["review"]) != ("train", kind, "accepted_before_inference") or row["place_filename"] in test_places or sha256(path) != row["sha256"]:
                raise ValueError(f"Unverified background {name}")
            with Image.open(path) as image:
                bg = image.convert("RGB")
            if bg.size != (256, 256) or pixel_hash(bg) != row["pixel_sha256"]:
                raise ValueError(f"Background pixels differ: {name}")
            backgrounds[name] = bg
            rows.append({k: row[k] for k in ("background_id", "source_bird_id", "source_split", "place_filename", "scene", "background_type", "pixel_sha256")})
    return backgrounds, rows


def build_pair(row, foregrounds, backgrounds):
    rgba, metadata = foregrounds.get(row.foreground_id)
    if metadata["quality_status"] != "verified_original_cub" or metadata["original_split"] != "test":
        raise ValueError(f"Foreground {row.foreground_id} is not an audited test cutout")
    # The prior train-side pilot used this fixed rule to suppress soft-mask background halos.
    official_alpha = rgba.getchannel("A")
    threshold = min(MASK_MIN_COVERAGE, official_alpha.getextrema()[1])
    rgba.putalpha(official_alpha.point(lambda a: 255 if a >= threshold else 0))
    layer = foreground_layer(rgba, sigma=0)
    result = []
    for name in SCENE_PAIRS[row.scene_pair]:
        image, audit = composite(layer, backgrounds[name])
        result.append((image, name, audit["alpha_sha256"], audit["foreground_layer_sha256"], threshold))
    if result[0][2:] != result[1][2:]:
        raise ValueError("Alpha or foreground changed between backgrounds")
    return result


def score(cohort, foregrounds, backgrounds, model, output, batch_size, smoke=False):
    n_per_batch = batch_size // 2
    partial = output / "prediction_batches"
    partial.mkdir(exist_ok=True)
    paths, preview_cases = [], []
    for start in range(0, len(cohort), n_per_batch):
        block = list(cohort.iloc[start:start+n_per_batch].itertuples(index=False))
        path = partial / f"{start:05}.csv"
        paths.append(path)
        if path.is_file() and not smoke:
            check = pd.read_csv(path)
            keys = [(r.source_image_id, name) for r in block for name in SCENE_PAIRS[r.scene_pair]]
            if list(zip(check.source_image_id, check.background_id)) != keys:
                raise ValueError(f"Restart cache differs: {path}")
            continue
        pairs = [build_pair(row, foregrounds, backgrounds) for row in block]
        images = [entry[0] for pair in pairs for entry in pair]
        predictions = model.predict_batch(images, batch_size=batch_size)
        records = []
        for i, (row, pair) in enumerate(zip(block, pairs)):
            scores = predictions[2*i:2*i+2]
            for side, ((image, name, alpha, layer, threshold), pred) in enumerate(zip(pair, scores)):
                records.append(dict(source_image_id=row.source_image_id, species_id=row.species_id, true_y=row.true_y,
                                    original_group=row.original_group, original_pred=row.pred_y, scene_pair=row.scene_pair,
                                    background_type=("land", "water")[side], background_id=name, pred_y=pred["pred_y"],
                                    p_waterbird=pred["probabilities"][1], logit_margin=pred["logits"][1]-pred["logits"][0],
                                    alpha_sha256=alpha, foreground_layer_sha256=layer, mask_min_coverage=threshold,
                                    raw_input_sha256=pixel_hash(image),
                                    model_input_sha256=pred["model_input_sha256"]))
            if smoke:
                preview_cases.append((row, scores))
        tmp = path.with_suffix(".tmp")
        pd.DataFrame(records).to_csv(tmp, index=False)
        tmp.replace(path)
        if start // n_per_batch % 20 == 0:
            print(f"{start + len(block):,}/{len(cohort):,} birds scored", flush=True)
    result = pd.concat([pd.read_csv(path, float_precision="round_trip") for path in paths], ignore_index=True)
    if smoke:
        contact_sheet(preview_cases, output / "preflight.png")
    return result


def contact_sheet(examples, path):
    tile, vertical = 132, 172
    canvas = Image.new("RGB", (4*(2*tile+10), 24+vertical*((len(examples)+3)//4)), "#fafaf5")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=12)
    for i, (row, pair) in enumerate(examples):
        x, y = i % 4 * (2*tile+10), 14+i//4*vertical
        draw.text((x+3,y), f"#{row.source_image_id} | {LABELS[row.true_y]}", fill="#192920", font=font)
        for j, pred in enumerate(pair):
            left = x+j*tile
            view = pred["preview"].resize((tile,tile), Image.Resampling.BILINEAR)
            canvas.paste(view, (left, y+18))
            draw.text((left+3,y+152), f"{('LAND', 'WATER')[j]}  {LABELS[pred['pred_y']]}", fill="#192920", font=font)
    canvas.save(path)


def summarize(predictions, cohort, output):
    data = predictions.copy()
    data["correct"] = data.pred_y.eq(data.true_y)
    paired = data.pivot(index="source_image_id", columns="background_type", values=["pred_y", "p_waterbird", "correct"])
    if len(data) != 2*len(cohort) or len(paired) != len(cohort) or paired.isna().any().any() or \
            data.groupby("source_image_id")[["alpha_sha256", "foreground_layer_sha256"]].nunique().ne(1).any().any():
        raise ValueError("Incomplete paired prediction set or changed foreground")
    original = cohort.set_index("source_image_id")
    paired_summary = original[["species_id", "true_y", "original_group", "pred_y", "scene_pair"]].rename(columns={"pred_y":"original_pred"}).copy()
    for kind in ("land", "water"):
        paired_summary[f"{kind}_pred"] = paired["pred_y", kind].astype(int)
        paired_summary[f"{kind}_p_waterbird"] = paired["p_waterbird", kind].astype(float)
    paired_summary["delta_p_waterbird"] = paired_summary.water_p_waterbird - paired_summary.land_p_waterbird
    paired_summary["flipped"] = paired_summary.land_pred.ne(paired_summary.water_pred)
    swapped = np.where(paired_summary.original_group % 2 == 0, paired_summary.water_pred, paired_summary.land_pred)
    original_correct = paired_summary.original_pred.eq(paired_summary.true_y)
    swapped_correct = swapped == paired_summary.true_y
    paired_summary["opposite_background_pred"] = swapped
    paired_summary["original_wrong_to_opposite_correct"] = ~original_correct & swapped_correct
    paired_summary["original_correct_to_opposite_wrong"] = original_correct & ~swapped_correct
    paired_summary.to_csv(output / "paired_cases.csv")
    original_groups = []
    for group, rows in original.groupby("original_group"):
        original_groups.append(dict(true_y=int(group)//2, background_type=("land", "water")[int(group)%2],
                                    correct=int(rows.pred_y.eq(rows.true_y).sum()), n=len(rows)))
    group_table = data.groupby(["true_y", "background_type"], as_index=False).correct.agg(correct="sum", n="size")
    group_table["accuracy"] = group_table.correct / group_table.n
    group_table.to_csv(output / "group_metrics.csv", index=False)
    by_background = data.groupby("background_type").correct.agg(correct="sum", n="size", accuracy="mean")
    by_scene = data.groupby(["background_id", "true_y"]).correct.agg(correct="sum", n="size", accuracy="mean")
    by_scene.to_csv(output / "scene_metrics.csv")
    strata = paired_summary.groupby("true_y").agg(n=("flipped", "size"), flips=("flipped", "sum"), mean_delta_p=("delta_p_waterbird", "mean"))
    report = dict(sources=len(cohort), composite_inputs=len(data), composite_accuracy=float(data.correct.mean()), composite_wga=float(group_table.accuracy.min()),
                  original_accuracy=float(original_correct.mean()), original_wga=min(g["correct"]/g["n"] for g in original_groups),
                  accuracy_by_treatment=by_background.to_dict("index"), paired_flips=int(paired_summary.flipped.sum()),
                  land_to_water=int((paired_summary.land_pred.eq(0) & paired_summary.water_pred.eq(1)).sum()),
                  water_to_land=int((paired_summary.land_pred.eq(1) & paired_summary.water_pred.eq(0)).sum()),
                  flips_by_class=strata.to_dict("index"), opposite_background_correct=int(swapped_correct.sum()),
                  low_coverage_mask_sources=int(data.loc[data.mask_min_coverage.lt(MASK_MIN_COVERAGE), "source_image_id"].nunique()),
                  original_wrong_to_opposite_correct=int(paired_summary.original_wrong_to_opposite_correct.sum()),
                  original_correct_to_opposite_wrong=int(paired_summary.original_correct_to_opposite_wrong.sum()))
    data.to_csv(output / "predictions.csv", index=False)
    write_json(output / "summary.json", report)
    print(json.dumps(report, indent=2))


def run(source, output, device, batch_size, smoke):
    started = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    cohort, foregrounds, checkpoint, original_hash = load_inputs(source)
    backgrounds, bank = load_backgrounds(source, foregrounds.root)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = ModelAdapter(checkpoint["checkpoint"], device, metadata=foregrounds.root / "metadata.csv")
    if model.checkpoint_sha256 != checkpoint["checkpoint_sha256"] or model.epoch != 258:
        raise ValueError("Expected the fixed POTER seed 1 checkpoint")
    config = dict(seed=SEED, batch_size=batch_size, smoke=smoke, masks="verified_original_cub", sigma=0,
                  mask_threshold=MASK_MIN_COVERAGE, low_coverage_mask_policy="retain maximum native coverage when below threshold",
                  scene_pairs=SCENE_PAIRS, backgrounds=bank, model=model.info(), original_predictions_sha256=original_hash,
                  compositor_sha256=sha256(ROOT / "src/counterfactual/compositor.py"), script_sha256=sha256(__file__))
    config_path = output / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != json.loads(json.dumps(config)):
        raise ValueError("Output directory already uses a different run configuration")
    write_json(config_path, config)
    if smoke:
        cohort = cohort.groupby(["original_group", "scene_pair"], group_keys=False).sample(n=2, random_state=SEED)
    result = score(cohort, foregrounds, backgrounds, model, output, batch_size, smoke)
    summarize(result, cohort, output)
    print(f"Completed in {perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 2 or args.batch_size % 2:
        parser.error("--batch-size must be an even integer >= 2")
    run(args.source_root.resolve(), args.output.resolve(), args.device, args.batch_size, args.smoke)
