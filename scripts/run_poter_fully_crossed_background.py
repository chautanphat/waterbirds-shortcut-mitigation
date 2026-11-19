"""Audit, preflight and score a fixed 160 by 40 POTER test intervention grid."""
import argparse
import itertools
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.optimize import linear_sum_assignment
import torch

from run_poter_test_background_swap import ROOT, load_inputs
from src.counterfactual.common import contained, pixel_hash, read_csv, sha256, write_csv, write_json
from src.counterfactual.compositor import composite, foreground_layer, mask_overlay
from src.counterfactual.model import ModelAdapter

CONFIG = ROOT / "configs/poter_fully_crossed_background.json"
OUTPUT = ROOT / "outputs/poter_fully_crossed_background"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def freeze(path, records):
    current = pd.DataFrame(records).to_csv(index=False, lineterminator="\n").encode("utf-8")
    if path.exists():
        require(path.read_bytes() == current, f"Locked data changed: {path}")
    else:
        path.write_bytes(current)


def fingerprint(image):
    thumbnail = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    return int.from_bytes(np.packbits(thumbnail[:, 1:] > thumbnail[:, :-1]).tobytes(), "big")


def select_backgrounds(output, source, settings, metadata):
    candidate_root = output / "assets/candidates"
    candidates = {int(r["source_bird_id"]): r for r in read_csv(candidate_root / "background_candidates.csv")}
    prior = read_csv(source / "outputs/vulnerability_ot_incremental/background_banks.csv")
    old_ids, old_places = {r["source_bird_id"] for r in prior}, {r["place_filename"] for r in prior}
    used_places = {r["place_filename"] for r in metadata if r["split"] != "0"}
    train = {r["img_id"]: r for r in metadata if r["split"] == "0"}
    entries, images = [], {}
    properties = ("brightness", "saturation", "mean_red", "mean_green", "mean_blue")
    for bank, category in settings["banks"].items():
        for kind, sources in category.items():
            require(len(sources) == 10, "Expected ten backgrounds per bank/type")
            for sid in sources:
                item = dict(candidates[sid])
                ident = item["background_id"]
                path = contained(candidate_root, ident + ".png")
                place = item["place_filename"]
                require(ident not in images and str(sid) not in old_ids and place not in old_places | used_places,
                        f"Previously used or overlapping background: {ident}")
                require(item["source_split"] == "train" and item["background_type"] == kind
                        and train[str(sid)]["place_filename"] == place
                        and int(train[str(sid)]["place"]) == (kind == "water"), f"Background provenance differs: {ident}")
                require(sha256(path) == item["sha256"], f"Background bytes differ: {ident}")
                with Image.open(path) as original:
                    rgb = original.convert("RGB")
                require(rgb.size == (224, 224) and pixel_hash(rgb) == item["pixel_sha256"], f"Background crop differs: {ident}")
                pixels = np.asarray(rgb.resize((256, 256), Image.Resampling.BILINEAR)).astype(np.float64) / 255.
                means = pixels.mean(axis=(0, 1))
                brightness = float(means @ np.array([.2126, .7152, .0722]))
                vmax, vmin = pixels.max(axis=2), pixels.min(axis=2)
                saturation = float(np.mean((vmax - vmin) / np.maximum(vmax, 1e-9)))
                data = {k: item[k] for k in ("background_id", "source_bird_id", "source_split", "place_filename", "scene",
                        "background_type", "crop_xyxy", "exclusion_margin_pixels", "source_image_sha256", "source_mask_sha256", "pixel_sha256")}
                entries.append(dict(**data, background_bank=bank, asset_path=str(path.relative_to(output)), background_sha256=item["sha256"],
                                    native_width=224, native_height=224, brightness=brightness, saturation=saturation,
                                    mean_red=float(means[0]), mean_green=float(means[1]), mean_blue=float(means[2]),
                                    dhash=f"{fingerprint(rgb):016x}", mirrored_dhash=f"{fingerprint(ImageOps.mirror(rgb)):016x}",
                                    source_seen_during_training=True, previously_scored_background=False,
                                    source_dataset="Places365 scene crop from Waterbirds train", visual_review="accepted_before_inference"))
                images[ident] = rgb.resize((256, 256), Image.Resampling.BILINEAR)
    require(len(entries) == 40 and len({r["place_filename"] for r in entries}) == 40
            and len({r["pixel_sha256"] for r in entries}) == 40, "Background bank contains duplicate assets")
    distance = min((int(a["dhash"], 16) ^ int(b[h], 16)).bit_count()
                   for a, b in itertools.combinations(entries, 2) for h in ("dhash", "mirrored_dhash"))
    require(distance > 6, f"Background dHash near-duplicate needs review (distance {distance})")
    freeze(output / "background_manifest.csv", entries)

    # Secondary color-matched subset assigned using scene properties only, before predictions.
    frame = pd.DataFrame(entries).set_index("background_id")
    features = frame[list(properties)]
    features = (features - features.mean()) / features.std(ddof=0)
    matches = []
    for name, bank in frame.groupby("background_bank"):
        land, water = [features.loc[bank.index[bank.background_type == kind]] for kind in ("land", "water")]
        cost = np.linalg.norm(land.to_numpy()[:, None] - water.to_numpy()[None, :], axis=2)
        li, wi = linear_sum_assignment(cost)
        pairs = sorted((float(cost[i, j]), land.index[i], water.index[j]) for i, j in zip(li, wi))[:5]
        matches.extend(dict(bank=name, cost=dist, land_id=l, water_id=w) for dist, l, w in pairs)
    freeze(output / "matched_background_subset.csv", matches)
    return entries, images, distance


def prepare_layer(rgba, settings):
    alpha = rgba.getchannel("A")
    require(alpha.getextrema()[1] >= settings["mask_min_coverage"], "Mask lacks native high-coverage foreground")
    rgba.putalpha(alpha.point(lambda x: 255 if x >= settings["mask_min_coverage"] else 0))
    layer = foreground_layer(rgba, sigma=0)
    array = np.asarray(layer.getchannel("A"))
    crop = array[16:240, 16:240]
    coverage = float((crop > 0).mean())
    retention = float(crop.sum() / array.sum())
    require(settings["minimum_visible_alpha"] <= coverage < .98 and retention >= settings["minimum_crop_retention"],
            f"Mask outside locked coverage/retention bounds: {coverage:.4f}/{retention:.4f}")
    boundary = float(((crop > 0) & (crop < 255)).sum() / (crop > 0).sum())
    return layer, coverage, retention, boundary


def candidate_order(group, rng):
    species = sorted({r.species_id for r in group})
    by_species = {s: [r for r in group if r.species_id == s] for s in species}
    for entries in by_species.values():
        rng.shuffle(entries)
    for depth in range(max(map(len, by_species.values()))):
        for name in rng.permutation(species):
            if depth < len(by_species[name]):
                yield by_species[name][depth]


def select_foregrounds(cohort, repository, output, settings):
    rng = np.random.default_rng(settings["cohort_seed"])
    rows, layers, sources, rejected = [], {}, {}, []
    used_hashes = set()
    for group, data in cohort.groupby("original_group"):
        accepted = 0
        for item in candidate_order(list(data.itertuples(index=False)), rng):
            rgba, meta = repository.get(item.foreground_id)
            require(meta["quality_status"] == "verified_original_cub" and meta["original_split"] == "test",
                    f"Unverified foreground {item.source_image_id}")
            try:
                layer, area, retention, edge = prepare_layer(rgba, settings)
                require(meta["original_cub_sha256"] not in used_hashes, "Duplicate native RGB asset")
            except ValueError as exc:
                rejected.append(dict(source_image_id=item.source_image_id, group=int(group), reason=str(exc)))
                continue
            sid = item.source_image_id
            used_hashes.add(meta["original_cub_sha256"])
            layers[sid], sources[sid] = layer, item
            rows.append(dict(source_image_id=sid, species_id=item.species_id, true_y=item.true_y, original_group=int(group),
                             foreground_split="test", source_seen_during_training=False, source_used_for_reference=False,
                             source_used_for_checkpoint_selection=False, prior_test_exposure=True,
                             foreground_path=meta["rgba_path"], foreground_sha256=meta["rgba_sha256"], mask_path=meta["mask_path"],
                             mask_sha256=meta["mask_sha256"], alpha_sha256=pixel_hash(layer.getchannel("A")),
                             foreground_layer_sha256=pixel_hash(layer), mask_area_fraction=area,
                             crop_retention=retention, soft_boundary_fraction=edge, cohort_seed=settings["cohort_seed"], pilot=False))
            accepted += 1
            if accepted == settings["foregrounds_per_original_group"]:
                break
        require(accepted == settings["foregrounds_per_original_group"], f"Insufficient qualified group {group} foregrounds")
    pilot_rng = np.random.default_rng(settings["pilot_seed"])
    for group in range(4):
        block = [r for r in rows if r["original_group"] == group]
        # Include the smallest/largest masks, then use a seeded selection among the rest.
        block.sort(key=lambda r: (r["mask_area_fraction"], r["source_image_id"]))
        to_review = [block[0], block[-1], *[block[i] for i in pilot_rng.choice(np.arange(1, len(block)-1),
                                                                                        settings["pilot_per_original_group"]-2, replace=False)]]
        for row in to_review:
            row["pilot"] = True
    freeze(output / "foreground_manifest.csv", rows)
    freeze(output / "foreground_rejections.csv", rejected)
    return rows, layers, sources


def preflight(output, foregrounds, layers, sources, entries, images, runner, repository, settings):
    pilot = [r for r in foregrounds if r["pilot"]]
    chosen = [next(r for r in entries if r["background_bank"] == bank and r["background_type"] == kind)
              for bank in "AB" for kind in ("land", "water")]
    chosen += [next(r for r in entries if r["background_bank"] == "A" and r["background_type"] == typ
                    and r["scene"] == scene) for typ,scene in [("land","forest/broadleaf"),("water","ocean")]]
    tile, height = 116, 135
    sheet = Image.new("RGB", (9*tile+14, 28+len(pilot)*height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=11)
    headers = ["Original", "Mask overlay", *[f"{r['background_bank']} {r['background_type']} #{r['source_bird_id']}" for r in chosen]]
    for x, header in enumerate(headers):
        draw.text((tile+8+x*tile, 8), header, fill="black", font=font)
    originals = []
    for i, item in enumerate(pilot):
        sid = item["source_image_id"]
        old = sources[sid]
        original = contained(repository.root, old.source_path)
        require(sha256(original) == old.source_sha256, f"Original bytes differ for {sid}")
        with Image.open(original) as photo:
            rgb = photo.convert("RGB")
        require(pixel_hash(rgb) == old.raw_input_sha256, f"Original raw RGB differs for {sid}")
        originals.append(rgb)
        layer = layers[sid]
        views = [runner.preview(rgb), mask_overlay(layer).convert("RGB").crop((16, 16, 240, 240))]
        for background in chosen:
            composed, audit = composite(layer, images[background["background_id"]])
            require(audit["alpha_sha256"] == item["alpha_sha256"], f"Preflight alpha changed: {sid}")
            views.append(runner.preview(composed))
        y = 27+i*height
        draw.text((5,y+4),f"G{item['original_group']} #{sid}", fill="black", font=font)
        draw.text((5,y+24),f"Area {item['mask_area_fraction']:.1%}", fill="black", font=font)
        for j, view in enumerate(views):
            sheet.paste(view.resize((tile-4, tile-4), Image.Resampling.BILINEAR), (tile+8+j*tile,y))
    sheet.save(output / "mask_contact_sheet.png")

    fresh = runner.predict_batch(originals, batch_size=len(pilot))
    parity = []
    for item, result in zip(pilot, fresh):
        old = sources[item["source_image_id"]]
        require(result["model_input_sha256"] == old.model_input_sha256 and result["pred_y"] == old.pred_y,
                f"Original model view or argmax differs for {old.source_image_id}")
        parity.append(dict(source_image_id=old.source_image_id,
                           p_abs_diff=abs(result["probabilities"][1]-old.p_waterbird),
                           logit_max_abs_diff=max(abs(f-s) for f,s in zip(result["logits"], (old.logit_landbird,old.logit_waterbird)))))
    freeze(output / "original_parity.csv", parity)
    pmax, lmax = max(r["p_abs_diff"] for r in parity), max(r["logit_max_abs_diff"] for r in parity)
    require(pmax <= settings["original_probability_parity_atol"] and lmax <= settings["original_logit_parity_atol"], "BLOCKED_MASK_OR_MODEL_PROVENANCE: parity tolerance exceeded")
    write_json(output / "pixel_integrity_preflight.json", dict(parity_sources=len(parity), p_max_abs_difference=pmax,
               logit_max_abs_difference=lmax, contact_sheet_sha256=sha256(output / "mask_contact_sheet.png"),
               manifest_sha256={name:sha256(output/name) for name in ("background_manifest.csv","foreground_manifest.csv")}))
    print(f"24-foreground preflight completed. Max probability difference {pmax:.3g}, max logit difference {lmax:.3g}.")


def score(output, foregrounds, layers, entries, images, model, stage):
    cache = output / "per_foreground_cache"
    cache.mkdir(exist_ok=True)
    selected = [r for r in foregrounds if stage != "pilot" or r["pilot"]]
    files, hits = [], 0
    expected = [r["background_id"] for r in entries]
    for k, row in enumerate(selected, 1):
        sid = row["source_image_id"]
        filepath = cache / f"{sid}.csv"
        files.append(filepath)
        if filepath.exists():
            cached = pd.read_csv(filepath)
            require(cached.background_id.tolist() == expected and cached.source_image_id.eq(sid).all()
                    and cached.alpha_sha256.eq(row["alpha_sha256"]).all() and cached.checkpoint_sha256.eq(model.checkpoint_sha256).all(),
                    f"Cached records changed: {sid}")
            hits += len(entries)
            continue
        layer = layers[sid]
        views = []
        for bg in entries:
            im, audit = composite(layer, images[bg["background_id"]])
            require(audit["alpha_sha256"] == row["alpha_sha256"] and
                    audit["foreground_layer_sha256"] == row["foreground_layer_sha256"], f"Foreground geometry changed: {sid}")
            views.append(im)
        outputs = model.predict_batch(views, batch_size=40)
        require(len(outputs) == 40 and len({p["model_input_sha256"] for p in outputs}) == 40,
                f"Missing or duplicate model-view raster: {sid}")
        records = []
        for bg, im, out in zip(entries, views, outputs):
            lo, hi = out["logits"]
            record = {key: row[key] for key in ("source_image_id", "species_id", "foreground_split", "source_seen_during_training",
                       "source_used_for_reference", "original_group", "true_y", "foreground_path", "foreground_sha256", "mask_sha256",
                       "alpha_sha256", "foreground_layer_sha256", "mask_area_fraction")}
            record.update(model_id="POTER", checkpoint_sha256=model.checkpoint_sha256,
                          **{key:bg[key] for key in ("background_id","background_type","background_bank","background_sha256")},
                          composite_sha256=pixel_hash(im), model_input_sha256=out["model_input_sha256"], logit_landbird=lo,
                          logit_waterbird=hi, margin_water_minus_land=hi-lo, p_waterbird=out["probabilities"][1],
                          pred_y=out["pred_y"], correct=out["pred_y"] == row["true_y"])
            records.append(record)
        tmp = filepath.with_suffix(".tmp")
        pd.DataFrame(records).to_csv(tmp, index=False)
        tmp.replace(filepath)
        if k % 20 == 0:
            print(f"{stage}: {k}/{len(selected)} birds", flush=True)
    frame = pd.concat([pd.read_csv(f, float_precision="round_trip") for f in files], ignore_index=True)
    require(len(frame) == len(selected)*40 and not frame.duplicated(["model_id","source_image_id","background_id"]).any(), "Grid incomplete")
    frame.to_csv(output / ("pilot_predictions.csv" if stage == "pilot" else "fully_crossed_predictions.csv"), index=False)
    return dict(foregrounds=len(selected), backgrounds=40, rows=len(frame), cache_hits=hits, forwards=len(frame)-hits)


def run(source, output, stage, device):
    started = perf_counter()
    settings = json.loads(CONFIG.read_text(encoding="utf-8"))
    cohort, repository, checkpoint, original_hash = load_inputs(source)
    metadata = read_csv(repository.root / "metadata.csv")
    entries, images, distance = select_backgrounds(output, source, settings, metadata)
    foregrounds, layers, sources = select_foregrounds(cohort, repository, output, settings)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    runner = ModelAdapter(checkpoint["checkpoint"], device, metadata=repository.root / "metadata.csv")
    require(runner.epoch == 258 and runner.checkpoint_sha256 == checkpoint["checkpoint_sha256"], "Historical checkpoint differs")
    stamp = dict(settings_sha256=sha256(CONFIG), runner_sha256=sha256(__file__), foreground_manifest_sha256=sha256(output/"foreground_manifest.csv"),
                 background_manifest_sha256=sha256(output/"background_manifest.csv"), compositor_sha256=sha256(ROOT/"src/counterfactual/compositor.py"),
                 original_predictions_sha256=original_hash, scene_min_dhash_distance=distance, model=runner.info())
    path = output / "run_lock.json"
    if path.is_file():
        require(json.loads(path.read_text(encoding="utf-8")) == json.loads(json.dumps(stamp)), "Run parameters changed")
    else:
        write_json(path, stamp)
    if stage == "preflight":
        preflight(output, foregrounds, layers, sources, entries, images, runner, repository, settings)
    else:
        decisions = json.loads((output/"pixel_integrity_preflight.json").read_text(encoding="utf-8"))
        review = json.loads((output/"preflight_review.json").read_text(encoding="utf-8"))
        require(review["status"] == "PASS" and review["contact_sheet_sha256"] == decisions["contact_sheet_sha256"]
                == sha256(output / "mask_contact_sheet.png"), "Audited visual preflight is required")
        if stage == "full":
            pilot = json.loads((output/"pilot_inference.json").read_text(encoding="utf-8"))
            require(pilot["rows"] == 960 and pilot["foregrounds"] == 24, "Completed 24-bird pilot required")
        stats = score(output, foregrounds, layers, entries, images, runner, stage)
        write_json(output/f"{stage}_inference.json", dict(**stats, wall_seconds=perf_counter()-started))
        print(stats, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "pilot", "full"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run(args.source_root.resolve(), args.output.resolve(), args.stage, args.device)
