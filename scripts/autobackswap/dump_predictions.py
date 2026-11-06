"""Dump per-example test errors for a trained Waterbirds checkpoint.

Standalone: does NOT modify run.py or method code. Run it from anywhere; it
chdirs into the spur-corr checkout (the repo reads ./data relative to it).
Images are never copied or uploaded: every output refers to images by their
path relative to <repo>/data/waterbird_complete95_forest2water2/images.

Outputs in <out_dir>/<run_label>/:
  errors.csv   every misclassified test example (all of them, not capped)
  index.md     at most --max_per_group errors per group (highest confidence first)
  index.html   same selection; <img src> points at "images/<path>", so it only
               shows pictures if you symlink the dataset's images/ next to it
  summary.md   counts per group, check against test_accuracies.txt
"""
import argparse
import csv
import html
import os
import sys

GROUP_NAMES = {
    1: "Landbird on Land",
    2: "Landbird on Water",
    3: "Waterbird on Land",
    4: "Waterbird on Water",
}
CLASS_NAMES = ("landbird", "waterbird")
PLACE_NAMES = ("land", "water")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="path to spur-corr checkout")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--run_label", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_dir", default="artifacts/misclassified")
    p.add_argument("--max_per_group", type=int, default=50)
    p.add_argument("--reported_wga", type=float, default=None,
                   help="Worst Group Accuracy (%%) from test_accuracies.txt, for the check")
    p.add_argument("--reported_overall", type=float, default=None,
                   help="Overall Accuracy (%%) from test_accuracies.txt, for the check")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.abspath(os.path.join(args.out_dir, args.run_label))
    checkpoint = os.path.abspath(args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)

    os.chdir(args.repo)
    sys.path.insert(0, os.getcwd())
    import torch
    from torch.utils.data import DataLoader

    from utils import Config, get_test_transform
    from methods.common.model import Classifier
    from methods.common.waterbird import WaterBird

    # same config file the run used; only the settings the run overrode are overridden here
    config = Config("./methods/ours/config_wb.yaml")
    config.device = args.device
    config.random_seed = args.seed
    config.waterbirds_use_minority = False
    config.cuda_optimizations = False

    model = Classifier(config).to(config.device)
    state = torch.load(checkpoint, map_location=config.device, weights_only=True)
    state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    dataset = WaterBird(
        root="./data", split="test", groups=(1, 2, 3, 4),
        transform=get_test_transform(config),
    )
    meta = dataset.metadata.reset_index(drop=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    preds, confs = [], []
    with torch.no_grad():
        for images, _ in loader:
            probs = torch.softmax(model(images.to(config.device)).float(), dim=1).cpu()
            conf, pred = probs.max(dim=1)
            preds += pred.tolist()
            confs += conf.tolist()
    assert len(preds) == len(meta), (len(preds), len(meta))

    rows, per_group = [], {g: [0, 0] for g in GROUP_NAMES}  # group -> [correct, total]
    for i, row in meta.iterrows():
        g = int(row.group)
        per_group[g][1] += 1
        if preds[i] == int(row.y):
            per_group[g][0] += 1
            continue
        rows.append({
            "image_path": row.img_filename,
            "true_label": CLASS_NAMES[int(row.y)],
            "predicted_label": CLASS_NAMES[preds[i]],
            "group": g,
            "group_name": GROUP_NAMES[g],
            "spurious_attribute": PLACE_NAMES[int(row.place)],
            "confidence": round(confs[i], 4),
            "run_label": args.run_label,
            "seed": args.seed,
        })

    fields = list(rows[0].keys()) if rows else []
    with open(os.path.join(out_dir, "errors.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["group"], -r["confidence"])))

    # index: at most max_per_group per group, highest-confidence errors first
    selected = {}
    for g in GROUP_NAMES:
        errs = [r for r in rows if r["group"] == g]
        errs.sort(key=lambda r: -r["confidence"])
        selected[g] = errs[: args.max_per_group]

    with open(os.path.join(out_dir, "index.md"), "w") as f:
        f.write(f"# Misclassified test examples: {args.run_label} (seed {args.seed})\n\n")
        f.write(f"At most {args.max_per_group} per group, highest confidence first. "
                "Paths are relative to `data/waterbird_complete95_forest2water2/images/`. "
                "No images are included.\n")
        for g, name in GROUP_NAMES.items():
            n_err = sum(1 for r in rows if r["group"] == g)
            f.write(f"\n## {name} (group {g}): {n_err} errors of {per_group[g][1]}, "
                    f"showing {len(selected[g])}\n\n")
            f.write("| image_path | true | predicted | confidence |\n|---|---|---|---|\n")
            for r in selected[g]:
                f.write(f"| {r['image_path']} | {r['true_label']} | "
                        f"{r['predicted_label']} | {r['confidence']} |\n")

    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write("<!doctype html><meta charset='utf-8'><title>Misclassified examples</title>\n")
        f.write("<style>body{font-family:sans-serif}figure{display:inline-block;margin:6px;"
                "width:160px;font-size:11px}img{width:160px}</style>\n")
        f.write(f"<h1>{html.escape(args.run_label)} (seed {args.seed})</h1>\n")
        f.write("<p>Images load from <code>images/&lt;path&gt;</code>; symlink the dataset's "
                "<code>images/</code> folder next to this file to see them.</p>\n")
        for g, name in GROUP_NAMES.items():
            f.write(f"<h2>{name}: {len(selected[g])} shown</h2>\n")
            for r in selected[g]:
                p = html.escape(r["image_path"])
                f.write(f"<figure><img loading='lazy' src='images/{p}'>"
                        f"<figcaption>true {r['true_label']}, pred {r['predicted_label']}, "
                        f"conf {r['confidence']}</figcaption></figure>\n")

    accs = {g: per_group[g][0] / per_group[g][1] for g in per_group}
    total = sum(v[1] for v in per_group.values())
    overall = sum(v[0] for v in per_group.values()) / total
    wga = min(accs.values())
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(f"# Summary: {args.run_label} (seed {args.seed})\n\n")
        f.write(f"Checkpoint: `{os.path.basename(checkpoint)}` (best val_acc checkpoint)\n\n")
        f.write("| group | test size | errors | accuracy % |\n|---|---|---|---|\n")
        for g, name in GROUP_NAMES.items():
            f.write(f"| {name} | {per_group[g][1]} | {per_group[g][1] - per_group[g][0]} "
                    f"| {accs[g] * 100:.3f} |\n")
        f.write(f"\nWorst group accuracy {wga * 100:.3f} %, overall accuracy {overall * 100:.3f} % "
                f"({len(rows)} errors of {total}).\n")
        if args.reported_wga is not None and args.reported_overall is not None:
            ok = (abs(wga * 100 - args.reported_wga) < 0.01
                  and abs(overall * 100 - args.reported_overall) < 0.01)
            f.write(f"\nCheck against test_accuracies.txt (WGA {args.reported_wga}, "
                    f"overall {args.reported_overall}): {'matches' if ok else 'DOES NOT MATCH'}.\n")
        f.write("\nErrors with a high-confidence wrong prediction are listed first in index.md; "
                "no per-example cause is claimed.\n")

    print(f"wrote {out_dir}")
    print({GROUP_NAMES[g]: round(accs[g] * 100, 3) for g in accs},
          "WGA", round(wga * 100, 3), "overall", round(overall * 100, 3), "errors", len(rows))


if __name__ == "__main__":
    main()
