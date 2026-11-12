"""Write predictions.csv for a Waterbirds checkpoint: every test example, correct and wrong.

Emits the shared predictions.csv v1 format (docs/prediction_protocol.md).
Standalone, does not touch run.py or method code. Images are never copied or uploaded;
paths are relative to <repo>/data/waterbird_complete95_forest2water2/images.
errors.csv for a run is the rows of predictions.csv with correct == 0.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.prediction_export import sha256, write_predictions

GN = {1: "Landbird on Land", 2: "Landbird on Water", 3: "Waterbird on Land", 4: "Waterbird on Water"}
CN = ("landbird", "waterbird")
p = argparse.ArgumentParser()
p.add_argument("--repo", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--run_label", required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--device", default="cuda:0")
p.add_argument("--out_dir", required=True)
a = p.parse_args()
out = os.path.abspath(os.path.join(a.out_dir, a.run_label))
ckpt = os.path.abspath(a.checkpoint)
if os.path.exists(out):
    raise FileExistsError(f"Preserve the existing export: {out}")
os.chdir(a.repo)
sys.path.insert(0, os.getcwd())
import torch
from torch.utils.data import DataLoader
from utils import Config, get_test_transform
from methods.common.model import Classifier
from methods.common.waterbird import WaterBird

cfg = Config("./methods/ours/config_wb.yaml")
cfg.device = a.device
cfg.random_seed = a.seed
cfg.cuda_optimizations = False
model = Classifier(cfg).to(cfg.device)
st = torch.load(ckpt, map_location=cfg.device, weights_only=True)
model.load_state_dict({k.replace("_orig_mod.", "", 1): v for k, v in st.items()})
model.eval()
ds = WaterBird(root="./data", split="test", groups=(1, 2, 3, 4), transform=get_test_transform(cfg))
meta = ds.metadata.reset_index(drop=True)
preds, confs = [], []
with torch.no_grad():
    for x, _ in DataLoader(ds, batch_size=64, shuffle=False, num_workers=2):
        c, pr = torch.softmax(model(x.to(cfg.device)).float(), 1).cpu().max(1)
        preds += pr.tolist()
        confs += c.tolist()
assert len(preds) == len(meta)
tot = {g: [0, 0] for g in GN}
rows = []
for i, r in meta.iterrows():
    g = int(r.group)
    tot[g][0] += preds[i] == int(r.y)
    tot[g][1] += 1
    rows.append(dict(image_path=r.img_filename, y=int(r.y), a=int(r.place), split="test"))
write_predictions(out, rows, None, predicted_labels=preds, confidence=confs,
    run_label=a.run_label, seed=a.seed,
    provenance=dict(method="erm-autobackswap" if a.run_label.startswith("erm") else "autobackswap",
        checkpoint=ckpt, checkpoint_sha256=sha256(ckpt),
        metadata_sha256=sha256("./data/waterbird_complete95_forest2water2/metadata.csv"),
        source_config="methods/ours/config_wb.yaml",
        source_config_sha256=sha256("./methods/ours/config_wb.yaml"),
        config_overrides=dict(device=a.device, random_seed=a.seed, cuda_optimizations=False),
        limitation="Legacy helper uses the original default config plus export overrides; retain original training records",
        selection="User-supplied checkpoint; no selection or fitting during export"))
n = sum(v[1] for v in tot.values())
right = sum(v[0] for v in tot.values())
print(a.run_label, {GN[g]: round(100 * v[0] / v[1], 3) for g, v in tot.items()},
      "WGA", round(100 * min(v[0] / v[1] for v in tot.values()), 3),
      "overall", round(100 * right / n, 3), "errors", n - right)
