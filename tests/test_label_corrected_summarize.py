import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "summarize", ROOT/"experiments/dpe_label_corrected_backbones/summarize.py")
summarize = importlib.util.module_from_spec(spec); spec.loader.exec_module(summarize)


def fake_test_set():
    files, y, a = [], [], []
    counts = {(0, 0): 2255, (0, 1): 2255, (1, 0): 642, (1, 1): 642}
    for (yy, aa), n in counts.items():
        for i in range(n):
            if (yy, aa) == (1, 0) and i < 43:
                name = f"021.Eastern_Towhee/img{i}.jpg"
            elif (yy, aa) == (1, 1) and i < 47:
                name = f"088.Western_Meadowlark/img{i}.jpg"
            else:
                name = f"999.Other/img{yy}{aa}{i}.jpg"
            files.append("/data/waterbirds/x/" + name); y.append(yy); a.append(aa)
    return np.array(files), np.array(y), np.array(a)


def write_job(d, name, seed, bb, val, head, pred_landbird_on_affected):
    files, y_off, a = fake_test_set()
    mask = summarize.affected(files)
    y = y_off.copy()
    if val == "corrected":
        y[mask] = 0
    pred = y_off.copy()
    pred[mask] = 0 if pred_landbird_on_affected else 1
    p = np.stack([1-pred, pred], 1).astype(float)
    job = d/name; job.mkdir()
    np.savez(job/"test_predictions.npz", probabilities=p, y=y,
             groups=np.array([f"{yy}{aa}" for yy, aa in zip(y, a)]), filenames=files)
    (job/"job.json").write_text(json.dumps(dict(name=name, backbone_seed=seed, backbone_labels=bb,
                                                val_labels=val, head_seed=head)))


def test_scoring_both_label_sets(tmp_path):
    write_job(tmp_path, "a", 5, "official", "official", 5, pred_landbird_on_affected=False)
    write_job(tmp_path, "b", 5, "official", "corrected", 5, pred_landbird_on_affected=True)
    r_off = summarize.score_job(tmp_path/"a")
    r_cor = summarize.score_job(tmp_path/"b")
    # Predicting waterbird on the affected images is perfect under official labels ...
    assert r_off["official"]["wga"] == 100 and r_off["corrected"]["wga"] < 100
    # ... predicting landbird is perfect under corrected labels.
    assert r_cor["corrected"]["wga"] == 100 and r_cor["official"]["wga"] < 100
    assert r_cor["affected_predicted_landbird"] == 90


def test_report_contrast(tmp_path):
    for s in (5, 6):
        write_job(tmp_path, f"a{s}", s, "official", "official", s, False)
        write_job(tmp_path, f"b{s}", s, "official", "corrected", s, True)
    text = summarize.summarize(tmp_path, tmp_path/"out")
    assert "primary: fix validation labels" in text
    line = next(l for l in text.splitlines() if l.startswith("| primary") and "s5" in l)
    assert "+" in line.split("|")[2]  # positive on corrected test labels (first section)
    assert (tmp_path/"out"/"summary.csv").exists()
