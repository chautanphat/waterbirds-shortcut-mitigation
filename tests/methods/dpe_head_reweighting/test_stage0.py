"""Stage-0 recipe and handoff guards; no real data or weight downloads."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.methods.dpe_head_reweighting import stage0


def shared_run(tmp_path, *, legacy=False):
    c = stage0.backbone_config(tmp_path / "data", tmp_path / "backbone", workers=0)
    out = Path(c.output_dir)
    out.mkdir()
    raw = c.as_dict()
    if legacy:
        del raw["dataset"]["augmentation"]
    (out / "config.yaml").write_text(yaml.safe_dump(raw))
    (out / "run.json").write_text(json.dumps(dict(
        config=raw, metadata_sha256="metadata-hash", git_commit="test-commit")))
    (out / "summary.json").write_text(json.dumps(dict(updates=3700, selected_checkpoint="ckpt_last.pt")))
    (out / "ckpt_last.pt").write_bytes(b"test checkpoint")
    return c, out


@pytest.mark.parametrize("legacy", [False, True])
def test_completed_backbone_recipe_and_legacy_standard_runs(tmp_path, legacy):
    c, out = shared_run(tmp_path, legacy=legacy)
    assert stage0.completed_backbone(c)["git_commit"] == "test-commit"
    # Moving a run or changing execution workers/device does not change its recipe.
    c.dataset.data_dir = "/other/data"
    c.dataset.workers = 2
    c.training.device = "cuda"
    stage0.completed_backbone(c)
    c.training.seed = 1
    with pytest.raises(ValueError, match="recipe"):
        stage0.completed_backbone(c)
    c.training.seed = 0
    (out / "summary.json").write_text(json.dumps(dict(updates=37, selected_checkpoint="ckpt_last.pt")))
    with pytest.raises(ValueError, match="final-epoch"):
        stage0.completed_backbone(c)
    (out / "summary.json").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        stage0.completed_backbone(c)


def test_handoff_is_atomic_and_rejects_changed_inputs(tmp_path, monkeypatch):
    c, backbone = shared_run(tmp_path)
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("source metadata")
    run_path = backbone / "run.json"
    run = json.loads(run_path.read_text())
    run["metadata_sha256"] = stage0.digest(metadata)
    run_path.write_text(json.dumps(run))
    rows = [dict(id="image1", split=1), dict(id="image2", split=2)]
    monkeypatch.setattr(stage0, "validate_dataset", lambda _: (metadata, rows))
    monkeypatch.setattr(stage0, "load_resnet_checkpoint", lambda *a, **kw: object())
    class Dataset:
        def __init__(self, root, split):
            self.filenames = ["image1" if split == "val" else "image2"]
    monkeypatch.setattr(stage0, "Waterbirds", Dataset)
    monkeypatch.setattr(stage0, "evaluation_loader", lambda data, *a: data)
    calls = []
    def extract(model, data, device):
        calls.append(data.filenames)
        if len(calls) == 2:
            raise RuntimeError("interrupted test extraction")
        return dict(features=np.arange(2048, dtype=np.float32)[None])
    monkeypatch.setattr(stage0, "extract_features", extract)
    output = tmp_path / "features"
    with pytest.raises(RuntimeError, match="interrupted"):
        stage0.export_inputs(c, output)

    assert not output.exists()  # No partly complete bundle remains to be reused.
    inputs = stage0.export_inputs(c, output)
    assert inputs["backbone"] == str(backbone / "ckpt_last.pt")
    assert not (output / "ckpt_last.pt").exists()
    before = len(calls)
    assert stage0.export_inputs(c, output) == inputs
    assert len(calls) == before  # A verified bundle needs no model/feature computation.
    (output / "test_raw.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        stage0.export_inputs(c, output)
    (backbone / "ckpt_last.pt").write_bytes(b"replaced checkpoint")
    with pytest.raises(ValueError, match="inputs or extraction implementation changed"):
        stage0.export_inputs(c, output)


def test_smoke_recipes_are_rejected_before_training(tmp_path):
    c = stage0.backbone_config(tmp_path, tmp_path / "backbone")
    c.training.epochs = 2
    with pytest.raises(ValueError, match="100-epoch"):
        stage0.train_or_reuse(c)
    c.training.epochs = 100
    c.model.weights = None
    with pytest.raises(ValueError, match="ImageNet"):
        stage0.completed_backbone(c)
