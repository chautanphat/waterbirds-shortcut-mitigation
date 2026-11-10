"""CPU export-contract tests. No downloads, datasets or trained checkpoints."""

import csv
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from PIL import Image

from src.evaluation.prediction_export import (align_ids, export_checkpoint, image_path, infer_images,
    load_image_model, read_metadata, resolve_images, sha256, write_predictions)
from src.evaluation.dpe_export import (capture_notebook_heads, export_dpe_heads,
    export_dpe_probabilities, mean_probabilities, package_notebook_features, predict_heads,
    save_dpe_heads, save_dpe_probabilities)

ROOT = Path(__file__).resolve().parents[1]


def fixture(root):
    metadata = root / "metadata.csv"
    with metadata.open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(["img_filename", "y", "place", "split"])
        for split in (0, 1, 2):
            for group in range(4):
                w.writerow([f"bird/{split}_{group}.jpg", group // 2, group % 2, split])
    return metadata


class PredictionExportTests(unittest.TestCase):
    def test_metadata_aliases_and_identity_alignment(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = fixture(root)
            rows = read_metadata(metadata)
            self.assertEqual(len(rows), 4)
            self.assertEqual([r["y"] for r in rows], [0, 0, 1, 1])
            ids = [f"/kaggle/data/images/{r['image_path']}" for r in reversed(rows)]
            np.testing.assert_array_equal(align_ids(ids, rows), [3, 2, 1, 0])
            with self.assertRaises(ValueError):
                align_ids([ids[0]] * 4, rows)
            with self.assertRaises(ValueError):
                align_ids(ids[:3], rows)
            prepared = root / "prepared.csv"
            prepared.write_text(metadata.read_text().replace("img_filename", "filename").replace("place", "a"))
            self.assertEqual(read_metadata(prepared), rows)
            for name in ("../image.jpg", "/absolute.jpg", "https://host/a.jpg", "species//a.jpg"):
                with self.assertRaises(ValueError):
                    image_path(name)

    def test_writer_reorders_probabilities_and_refuses_overwrite(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = read_metadata(fixture(root))
            p = np.array([[.1, .9], [.1, .9], [.8, .2], [.8, .2]])
            output = root / "export"
            report = write_predictions(output, rows, p, ids=[r["image_path"] for r in reversed(rows)],
                                       run_label="DPE", seed=0, provenance={"test": True})
            with (output / "predictions.csv").open(newline="") as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual([r["correct"] for r in records], ["1"] * 4)
            self.assertEqual([r["group"] for r in records], ["1", "2", "3", "4"])
            self.assertEqual(report["metrics"]["worst_group_accuracy"], 1.)
            self.assertEqual(report["csv_sha256"], sha256(output / "predictions.csv"))
            self.assertTrue((output / "provenance.json").is_file())
            with self.assertRaises(FileExistsError):
                write_predictions(output, rows, p, run_label="DPE", seed=0, provenance={})

    def test_invalid_probabilities_leave_no_export(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = read_metadata(fixture(root))
            for p in (np.zeros((4, 2)), np.full((4, 2), np.nan), np.array([[.2, .8]]), np.full((4, 2), -.5)):
                with self.assertRaises(ValueError):
                    write_predictions(root / "bad", rows, p, run_label="test", seed=0, provenance={})
                self.assertFalse((root / "bad").exists())

    def test_original_and_images_subfolder_layouts_and_singleton_batch(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = read_metadata(fixture(root))[:3]
            image_root = root / "waterbird_complete95_forest2water2/images/bird"
            image_root.mkdir(parents=True)
            for row in rows:
                Image.new("RGB", (280, 300), (60, 100, 150)).save(image_root / Path(row["image_path"]).name)
            paths = resolve_images(root, rows)
            model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2))
            p = infer_images(model, paths, batch_size=2)
            self.assertEqual(p.shape, (3, 2))
            np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
            with self.assertRaises(FileNotFoundError):
                resolve_images(root, rows + [{"image_path": "bird/missing.jpg"}])

    def test_poter_wrapped_checkpoints_load_strictly(self):
        class SmallResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 2)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = SmallResNet()
            checkpoint = root / "best_validation.pt"
            torch.save({"model": model.state_dict(), "config": {"seed": 1}, "epoch": 258}, checkpoint)
            with patch("torchvision.models.resnet50", side_effect=lambda **kw: SmallResNet()) as build:
                loaded, config = load_image_model("poter", checkpoint)
                self.assertEqual(config["seed"], 1)
                torch.testing.assert_close(loaded.fc.weight, model.fc.weight)
                self.assertEqual(build.call_args.kwargs, {"weights": None})
                torch.save({"fc.weight": model.fc.weight}, root / "bad.pt")
                with self.assertRaises(RuntimeError):
                    load_image_model("poter", root / "bad.pt")

    def test_poter_checkpoint_to_csv_with_real_image_preprocessing(self):
        class SmallResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 2)
            def forward(self, inputs):
                return self.fc(inputs.mean((2, 3)))
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = fixture(root)
            rows = read_metadata(metadata)
            (root / "images/bird").mkdir(parents=True)
            for row in rows:
                Image.new("RGB", (290, 260), (80, 110, 160)).save(root / "images" / row["image_path"])
            model = SmallResNet()
            with torch.no_grad():
                model.fc.weight.zero_(); model.fc.bias.copy_(torch.tensor([2., -2.]))
            config = {"seed": 1, "metadata_sha256": sha256(metadata)}
            torch.save({"model": model.state_dict(), "config": config}, root / "best.pt")
            with patch("torchvision.models.resnet50", side_effect=lambda **kw: SmallResNet()):
                report = export_checkpoint("poter", root / "best.pt", metadata, root, root / "out",
                                          run_label="poter_test", seed=1, batch_size=3)
                self.assertEqual(report["metrics"]["count"], 4)
                self.assertEqual(report["metrics"]["errors"], 2)
                self.assertEqual(report["metrics"]["accuracy"], .5)
                with (root / "out/predictions.csv").open() as stream:
                    records = list(csv.DictReader(stream))
                self.assertTrue(all(r["predicted_label"] == "landbird" for r in records))
                self.assertAlmostEqual(float(records[0]["confidence"]), .98201379, places=6)
                with self.assertRaisesRegex(ValueError, "seed"):
                    export_checkpoint("poter", root / "best.pt", metadata, root, root / "bad-seed",
                                      run_label="bad", seed=0)

    def test_erm_and_autobackswap_checkpoint_adapters_do_not_download_weights(self):
        class SmallBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 2)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            erm = nn.Sequential(nn.Linear(3, 3), nn.Flatten(), nn.Linear(3, 2))
            torch.save(erm.state_dict(), root / "erm.pt")
            with patch("src.models.resnet.ResNetClassifier", return_value=erm):
                loaded, _ = load_image_model("erm", root / "erm.pt")
                self.assertIs(loaded, erm)
            spec = importlib.util.spec_from_file_location("reference_model", ROOT / "baselines/autobackswap/methods/common/model.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            from types import SimpleNamespace
            with patch("torchvision.models.resnet50", side_effect=lambda **kw: SmallBackbone()) as build:
                original = module.Classifier(SimpleNamespace(dataset_type="waterbird", model_class="resnet"), pre_trained=False)
                torch.save({"_orig_mod." + k: v for k, v in original.state_dict().items()}, root / "auto.pt")
                (root / "config.yaml").write_text("dataset_type: waterbird\nmodel_class: resnet\n")
                loaded, _ = load_image_model("autobackswap", root / "auto.pt", config_path=root / "config.yaml")
                torch.testing.assert_close(loaded.head.weight, original.head.weight)
                self.assertTrue(all(call.kwargs == {"weights": None} for call in build.call_args_list))


class DPEExportTests(unittest.TestCase):
    def test_saved_probabilities_roundtrip_with_metadata_identity_and_seed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = fixture(root)
            rows = read_metadata(metadata)
            p = np.array([[.8, .2], [.7, .3], [.3, .7], [.1, .9]])
            artifact = root / "p.npz"
            save_dpe_probabilities(artifact, p[::-1], [r["image_path"] for r in reversed(rows)], metadata=metadata, seed=0, source="fixture original DPE")
            report = export_dpe_probabilities(artifact, metadata, root / "out", run_label="DPE", seed=0)
            self.assertEqual(report["metrics"]["accuracy"], 1.)
            with self.assertRaises(ValueError):
                export_dpe_probabilities(artifact, metadata, root / "bad", run_label="DPE", seed=1)
            with self.assertRaises(FileExistsError):
                save_dpe_probabilities(artifact, p, [r["image_path"] for r in rows], metadata=metadata, seed=0, source="same")

    def test_original_dpe_matches_upstream_head_softmax_and_keeps_singleton_batch(self):
        spec = importlib.util.spec_from_file_location("reference_isomax", ROOT / "baselines/dpe4subpop/src/dpe/isomaxplus.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        raw = np.array([[3., 2., 1.], [7., 2., 3.], [2., 1., 4.], [1., 4., 2.], [4., 1., 2.]], dtype=np.float32)
        normalized = torch.from_numpy((raw - raw.mean(1, keepdims=True)) / raw.std(1, keepdims=True))
        heads = [(torch.tensor([[[1., 0., 1.]], [[0., 1., 0.]]]), torch.tensor([2.])),
                 (torch.tensor([[[0., 1., 1.]], [[1., 0., 1.]]]), torch.tensor([-1.]))]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = fixture(root)
            backbone = root / "backbone.pt"
            backbone.write_bytes(b"selected backbone")
            save_dpe_heads(root / "heads.pt", heads, metadata=metadata, backbone=backbone, seed=0, source="test")
            record = torch.load(root / "heads.pt", weights_only=True)
            actual = predict_heads(record, raw, batch_size=2)
            expected = []
            for prototypes, scale in heads:
                head = module.IsoMaxPlusLossFirstPart(3, 2)
                with torch.no_grad():
                    head.prototypes.copy_(prototypes[:, 0, :]); head.distance_scale.copy_(scale)
                    expected.append(head(normalized).softmax(1))
            np.testing.assert_allclose(actual, torch.stack(expected).mean(0).numpy(), atol=1e-7)
            with self.assertRaises(ValueError):
                predict_heads(record, np.ones((5, 3), dtype=np.float32))
            with self.assertRaises(ValueError):
                mean_probabilities(np.zeros((2, 5, 2)), 5)

    def test_head_export_verifies_feature_backbone_and_notebook_manifest(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = fixture(root)
            rows = read_metadata(metadata)
            backbone = root / "backbone.pt"
            backbone.write_bytes(b"backbone")
            heads = [(torch.randn(2, 1, 3), torch.ones(1))]
            save_dpe_heads(root / "heads.pt", heads, metadata=metadata, backbone=backbone, seed=0, source="test")
            raw = np.array([[3., 2., 1.], [7., 2., 3.], [2., 1., 4.], [1., 4., 2.]], dtype=np.float32)
            np.save(root / "features.npy", raw)
            manifest = dict(checkpoint_sha256=sha256(backbone), metadata_sha256=sha256(metadata),
                features_sha256=sha256(root / "features.npy"), split="test", image_paths=[r["image_path"] for r in rows],
                format="raw;Resize((256,256));CenterCrop(224);ImageNet normalization")
            (root / "manifest.json").write_text(json.dumps(manifest))
            package_notebook_features(root / "features.npz", root / "features.npy", root / "manifest.json", metadata, backbone)
            report = export_dpe_heads(root / "heads.pt", root / "features.npz", metadata, root / "export", run_label="DPE", seed=0)
            self.assertEqual(report["metrics"]["count"], 4)
            np.savez(root / "bad.npz", ids=np.array([r["image_path"] for r in rows]), features=raw, backbone_sha256="different")
            with self.assertRaises(ValueError):
                export_dpe_heads(root / "heads.pt", root / "bad.npz", metadata, root / "bad-export", run_label="DPE", seed=0)
            manifest["features_sha256"] = "changed"
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                package_notebook_features(root / "repacked.npz", root / "features.npy", root / "manifest.json", metadata, backbone)

    def test_notebook_capture_preserves_result_and_rng_and_takes_detached_copies(self):
        original = (torch.tensor([1.], requires_grad=True), torch.tensor([2.]))
        captured = []
        wrapped = capture_notebook_heads(lambda: original, captured)
        rng = torch.get_rng_state().clone()
        self.assertIs(wrapped(), original)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        self.assertFalse(captured[0][0].requires_grad)
        with torch.no_grad():
            original[0].add_(10)
        self.assertEqual(float(captured[0][0]), 1.)


if __name__ == "__main__":
    unittest.main()
