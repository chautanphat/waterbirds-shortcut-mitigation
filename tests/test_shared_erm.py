"""CPU regression tests for the shared ERM contracts (no downloads required)."""

import ast
import json
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import RandomSampler, WeightedRandomSampler

from src.datasets.loaders import evaluation_loader, training_loader
from src.datasets.waterbirds import Waterbirds, waterbirds_training_transform
from src.evaluation.classification import group_metrics
from src.evaluation.features import extract_features, save_features
from src.methods.erm.config import ERMConfig, load_config
from src.methods.erm.train import run_training, train_epoch
from src.models.resnet import ResNetClassifier, load_resnet_checkpoint
from src.utils.reproducibility import seed_everything

ROOT = Path(__file__).resolve().parents[1]


def make_fixture(root, *, include_test=True, original=False):
    images = root / "waterbird_complete95_forest2water2"
    images.mkdir()
    rows = []
    # Deliberately interleave split rows; caches must preserve within-split order.
    for group in (3, 0, 2, 1):
        for split in range(3 if include_test else 2):
            filename = f"{split}_{group}.jpg"
            Image.new("RGB", (300, 230), (group * 70, split * 70, 30)).save(images / filename)
            rows.append({"filename": filename, "y": group // 2, "a": group % 2, "split": split})
    frame = pd.DataFrame(rows)
    if original:
        frame.rename(columns={"filename": "img_filename", "a": "place"}).to_csv(images / "metadata.csv", index=False)
    else:
        frame.to_csv(root / "metadata_waterbirds.csv", index=False)


class TinyClassifier(nn.Sequential):
    """Fast genuine image training for integration tests."""

    def __init__(self, *args):
        super().__init__(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(3, 2))

    def extract_features(self, inputs):
        return self[1](self[0](inputs))


class SharedERMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_config_matches_effective_dpe_notebook_recipe(self):
        config = load_config(ROOT / "configs/erm/waterbirds_standard.yaml")
        # Compare to the actual notebook command plus DPE parser defaults,
        # rather than a YAML file that the notebook never loads.
        tree = ast.parse((ROOT / "baselines/dpe4subpop/main.py").read_text())
        parser_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_args")
        effective = {}
        for node in ast.walk(parser_function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                for keyword in node.keywords:
                    if keyword.arg == "default":
                        flags = [ast.literal_eval(arg) for arg in node.args]
                        flag = next(flag for flag in flags if flag.startswith("--"))
                        effective[flag] = ast.literal_eval(keyword.value)
        notebook = json.loads((ROOT / "notebooks/dpe_waterbirds_end_to_end.ipynb").read_text())
        source = next("".join(cell.get("source", [])) for cell in notebook["cells"]
                      if "stage0_command =" in "".join(cell.get("source", [])))
        command = next(node.value for node in ast.parse(source).body
                       if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "stage0_command" for t in node.targets))
        tokens = [node.value if isinstance(node, ast.Constant) else "placeholder" for node in command.elts]
        index = 2
        while index < len(tokens):
            flag = tokens[index]
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                effective[flag] = tokens[index + 1]
                index += 2
            else:
                effective[flag] = True
                index += 1
        self.assertEqual(config.as_dict(), ERMConfig().as_dict())
        for key in ("epochs", "lr", "momentum", "seed"):
            self.assertEqual(getattr(config.training, key), float(effective[f"--{key}"]))
        self.assertEqual(config.training.weight_decay, float(effective["--optim_weight_decay"]))
        for key in ("batch_size", "batch_size_eval", "workers"):
            self.assertEqual(getattr(config.dataset, key), int(effective[f"--{key}"]))
        self.assertTrue(effective["--no_augmentation"])
        self.assertTrue(effective["--pretrained_imgnet"])
        self.assertEqual(effective["--optim"], "sgd")
        self.assertEqual(effective["--scheduler"], "none")
        self.assertEqual(effective["--train_mode"], "full")
        self.assertEqual(effective["--loss_name"], "ce")
        self.assertEqual(config.model.weights, "IMAGENET1K_V2")
        self.assertFalse(config.model.freeze_bn)
        self.assertEqual(config.training.sampling, "class_balanced")

    def test_invalid_config_is_rejected(self):
        config = ERMConfig()
        config.training.steps = 3
        with self.assertRaisesRegex(ValueError, "exactly one"):
            config.validate()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.yaml"
            path.write_text("training:\n  weight_deacy: 0.1\n")
            with self.assertRaisesRegex(ValueError, "Unknown training fields"):
                load_config(path)

    def test_enhanced_training_augmentation_and_deterministic_evaluation(self):
        config = load_config(ROOT / "configs/erm/waterbirds_enhanced.yaml")
        self.assertTrue(config.dataset.augmentation)
        self.assertEqual(config.training, ERMConfig().training)
        from torchvision.transforms import v2
        transform = waterbirds_training_transform()
        self.assertIsInstance(transform.transforms[0], v2.RandomResizedCrop)
        self.assertIsInstance(transform.transforms[1], v2.RandomPhotometricDistort)
        self.assertIsInstance(transform.transforms[2], v2.RandomHorizontalFlip)
        with tempfile.TemporaryDirectory() as temp:
            make_fixture(Path(temp))
            train = Waterbirds(temp, "train", transform=transform)
            val = Waterbirds(temp, "val")
            self.assertEqual(tuple(train[0][1].shape), (3, 224, 224))
            self.assertEqual(train[0][1].dtype, torch.float32)
            self.assertTrue(torch.isfinite(train[0][1]).all())
            self.assertTrue(torch.equal(val[0][1], val[0][1]))

    def test_dataset_formats_groups_and_row_order(self):
        for original in (False, True):
            with self.subTest(original=original), tempfile.TemporaryDirectory() as temp:
                make_fixture(Path(temp), original=original)
                data = Waterbirds(temp, "val")
                self.assertEqual(data.g.tolist(), [3, 0, 2, 1])
                self.assertEqual(data.filenames, ["1_3.jpg", "1_0.jpg", "1_2.jpg", "1_1.jpg"])
                index, image, y, a = data[0]
                self.assertEqual((index, y, a), (0, 1, 1))
                self.assertEqual(tuple(image.shape), (3, 224, 224))
                self.assertTrue(torch.equal(image, data[0][1]))
                self.assertEqual(data.transform.transforms[0].size, (256, 256))

    def test_missing_image_and_invalid_metadata_fail_early(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_fixture(root)
            (root / "waterbird_complete95_forest2water2/0_3.jpg").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "missing"):
                Waterbirds(root, "train")
            frame = pd.read_csv(root / "metadata_waterbirds.csv")
            frame.loc[0, "a"] = 9
            frame.to_csv(root / "metadata_waterbirds.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Invalid Waterbirds a"):
                Waterbirds(root, "val")

    def test_sampling_weights_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as temp:
            make_fixture(Path(temp))
            data = Waterbirds(temp, "train")
            data.y[:] = [0, 0, 0, 1]
            seed_everything(3)
            loader = training_loader(data, 3, sampling="class_balanced")
            self.assertIsInstance(loader.sampler, WeightedRandomSampler)
            torch.testing.assert_close(loader.sampler.weights, torch.tensor([1/3, 1/3, 1/3, 1.], dtype=torch.double))
            self.assertTrue(loader.sampler.replacement)
            first = list(loader.sampler)
            seed_everything(3)
            self.assertEqual(first, list(training_loader(data, 3, sampling="class_balanced").sampler))
            self.assertEqual(len(loader), 1)  # DPE drops incomplete training batches.
            uniform = training_loader(data, 3, sampling="uniform", steps=5)
            self.assertIsInstance(uniform.sampler, RandomSampler)
            self.assertTrue(uniform.sampler.replacement)
            self.assertEqual(len(uniform), 5)
            self.assertEqual(len(evaluation_loader(data, 3)), 2)

    def test_metrics_include_group_counts_and_missing_groups(self):
        for predictions, targets, attributes in (([0.5], [0], [0]), ([0], [0.5], [0]), ([0], [0], [0.5])):
            with self.assertRaisesRegex(ValueError, "binary"):
                group_metrics(predictions, targets, attributes)
        metrics = group_metrics([0, 1, 1, 0, 1], [0, 0, 1, 1, 1], [0, 1, 0, 1, 1])
        self.assertAlmostEqual(metrics["accuracy"], 0.6)
        self.assertEqual(metrics["worst_group_accuracy"], 0)
        self.assertEqual(metrics["per_group"]["3"]["count"], 2)
        self.assertEqual(metrics["per_group"]["3"]["accuracy"], 0.5)
        self.assertIsNone(group_metrics([0], [0], [0])["worst_group_accuracy"])

    def test_model_bn_policy_and_dpe_checkpoint_layout(self):
        model = ResNetClassifier(weights=None, freeze_bn=False)
        model.train()
        self.assertTrue(model[0][1].training)
        model.freeze_bn = True
        model.train()
        self.assertFalse(model[0][1].training)
        self.assertTrue(model[0][1].weight.requires_grad)
        # This is the actual Sequential architecture built by DPE get_model().
        from torchvision.models import resnet50
        backbone = resnet50(weights=None)
        legacy = nn.Sequential(nn.Sequential(*list(backbone.children())[:-1]),
                               nn.Flatten(), nn.Linear(2048, 2))
        legacy.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        legacy.eval()
        inputs = torch.randn(1, 3, 32, 32)
        with torch.inference_mode():
            torch.testing.assert_close(model(inputs), legacy(inputs))
            self.assertEqual(tuple(model.extract_features(inputs).shape), (1, 2048))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ckpt.pt"
            torch.save(model.state_dict(), path)
            with patch("src.models.resnet.resnet50", wraps=resnet50) as factory:
                restored = load_resnet_checkpoint(path)
            factory.assert_called_once_with(weights=None)
            torch.testing.assert_close(restored[2].weight, model[2].weight)

    def test_training_updates_backbone_without_attributes(self):
        seed_everything(0)
        model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
        before = [p.detach().clone() for p in model.parameters()]
        inputs, labels = torch.randn(4, 3), torch.tensor([0, 1, 0, 1])
        # Attributes deliberately unusable; ERM must not inspect them.
        batch = (torch.arange(4), inputs, labels, None)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        train_epoch(model, [batch], optimizer, "cpu")
        self.assertTrue(all(not torch.equal(old, new) for old, new in zip(before, model.parameters())))

    def test_train_select_export_without_test_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_fixture(root, include_test=False)
            config = ERMConfig()
            config.dataset.data_dir = temp
            config.dataset.workers = 0
            config.dataset.batch_size = 3
            config.dataset.batch_size_eval = 3  # exercises a singleton final batch
            config.training.device = "cpu"
            config.training.epochs = 2
            config.output_dir = str(root / "run")
            # A worse final validation result must not overwrite the selected model.
            import src.methods.erm.train as training
            real_evaluate = training.evaluate
            seen = []

            def controlled_eval(model, loader, device):
                metrics = real_evaluate(model, loader, device)
                metrics["worst_group_accuracy"] = [0.8, 0.4][len(seen)]
                seen.append(model[2].weight.detach().clone())
                return metrics

            with patch.object(training, "ResNetClassifier", TinyClassifier), patch.object(training, "evaluate", controlled_eval):
                summary = run_training(config)
            self.assertEqual(summary["updates"], 2)
            self.assertEqual(summary["best_position"], {"epoch": 1})
            self.assertEqual(summary["selected_checkpoint"], "ckpt_last.pt")
            self.assertEqual(summary["best_checkpoint"], "ckpt_best_wga.pt")
            out = Path(config.output_dir)
            best = torch.load(out / "ckpt_best_wga.pt", weights_only=True)
            last = torch.load(out / "ckpt_last.pt", weights_only=True)
            torch.testing.assert_close(best["2.weight"], seen[0])
            torch.testing.assert_close(last["2.weight"], seen[1])
            state = torch.load(out / "training_state.pt", weights_only=True)
            self.assertEqual(state["optimizer"]["param_groups"][0]["weight_decay"], 0.0)
            self.assertEqual(json.loads((out / "run.json").read_text())["split_counts"], {"train": 4, "val": 4})
            with self.assertRaises(FileExistsError):
                run_training(config)
            model = TinyClassifier()
            model.load_state_dict(best)
            data = Waterbirds(root, "val")
            loader = evaluation_loader(data, 3)
            features = extract_features(model, loader)
            self.assertEqual(features["features"].shape, (4, 3))
            save_features(model, loader, out / "ckpt_best_wga.pt", out)
            np.testing.assert_array_equal(np.load(out / "feats_val.npy"), features["features"])
            with np.load(out / "feats_val.npz") as labels:
                np.testing.assert_array_equal(labels["filenames"], data.filenames)
            with self.assertRaises(FileExistsError):
                save_features(model, loader, out / "ckpt_best_wga.pt", out)

    def test_step_budget_and_feature_order_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_fixture(root, include_test=False)
            config = ERMConfig()
            config.dataset.data_dir = temp
            config.dataset.workers = 0
            config.dataset.batch_size = 3
            config.training.device = "cpu"
            config.training.epochs, config.training.steps = None, 5
            config.training.eval_every = 2
            config.output_dir = str(root / "run")
            with patch("src.methods.erm.train.ResNetClassifier", TinyClassifier):
                summary = run_training(config)
            self.assertEqual(summary["updates"], 5)
            records = [json.loads(line) for line in (root / "run/metrics.jsonl").read_text().splitlines()]
            self.assertEqual([r["updates"] for r in records], [2, 4, 5])
            data = Waterbirds(root, "val")
            reordered = torch.utils.data.DataLoader(data, batch_size=3, sampler=[3, 2, 1, 0])
            with self.assertRaisesRegex(ValueError, "metadata order"):
                extract_features(TinyClassifier(), reordered)


if __name__ == "__main__":
    unittest.main()
