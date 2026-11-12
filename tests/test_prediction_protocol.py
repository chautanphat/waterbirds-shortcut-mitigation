"""Canonical prediction CSV contract checks; standard library only."""

import csv
import ast
import io
from pathlib import Path
import subprocess
import sys
import re
from tempfile import TemporaryDirectory
import unittest

from src.evaluation.prediction_protocol import FIELDS, validate_prediction_csv, validate_records

ROOT = Path(__file__).resolve().parents[1]


def prediction(**changes):
    row = dict(image_path="bird/a.jpg", true_label="waterbird", predicted_label="landbird", correct="0",
        group="3", group_name="Waterbird on Land", spurious_attribute="land", confidence="0.93",
        run_label="method_seed0_test", seed="0", split="test")
    return dict(row, **changes)


def write_csv(path, rows, header=None):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS if header is None else header)
        writer.writeheader()
        writer.writerows(rows)


class PredictionProtocolTests(unittest.TestCase):
    def test_documented_example_and_viewer_download_match_shared_contract(self):
        documentation = (ROOT / "docs/prediction_protocol.md").read_text(encoding="utf-8")
        example = documentation.split("```csv\n", 1)[1].split("```", 1)[0]
        html = (ROOT / "scripts/viewer.html").read_text(encoding="utf-8")
        literal = re.search(r"\$\('sample'\)\.onclick = \(\) => downloadCSV\(('(?:[^'\\]|\\.)*')", html).group(1)
        downloaded = ast.literal_eval(literal)
        for text in (example, downloaded):
            reader = csv.DictReader(io.StringIO(text))
            self.assertEqual(reader.fieldnames, FIELDS)
            self.assertEqual(validate_records(list(reader))["schema_version"], 1)

    def test_canonical_example_and_missing_confidence(self):
        result = validate_records([prediction(confidence="")])
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["missing_confidence"], 1)
        self.assertFalse(result["metadata_coverage_verified"])
        self.assertEqual(result["errors"], 1)
        self.assertIsNone(result["worst_group_accuracy"])

    def test_invalid_values_are_rejected(self):
        changes = [
            dict(correct="1"), dict(correct="false"), dict(group="2"),
            dict(group_name="Waterbird on land"), dict(true_label="1"), dict(predicted_label="2"),
            dict(confidence="NaN"), dict(confidence="inf"), dict(confidence="93"), dict(confidence="-.1"),
            dict(split="2"), dict(split="validation"), dict(seed="1.0"), dict(seed="-1"), dict(seed="01"),
            dict(run_label=""), dict(run_label=" padded"), dict(run_label="line\nbreak"),
            dict(image_path="images/bird/a.jpg"), dict(image_path="waterbird_complete95_forest2water2/bird/a.jpg"),
            dict(image_path="bird\\a.jpg"), dict(image_path="/bird/a.jpg"), dict(image_path="../a.jpg"),
            dict(image_path="bird/./a.jpg"), dict(image_path="bird//a.jpg"),
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_records([prediction(**change)])

    def test_one_run_seed_and_split_and_unique_identity(self):
        for second in (prediction(), prediction(image_path="bird/b.jpg", run_label="other"),
                       prediction(image_path="bird/b.jpg", seed="1"), prediction(image_path="bird/b.jpg", split="val")):
            with self.assertRaises(ValueError):
                validate_records([prediction(), second])

    def test_four_group_mapping_counts_accuracy_and_wga(self):
        rows = []
        names = ["Landbird on Land", "Landbird on Water", "Waterbird on Land", "Waterbird on Water"]
        for i in range(4):
            truth = "landbird" if i < 2 else "waterbird"
            rows.append(prediction(image_path=f"bird/{i}.jpg", true_label=truth,
                predicted_label=truth if i else "waterbird", correct="1" if i else "0",
                group=str(i + 1), group_name=names[i], spurious_attribute="water" if i % 2 else "land"))
        report = validate_records(rows)
        self.assertEqual(report["accuracy"], .75)
        self.assertEqual(report["worst_group_accuracy"], 0.)
        self.assertTrue(all(g["count"] == 1 for g in report["per_group"].values()))

    def test_exact_header_order_and_no_extra_or_legacy_columns(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.csv"
            write_csv(path, [prediction()])
            self.assertEqual(validate_prediction_csv(path)["count"], 1)
            for header in (FIELDS[::-1], FIELDS[:-1], FIELDS + ["index"]):
                write_csv(path, [{k: prediction().get(k, "") for k in header}], header)
                with self.assertRaisesRegex(ValueError, "header"):
                    validate_prediction_csv(path)

    def test_metadata_proves_coverage_and_truth_and_does_not_require_row_order(self):
        rows = [prediction(), prediction(image_path="bird/b.jpg")]
        reference = [dict(image_path="bird/b.jpg", y=1, a=0, split="test"),
                     dict(image_path="bird/a.jpg", y=1, a=0, split="test")]
        self.assertTrue(validate_records(rows, reference)["metadata_coverage_verified"])
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_records(rows[:1], reference)
        reference[0]["a"] = 1
        with self.assertRaisesRegex(ValueError, "disagrees"):
            validate_records(rows, reference)

    def test_csv_metadata_aliases_and_cli_success_failure(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "predictions.csv"
            metadata = root / "metadata.csv"
            metadata.write_text("filename,y,a,split\nbird/a.jpg,1,0,2\n")
            write_csv(path, [prediction()])
            report = validate_prediction_csv(path, metadata)
            self.assertTrue(report["metadata_coverage_verified"])
            command = [sys.executable, "-S", str(ROOT / "scripts/validate_predictions.py"), str(root), "--metadata", str(metadata)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('"metadata_coverage_verified": true', result.stdout)
            write_csv(path, [prediction(correct="1")])
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn('"valid": false', result.stdout)

    def test_empty_truncated_and_extra_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_records([])
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.csv"
            for tail in ("bird/a.jpg,waterbird\n", '"unclosed'):
                path.write_text(",".join(FIELDS) + "\n" + tail)
                with self.assertRaises(ValueError):
                    validate_prediction_csv(path)
            write_csv(path, [prediction()])
            path.write_text(path.read_text(encoding="utf-8-sig").rstrip() + ",extra\n")
            with self.assertRaises(ValueError):
                validate_prediction_csv(path)


if __name__ == "__main__":
    unittest.main()
