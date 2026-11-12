"""Waterbirds predictions.csv v1 contract and validation (standard library only)."""

import csv
import math
from pathlib import Path, PurePosixPath
import re

SCHEMA_VERSION = 1
LABELS = ("landbird", "waterbird")
BACKGROUNDS = ("land", "water")
GROUP_NAMES = ("Landbird on Land", "Landbird on Water", "Waterbird on Land", "Waterbird on Water")
SPLITS = {"train": "0", "val": "1", "test": "2"}
FIELDS = ["image_path", "true_label", "predicted_label", "correct", "group", "group_name",
          "spurious_attribute", "confidence", "run_label", "seed", "split"]


def image_path(value):
    """Normalize known dataset prefixes when reading source metadata, not CSVs."""
    text = str(value).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    prefix = "waterbird_complete95_forest2water2/"
    if text.startswith(prefix):
        text = text[len(prefix):]
    if text.startswith("images/"):
        text = text[len("images/"):]
    parts = text.split("/")
    if (not text or text != text.strip() or text.startswith("/") or ":" in text
        or any(ord(c) < 32 for c in text) or any(p in ("", ".", "..") for p in parts)):
        raise ValueError(f"Invalid relative image path: {value!r}")
    return str(PurePosixPath(text))


def read_metadata(path, split="test"):
    if split not in SPLITS:
        raise ValueError("split must be train, val or test")
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        records = list(csv.DictReader(stream, strict=True))
    rows, seen = [], set()
    for line, record in enumerate(records, 2):
        try:
            name = image_path(record.get("img_filename") or record.get("filename") or "")
            y = record["y"]
            a = record.get("place", record.get("a"))
            part = record["split"]
            if y not in ("0", "1") or a not in ("0", "1") or part not in SPLITS.values():
                raise ValueError("Labels/backgrounds must be 0/1 and split must be 0/1/2")
            if name in seen:
                raise ValueError(f"Duplicate metadata image: {name}")
            seen.add(name)
            if part == SPLITS[split]:
                rows.append(dict(image_path=name, y=int(y), a=int(a), split=split))
        except (KeyError, ValueError) as error:
            raise ValueError(f"Metadata record {line}: {error}") from error
    if not rows:
        raise ValueError(f"No metadata rows for {split}")
    return rows


def validate_records(records, reference=None):
    """Validate a single run/seed/split; optionally verify full metadata coverage."""
    seen, identity = set(), None
    counts, errors = [0] * 4, [0] * 4
    rows, missing_confidence = [], 0
    for line, row in enumerate(records, 2):
        try:
            if set(row) != set(FIELDS):
                raise ValueError("Missing or extra columns")
            for key in FIELDS:
                value = row[key]
                if not isinstance(value, str) or value != value.strip() or any(ord(c) < 32 for c in value):
                    raise ValueError(f"Invalid whitespace/control characters or missing value in {key}")
                if key != "confidence" and not value:
                    raise ValueError(f"{key} is required")
            name = image_path(row["image_path"])
            if name != row["image_path"]:
                raise ValueError("image_path must use canonical forward slashes without dataset/images prefixes")
            if name in seen:
                raise ValueError(f"Duplicate image_path: {name}")
            seen.add(name)
            if row["true_label"] not in LABELS or row["predicted_label"] not in LABELS or row["spurious_attribute"] not in BACKGROUNDS:
                raise ValueError("Use landbird/waterbird labels and land/water backgrounds")
            y, a = LABELS.index(row["true_label"]), BACKGROUNDS.index(row["spurious_attribute"])
            g = 2 * y + a
            if row["group"] != str(g + 1) or row["group_name"] != GROUP_NAMES[g]:
                raise ValueError("group/group_name disagrees with the 1 + 2*y + background mapping")
            correct = int(row["true_label"] == row["predicted_label"])
            if row["correct"] != str(correct):
                raise ValueError("correct must be 0/1 and agree with true_label/predicted_label")
            if row["split"] not in SPLITS:
                raise ValueError("split must be train, val or test")
            if re.fullmatch(r"0|[1-9][0-9]*", row["seed"]) is None:
                raise ValueError("seed must be a nonnegative decimal integer")
            current = (row["run_label"], int(row["seed"]), row["split"])
            if identity is not None and current != identity:
                raise ValueError("One predictions.csv must contain one run_label, seed and split")
            identity = current
            if row["confidence"]:
                confidence = float(row["confidence"])
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError("confidence must be a finite probability in [0, 1] or blank")
            else:
                missing_confidence += 1
            counts[g] += 1
            errors[g] += 1 - correct
            rows.append(dict(image_path=name, y=y, a=a, split=row["split"]))
        except (ValueError, TypeError) as error:
            raise ValueError(f"Prediction record {line}: {error}") from error
    if not rows:
        raise ValueError("A predictions.csv must contain at least one evaluated example")
    if reference is not None:
        expected = {r["image_path"]: r for r in reference}
        if len(expected) != len(reference):
            raise ValueError("Duplicate reference metadata IDs")
        missing, extra = set(expected) - seen, seen - set(expected)
        if missing or extra:
            raise ValueError(f"Split coverage mismatch: {len(missing)} missing, {len(extra)} unexpected images")
        for row in rows:
            if row != expected[row["image_path"]]:
                raise ValueError(f"Ground truth/background/split disagrees with metadata: {row['image_path']}")
    per_group = {str(i + 1): dict(count=counts[i], accuracy=1 - errors[i] / counts[i] if counts[i] else None) for i in range(4)}
    return dict(schema_version=SCHEMA_VERSION, run_label=identity[0], seed=identity[1], split=identity[2],
        count=len(rows), errors=sum(errors), accuracy=1 - sum(errors) / len(rows),
        worst_group_accuracy=min(g["accuracy"] for g in per_group.values()) if all(counts) else None,
        per_group=per_group, missing_confidence=missing_confidence, metadata_coverage_verified=reference is not None)


def validate_prediction_csv(path, metadata=None):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, strict=True)
        try:
            if reader.fieldnames != FIELDS:
                raise ValueError("Expected predictions.csv v1 header exactly: " + ",".join(FIELDS))
            records = list(reader)
        except csv.Error as error:
            raise ValueError(f"Malformed prediction CSV: {error}") from error
    result = validate_records(records)
    if metadata is not None:
        result = validate_records(records, read_metadata(metadata, result["split"]))
    return result
