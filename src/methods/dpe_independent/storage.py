"""Atomic progress, immutable inputs and source snapshots for resumable studies."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import subprocess
import zipfile

import numpy as np

from src.evaluation.prediction_export import sha256

ROOT = Path(__file__).resolve().parents[3]


def read_json(path):
    return json.loads(Path(path).read_text())


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n"
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def frozen_json(path, value):
    path = Path(path)
    if path.exists():
        if object_hash(read_json(path)) != object_hash(value):
            raise ValueError(f"Frozen inputs changed; preserve output and use a fresh directory: {path}")
    else:
        atomic_json(path, value)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-", suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def implementation(output):
    files = sorted([*ROOT.glob("src/**/*.py"), ROOT/"configs/dpe_independent/study.yaml"])
    identity = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    path = Path(output)/"implementation.zip"
    marker = Path(output)/"implementation.json"
    frozen_json(marker, identity)
    if not path.exists():
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for p in files:
                archive.write(p, p.relative_to(ROOT))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    return dict(repository="https://github.com/chautanphat/waterbirds-shortcut-mitigation",
                revision=revision.stdout.strip() if revision.returncode == 0 else None,
                dirty=bool(status.stdout) if status.returncode == 0 else None,
                source_snapshot_is_authoritative=True, files=identity, snapshot=str(path), sha256=sha256(path))


class Budget:
    def __init__(self, hours):
        import time
        self.started = time.monotonic()
        self.seconds = hours*3600

    def check(self):
        import time
        if time.monotonic()-self.started >= self.seconds:
            raise TimeoutError("Eight-hour session budget reached; saved outputs can be resumed")
