"""
Quét toàn bộ workspace để tìm các file prediction cache thật của Waterbirds (N ~= 4974, 15 heads).
Chạy lệnh: python -m experiments.gcaw_study.locate_real_data
"""

from __future__ import annotations
import zipfile
from pathlib import Path
import numpy as np


def inspect_npz_bytes(fobj, source_desc: str) -> bool:
    try:
        raw = np.load(fobj, allow_pickle=True)
        if not hasattr(raw, "files"):
            return False
        keys = raw.files
        for k in keys:
            arr = raw[k]
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[1] == 15 and arr.shape[2] == 2:
                if arr.shape[0] >= 2000:
                    print(f"[FOUND REAL CACHE] {source_desc} | key='{k}' shape={arr.shape} | all_keys={keys}")
                    return True
    except Exception:
        pass
    return False


def scan_workspace(root: Path = Path(".")) -> None:
    print(f"Scanning workspace '{root.resolve()}' for real Waterbirds caches (N >= 2000, shape=[N, 15, 2])...")
    found_any = False

    for p in sorted(root.rglob("*")):
        if any(part.startswith(".") or part in ("venv", ".venv", "__pycache__", "node_modules") for part in p.parts):
            continue
        if p.is_file():
            if p.suffix in (".npz", ".npy"):
                try:
                    if p.suffix == ".npz":
                        with open(p, "rb") as f:
                            if inspect_npz_bytes(f, str(p)):
                                found_any = True
                    else:
                        arr = np.load(p, mmap_mode="r", allow_pickle=True)
                        if arr.ndim == 3 and arr.shape[0] >= 2000 and arr.shape[1:] == (15, 2):
                            print(f"[FOUND REAL NPY] {p} | shape={arr.shape}")
                            found_any = True
                except Exception:
                    pass
            elif p.suffix == ".zip" and "review_bundle" not in p.name:
                try:
                    with zipfile.ZipFile(p, "r") as zf:
                        for member in zf.namelist():
                            if member.endswith(".npz"):
                                with zf.open(member) as zf_obj:
                                    if inspect_npz_bytes(zf_obj, f"{p}::{member}"):
                                        found_any = True
                except Exception:
                    pass

    if not found_any:
        print("[WARNING] Không tìm thấy file .npz/.npy/.zip nào có N >= 2000. Hãy kiểm tra các file .pt / .pkl / .joblib!")


if __name__ == "__main__":
    scan_workspace()
