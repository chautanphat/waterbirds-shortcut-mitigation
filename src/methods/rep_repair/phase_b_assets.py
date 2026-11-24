"""Validation-only asset recovery (backbones, Waterbirds images, CUB masks) for Phase B."""

from __future__ import annotations

import json
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.methods.dpe_independent.recovery import list_files
from src.methods.rep_repair.phase_a_data import file_sha256

WATERBIRDS_URLS = (
    "https://downloads.cs.stanford.edu/nlp/data/dro/waterbird_complete95_forest2water2.tar.gz",
    "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz",
)
CUB_SEG_URLS = (
    "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1",
)
BACKBONE_SPECS = (
    ("standard_erm", 0, "dpe-mlp-dropout-verification",
     "dpe_five_seed_verification/original_evidence/dpe_head_aggregation_session/backbones/standard/seed0/ckpt_last.pt"),
    ("standard_erm", 1, "dpe-mlp-dropout-verification",
     "dpe_five_seed_verification/original_evidence/dpe_head_aggregation_session/backbones/standard/seed1/ckpt_last.pt"),
    ("standard_erm", 2, "dpe-mlp-dropout-verification",
     "dpe_five_seed_verification/original_evidence/dpe_head_aggregation_session/backbones/standard/seed2/ckpt_last.pt"),
    ("standard_erm", 3, "dpe-mlp-dropout-verification",
     "dpe_five_seed_verification/additional_sources/backbones/standard/seed3/ckpt_last.pt"),
    ("standard_erm", 4, "dpe-mlp-dropout-verification",
     "dpe_five_seed_verification/additional_sources/backbones/standard/seed4/ckpt_last.pt"),
    ("erm_star", 0, "dpe-exploration", "dpe_ermstar/jobs/seed0/backbone/ckpt_last.pt"),
    ("erm_star", 1, "dpe-exploration", "dpe_ermstar/jobs/seed1/backbone/ckpt_last.pt"),
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def eval_image_transform():
    """Exact canonical Waterbirds evaluation transform matching src/datasets/waterbirds.py."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def pil_eval_crop(img: Image.Image) -> Image.Image:
    """Exact Resize(256, 256) -> CenterCrop(224) in PIL space (used for images and masks)."""
    resized = img.resize((256, 256), Image.BILINEAR)
    left = (256 - 224) // 2
    return resized.crop((left, left, left + 224, left + 224))


def _download_with_headers(urls: tuple[str, ...], dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in urls:
        tmp = dest.with_name(dest.name + ".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as resp, tmp.open("wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
            tmp.replace(dest)
            return
        except Exception as err:
            last_err = err
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Failed downloading {dest.name} from {urls}: {last_err}")


def _safe_extract_tar(archive_path: Path, dest_dir: Path):
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            rel = PurePosixPath(member.name)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"Unsafe tar member path: {member.name}")
        tar.extractall(path=dest_dir)


def ensure_waterbirds_images(work_data_dir: Path) -> Path:
    """Locate existing Waterbirds image root or download from Stanford mirror."""
    candidates = [
        work_data_dir / "waterbird_complete95_forest2water2",
        work_data_dir,
        Path("/kaggle/input/waterbirds/waterbird_complete95_forest2water2"),
        Path("/kaggle/input/waterbird-complete95-forest2water2/waterbird_complete95_forest2water2"),
    ]
    for cand in candidates:
        if (cand / "metadata.csv").is_file() and (cand / "001.Black_footed_Albatross").is_dir():
            return cand.resolve()

    archive = work_data_dir / "waterbird_complete95_forest2water2.tar.gz"
    if not archive.is_file():
        print("Downloading Waterbirds archive (~490 MB)...", flush=True)
        _download_with_headers(WATERBIRDS_URLS, archive)
    print("Extracting Waterbirds archive...", flush=True)
    _safe_extract_tar(archive, work_data_dir)
    extracted = work_data_dir / "waterbird_complete95_forest2water2"
    if not (extracted / "metadata.csv").is_file():
        raise FileNotFoundError(f"Expected metadata.csv under {extracted}")
    return extracted.resolve()


def ensure_cub_segmentations(work_mask_dir: Path) -> Path:
    """Locate existing CUB_200_2011 segmentations/ directory or download from Caltech."""
    for cand in (work_mask_dir / "segmentations", work_mask_dir):
        if (cand / "001.Black_footed_Albatross").is_dir():
            return cand.resolve()

    archive = work_mask_dir / "segmentations.tgz"
    if not archive.is_file():
        print("Downloading Caltech CUB_200_2011 segmentations (~39 MB)...", flush=True)
        _download_with_headers(CUB_SEG_URLS, archive)
    print("Extracting CUB segmentations...", flush=True)
    _safe_extract_tar(archive, work_mask_dir)
    seg_dir = work_mask_dir / "segmentations"
    if not (seg_dir / "001.Black_footed_Albatross").is_dir():
        raise FileNotFoundError(f"Expected CUB segmentations under {seg_dir}")
    return seg_dir.resolve()


def recover_backbones(
    output_dir: Path,
    cohorts: tuple[str, ...] = ("standard_erm", "erm_star"),
    seeds: dict[str, list[int]] | None = None,
    workers: int = 4,
) -> list[dict]:
    """Download saved ResNet-50 ckpt_last.pt checkpoints from the public Kaggle outputs."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    wanted = []
    for cohort, seed, slug, member in BACKBONE_SPECS:
        if cohort not in cohorts:
            continue
        if seeds is not None and seed not in seeds.get(cohort, []):
            continue
        dest = output_dir / f"{cohort}_s{seed}_ckpt_last.pt"
        wanted.append((cohort, seed, slug, member, dest))

    # Query public Kaggle API only for slugs that have missing local files.
    slug_index = {}
    for _, _, slug, _, dest in wanted:
        if not dest.is_file() and slug not in slug_index:
            slug_index[slug] = {row["fileName"]: row["url"] for row in list_files(slug)}

    def fetch(item):
        cohort, seed, slug, member, dest = item
        if not dest.is_file():
            url = slug_index[slug].get(member)
            if not url:
                raise FileNotFoundError(f"Backbone {member} not found in Kaggle output {slug}")
            print(f"Downloading {cohort} seed {seed} backbone -> {dest.name}...", flush=True)
            _download_with_headers((url,), dest)
        return {
            "cohort": cohort,
            "seed": seed,
            "source_id": f"{cohort}_s{seed}",
            "kaggle_slug": slug,
            "member": member,
            "path": str(dest),
            "sha256": file_sha256(dest),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(fetch, wanted))
    (output_dir / "backbones_manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


def load_val_mask_224(seg_root: Path, img_rel_path: str) -> np.ndarray:
    """Return float32 [224, 224] foreground mask in [0, 1] aligned with CenterCrop(224)."""
    rel = Path(img_rel_path)
    mask_path = seg_root / rel.parent / f"{rel.stem}.png"
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing CUB segmentation mask: {mask_path}")
    with Image.open(mask_path) as m:
        cropped = pil_eval_crop(m.convert("L"))
    return (np.asarray(cropped, dtype=np.float32) / 255.0).clip(0.0, 1.0)


def recover_kaggle_kernel_files(
    slug: str,
    wanted_members: list[str],
    dest_dir: Path,
    *,
    user_name: str = "noerula",
    input_search_root: Path = Path("/kaggle/input"),
    workers: int = 3,
) -> dict[str, Path]:
    """Locate saved .pt checkpoints in dest_dir or /kaggle/input, else fetch via public Kaggle API (early-stop)."""
    import urllib.parse

    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    missing: list[str] = []

    for member in wanted_members:
        fname = PurePosixPath(member).name
        local = dest_dir / fname
        if local.is_file():
            resolved[member] = local
            continue
        found_input = None
        if input_search_root.is_dir():
            matches = list(input_search_root.rglob(fname))
            if matches:
                found_input = matches[0].resolve()
        if found_input is not None:
            resolved[member] = found_input
        else:
            missing.append(member)

    if missing:
        wanted_set = set(missing)
        url_map: dict[str, str] = {}
        token = None
        while wanted_set - set(url_map):
            query = {"userName": user_name, "kernelSlug": slug}
            if token:
                query["pageToken"] = token
            api_url = "https://www.kaggle.com/api/v1/kernels/output?" + urllib.parse.urlencode(query)
            req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.load(resp)
            for row in data.get("files", []):
                name = row.get("fileName", "")
                if name in wanted_set:
                    url_map[name] = row["url"]
            if not data.get("hasNextPageToken"):
                break
            token = data["nextPageToken"]

        still_missing = sorted(wanted_set - set(url_map))
        if still_missing:
            raise FileNotFoundError(f"Members not found in Kaggle output {user_name}/{slug}: {still_missing}")

        def _fetch(member: str) -> tuple[str, Path]:
            dest = dest_dir / PurePosixPath(member).name
            print(f"Downloading {slug}/{member} -> {dest.name}...", flush=True)
            _download_with_headers((url_map[member],), dest)
            return member, dest

        with ThreadPoolExecutor(max_workers=min(workers, len(missing))) as pool:
            for member, path in pool.map(_fetch, missing):
                resolved[member] = path
    return resolved


def downsample_mask_to_grid(mask_224: np.ndarray, grid: int = 7) -> np.ndarray:
    """Area-average [224, 224] foreground mask into a [grid, grid] spatial map."""
    t = torch.from_numpy(mask_224)[None, None, :, :]
    pooled = torch.nn.functional.adaptive_avg_pool2d(t, (grid, grid))
    return pooled[0, 0].numpy().astype(np.float32)
