"""Recover public saved output files without Kaggle credentials or retraining."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
import json
import urllib.parse
import urllib.request

from src.evaluation.prediction_export import sha256
from .storage import atomic_json

RUNS = ("dpe-exploration", "dpe-mlp-dropout-verification")


def list_files(slug):
    files, token = [], None
    while True:
        query = dict(userName="noerula", kernelSlug=slug)
        if token:
            query["pageToken"] = token
        url = "https://www.kaggle.com/api/v1/kernels/output?"+urllib.parse.urlencode(query)
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.load(response)
        files.extend(data["files"])
        if not data.get("hasNextPageToken"):
            return files
        token = data["nextPageToken"]


def wanted(slug, name):
    if slug == "dpe-exploration":
        return name.startswith("dpe_ermstar/jobs/") and "/backbone/" not in name and not name.endswith(".zip")
    if "/features/standard/" in name:
        return True
    return ("/results/five_seed_verification-" in name and
            not any(s in name for s in ("/.git/", "/repository/", "/source_snapshot/")) and
            name.endswith((".json", ".npz", ".pt", ".csv", ".jsonl")))


def recover(output, workers=4):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for slug in RUNS:
        for row in list_files(slug):
            name = row["fileName"]
            if wanted(slug, name):
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts or "\\" in name:
                    raise ValueError("Unsafe output filename")
                jobs.append((slug, row))

    def download(job):
        slug, row = job
        path = output/slug/row["fileName"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_name(path.name+".download-part")
            with urllib.request.urlopen(row["url"], timeout=120) as response, temporary.open("wb") as stream:
                while block := response.read(1024*1024):
                    stream.write(block)
            temporary.replace(path)
        # Retain public retrieval context, not expiring signed download URLs.
        return dict(notebook=f"https://www.kaggle.com/code/noerula/{slug}",
                    member=row["fileName"], location=str(path), sha256=sha256(path))

    with ThreadPoolExecutor(workers) as pool:
        records = list(pool.map(download, jobs))
    atomic_json(output/"recovered_artifacts.json", dict(files=records, backbone_training=False,
                note="Saved outputs retrieved via public API; hashes identify exact recovered bytes"))
    return records
