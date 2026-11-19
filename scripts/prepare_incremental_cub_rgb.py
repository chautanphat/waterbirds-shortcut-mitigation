"""Extract non-test CUB RGB for the provisional OT cohort from the official archive."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from tempfile import TemporaryDirectory
import time

import requests

from audit_vulnerability_ot_incremental import OUTPUT
from src.counterfactual.common import read_csv, sha256, write_json
from src.counterfactual.downloads import RangeStream

URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
RGB_SHA = "0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081"


class RetryingRangeStream(RangeStream):
    def _fetch(self, start, end):
        for delay in (0, 2, 5, 10, 20):
            if delay:
                time.sleep(delay)
            try:
                return super()._fetch(start, end)
            except (requests.ConnectionError, requests.Timeout):
                if delay == 20:
                    raise


def extract(source, cohort, target):
    wanted = {r["canonical_filename"] for r in read_csv(cohort) if r["official_split"] == "train"
              and r["planning_role"] in {"smoke_candidate", "primary_candidate"}}
    if len(wanted) != 224 or target.exists():
        raise ValueError("Expected 224 provisional train IDs and a fresh RGB directory")
    digest = hashlib.sha256()

    class CheckedStream:
        def read(self, count):
            block = source.read(count)
            digest.update(block)
            return block

    stream = CheckedStream()
    found = {}
    with TemporaryDirectory(dir=target.parent) as staging:
        stage = Path(staging) / target.name
        stage.mkdir()
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if not member.isfile() or len(path.parts) != 4 or path.parts[:2] != ("CUB_200_2011", "images"):
                    continue
                name = "/".join(path.parts[-2:])
                if name not in wanted:
                    continue
                rgb = stage / name
                rgb.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, rgb.open("xb") as dest:
                    shutil.copyfileobj(src, dest)
                found[name] = sha256(rgb)
                if len(found) % 40 == 0:
                    print(f"Staged {len(found)}/224 JPEGs, awaiting full archive verification", flush=True)
        while stream.read(1024 * 1024):
            pass
        if digest.hexdigest() != RGB_SHA or set(found) != wanted:
            raise ValueError("CUB RGB checksum or selected filename coverage mismatch")
        write_json(stage / "provenance.json", dict(source=URL, archive_sha256=RGB_SHA, image_sha256=found))
        stage.rename(target)
    print(f"Verified {len(found)} official CUB RGB files under {target}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Read a local official archive instead of streaming")
    parser.add_argument("--cohort", type=Path, default=OUTPUT / "cohort_manifest.csv")
    parser.add_argument("--target", type=Path, default=OUTPUT / "assets/cub_train_rgb")
    args = parser.parse_args()
    if args.archive:
        with args.archive.open("rb") as stream:
            extract(stream, args.cohort, args.target)
    else:
        origin = URL
        try:
            response = requests.get(origin, stream=True, timeout=(12, 35))
        except requests.exceptions.SSLError:
            origin = origin.replace("https:", "http:", 1)
            response = requests.get(origin, stream=True, timeout=(12, 35))
        with response:
            response.raise_for_status()
            if not response.url.startswith("https://"):
                raise ValueError("CUB RGB archive did not resolve to HTTPS")
            length = int(response.headers["Content-Length"])
            resolved = response.url
        print(f"Streaming {length} bytes for 224 selected RGB images; archive stays off disk", flush=True)
        stream = RetryingRangeStream(origin, resolved, length)
        try:
            extract(stream, args.cohort, args.target)
        finally:
            stream.close()
