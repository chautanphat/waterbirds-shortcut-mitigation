#!/usr/bin/env python3
"""Lay out one dataset the way SubpopBench expects and generate its metadata with SubpopBench's own code.

    python prepare_data.py --subpop-root SUBPOP --data-dir DATA --dataset CelebA [--search /kaggle/input]

Metadata always comes from `subpopbench/scripts/download.py` (generate_metadata_*) at the pinned
commit, so labels, attributes, splits and the CivilComments "fine" groups are SubpopBench's.
Raw files are taken from attached inputs when found under --search, otherwise downloaded from the
URLs SubpopBench itself uses.

CelebA: the Kaggle dataset `jessicali9530/celeba-dataset` ships CSV versions of the official
list_attr_celeba.txt / list_eval_partition.txt; they are rewritten in the official text layout
(same rows, same order, values -1/1) before SubpopBench's generator reads them.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path


def find(roots, name):
    hits = []
    for root in roots:
        if Path(root).exists():
            hits += [p for p in Path(root).rglob(name)]
    return sorted(hits)


def safe_extract(archive, target):
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            dest = (Path(target) / m.name).resolve()
            if not dest.is_relative_to(Path(target).resolve()) or m.issym() or m.islnk() or m.isdev():
                raise RuntimeError(f"Unexpected archive entry: {m.name}")
        tar.extractall(target)


def download(url, target):
    part = Path(str(target) + ".part")
    urllib.request.urlretrieve(url, part)
    part.replace(target)


def generator(subpop_root, name):
    sys.path.insert(0, str(subpop_root))
    from subpopbench.scripts import download as d
    return getattr(d, name)


def celeba(a):
    root = a.data_dir / "celeba"
    root.mkdir(parents=True, exist_ok=True)
    images = root / "img_align_celeba"
    if not images.exists():
        firsts = [p.parent for p in find(a.search, "000001.jpg")]
        if not firsts:
            raise SystemExit("CelebA images not found: attach the Kaggle dataset jessicali9530/celeba-dataset")
        images.symlink_to(firsts[0].resolve(), target_is_directory=True)
    attr_txt, part_txt = root / "list_attr_celeba.txt", root / "list_eval_partition.txt"
    if not attr_txt.exists():
        txt = find(a.search, "list_attr_celeba.txt")
        if txt:
            shutil.copy(txt[0], attr_txt)
        else:
            src = find(a.search, "list_attr_celeba.csv")
            if not src:
                raise SystemExit("list_attr_celeba not found")
            with src[0].open(newline="") as f:
                rows = list(csv.reader(f))
            header, body = rows[0], rows[1:]
            with attr_txt.open("w") as f:
                f.write(f"{len(body)}\n")
                f.write(" ".join(header[1:]) + "\n")
                for r in body:
                    f.write(" ".join(r) + "\n")
    if not part_txt.exists():
        txt = find(a.search, "list_eval_partition.txt")
        if txt:
            shutil.copy(txt[0], part_txt)
        else:
            src = find(a.search, "list_eval_partition.csv")
            if not src:
                raise SystemExit("list_eval_partition not found")
            with src[0].open(newline="") as f:
                rows = list(csv.reader(f))[1:]
            with part_txt.open("w") as f:
                for r in rows:
                    f.write(f"{r[0]} {r[1]}\n")
    # Header check: Blond_Hair is attribute 10 and Male attribute 21 in the official order.
    lines = attr_txt.read_text().splitlines()
    names = lines[1].split()
    if names[9] != "Blond_Hair" or names[20] != "Male":
        raise SystemExit(f"Unexpected attribute order: {names[:21]}")
    # SubpopBench's generator pairs the two files line by line; the image ids must agree.
    attr_ids = [l.split()[0] for l in lines[2:]]
    part_ids = [l.split()[0] for l in part_txt.read_text().splitlines() if l.strip()]
    if attr_ids != part_ids or len(attr_ids) != 202599:
        raise SystemExit("list_attr_celeba and list_eval_partition disagree on image order or count")
    generator(a.subpop_root, "generate_metadata_celeba")(str(a.data_dir))


def civilcomments(a):
    root = a.data_dir / "civilcomments"
    root.mkdir(parents=True, exist_ok=True)
    target = root / "all_data_with_identities.csv"
    if not target.exists():
        hits = find(a.search, "all_data_with_identities.csv")
        if hits:
            target.symlink_to(hits[0].resolve())
        else:
            archive = root / "civilcomments.tar.gz"
            # The WILDS CivilComments v1.0 bundle used by SubpopBench's download_civilcomments.
            download("https://worksheets.codalab.org/rest/bundles/0x8cd3de0634154aeaad2ee6eb96723c6e/contents/blob/", archive)
            safe_extract(archive, root)
            archive.unlink()
            if not target.exists():
                found = list(root.rglob("all_data_with_identities.csv"))
                if not found:
                    raise SystemExit("all_data_with_identities.csv missing from the archive")
                target.symlink_to(found[0].resolve())
    generator(a.subpop_root, "generate_metadata_civilcomments")(str(a.data_dir))


def multinli(a):
    root = a.data_dir / "multinli"
    glue = root / "glue_data/MNLI"
    glue.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    needed = ["cached_train_bert-base-uncased_128_mnli", "cached_dev_bert-base-uncased_128_mnli",
              "cached_dev_bert-base-uncased_128_mnli-mm"]
    if not all((glue / n).exists() for n in needed):
        hits = {n: find(a.search, n) for n in needed}
        if all(hits.values()):
            for n in needed:
                (glue / n).symlink_to(hits[n][0].resolve())
        else:
            archive = glue / "multinli_bert_features.tar.gz"
            download("https://nlp.stanford.edu/data/dro/multinli_bert_features.tar.gz", archive)
            safe_extract(archive, glue)
            archive.unlink()
    meta = root / "data/metadata_random.csv"
    if not meta.exists():
        hits = find(a.search, "metadata_random.csv")
        if hits:
            shutil.copy(hits[0], meta)
        else:
            download("https://raw.githubusercontent.com/kohpangwei/group_DRO/master/dataset_metadata/multinli/metadata_random.csv", meta)
    generator(a.subpop_root, "generate_metadata_multinli")(str(a.data_dir))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subpop-root", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--dataset", choices=["CelebA", "CivilCommentsFine", "MultiNLI"], required=True)
    p.add_argument("--search", nargs="*", default=["/kaggle/input"])
    a = p.parse_args()
    a.data_dir = a.data_dir.resolve()
    {"CelebA": celeba, "CivilCommentsFine": civilcomments, "MultiNLI": multinli}[a.dataset](a)
    print("prepared", a.dataset, "in", a.data_dir)


if __name__ == "__main__":
    main()
