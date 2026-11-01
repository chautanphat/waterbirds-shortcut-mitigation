#!/usr/bin/env python
"""
prepare_waterbirds.py - Download, extract, and format the Waterbirds dataset.

Follows the project data protocol in docs/dataset.md and formats files
specifically for SubpopBench and DPE (baselines/dpe4subpop).

Usage:
    python scripts/prepare_waterbirds.py --data_dir data
"""

import os
import sys
import tarfile
import urllib.request
import argparse
import pandas as pd
from tqdm import tqdm


# Stanford CS downloads mirror (primary, active) and legacy nlp mirror (fallback)
WATERBIRDS_URLS = [
    "https://downloads.cs.stanford.edu/nlp/data/dro/waterbird_complete95_forest2water2.tar.gz",
    "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz",
]
ARCHIVE_NAME = "waterbird_complete95_forest2water2.tar.gz"


def download_file(urls, output_path):
    if isinstance(urls, str):
        urls = [urls]

    success = False
    for url in urls:
        print(f"[*] Downloading {url} -> {output_path}...")
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp, open(output_path, "wb") as out_file:
                total_size = int(resp.headers.get("content-length", 0))
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=os.path.basename(output_path)) as pbar:
                    while True:
                        chunk = resp.read(1024 * 1024)  # 1 MB chunks
                        if not chunk:
                            break
                        out_file.write(chunk)
                        pbar.update(len(chunk))
            success = True
            print(f"[+] Download complete: {output_path}")
            break
        except Exception as e:
            print(f"[!] Warning: Download from {url} failed: {e}")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            continue

    if not success:
        raise RuntimeError(f"Failed to download Waterbirds dataset from all available sources: {urls}")


def extract_archive(archive_path, extract_dir):
    print(f"[*] Extracting {archive_path} to {extract_dir}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)


def format_metadata(waterbirds_dir):
    """
    Ensures metadata_waterbirds.csv exists with columns:
    filename, y, a, split, g
    """
    raw_metadata_path = os.path.join(waterbirds_dir, "waterbird_complete95_forest2water2", "metadata.csv")
    target_metadata_path = os.path.join(waterbirds_dir, "metadata_waterbirds.csv")

    if not os.path.exists(raw_metadata_path):
        # Check if metadata.csv is in the parent directory
        alt_path = os.path.join(waterbirds_dir, "metadata.csv")
        if os.path.exists(alt_path):
            raw_metadata_path = alt_path
        else:
            raise FileNotFoundError(f"Cannot find metadata.csv in {waterbirds_dir}")

    print(f"[*] Formatting metadata from {raw_metadata_path} -> {target_metadata_path}...")
    df = pd.read_csv(raw_metadata_path)

    # Standardize column names expected by SubpopBench/DPE
    if 'filename' not in df.columns and 'img_filename' in df.columns:
        df['filename'] = df['img_filename']
    if 'a' not in df.columns and 'place' in df.columns:
        df['a'] = df['place']
    if 'g' not in df.columns:
        df['g'] = df['y'].astype(str) + df['a'].astype(str)

    # Save standardized metadata
    df.to_csv(target_metadata_path, index=False)
    print(f"[+] Successfully saved {target_metadata_path} ({len(df)} samples).")
    return df


def verify_dataset(df):
    """
    Verifies split counts and group distribution according to docs/dataset.md.
    """
    print("\n" + "=" * 60)
    print("         WATERBIRDS DATASET VERIFICATION (docs/dataset.md)")
    print("=" * 60)
    split_names = {0: 'Train', 1: 'Validation', 2: 'Test'}
    for split_id, split_name in split_names.items():
        sub_df = df[df['split'] == split_id]
        print(f"\n--- {split_name} Split (Total: {len(sub_df)}) ---")
        counts = sub_df.groupby(['y', 'a']).size()
        for (y, a), cnt in counts.items():
            bird = "Waterbird (y=1)" if y == 1 else "Landbird (y=0)"
            bg = "Water (a=1)" if a == 1 else "Land (a=0)"
            print(f"  Target: {bird:<18} | Background: {bg:<15} | Count: {cnt:>5}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Waterbirds dataset.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root data directory (default: 'data').")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip downloading if the archive already exists.")
    args = parser.parse_args()

    waterbirds_dir = os.path.join(args.data_dir, "waterbirds")
    os.makedirs(waterbirds_dir, exist_ok=True)

    archive_path = os.path.join(waterbirds_dir, ARCHIVE_NAME)

    # 1. Download
    if not os.path.exists(archive_path) or not args.skip_download:
        if not os.path.exists(archive_path):
            download_file(WATERBIRDS_URLS, archive_path)
        else:
            print(f"[*] Archive already exists at {archive_path}.")
    else:
        print(f"[*] Skipping download, using existing archive at {archive_path}.")

    # 2. Extract
    extracted_folder = os.path.join(waterbirds_dir, "waterbird_complete95_forest2water2")
    if not os.path.exists(extracted_folder):
        extract_archive(archive_path, waterbirds_dir)
    else:
        print(f"[*] Extracted folder already exists at {extracted_folder}.")

    # 3. Format metadata
    df = format_metadata(waterbirds_dir)

    # 4. Verify
    verify_dataset(df)
    print("[+] Waterbirds dataset is ready for training and feature extraction!")


if __name__ == "__main__":
    main()
