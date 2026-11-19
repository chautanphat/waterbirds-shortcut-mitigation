#!/usr/bin/env python3
"""
download_error_images.py
Downloads only the 346 error images from Stanford Waterbirds archive.
Extracts directly to results/poter/visualizer/images/
"""

import os
import sys
import json
import tarfile
import urllib.request
import ssl
import time

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

WATERBIRDS_URLS = [
    "https://downloads.cs.stanford.edu/nlp/data/dro/waterbird_complete95_forest2water2.tar.gz",
    "http://downloads.cs.stanford.edu/nlp/data/dro/waterbird_complete95_forest2water2.tar.gz",
    "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
]

OUTPUT_DIR = os.path.join("results", "poter", "visualizer", "images")
JSON_PATH = os.path.join("results", "poter", "visualizer", "error_samples_summary.json")

def get_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

def main():
    if not os.path.exists(JSON_PATH):
        print(f"Error: {JSON_PATH} not found.")
        sys.exit(1)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_paths = set(item["path"].replace("\\", "/") for item in data.get("errors", []))
    print(f"[*] Total target error images: {len(target_paths)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check already downloaded
    already_done = set()
    for p in target_paths:
        local_p = os.path.join(OUTPUT_DIR, os.path.normpath(p))
        if os.path.exists(local_p) and os.path.getsize(local_p) > 0:
            already_done.add(p)

    remaining = target_paths - already_done
    print(f"[*] Already extracted: {len(already_done)}, Remaining to download: {len(remaining)}")

    if not remaining:
        print("[+] All 346 images are already present!")
        return

    # Attempt download and stream-extraction
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    ssl_ctx = get_ssl_context()

    for url in WATERBIRDS_URLS:
        print(f"\n[*] Connecting to {url}...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=60) as resp:
                print("[*] Connected! Stream-inspecting tar.gz archive...")
                start_time = time.time()
                extracted_count = 0
                bytes_read = 0

                # Open stream tar
                with tarfile.open(mode="r|gz", fileobj=resp) as tar:
                    for member in tar:
                        # normalize path
                        name = member.name.replace("\\", "/")
                        # strip prefix e.g. "waterbird_complete95_forest2water2/"
                        parts = name.split("/")
                        if len(parts) >= 2:
                            rel_key = "/".join(parts[-2:])
                        else:
                            rel_key = name

                        if rel_key in remaining:
                            f = tar.extractfile(member)
                            if f:
                                dest_path = os.path.join(OUTPUT_DIR, os.path.normpath(rel_key))
                                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                                with open(dest_path, "wb") as out:
                                    out.write(f.read())
                                remaining.remove(rel_key)
                                extracted_count += 1
                                if extracted_count % 25 == 0 or len(remaining) == 0:
                                    elapsed = time.time() - start_time
                                    print(f"  -> Extracted {extracted_count}/{len(target_paths) - len(already_done)} images (Remaining: {len(remaining)}) [{elapsed:.1f}s]")

                        if len(remaining) == 0:
                            print(f"\n[+] Success! All {len(target_paths)} error images extracted!")
                            return

        except Exception as e:
            print(f"[!] Error with {url}: {e}")
            if len(remaining) < len(target_paths) - len(already_done):
                print(f"    (Resuming with next mirror, {len(remaining)} images remaining...)")
            continue

    if remaining:
        print(f"\n[!] Finished archive scan but {len(remaining)} images were not found in archive.")
    else:
        print(f"\n[+] Completed successfully. Total images in {OUTPUT_DIR}: {len(target_paths)}")

if __name__ == "__main__":
    main()
