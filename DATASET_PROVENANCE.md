# Waterbirds dataset provenance

Required source: [arubique/waterbirds](https://huggingface.co/datasets/arubique/waterbirds), pinned original revision `0b3b8b49a57aecf49b6e9ea707d9f120535c20d5`. Official auto-converted Parquet revision: `110dcb098399c0fc5928e9f9ce52e0fa19b1b4f7`. Setup uses the official `huggingface_hub` client and materializes the original JPEG bytes from twelve bundled Parquet files. This avoids the request limit encountered while downloading 17,385 individual files. `--representation files` remains supported. Already-present original JPEGs must match the converted bytes exactly.

Local roots: `data/waterbirds/hf/` (unaltered originals), `parquet/` (bundled download cache), `foregrounds/` (versioned derived PNG cutouts and masks), `cub/` (optional original CUB assets), and root CSV/JSON audit manifests. The entire data directory is already gitignored. Tokens and caches are never served by the local apps.

The snapshot has **17,382 JPEGs: 5,794 scenes, 5,794 foreground-only and 5,794 background-only images**. All 5,794 canonical bird IDs join uniquely to metadata, exclusively to official **test**. Its Hugging Face loader's `train` label is not the official split. Metadata contains 11,788 rows: train 4,795; validation 1,199; test 5,794. No train/validation images are falsely manufactured or relabeled.

Canonical labels are `y=0` landbird, `y=1` waterbird; `place=0` Land, `place=1` Water. The card's statement about reversed folder digits disagrees with the pinned files: actual digits match canonical y. The code asserts observed encoding but obtains labels exclusively from metadata. Known example: `img_id=1`, `001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg` is a waterbird on Water, official test. Canonical metadata SHA-256: `2f023b9dc371f6fbb82c7cb810ff0f03e19c54c8055a2e37517ae7bb56644362`.

| Bank | Images |
|---|---:|
| Land | 2,897 |
| Water | 2,897 |
| bamboo_forest | 1,471 |
| forest/broadleaf | 1,426 |
| lake/natural | 1,467 |
| ocean | 1,430 |

The scene categories come from `place_filename`, not from predictions. All downloaded images are decoded and checked; each manifest records byte SHA-256, pixel hash, dimensions, canonical split, Places filename, revision and source path. `dataset_audit.json` is the authoritative execution status. `quarantine.json` lists invalid images without destroying originals. The initial full HF audit found no invalid images, two exact duplicate background sets and 63 average-hash collision sets. The latter are candidate near duplicates requiring human inspection, not a semantic duplicate guarantee. Default background selection deduplicates exact pixels, balances Land/Water quotas, fixes the seed and rejects cross-split borrowing. Source IDs remain in the bird's canonical split. `background_reuse_audit.json` discloses fixed background reuse within a run.

HF foreground-only JPEGs contain a gray matte, not transparent alpha. Approximate extraction (`gray150-threshold12-erode1-v1`) is labeled and requires explicit opt-in. It never qualifies as a verified original cutout. For exact original cutouts, setup can stream the official [CUB RGB archive](https://data.caltech.edu/records/65de6-vp158) and [Farrell segmentation masks](https://data.caltech.edu/records/w9d68-gec53), retaining only the 5,794 supplied test birds. Published archive MD5s are checked: RGB `97eceeb196236b17998738112f37df78`; masks `4d47ba1228eae64f2fa547c47bc65255`. SHA-256s and source checksums are also recorded locally. Caltech's expired record certificate is handled through its public HTTP record redirect to verified HTTPS OSN object storage; certificate verification is never disabled. Extraction rejects paths outside the asset root and ignores links. Bounded parallel reads avoid keeping the 1.15 GB compressed archive on disk.

Use is restricted to noncommercial research and education; original image rights remain with image owners. Attribute Waterbirds (Sagawa et al., ICLR 2020), CUB (Wah et al., 2011), Places (Zhou et al., 2017), Farrell masks, and this derivative packaging/OCCAM when appropriate. See the [official Waterbirds generation code](https://github.com/kohpangwei/group_DRO/blob/master/dataset_scripts/generate_waterbirds.py), [WILDS dataset/license implementation](https://github.com/p-lambda/wilds/blob/main/wilds/datasets/waterbirds_dataset.py), and [Caltech dataset page](https://www.vision.caltech.edu/datasets/cub_200_2011/).

No human/animal detector is installed: manifest status explicitly says `not_screened_for_humans_or_animals`. No full Places365 download or generative image editing occurs. Optional local Places ingestion needs explicit category/source-split columns and records source hashes and quarantine results.

Executed final status: **all 5,794 cutouts use verified original CUB RGB/masks**, no quarantined images, full metadata join, and no Places-filename reuse across canonical splits. Both official archive checksums passed. RGB archive SHA-256 is `0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081`; mask archive SHA-256 is `dc77f6cffea0cbe2e41d4201115c8f29a6320ecb04fffd2444f51b8066e4b84f`. An offline audit rerun preserved selected raw metadata, RGBA and mask hashes exactly. Redundant Parquet bundles were pruned after materialization; `hf_snapshot_sha256.json` retains all original file hashes and permits verified reuse without network downloads.
