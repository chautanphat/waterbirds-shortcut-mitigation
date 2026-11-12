# Waterbirds Prediction Viewer Guide

The standalone viewer in [`scripts/viewer.html`](../scripts/viewer.html) compares saved per-image predictions across **DPE**, **AutoBackSwap**, **POTER**, **ERM**, and future baselines (such as **GroupDRO**).

Open [`scripts/viewer.html`](../scripts/viewer.html) directly in Chrome, Edge, or Firefox (`file://`) — **no Python server, installation, account, or internet connection is required**. All files stay locally in your browser tab.

---

## Quick Start (2 Steps)

1. **Open `scripts/viewer.html` in your browser:**
   - All **13 exported research runs** (`dpe_seed0`, `autobackswap_seed0..2`, `poter_seed1`, `erm_stage0_seed0`, `autobackswap_w_minority_seed0`, `erm_baseline_seed0..2`, `erm_w_minority_seed0..2`) are pre-bundled and **auto-load automatically** on startup.
   - By default, the viewer selects one primary representative per method family on the image cards:
     - **DPE (`s0`)** — `93.9% WGA` (`95.8%` overall acc, `246` errors)
     - **AutoBackSwap (`s0`)** — `92.2% WGA` (`94.0%` overall acc, `346` errors; `s2` reaches `92.5% WGA`)
     - **POTER (`s1`)** — `91.0% WGA` (`92.0%` overall acc, `464` errors)
     - **ERM Stage-0 (`s0`)** — `76.6% WGA` (`92.4%` overall acc, `443` errors)
   - Click `+ 9 other seed(s) / baseline variant(s)` in the **Methods on cards** bar to toggle additional seeds or baseline variants at any time.
2. **Load Bird Photos (`data/waterbirds/`):**
   - Click **`2. 🖼️ Load Bird Photos (data/waterbirds/)`** in the top bar and select the local `data/waterbirds` directory (or `data/waterbirds/waterbird_complete95_forest2water2`).
   - Why this click is needed once per session: browsers block `file://` pages from reading arbitrary folders on disk without user permission, and embedding all 11,788 dataset JPEGs (~1.2 GB) into the HTML file would exceed browser memory limits. Once you select `data/waterbirds/`, photos render immediately on every card and in the zoom modal.
   - *(Optional)* If you export new runs into `experiments/` (for example `groupdro_seed0`), click **`1. 📂 Load Predictions (experiments/)`** and select the `experiments` folder to load them.

---

## How to Use the Viewer Controls

### 1. Four View Tabs
Use the **View** tabs to choose which test images to inspect:
- **`🔴 Common Failures (All Wrong)`** *(Default)* — Shows only images where **every** active method on the cards predicted the wrong bird label. Use this to spot the hardest benchmark examples and potential dataset label noise.
- **`⚡ Methods Disagree`** — Shows images where at least one active method is right and at least one is wrong.
- **`🔍 Compare A vs B`** — Opens the pairwise comparison bar (`Focus Method A`, `Rule`, `Compare Method B`) to filter for:
  - `Method A is wrong`
  - `Method A wrong, Method B correct` (cases fixed by Method B)
  - `Method A correct, Method B wrong` (regressions in Method B)
  - `Both Method A and B wrong`
  - `Method A and B disagree`
- **`📋 All Images`** — Shows all 5,794 held-out test images.

### 2. Subpopulation Group Cards (`G1–G4`)
Click any of the four group cards (`G1 · Landbird on land`, `G2 · Landbird on water`, `G3 · Waterbird on land`, `G4 · Waterbird on water`) to filter the gallery and species ranking to that subpopulation. Click the active card again to clear the group filter.

### 3. Bird Species Failure Ranking (`🏆 Bird Species Ranking`)
- Expanded by default right below the search bar, this panel ranks all 200 bird species (`#1`, `#2`, `#3`, …) by how often they match the **currently active View Tab** (and selected `G1–G4` group, if any):
  - In **Common Failures (All Wrong)**, `4/30 (13%)` means all active methods simultaneously failed on 4 of the 30 test images of that bird species.
  - In **Methods Disagree**, `26/30 (87%)` means active methods disagreed on 26 of the 30 test images of that species.
  - In **Compare A vs B (`Method A is wrong`)**, `15/30 (50%)` means Method A misclassified 15 of the 30 test images of that species.
- Click any bird species row in the ranking panel (or choose it from the **Kind of Bird (Species)** dropdown) to filter the image cards to that species. Click it again to clear.

### 4. Inspecting Cards & Exporting Filtered CSVs
- **Card Grouping:** Use **Group Cards By** to switch between a flat grid, grouping by **Subpopulation (`G1–G4`)**, or grouping by **Kind of bird (Species)**.
- **Zoom & All-Run Inspection:** Click any image card to open the detail dialog with the full-resolution bird photo and a table of predictions across all 13 loaded runs. Use `Left` / `Right` arrow keys to step through matching cases and `Escape` to close.
- **Export CSV:** Click **`⬇ Export CSV`** to download the currently filtered subset as `waterbirds_filtered_cases.csv`.

---

## Expected Local Directory Layout

```text
experiments/                                        # Clean method-first experiment IDs
├── dpe_seed0/predictions/test/predictions.csv      # DPE 15-head prototype ensemble (93.9% WGA)
├── poter_seed1/predictions/test/predictions.csv    # POTER baseline (91.0% WGA)
├── autobackswap_seed0/predictions/test/predictions.csv
├── autobackswap_seed1/predictions/test/predictions.csv
├── autobackswap_seed2/predictions/test/predictions.csv
├── autobackswap_w_minority_seed0/predictions/test/predictions.csv
├── erm_stage0_seed0/predictions/test/predictions.csv
├── erm_baseline_seed0/predictions/test/predictions.csv
├── erm_baseline_seed1/predictions/test/predictions.csv
├── erm_baseline_seed2/predictions/test/predictions.csv
├── erm_w_minority_seed0/predictions/test/predictions.csv
├── erm_w_minority_seed1/predictions/test/predictions.csv
└── erm_w_minority_seed2/predictions/test/predictions.csv

data/waterbirds/                                    # Local dataset root (gitignored)
├── metadata_waterbirds.csv
└── waterbird_complete95_forest2water2/
    ├── metadata.csv
    └── <species>/<image>.jpg
```

---

## Prediction CSV Compatibility

All new method exports must follow the [shared `predictions.csv` v1 protocol](prediction_protocol.md):

```csv
image_path,true_label,predicted_label,correct,group,group_name,spurious_attribute,confidence,run_label,seed,split
001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg,waterbird,landbird,0,4,Waterbird on Water,water,0.93,poter_seed1,1,test
```

For backward compatibility with historical exports, the viewer also accepts legacy column aliases (`img_filename`/`filename`, `y`/`target`, `y_pred`/`pred`, `place`/`a`/`background`, `group_id`) via **⚙️ CSV Import Options** in the top bar.

See [Export predictions](export_predictions.md) for CLI commands to export new checkpoints (`poter`, `autobackswap`, `erm`, `dpe-heads`, `dpe-probabilities`).

---

## Regression Testing

The viewer's data parser, normalization, multi-method filtering, species ranking, and DOM interactions are tested with Node's built-in test runner (no external npm packages required):

```sh
node --test tests/test_viewer.cjs
python -m unittest discover -s tests -p "test_prediction*.py" -v
```
