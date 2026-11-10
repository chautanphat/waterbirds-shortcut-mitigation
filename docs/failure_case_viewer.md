# Failure case viewer

The local viewer in [`scripts/failure_case_viewer.html`](../scripts/failure_case_viewer.html) displays saved per-image predictions from different approaches. Open the HTML file in Chrome or Edge; no server, installation, account, or uploads are needed. Files stay in the tab and are cleared on reload. This tool does not run inference or modify method implementations.

A hosted copy works the same way: open its Sites link in Chrome or Edge and select local inputs. A private Site requires its owner's signed-in account. Hosting the viewer does not upload the selected predictions or images.

## Start

1. Open `scripts/failure_case_viewer.html` in your browser, or open the private Sites address if a hosted copy is configured. **Try a synthetic demo** exercises the controls with clearly labeled illustrative data.
2. For the quickest setup, choose **Open experiment folder** and select a folder containing prediction CSVs and images. The viewer finds both, skips metadata/summary CSVs, and prefers `predictions.csv` over `errors.csv` in the same directory. With two or more runs, it selects B automatically.
3. Alternatively, choose prediction CSVs and the image folder separately. You can add more CSVs afterward. A file can contain multiple runs, distinguished by `run_label` and `seed`. Choose the dataset folder, its `images/` folder, or a parent containing them. Images are matched by relative path; ambiguous paths are displayed as missing. **Import settings (optional)** contains the group-ID convention, file scope, and missing-split settings. **Download example CSV** provides an input template.
4. Choose approach **A** and optionally **B**. Use **A wrong / B correct**, **Both wrong**, **A and B disagree**, or the other case filters. Group buttons filter by the true bird/background combination. Search matches the species folder or image path. The confidence threshold and confidence sort use A's predicted-class confidence.
5. Click an image to enlarge it and see predictions from every loaded run. Arrow keys navigate matching cases; Escape closes the dialog. **Export filtered CSV** downloads the matching images' predictions, including missing predictions as blank fields.

## Expected local files

This is a suggested input layout for the viewer, not a change to the project's dataset convention. Existing method-specific layouts can remain as they are.

```text
data/waterbirds/                         # local, ignored by Git
├── metadata.csv                        # retained dataset metadata; not read by this viewer
└── images/
    ├── 001.Black_footed_Albatross/
    │   └── Black_Footed_Albatross_0046_18.jpg
    └── ...

outputs/failure_cases/                   # local, ignored by Git
├── erm_seed0/predictions.csv
├── autobackswap_seed0/predictions.csv
└── another_approach_seed0/predictions.csv
```

The original `waterbird_complete95_forest2water2/<species>/<image>.jpg` layout also works. The CSV's `image_path` should be `<species>/<image>.jpg` in either layout. Images are not resized or cropped by the viewer; they are fitted inside each card. Selecting the full dataset can take time because the browser enumerates its files. A folder containing only error images is also usable, with missing images shown explicitly.

For one-folder setup, a convenient local arrangement is:

```text
failure_cases/
├── images/<species>/<image>.jpg
├── erm/predictions.csv
└── approach_b/predictions.csv
```

Choose `failure_cases/` once. Files are read locally even when using the hosted Site. The browser requires you to select local files for each session; the Site cannot automatically read a filesystem path. Keep generated inspection folders outside tracked source or in an ignored output location.

## Prediction CSV

The existing exports from [`dump_all_predictions.py`](../scripts/autobackswap/dump_all_predictions.py) and [`dump_predictions.py`](../scripts/autobackswap/dump_predictions.py) are supported. For other approaches, export each evaluated example using this input format:

```csv
image_path,true_label,predicted_label,spurious_attribute,confidence,run_label,seed,split
001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg,waterbird,landbird,water,0.93,approach_a,0,test
```

Required columns:

| Information | Accepted columns / values |
| --- | --- |
| Relative image path | `image_path`, `img_filename`, or `filename` |
| True bird label | `true_label`, `y`, or `target`; `landbird`/`waterbird` or `0`/`1` |
| Predicted label | `predicted_label`, `y_pred`, or `pred`; same label mapping |
| True background | `spurious_attribute`, `place`, `a`, or `background`; `land`/`water` or `0`/`1` |

Alternatively, provide numeric `group` or `group_id` instead of background. Select the correct group convention **before importing**:

| True label / background | AutoBackSwap (default) | Shared evaluation |
| --- | --- | --- |
| Landbird / land | 1 | 0 |
| Landbird / water | 2 | 1 |
| Waterbird / land | 3 | 2 |
| Waterbird / water | 4 | 3 |

When both group and background are supplied, they must agree. Display names in a `group_name` column are ignored; the viewer derives names from the labels and background. This avoids inheriting incorrect display names from an exporter.

Optional columns:

- `confidence`: probability of the predicted class, between 0 and 1. Missing confidence displays as a dash and is excluded by positive confidence thresholds.
- `run_label` (also `method` or `run`) and `seed`: distinguish runs. Without a label, the filename is used. Repeated display names receive a numerical suffix.
- `split`: `train`, `val`/`validation`, `test`, or official split codes `0`/`1`/`2`. If absent, select **Split if missing** before importing, or leave it explicitly unspecified. Runs are compared only on matching image paths **and** split labels. Only one split is displayed at a time.
- `correct`: `0`/`1` or `false`/`true`; checked against labels if present.

Duplicate images within a run/split, conflicting ground truth across runs, invalid labels, or malformed CSVs produce an import error. Each file is imported atomically; other successfully loaded files remain available. Relative paths must match across approaches; the viewer cannot align row indices or infer equivalence between different filenames.

Aggregate metrics reports and DPE `.npz` caches are not per-image CSV inputs. They need an explicit export containing image paths and the predictions for the intended checkpoint or selected ensemble; this viewer does not convert them or select models.

See [Export predictions](export_predictions.md) for ready recovered AutoBackSwap/ERM CSVs, checkpoint export commands for POTER/AutoBackSwap/ERM, original-DPE export helpers, and the concrete image/output folder layout.

## Interpreting the display

- **Predictions** files can include correct and incorrect examples. Summary accuracy and per-group accuracy describe the loaded rows in the selected split, before gallery filters. Split completeness is not verified. Worst-group accuracy is shown only when all four groups are present.
- **Errors only** files have unknown denominators: accuracy and worst-group accuracy remain unavailable. Auto mode identifies error-named files or files with no correct predictions as errors-only. Choose **All predictions** explicitly for a genuine full export with zero correct predictions.
- Missing rows in another run mean **unknown**, including in errors-only CSVs. They are never counted as correct. To compare **A wrong / B correct**, load B's full predictions.
- A high-confidence mistake or a background-conflicting image is a case to inspect; it does not establish the cause of failure.
- Test failures are for analysis after final evaluation. Preserve the [research protocol](research_protocol.md): do not use test cases for checkpoint or hyperparameter selection. Retain the original configuration, checkpoint identity, command, environment, and seed with the run artifacts. The exported inspection CSV is a filtered view, not a substitute for those records.

## Maintaining and checking the viewer

The viewer is self-contained; keep its input contract and this guide consistent when changing it. Regression checks use Node's built-in test runner, without installing JavaScript packages:

```sh
node --test tests/test_failure_case_viewer.cjs
```

For visual verification, open the HTML file, try the demo, choose A/B comparison modes, click a group and image, navigate the detail dialog, and download a filtered CSV. Then load real predictions and choose their matching local image directory.
