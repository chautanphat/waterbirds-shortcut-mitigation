# POTER image preview

This app lets you test a trained POTER Waterbirds classifier by importing an image, painting interventions directly on it, and comparing model predictions with ground truth. It uses three panels: **Input**, **Model output**, and **Ground truth**.

## Local artifacts

Model weights and dataset images are not included in Git. Supply your recovered Kaggle `best_validation.pt`, the official Waterbirds `metadata.csv`, and the directory containing the species folders and JPEGs. The selected checkpoint used during development was seed 1, epoch 258 (zero-based); another checkpoint can be supplied explicitly.

```sh
python scripts/poter_preview.py \
  --checkpoint /path/to/best_validation.pt \
  --metadata /path/to/metadata.csv \
  --data-dir /path/to/waterbird_complete95_forest2water2
```

The command above uses Bash line continuations. In PowerShell, put the command on one line or use backticks instead. `--data-dir` must point directly to the directory containing folders such as `001.Black_footed_Albatross`.

## Start with recovered repository-local artifacts

Run from the repository root in a Python environment containing PyTorch, torchvision and Pillow:

```sh
python scripts/poter_preview.py
```

Open <http://127.0.0.1:8765>. The server binds only to the local machine. It loads the recovered `best_validation.pt` using `weights_only=True`; no training occurs. The default is CPU. Pass `--device cuda` for GPU inference. Use `--checkpoint`, `--metadata`, `--data-dir` or `--port` to change local artifact locations.

Import an image or choose a Waterbirds test sample. Draw with a brush or a filled rectangle; undo and reset are available. Click **Run comparison** to evaluate both the original and the edited input. The three panels show editable input, actual model crop with output probabilities, and ground truth. Save the edited image and comparison JSON separately.

Inference preserves the Waterbirds evaluation transform: resize to 256 × 256, center crop 224 × 224, ImageNet normalization, ResNet-50 binary softmax in class order `[landbird, waterbird]`. The tool reduces imported images to a maximum side of 1,600 pixels for drawing; it evaluates this displayed original and its edit. A single solid color is painted at full opacity. This is an intervention for observation, not a segmentation model or an explanation of causality.

Ground truth comes from metadata when the image path or filename uniquely matches. An external file with the same filename may be a different image, so verify identity. If there is no match, the label remains unknown unless entered manually. Editing the image never changes its ground truth automatically. Removing the bird can make the inherited label unsuitable. Model probabilities are softmax scores, not calibrated certainty.

## Verification

The recovered checkpoint loaded successfully with strict model-state validation. An end-to-end browser check loaded a Waterbirds test sample, drew a brush stroke, and reran the real model: its waterbird probability changed from 79.91% to 80.24%, while the metadata label remained waterbird. This verifies the intervention workflow; it is not a benchmark result or evidence of causality.
