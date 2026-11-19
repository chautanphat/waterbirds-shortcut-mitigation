# Counterfactual tools: integration map

The working checkout is the repository containing this file. The handoff was read from the user-designated sibling checkout; implementation stays here. Existing uncommitted reports, galleries and images were preserved.

| Existing implementation | Reused integration |
|---|---|
| `scripts/image_mixer.html`: `load`, `aligned`, `blend`, `render`, undo-independent export | Existing plain HTML/canvas UI retained. `aligned` and weighted-alpha `blend` extracted without algorithm changes into `scripts/mixer_core.js`; Mixer and Lab's mask-overlay preview use it. Scientific compositing uses explicit foreground alpha, never the Mixer's whole-photo blend. |
| `scripts/poter_preview.py`: `create_app`, `/api/predict`, `/api/sample` | Runner now delegates loading and batch probabilities to `src/counterfactual/model.py:ModelAdapter`. Its HTTP payload, labels, metadata matching and PNG previews remain intact. |
| `src/evaluation/prediction_export.py:load_image_model` | Strict state-dict loading reused; no ImageNet download, retraining, head substitution or POTER baseline changes. |
| `src/datasets/waterbirds.py:waterbirds_transform` | Reused square 256 resize, 224 center crop, float32 conversion, ImageNet normalization. Preview is split from this same transform, not a second implementation. |
| `baselines/poter/data/cub_dataset.py:get_transform_cub` | Inspected to confirm square evaluation resize and crop match Runner. Preserved. |
| `src/evaluation/dpe_export.py:predict_heads` | Optional original-DPE adapter reuses equal head-softmax averaging, row z-score, L2 prototype distances and temperature 1. Requires exact backbone and metadata hashes. |
| `src/models/resnet.py:ResNetClassifier` | Existing ERM loader and feature extraction reused for optional ERM/DPE. |
| `data/waterbirds/metadata.csv` | Canonical metadata retained byte-for-byte. Checkpoint's metadata hash checked before inference. |
| `checkpoints/poter_kaggle_recovered/CUB/Clean/POTER_CUB_seed1_kaggle/best_validation.pt` | Actual validation-selected ResNet-50, two outputs `[landbird, waterbird]`, epoch 258 (zero-based). No test-label selection. |

Both new entrypoints use `src/counterfactual/server.py`, the existing standard-library threaded HTTP architecture, with static HTML/CSS/JS. They share foreground/background repositories, image operations, metrics, storage, exports and the model adapter. No web service receives research images.

Implementation sequence: canonical download/join audit → mask and source verification → shared inference/compositing → factorial storage → Lab and Explorer → scientific-contract and real-inference tests. `tests/test_counterfactual.py` includes an independent frozen reference of Runner's pre-refactor loading/transforms, plus its live HTTP contract, single/batch parity, and the full 4×2 experiment.

No pre-existing Mixer or Runner automated test suite was found. New tests exercise Runner's public API and compare the extracted Mixer source functions; browser verification covers the new interfaces.
