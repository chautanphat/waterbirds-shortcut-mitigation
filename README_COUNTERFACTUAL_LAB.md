# Counterfactual Lab

From the repository root in PowerShell, use a Python environment with a compatible Torch/torchvision pair:

```powershell
python -m pip install -r requirements-counterfactual.txt
python scripts/setup_waterbirds_data.py --download-cub
python scripts/counterfactual_lab.py --device cuda
```

Open [Counterfactual Lab](http://127.0.0.1:8786). Use `--device cpu` if needed. In the repository-local environment created during implementation, substitute `.\.venv\Scripts\python.exe` for `python`. Default ports 8786/8787 keep the existing Runner separate. Launches are offline; setup supports `--skip-download` for repeat validation and `--cub-root <directory>` for existing original RGB/masks. The official Hugging Face dataset is installed automatically; no manual drag-and-drop setup is required.

Setup pins both original and Parquet commits. Subsequent setup runs verify the materialized snapshot hashes and avoid network downloads. `--skip-download --prune-download-cache --cache-only` removes redundant Parquet bundles while retaining original JPEGs and their hash inventory; this cleanup was executed locally. `--force-download` bypasses the materialized cache and repairs HF image copies from pinned bundles. A custom original revision requires its matching `--parquet-revision` or the direct `--representation files` route.

Select one or more canonical birds, balanced Land/Water backgrounds or a seeded quota, models, opacity and severity. **Run all** persists the entire matrix; **Compose only** saves images with `status=not_run` and empty probability fields. Presets are clean 0, mild 1.5, moderate 4 and strong 9 Gaussian sigma pixels in the final model view. Custom sigma is saved separately. Foreground geometry is fixed on a 256-square canvas; Runner's exact center 224 crop follows. There is no rescale between blur and the model crop. Background alignment offers the Mixer's cover/contain/stretch choices; contain uses logged gray padding.

Foreground resampling uses premultiplied alpha. Blur uses normalized convolution of foreground color times alpha divided by blurred alpha, retaining original alpha support. Every Land/Water pair reuses byte-identical RGBA layers, masks, geometry and opacity. Opaque foreground and outside-support background pixels must pass exact invariance checks. Antialiased edges depend on their background through alpha, as expected. Each run saves raw composites, exact 224 RGB previews, normalized tensor hashes, source/mask assets, manifests, model checkpoint/preprocessing fingerprints, config, all probabilities/logits, summaries, pair metrics, and PNG/PDF case packets.

After the primary Gaussian gates passed, foreground contrast and partial occlusion were enabled as separate operators. Contrast factors are clean 1, mild .75, moderate .4 and strong .1, measured around the alpha-weighted bird-color mean. Occlusion covers 0, .1, .3 and .5 of alpha-support pixels with RGB 127, using a fixed source-seeded center and nested radial regions. Both preserve alpha and geometry across backgrounds. A fully occluded custom variant is excluded from valid-target accuracy. Each run contains **one operator**; contrast/occlusion require sigma 0 and saved `operator_strengths`. Their charts are never merged with Gaussian severities. Opacity remains independent.

Approximate HF-matte or manually corrected masks require **Allow approximate exploratory cutouts**. Original CUB pixels/masks are preferred and enabled by the setup command above. Manual mask corrections support brush add/remove, undo/reset and immutable version history. External image + explicit mask import preserves unknown ground truth; accuracy, failures and WGA stay undefined. No-bird backgrounds and neutral controls are clearly marked OOD and never enter natural-image accuracy.

Reproduce the known Waterbird demo from a saved config:

```powershell
python scripts/run_counterfactual_experiment.py --config configs/counterfactual_lab/waterbird_demo.json --device cuda
python scripts/run_counterfactual_experiment.py --config configs/counterfactual_lab/contrast_demo.json --device cuda
python scripts/run_counterfactual_experiment.py --config configs/counterfactual_lab/occlusion_demo.json --device cuda
```

The demo has 8 natural interventions, 2 background-only controls, 4 neutral controls and 1 separately evaluated original-scene baseline (15 rows). Every prediction records its probability change and flip relative to that original. Reload results after restart using the saved-run selector. Re-run/resume creates a new immutable run with parent ID, preserving earlier runs. CLI supports `--resume --config <saved-run>/run_config.json`. Use the same checkpoint and recorded software versions for numerical reproduction. The UI checks checkpoint, DPE-head and preprocessing hashes before rerunning a saved configuration.

Optional adapters:

```powershell
python scripts/counterfactual_lab.py --erm-checkpoint checkpoints/your_erm.pt
python scripts/counterfactual_lab.py --dpe-heads checkpoints/original_dpe_heads.pt --dpe-backbone checkpoints/exact_erm_backbone.pt
```

DPE requires the repository's `src.evaluation.dpe_export.save_dpe_heads` format, canonical metadata hash and matching backbone SHA. No training/conversion of unbound artifacts or mislabeling POTER as DPE occurs. DPE mean-probability scores are logged as **log probability**, not native network logits. POTER remains unchanged in `baselines/poter/`.

Local Places manifest example (`path` relative to manifest location):

```csv
path,bg_type,scene_category,original_split,license_note
land/example.jpg,land,bamboo_forest,test,local authorized research asset
water/example.jpg,water,ocean,test,local authorized research asset
```

```powershell
python scripts/import_counterfactual_assets.py backgrounds --manifest local_places.csv
```

Tests:

```powershell
python -m unittest discover -s tests -v
$env:RUN_REAL_INFERENCE="1"
python -m unittest discover -s tests -p test_counterfactual.py -v
```

Full dataset preparation and the real checkpoint are required only for the opt-in real tests. Local HTTP tests need localhost socket access. The source code and exact preprocessing parity tolerances are in `tests/test_counterfactual.py`.

Frontend configuration and existing Mixer contracts can also be checked with `node tests/test_lab_controls.cjs` and `node tests/test_mixer_core.cjs`. See `docs/counterfactual_validation.md` for the executed suite, exact artifacts and Windows-only exclusions.
