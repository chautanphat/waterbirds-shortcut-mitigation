# Direction 2 · Interaction-Targeted Minimal Representation Repair (`rep_repair`)

Project-owned mechanism study and representation repair method addressing **when and why ERM representations fail under background shifts** — spanning readout falsification, CUB counterfactual probes, causal block patching, `PreLayer4RepairModel`, `ViT-B/16` transfer, `Spawrious (o2o_hard)` OOD generalization, and a local zero-GPU statistical audit.

## Architecture Overview

Causal localization (Step 5) shows that ResNet-50 preserves foreground–background spatial separation through `layer3.5` ($14 \times 14$), then irreversibly mixes background activation into foreground spatial cells via `layer4.{0,1,2}` $3\times 3$ residual convolutions (`0.851` Layer-4 foreground drift on persistent ERM failures vs. `0.686` on 400-epoch `ERM*`).

```text
Input x ──▶ [layer1 → layer2 → layer3] (100% frozen convs; 14×14 spatial separation intact)
                   │
                   ▼  PreLayer4SpatialGate (1×1 -> ReLU -> 1×1 -> Sigmoid, mask-free 14×14 gate)
              g' = g / max_hw(g)   (peak_neutral: preserves peak foreground magnitude!)
              h' = g' ⊙ h + (1 - g') ⊙ neutral_c3   (neutral_c3 = fold-train group-balanced mean)
                   │
                   ▼  [layer4] (convs frozen; only BN affine γ,β adapted: bn_all)
                   ▼  [GAP → Linear Head]
                   │
      Losses: L_CE(x) + 0.05 · ||f_4(x) - f_4(x_bg_swap)||_cos
                      + 0.10 · JS(p(x), p(x_bg_swap))
                      + 0.005 · mean(g)
```

Key design files in [repair.py](repair.py):
- `PreLayer4SpatialGate`: 2-layer $1\times 1$ spatial attention predictor producing a $[B, 1, 14, 14]$ mask-free gate without ground-truth segmentation supervision.
- `PreLayer4RepairModel`: Wraps a frozen ResNet-50 backbone (`gate_mode="peak_neutral"`, `bn_scope="bn_all"`, `head_type="linear"`) with group-balanced `neutral_c3` registered as a persistent buffer.
- `context_invariance_loss`: Pairs each training sample with an in-batch same-class, different-background counterpart inside `layer4` (implemented via `F.adaptive_avg_pool2d` for shape generality).

---

## Frozen Hyperparameters (`pre_l4_repair_w005`)

Frozen strictly on cleaned Waterbirds validation (`split == 1`) after Step 6b — never retuned on Waterbirds test or Spawrious OOD:

| Parameter | Value | Parameter | Value |
|---|---|---|---:|
| `gate_mode` | `peak_neutral` | `bn_scope` | `bn_all` (convs 100% frozen) |
| `head_type` | `linear` | `lr` / `epochs` / `batch_size` | `3e-3` / `20` / `64` (AdamW) |
| `drift_weight` | `0.05` | `cons_weight` | `0.10` |
| `gate_sparsity` | `0.005` | `trainable_params` | `124,931` (2-cls) / `129,029` (4-cls) |

---

## Module & Script Inventory

### Core Library (`src/`)
- [phase_a_data.py](phase_a_data.py): Zero-test-leakage validation loader (`safe_val_file`), `"tern"` label cleaning (`official` / `relabel` / `drop`), deterministic 5-fold group-stratified splits, and `group_metrics()` (returns `[0, 100]` percentages).
- [phase_b_assets.py](phase_b_assets.py): Kaggle backbone recovery, Waterbirds / CUB-200 segmentation downloader, deterministic `eval_image_transform()`.
- [controls.py](controls.py): Parameterized Phase B controls (`DARHead`, `OracleMaskPoolHead`, `H2T-DFR`, `BN-Stat-Only`, `Layer4-FT`, `Affine-DFR`).
- [counterfactuals.py](counterfactuals.py): 5-view CUB-masked counterfactual image synthesis (`orig`, `matched_bg`, `flipped_bg`, `neutral_gray`, `ood_bg`) and foreground-cell relative drift computation.
- [patching.py](patching.py): Causal block-level activation patching (`bg_only` / `fg_only` / `full`) and bottleneck residual-branch vs. skip decomposition.
- [repair.py](repair.py): `PreLayer4SpatialGate`, `PreLayer4RepairModel`, `LoRAConv2d`, and `context_invariance_loss`.
- [src/models/vit.py](../../models/vit.py): `ViTClassifier` and `PreBlockViTRepairModel` for cross-architecture ViT-B/16 mechanism localization (`block_0..block_11`) and late-LN / pre-`block_6` repair.
- [src/datasets/spawrious_o2o.py](../../datasets/spawrious_o2o.py): `Spawrious (o2o_hard)` downloader, sealed 85/15 train/val split loader (`assert_not_o2o_hard_test` guard during Steps 7b), and 25,344-image shifted OOD test loader (`Step 7c` only).

### Execution Scripts (`scripts/analysis/`)
1. [rep_repair_failure_audit.py](../../../scripts/analysis/rep_repair_failure_audit.py) — Phase A Step 1: Error overlap & `"tern"` mislabeling audit.
2. [rep_repair_frozen_readouts.py](../../../scripts/analysis/rep_repair_frozen_readouts.py) — Phase A Step 2: Frozen post-GAP readout falsification (`DFR`, `MLP`, `RBF-SVM`, `Cosine kNN`, `DPE`).
3. [rep_repair_phase_b_controls.py](../../../scripts/analysis/rep_repair_phase_b_controls.py) — Phase B Step 3: Non-linear / spatial / BN control ladder (`DAR`, `Oracle-Mask-Pool`, `H2T-DFR`, `Layer4-FT`, `Affine-DFR`).
4. [rep_repair_phase_b_counterfactuals.py](../../../scripts/analysis/rep_repair_phase_b_counterfactuals.py) — Phase B Step 4: 5-view CUB counterfactuals & stage-wise FG drift.
5. [rep_repair_phase_c_patching.py](../../../scripts/analysis/rep_repair_phase_c_patching.py) — Phase C Step 5: Causal block patching, branch decomposition, channel-bleed audit, BN stage ablation.
6. [rep_repair_phase_c_step6.py](../../../scripts/analysis/rep_repair_phase_c_step6.py) — Phase C Step 6: Initial Pre-`layer4` gate & invariance 6-arm ablation.
7. [rep_repair_phase_c_step6b.py](../../../scripts/analysis/rep_repair_phase_c_step6b.py) — Phase C Step 6b: `peak_neutral` Pre-`layer4` repair vs. `layer4.conv2` LoRA ablation (freezes `pre_l4_repair_w005`).
8. [rep_repair_step7a_vit_diag.py](../../../scripts/analysis/rep_repair_step7a_vit_diag.py) — Phase C Step 7a: Train 3 `ViT-B/16` ERM seeds + block-wise token patching & counterfactual mechanism diagnostic.
9. [rep_repair_step7b_spawrious.py](../../../scripts/analysis/rep_repair_step7b_spawrious.py) — Phase C Step 7b: Train 3 ResNet-50 `Spawrious (o2o_hard)` ERM seeds + validation control ladder.
10. [rep_repair_step7c_waterbirds_test.py](../../../scripts/analysis/rep_repair_step7c_waterbirds_test.py) — Phase C Step 7c-1: One-shot held-out Waterbirds test (`ResNet-50` + `ViT-B/16`).
11. [rep_repair_step7c_spawrious_test.py](../../../scripts/analysis/rep_repair_step7c_spawrious_test.py) — Phase C Step 7c-2: One-shot sealed OOD `Spawrious (o2o_hard)` test (`25,344` images).
12. [rep_repair_step7d_posthoc_audit.py](../../../scripts/analysis/rep_repair_step7d_posthoc_audit.py) — Phase C Step 7d: **Zero-GPU local** threshold-sensitivity sweep, 2,000-replicate paired stratified bootstrap CIs, and Spawrious spurious-BG confusion audit.

---

## Public Pretrained Checkpoints & Reproducibility

All backbone checkpoints used by Step 7c are publicly hosted on Kaggle Outputs (no retraining needed):

| Cohort | Kaggle Kernel Output Slug | Seeds | Role |
|---|---|---:|---|
| Waterbirds `standard_erm` ResNet-50 | `noerula/dpe-mlp-dropout-verification` | 0–4 | 100-epoch SubpopBench deterministic ERM |
| Waterbirds `erm_star` ResNet-50 | `noerula/dpe-exploration` | 0–1 | 400-epoch augmented `ERM*` upper bound |
| Waterbirds `ViT-B/16` ERM | `noerula/rep-repair-step7a-vit-kaggle` | 0–2 | Cross-architecture mechanism & test transfer |
| Spawrious `o2o_hard` ResNet-50 | `noerula/rep-repair-step7b-spawrious-kaggle` | 0–2 | Unseen OOD background shift generalization |

Exact SHA-256 digests are recorded in [Waterbirds provenance](../../../results/rep_repair/step7c_waterbirds_test_20261124/provenance.json) and [Spawrious provenance](../../../results/rep_repair/step7c_spawrious_test_20261124/provenance.json).

> **Note on `.gitignore` & `.npz` probability caches:** Per repo conventions (`*.npz`, `*.pt`), large binary checkpoint and raw probability cache files (`step7c_waterbirds_test_probs.npz`, `step7c_spawrious_ood_probs.npz`) are excluded from Git. Running `rep_repair_step7c_waterbirds_test.py` and `rep_repair_step7c_spawrious_test.py` regenerates both `.npz` files automatically; running `rep_repair_step7d_posthoc_audit.py` locally consumes those `.npz` files on CPU in `< 2 min`.

### Local CPU Reproduction Commands

```bash
# 1. Install dependencies (CPU or CUDA torch + analysis libs)
pip install -r requirements.txt

# 2. Run all Direction 2 unit tests (35+ checks across Phases A-C + Step 7d)
python -m unittest discover -s tests -p "test_rep_repair_*.py" -v

# 3. Re-run Step 7d post-hoc audit (requires local step7c_*_probs.npz + Waterbirds metadata.csv)
python scripts/analysis/rep_repair_step7d_posthoc_audit.py
```

For GPU execution order on Kaggle (Phases A -> C), see [notebooks/README.md](../../../notebooks/README.md).

---

## Known Limitations & Scope Boundaries

1. **Spatial background-shortcut scope:** The localized `layer3 -> layer4` mixing mechanism targets foreground-vs-background shifts rather than non-spatial attribute shortcuts (e.g., demographic gender/age attributes on CelebA or text shortcuts on MultiNLI).
2. **Waterbirds scalar threshold variance (`rel_val_thr_wga`):** Optimizing a discrete 1D threshold on the small Waterbirds validation minority split (`266` total rows) produced boundary thresholds (`0.05`, `0.44`) on two seeds, yielding `92.52%` vs. `Affine-DFR`'s `93.75%` (`rel_val_thr_wga`), even though `pre_l4_repair_w005` wins `5/5` seeds on threshold-free Min-Grp AUROC (`98.95%`), `4/5` on Raw `p=0.5` WGA (`91.22%`), and `5/5` on diagnostic ceiling (`95.24%`).
3. **Spawrious worst-class residual BN trade-off:** `pre_l4_repair_w005` leads overall OOD Mean Accuracy (`88.14%`, `3/3` vs. `Affine-DFR` `+3.30 pp`), cuts non-spurious errors lowest (`1.82%`), and rescues `+5.58 pp` Worst-Class over `affine_dfr_bn_all`, yet trails frozen-backbone `dfr_l2` by `1.44 pp` on single worst class `dachshund @ snow` (`70.92%` vs. `72.36%`) because adapting `bn_all` slightly sensitizes the hardest OOD breed.
4. **ViT global self-attention boundary:** While `ViT-B/16` exhibits the same late-stage spatial crossover (`block_6..8`), global attention mixes background tokens into `[CLS]` from `block_0`. Consequently, gating spatial patch tokens before `block_6` does not retroactively clean `[CLS]`, making late-block LayerNorm recalibration (`vit_ln_dfr_late`, `91.52%` Cln WGA) superior to patch-only gating (`88.16%`) on ViT.
