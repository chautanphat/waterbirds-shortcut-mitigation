# Experiment Log

Record every meaningful run here. This is an index, not a replacement for configurations, method READMEs, or untracked output files.

## Entry template

```markdown
## <experiment-id> — YYYY-MM-DD

- **Method / version:**
- **Purpose:** smoke test / original reproduction / Waterbirds adaptation / final comparison
- **Branch or commit:**
- **Configuration:** `configs/...` or `experiments/...`
- **Dataset and split:**
- **Seed:**
- **Environment and hardware:**
- **Command:**
- **Output location:** local, untracked path
- **Results:** overall accuracy; per-group accuracy; worst-group accuracy
- **Paper comparison or expected behavior:**
- **Status:** exploratory / smoke test / partial reproduction / verified reproduction / final
- **Notes and follow-up:**
```

Use one entry per meaningful run or a short grouped entry for repeated seeds with the same configuration. Never present a result without its configuration, command, seed, and evaluation context.

Use repository-relative paths in every entry. Link implementation changes to `baselines/` or `src/methods/`, tracked settings to `configs/`, run metadata to `experiments/`, and lightweight summaries to `results/`; keep large outputs and checkpoints local and untracked.

---

## exp-dpe-001-smoke — 2026-11-01

- **Method / version:** Diverse Prototypical Ensembles (DPE) / commit `68b811461e238d44cced6ae6c8ce75b66cad5449`
- **Purpose:** smoke test
- **Branch or commit:** `main`
- **Configuration:** `configs/dpe4subpop/synthetic.yaml`
- **Dataset and split:** Synthetic 4-group subpopulation benchmark (D=256, 400 samples/split: train/val/test)
- **Seed:** 0
- **Environment and hardware:** Windows 11, Intel Core i5-12450H CPU, Python 3.11, PyTorch 2.5.1
- **Command:** `python baselines/dpe4subpop/reproduce_demo.py` & `python tests/test_dpe_smoke.py`
- **Output location:** Temporary execution buffers (untracked)
- **Results:**
  - Overall accuracy: 92.25%
  - Balanced accuracy: 92.24%
  - Worst-group accuracy: 90.62%
  - Per-group accuracy: Group 0: 93.26%, Group 1: 93.75%, Group 2: 90.62%, Group 3: 91.26%
- **Paper comparison or expected behavior:** Verifies that IsoMax+ distance classifier, covariance regularization diversity penalty, bootstrap class-balanced sampling, and probability averaging converge and evaluate correctly across multiple stages without errors.
- **Status:** smoke test
- **Notes and follow-up:** Successfully validated core DPE pipeline in ~8 seconds on CPU. Ready for full Waterbirds feature extraction and benchmark run.

