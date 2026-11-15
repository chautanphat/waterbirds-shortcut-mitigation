# Cross-dataset DPE post-hoc study: how to run it

What and why: [`PLAN.md`](PLAN.md). This folder holds everything needed to run that plan on Kaggle.

| File | Role |
| --- | --- |
| `backbones_kaggle.ipynb` | GPU T4 x2. Prepares one dataset, runs a timing smoke test, trains two SubpopBench ERM backbones (one per GPU) and saves frozen val/test features. |
| `nested_kaggle.ipynb` | CPU. Runs the strictly nested post-hoc study on the attached backbone outputs and, once every cell is done, the declared analysis. |
| `build_notebooks.py` | Regenerates both notebooks from `code/`. Run it after any code change. |
| `code/xdpe/` | Dataset-agnostic port of the Waterbirds strict nested pipeline (`experiments/dpe_nested_corrected/code/corrected_nested.py`). |
| `code/summarize.py` | The declared analysis (H1–H4, decision rule). Do not edit after the freeze commit. |
| `code/train_backbone.py`, `code/prepare_data.py` | Backbone training + feature extraction; data layout with SubpopBench's own metadata generators. |
| `code/tests/` | `equivalence.py` (port vs. original on Waterbirds), `integration.py` (synthetic end to end), `synthetic_summary.py` (analysis with known answers). |

## Order of work

1. **Timing smoke (step 0.1–0.2).** Open `backbones_kaggle.ipynb`, set `DATASET`, attach the data
   (CelebA: Kaggle dataset `jessicali9530/celeba-dataset`), keep `RUN_SMOKE = True`, **Save & Run All**.
   The smoke part prints `smoke_report.json`: seconds per step, an estimate per 30,001-step backbone,
   the verified split/group counts and whether SubpopBench resume works. Copy those numbers into
   PLAN.md Sect. 12. If two backbones do not fit in one session, the notebook stops before training
   unless `ALLOW_RESUME = True` (see "Resume" below).
2. **Pipeline timing (step 0.4).** After the first backbone session finishes, open `nested_kaggle.ipynb`
   (CPU), attach that output, set `MODE = 'smoke'`, run; then `MODE = 'timing'` (full recipe, one cell,
   test not scored). Record the CPU minutes per cell in PLAN.md Sect. 12 and confirm 10 split seeds fit.
3. **Freeze (step 0.5).** Commit PLAN.md with the measured numbers and the final split-seed count,
   together with `code/` and the notebooks; write the commit hash into PLAN.md Sect. 12. Nothing in
   PLAN Sects. 2–7 or in `summarize.py` changes after this commit.
4. **Backbones (step 1).** Seeds 0–4 per dataset: sessions with `SEEDS = [0, 1]`, `[2, 3]`, `[4]`, on any
   of the four accounts. `RUN_SMOKE` can be `False` after the first session of a dataset if that session's
   output (it holds `smoke_report.json`) is attached; otherwise the time guard has no estimate and the
   main run is skipped. MultiNLI: one seed per session.
5. **Nested run (step 2).** `nested_kaggle.ipynb`, `MODE = 'run'`, attach **all five** backbone outputs of
   the dataset. If the 11 h budget ends first, save the version, attach its own output as well and run again:
   finished cells and head checkpoints are reused. The analysis is printed only when all 50 cells exist.
6. **Results (step 3).** Download the bundle `xdpe-<dataset>-run.zip` and put its contents under
   `results/dpe/cross_dataset_<date>/` with a README, as for the earlier runs.

Secondary analysis (PLAN Sect. 7): `nested_kaggle.ipynb` with `DATASET = 'CelebA'`, `VAL_SUBSAMPLE = 1199`.

## Checks already done locally (2026-11-15)

* `tests/equivalence.py`: on the local Waterbirds features (`checkpoints/waterbirds_stage0_standard_erm`),
  one cell through the original `corrected_nested.run_task` and through `xdpe.nested.run_task` gives
  identical results for all 27 arms, all 28 settings, the inner scores, the threshold fits and all
  41 cached head-score arrays (bit for bit), at 1 epoch (split 0) and at the full 20-epoch recipe
  (split 3, about 6 CPU minutes per cell on one core).
* `tests/integration.py`: synthetic CelebA-, CivilCommentsFine- and MultiNLI-shaped backbones through
  `discover`, `run --smoke`, scored cells and `summarize.py` (27, 27 and 25 arms; 4, 16 and 6 groups;
  CivilComments duplicates never split across folds or the 60/40 partition). The 3-class offset map
  also recovers a planted class shift on a controlled problem.
* `tests/synthetic_summary.py`: the analysis reproduces known differences, intervals and decisions, and
  flags a partial run.
* `prepare_data.py` (CelebA path) on full-size synthetic CSVs with SubpopBench's real generator.
* An independent review of the code against SubpopBench's source (2026-11-15), whose findings were
  fixed: a missing `netcal` pin that would crash every training run, SubpopBench's broken `--resume`, an
  infeasible Sect. 7 subsample, the time guard in sessions without a smoke test, and smaller issues.
  The notebook's control cells (smoke, estimate, resume and extract-only, time guard, clean-up) were
  dry-run with stubbed jobs; the Sect. 7 subsample was checked on CelebA's real group sizes (smallest
  balanced head pool at any nesting level: 356 rows, at least one 256-row batch).

Not run locally: GPU training, feature extraction and the CivilComments/MultiNLI downloads. The
backbone notebook's smoke test is the first real check of those.

## Things that differ from the Waterbirds run, and why

* **Group and class counts come from the data.** Group means, folds, the 60/40 partition and the
  head-training interleave run over the groups present (sorted). On Waterbirds this is the original order.
* **Duplicated inputs (CivilCommentsFine).** SubpopBench's fine version repeats a comment once per
  identity it mentions. Rows with the same comment text share a `cluster` and always land in the same
  fold and on the same side of the 60/40 split, so no stage is scored on a copy of its own training input.
  Test WGA keeps SubpopBench's definition (every row counts in its group).
* **Head settings per dataset** from the authors' `train_all_pe.sh` (PLAN Sect. 4). Everything else is the
  Waterbirds standard-ERM head recipe.
* **No DFR-lite** (it served only the label lever).
* **MultiNLI (3 classes):** calibrators of PLAN Sect. 9, fitted with the same 400-step Adam; no exact-WGA
  threshold arms (25 arms instead of 27).
* **Resume.** At the pinned commit SubpopBench's `--resume` always fails (it reads a `best_val_acc` key its
  checkpoints never contain). `train_backbone.py` resumes from a copy of the checkpoint with that key added;
  SubpopBench itself is unchanged. Resuming restores weights and the step counter, not the optimizer state, the
  BERT learning-rate schedule or the data order, so a resumed backbone is recorded in `backbone.json`
  (`resumed_from_step`, `deviation`) and must be reported as such. The notebook resumes only with
  `ALLOW_RESUME = True` and a passing resume check. Prefer finishing each backbone in one session.
* **Extract-only.** If training finished but feature extraction did not, attaching that output lets the next
  session extract features without retraining (not a deviation).
* **MultiNLI memory.** SubpopBench keeps all cached MultiNLI features in every dataset object, so the
  notebook runs one MultiNLI job per session (`PARALLEL_JOBS = 1`); the smoke test shows whether even that fits.
* **Kaggle environment.** For nested runs spread over several sessions, pin the notebook environment
  (Settings -> Environment -> Pin to original environment): checkpoints are bound to the package versions.
  During step 0.4, also check that the saved output keeps all checkpoint files (about 130 per cell).
* **torch 2.6 and pickles.** Training processes set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` because SubpopBench
  loads its own checkpoints and the MultiNLI feature pickles with `torch.load` defaults.

## Test-set rules in practice

The backbone logs contain ERM test accuracy (SubpopBench prints it). The nested notebook writes test
scores only after each cell's fits and selections are saved, prints nothing about them during the run,
and summarises only a complete run. Do not open `results/*/split*.json` before the run is complete.
