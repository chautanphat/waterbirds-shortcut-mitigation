# AutoBackSwap on Waterbirds: short report

**Status: partial reproduction.** Waterbirds, ResNet50 only (10 runs). Spawrious and Spurious Vehicles were not run.
Paper: arXiv 2606.32018. Code: `cesro7/spur-corr` at commit `b060532c` (preserved in `baselines/autobackswap/`).
Details and every run: [`runs_log.md`](runs_log.md), [`report.md`](report.md), [`summary.md`](summary.md).

## 1. Reproduction results

Mean over our seeds (paper: 5 seeds, mean ± std). WGA = worst-group test accuracy, Acc = overall test accuracy on all 5794 test images.

| Setting | Seeds | WGA (ours / paper) | Acc (ours / paper) |
|---|---|---|---|
| AutoBackSwap, without minority | 0, 1, 2 | 92.33 / 92.2 ± 0.4 | 94.49 / 95.1 ± 0.3 |
| AutoBackSwap, with minority | 0 | 92.37 / 93.0 ± 0.9 | 95.44 / 95.0 ± 0.5 |
| ERM, without minority | 0, 1, 2 | 33.38 / 33.4 ± 4.4 | 68.68 / 68.4 ± 1.9 |
| ERM, with minority | 0, 1, 2 | 78.45 / 74.9 ± 3.3 | 93.67 / 93.6 ± 0.4 |

- WGA is within 2σ of the paper in every setting.
- Acc of AutoBackSwap without minority is below the paper in all three seeds (94.03, 94.72, 94.72; 3.6σ, 1.27σ, 1.27σ). The cause is **not established**; no experiment isolated it. Differences from the paper setup that were not tested: fp32 instead of bf16 + `torch.compile`, chunked generation, `detector_n_samples` 200 vs 287, the WILDS copy of the data, how the paper defines Acc.
- AutoBackSwap with minority has one seed only.

## 2. Method steps (from the code)

**Evaluation setup (affects the results):**

- Split: the `split` column of `metadata.csv` (0 train, 1 valid, 2 test). No random split, identical for every seed.
- Group = `y*2 + place + 1` (y: 0 landbird, 1 waterbird; place: 0 land, 1 water).

| split | G1 Landbird/Land | G2 Landbird/Water | G3 Waterbird/Land | G4 Waterbird/Water | total |
|---|---|---|---|---|---|
| train | 3498 | 184 | 56 | 1057 | 4795 |
| valid | 467 | 466 | 133 | 133 | 1199 |
| test | 2255 | 2255 | 642 | 642 | 5794 |

- Without minority, training and validation keep only G1 and G4 (background fully correlated with the label). With minority, all four groups.
- Test always has all four groups, as original images. Only training images are background-swapped.
- Checkpoint selection: early stopping on overall validation accuracy (`val_acc`), not on worst-group accuracy. Test is scored once with the best-validation checkpoint.
- Classifier: ResNet50, 50 epochs, SGD (momentum 0.9), label smoothing 0.1, class-weighted loss.

**Steps:**

1. **Detector.** Train an EfficientNet-B0 (ImageNet-pretrained) that predicts, per cell of a 16x16 grid over the 224x224 image, whether it is the bird. Labels come from the CUB segmentation masks of 200 training images from the majority groups (160 train / 40 valid).
2. **Foreground/background separation.** For every image, the detector gives a bird-probability map. Threshold 0.5 gives the "tight" mask (cuts the bird out), threshold 0.9 gives the "loose" mask (covers the whole bird, used for removal).
3. **Inpainting.** AOT-GAN fills the loose-mask region, producing a background-only image.
4. **Classifier training with swapping at load time.** Each time a training image is fetched, pick a random class (landbird or waterbird), then a random inpainted background of that class from the training set. The bird (tight mask) is randomly translated (±32 px), scaled (0.5 to 1.5) and rotated (±30 degrees), then pasted on that background. Every training sample is swapped and gets a new background each epoch, so for the majority groups the background becomes close to independent of the label.
5. **Evaluation** on the original test images as described above.

## 3. Failure cases and observations

Observed by the author (looking at misclassified images; not quantified):
- Errors are similar across seeds.
- Some swapped training images show leftover birds from the background source image, or backgrounds that are not plausible; some objects are blurred or keep an outline from the original image.
- Waterbirds itself is made of pasted birds on other scenes, so blur and outlines are also present in the original data.
- Some waterbird-on-water test images predicted as landbird look strange.

Checked on the predictions (script: [`scripts/autobackswap/species_error_check.py`](../../scripts/autobackswap/species_error_check.py)):
- One species is wrong 27/30 times in seed 0 (and 27, 26 in seeds 1, 2): `102.Western_Wood_Pewee`. `021.Eastern_Towhee` (20/30) and `088.Western_Meadowlark` (17/30) follow. In `metadata.csv` all images of these three species have `y = 1` (waterbird). That these species are land birds is outside knowledge, not checked from the data. The three species make up 90 test images (43 in G3, 47 in G4).
- AutoBackSwap without minority, seed 0: 64 of 90 are wrong, which is 64 of 346 errors, and 37 of the 46 errors in G4. For seeds 1 and 2 it is 46 of 306 each.
- Excluding those 90 images (exploratory only): Acc 95.06 / 95.44 / 95.44 for the three seeds, WGA unchanged because the worst group is Landbird on Water. For ERM without minority Acc moves away from the paper (69.76 to 70.27 vs paper 68.4), so the AutoBackSwap match to 95.1 may be a coincidence and is **not** claimed as the cause of the Acc gap.
- Errors are consistent: 202 test images are wrong in all three AutoBackSwap-without-minority seeds (109 in Landbird on Water).
- Seed 0 error counts: Landbird on Land 75 and Landbird on Water 176 (all predicted waterbird), Waterbird on Land 49 and Waterbird on Water 46 (all predicted landbird).

Hypotheses (not tested):
- Part of the remaining errors comes from imperfect detection or inpainting (leaked birds, implausible backgrounds), so background influence is not fully removed.
- Landbird on Water remains the worst group; whether the background still pulls predictions there is open.

## 4. Planned next experiments

- Look at errors object by object (species, scene, confidence) and separate label-related errors from background-related ones.
- Perturb the background of test images (paint over, cut holes, replace) and measure how accuracy per group changes, to check whether the background influence is removed. Points raised in review and not yet decided: define what result counts as "background removed", and run ERM as a control, because altering the background makes images unfamiliar to both models.
- Inspect the swapped training images directly (they were not examined for this report).

## 5. Limits of this report

- 3 seeds (1 for AutoBackSwap with minority) against the paper's 5 seeds.
- Per-group numbers of the paper were not available, so the Acc gap cannot be located by group.
- The swapped training images were not inspected; statements about their artifacts are the author's observations.
- The species analysis is exploratory and was run after seeing the results.

Misclassified examples for all 10 runs (error images, per-run lists, predictions): https://drive.google.com/drive/folders/143VTxJQRKtkX4wzAzCz2cFLhIrruIoy8?usp=sharing
