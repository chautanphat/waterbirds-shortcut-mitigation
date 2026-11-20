# AutoBackSwap and ERM seed 0 weights (retrain of 2026-11-20)

Weights are outside Git, as required by `docs/model_artifact_contract.md`. This file only records where they are and their SHA-256. It is not a `model.json`; no experiment ID has been assigned yet, and it does not claim the artifacts are verified or complete under that contract.

- Location: Google Drive folder, anyone with the link can view: https://drive.google.com/drive/folders/1yZIUqhab3SHWEKNpvbqAF5Dg_gT-WEMf
- Source: Kaggle notebook `kaggle_autobackswap_full.ipynb` of `ndmhuy/AutoBackSwap-repro` (Save & Run All, GPU T4 x2, spur-corr commit b060532, `--cuda_optimizations=false`), seed 0.
- Stack: Python 3.12, torch 2.11.0+cu126, torchvision 0.26.0+cu126.
- Files are `checkpoint.pt` and `final.pt` as saved by the training script. Which of them is the best-validation or the last-epoch state has not been checked; do not treat them as interchangeable.

| run | WGA | overall acc |
|---|---|---|
| ours_minority_false_seed0 | 92.195 | 94.028 |
| ours_minority_true_seed0 | 92.368 | 95.444 |
| erm_minority_false_seed0 | 34.900 | 69.762 |
| erm_minority_true_seed0 | 77.414 | 93.890 |

Scores are from the notebook log; the notebook's check against the dumped predictions reported no mismatch. The Drive folder also has `failure_viewer.zip` (1896 misclassified test images) and `SHA256SUMS.txt`.

## SHA-256

```text
edba28d4016c0fa9e823369633e897ca26b977c724f17a072c36a6d1049ca653  erm_minority_false_seed0/checkpoint.pt
e5e68710af19d16fe5bde8325763ef999da3e82d9f3749d11817a14bd7b1f55e  erm_minority_false_seed0/final.pt
684c040ed4c3855adc3f03dd60ec34ca108bf0b8c6a97aa721296ddf2d4c0dd5  erm_minority_true_seed0/checkpoint.pt
77beffbda5ba6e898a989b8c7d4315e3aa20c678690289d918ba89dc66aa5318  erm_minority_true_seed0/final.pt
868742b8f4460d6b658b7d10fa5ddbdd3567cf93ab198beb4218d2847f303d1d  ours_minority_false_seed0/checkpoint.pt
4bfab2588173db431536d81a340532c913067e02f4ba30c225f003afdc5dad0c  ours_minority_false_seed0/final.pt
284a190f780f0ee263617663344a5396cc741da191bc7b895bf1addab2fc73dc  ours_minority_true_seed0/checkpoint.pt
d638995700f2a220dfe4000e6ebfdc3424e30df71b5987716b805cd6f810c5de  ours_minority_true_seed0/final.pt
a39a2a01fbbc1209befc219593b28c640f1776350b3660f5bc4eaef7c1abb690  failure_viewer.zip
```
