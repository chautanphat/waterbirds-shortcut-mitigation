# Historical two-pair audit

The repository CSVs are the source for this audit. The separate uploaded `predictions(6).csv` was not present at the named local locations. The repository originals reproduce the reported results without using rounded summaries.

The join verifies 5,704 unique already-exposed test bird IDs, 11,408 predictions, and exactly one Land and one Water view per bird. All paired foreground layer and alpha hashes match. Saved labels, groups, probabilities, argmax classes, and all paired differences reconcile. Exactly 370 birds change class.

| Pair | Land scene | Water scene | Foreground class | N | Mean Δp(Waterbird) | Flips |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 0 | train-bg-7127 | train-bg-5143 | 0 | 2260 | +0.076722688 | 253 |
| 0 | train-bg-7127 | train-bg-5143 | 1 | 592 | +0.048329448 | 10 |
| 1 | train-bg-6661 | train-bg-1324 | 0 | 2250 | -0.009896250 | 104 |
| 1 | train-bg-6661 | train-bg-1324 | 1 | 602 | -0.035549136 | 3 |

The two Landbird signs are opposite. The past foreground count cannot compensate for the four scene identities. New testing uses a fully crossed scene design.

- `paired_cases.csv` SHA-256 `e1699639eec6451a4962a881011b4644be506a8c4be3fc08f8659916f40e3f00`
- `predictions.csv` SHA-256 `1b17fc80c1a5843ad95580b4810b5518b633b8dadd3ea52b9fb33b3c636a1880`
