# Phase 0 audit

The requested `predictions(1).csv` was not attached/found. Audited the preserved 400-row preliminary run instead, with core SHA-256 verification.

16 foreground IDs, 2 species, 4 natural backgrounds, 340 unique raw images. 256 natural composites + 64 background-only rows (4 unique images repeated across 16 birds) + 64 neutral controls + 16 originals. No missing or duplicate prediction keys; repeated controls are not independent.

Labels: 0=Landbird, 1=Waterbird; group=2*y+background (Land=0, Water=1). Gaussian sigma is 0, 1.5, 4, 9 pixels in the model crop. Premultiplied bilinear geometry is fixed; normalized Gaussian convolution operates on masked foreground RGB only. Alpha bytes are unchanged across severity/background; outside-support background and opaque foreground equality assertions passed for all 16 birds.

Original CUB segmentation has been checksum verified. Pixel assertions do not establish semantic mask quality; segmentation artifacts remain possible. New representative masks/composites are saved for inspection.

Known official label issue: 021.Eastern_Towhee, 088.Western_Meadowlark, 102.Western_Wood_Pewee. The upstream Tern substring rule matches Eastern/Western. Affected counts by split: {'2': 90, '0': 72, '1': 18}; preliminary affected rows=0. Official labels remain unchanged; corrected and species-excluded sensitivity analyses are secondary. The new 100-image replication excludes these known ambiguous species and the original 16 foregrounds, before any new predictions.

Direct CSV recomputation:

|Severity|n|Accuracy|WGA|Landbird|Waterbird|
|---|---:|---:|---:|---:|---:|
|clean|64|0.968750|0.937500|0.937500|1.000000|
|mild|64|0.703125|0.312500|0.406250|1.000000|
|moderate|64|0.531250|0.000000|0.062500|1.000000|
|strong|64|0.546875|0.000000|0.093750|1.000000|
