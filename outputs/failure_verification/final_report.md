# POTER Failure Verification: Experiments1 &2

## 1. Executive summary

**Experiment1 đã hoàn thành; Experiment2 chưa hoàn thành xác nhận đa seed.** POTER seed1, Landbird tại sigma2: natural−gray interference **-21.00 [-30.00; -12.00] pp**; amplification so với clean **-15.00 [-25.00; -5.00] pp**. Đây là bằng chứng can thiệp input có điều kiện trên checkpoint và bốn nền đã chọn. Chưa xác nhận tính ổn định giữa các seed huấn luyện độc lập.

Land→Water tại sigma2 cho Landbird có signed ΔP(Water) **-8.77 [-10.72; -6.76] pp**. Nền Water trong ngân hàng này làm giảm P(Water) so với Land; không suy ra cơ chế water-background→Waterbird bias.

**Quyết định: NO-GO / INCONCLUSIVE** cho chuyển sang Mechanism Discovery ở cấp phương pháp hoặc phát triển mitigation. Cần xác nhận đa checkpoint và rà soát artifacts/recognizability trước. Không triển khai phương pháp robustness mới.

Không có training hoặc inference mới trong verification này. Experiment2 tái sử dụng6.300 rows của ba checkpoint hiện có (2.100/model:1.600 natural,400 gray,100 original), so với mục tiêu18.900. Thiếu12.600 rows từ6 checkpoint độc lập còn thiếu. Yêu cầu mới nhất của người dùng: báo trước và chờ xác nhận nếu cần training, người dùng không thể train local.

## 2. Experiment1 results

| Model | Lớp | Sigma | Interference [CI95%] (pp) | Amplification [CI95%] (pp) |
|---|---|---|---|---|
| POTER | landbird | 0.0 | -6.00 [-12.50; -1.00] | +0.00 [+0.00; +0.00] |
| POTER | landbird | 2.0 | -21.00 [-30.00; -12.00] | -15.00 [-25.00; -5.00] |
| POTER | waterbird | 0.0 | +0.50 [-1.50; +3.00] | +0.00 [+0.00; +0.00] |
| POTER | waterbird | 2.0 | +2.00 [+0.00; +6.00] | +1.50 [+0.00; +4.00] |
| ERM enhanced | landbird | 0.0 | +3.50 [-3.00; +10.50] | +0.00 [+0.00; +0.00] |
| ERM enhanced | landbird | 2.0 | +6.00 [-4.00; +17.00] | +2.50 [-7.00; +13.00] |
| ERM enhanced | waterbird | 0.0 | -4.00 [-8.00; -1.00] | +0.00 [+0.00; +0.00] |
| ERM enhanced | waterbird | 2.0 | -18.00 [-23.00; -12.00] | -14.00 [-20.00; -7.00] |
| DPE legacy | landbird | 0.0 | +3.00 [-4.00; +10.01] | +0.00 [+0.00; +0.00] |
| DPE legacy | landbird | 2.0 | +9.00 [-0.50; +19.00] | +6.00 [-4.00; +16.01] |
| DPE legacy | waterbird | 0.0 | -0.50 [-1.50; +0.00] | +0.00 [+0.00; +0.00] |
| DPE legacy | waterbird | 2.0 | +1.00 [-3.00; +6.00] | +1.50 [-2.00; +6.50] |

Các background replicates được giữ riêng trong exp1_paired_interventions.csv, sau đó trung bình trong source foreground trước CI. exp1_foreground_effects.csv giữ effect từng foreground; exp1_species_summary.csv giữ từng species. Paired flip, signed/absolute probability shift và margin shift được tính cho cả gray→Land, gray→Water, Land→Water. Xem exp1_summary.md và statistical_analysis.csv để đọc mọi sigma/contrast/class, bao gồm source và species sensitivity bootstrap.

Tại sigma2/Landbird, POTER có gray→Land flip rate +28.00 [+17.00; +39.00]%, gray→Water +14.00 [+6.00; +23.00]%, Land→Water +18.00 [+10.00; +26.00]%. Điều kiện Land có Landbird accuracy58%, Water72%, gray86%. Water thay vì Land giảm P(Water) và tăng correctness trong các cặp này, dù cả hai nhóm natural backgrounds có accuracy thấp hơn gray. Hướng effect phụ thuộc contrast; không gọi đây là một semantic water-cue mechanism đã xác nhận.

![Interference](landbird_vs_waterbird_interference.png)

## 3. Experiment2 results

| Model | Seed có sẵn | Seed còn thiếu | Trạng thái |
|---|---|---|---|
| POTER | 1 | 0, 2 | 1/3; chưa xác nhận đa seed |
| ERM enhanced | 0 | 1, 2 | 1/3; chưa xác nhận đa seed |
| DPE legacy | 0 | 1, 2 | 1/3; chưa xác nhận đa seed |

Có hai image backbones duy nhất: POTER và enhanced ERM; DPE dùng chung backbone ERM. Trong từng method chỉ có một training seed hợp lệ. Bản sao byte-identical không tăng n; các head-fitting seed/fold không thay thế image-backbone training seeds. last_training.pt là snapshot cùng POTER seed1; checkpoint bs16/smoke cũng seed1 và khác recipe. Log POTER seed2/3 trên D chỉ dừng ở epoch0, chưa có checkpoint; nguyên nhân lỗi không được ghi trong log.

Downloads có thêm standard-ERM seed3 và DPE partition0 tương ứng, với hash khớp archived lineage. Chúng khác enhanced-ERM400 và DPE legacy/full-validation recipe, nên không dùng để giả hoàn thành seed target. Manifest còn ghi các recovery-test artifacts và những head-only files đã loại. Một số manifest tham chiếu checkpoint ở đường dẫn /kaggle/working nhưng file không có trong checkout; đường dẫn metadata không được tính như một checkpoint đã có/evaluate.

### Từng checkpoint hiện có

| Model | Seed | Sigma | AA (%) | WGA (%) | Landbird (%) | Waterbird (%) | Asymmetry (pp) |
|---|---|---|---|---|---|---|---|
| POTER | 1 | 0.0 | 93.25 | 86.00 | 88.00 | 98.50 | 0.00 |
| POTER | 1 | 0.5 | 92.00 | 83.00 | 86.00 | 98.00 | 1.50 |
| POTER | 1 | 1.0 | 88.75 | 74.00 | 78.50 | 99.00 | 10.00 |
| POTER | 1 | 2.0 | 82.50 | 58.00 | 65.00 | 100.00 | 24.50 |
| ERM enhanced | 0 | 0.0 | 94.75 | 87.00 | 93.50 | 96.00 | 0.00 |
| ERM enhanced | 0 | 0.5 | 93.00 | 86.00 | 93.00 | 93.00 | -2.50 |
| ERM enhanced | 0 | 1.0 | 85.25 | 69.00 | 84.00 | 86.50 | 0.00 |
| ERM enhanced | 0 | 2.0 | 74.00 | 39.00 | 68.00 | 80.00 | 9.50 |
| DPE legacy | 0 | 0.0 | 96.25 | 90.00 | 93.00 | 99.50 | 0.00 |
| DPE legacy | 0 | 0.5 | 96.25 | 91.00 | 93.00 | 99.50 | 0.00 |
| DPE legacy | 0 | 1.0 | 90.25 | 80.00 | 81.50 | 99.00 | 11.00 |
| DPE legacy | 0 | 2.0 | 85.00 | 64.00 | 71.00 | 99.00 | 21.50 |

Mean với n=1 bằng đúng giá trị của checkpoint; **SD giữa training seeds không xác định**, không ghi0. exp2_across_seed_summary.csv đánh dấu thiếu6 checkpoint. Foreground-bootstrap CIs trong statistical_analysis.csv không phải training-seed CI. Các plot across_seeds hiển thị một seed/method và ghi rõ1/3, không vẽ seed giả.

### Training chưa khởi chạy

Run POTER gốc ghi peak VRAM **10.63 GiB** (batch128, FP32), vượt GPU local6 GiB. Không đổi batch, AMP, BN hoặc loss để ép train. Log300 epochs có trung bình 108.65s/epoch; hai POTER seeds bổ sung ước tính 18.11 giờ nối tiếp trên GPU tương đương T4. Đây không phải benchmark mới hoặc bảo đảm thời gian cloud. ERM/DPE sẽ cần thêm thời gian. Train4795/val1199 images có đủ ởD và metadata canonical khớp; dữ liệu không phải blocker. Training chưa được bắt đầu theo chỉ dẫn báo trước của người dùng; cần cloud đủ VRAM và xác nhận trước khi launch. pending_training_plan.json khóa các yêu cầu seed/recipe để xem xét; file này không thực thi training.

![WGA seeds](wga_across_seeds.png)

![Asymmetry seeds](class_asymmetry_across_seeds.png)

![Interference seeds](background_interference_across_seeds.png)

![Class seed](class_accuracy_per_seed.png)

## 4. Model comparison

| POTER−baseline tại sigma2 | Land interference difference [CI95%] (pp) | Asymmetry difference [CI95%] (pp) |
|---|---|---|
| POTER−ERM enhanced | -27.00 [-42.00; -13.49] | +15.00 [+3.50; +25.50] |
| POTER−DPE legacy | -30.00 [-43.50; -17.00] | +3.00 [-9.51; +14.50] |

Tách clean performance, drop do blur và effect background. POTER/DPE có class-asymmetric blur response; enhanced ERM có WGA thấp hơn ởsigma2. Phản ứng natural−gray khác nhau ở checkpoint hiện có, nhưng seed/recipe không khớp và DPE legacy thiếu historical backbone binding. Không quy khác biệt quan sát được cho thuật toán POTER. Xác suất chưa calibration chung; DPE log-mean-probability margin không cùng thang native CNN logits.

![Sigma2](model_comparison_sigma2.png)

## 5. Hypothesis evaluation

| Hypothesis | Trạng thái | Phạm vi | Effect / CI95% |
|---|---|---|---|
| H1 | Supported | available POTER seed1, sigma2, selected foreground/background cohort | +24.50 [+14.00; +35.00] pp |
| H2 | Supported | available POTER seed1, sigma2, selected foreground/background cohort | -21.00 [-30.00; -12.00] pp |
| H3 | Supported | available POTER seed1, sigma2, selected foreground/background cohort | -15.00 [-25.00; -5.00] pp |
| H4 | Inconclusive | independent POTER training seeds | not estimable |
| H5 | Inconclusive | method-specific difference from ERM/DPE | see paired contrasts; seed-level replication missing |

H1: Landbird degradation lớn hơn Waterbird. H2: natural backgrounds làm giảm Landbird accuracy so với gray khi blur. H3: interference tăng thêm dưới blur. H4: tái hiện giữa POTER training seeds độc lập. H5: POTER failure behavior khác ERM/DPE ở cấp phương pháp. Supported choH1–H3 nếu có chỉ áp dụng POTER seed1 và input cohort hiện tại; không suy sangH4/H5. Không hiệu chỉnh multiple comparisons.

## 6. Mechanism Discovery decision

**NO-GO / INCONCLUSIVE.** Các can thiệp input cung cấp giả thuyết background interference đáng kiểm tra tiếp. Thiếu independent POTER seeds ngăn xác nhận tính ổn định ở cấp method. Gray thay đổi scene statistics; bốn nền có người/động vật/objects và một số mask/crop có artifacts. Human recognizability chưa được đánh giá độc lập. Không chọn GO cho internal mechanism chỉ từ các output shifts.

Bằng chứng tối thiểu còn thiếu: bổ sung đúng2 training seeds/method với fixed recipe và validation rule, gắn backbone/head hashes và lineage DPE; đánh giá2.100 inputs/checkpoint đã khóa; báo individual seeds và mean±SD thật; kiểm tra mask/recognizability độc lập và backgrounds/cohort mới để xác nhận sau nghiên cứu exploratory. Ba seeds chỉ là bằng chứng training variation sơ bộ. Không dùng cohort test đã xem để phát triển/tune mitigation.

## 7. Research contribution assessment

- Tái lập được cặp input/hash và tính effect trong một checkpoint; chưa xác nhận lặp giữa training seeds.
- POTER-specificity chưa được hỗ trợ ở cấp phương pháp; DPE cũng có class-asymmetric degradation.
- Bằng chứng vượt một accuracy-drop plot đơn thuần: cùng foreground/blur/mask/crop được can thiệp nền, có gray reference, replicate-level pairs, interference/amplification, distributions và species sensitivity.
- Hướng ưu tiên là xác minh background-induced interference bằng checkpoints mới trước khi phân tích representation/logit shifts.
- Chưa đủ để phát triển mitigation hoặc tuyên bố novelty/paper-readiness. Học class priors, calibration và internal representation geometry vẫn là giả thuyết chưa được xác định.

### Integrity, execution and files

Nguồn được gọi predictions(5).csv trong handoff không được đính kèm; dùng canonical outputs/mild_blur_v2/predictions.csv9.000 rows,34 columns, hash khóa trong experiment_config.json. Không thay sigma/cohort/checkpoint hoặc sửa kết quả cũ. statistical_analysis.csv ghi2.000 class-stratified bootstrap draws theo source foreground và sensitivity theo species. Mọi background/severity của cùng cluster đi cùng nhau; model comparisons paired cùng input. Không pool composite hay seed như quan sát độc lập.

Required outputs đã có: exp1_paired_interventions.csv, exp1_summary.md, exp2_seed_metrics.csv, exp2_group_metrics.csv, exp2_class_metrics.csv, exp2_predictions.csv, statistical_analysis.csv, checkpoint_manifest.json, experiment_config.json, final_report.md và10 requested PNG plots. Các file exp2 chứa đúng available seeds, không chứa rows giả cho seeds thiếu. files_manifest.csv liệt kê toàn bộ file/bytes/SHA256. validation.json ghi coverage vàimage audits; checkpoint_candidates.csv ghi lý do chấp nhận/loại từng asset.

Chạy lại analysis:
```powershell
.\.venv\Scripts\python.exe scripts\run_failure_verification.py
```
Đổi output_root trong config để giữ một version mới. Script không train hoặc chạy model inference.

Automated checks: 18 passed in 44.02s
