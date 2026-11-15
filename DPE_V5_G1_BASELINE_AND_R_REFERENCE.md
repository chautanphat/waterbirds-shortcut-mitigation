# DPE v5 — G1 baseline và G1 + R tham khảo

**Branch:** `dpe_optimized_version_5`. **Ngày:** 15/11/2026.

Trong cách trình bày của nhánh này, **G1 là baseline so sánh** và **G1 + R là phương pháp tham khảo**. `G1 + R` tương ứng ID `G1_R` trong các file kết quả v7: G1 cộng continuous pooled relative-advantage residual. Uniform vẫn được lưu làm đối chứng matched.

**EXPLORATORY_ONLY — REUSED DEVELOPMENT — NOT INDEPENDENT CONFIRMATION**

**INDEPENDENT_CONFIRMATION_NOT_PERFORMED.** Các số liệu dưới đây lấy từ 9 cache C đã khảo sát, dùng chung một backbone ERM* và 1.199 ID ảnh lặp lại qua ba seed. Đây không phải kết quả ERM chuẩn hoặc test độc lập mới.

## Kết quả G1 và G1 + R

| Vai trò trên nhánh | Phương pháp | Mean AA (%) | Mean WGA (%) | ΔAA so với G1 (điểm %) | ΔWGA so với G1 (điểm %) |
|---|---|---:|---:|---:|---:|
| **Baseline** | **G1** | **95,857659** | **92,982456** | 0 | 0 |
| **Tham khảo** | **G1 + R** | **95,829858** | **93,233083** | **−0,027801** | **+0,250627** |

| Seed | G1 AA (%) | G1 + R AA (%) | ΔAA (điểm %) | G1 WGA (%) | G1 + R WGA (%) | ΔWGA (điểm %) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 95,829858 | 95,913261 | +0,083403 | 90,977444 | 91,729323 | +0,751880 |
| 1 | 96,080067 | 95,913261 | −0,166806 | 93,984962 | 93,984962 | 0 |
| 2 | 95,663053 | 95,663053 | 0 | 93,984962 | 93,984962 | 0 |

G1 + R đạt điều kiện giảm AA <1 điểm % ở từng seed so với Uniform và G1. Khoảng tin cậy bootstrap ghép cặp 95% cho ΔWGA so với G1 là **[−0,143062; +0,751880] điểm %**, vẫn chứa 0. Không coi chênh lệch quan sát này là bằng chứng xác nhận khả năng tổng quát hóa.

Mean WGA được tính bằng cách gộp ba outer fold trong từng seed, lấy minimum accuracy của bốn nhóm cho seed đó, rồi trung bình qua ba seed. Không lấy trung bình WGA của từng fold. Bootstrap dùng 2.000 lượt lấy mẫu theo **1.199 ID duy nhất**, với cùng lượt lấy mẫu cho mọi seed/phương pháp.

## G1 + R giữ gì và bổ sung gì?

- Giữ nguyên G1 geometry `g_j = -50 * min_k distance_jk`, 15 prototype head, backbone, member probabilities và IPS training.
- Fit no-intercept pooled ridge từ 6 đặc trưng liên tục, penalty 10, để dự đoán relative advantage so với Uniform. Tất cả thống kê và nhãn fitting lấy từ B-training.
- Fit scale từ 3-fold secondary OOF nằm hoàn toàn bên trong mỗi tập B-training; dùng guard `s_R < 0.002` và clipping ±2s_R theo handoff.
- Cộng residual với hệ số cố định **λ=0,5** vào G1 score; giữ **γ=50**, **α=0,25**, `w_j = 0.05 + 0.25 * softmax(score)_j`.
- G1 + R là bản **pooled-only**. Hiệu ứng head/class partial pooling thuộc ablation `G1_RH`; margin gate thuộc `G1_RHM`.

WGA tăng so với G1 đến từ việc sửa một ảnh waterbird/land ở seed 0: `003.Sooty_Albatross/Sooty_Albatross_0034_1154.jpg`. Nhóm waterbird/land (group 2, 133 ảnh/seed) vẫn là nhóm tệ nhất ở cả ba seed. Tất cả breakages ở các nhóm khác cũng được giữ trong bảng đầy đủ.

## Bằng chứng B và kết luận nghiên cứu gốc

G1 + R là phương pháp **tham khảo trên nhánh này**, chưa được xác nhận cho production. Trên B-OOF, signed gain của R trên ảnh Uniform dự đoán sai giảm ở cả ba seed so với G1; chênh lệch trung bình **−0,00217159**, paired 95% CI **[−0,00274243; −0,00161277]**. Vì vậy tăng WGA trên C không đủ để kết luận đã học được cách chọn expert tốt hơn ở ảnh khó.

Primary đã đăng ký trước của nghiên cứu v7 vẫn là **G1_RH**, với kết luận gốc **B_DIRECTION_NO_GO** và **EXPLORATORY_C_WGA_EQUAL**. Vai trò trình bày của R theo yêu cầu trên branch không sửa protocol, không chọn lại primary bằng C và không thay bất kỳ kết quả cũ nào.

| Phương pháp v7 | Mean AA (%) | Mean WGA (%) | Ghi chú |
|---|---:|---:|---|
| Uniform | 95,746455 | 92,481203 | Đối chứng matched |
| G1 | 95,857659 | 92,982456 | **Baseline trên nhánh** |
| G1_R | 95,829858 | 93,233083 | **G1 + R tham khảo** |
| G1_RH | 95,829858 | 92,982456 | Primary v7 gốc, B-direction NO-GO |
| G1_RHM | 95,857659 | 92,982456 | Margin-gating ablation |

Inference mặc định gốc, Uniform và chế độ G1 opt-in đều được bảo toàn. Không retrain, không mở rộng search và không chạy lại thí nghiệm khi export/push nhánh này.

## Đọc kết quả và mã nguồn

- [Báo cáo v7 đầy đủ](results/dpe/g1_residual_v7_20261115/EXPLORATORY_C_REPORT.md), [phân tích cơ chế và ảnh thay đổi](results/dpe/g1_residual_v7_20261115/DELIVERY_SUPPLEMENT.md).
- [Bảng mean AA/WGA và chênh lệch](results/dpe/g1_residual_v7_20261115/metrics/MEAN_AA_WGA_COMPARISON.csv), [kết quả từng seed](results/dpe/g1_residual_v7_20261115/metrics/C_PER_SEED_METRICS.csv), [từng subgroup](results/dpe/g1_residual_v7_20261115/metrics/PER_SEED_AND_GROUP_METRICS.csv).
- [Flip audit đầy đủ mọi ảnh](results/dpe/g1_residual_v7_20261115/metrics/FLIP_AUDIT.csv), [B-OOF mechanism](results/dpe/g1_residual_v7_20261115/metrics/B_OOF_MECHANISM.csv), [paired CI](results/dpe/g1_residual_v7_20261115/metrics/BOOTSTRAP_CI.json), [permutation controls](results/dpe/g1_residual_v7_20261115/metrics/PERMUTATION_CONTROLS.json).
- [Score/weight/probability cache](results/dpe/g1_residual_v7_20261115/metrics/WEIGHT_AND_SCORE_DECOMPOSITION.npz), [C labels/group/ID cho evaluator](results/dpe/g1_residual_v7_20261115/metrics/C_EVALUATION_LABELS.npz), [schema](results/dpe/g1_residual_v7_20261115/metrics/CACHE_SCHEMA.json). Nhãn C nằm riêng với inference decomposition.
- [Mã thuật toán](experiments/g1_residual_v7/model.py), [CLI và hướng dẫn chạy](experiments/g1_residual_v7/README.md), [manifest bản export](results/dpe/g1_residual_v7_20261115/EXPORT_MANIFEST.json).

Đã pass 13 kiểm thử, audit độc lập 180 fit ridge/head, kiểm tra kill/resume thật ở secondary fit và C inference, và xác minh 591 completion receipt. Resume tạo 0 task khoa học mới. Checkpoint local, kết quả v1–v6 và NO-GO gốc được giữ nguyên. Git export chứa báo cáo, bảng, fitted models và hai cache phân tích đã chọn; ảnh gốc, trained prototype/backbone checkpoints, per-fold prediction trùng lặp và ZIP được giữ ngoài Git.
