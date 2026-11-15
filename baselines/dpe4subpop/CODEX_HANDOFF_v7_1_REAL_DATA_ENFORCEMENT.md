# CODEX HANDOFF v7.1 — GCAW Real-Data Enforcement & Algorithmic Fix Protocol

**Document Status:** Mandatory Re-Execution Handoff (Supersedes v6 & v7)

**Target Branch:** `dpe_optmized_version_3-(temporal-best)` / `feature/gcaw-v7-oracle-study`

**Action Required:** Reject synthetic run in `review_bundle_gcaw_v7.zip`; re-run on real Waterbirds prediction caches ($N = 4,974$ samples/seed) with patched calibration and hard data-lineage guards.

---

## 0. Lý do Bác bỏ Gói Kết quả `review_bundle_gcaw_v7.zip` (Audit Rejection Notice)

Qua kiểm toán độc lập trên gói `review_bundle_gcaw_v7.zip`, đợt chạy trước **không đạt yêu cầu nghiệm thu** vì 3 lỗi nghiêm trọng sau:

1. **Lỗi 1 — Chạy trên Dữ liệu Giả lập (Synthetic Data Substitution) thay vì Waterbirds thật:**
* Trong `GCAW_DECISION_REPORT.md` và `gcaw_ablation_metrics.csv`, `00_Uniform_Baseline` có `WGA = 0.28%`, `G0_Acc = 100%`, `G1_Acc = 100%`, `G3_Acc = 100%` và sửa đúng `110, 118, 120` mẫu trên tổng số $N = 1,200$ mẫu/seed để đạt `WGA = 100.00%`.
* Đây là dữ liệu sinh từ hàm unit test `_generate_synthetic_gcaw_data`, **không phải** tập test Waterbirds thực tế ($N = 4,974$ mẫu/seed, nơi `Uniform Baseline` có `WGA = 92.76%` và `G1` tăng `+0.5013 pp`).

2. **Lỗi 2 — Suy biến biên độ trong `06_Ablation_B2_MarginalFree_Lift_Only`:**
* Trên `GCAW_DECISION_REPORT.md`, biến thể `B2` đạt `AUROC_UnifErrors = 100.00%` nhưng `WGA` tụt về `0.83%`.
* Nguyên nhân hiển thị rõ trên `gcaw_preflight_01_reliability_surface.png`: hiệu số xác suất thô làm bảng `Marginal-Free Lift` suy biến về `+0.000` và `+0.002`, khiến $\operatorname{softmax}(\beta g_j)$ phẳng hoàn toàn (trọng số giữ nguyên ở $1/15$) vì thiếu bước chuẩn hóa biên độ nội mẫu (Within-Sample Z-score Standardization).

3. **Lỗi 3 — Gom trung bình mẫu làm sập lưới Heatmap 2 & Sai lệch văn bản báo cáo:**
* Trên `gcaw_preflight_02_oracle_weights.png`, việc lấy trung bình khoảng cách và trung bình đồng thuận của cả 15 heads trên mỗi ảnh khiến toàn bộ mẫu Group 2 bị dồn vào duy nhất hàng thứ 8 (`Far (Q3) | MidAgree`), trong khi 8 hàng còn lại phẳng hoàn toàn.
* Thế nhưng `README_REVIEWER.md` lại mô tả sai thực tế rằng trọng số Oracle tập trung ở hàng trên cùng `Near | LowAgree`.

---

## 1. Ràng buộc Kỹ thuật Bất khả xâm phạm (Non-Negotiable Constraints)

Người thực hiện bắt buộc phải tuân thủ 4 ràng buộc cứng sau trước khi đóng gói kết quả mới:

* **Ràng buộc 1 (Cấm tuyệt đối dữ liệu giả lập trong `run_gcaw.py`):**
  * Tuyệt đối không gọi bất kỳ hàm sinh dữ liệu ngẫu nhiên nào để tạo `probs`, `distances`, `y`, hoặc `group` khi chạy `run_gcaw.py`.
  * Nếu chưa rõ đường dẫn file cache thực tế, phải chạy script tự động dò tìm `locate_real_data.py` (cung cấp ở Mục 2.1) để định vị chính xác nơi lưu trữ dự đoán của 3 seeds (`seed0`, `seed1`, `seed2`).

* **Ràng buộc 2 (Chốt chặn Kiểm định Vân tay Dữ liệu Thực — Hard Fingerprint Assertions):**
  * Ngay khi nạp mỗi seed, code sẽ tự động kiểm tra:
    1. Số mẫu $N \ge 4,000$ (chuẩn Waterbirds test là $N = 4,974$).
    2. Độ chính xác `Uniform Baseline` phải nằm trong dải thực tế: $80.0\% \le \text{WGA}_{\text{uniform}} \le 96.0\%$ và không có bất kỳ nhóm nào (`G0, G1, G2, G3`) đạt độ chính xác tuyệt đối `100.0%` hay `0.0%`.
  * Nếu vi phạm bất kỳ điều kiện nào, chương trình lập tức dừng với `RuntimeError: [FATAL SYNTHETIC DATA DETECTED]`.

* **Ràng buộc 3 (Bắt buộc xuất `data_lineage_manifest.json`):**
  * Script phải tự động ghi lại đường dẫn tuyệt đối/tương đối của file dữ liệu gốc, mã băm `SHA256` của file đó, kích thước mảng `probs.shape`, `distances.shape`, số lượng mẫu từng nhóm (`G0, G1, G2, G3`) vào file `data_lineage_manifest.json` trong thư mục kết quả.

* **Ràng buộc 4 (Đồng bộ 100% giữa Biểu đồ, CSV và Markdown):**
  * Mọi nhận xét trong `GCAW_DECISION_REPORT.md` và `README_REVIEWER.md` phải đọc trực tiếp từ con số thực tế sau khi chạy, tuyệt đối không viết sẵn theo giả thuyết.

---

## 2. Mã nguồn Thực thi Đầy đủ (Cập nhật v7.1)

Cấu trúc các file tại `experiments/gcaw_study/`:

```text
experiments/gcaw_study/
├── __init__.py
├── locate_real_data.py    # [MỚI] Tự động quét workspace tìm cache Waterbirds thật (N=4974)
├── config.yaml            # Cấu hình đường dẫn và siêu tham số GCAW v7.1
├── calibration.py         # [ĐÃ VÁ] Thêm Within-Sample Z-score & sửa công thức B2 Lift
├── oracle_gate.py         # [ĐÃ VÁ] Phân rã ô theo Minority-Head State tránh sập vào 1 hàng
└── run_gcaw.py            # [ĐÃ VÁ] Chốt chặn chống dữ liệu giả + xuất data_lineage_manifest.json
```

---

## 3. Quy trình Thực thi Bắt buộc cho Người Chạy (Step-by-Step Execution)

Người thực hiện cần chạy đúng 3 bước sau trên môi trường có chứa dữ liệu gốc:

```bash
# Bước 1: Quét tìm đường dẫn thực tế của prediction cache Waterbirds (N=4974)
python -m experiments.gcaw_study.locate_real_data

# Bước 2: Cập nhật đúng đường dẫn vừa tìm được vào experiments/gcaw_study/config.yaml
# rồi chạy pipeline chính (TUYỆT ĐỐI KHÔNG truyền cờ --allow-synthetic-pytest)
python -m experiments.gcaw_study.run_gcaw --config experiments/gcaw_study/config.yaml

# Bước 3: Kiểm tra nhanh file chứng thư dữ liệu và bảng kết quả trước khi nén gửi lại
cat results/dpe/gcaw_study_v7_1_real/data_lineage_manifest.json
cat results/dpe/gcaw_study_v7_1_real/GCAW_DECISION_REPORT.md
```

---

## 4. Danh mục Kiểm tra Nghiệm thu (Acceptance Checklist trước khi bàn giao)

Gói kết quả gửi lại (`review_bundle_gcaw_v7_1.zip`) chỉ được chấp nhận khi thỏa mãn đầy đủ 5 tiêu chí sau:

* [ ] **Có file `data_lineage_manifest.json`** ghi rõ `n_samples: 4974` (hoặc kích thước tập test thật $\ge 4,000$), mã băm `SHA256` của file gốc, và `uniform_baseline_wga_pct` khớp với mốc lịch sử ($\approx 92.76\%$).
* [ ] **Trong `gcaw_ablation_metrics.csv`**, không có phương pháp nào có `WGA = 100.00%` hay `0.00%` giả tạo như ở bản v7.0.
* [ ] **Trên `gcaw_preflight_01_reliability_surface.png`**, bảng bên phải (`Marginal-Free Lift`) hiển thị giá trị độ nâng khác `0` rõ ràng giữa các tầng khoảng cách `Near (Q1)`, `Medium (Q2)`, `Far (Q3)`.
* [ ] **Trên `gcaw_preflight_02_oracle_weights.png`**, nhãn trục tung ghi rõ số lượng mẫu `(n=...)` trong từng ô và trọng số $W^*$ được hiển thị bằng con số cụ thể trên từng ô thay vì phẳng một màu.
* [ ] **Báo cáo `GCAW_DECISION_REPORT.md`** phản ánh trung thực kết quả thực tế (dù `PASS` hay `KILL BRANCH`), không chỉnh sửa tay.
