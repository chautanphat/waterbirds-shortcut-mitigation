# CODEX HANDOFF v6 — DPE Oracle & Test-Optimal Diagnostic Study

**Document status:** Diagnostic Research Protocol & Execution Handoff  
**Date:** 15 November 2026  
**Target Branch:** `dpe_optmized_version_3-(temporal-best)`  
**Scope:** Strictly Diagnostic / Upper-Bound Analysis (Non-Deployable)  

---

## 0. Sứ mệnh, Mục tiêu và Ranh giới Khoa học (Mission & Scientific Scope)

### 0.1. Bài toán đặt ra
Trong các nghiên cứu trước đây trên Waterbirds:
- **DPE Version 1 (`D19` - Neural Gate):** Mạng nơ-ron học cách gán trọng số động cho 15 prototype heads. Kết quả: Overall Accuracy tăng nhẹ (+0.25 pp) nhưng Worst-Group Accuracy (WGA) sụt giảm thảm hại (−9.12 pp, từ 92.76% xuống 83.65%). Mạng bị overfit nghiêm trọng vào majority shortcut.
- **DPE Version 2 (`D5` - Discrete Reliability Tables):** Tra bảng tin cậy rời rạc theo agreement và confidence. WGA vẫn giảm (−2.42 pp).
- **DPE Version 3 (`G1` - Geometry Aggregation):** Dùng khoảng cách Euclidean tới prototype kết hợp residual uniform bảo thủ (75% uniform + 25% distance). WGA tăng +0.5013 pp, vượt qua gate kiểm định khắt khe.

Tuy nhiên, câu hỏi cốt lõi vẫn chưa được giải đáp trực quan:
> **"Ở chế độ tối ưu nhất (nếu được tối ưu trực tiếp để cực đại hóa WGA trên tập dữ liệu đánh giá), phân phối trọng số của các head sẽ trông như thế nào? Khoảng cách cơ chế (mechanism gap) giữa những gì mạng Neural Gate học được và những gì Oracle thực sự cần là gì?"**

### 0.2. Mục tiêu nghiên cứu (Mission Objectives)
1. **Xây dựng Test-Optimal / Oracle Router:** Tối ưu hóa có ràng buộc phân phối trọng số của 15 heads trực tiếp trên tập test lịch sử để tìm ra phân phối trọng số lý tưởng tối đa hóa WGA (đồng thời giữ vững AA).
2. **Trích xuất Heatmap đối đầu:**
   - **Heatmap A (Neural Gate D19):** Mô hình neural thực tế đã phân bổ trọng số như thế nào theo các chiều đặc trưng `(Distance Quantile, Confidence Bin, Predicted Class)`.
   - **Heatmap B (Oracle / Test-Optimized):** Phân bổ trọng số lý tưởng ở chế độ tối ưu nhất trên cùng các ô đặc trưng đó.
   - **Heatmap C (Gap / Residual Analysis):** Vùng sai lệch $\Delta W = W^*_{\text{oracle}} - W_{\text{D19}}$, chỉ rõ chính xác những ô mà Neural Gate bị shortcut correlation đánh lừa.
3. **Rút ra Quy luật (Rule Extraction):** Xác định xem có quy luật tất định (deterministic rules) hoặc cấu trúc hình học nào có thể chưng cất thành inductive bias cho các phiên bản tiếp theo.

### 0.3. Ranh giới khoa học bất khả xâm phạm (Non-Negotiables)
- **DIAGNOSTIC ONLY — NEVER DEPLOY:** Nghiên cứu này là công cụ chẩn đoán hậu nghiệm (post-hoc diagnostic). Kết quả tối ưu trên test tuyệt đối KHÔNG được báo cáo như một phương pháp mới, KHÔNG được dùng để cherry-pick mô hình hay tinh chỉnh siêu tham số cho các mô hình cạnh tranh benchmark.
- **Watermark:** Tất cả các biểu đồ, bảng dữ liệu và báo cáo phải được đóng dấu tiêu đề:
  `[DIAGNOSTIC ONLY — ORACLE TEST UPPER BOUND — NON-DEPLOYABLE]`.
- **Bảo toàn hiện trạng:** Không sửa đổi, không ghi đè bất kỳ checkpoint, cache hay mã nguồn đã khóa nào của `dpe_optimized_version_1`, `version_2` hay `v3_geometry`.

---

## 1. Dữ liệu Đầu vào & Tuyến tính Dữ liệu (Inputs & Data Lineage)

Người thực hiện cần sử dụng các artifacts đã được xác thực có sẵn trong workspace:

1. **Prediction Cache lịch sử (`prediction_bundle.zip` hoặc `outputs/weighting/fits/`):**
   - Đọc 3 seed (`seed0`, `seed1`, `seed2`) của tập test Waterbirds (mỗi seed 4,974 ảnh test).
   - Các mảng cần nạp (shape chuẩn):
     ```python
     probs       # [N, 15, 2]  - Xác suất dự đoán của 15 prototype heads
     logits      # [N, 15, 2]  - Logits gốc của 15 heads
     distances   # [N, 15, 2]  - Khoảng cách Euclidean chuẩn hóa tới prototype
     y           # [N]         - Nhãn thực (0: Landbird, 1: Waterbird)
     group       # [N]         - Nhóm thực (0: Land/Land, 1: Land/Water, 2: Water/Land, 3: Water/Water)
     sample_id   # [N]         - ID ảnh duy nhất
     ```
2. **D19 Saved Checkpoint & Outputs:**
   - Tải trọng số $W_{\text{D19}} \in \mathbb{R}^{N \times 15}$ từ `results/dpe/score_audit_v4_20261115/phase0/` hoặc checkpoint joblib tương ứng.

---

## 2. Thiết kế Bài toán Tối ưu Oracle trên Tập Test (Test-Optimal Formulation)

### 2.1. Cảnh báo lỗi Overfitting cục bộ (Memorization Trap)
Nếu cho phép tối ưu unconstrained trọng số $w_i \in \Delta^{14}$ tự do cho từng mẫu ảnh $i$, optimizer sẽ chỉ đơn giản gán $w_{ij}=1$ cho bất kỳ head nào đoán đúng mẫu đó (đây chỉ là Any-Head-Correct Oracle X0, đạt 99.68% nhưng không có cấu trúc hay quy luật học được).

### 2.2. Phương pháp Chia lưới Đặc trưng (Feature Grid / Binning Formulation)
Để rút ra **quy luật**, ta buộc trọng số phải là một hàm của các đặc trưng quan sát được (Observable Features):
$$w_j = f_j(\text{Distance Quantile}, \text{Confidence Bin}, \text{Predicted Class})$$

1. **Phân chia không gian đặc trưng thành các ô (Cells $c \in \{1, \dots, C\}$):**
   - **Trục 1: Prototype Distance Metric ($d_{\min}$ hoặc Margin):**
     - $d_{\min, j} = \min_k \text{distance}[j, k]$: khoảng cách tới prototype gần nhất.
     - Chia thành 3–5 quantiles (Tertiles / Quintiles): `[Near, Medium, Far]`.
   - **Trục 2: Head Confidence / Agreement ($p_{\max}$ hoặc Agreement):**
     - Confidence: $\max_k P_j(k \mid x)$, chia thành 3 bins: `[Low, Medium, High]`.
     - Hoặc LOO Agreement: tỷ lệ các head khác đồng thuận với head $j$.
   - **Trục 3: Predicted Class:** Tách riêng cho Class 0 (Landbird) và Class 1 (Waterbird).
   - Tổng cộng: Mỗi class có $3 \times 3 = 9$ ô (hoặc $5 \times 5 = 25$ ô).

2. **Hàm mục tiêu Tối ưu hóa (Constrained Group-WGA Objective):**
   Tìm ma trận trọng số $\mathbf{W}^* \in \mathbb{R}^{C \times 15}$ (mỗi ô $c$ có 1 phân phối trọng số trên 15 heads) giải bài toán:
   $$\max_{\mathbf{W}} \min_{g \in \{0,1,2,3\}} \text{Accuracy}_g(\mathbf{W})$$
   $$\text{thỏa mãn:} \quad \begin{cases} 
   \text{AA}(\mathbf{W}) \ge \text{AA}_{\text{uniform}} - 1.0\% \\
   \sum_{j=1}^{15} W_{c, j} = 1, \quad \forall c \\
   W_{c, j} \ge 0, \quad \forall c, j \\
   \mathcal{R}(\mathbf{W}) \le \epsilon \quad (\text{Regularization chống cực đoan, ví dụ } W_{c,j} \le 0.40)
   \end{cases}$$

   *Gợi ý thuật toán giải:* Có thể giải bằng SLSQP / Linear Programming / Epigraph formulation tương tự cách đã kiểm chứng trong `experiments/wga_verification/solver.py`.

---

## 3. Danh mục Deliverables & Visualizations cần tạo

Người thực hiện cần sinh ra toàn bộ các biểu đồ và bảng dữ liệu sau đặt tại thư mục:  
`results/dpe/oracle_diagnostic_20261115/`

### 3.1. Bộ 4 Heatmaps Trọng tâm (PNG & CSV)
1. `heatmap_01_neural_d19.png`:
   - Phân bổ trọng số của Neural Gate D19 theo từng ô `(Distance Bin × Confidence Bin)`.
   - Thể hiện rõ: D19 dồn trọng số vào đâu khi gặp mẫu thuộc Group 2 (Waterbird on Land - nhóm bị shortcut hại).
2. `heatmap_02_oracle_optimal.png`:
   - Phân bổ trọng số của mô hình Test-Optimized Oracle trên cùng hệ tọa độ.
   - Trả lời: Ở trạng thái lý tưởng, các head nào được kích hoạt ở từng ô?
3. `heatmap_03_mechanism_gap.png`:
   - Hiệu số trọng số: $\Delta W = W^*_{\text{oracle}} - W_{\text{D19}}$.
   - Thang màu Diverging (Xanh = Oracle cần nhiều hơn; Đỏ = D19 dồn quá mức).
4. `heatmap_04_empirical_head_accuracy.png`:
   - Độ chính xác thực tế (Empirical Correctness Rate) của từng head trong từng cell.
   - Đối chiếu: Trọng số của Oracle có tỷ lệ thuận với độ chính xác thực tế không, hay nó chọn head vì tính bù trừ (complementarity)?

### 3.2. Bảng số liệu Tổng hợp (CSV & Markdown)
1. `oracle_vs_neural_metrics.csv`:
   - So sánh per-group accuracy (Group 0, 1, 2, 3), WGA, AA giữa:
     - Uniform Baseline
     - Neural Gate D19
     - Test-Optimal Grid Oracle
     - Any-Correct Upper Bound (X0)
2. `flip_analysis_oracle_vs_d19.csv`:
   - Danh sách các mẫu ảnh mà D19 đoán sai nhưng Oracle sửa đúng (và ngược lại), kèm metadata: true group, distance to prototype, confidence score.
3. `DIAGNOSTIC_REPORT.md`:
   - Báo cáo phân tích cơ chế hoàn chỉnh (chi tiết ở Mục 4).

---

## 4. Bộ Câu hỏi Nghiên cứu Bắt buộc Phải Trả lời trong Báo cáo (Rule Extraction Checklist)

Trong báo cáo `DIAGNOSTIC_REPORT.md`, người thực hiện phải trả lời dứt khoát 4 câu hỏi:

1. **Câu hỏi 1 (Shortcut Trap):**
   - *Trên Heatmap Gap, Neural Gate D19 bị "lừa" nặng nhất ở những ô đặc trưng nào?*
   - Có phải D19 luôn tin vào các head có confidence cao bất chấp khoảng cách prototype xa hay không?
2. **Câu hỏi 2 (Oracle Behavior in Ambiguous Regions):**
   - *Khi độ tự tin thấp hoặc khoảng cách prototype mập mờ, Oracle hành xử thế nào?*
   - Nó có xu hướng san đều trọng số (Uniform Fallback) hay vẫn chọn lọc 1 head cụ thể?
3. **Câu hỏi 3 (Giải mã thành công của G1 Geometry):**
   - *So sánh Heatmap của Oracle với công thức G1 ($w = 0.75/15 + 0.25 \cdot \text{softmax}(-50 d_{\min})$):*
   - Công thức G1 có bao phủ được hướng đi của Oracle không? Thành phần $75\%$ Uniform residual của G1 đóng vai trò gì khi đối chiếu với Oracle?
4. **Câu hỏi 4 (Tính khả thi của Router):**
   - *Các quy luật rút ra từ Oracle có thể chuyển hóa thành một heuristic không cần nhãn (unsupervised rule) cho tương lai hay không?* Hay bài toán router về bản chất bị nghẽn thông tin ở tầng embedding?

---

## 5. Quy trình Thực thi Kỹ thuật & Kiểm thử (Execution Protocol)

### 5.1. Mã nguồn module đề xuất
Tạo package thực nghiệm cô lập:
```text
experiments/oracle_diagnostic/
├── __init__.py
├── config.yaml          # Định nghĩa cấu hình bins, quantiles, regularization
├── optimize.py          # Bộ tối ưu hóa có ràng buộc trên test
├── extract_rules.py     # Phân tích quy luật và so sánh gap
├── visuals.py          # Render 4 heatmaps chuẩn matplotlib (dpi=150, layout='constrained')
└── run.py               # CLI entry point (--phase all, --resume, --status)
```

### 5.2. Lệnh thực thi & Kiểm tra
```bash
# 1. Chạy toàn bộ pipeline chẩn đoán Oracle
python -m experiments.oracle_diagnostic.run --phase all

# 2. Kiểm tra trạng thái và tính toàn vẹn
python -m experiments.oracle_diagnostic.run --status

# 3. Chạy unit tests bảo đảm không can thiệp vào mã nguồn gốc
pytest tests/methods/test_oracle_diagnostic.py -v
```

### 5.3. Tiêu chí Nghiệm thu Handoff (Acceptance Criteria)
- [ ] Chạy độc lập, atomic write, crash-safe (gián đoạn giữa chừng có thể resume không lỗi).
- [ ] 4 file PNG heatmaps và các file CSV được sinh đầy đủ tại `results/dpe/oracle_diagnostic_20261115/`.
- [ ] Tuyệt đối không thay đổi file dữ liệu gốc và không import test labels vào bất kỳ training routine nào của các baseline khác.
- [ ] Báo cáo `DIAGNOSTIC_REPORT.md` trả lời đầy đủ 4 câu hỏi nghiên cứu ở Mục 4.

---
*Handoff document được tạo để làm cơ sở kỹ thuật và chuẩn mực nghiên cứu trước khi triển khai thực nghiệm chẩn đoán.*
