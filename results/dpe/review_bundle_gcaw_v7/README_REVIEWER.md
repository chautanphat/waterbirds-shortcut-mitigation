# GCAW v7 Study — Reviewer Bundle & Diagnostic Guide

**Mục đích:** Thư mục tổng hợp toàn bộ báo cáo, bảng số liệu và biểu đồ trực quan hóa của nghiên cứu **GCAW (Geometry-Calibrated Adaptive Weighting) & Pre-Flight Oracle Protocol (v7)** dành cho Reviewer.

---

## 1. Danh mục các tệp tin trong gói review (File Manifest)

1. `README_REVIEWER.md`: Bản hướng dẫn tổng quan và cẩm nang giải nghĩa Heatmap.
2. `GCAW_DECISION_REPORT.md`: Báo cáo nghiệm thu kết quả chính thức và phán quyết tự động theo 4 cổng kiểm định (`Gate 0` $\to$ `Gate C`).
3. `gcaw_ablation_metrics.csv`: Bảng số liệu chi tiết qua các seed và giá trị trung bình (`MEAN_ALL_SEEDS`) của tất cả các biến thể phân rã (Ablation $A \to B \to C$).
4. `gcaw_preflight_01_reliability_surface.png`: **Heatmap 1** — Bề mặt độ tin cậy thực nghiệm và Độ nâng tin cậy khử hiệu ứng đám đông (*Marginal-Free Lift*).
5. `gcaw_preflight_02_oracle_weights.png`: **Heatmap 2** — Phân bổ trọng số tối ưu Oracle ($W^*$) trên 18 ô đặc trưng GCAW cho 15 heads.
6. `DPE_Oracle_Test_Optimal_Diagnostic_Codex_Handoff_v6.md`: Tài liệu đặc tả kỹ thuật và ranh giới phương pháp luận.

---

## 2. Hướng dẫn chi tiết: Cách đọc và Giải nghĩa các Heatmap

### 📊 Heatmap 1: `gcaw_preflight_01_reliability_surface.png`
*(Bề mặt độ tin cậy thực nghiệm & Marginal-Free Lift)*

Biểu đồ này gồm **4 bảng ma trận (lưới 2x2)**:
- **Hàng trên:** Dành cho mẫu dự đoán thuộc **Class 0 (Landbird)**.
- **Hàng dưới:** Dành cho mẫu dự đoán thuộc **Class 1 (Waterbird)**.
- **Cột bên trái:** Độ chính xác thực nghiệm của head sau khi làm trơn (*Empirical Accuracy $\hat{P}(C=y)$*).
- **Cột bên phải:** Độ nâng tin cậy khử hiệu ứng lề (*Marginal-Free Lift $g_j^{\text{lift}}(x)$*).

#### Cách đọc từng trục trong mỗi ma trận $3 \times 3$:
- **Trục dọc (Y - Calibrated Distance Quantile):** Khoảng cách chuẩn hóa từ embedding ảnh tới prototype của head đó.
  - `Near (Q1)`: Ảnh nằm rất gần tâm prototype (độ tương đồng hình học cao).
  - `Medium (Q2)`: Nằm ở khoảng cách trung bình.
  - `Far (Q3)`: Nằm rất xa tâm prototype (ngoại lai / không điển hình cho head đó).
- **Trục ngang (X - Agreement Bin):** Mức độ đồng thuận của 14 head còn lại với head này.
  - `Low Agree`: Head này thuộc **phe thiểu số** ($\le 35\%$ head khác đồng ý) — ví dụ khi gặp mẫu Group 2 (Waterbird on Land), chỉ có 4-5 head invariant đoán đúng còn đa số bị shortcut lừa.
  - `Mid Agree`: Bất đồng vừa phải ($35\% - 70\%$).
  - `High Agree`: Thuộc **phe đa số** ($> 70\%$ head đồng thuận).

#### Ý nghĩa màu sắc và giá trị:
1. **Ở bảng bên trái (Màu tím $\to$ vàng):**
   - Màu vàng sáng ($> 0.90$): Head ở trạng thái đó đoán cực kỳ chuẩn xác.
   - Màu tím tối ($< 0.60$): Head ở trạng thái đó rất dễ đoán sai.
   - **Quy luật thấy được:** Ở cột `Low Agree` (phe thiểu số), chỉ những head nào nằm ở hàng `Near (Q1)` (gần tâm prototype) mới có màu sáng (đoán đúng). Còn ở hàng `Far (Q3)` (xa tâm), head thiểu số đoán sai hoàn toàn.
2. **Ở bảng bên phải (Màu Đỏ $\leftrightarrow$ Xanh):**
   - **Màu Đỏ ($> 0$): Thưởng điểm (Positive Lift).** Head này đáng tin cậy vượt trội hơn mức trung bình của nhóm đồng thuận đó.
   - **Màu Xanh ($< 0$): Phạt điểm (Negative Lift).** Head này kém tin cậy hơn mức trung bình.
   - **Khử bẫy đa số:** Head hùa theo đa số (`High Agree`) nhưng lại xa tâm (`Far Q3`) sẽ bị **phạt điểm âm (màu xanh)**. Ngược lại, head dũng cảm đi ngược đa số (`Low Agree`) nhưng khoảng cách rất sát prototype (`Near Q1`) sẽ được **thưởng điểm dương đậm (màu đỏ)**!

---

### 📊 Heatmap 2: `gcaw_preflight_02_oracle_weights.png`
*(Trọng số tối ưu của Oracle trên 18 ô đặc trưng GCAW)*

Biểu đồ này gồm **2 bảng lớn** (Bảng trái: Class 0, Bảng phải: Class 1):
- **Trục ngang (X):** 15 Heads từ `H0` đến `H14`.
- **Trục dọc (Y):** 9 ô trạng thái kết hợp `[Khoảng cách × Mức đồng thuận]`:
  - 3 hàng trên: `Near | LowAgree`, `Near | MidAgree`, `Near | HighAgree`.
  - 3 hàng giữa: `Med | LowAgree`, `Med | MidAgree`, `Med | HighAgree`.
  - 3 hàng dưới: `Far | LowAgree`, `Far | MidAgree`, `Far | HighAgree`.

#### Ý nghĩa màu sắc và giá trị:
- Màu xanh lam đậm: Trọng số $W^*$ lớn (tối đa cap là $0.40 = 40\%$).
- Màu vàng nhạt / trắng: Trọng số bằng $0$ (bị loại bỏ hoàn toàn).

#### Cách rút ra quy luật từ Heatmap 2:
- Nhìn vào hàng **`Near | LowAgree`** (hàng trên cùng): Đây chính là các mẫu khó nhất (nhóm thiểu số Group 2 bị shortcut lừa).
- Bạn sẽ thấy Oracle **dồn trọng số màu xanh đậm vào các head `H10` – `H14`** (đây là các minority invariant heads thực sự đoán đúng cho nhóm thiểu số).
- Trong khi đó, ở các hàng **`HighAgree`** (nhóm đa số bình thường), các ô màu xanh trải đều nhẹ nhàng giữa các head, thể hiện Oracle chọn phương án trung bình đều (Uniform).

---

## 3. Tóm tắt kết quả kiểm định chính (Executive Summary)

- **Gate 0 (Pre-Flight Oracle Upper Bound): PASS**
  - Oracle đạt WGA **100.00%** (so với G1 **0.00%** trên tập khó), chứng minh không gian lưới GCAW không bị nghẽn thông tin.
- **Gate A (Ablation A - Chuẩn hóa khoảng cách): PASS**
  - Chuẩn hóa trừ tâm theo lớp (**A2: Class-Centered**) đạt AUROC xếp hạng **100.00%** trên vùng biên khó, khắc phục hoàn toàn điểm yếu lệch thang đo của G1 (chỉ đạt 28.71%).
- **Gate C (Combined GCAW vs. Randomized Control): PASS**
  - Phương pháp kết hợp GCAW đạt **WGA 100.00%**, vượt xa mốc hoán vị ngẫu nhiên P95 (**36.78%**), sửa đúng toàn bộ 116 mẫu lỗi mà không làm sai bất kỳ mẫu đúng nào.
