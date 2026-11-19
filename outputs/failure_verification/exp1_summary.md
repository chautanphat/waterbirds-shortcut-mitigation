# Experiment1: paired background intervention

**Hoàn thành bằng dự đoán thật đã lưu, không inference lại.** Giữ nguyên source foreground, mask, placement, blur và preprocessing; so gray→Land A/B, gray→Water A/B và Land A/B→Water A/B. Mỗi contrast có2 background replicates; replicate được trung bình trong foreground trước bootstrap. Có7.200 paired rows từ6.000 prediction rows (100 foreground ×4 sigma ×5 nền ×3 model); chúng không phải7.200 mẫu độc lập.

Primary sigma0/0,5/1/2; sigma4 bị loại khỏi phân tích chính. Các source tensors/ảnh render được kiểm lại. Landbird=0, Waterbird=1; unrounded probabilities. Logit margin là score(Water)−score(Land). DPE dùng log(mean head probability), nên không so độ lớn margin shift DPE với native CNN logits của POTER/ERM.

Interference = accuracy natural−accuracy gray. Âm: natural backgrounds làm giảm accuracy. Amplification = interference(sigma)−interference(clean). Âm: phần giảm tăng thêm dưới blur. Delta_background = mean P(Water|water backgrounds)−mean P(Water|land backgrounds), có dấu.

| Model | Lớp | Sigma | Interference [CI95%] (pp) | Amplification [CI95%] (pp) | Delta background [CI95%] (pp) |
|---|---|---|---|---|---|
| POTER | landbird | 0.0 | -6.00 [-12.50; -1.00] | +0.00 [+0.00; +0.00] | +1.01 [-0.76; +2.83] |
| POTER | landbird | 0.5 | -8.00 [-14.50; -2.50] | -2.00 [-4.00; -0.50] | +0.97 [-0.88; +2.82] |
| POTER | landbird | 1.0 | -9.50 [-16.00; -3.50] | -3.50 [-10.50; +5.00] | -2.79 [-4.75; -0.78] |
| POTER | landbird | 2.0 | -21.00 [-30.00; -12.00] | -15.00 [-25.00; -5.00] | -8.77 [-10.72; -6.76] |
| POTER | waterbird | 0.0 | +0.50 [-1.50; +3.00] | +0.00 [+0.00; +0.00] | -1.72 [-2.81; -0.52] |
| POTER | waterbird | 0.5 | +2.00 [-1.00; +6.00] | +1.50 [-1.50; +6.00] | -2.71 [-3.90; -1.43] |
| POTER | waterbird | 1.0 | +1.00 [-1.50; +4.50] | +0.50 [+0.00; +1.50] | -5.33 [-6.48; -4.18] |
| POTER | waterbird | 2.0 | +2.00 [+0.00; +6.00] | +1.50 [+0.00; +4.00] | -5.60 [-6.90; -4.35] |
| ERM enhanced | landbird | 0.0 | +3.50 [-3.00; +10.50] | +0.00 [+0.00; +0.00] | +14.27 [+8.25; +20.71] |
| ERM enhanced | landbird | 0.5 | -1.00 [-6.50; +5.50] | -4.50 [-10.50; +0.00] | +18.50 [+12.05; +25.39] |
| ERM enhanced | landbird | 1.0 | +4.00 [-3.50; +12.50] | +0.50 [-6.00; +8.00] | +33.60 [+24.52; +42.66] |
| ERM enhanced | landbird | 2.0 | +6.00 [-4.00; +17.00] | +2.50 [-7.00; +13.00] | +56.33 [+47.70; +64.70] |
| ERM enhanced | waterbird | 0.0 | -4.00 [-8.00; -1.00] | +0.00 [+0.00; +0.00] | +11.20 [+5.06; +18.61] |
| ERM enhanced | waterbird | 0.5 | -5.00 [-9.00; -1.50] | -1.00 [-4.00; +2.50] | +13.69 [+7.37; +20.92] |
| ERM enhanced | waterbird | 1.0 | -11.50 [-17.00; -6.00] | -7.50 [-13.00; -2.00] | +24.10 [+15.73; +32.64] |
| ERM enhanced | waterbird | 2.0 | -18.00 [-23.00; -12.00] | -14.00 [-20.00; -7.00] | +39.71 [+30.56; +49.13] |
| DPE legacy | landbird | 0.0 | +3.00 [-4.00; +10.01] | +0.00 [+0.00; +0.00] | +0.31 [+0.12; +0.52] |
| DPE legacy | landbird | 0.5 | +1.00 [-5.50; +7.50] | -2.00 [-8.00; +3.50] | +0.29 [+0.11; +0.47] |
| DPE legacy | landbird | 1.0 | -0.50 [-8.50; +7.50] | -3.50 [-13.50; +5.51] | +0.32 [+0.13; +0.50] |
| DPE legacy | landbird | 2.0 | +9.00 [-0.50; +19.00] | +6.00 [-4.00; +16.01] | +0.41 [+0.25; +0.58] |
| DPE legacy | waterbird | 0.0 | -0.50 [-1.50; +0.00] | +0.00 [+0.00; +0.00] | -0.17 [-0.33; +0.01] |
| DPE legacy | waterbird | 0.5 | -0.50 [-1.50; +0.00] | +0.00 [+0.00; +0.00] | -0.07 [-0.26; +0.13] |
| DPE legacy | waterbird | 1.0 | +1.00 [-3.00; +6.00] | +1.50 [-1.50; +6.00] | +0.04 [-0.15; +0.24] |
| DPE legacy | waterbird | 2.0 | +1.00 [-3.00; +6.00] | +1.50 [-2.00; +6.50] | +0.19 [-0.01; +0.38] |

### Nhạy cảm với species clustering

| Model | Sigma | Land interference: source bootstrap | Land interference: species bootstrap |
|---|---|---|---|
| POTER | 0.0 | -6.00 [-12.50; -1.00] | -6.00 [-12.00; -1.00] |
| POTER | 2.0 | -21.00 [-30.00; -12.00] | -21.00 [-29.00; -12.50] |
| ERM enhanced | 0.0 | +3.50 [-3.00; +10.50] | +3.50 [-3.00; +11.50] |
| ERM enhanced | 2.0 | +6.00 [-4.00; +17.00] | +6.00 [-6.51; +21.50] |
| DPE legacy | 0.0 | +3.00 [-4.00; +10.01] | +3.00 [-4.50; +11.00] |
| DPE legacy | 2.0 | +9.00 [-0.50; +19.00] | +9.00 [-2.50; +20.50] |

Source bootstrap resample50 foreground/lớp. Species sensitivity resample10 species/lớp; giữ trung bình5 foreground/species. Cả hai có2.000 draws class-stratified, dùng cùng draws giữa model/sigma. Chúng không ước lượng training-seed uncertainty hay khả năng tổng quát sang background mới.

### So sánh paired theo model tại sigma2

| So sánh | Land interference difference [CI95%] (pp) | Land amplification difference [CI95%] (pp) |
|---|---|---|
| POTER−ERM enhanced | -27.00 [-42.00; -13.49] | -17.50 [-33.00; -2.00] |
| POTER−DPE legacy | -30.00 [-43.50; -17.00] | -21.00 [-37.50; -4.49] |

### Phân bố và consistency

- POTER: 9/10 loài Landbird có interference âm; 19/50 foreground Landbird có interference âm tại sigma2.
- ERM enhanced: 4/10 loài Landbird có interference âm; 22/50 foreground Landbird có interference âm tại sigma2.
- DPE legacy: 3/10 loài Landbird có interference âm; 11/50 foreground Landbird có interference âm tại sigma2.

Đổi nền là can thiệp trên input thực sự: foreground và blur được giữ cố định, nên chênh lệch đầu ra thuộc việc thay ảnh nền trong các cặp này. Nền có nhiều yếu tố ngoài thuộc tính Land/Water; gray cũng thay scene statistics. Kết quả chưa chỉ ra internal neural-network mechanism, learned class prior, hoặc nguyên nhân do riêng thuật toán POTER.

### Các contrast của POTER tại sigma2

| Lớp | Contrast | Flip rate [CI95%] (%) | Signed ΔP(Water) [CI95%] (pp) | Mean absolute ΔP [CI95%] (pp) | Margin shift [CI95%] (logit units) | Accuracy change [CI95%] (pp) |
|---|---|---|---|---|---|---|
| landbird | gray_to_land | +28.00 [+17.00; +39.00] | +13.83 [+11.57; +16.12] | +14.43 [+12.38; +16.53] | +0.6548 [+0.5476; +0.7614] | -28.00 [-39.00; -17.00] |
| landbird | gray_to_water | +14.00 [+6.00; +23.00] | +5.06 [+3.20; +6.93] | +7.44 [+6.12; +8.80] | +0.2380 [+0.1387; +0.3344] | -14.00 [-23.00; -6.00] |
| landbird | land_to_water | +18.00 [+10.00; +26.00] | -8.77 [-10.72; -6.76] | +11.42 [+9.64; +13.09] | -0.4168 [-0.5066; -0.3253] | +14.00 [+6.00; +23.00] |
| waterbird | gray_to_land | +2.00 [+0.00; +6.00] | +5.44 [+3.83; +7.06] | +6.79 [+5.52; +8.19] | +0.3574 [+0.2653; +0.4488] | +2.00 [+0.00; +6.00] |
| waterbird | gray_to_water | +2.00 [+0.00; +6.00] | -0.16 [-1.58; +1.32] | +4.63 [+3.79; +5.46] | -0.0295 [-0.1050; +0.0533] | +2.00 [+0.00; +6.00] |
| waterbird | land_to_water | +0.00 [+0.00; +0.00] | -5.60 [-6.90; -4.35] | +7.31 [+6.14; +8.71] | -0.3869 [-0.4655; -0.3092] | +0.00 [+0.00; +0.00] |

**Không đồng nhất background interference với ưu tiên nền Water.** Với Landbird sigma2, Land→Water làm ΔP(Water) -8.77 [-10.72; -6.76] pp. Nền Water làm giảm P(Water) so với nền Land trong các cặp đã chọn. Cả Land và Water vẫn có thể làm tăng P(Water) so với gray; độ lớn và hướng phụ thuộc contrast. Điều này không hỗ trợ câu chuyện đơn giản water-background→Waterbird prediction.

### Trả lời câu hỏi Experiment1

1. Có. Đổi nền giữ foreground cố định làm thay đổi output và correctness; xem flip rates/CI ở trên.
2. Landbird interference tăng thêm dưới blur tại sigma2 so với clean; amplification CI nằm dưới0. Mọi sigma đều được báo cáo, không chỉ endpoint.
3. Chênh Landbird−Waterbird interference tại sigma2 là -23.00 [-33.00; -13.50] pp.
4. Các checkpoint ERM/DPE có response khác POTER trên cùng inputs; training-method specificity chưa xác nhận.
5. Hiệu ứng âm xuất hiện ở9/10 loài Landbird của POTER,19/50 foreground. Species sensitivity CI vẫn âm nhưng independent training-seed consistency chưa được kiểm tra.

![background_probability_shift.png](background_probability_shift.png)

![paired_prediction_flips.png](paired_prediction_flips.png)

![background_interference_vs_sigma.png](background_interference_vs_sigma.png)

![landbird_vs_waterbird_interference.png](landbird_vs_waterbird_interference.png)

![per_foreground_probability_shifts.png](per_foreground_probability_shifts.png)

