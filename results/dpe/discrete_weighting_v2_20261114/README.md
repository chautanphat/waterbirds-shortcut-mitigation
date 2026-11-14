# DPE v2 — D5 đạt 96,076% accuracy, đánh đổi 2,418 điểm phần trăm WGA

**D5 là cấu hình tốt nhất theo tiêu chí lựa chọn trên validation trong nghiên cứu v2.** Sau khi khóa cấu hình, accuracy trung bình trên test đạt **96,076%**, tăng **0,368 điểm phần trăm** so với uniform averaging dùng cùng 15 heads. Tuy nhiên, WGA giảm từ **92,761% xuống 90,343%**, tức **−2,418 điểm phần trăm**.

Đây là winner của quy trình chọn bằng validation đã định trước, không phải khẳng định D5 đứng đầu mọi ablation trên test hoặc mọi tiêu chí robustness.

## Kết quả chính

Các giá trị là **trung bình ± độ lệch chuẩn mẫu qua 3 seed (0, 1, 2)**. Ba seed refit heads/gate và dùng chung một backbone đóng băng.

| Phương pháp | Test accuracy (%) | Balanced accuracy (%) | WGA (%) |
|---|---:|---:|---:|
| Uniform, cùng 15 heads | 95,708 ± 0,117 | 95,182 ± 0,053 | 92,761 ± 0,552 |
| **D5 — winner đã khóa** | **96,076 ± 0,339** | 94,843 ± 0,308 | 90,343 ± 1,685 |
| Chênh lệch trung bình | **+0,368 điểm %** | −0,339 điểm % | **−2,418 điểm %** |

“WGA giảm 2,4%” được hiểu chính xác là giảm khoảng **2,4 điểm phần trăm**, không phải mức giảm tương đối 2,4%.

| Seed | Accuracy D5 (%) | Tăng so với uniform (điểm %) | WGA D5 (%) | Ảnh sửa đúng | Ảnh bị làm sai thêm |
|---|---:|---:|---:|---:|---:|
| 0 | 95,685 | +0,017 | 91,745 | 20 | 19 |
| 1 | 96,289 | +0,449 | 88,474 | 61 | 35 |
| 2 | 96,255 | +0,639 | 90,810 | 73 | 36 |

D5 tăng accuracy ở cả ba seed, nhưng mức tăng khác nhau đáng kể. Tổng cộng qua ba lượt test, D5 sửa đúng 154 dự đoán và làm sai thêm 90 dự đoán so với uniform, tăng ròng 64 dự đoán đúng trên 17.382 lượt ảnh/seed. Đây là cùng 5.794 ảnh được đánh giá bằng ba mô hình, không phải 17.382 ảnh độc lập.

## Cơ chế D5

Uniform gán cho mỗi prototype classifier trọng số 1/15. D5 học bảng reliability từ dự đoán OOF của dữ liệu phát triển, sau đó điều chỉnh trọng số theo từng ảnh. Các thông tin dùng khi suy luận gồm:

1. **Lớp classifier dự đoán**: landbird hoặc waterbird.
2. **Confidence**: chia thành ba khoảng quantile, ước lượng trong từng lớp dự đoán trên tập học reliability.
3. **Mức đồng thuận**: tỷ lệ 14 classifier còn lại có cùng dự đoán, chia ba khoảng theo ngưỡng 1/3 và 2/3.

Reliability pooled được ước lượng từ các heads. Bảng riêng từng head được làm trơn về pooled với `kappa=50`; các ô ít dữ liệu có backoff về bảng thô hơn. Reliability `r` được chuyển thành điểm liên tục `s=2r−1`, rồi qua softmax với `beta=5`. Đầu vào dùng bins rời rạc; winner dùng ánh xạ điểm đầu ra liên tục, được đối chiếu với ordinal scoring trong ablation.

Winner dùng **uniform residual `alpha=0.5`**:

```text
v_j(x) = softmax_j(5 * (2*r_j(x) - 1))
w_j(x) = 0.5/15 + 0.5*v_j(x)
p_D5(x) = sum_j w_j(x) * p_j(x)
```

50% trọng số giữ theo uniform, 50% được phân bổ theo reliability. Trọng số không âm và tổng bằng 1. Winner dùng xác suất raw và không bật consensus guard. Nhãn thật/correctness chỉ dùng để học bảng; khi suy luận không biết nhãn thật, group hoặc classifier nào đang đúng/sai.

## Quy trình lựa chọn và chống leakage

- Tái sử dụng OOF cache hợp lệ từ DPE refit theo 3-fold cho ba seed. Mỗi ảnh OOF được dự đoán bởi bộ heads không huấn luyện trên ảnh đó.
- Chia OOF theo lớp thành `meta_train/meta_select` tỷ lệ 80/20, dùng cùng split IDs cho ứng viên. Bins, smoothing, calibration và ngưỡng đều fit trên `meta_train`. Group không tham gia fit hay chọn cấu hình.
- Phase 1 khảo sát **29 cấu hình định trước**, tức 87 kết quả cấu hình/seed, cùng saved controls và oracle chỉ phục vụ chẩn đoán.
- Chọn bằng accuracy trên `meta_select`; hòa thì xét balanced accuracy và NLL. D5 đạt **96,806% accuracy trung bình trên selection**.
- Khóa winner và ma trận ablation trong `SELECTION_LOCK.json` trước khi mở test của lượt chạy này. Không đổi winner theo kết quả test.
- Phase 2 thực hiện A0–A20: confidence, agreement, smoothing, alpha/beta, calibration, top-k, số heads, split và 3/5-fold. Cấu hình tương đương được tái sử dụng; có **132 đánh giá test riêng biệt**.

Các đối chứng dùng cùng heads, sample IDs và thứ tự lớp. Ablation thay số heads có uniform control tương ứng. Group chỉ dùng để báo cáo sau khi khóa lựa chọn.

## Vì sao accuracy tăng nhưng WGA giảm?

Accuracy đo tỷ lệ đúng trên toàn bộ test; WGA là accuracy thấp nhất trong bốn nhóm nhãn chim × nền ảnh. Test gồm 4.510 landbird và 1.284 waterbird. Vì vậy, cải thiện trên lớp đông hơn có thể bù số lỗi tăng ở một nhóm ít mẫu hơn.

Nhóm yếu nhất của D5 qua ba seed là **waterbird trên nền land**, có accuracy lần lượt 91,745%, 88,474% và 90,810%. Kết quả không thể được diễn giải thành tăng robustness đồng đều. Nếu ưu tiên micro accuracy, D5 là ứng viên tốt hơn matched uniform trong lượt chạy này; nếu ưu tiên nhóm yếu nhất, uniform vẫn có lợi thế.

## Độ tin cậy và giới hạn

Khoảng bootstrap ghép cặp 95% cho mức tăng accuracy trung bình là **[+0,178; +0,575] điểm phần trăm**, có điều kiện trên các mô hình đã fit và tập test hiện tại. Khoảng này không đo độ bất định giữa các backbone độc lập.

Test đã được khảo sát nhiều lần trong các nghiên cứu trước, nên đây là **kết quả thăm dò**, chưa phải kiểm chứng trên test hoàn toàn mới. Selection khoảng 240 ảnh/seed tạo biến thiên và selection optimism. Ba seed dùng chung backbone; checkpoint backbone không lưu đủ lịch sử train để chứng minh toàn bộ provenance. Heads OOF và heads cuối fit toàn bộ validation cũng có thể khác phân phối.

Kết luận được hỗ trợ: **D5 cải thiện accuracy so với matched uniform trong nghiên cứu v2; lợi ích đó đi kèm giảm WGA khoảng 2,4 điểm phần trăm.** Chưa có cơ sở nói D5 tốt hơn trên mọi phân phối hoặc giải quyết hoàn toàn shortcut learning.

## Checkpoint và chạy lại

Run ID: `64b58d2c31b62c9d`. Đã lưu **831 task hoàn thành** bằng atomic writes, artifact hashes và completion markers. Resume toàn bộ tạo **0 task commit mới**. Năm test recovery/method đã qua, gồm resume giữa các tiến trình, file tạm bị cắt và output hỏng. Kết quả hợp lệ trước gián đoạn được giữ lại và tái sử dụng.

```powershell
.venv/Scripts/python.exe -m experiments.discrete_weighting.run --config experiments/discrete_weighting/config.yaml --phase all --resume
```

Code và hướng dẫn: [experiments/discrete_weighting](../../../experiments/discrete_weighting/README.md).

Các artifact lớn được lưu cục bộ tại `outputs/discrete_weighting_v2/64b58d2c31b62c9d/`, gồm state, handoff, fitted tables, bảng đầy đủ và dự đoán từng ảnh. Cache/checkpoint được loại khỏi Git theo quy ước repository; clone branch không tự tải các artifact này.

- [Báo cáo kỹ thuật đầy đủ](FULL_REPORT.md)
- [Bảng tổng hợp ablation](phase2_summary.csv)
- [Audit kết quả](audit.json)
- [Kiểm tra crash recovery](recovery_acceptance.json)
