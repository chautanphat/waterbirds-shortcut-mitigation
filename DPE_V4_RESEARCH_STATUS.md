# DPE v4 — Tạm dừng thử thêm biến thể, chuyển sang điểm yếu của G1

**Branch:** `dpe_optimized_version_4`. **Ngày:** 15/11/2026.

**Quyết định hiện tại:** tạm dừng mở rộng các biến thể weighting/normalization/advantage. Các biến thể đã chạy trong nghiên cứu CCW v6 đang bị kẹt ở cùng mức WGA của G1. Bước nghiên cứu tiếp theo sẽ tập trung trực tiếp vào các lỗi còn lại và cơ chế thất bại của G1. Commit này lưu kết quả và đánh dấu chuyển hướng; chưa triển khai hay chạy một thuật toán mới.

## Bằng chứng cho quyết định

**EXPLORATORY_ONLY — REUSED DEVELOPMENT — NOT INDEPENDENT CONFIRMATION**

So sánh năm phương pháp với chính xác cùng 15 prototype head mỗi ensemble, trên 3 seed × 3 outer fold. Mỗi seed gộp 1.199 ảnh C duy nhất trước khi tính AA/WGA. Các seed dùng lại cùng ảnh và một backbone ERM* đóng băng. Đây không phải kết quả ERM chuẩn hay test độc lập mới.

| Phương pháp | Mean AA (%) | Mean WGA (%) | ΔWGA so với G1 (điểm %) |
|---|---:|---:|---:|
| Uniform DPE | 95,746455 | 92,481203 | −0,501253 |
| G1 minimum-distance | **95,857659** | **92,982456** | 0 |
| G2 z-normalization | 95,829858 | 92,982456 | 0 |
| CCW-A | 95,802057 | 92,982456 | 0 |
| CCW-Full | 95,829858 | 92,982456 | 0 |

| Seed | Uniform WGA (%) | G1 WGA (%) | CCW-Full WGA (%) | G1 AA (%) | CCW-Full AA (%) |
|---:|---:|---:|---:|---:|---:|
| 0 | 90,977444 | 90,977444 | 90,977444 | 95,829858 | 95,996664 |
| 1 | 93,233083 | 93,984962 | 93,984962 | 96,080067 | 95,913261 |
| 2 | 93,233083 | 93,984962 | 93,984962 | 95,663053 | 95,579650 |

- CCW-Full bằng WGA của G1 ở **cả ba seed**; AA trung bình thấp hơn G1 0,027801 điểm %. Điều kiện giảm AA <1 điểm % mỗi seed vẫn đạt.
- Trong nhóm khó nhất waterbird/land (group 2, 133 ảnh/seed), CCW-Full không tạo thêm correction hoặc breakage nào so với G1. So với Uniform, seed 1 và 2 mỗi seed sửa thêm một ảnh; seed 0 sửa một ảnh và làm sai một ảnh.
- Advantage term có thay đổi trọng số: mean total variation 0,00688089; thay expert trọng số cao nhất trên 134/3.597 hàng seed/ảnh. Tuy nhiên chỉ đổi **một dự đoán** so với CCW-A, thuộc group 1, và **không tăng WGA**.
- Trước exploratory C, CCW-Full đã thua frozen G1 trên proxy B-only worst-class NLL: **0,62518272 so với 0,62063432** (thấp hơn tốt hơn). **NO-GO gốc được giữ nguyên.**
- WGA của CCW-Full cao hơn Uniform 0,501253 điểm %, nhưng paired 95% CI là [−0,286123; +1,253133]. So với G1, chênh lệch quan sát bằng 0; CI [−0,501253; +0,033359]. Bootstrap ghép cặp theo ID ảnh và dùng cùng lượt lấy mẫu cho cả ba seed.

Nhận định “kẹt WGA” áp dụng cho các biến thể và dữ liệu đã đo ở đây. Kết quả không được dùng để chọn lại beta/rho/alpha hoặc sửa quyết định của nghiên cứu gốc.

## Hướng tiếp theo: đánh thẳng vào điểm yếu của G1

Tạm dừng thêm grid cấu hình. Hướng phân tích kế tiếp cần trả lời các câu hỏi cụ thể:

1. **G1 còn sai ở đâu?** Lập danh sách lỗi còn lại, corrections/breakages so với Uniform, tập trung group 2; đối chiếu từng ảnh và lỗi nhãn đã được ghi nhận mà vẫn giữ nguyên báo cáo theo nhãn gốc.
2. **Có expert nào cứu được ảnh G1 sai không?** Tách trường hợp tất cả 15 expert cùng sai khỏi trường hợp vẫn có expert đúng. Từ đó đo phần còn có thể cải thiện bằng aggregation trong ensemble hiện có.
3. **Geometry có ưu tiên nhầm expert không?** Trên ảnh G1 sai, kiểm tra minimum distance, thứ hạng expert, confidence và lượng trọng số đặt vào expert sai; đo sự thay đổi khi bỏ từng expert. Đây là các phép chẩn đoán được đề xuất, chưa được chạy trong commit này.
4. **Điểm yếu có nhận diện được từ B không?** Chỉ sau khi xác định cơ chế thất bại mới thiết kế một can thiệp có mục tiêu và protocol riêng. Mọi fitting/selection tiếp tục chỉ dùng B/development được phép; các C cache đã xem chỉ mang giá trị exploratory. Xác nhận khả năng tổng quát hóa cần dữ liệu thực sự chưa được khảo sát.

Không thay production inference trong bước chuyển hướng này. Uniform vẫn mặc định; G1 vẫn là chế độ opt-in đã có. Không retrain head, không thêm kiến trúc và không phát động thí nghiệm mới khi tạo branch.

## Tài liệu và kết quả đã lưu

- [Báo cáo đầy đủ exploratory C, gồm kết quả từng seed](results/dpe/ccw_v6_exploratory_c_20261115/EXPLORATORY_C_REPORT.md).
- [Bảng AA/WGA/WCA/BA, paired differences và CI](results/dpe/ccw_v6_exploratory_c_20261115/EXPLORATORY_C_RESULTS.csv).
- [Độ chính xác từng subgroup và corrections/breakages](results/dpe/ccw_v6_exploratory_c_20261115/EXPLORATORY_C_PER_GROUP.csv).
- [Các ảnh đổi dự đoán](results/dpe/ccw_v6_exploratory_c_20261115/EXPLORATORY_C_FLIPS.csv).
- [Prediction cache: xác suất, đủ 15 trọng số, nhãn và ID](results/dpe/ccw_v6_exploratory_c_20261115/EXPLORATORY_C_PREDICTIONS.npz), [schema và thứ tự head](results/dpe/ccw_v6_exploratory_c_20261115/PREDICTION_SCHEMA.json).
- [NO-GO B-selection gốc](results/dpe/ccw_v6_20261115/LOCKED_CANDIDATE.json), [báo cáo gốc](results/dpe/ccw_v6_20261115/FINAL_REPORT.md).
- [Mã nguồn và lệnh chạy](experiments/ccw_v6/README.md), [manifest bản export](results/dpe/ccw_v6_exploratory_c_20261115/EXPORT_MANIFEST.json).

Các kết quả cũ và checkpoint local được bảo toàn. Đã pass 12 kiểm thử; kill/resume trên full-B thật và resume toàn workflow không tính lại task đã hoàn thành. 94 receipt diagnostic/delivery và 1.586 file gốc được kiểm tra toàn vẹn. Bản export Git chứa báo cáo, bảng, cache tổng hợp và fitted CCW tables; dữ liệu ảnh, checkpoint prototype và ZIP trùng lặp tiếp tục được giữ ngoài Git.
