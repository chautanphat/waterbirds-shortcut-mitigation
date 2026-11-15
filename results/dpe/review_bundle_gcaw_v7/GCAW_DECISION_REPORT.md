# [DIAGNOSTIC & ABLATION STUDY — GCAW PROTOCOL v7]
# Báo cáo Nghiệm thu Thực nghiệm GCAW & Pre-Flight Oracle

## 1. Bảng Tổng hợp Kết quả Trung bình (MEAN_ALL_SEEDS)

| method                                | seed           |    WGA |     AA |   G0_Acc |   G1_Acc |   G2_Acc |   G3_Acc |   AUROC_All |   AUROC_MarginZone |   AUROC_UnifErrors |   WrongWeight_G2 |   Corrected_Samples |   Broken_Samples |   Net_Gain_Samples |
|:--------------------------------------|:---------------|-------:|-------:|---------:|---------:|---------:|---------:|------------:|-------------------:|-------------------:|-----------------:|--------------------:|-----------------:|-------------------:|
| 00_Uniform_Baseline                   | MEAN_ALL_SEEDS |   0.28 |  90.33 |      100 |      100 |     0.28 |      100 |       50    |              50    |              50    |            66.67 |                0    |             0    |               0    |
| 01_G1_Raw_Distance                    | MEAN_ALL_SEEDS |   0    |  90.31 |      100 |      100 |     0    |      100 |       28.71 |              28.71 |              28.71 |            74.99 |                0    |             0.33 |              -0.33 |
| 02_Ablation_A1_Head_ZScore            | MEAN_ALL_SEEDS | 100    | 100    |      100 |      100 |   100    |      100 |       98.9  |              98.9  |              98.9  |            50.06 |              116    |             0    |             116    |
| 03_Ablation_A2_Class_Centered         | MEAN_ALL_SEEDS | 100    | 100    |      100 |      100 |   100    |      100 |      100    |             100    |             100    |            50.09 |              116    |             0    |             116    |
| 04_Ablation_A3_Class_ZScore           | MEAN_ALL_SEEDS | 100    | 100    |      100 |      100 |   100    |      100 |       96.18 |              96.18 |              96.17 |            50.23 |              116    |             0    |             116    |
| 05_Ablation_B1_Std_Reliability_Only   | MEAN_ALL_SEEDS | 100    | 100    |      100 |      100 |   100    |      100 |      100    |             100    |             100    |            50.89 |              116    |             0    |             116    |
| 06_Ablation_B2_MarginalFree_Lift_Only | MEAN_ALL_SEEDS |   0.83 |  90.39 |      100 |      100 |     0.83 |      100 |      100    |             100    |             100    |            66.28 |                0.67 |             0    |               0.67 |
| 07_Ablation_C1_GCAW_A2_Plus_Lift      | MEAN_ALL_SEEDS | 100    | 100    |      100 |      100 |   100    |      100 |      100    |             100    |             100    |            50.08 |              116    |             0    |             116    |
| 08_Ablation_C2_GCAW_A3_Plus_Lift      | MEAN_ALL_SEEDS | 100    | 100    |      100 |      100 |   100    |      100 |       96.4  |              96.4  |              96.38 |            50.22 |              116    |             0    |             116    |
| 99_Stage1_GCAW_Grid_Oracle_UpperBound | MEAN_ALL_SEEDS | 100    | 100    |      100 |      100 |   100    |      100 |        0    |               0    |               0    |            37.04 |                0    |             0    |               0    |
| 90_Randomized_Weight_Control_P95      | MEAN_ALL_SEEDS |  36.78 |  93.14 |        0 |        0 |     0    |        0 |       50    |              50    |              50    |             0    |                0    |             0    |               0    |

---

## 2. Phán quyết Tự động theo 4 Cổng Kiểm định (Go / No-Go Gates)

* **Gate 0 (Pre-Flight GCAW Grid Oracle):** `PASS`
  * Oracle WGA = **100.00%** vs. G1 WGA = **0.00%** (Độ chênh: **+100.00 pp**, yêu cầu $\ge +1.50\text{ pp}$).
* **Gate A (Calibrated Distance Ranking trên Vùng Biên):** `PASS`
  * Within-Sample AUROC trên Margin Subset: G1 = **28.71%**, A2 (Class-Centered) = **100.00%**, A3 (Class-ZScore) = **96.18%**.
* **Gate B (Marginal-Free Lift vs. Standard OOF Reliability trên Ảnh Uniform Sai):** `FAIL`
  * AUROC trên ảnh Uniform sai: B1 (Standard OOF) = **100.00%** vs. B2 (Marginal-Free Lift) = **100.00%**.
* **Gate C (Combined GCAW vs. G1 & Randomized Control P95):** `PASS`
  * Best GCAW WGA = **100.00%** vs. G1 = **0.00%** và Randomized Control P95 = **36.78%**.
