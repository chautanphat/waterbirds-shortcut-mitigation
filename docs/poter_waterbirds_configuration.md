# POTER Waterbirds: Configuration va Hyperparameters

Tài liệu này mô tả implementation Waterbirds đang có trong repository và cấu hình notebook Kaggle đã chạy. Các giá trị dưới đây không phải toàn bộ giá trị mặc định của CLI và không mặc nhiên đại diện cho mọi cấu hình trong paper.

## Pipeline

1. Bat dau tu **ResNet-50 pretrained ImageNet**.
2. Trich xuat feature 2.048 chieu de tinh Optimal Transport (OT); feature cache **khong** phai checkpoint ERM da fine-tune tren Waterbirds.
3. Tinh trong so tung mau train bang class-conditioned entropic OT giua feature train va tap tham chieu validation.
4. Khoi tao **ResNet-50 pretrained ImageNet moi**, thay classifier thanh 2 lop va fine-tune toan bo model voi WeightedRandomSampler theo trong so POTER.

Trong pipeline này không có bước train ERM Waterbirds riêng trước POTER. Model sử dụng trọng số ImageNet có sẵn rồi fine-tune trên Waterbirds. Nếu bỏ `--ot_cache`, `run_expt.py` chạy plain ERM; cần truyền cùng các tham số tối ưu hóa và augmentation để so sánh.

## Dataset va moi truong

| Muc | Gia tri |
| --- | --- |
| Dataset | Waterbirds / `CUB` |
| Archive | `waterbird_complete95_forest2water2` |
| Target | `waterbird_complete95` |
| Confounder | `forest2water2` |
| Split | Official train/validation/test: 4.795 / 1.199 / 5.794 mau |
| GPU | Kaggle Tesla T4, GPU `0` |
| So GPU su dung | 1 (khong dung distributed training) |
| Framework da chay | PyTorch `2.11.0+cu130`, CUDA available |
| Seed | `1` |
| Epoch | `300` (`0` den `299`) |
| Batch size | `128` |
| DataLoader workers | `0` |
| Pin memory | Bat |

## Backbone va toi uu hoa

| Hyperparameter | Gia tri |
| --- | --- |
| Backbone | `torchvision.models.resnet50(pretrained=True)` |
| Khoi tao | ImageNet pretrained |
| Classifier | `Linear(2048, 2)` |
| Fine-tuning | Toan bo ResNet-50 |
| Loss | Cross-entropy, `reduction="none"`, sau đó lấy trung bình; OT weight áp dụng qua sampler |
| Optimizer | SGD |
| Learning rate | `1e-5` |
| Momentum | `0.9` |
| Weight decay | `1.0` |
| Learning-rate scheduler | Tat |
| Train from scratch | Tat |
| Label noise | Khong co (`None`, fraction `0.0`) |
| GroupDRO / robust loss | Tat |
| Group reweighting thong thuong | Tat; POTER dung sample weights |

## Data augmentation

Augmentation chi ap dung cho tap train:

| Bien doi | Gia tri |
| --- | --- |
| Random resized crop | Kich thuoc 224, scale `(0.7, 1.0)`, ratio `(1.0, 1.3333333333)` |
| Color jitter | `brightness=0.125`, `contrast=0.5`, `saturation=0.5`, `hue=0.05`; áp dụng với xác suất `0.5` |
| Horizontal flip | Xác suất `0.5` |
| Rotation | Tat |
| Normalization | ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |

Validation va test dung resize 256, center crop 224 va ImageNet normalization.

## POTER / Optimal Transport

| Hyperparameter | Gia tri | Vai tro |
| --- | ---: | --- |
| Feature cache | `/kaggle/working/poter-features/waterbirds_standard.pt` | Feature dung de tinh OT |
| Feature backbone | ResNet-50 ImageNet pretrained | Dung de trich feature, khong phai ERM checkpoint Waterbirds |
| Train feature shape | `(4795, 2048)` | Feature cua tap train |
| Reference feature shape | `(599, 2048)` | Feature cua tap tham chieu |
| Feature extraction batch size | `32` | Khác batch size huấn luyện `128` |
| Reference seed | `0` | Khác training seed `1` |
| `ref_per_group` | `{0: 0, 1: 466, 2: 133, 3: 0}` | Thanh phan tap tham chieu |
| OT cost | Squared Euclidean distance | Khoang cach feature theo tung class |
| `reg_scale` | `0.03` | `epsilon = reg_scale * mean(cost_matrix)` |
| `f_power` | `0.1` | Do sac cua sample reweighting |
| `clip_cap` | `20` | Gia tri weight toi da truoc khi chuan hoa lai |
| Sinkhorn `max_iter` | `300` | So vong lap toi da |
| Sinkhorn `stop_thr` | `1e-6` | Nguong dung |
| Normalization | Per-class median-MAD, can bang mass moi class, global normalization | On dinh thang do va tranh class dong chi phoi |

Trong code, `f0 = f_star - min(f_star)`, `tau = median(f0) * f_power`, rồi `w = exp(-f0 / (tau + 1e-12))`. Weight được chuẩn hóa trong từng class, cân bằng tổng mass giữa các class, chuẩn hóa toàn cục, clip ở 20 và chuẩn hóa lại. Vì có bước chuẩn hóa sau clipping, weight cuối cùng có thể lớn hơn 20.

Weight được đưa vào `WeightedRandomSampler(num_samples=4795, replacement=True)`: mỗi epoch lấy 4.795 lượt mẫu, có thể lặp ảnh. Loss không nhân trực tiếp với OT weight. Vì vậy cách gọi chính xác là **fine-tuning bằng cross-entropy với OT-weighted sampling**; câu trả lời trước gọi là weighted cross-entropy chưa mô tả đúng implementation.

OT và weight được tính một lần trước vòng lặp epoch từ feature cache cố định; không tính lại theo backbone sau mỗi epoch. Tập tham chiếu lấy từ validation, không lấy từ test.

## Nguồn đối chiếu trong repository

- `baselines/poter/scripts/kaggle_waterbirds.py`: cấu hình thực tế, khởi tạo model, optimizer, checkpoint và evaluation.
- `baselines/poter/extract_features.py`: backbone trích feature và reference sampling.
- `baselines/poter/ot_reweight.py`: Sinkhorn, công thức weight và thứ tự chuẩn hóa/clipping.
- `baselines/poter/data/dro_dataset.py`: weighted sampling có hoàn lại.
- `baselines/poter/data/cub_dataset.py`: augmentation.
- `baselines/poter/run_expt.py`: CLI và nhánh plain ERM khi không có OT cache.

## Checkpoint va chon model

| File | Noi dung |
| --- | --- |
| `last_training.pt` | Model, optimizer, RNG state va loss state sau moi epoch; dung de resume cung mot run |
| `best_validation.pt` | Checkpoint co validation worst-group accuracy (WGA) cao nhat |
| `train.csv`, `val.csv`, `test.csv` | Metric theo tung epoch |
| `config.json` | Cau hinh va checksum cua run |

Checkpoint duoc luu sau moi epoch. Model duoc chon bang **validation WGA**; sau do phai doc test metric cua dung epoch da chon. Khong chon checkpoint theo test metric.

## Lenh Kaggle da chay

```bash
python scripts/kaggle_waterbirds.py \
  --data-dir /kaggle/working/poter-data \
  --cache /kaggle/working/poter-features/waterbirds_standard.pt \
  --run-dir /kaggle/working/poter-results/CUB/Clean/POTER_CUB_seed1_kaggle \
  --seed 1 \
  --epochs 300 \
  --batch-size 128 \
  --max-hours 9.5
```

De reproduce day du, lap lai cho `--seed 2` va `--seed 3`, sau do tong hop:

```bash
python scripts/summarize_waterbirds.py /kaggle/working/poter-results --seeds 1 2 3
```

## Lenh ERM de so sanh cong bang

ERM bo qua OT cache va sample weighting, nhung giu nguyen backbone, epoch, batch size, learning rate, weight decay, augmentation va seed:

```bash
python run_expt.py \
  --dataset CUB \
  --data_dir /kaggle/working/poter-data \
  --n_epochs 300 \
  --lr 1e-5 \
  --weight_decay 1.0 \
  --batch_size 128 \
  --augment_data \
  --seed 1 \
  --gpu 0
```
