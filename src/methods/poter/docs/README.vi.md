# POTER — Bản thực hành và chia sẻ mã nguồn

Repository này được clone từ [Sung-Ho-Jo/POTER](https://github.com/Sung-Ho-Jo/POTER) để tìm hiểu, chạy thử và chia sẻ các chỉnh sửa cá nhân. Đây là bản phát triển dựa trên mã nguồn gốc; công trình nghiên cứu và phương pháp POTER thuộc các tác giả bên dưới.

**POTER** sử dụng Optimal Transport (OT) để tính trọng số cho từng mẫu trước khi huấn luyện, nhằm tăng độ bền vững của mô hình trước tương quan giả (spurious correlations) và nhiễu nhãn (label noise).

- **Công trình gốc:** *Optimal Transport Reweighting for Robust Learning under Spurious Correlations and Label Noise*.
- **Tác giả:** Sung Ho Jo, Seonghwi Kim, Wonsang Yun và Minwoo Chae.
- **Tài liệu gốc:** [README upstream](README.upstream.md), giữ lại hướng dẫn thí nghiệm cho cả bốn bộ dữ liệu.

## Các chỉnh sửa trong bản này

So với mã nguồn gốc ở commit `a4a46ff`:

- Bật thanh tiến trình khi huấn luyện (`show_progress=True`).
- Lưu mô hình cuối cùng tại `last_model.pth` (`save_last=True`).
- Đặt `num_workers=0` cho DataLoader trong `run_expt.py`.
- Chuyển nhãn, nhóm và chỉ số mẫu sang kiểu `long` trong vòng lặp huấn luyện (`train.py`).
- Bổ sung README tiếng Việt để người xem có thể nắm cấu trúc và cách chạy.

Các mục trên mô tả thay đổi trong code; repository chưa kèm kết quả benchmark cho các thay đổi này.

## Cấu trúc chính

```text
POTER/
├── run_expt.py              # Chạy thí nghiệm POTER hoặc baseline ERM
├── extract_features.py      # Trích xuất đặc trưng và tạo cache cho OT
├── ot_reweight.py           # Tính OT potentials và trọng số mẫu
├── train.py                 # Vòng lặp huấn luyện và đánh giá
├── noise.py                 # Tạo nhiễu nhãn symmetric / subgroup
├── models.py               # Cấu hình mô hình
├── loss.py                 # Tính loss và thống kê theo nhóm
├── utils.py                # Logging, seed và tiện ích
├── data/                   # Mã nguồn đọc và xử lý dataset
├── scripts/run_experiments.sh
└── docs/README.upstream.md # Hướng dẫn gốc, đầy đủ cấu hình thí nghiệm
```

**Lưu ý:** `data/` trong repo chứa mã Python xử lý dataset. Dữ liệu ảnh/văn bản thực tế cần chuẩn bị riêng.

## Cài đặt môi trường

Cần Python, môi trường PyTorch phù hợp và **GPU NVIDIA có CUDA**: code hiện tại gọi trực tiếp `.cuda()` và `torch.cuda.set_device()`. Chỉ đọc mã nguồn thì không cần cài môi trường.

Tạo môi trường ảo:

```sh
python -m venv .venv
```

Kích hoạt trên Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Hoặc trên Linux / macOS:

```sh
source .venv/bin/activate
```

Cài `torch` và `torchvision` theo hệ điều hành/CUDA tại [hướng dẫn PyTorch](https://pytorch.org/get-started/locally/). README gốc liệt kê `torch >= 1.12`; repo chưa có file khóa phiên bản môi trường.

Cài các thư viện còn lại theo danh sách của repo gốc:

```sh
python -m pip install numpy scipy pandas tqdm Pillow scikit-learn transformers pytorch-transformers
```

Kiểm tra PyTorch nhận GPU:

```sh
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## Chuẩn bị dữ liệu

Truyền thư mục gốc chứa dataset qua `--data_dir`. Ví dụ dùng một thư mục riêng tên `datasets/`:

| Dataset | Giá trị `--dataset` | Thư mục dưới `datasets/` | Nguồn tham khảo |
| --- | --- | --- | --- |
| Waterbirds | `CUB` | `cub/` | [group_DRO](https://github.com/kohpangwei/group_DRO) |
| CelebA | `CelebA` | `celebA/` | [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
| ColoredMNIST | `CMNIST` | `CMNIST/` | [IRM](https://github.com/facebookresearch/InvariantRiskMinimization) |
| CivilComments | `CivilComments` | `civilcomments/` | [WILDS](https://wilds.stanford.edu/) |

Các loader yêu cầu đúng cấu trúc con và metadata, không chỉ thư mục trống. Ví dụ Waterbirds cần `datasets/cub/data/waterbird_complete95_forest2water2/metadata.csv` cùng các ảnh được metadata tham chiếu. Xem các loader trong `data/` và [tài liệu gốc](README.upstream.md#dataset-preparation) cho bộ dữ liệu tương ứng.

## Chạy thử với Waterbirds

Các lệnh bên dưới viết trên một dòng để dùng được trong PowerShell hoặc Bash. Thay `./datasets` bằng đường dẫn dữ liệu trên máy.

**1. Trích xuất đặc trưng** (thực hiện trước khi chạy POTER):

```sh
python extract_features.py --dataset CUB --data_dir ./datasets --ref_per_group 0:0,1:466,2:133,3:0 --output features/cub_features.pt --gpu 0
```

`--ref_per_group` chỉ định số mẫu validation lấy từ từng nhóm làm tập tham chiếu cho OT; số `0` nghĩa là bỏ nhóm đó. Cần đủ mẫu trong từng nhóm được chỉ định.

**2. Chạy POTER:**

```sh
python run_expt.py --dataset CUB --data_dir ./datasets --ot_cache features/cub_features.pt --f_power 0.1 --clip_cap 20 --n_epochs 300 --lr 1e-5 --weight_decay 1.0 --batch_size 128 --augment_data --seed 1 --gpu 0
```

**3. Hoặc chạy baseline ERM** bằng cách bỏ `--ot_cache`:

```sh
python run_expt.py --dataset CUB --data_dir ./datasets --n_epochs 300 --lr 1e-5 --weight_decay 1.0 --batch_size 128 --seed 1 --gpu 0
```

Muốn kiểm tra pipeline nhanh có thể giảm `--n_epochs` xuống `1`; đây không phải cấu hình tái lập kết quả nghiên cứu. Lần chạy đầu có thể tải trọng số mô hình pretrained.

Các cấu hình CelebA, ColoredMNIST, CivilComments và nhiễu nhãn nằm trong [README gốc](README.upstream.md#reproducing-results). Script `scripts/run_experiments.sh` dùng Bash (Linux, WSL hoặc Git Bash); sửa `DATA_DIR` và `FEATURE_DIR` trước khi chạy.

## Kết quả đầu ra

Mỗi lần chạy tạo thư mục riêng dưới `logs/<dataset>/<noise_bucket>/<run_name>/` (hoặc dưới đường dẫn `--log_dir`):

- `log.txt`: thông tin cấu hình và log huấn luyện.
- `train.csv`, `val.csv`, `test.csv`: thống kê theo epoch và theo nhóm.
- `last_model.pth`: checkpoint mô hình cuối cùng trong bản chỉnh sửa này.
- `val_16group.csv`, `test_16group.csv`: thống kê 16 nhóm cho CivilComments.

Feature cache, checkpoint, môi trường ảo và log được bỏ qua bằng `.gitignore`. Bạn clone repo cần tự chuẩn bị dữ liệu và tạo cache để chạy.

## Nguồn gốc

Khi sử dụng hoặc chia sẻ lại, giữ thông tin tác giả và liên kết [repository gốc](https://github.com/Sung-Ho-Jo/POTER). Tham khảo [README upstream](README.upstream.md) để đọc mô tả phương pháp và các thiết lập thí nghiệm của tác giả.
