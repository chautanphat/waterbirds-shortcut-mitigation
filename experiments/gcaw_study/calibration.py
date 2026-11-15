"""
Triển khai:
1. Calibrated Prototype Distance (A1: Head Z-score, A2: Class-Centered, A3: Class Z-score)
2. Conditional Expert Agreement & Marginal-Free Reliability Lift Table (OOF + Shrinkage)
3. Within-Sample Correct-Expert AUROC & Randomized-Weighting Control
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
from scipy.special import softmax


def compute_head_predictions_and_agreement(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Từ probs [N, 15, 2], tính:
      - head_preds: [N, 15] nhãn dự đoán của từng head
      - agreement: [N, 15] tỷ lệ 14 head còn lại đồng thuận với head j
    """
    head_preds = np.argmax(probs, axis=-1)  # [N, K]
    N, K = head_preds.shape
    # So sánh từng cặp head: eq[n, j, m] = 1 nếu head_preds[n, j] == head_preds[n, m]
    eq = (head_preds[:, :, None] == head_preds[:, None, :]).astype(np.float64)
    # Trừ đi chính nó (đường chéo = 1) và chia cho (K - 1)
    agreement = (np.sum(eq, axis=-1) - 1.0) / max(1, K - 1)
    return head_preds, agreement


def within_sample_zscore(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Chuẩn hóa vector 15 heads trên từng mẫu ảnh về mean=0, std=1.
    Nếu tất cả 15 heads có giá trị bằng nhau trên mẫu i (std < eps), trả về 0.
    Giúp bảo đảm khi AUROC nội mẫu cao thì biên độ đưa vào softmax luôn nhất quán.
    """
    mean = np.mean(mat, axis=1, keepdims=True)
    std = np.std(mat, axis=1, keepdims=True)
    return np.where(std > eps, (mat - mean) / (std + eps), 0.0)


class DistanceCalibrator:
    """
    Chuẩn hóa khoảng cách d_j^{min}(x) theo thống kê trên dữ liệu fitting/OOF
    mà KHÔNG sử dụng nhãn thực hay nhãn nhóm lúc suy diễn.
    Hỗ trợ 3 chế độ:
      - 'raw': Giữ nguyên d_j^{min} (Baseline G1)
      - 'A1_head_zscore': (d - mu_j) / (sigma_j + eps)
      - 'A2_class_center': d - mu_{j, y_hat} (Bảo toàn độ sắc nét phương sai của từng head)
      - 'A3_class_zscore': (d - mu_{j, y_hat}) / (sigma_{j, y_hat} + eps)
    """

    def __init__(self, mode: str = "A3_class_zscore", eps: float = 1e-6):
        assert mode in ("raw", "A1_head_zscore", "A2_class_center", "A3_class_zscore")
        self.mode = mode
        self.eps = eps
        self.mu_head_: Optional[np.ndarray] = None        # [K]
        self.std_head_: Optional[np.ndarray] = None       # [K]
        self.mu_cls_: Optional[np.ndarray] = None         # [K, 2]
        self.std_cls_: Optional[np.ndarray] = None        # [K, 2]

    def fit(self, distances_oof: np.ndarray, probs_oof: np.ndarray) -> "DistanceCalibrator":
        d_min = np.min(distances_oof, axis=-1)            # [N, K]
        head_preds = np.argmax(probs_oof, axis=-1)        # [N, K]
        _, K = d_min.shape

        self.mu_head_ = np.mean(d_min, axis=0)
        self.std_head_ = np.std(d_min, axis=0)

        self.mu_cls_ = np.zeros((K, 2), dtype=np.float64)
        self.std_cls_ = np.ones((K, 2), dtype=np.float64)

        for j in range(K):
            for c in (0, 1):
                mask = head_preds[:, j] == c
                if np.sum(mask) >= 5:
                    self.mu_cls_[j, c] = float(np.mean(d_min[mask, j]))
                    self.std_cls_[j, c] = float(np.std(d_min[mask, j]))
                else:
                    self.mu_cls_[j, c] = float(self.mu_head_[j])
                    self.std_cls_[j, c] = float(self.std_head_[j])
        return self

    def transform(self, distances: np.ndarray, probs: np.ndarray) -> np.ndarray:
        d_min = np.min(distances, axis=-1)                # [N, K]
        if self.mode == "raw":
            return d_min.copy()

        assert self.mu_head_ is not None and self.mu_cls_ is not None
        if self.mode == "A1_head_zscore":
            return (d_min - self.mu_head_[None, :]) / (self.std_head_[None, :] + self.eps)

        head_preds = np.argmax(probs, axis=-1)            # [N, K]
        N, K = d_min.shape
        out = np.zeros_like(d_min, dtype=np.float64)
        for j in range(K):
            preds_j = head_preds[:, j]
            mu_jc = self.mu_cls_[j, preds_j]
            if self.mode == "A2_class_center":
                out[:, j] = d_min[:, j] - mu_jc
            else:  # A3_class_zscore
                std_jc = self.std_cls_[j, preds_j]
                out[:, j] = (d_min[:, j] - mu_jc) / (std_jc + self.eps)
        return out


class ConditionalReliabilityTable:
    """
    Bảng ước lượng độ tin cậy có điều kiện g_j(x) trên dữ liệu OOF với Empirical Bayes Shrinkage.
    Hỗ trợ 2 chế độ:
      - 'B1_absolute_prob': g_j(x) = 2 * P_shrunk(C_j=y | d_tilde, A_j, y_hat_j) - 1 (Công thức gốc)
      - 'B2_marginal_free_lift': g_j(x) = 2 * [P_shrunk(C_j=y | d_tilde, A_j, y_hat_j) - P_marginal(C_j=y | A_j, y_hat_j)]
        (Khử hoàn toàn hiệu ứng lề của Agreement để không thưởng cho phe đa số shortcut).
    """

    def __init__(
        self,
        mode: str = "B2_marginal_free_lift",
        n_dist_bins: int = 3,
        agree_thresholds: Tuple[float, float] = (0.35, 0.70),
        shrinkage_m: float = 25.0,
    ):
        assert mode in ("B1_absolute_prob", "B2_marginal_free_lift")
        self.mode = mode
        self.n_dist_bins = n_dist_bins
        self.agree_thresholds = np.array(agree_thresholds, dtype=np.float64)
        self.shrinkage_m = float(shrinkage_m)

        self.dist_edges_: Optional[np.ndarray] = None
        # Bảng lưu xác suất đúng đã làm trơn: shape [n_dist_bins, 3 (agree_bins), 2 (pred_class)]
        self.p_shrunk_table_: Optional[np.ndarray] = None
        # Bảng lưu xác suất lề theo (agree_bin, pred_class): shape [3, 2]
        self.p_marginal_table_: Optional[np.ndarray] = None
        # Bảng đếm số lượng mẫu OOF trong từng ô: shape [n_dist_bins, 3, 2]
        self.counts_table_: Optional[np.ndarray] = None

    def _bin_features(
        self, d_tilde: np.ndarray, agreement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self.dist_edges_ is not None
        q_bin = np.clip(
            np.digitize(d_tilde, self.dist_edges_[1:-1], right=False),
            0,
            self.n_dist_bins - 1,
        )
        a_bin = np.clip(
            np.digitize(agreement, self.agree_thresholds, right=True),
            0,
            2,
        )
        return q_bin, a_bin

    def fit(
        self,
        d_tilde_oof: np.ndarray,
        probs_oof: np.ndarray,
        y_oof: np.ndarray,
    ) -> "ConditionalReliabilityTable":
        """
        Ước lượng bảng tin cậy từ dữ liệu OOF (gộp toàn bộ N x 15 quan sát head-level).
        Không sử dụng nhãn nhóm (group labels).
        """
        head_preds, agreement = compute_head_predictions_and_agreement(probs_oof)
        head_correct = (head_preds == y_oof[:, None]).astype(np.float64)  # [N, K]

        # Chia phân vị trên phân phối khoảng cách chuẩn hóa toàn cục
        self.dist_edges_ = np.quantile(
            d_tilde_oof.ravel(), np.linspace(0.0, 1.0, self.n_dist_bins + 1)
        )
        q_bins, a_bins = self._bin_features(d_tilde_oof, agreement)

        global_acc = float(np.mean(head_correct))
        self.p_marginal_table_ = np.full((3, 2), global_acc, dtype=np.float64)
        for a in range(3):
            for c in range(2):
                mask_ac = (a_bins == a) & (head_preds == c)
                n_ac = int(np.sum(mask_ac))
                k_ac = float(np.sum(head_correct[mask_ac]))
                # Làm trơn nhẹ mức lề về global_acc
                self.p_marginal_table_[a, c] = (k_ac + 10.0 * global_acc) / (n_ac + 10.0)

        self.p_shrunk_table_ = np.zeros((self.n_dist_bins, 3, 2), dtype=np.float64)
        self.counts_table_ = np.zeros((self.n_dist_bins, 3, 2), dtype=np.int64)

        for q in range(self.n_dist_bins):
            for a in range(3):
                for c in range(2):
                    mask_qac = (q_bins == q) & (a_bins == a) & (head_preds == c)
                    n_qac = int(np.sum(mask_qac))
                    k_qac = float(np.sum(head_correct[mask_qac]))
                    prior_p = self.p_marginal_table_[a, c]

                    self.counts_table_[q, a, c] = n_qac
                    self.p_shrunk_table_[q, a, c] = (
                        k_qac + self.shrinkage_m * prior_p
                    ) / (n_qac + self.shrinkage_m)

        return self

    def predict_g(self, d_tilde: np.ndarray, probs: np.ndarray) -> np.ndarray:
        """
        Trả về ma trận tín hiệu tin cậy g_j(x) kích thước [N, 15]:
        - B1_absolute_prob: 2 * P_shrunk(C=y | q, a, c) - 1
        - B2_marginal_free_lift:
          Khử hiệu ứng đám đông bằng cách trừ đi kỳ vọng trung bình tại tầng khoảng cách Far (Q3)
          của nhóm đa số, đồng thời chuẩn hóa theo lớp dự đoán để bảo toàn độ chênh giữa
          phe thiểu số gần tâm (Near, LowAgree) và phe đa số xa tâm (Far, HighAgree).
        """
        assert self.p_shrunk_table_ is not None and self.p_marginal_table_ is not None
        head_preds, agreement = compute_head_predictions_and_agreement(probs)
        q_bins, a_bins = self._bin_features(d_tilde, agreement)

        p_cell = self.p_shrunk_table_[q_bins, a_bins, head_preds]
        if self.mode == "B1_absolute_prob":
            return 2.0 * p_cell - 1.0

        # B2: Độ nâng của ô (q, a, c) so với trung bình khoảng cách trong cùng (a, c)
        # CỘNG VỚI hiệu chỉnh chống đa số xa tâm (Interaction Contrast)
        p_col_mean = np.mean(self.p_shrunk_table_, axis=0)[a_bins, head_preds]
        within_col_lift = p_cell - p_col_mean

        # Điểm thưởng tương tác hình học: thưởng cho head ở Near (q=0) và phạt Far (q=2)
        # tỷ lệ thuận với độ nhạy thực nghiệm của cột đó
        class_baseline = np.mean(self.p_marginal_table_, axis=0)[head_preds]
        debiased_score = within_col_lift + 0.5 * (p_cell - class_baseline)
        return np.clip(2.0 * debiased_score, -1.0, 1.0)


def compute_gcaw_weights(
    d_tilde: np.ndarray,
    g_rel: Optional[np.ndarray],
    gamma: float,
    beta: float = 0.0,
    alpha: float = 0.25,
    standardize_within_sample: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính điểm số a_j(x) và trọng số w_j(x) với tùy chọn chuẩn hóa Z-score nội mẫu
    để tránh suy biến biên độ softmax.
    """
    K = d_tilde.shape[1]
    d_comp = within_sample_zscore(d_tilde) if (standardize_within_sample and gamma > 0) else d_tilde
    scores = -gamma * d_comp

    if g_rel is not None and beta != 0.0:
        g_comp = within_sample_zscore(g_rel) if standardize_within_sample else g_rel
        scores = scores + beta * g_comp

    r = softmax(scores, axis=1)
    w = ((1.0 - alpha) / K) + alpha * r
    return scores, w


def compute_within_sample_auroc(
    scores: np.ndarray,
    probs: np.ndarray,
    y: np.ndarray,
    sample_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Tính Within-Sample Correct-Expert AUROC:
    Xác suất trung bình trong cùng một bức ảnh, một head đoán đúng được gán điểm
    a_j(x) lớn hơn một head đoán sai. Chỉ tính trên các ảnh có cả head đúng lẫn head sai.
    """
    if sample_mask is not None:
        scores = scores[sample_mask]
        probs = probs[sample_mask]
        y = y[sample_mask]

    head_preds = np.argmax(probs, axis=-1)  # [M, K]
    correct_mat = head_preds == y[:, None]  # [M, K]

    aurocs = []
    for i in range(len(y)):
        c_idx = np.where(correct_mat[i])[0]
        w_idx = np.where(~correct_mat[i])[0]
        if len(c_idx) == 0 or len(w_idx) == 0:
            continue
        s_c = scores[i, c_idx][:, None]  # [n_c, 1]
        s_w = scores[i, w_idx][None, :]  # [1, n_w]
        wins = np.sum(s_c > s_w) + 0.5 * np.sum(s_c == s_w)
        aurocs.append(float(wins) / (len(c_idx) * len(w_idx)))

    return float(np.mean(aurocs)) if aurocs else 0.5


def evaluate_ensemble_metrics(
    weights: np.ndarray,
    probs: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
) -> Dict[str, float]:
    """
    Tính WGA, AA, độ chính xác từng nhóm và tổng trọng số đặt lên các head đoán sai
    trên các mẫu thuộc Group 2 (Worst Minority).
    """
    weighted_probs = np.einsum("nj,njk->nk", weights, probs)
    preds = np.argmax(weighted_probs, axis=1)
    correct = (preds == y).astype(np.float64)

    head_preds = np.argmax(probs, axis=-1)
    wrong_head_mask = (head_preds != y[:, None]).astype(np.float64)
    wrong_weight_per_sample = np.sum(weights * wrong_head_mask, axis=1)

    res = {
        "AA": float(np.mean(correct)),
        "WrongHeadWeight_All": float(np.mean(wrong_weight_per_sample)),
    }
    g_accs = []
    for g in range(4):
        m = group == g
        acc_g = float(np.mean(correct[m])) if np.any(m) else 0.0
        res[f"Group_{g}_Acc"] = acc_g
        g_accs.append(acc_g)

    g2_mask = group == 2
    res["WrongHeadWeight_G2"] = (
        float(np.mean(wrong_weight_per_sample[g2_mask])) if np.any(g2_mask) else 0.0
    )
    res["WGA"] = float(np.min(g_accs))
    return res


def run_randomized_weight_control(
    base_weights: np.ndarray,
    probs: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
    n_permutations: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Đối chứng trọng số ngẫu nhiên (Randomized-Weighting Control):
    Giữ nguyên phân phối độ lớn trọng số w_j(x) trên từng ảnh nhưng hoán vị ngẫu nhiên
    thứ tự 15 heads để kiểm tra xem mức tăng WGA đến từ chất lượng ranking thực sự
    hay chỉ do nhiễu phương sai ngẫu nhiên.
    """
    rng = np.random.default_rng(seed)
    N, K = base_weights.shape
    wga_samples = []
    aa_samples = []

    for _ in range(n_permutations):
        # Sinh ma trận hoán vị độc lập cho từng mẫu hoặc từng batch head
        perm_indices = np.argsort(rng.random((N, K)), axis=1)
        w_perm = np.take_along_axis(base_weights, perm_indices, axis=1)
        m = evaluate_ensemble_metrics(w_perm, probs, y, group)
        wga_samples.append(m["WGA"])
        aa_samples.append(m["AA"])

    wga_arr = np.array(wga_samples)
    return {
        "Random_WGA_Mean": float(np.mean(wga_arr)),
        "Random_WGA_Std": float(np.std(wga_arr)),
        "Random_WGA_P95": float(np.quantile(wga_arr, 0.95)),
        "Random_AA_Mean": float(np.mean(aa_samples)),
    }
