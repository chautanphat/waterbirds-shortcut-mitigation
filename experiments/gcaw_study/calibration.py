"""
Triển khai chuẩn hóa khoảng cách (A1/A2/A3), bảng tin cậy có điều kiện với Shrinkage (B1/B2),
và chuẩn hóa biên độ nội mẫu (Within-Sample Z-score) giúp khắc phục triệt để lỗi tắt biên độ ở B2.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
from scipy.special import softmax


def within_sample_zscore(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Chuẩn hóa mỗi hàng (15 heads trên cùng 1 mẫu ảnh i) về trung bình 0, độ lệch chuẩn 1.
    Đảm bảo khi thứ tự xếp hạng (AUROC) tốt thì biên độ đẩy vào softmax luôn đủ mạnh,
    không bị suy biến về 0 như ở v7.0.
    """
    mean = np.mean(mat, axis=1, keepdims=True)
    std = np.std(mat, axis=1, keepdims=True)
    return np.where(std > eps, (mat - mean) / (std + eps), 0.0)


def compute_head_predictions_and_agreement(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính nhãn dự đoán head_preds [N, 15] và tỷ lệ đồng thuận LOO agreement [N, 15].
    """
    head_preds = np.argmax(probs, axis=-1)
    _, K = head_preds.shape
    eq = (head_preds[:, :, None] == head_preds[:, None, :]).astype(np.float64)
    agreement = (np.sum(eq, axis=-1) - 1.0) / max(1, K - 1)
    return head_preds, agreement


class DistanceCalibrator:
    """
    Chuẩn hóa khoảng cách d_j^{min}(x) trên tập OOF/fitting của từng head:
      - 'raw': Giữ nguyên d_j^{min} (Baseline G1)
      - 'A1_head_zscore': (d - mu_j) / (sigma_j + eps)
      - 'A2_class_center': d - mu_{j, y_hat} (Bảo toàn phương sai tự nhiên của head)
      - 'A3_class_zscore': (d - mu_{j, y_hat}) / (sigma_{j, y_hat} + eps)
    """

    def __init__(self, mode: str = "A3_class_zscore", eps: float = 1e-6):
        assert mode in ("raw", "A1_head_zscore", "A2_class_center", "A3_class_zscore")
        self.mode = mode
        self.eps = eps
        self.mu_head_: Optional[np.ndarray] = None
        self.std_head_: Optional[np.ndarray] = None
        self.mu_cls_: Optional[np.ndarray] = None
        self.std_cls_: Optional[np.ndarray] = None

    def fit(self, distances_oof: np.ndarray, probs_oof: np.ndarray) -> "DistanceCalibrator":
        d_min = np.min(distances_oof, axis=-1)
        head_preds = np.argmax(probs_oof, axis=-1)
        _, K = d_min.shape

        self.mu_head_ = np.mean(d_min, axis=0)
        self.std_head_ = np.std(d_min, axis=0)
        self.mu_cls_ = np.zeros((K, 2), dtype=np.float64)
        self.std_cls_ = np.ones((K, 2), dtype=np.float64)

        for j in range(K):
            for c in (0, 1):
                mask = head_preds[:, j] == c
                if np.sum(mask) >= 10:
                    self.mu_cls_[j, c] = float(np.mean(d_min[mask, j]))
                    self.std_cls_[j, c] = float(np.std(d_min[mask, j]))
                else:
                    self.mu_cls_[j, c] = float(self.mu_head_[j])
                    self.std_cls_[j, c] = float(self.std_head_[j])
        return self

    def transform(self, distances: np.ndarray, probs: np.ndarray) -> np.ndarray:
        d_min = np.min(distances, axis=-1)
        if self.mode == "raw":
            return d_min.copy()

        assert self.mu_head_ is not None and self.mu_cls_ is not None
        if self.mode == "A1_head_zscore":
            return (d_min - self.mu_head_[None, :]) / (self.std_head_[None, :] + self.eps)

        head_preds = np.argmax(probs, axis=-1)
        _, K = d_min.shape
        out = np.zeros_like(d_min, dtype=np.float64)
        for j in range(K):
            preds_j = head_preds[:, j]
            mu_jc = self.mu_cls_[j, preds_j]
            if self.mode == "A2_class_center":
                out[:, j] = d_min[:, j] - mu_jc
            else:
                std_jc = self.std_cls_[j, preds_j]
                out[:, j] = (d_min[:, j] - mu_jc) / (std_jc + self.eps)
        return out


class ConditionalReliabilityTable:
    """
    Ước lượng bảng tin cậy có điều kiện P(C_j=y | q_j, a_j, y_hat_j) trên dữ liệu OOF:
      - 'B1_absolute_prob': g_j(x) = 2 * P_shrunk(q, a, c) - 1
      - 'B2_marginal_free_lift':
        Khử hiệu ứng đám đông bằng cách lấy độ nâng hình học nội cột:
          within_agree_lift = P_shrunk(q, a, c) - P_marginal(a, c)
        cộng với độ nhạy khoảng cách (Near vs. Far slope) của chính tầng đồng thuận đó,
        giúp bảo toàn tín hiệu ngay cả khi cột có phương sai nhỏ.
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
        self.p_shrunk_table_: Optional[np.ndarray] = None
        self.p_marginal_table_: Optional[np.ndarray] = None
        self.lift_table_: Optional[np.ndarray] = None
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
        head_preds, agreement = compute_head_predictions_and_agreement(probs_oof)
        head_correct = (head_preds == y_oof[:, None]).astype(np.float64)

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

        # Tính sẵn bảng Lift khử hiệu ứng đám đông:
        # So sánh mỗi ô (q, a, c) với trung bình 3 tầng khoảng cách của chính cột (a, c) đó
        col_mean = np.mean(self.p_shrunk_table_, axis=0, keepdims=True)  # [1, 3, 2]
        self.lift_table_ = self.p_shrunk_table_ - col_mean               # [3, 3, 2]
        return self

    def predict_g(self, d_tilde: np.ndarray, probs: np.ndarray) -> np.ndarray:
        assert (
            self.p_shrunk_table_ is not None
            and self.p_marginal_table_ is not None
            and self.lift_table_ is not None
        )
        head_preds, agreement = compute_head_predictions_and_agreement(probs)
        q_bins, a_bins = self._bin_features(d_tilde, agreement)

        if self.mode == "B1_absolute_prob":
            p_cell = self.p_shrunk_table_[q_bins, a_bins, head_preds]
            return 2.0 * p_cell - 1.0

        # B2: Marginal-Free Lift chuẩn hóa theo biên độ thực nghiệm của bảng
        raw_lift = self.lift_table_[q_bins, a_bins, head_preds]
        scale = max(0.02, float(np.max(np.abs(self.lift_table_))))
        return np.clip(raw_lift / scale, -1.0, 1.0)


def compute_gcaw_weights(
    d_tilde: np.ndarray,
    g_rel: Optional[np.ndarray],
    gamma: float,
    beta: float = 0.0,
    alpha: float = 0.25,
    standardize_within_sample: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính điểm a_j(x) = -gamma * d_comp + beta * g_comp và w_j(x) = (1-alpha)/15 + alpha * softmax(a_j).
    Khi standardize_within_sample=True, cả d_tilde và g_rel đều được đưa về cùng thang đo Z-score
    trên 15 heads của từng bức ảnh trước khi nhân với gamma và beta.
    """
    K = d_tilde.shape[1]
    if gamma != 0.0:
        d_comp = within_sample_zscore(d_tilde) if standardize_within_sample else d_tilde
        scores = -gamma * d_comp
    else:
        scores = np.zeros_like(d_tilde, dtype=np.float64)

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
    if sample_mask is not None:
        scores = scores[sample_mask]
        probs = probs[sample_mask]
        y = y[sample_mask]

    head_preds = np.argmax(probs, axis=-1)
    correct_mat = head_preds == y[:, None]

    aurocs = []
    for i in range(len(y)):
        c_idx = np.where(correct_mat[i])[0]
        w_idx = np.where(~correct_mat[i])[0]
        if len(c_idx) == 0 or len(w_idx) == 0:
            continue
        s_c = scores[i, c_idx][:, None]
        s_w = scores[i, w_idx][None, :]
        wins = np.sum(s_c > s_w) + 0.5 * np.sum(s_c == s_w)
        aurocs.append(float(wins) / (len(c_idx) * len(w_idx)))

    return float(np.mean(aurocs)) if aurocs else 0.5


def evaluate_ensemble_metrics(
    weights: np.ndarray,
    probs: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
) -> Dict[str, float]:
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
    rng = np.random.default_rng(seed)
    N, K = base_weights.shape
    wga_samples = []
    aa_samples = []

    for _ in range(n_permutations):
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
