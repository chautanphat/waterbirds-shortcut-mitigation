"""Per-dataset settings fixed in PLAN.md Sect. 4 (head settings from dpe4subpop train_all_pe.sh)."""

COMMON_HEAD = dict(heads=15, epochs=20, batch_size=256, momentum=0.9, entropic_scale=30.0, wd_weight=10.0)

DATASETS = {
    # Reference recipe; used only by the equivalence test against the Waterbirds code.
    "Waterbirds": dict(classes=2, feature_dim=2048, head=dict(COMMON_HEAD, lr=1e-3, cov_reg=5e5)),
    "CelebA": dict(classes=2, feature_dim=2048, head=dict(COMMON_HEAD, lr=5e-4, cov_reg=5e5)),
    "CivilCommentsFine": dict(classes=2, feature_dim=768, head=dict(COMMON_HEAD, lr=1e-4, cov_reg=1e5)),
    "MultiNLI": dict(classes=3, feature_dim=768, head=dict(COMMON_HEAD, lr=1e-4, cov_reg=5e6)),
}

# Calibration settings shared with the Waterbirds study (configs/dpe_independent/study.yaml).
CALIBRATION = dict(steps=400, learning_rate=0.01, hidden_units=8, seed_offset=10000)


def head_settings(dataset):
    return dict(DATASETS[dataset]["head"])


def config(dataset):
    """The cfg dict consumed by the ported fitting code (same keys as study.yaml)."""
    return dict(CALIBRATION, head_training=head_settings(dataset))
