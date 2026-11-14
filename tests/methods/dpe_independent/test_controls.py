import numpy as np
import pytest

from src.methods.dpe_independent import controls

CONFIG = dict(learning_rate=0.01, steps=400)


def toy(n=200, shift=0.0, seed=0):
    rng = np.random.RandomState(seed)
    y = np.repeat([0, 0, 1, 1], n // 4)
    a = np.tile([0, 1], n // 2)
    groups = np.array([f"{yi}{ai}" for yi, ai in zip(y, a)])
    s = (2*y - 1) * 1.5 + rng.normal(size=n) + shift
    return s, y, groups


def test_score_is_logit_and_clipped():
    assert controls.score([0.5])[0] == pytest.approx(0.0)
    assert controls.score([0.9])[0] == pytest.approx(np.log(9))
    assert np.isfinite(controls.score([0.0, 1.0])).all()


def test_offset_recovers_shift_direction():
    s, y, g = toy(shift=2.0)
    b = controls.fit_offset(s, y, g, "average", 0.01, CONFIG, seed=0)
    assert b < -0.5  # scores pushed toward waterbird should be pulled back


def test_offset_penalty_shrinks_toward_zero():
    s, y, g = toy(shift=2.0)
    weak = controls.fit_offset(s, y, g, "average", 0.01, CONFIG, seed=0)
    strong = controls.fit_offset(s, y, g, "average", 10.0, CONFIG, seed=0)
    assert abs(strong) < abs(weak)


def test_offset_is_deterministic():
    s, y, g = toy()
    assert controls.fit_offset(s, y, g, "maximum", 0.1, CONFIG, 3) == controls.fit_offset(s, y, g, "maximum", 0.1, CONFIG, 3)


def test_wga_threshold_beats_or_matches_zero():
    s, y, g = toy(shift=1.5)
    t, wga = controls.wga_threshold(s, y, g)
    assert wga >= controls.worst_group_accuracy((s > 0).astype(int), y, g)
    assert wga == pytest.approx(controls.worst_group_accuracy((s > t).astype(int), y, g))


def test_wga_threshold_prefers_zero_on_ties():
    y = np.array([0, 0, 1, 1]); g = np.array(["00", "01", "10", "11"])
    s = np.array([-5.0, -4.0, 4.0, 5.0])
    t, wga = controls.wga_threshold(s, y, g)
    assert t == 0.0 and wga == 1.0


def test_offset_average_matches_uniform_at_zero():
    folds = [np.array([0.0, 1.0]), np.array([-1.0, 2.0])]
    expected = np.mean([1/(1+np.exp(-f)) for f in folds], axis=0)
    assert np.allclose(controls.offset_average(folds, 0.0), expected)


def test_missing_group_rejected():
    with pytest.raises(ValueError):
        controls.group_index(["00", "01", "10"])
