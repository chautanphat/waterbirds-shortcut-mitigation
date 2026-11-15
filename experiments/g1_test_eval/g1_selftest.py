"""Software checks for g1_eval.py; synthetic data only, no Waterbirds performance claims."""
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import g1_eval as g


def teammate_distance_mix(probabilities, distances, gamma=50.0, alpha=0.25):
    """Verbatim logic of the teammate's verified G1 (baselines/dpe4subpop/src/dpe/geometry.py, v5 branch),
    in its [samples, 15, 2] layout."""
    p, d = np.asarray(probabilities), np.asarray(distances)
    score = -gamma * d.min(-1)
    r = np.exp(score - score.max(1, keepdims=True))
    r = r / r.sum(1, keepdims=True)
    weights = (1 - alpha) / 15 + alpha * r
    return (p * weights[..., None]).sum(1)


class G1Tests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        logits = rng.normal(size=(15, 300, 2))
        self.q = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
        self.d = rng.uniform(0.3, 1.6, size=(15, 300, 2))

    def test_g1_matches_teammate_recipe(self):
        mine = g.mix(self.q, g.g1_route(self.d), g.ALPHA)
        theirs = teammate_distance_mix(self.q.transpose(1, 0, 2), self.d.transpose(1, 0, 2))
        np.testing.assert_allclose(mine, theirs, rtol=0, atol=1e-12)

    def test_alpha_zero_is_uniform_and_weights_sum_to_one(self):
        np.testing.assert_allclose(g.mix(self.q, g.g1_route(self.d), 0.0), self.q.mean(0), atol=1e-12)
        np.testing.assert_allclose(g.mix(self.q, g.g1_route(self.d), g.ALPHA).sum(1), 1, atol=1e-12)

    def test_permutation_keeps_each_images_weight_multiset(self):
        out = g.arms(self.q, self.d, seed=3)
        self.assertEqual(len(out['perm']), g.PERMUTATIONS)
        route = g.g1_route(self.d)
        rng = np.random.RandomState(3)
        order = np.argsort(rng.rand(*route.shape), axis=0)
        permuted = np.take_along_axis(route, order, axis=0)
        np.testing.assert_allclose(np.sort(permuted, 0), np.sort(route, 0))
        np.testing.assert_allclose(out['perm'][0], g.mix(self.q, permuted, g.ALPHA))


def smoke():
    """Both real head trainers, one epoch, synthetic 32-D features; then the report path."""
    import torch
    torch.set_num_threads(1)
    from corrected_selftest import synthetic_source
    cfg = copy.deepcopy(g.old.adapters()['grid'].load_config())
    cfg['head_training']['epochs'] = 1
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for recipe in ('normal_erm', 'ermstar'):
            source = synthetic_source(1200, 32, recipe, root / recipe)
            with patch.object(g.old, 'get_source', return_value=source):
                for split in (0, 1):
                    status = g.run_cell((source.spec, split, str(root / 'out'), cfg, None))
                    if status[2] != 'done':
                        raise AssertionError(str(status) + '\n' + '\n'.join(
                            p.read_text() for p in (root / 'out/errors').glob('*.txt')))
                    assert g.run_cell((source.spec, split, str(root / 'out'), cfg, None))[2] == 'cached'
        cell = json.loads((root / 'out/results/synthetic-normal_erm/split00.json').read_text())
        assert set(cell['arms']) == {'uniform', 'g1', 'g1_a0.1', 'g1_a0.5', 'perm', 'conf'}, cell['arms'].keys()
        report = g.summarize(root / 'out', None)
        assert 'INCOMPLETE' in report  # two synthetic sources only
    print('G1 SYNTHETIC SMOKE PASSED')


if __name__ == '__main__':
    import sys
    if '--smoke' in sys.argv:
        smoke()
    else:
        unittest.main(argv=[sys.argv[0], '-v'])
