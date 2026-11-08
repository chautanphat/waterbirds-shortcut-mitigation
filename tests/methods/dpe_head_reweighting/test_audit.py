"""Independent aggregation oracle, nonuniform winners and archived-run auditing."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.methods.dpe_head_reweighting import experiment as ex
from src.methods.dpe_head_reweighting.reweighting import combine, fit_select, metrics

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('independent_audit', ROOT / 'scripts/audit_dpe_head_reweighting.py')
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


@pytest.mark.parametrize('mode', ['weighted_wins', 'uniform_wins', 'same_labels_different_probabilities'])
def test_independent_saved_prediction_audit(tmp_path, mode):
    y = np.array([0, 0, 1, 1])
    g = np.array(['00', '01', '10', '11'])
    perfect = .1 + .8 * np.eye(2)[y]
    bad = perfect[:, ::-1]
    # 15 heads exercise the experiment's real head axis, not just two heads.
    pb = np.stack([perfect] * 5 + [bad] * 10)
    if mode == 'weighted_wins':
        pc = pb.copy()
    elif mode == 'uniform_wins':
        pc = np.stack([bad] * 5 + [perfect] * 10)
    else:
        pc = np.stack([perfect] * 5 + [.3 + .4*np.eye(2)[y]] * 10)
    result = fit_select(pb, y, g, pc, y, g)
    if mode == 'weighted_wins':
        assert result['selected']['alpha'] > 0
        assert result['selected']['metrics']['wga'] == 1
    else:
        assert result['selected']['alpha'] == 0
    if mode == 'same_labels_different_probabilities':
        assert all(r['c_prediction_changes'] == 0 for r in result['candidates'])
        assert any(r['c_probability_max_abs_delta'] > 0 for r in result['candidates'])
    parts = {role: [dict(id=f'{role}-{k}', y=int(label), group=group)
                   for k, (label, group) in enumerate(zip(y, g))] for role in 'ABC'}
    manifest = dict(parts=parts, validation=sum(parts.values(), []))
    ex.write_json(tmp_path / 'experiment/manifest.json', manifest)
    checkpoint = tmp_path / 'checkpoints/ensemble.pt'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'fixture identity only; audit does not load a model')
    lock = dict(manifest=ex.digest(tmp_path / 'experiment/manifest.json'), ensemble=ex.digest(checkpoint))
    result.update(lock=lock, head_ids=[f'h{i}' for i in range(15)])
    selection = tmp_path / 'experiment/selection.json'
    ex.write_json(selection, result)
    prediction_dir = tmp_path / 'experiment/predictions'
    prediction_dir.mkdir()
    for role, p in [('B', pb), ('C', pc), ('test', pc)]:
        np.savez(prediction_dir / f'{role}.npz', probabilities=p, labels=y, groups=g,
                 ids=[f'{role}-{i}' for i in range(4)], role=role, ensemble_hash=lock['ensemble'])
    w = np.array(result['selected']['weights'])
    uniform, weighted = combine(pc, np.ones(15)/15), combine(pc, w)
    u, v = metrics(uniform, y, g), metrics(weighted, y, g)
    uc, vc = uniform.argmax(1) == y, weighted.argmax(1) == y
    ids = np.array([f'test-{i}' for i in range(4)])
    saved = dict(lock=lock, selection_hash=ex.digest(selection), uniform=u, selected=v,
                 alpha=result['selected']['alpha'], gamma=result['selected']['gamma'], weights=w.tolist(),
                 delta_pp={k: 100*(v[k]-u[k]) for k in ('overall','wga')},
                 fixed_ids=ids[~uc & vc].tolist(), new_error_ids=ids[uc & ~vc].tolist(),
                 effective_heads=float(1/np.square(w).sum()))
    path = tmp_path / 'results/metrics.json'
    ex.write_json(path, saved)
    checked = audit_module.audit(tmp_path)
    assert checked['status'] == 'passed'
    assert checked['selected']['alpha'] == result['selected']['alpha']
    # New bundles detect even a selection-file edit after final evaluation.
    original = selection.read_text()
    selection.write_text(original + '\n')
    with pytest.raises(ValueError, match='Selection changed'):
        audit_module.audit(tmp_path)
    selection.write_text(original)
    # Existing pre-audit bundles still receive an independent metric/grid check.
    del saved['selection_hash']
    ex.write_json(path, saved)
    assert not audit_module.audit(tmp_path)['selection_hash_recorded']
    saved['selected']['wga'] = .123
    ex.write_json(path, saved)
    with pytest.raises(ValueError, match='Metric wga differs'):
        audit_module.audit(tmp_path)


def test_randomized_selection_against_independent_oracle():
    rng = np.random.RandomState(42)
    y = np.tile([0, 0, 1, 1], 20)
    groups = np.tile(['00', '01', '10', '11'], 20)
    wins = set()
    for _ in range(100):
        pb, pc = [rng.dirichlet([.5,.5], size=(15,len(y))) for _ in range(2)]
        selected = fit_select(pb,y,groups,pc,y,groups)
        scores = np.array([audit_module.measure(p,y,groups)['wga'] for p in pb])
        oracle = []
        for row in selected['candidates']:
            powers = scores ** row['gamma']
            w = (1-row['alpha'])/15 + row['alpha']*powers/powers.sum()
            pred = sum(w[h]*pc[h] for h in range(15))
            np.testing.assert_allclose(combine(pc,w),pred,atol=1e-15,rtol=0)
            m = audit_module.measure(pred,y,groups)
            oracle.append((m['wga'],m['overall'],-row['alpha'],-row['gamma']))
        winner = selected['candidates'][max(range(13),key=lambda i:oracle[i])]
        assert winner == selected['selected']
        wins.add(winner['alpha'] > 0)
    assert wins == {False, True}
