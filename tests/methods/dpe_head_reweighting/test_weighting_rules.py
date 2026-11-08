"""Rule fits and cached-suite isolation; all accuracy fixtures are synthetic."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.methods.dpe_head_reweighting import rule_suite as suite
from src.methods.dpe_head_reweighting import weighting_rules as rules
from src.methods.dpe_head_reweighting.ensemble import PREPROCESSING
from src.methods.dpe_head_reweighting.experiment import digest, write_json
from src.methods.dpe_head_reweighting.reweighting import fit_select

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / 'configs/dpe_head_reweighting/weighting_rules.yaml'


def config():
    return yaml.safe_load(CONFIG.read_text())


def probabilities(correct, y):
    correct = np.asarray(correct)
    return np.stack([np.where(y == 0, correct, 1 - correct),
                     np.where(y == 1, correct, 1 - correct)], axis=-1)


def arrays():
    g = np.repeat(['00', '01', '10', '11'], 3)
    y = np.array([int(k[0]) for k in g])
    perfect = probabilities(np.full(len(y), .9), y)
    bad = perfect[:, ::-1]
    return np.stack([perfect] * 5 + [bad] * 10), y, g


def make_bundle(tmp_path):
    source = tmp_path / 'original'
    p, y, g = arrays()
    head_ids = [f'head-{i:02d}' for i in range(15)]
    parts = {role: [dict(id=f'{role}-{i}', y=int(yy), group=gg) for i, (yy, gg) in enumerate(zip(y, g))]
             for role in 'ABC'}
    manifest = dict(parts=parts, validation=sum(parts.values(), []), provenance=dict(track='standard', seed=0))
    write_json(source / 'experiment/manifest.json', manifest)
    model = source / 'checkpoints/ensemble.pt'
    model.parent.mkdir(parents=True)
    model.write_bytes(b'Synthetic identity only; no model is loaded by the suite.')
    original = fit_select(p, y, g, p, y, g)
    original.update(head_ids=head_ids, lock=dict(ensemble=digest(model), manifest=digest(source / 'experiment/manifest.json')))
    write_json(source / 'experiment/selection.json', original)

    def cache(role):
        path = source / f'experiment/predictions/{role}.npz'
        path.parent.mkdir(exist_ok=True)
        np.savez(path, probabilities=p, labels=y, groups=g, ids=[f'{role}-{i}' for i in range(len(y))],
                 role=role, ensemble_hash=digest(model), preprocessing=PREPROCESSING)
    cache('B')
    cache('C')
    return source, tmp_path / 'suite', cache


def test_joint_known_complementary_optimum_and_uniform_penalty():
    y = np.array([0, 0, 1, 1])
    g = np.array(['00', '01', '10', '11'])
    p = np.stack([probabilities(q, y) for q in ([.9, .9, .3, .3], [.3, .3, .9, .9], [.1] * 4)])
    w, record = rules.joint_weights(p, y, g, .01, config()['solver'])
    np.testing.assert_allclose(w, [.5, .5, 0], atol=1e-5)
    assert record['success'] and record['max_group_nll'] == pytest.approx(-np.log(.6), abs=1e-6)
    identical = np.repeat(p[:1], 15, axis=0)
    w, _ = rules.joint_weights(identical, y, g, 1, config()['solver'])
    np.testing.assert_allclose(w, np.ones(15) / 15, atol=1e-7)


def test_joint_rejects_solver_failure(monkeypatch):
    from scipy.optimize import OptimizeResult
    monkeypatch.setattr(rules, 'minimize', lambda *a, **k: OptimizeResult(success=False, message='iteration limit'))
    p, y, g = arrays()
    with pytest.raises(ValueError, match='Joint solver failed'):
        rules.joint_weights(p, y, g, .1, config()['solver'])


def test_all_rules_and_original_power_parity():
    p, y, g = arrays()
    result = rules.fit_rules(p, y, g, p, y, g, config())
    assert len(result['candidates']) == 20
    old = fit_select(p, y, g, p, y, g)
    for actual, expected in zip(result['candidates'][:13], old['candidates']):
        np.testing.assert_allclose(actual['weights'], expected['weights'], atol=1e-12)
        assert actual['metrics'] == expected['metrics']
    assert result['per_family_selected']['power'] == result['candidates'][
        old['candidates'].index(old['selected'])]['id']
    rows = {r['id']: r for r in result['candidates']}
    for family in ('softmax', 'top_k', 'joint'):
        assert rows[result['per_family_selected'][family]]['metrics']['wga'] == 1
    # B-only fitting: reversing all C outcomes may change selection, never fits.
    changed = rules.fit_rules(p, y, g, p[:, :, ::-1], y, g, config())
    for left, right in zip(result['candidates'], changed['candidates']):
        np.testing.assert_array_equal(left['weights'], right['weights'])


def test_uniform_fallback_and_stable_topk_ties():
    p, y, g = arrays()
    p = np.repeat(p[:1], 15, axis=0)
    result = rules.fit_rules(p, y, g, p, y, g, config())
    assert result['selected'] == 'uniform'
    assert set(result['per_family_selected'].values()) == {'uniform'}
    row = next(r for r in result['candidates'] if r['id'] == 'top-k5')
    np.testing.assert_array_equal(row['weights'], [.2] * 5 + [0] * 10)


def test_cached_lifecycle_needs_no_training_or_test_for_selection(tmp_path):
    source, output, cache = make_bundle(tmp_path)
    original = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    decision = suite.select(source, CONFIG, output)  # No test file exists.
    assert len(decision['candidates']) == 20
    (output / 'candidate_comparison.csv').unlink()  # Interrupted after the JSON write.
    assert suite.select(source, CONFIG, output) == decision
    assert (output / 'candidate_comparison.csv').is_file()
    cache('test')
    result = suite.evaluate(source, CONFIG, output)
    assert len(result['results']) == 6
    assert result['results']['joint']['metrics']['wga'] == 1
    assert suite.evaluate(source, CONFIG, output) == result
    assert (output / 'report.md').is_file() and (output / 'run.json').is_file()
    with (output / 'candidate_comparison.csv').open() as f:
        assert len(list(csv.DictReader(f))) == 20
    for path, content in original.items():
        assert (source / path).read_bytes() == content
    # Tampering with a report or changing the frozen choice cannot reuse test results.
    edited = json.loads((output / 'metrics.json').read_text())
    edited['results']['joint']['metrics']['wga'] = .5
    write_json(output / 'metrics.json', edited)
    with pytest.raises(ValueError, match='Existing test result'):
        suite.evaluate(source, CONFIG, output)
    decision['selected'] = 'top-k5'
    write_json(output / 'selection.json', decision)
    with pytest.raises(ValueError, match='Frozen suite selection'):
        suite.select(source, CONFIG, output)


def test_invalidated_config_and_cache_and_source_protection(tmp_path):
    source, output, cache = make_bundle(tmp_path)
    suite.select(source, CONFIG, output)
    with pytest.raises(ValueError, match='separate output'):
        suite.select(source, CONFIG, source / 'new-suite')
    changed_config = tmp_path / 'rules.yaml'
    c = config()
    c['softmax_betas'] = [20.]
    changed_config.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError, match='Frozen suite selection'):
        suite.select(source, changed_config, output)
    cache('test')
    path = source / 'experiment/predictions/test.npz'
    with np.load(path) as f:
        data = dict(f)
    data['ids'][0] = 'B-0'
    np.savez(path, **data)
    with pytest.raises(ValueError, match='overlaps validation'):
        suite.evaluate(source, CONFIG, output)
    cache('test')
    path = source / 'experiment/predictions/B.npz'
    with np.load(path) as f:
        data = dict(f)
    data['ids'][1] = data['ids'][0]
    np.savez(path, **data)
    with pytest.raises(ValueError, match='duplicate'):
        suite.select(source, CONFIG, tmp_path / 'other-suite')


def test_rules_notebook_execution_and_reuse(tmp_path, monkeypatch):
    nbformat = pytest.importorskip('nbformat')
    nbclient = pytest.importorskip('nbclient')
    pytest.importorskip('ipykernel')
    import sys
    import shutil
    source, _, cache = make_bundle(tmp_path)
    cache('test')
    # Run a real editable copy, autosaving after every cell, as Jupyter does.
    repo = tmp_path / 'checkout'
    for folder in ('src', 'scripts', 'configs'):
        shutil.copytree(ROOT / folder, repo / folder)
    notebook_path = repo / 'notebooks/dpe_weighting_rules.ipynb'
    notebook_path.parent.mkdir()
    template = nbformat.read(ROOT / 'notebooks/dpe_weighting_rules.ipynb', as_version=4)
    output = tmp_path / 'notebook-suite'
    for cell in template.cells:
        if cell.cell_type == 'code':
            cell.source = cell.source.replace('REPO_PATH = None', f'REPO_PATH = {str(repo)!r}')
            if 'SOURCE_RUN_DIR =' in cell.source:
                # Keep the actual readiness checks, changing only user settings.
                cell.source = cell.source.replace(
                    '(REPO / "outputs/dpe_head_reweighting_runs/h2-standard-seed0").resolve()',
                    f'Path({str(source)!r})')
                cell.source = cell.source.replace(
                    '(REPO / "outputs/dpe_weighting_rule_runs" / SUITE_NAME).resolve()',
                    f'Path({str(output)!r})')
    nbformat.write(template, notebook_path)
    kernels = tmp_path / 'jupyter/kernels/h2-rules'
    kernels.mkdir(parents=True)
    (kernels / 'kernel.json').write_text(json.dumps(dict(
        argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        display_name='H2 rules test', language='python',
        env=dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))))
    monkeypatch.setenv('JUPYTER_PATH', str(tmp_path / 'jupyter'))
    identities = None
    for _ in range(2):
        notebook = nbformat.read(notebook_path, as_version=4)
        nbclient.NotebookClient(notebook, timeout=120, kernel_name='h2-rules',
                                on_cell_executed=lambda **kwargs: nbformat.write(notebook, notebook_path),
                                resources={'metadata': {'path': str(repo / 'notebooks')}}).execute()
        current = [(output / name).read_bytes() for name in ('selection.json', 'metrics.json')]
        if identities is not None:
            assert identities == current
        identities = current
        assert output.with_suffix('.zip').is_file()


def test_notebook_hash_ignores_execution_but_detects_code_edits(tmp_path):
    path = tmp_path / 'notebook.ipynb'
    notebook = json.loads((ROOT / 'notebooks/dpe_weighting_rules.ipynb').read_text())
    path.write_text(json.dumps(notebook))
    original = suite.notebook_code_hash(path)
    code = next(c for c in notebook['cells'] if c['cell_type'] == 'code')
    code['execution_count'] = 42
    code['outputs'] = [dict(output_type='stream', name='stdout', text='saved output')]
    notebook['metadata']['modified'] = 'autosaved'
    path.write_text(json.dumps(notebook))
    assert suite.notebook_code_hash(path) == original
    code['source'].append('\nprint("changed code")\n')
    path.write_text(json.dumps(notebook))
    assert suite.notebook_code_hash(path) != original


def test_incomplete_source_reports_missing_files(tmp_path):
    with pytest.raises(ValueError, match='Incomplete H2 source bundle.*experiment/manifest.json'):
        suite.select(tmp_path / 'wrong-level', CONFIG, tmp_path / 'suite')
