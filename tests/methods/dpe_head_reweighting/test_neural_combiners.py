"""Synthetic functional checks; no Waterbirds accuracy is established here."""
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch
import yaml

from src.methods.dpe_head_reweighting import neural_combiner as neural
from src.methods.dpe_head_reweighting import neural_suite as suite
from src.methods.dpe_head_reweighting import neural_study as study
from src.methods.dpe_head_reweighting.ensemble import PREPROCESSING
from src.methods.dpe_head_reweighting.experiment import digest, object_hash, write_json
from src.methods.dpe_head_reweighting.reweighting import fit_select as old_select

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / suite.DEFAULT_CONFIG


def config():
    return yaml.safe_load(CONFIG.read_text())


def routing_problem():
    # Expert 0 works on background 0; expert 1 on background 1.
    # Head 2 reveals background, not class. The task is XOR-like for a stacker.
    g = np.repeat(['00', '01', '10', '11'], 8)
    y = np.array([int(v[0]) for v in g])
    a = np.array([int(v[1]) for v in g])
    q = np.full((15, len(y)), .5)
    q[0] = np.where(y == a, .1, .9)
    q[1] = 1 - q[0]
    q[2] = np.where(a == 0, .6, .4)
    return np.stack([1 - q, q], axis=-1), y, g


def make_bundle(tmp_path, seed=0):
    source = tmp_path / f'original-{seed}'
    p, y, g = routing_problem()
    parts = {role: [dict(id=f'{role}-{i}', y=int(yy), group=str(gg)) for i, (yy, gg) in enumerate(zip(y, g))]
             for role in 'ABC'}
    write_json(source / 'experiment/manifest.json', dict(parts=parts, validation=sum(parts.values(), []),
                                                        provenance=dict(track='standard', seed=seed)))
    model = source / 'checkpoints/ensemble.pt'
    model.parent.mkdir(parents=True)
    model.write_bytes(b'Identity only. This suite must never load or modify a head.')
    original = old_select(p, y, g, p, y, g)
    original.update(head_ids=[f'h{i}' for i in range(15)],
                    lock=dict(ensemble=digest(model), manifest=digest(source / 'experiment/manifest.json')))
    write_json(source / 'experiment/selection.json', original)

    def cache(role):
        path = source / f'experiment/predictions/{role}.npz'
        path.parent.mkdir(exist_ok=True)
        np.savez(path, probabilities=p, labels=y, groups=g, ids=[f'{role}-{i}' for i in range(len(y))],
                 role=role, ensemble_hash=digest(model), preprocessing=PREPROCESSING)
    cache('B')
    cache('C')
    return source, cache


@pytest.fixture
def small_config(tmp_path):
    c = config()
    c.update(steps=4, weight_decays=[.01])
    path = tmp_path / 'small.yaml'
    path.write_text(yaml.safe_dump(c))
    return path


@pytest.mark.parametrize('kind', ['mlp', 'gate'])
def test_learns_complementary_experts_and_roundtrips(kind):
    p, y, g = routing_problem()
    c = config()
    record, history = neural.fit(p, y, g, c, kind, .001, .25, 10000)
    prediction, w = neural.predict(json.loads(json.dumps(record)), p)
    assert neural.metrics(prediction, y, g)['wga'] == 1.
    assert history['history'][-1]['b_group_balanced_bce'] < history['history'][0]['b_group_balanced_bce']
    if kind == 'gate':
        assert (w >= 0).all()
        np.testing.assert_allclose(w.sum(axis=1), 1, atol=1e-12)
        assert np.max(np.ptp(w, axis=0)) > .1  # Learned input-dependent routing.
        assert (prediction[:, 1] >= p[:, :, 1].min(axis=0)).all()
        assert (prediction[:, 1] <= p[:, :, 1].max(axis=0)).all()
    else:
        assert w is None  # Stacking coefficients are not convex head weights.
    again, _ = neural.predict(record, p)
    np.testing.assert_array_equal(prediction, again)


def test_logistic_learns_reversed_signal():
    p, y, g = routing_problem()
    q = np.tile(np.where(y == 0, .9, .1), (15, 1))
    p = np.stack([1 - q, q], axis=-1)
    model, _ = neural.fit(p, y, g, config(), 'logistic', .01, 0., 10000)
    prediction, _ = neural.predict(model, p)
    assert neural.metrics(prediction, y, g)['wga'] == 1.


def test_whole_prediction_dropout_and_inference():
    q = torch.full((200, 15), .7, dtype=torch.float64)
    keep = neural.prediction_mask(q, 1., torch.Generator().manual_seed(0))
    assert (keep.sum(dim=1) == 1).all()
    with neural.cpu_randomness(1):
        gate = neural.Combiner('gate', 15, 8)
        q[:, 0] = .2
        prediction, w = gate(q, keep)
        assert (w[~keep] == 0).all()
        torch.testing.assert_close(w.sum(dim=1), torch.ones(200, dtype=torch.float64))
        torch.testing.assert_close(prediction, q[keep])
        inference, weights = gate(q)
        torch.testing.assert_close(inference, q.mean(dim=1))
        torch.testing.assert_close(weights, torch.full_like(q, 1 / 15))


def test_c_does_not_affect_fits_and_global_rng_is_preserved():
    p, y, g = routing_problem()
    c = config()
    c.update(steps=8, weight_decays=[.01])
    torch.manual_seed(99)
    rng = torch.get_rng_state().clone()
    first = neural.fit_select(p, y, g, p, y, g, c, 0)
    assert first['candidates'][0]['c_probability_max_abs_delta'] == 0.
    assert first['candidates'][0]['c_prediction_changes'] == 0
    torch.testing.assert_close(torch.get_rng_state(), rng)
    second = neural.fit_select(p, y, g, p[:, :, ::-1], y, g, c, 0)
    assert [r['model'] for r in first['candidates']] == [r['model'] for r in second['candidates']]
    assert first == neural.fit_select(p, y, g, p, y, g, c, 0)


def test_uniform_wins_complete_ties():
    p, y, g = routing_problem()
    q = np.tile(np.where(y == 0, .1, .9), (15, 1))
    p = np.stack([1 - q, q], axis=-1)
    c = config()
    c.update(steps=20, weight_decays=[.01])
    result = neural.fit_select(p, y, g, p, y, g, c, 0)
    assert result['selected'] == 'uniform'
    assert set(result['per_family_selected'].values()) == {'uniform'}
    assert result['complementarity']['B']['uniform_errors_with_correct_head'] == 0


def test_select_without_test_and_frozen_lifecycle(tmp_path, small_config, monkeypatch):
    source, cache = make_bundle(tmp_path)
    before = {str(p): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    output = tmp_path / 'neural'
    roles = []
    original_loader = suite.validation_context.__globals__['load_cache']
    def spy(root, role, *args):
        roles.append(role)
        return original_loader(root, role, *args)
    monkeypatch.setitem(suite.validation_context.__globals__, 'load_cache', spy)
    decision = suite.select(source, small_config, output)
    assert set(roles) == {'B', 'C'}
    assert not (source / 'experiment/predictions/test.npz').exists()
    (output / 'candidate_comparison.csv').unlink()
    assert suite.select(source, small_config, output) == decision
    assert (output / 'candidate_comparison.csv').exists()
    cache('test')
    result = suite.evaluate(source, small_config, output)
    assert len(result['results']) == 7
    assert suite.evaluate(source, small_config, output) == result
    for path, content in before.items():
        assert Path(path).read_bytes() == content
    edited = json.loads((output / 'metrics.json').read_text())
    edited['results']['uniform']['metrics']['wga'] = -1
    write_json(output / 'metrics.json', edited)
    with pytest.raises(ValueError, match='Existing neural test result'):
        suite.evaluate(source, small_config, output)
    decision['candidates'][1]['model']['state']['network.bias'][0] += .1
    write_json(output / 'selection.json', decision)
    with pytest.raises(ValueError, match='Frozen neural selection'):
        suite.select(source, small_config, output)


def test_source_protection_changed_inputs_and_bad_cache(tmp_path, small_config):
    source, cache = make_bundle(tmp_path)
    output = tmp_path / 'neural'
    suite.select(source, small_config, output)
    with pytest.raises(ValueError, match='separate output'):
        suite.select(source, small_config, source / 'new')
    c = yaml.safe_load(small_config.read_text())
    c['steps'] += 1
    small_config.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError, match='Frozen neural selection'):
        suite.evaluate(source, small_config, output)
    cache('test')
    path = source / 'experiment/predictions/test.npz'
    with np.load(path) as f:
        data = dict(f)
    data['ids'][0] = 'B-0'
    np.savez(path, **data)
    new = tmp_path / 'changed-config'
    suite.select(source, small_config, new)
    with pytest.raises(ValueError, match='overlaps validation'):
        suite.evaluate(source, small_config, new)


def test_prior_study_zip_freezes_all_seeds_before_test(tmp_path, small_config, monkeypatch):
    outer = tmp_path / 'previous-study.zip'
    with zipfile.ZipFile(outer, 'w') as archive:
        for seed in (0, 1):
            source, cache = make_bundle(tmp_path, seed)
            cache('test')
            nested = tmp_path / f'seed{seed}.zip'
            with zipfile.ZipFile(nested, 'w') as z:
                for path in source.rglob('*'):
                    if path.is_file():
                        z.write(path, str(path.relative_to(source)))
            archive.write(nested, f'source_archives/h2-standard-seed{seed}.zip')
    before = outer.read_bytes()
    stages = []
    select, evaluate = suite.select, suite.evaluate
    def freeze(*args):
        stages.append('select')
        return select(*args)
    def score(*args):
        stages.append('test')
        return evaluate(*args)
    monkeypatch.setattr(suite, 'select', freeze)
    monkeypatch.setattr(suite, 'evaluate', score)
    result = study.run_study(tmp_path / 'work', source=outer, seeds=[0, 1], config_path=small_config)
    assert stages == ['select', 'select'] and result['summary'] is None
    stages.clear()
    result = study.run_study(tmp_path / 'work', source=outer, seeds=[0, 1], config_path=small_config, evaluate_test=True)
    assert stages == ['select', 'select', 'test', 'test']
    assert result['summary']['status'] == 'complete'
    assert result['summary']['results']['uniform']['wga']['sample_std'] == 0
    assert outer.read_bytes() == before and Path(result['archive']).is_file()
    with pytest.raises(ValueError, match='Missing frozen H2 bundles'):
        study.run_study(tmp_path / 'other', source=outer, seeds=[2], config_path=small_config)


@pytest.mark.parametrize('name,value', [('steps', True), ('hidden_units', 0), ('prediction_dropout', 1),
                                      ('weight_decays', [float('nan')]), ('heads', 14), ('loss', 'test_accuracy')])
def test_invalid_config(name, value):
    c = config()
    c[name] = value
    with pytest.raises(ValueError):
        neural.validate_config(c, 15)


def test_summary_rejects_different_test_sets(tmp_path):
    with pytest.raises(ValueError, match='identical test'):
        suite.summarize(tmp_path, {0: dict(test_identity='a'), 1: dict(test_identity='b')}, [0, 1], 'standard')


@pytest.mark.parametrize('hosted', [False, True])
def test_notebook_runs_and_reuses_saved_models(tmp_path, small_config, monkeypatch, hosted):
    import shutil
    import subprocess
    import sys
    nbformat = pytest.importorskip('nbformat')
    nbclient = pytest.importorskip('nbclient')
    source, cache = make_bundle(tmp_path)
    cache('test')
    repo = tmp_path / 'checkout'
    for folder in ('src', 'scripts', 'configs'):
        shutil.copytree(ROOT / folder, repo / folder)
    notebook_path = repo / 'notebooks/dpe_neural_combiners.ipynb'
    notebook_path.parent.mkdir()
    notebook = nbformat.read(ROOT / 'notebooks/dpe_neural_combiners.ipynb', as_version=4)
    work = tmp_path / 'work'
    for cell in notebook.cells:
        if cell.cell_type == 'code' and 'SOURCE = None' in cell.source:
            cell.source = cell.source.replace('SOURCE = None', f'SOURCE = {str(source)!r}')
            cell.source = cell.source.replace('SEEDS = [0, 1, 2]', 'SEEDS = [0]')
            cell.source = cell.source.replace('EVALUATE_TEST = False', 'EVALUATE_TEST = True')
            cell.source = cell.source.replace('WORK_DIR = None', f'WORK_DIR = {str(work)!r}')
            cell.source = cell.source.replace('CONFIG_PATH = "configs/dpe_head_reweighting/neural_combiners.yaml"',
                                              f'CONFIG_PATH = {str(small_config)!r}')
            if hosted:
                cell.source = cell.source.replace('REPO_URL = "https://github.com/chautanphat/waterbirds-shortcut-mitigation.git"',
                                                  f'REPO_URL = {str(repo)!r}')
                cell.source = cell.source.replace('REPO_REF = "experiment/dpe-neural-combiner"', 'REPO_REF = "fixture"')
            else:
                cell.source = cell.source.replace('REPO_PATH = None', f'REPO_PATH = {str(repo)!r}')
    nbformat.write(notebook, notebook_path)
    if hosted:
        subprocess.run(['git', 'init', '-q', '-b', 'fixture', str(repo)], check=True)
        subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Notebook Test', '-c',
                        'user.email=notebook-test@example.invalid', 'commit', '-qm', 'Synthetic fixture'], check=True)
    kernels = tmp_path / 'jupyter/kernels/neural-test'
    kernels.mkdir(parents=True)
    (kernels / 'kernel.json').write_text(json.dumps(dict(
        argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        display_name='Neural combiner test', language='python',
        env=dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', IPYTHONDIR=str(tmp_path / 'ipython'),
                 JUPYTER_RUNTIME_DIR=str(tmp_path / 'jupyter-runtime')))))
    monkeypatch.setenv('JUPYTER_PATH', str(tmp_path / 'jupyter'))
    working = tmp_path / 'empty-session'
    working.mkdir()
    active_notebook = working / 'run.ipynb' if hosted else notebook_path
    nbformat.write(notebook, active_notebook)
    frozen = None
    for _ in range(2):
        actual = nbformat.read(active_notebook, as_version=4)
        nbclient.NotebookClient(actual, timeout=120, kernel_name='neural-test',
                                resources={'metadata': {'path': str(working)}},
                                on_cell_executed=lambda **kwargs: nbformat.write(actual, active_notebook)).execute()
        out, = [p for p in (work / 'studies').iterdir() if p.is_dir()]
        model = (out / 'comparisons/seed0/selection.json').read_bytes()
        if frozen is not None:
            assert model == frozen
        frozen = model
        assert json.loads((out / 'summary.json').read_text())['status'] == 'complete'
        assert (out / 'report.md').is_file() and out.with_suffix('.zip').is_file()
