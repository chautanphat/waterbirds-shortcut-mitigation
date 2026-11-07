import copy
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.methods.dpe_head_reweighting import ensemble as e
from src.methods.dpe_head_reweighting import experiment as ex
from src.methods.dpe_head_reweighting.reweighting import combine, fit_select, metrics, weights
from src.methods.dpe_head_reweighting.train_heads import train

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_settings():
    return dict(heads=3, epochs=2, batch_size=8, lr=.001, momentum=.9,
                entropic_scale=30., cov_reg=5., wd_weight=10.)


def grouped(n=10):
    g = np.repeat(['00', '01', '10', '11'], n)
    y = np.array([int(k[0]) for k in g])
    x = np.random.RandomState(5).normal(size=(len(g), 8)).astype('float32')
    return x, y, g


def test_reference_parity_and_roundtrip(tmp_path):
    spec = importlib.util.spec_from_file_location('upstream_isomax', ROOT / 'baselines/dpe4subpop/utils/isomaxplus.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    raw, y, g = grouped()
    a = train(raw, y, g, small_settings(), 0, {'fixture': 'synthetic'})
    a['distance_scales'] = torch.tensor([-.7, 2.3, 1.1])
    logits = []
    for p, s in zip(a['prototypes'], a['distance_scales']):
        h = ref.IsoMaxPlusLossFirstPart(8, 2)
        with torch.no_grad():
            h.prototypes.copy_(p)
            h.distance_scale.copy_(s.reshape(1))
            logits.append(h(e.standardize(raw[:9])))
    expected = torch.stack(logits)
    np.testing.assert_allclose(e.predict(a, raw[:9], batch_size=4, logits=True), expected.numpy(), atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(e.predict(a, raw[:9], batch_size=4).mean(0), expected.softmax(-1).mean(0), atol=1e-6, rtol=1e-5)
    e.save(tmp_path / 'a.pt', a)
    loaded = e.load(tmp_path / 'a.pt', 8, 3)
    np.testing.assert_array_equal(e.predict(loaded, raw[:9]), e.predict(a, raw[:9]))
    with pytest.raises(ValueError):
        e.load(tmp_path / 'a.pt', 2048, 3)
    for key, value in [('classes', [1, 0]), ('preprocessing', 'wrong'), ('distance_scales', torch.tensor([float('nan')]*3))]:
        b = dict(a, **{key: value})
        with pytest.raises(ValueError):
            e.validate(b)


def test_weights_metrics_and_selection():
    w, fallback = weights([.25, .75], .5, 2)
    np.testing.assert_allclose(w, [.3, .7])
    assert not fallback
    w, fallback = weights([0, 0], 1, 4)
    np.testing.assert_array_equal(w, [.5, .5])
    assert fallback
    y = np.array([0, 0, 1, 1])
    g = np.array(['00', '01', '10', '11'])
    perfect = np.eye(2)[y]
    bad = perfect[:, ::-1]
    p = np.stack([perfect, bad])
    m = metrics(combine(p, [.5, .5]), y, g)
    assert m['overall'] == .5 and m['wga'] == 0
    result = fit_select(p, y, g, p, y, g)
    assert len(result['candidates']) == 13
    assert result['selected']['alpha'] == .25 and result['selected']['gamma'] == 1
    # Uniform wins when all candidate predictions tie.
    same = np.stack([perfect, perfect])
    assert fit_select(same, y, g, same, y, g)['selected']['alpha'] == 0
    for invalid in ([float('nan')], [-.2], [1.1], []):
        with pytest.raises(ValueError):
            weights(invalid, .5, 1)
    for invalid in ([1., 1.], [-.1, 1.1], [float('nan'), 0]):
        with pytest.raises(ValueError):
            combine(p, invalid)
    with pytest.raises(ValueError):
        metrics(perfect[:3], y[:3], g[:3])
    with pytest.raises(ValueError):
        fit_select(p * float('nan'), y, g, p, y, g)


def test_partition_isolation_and_feature_ids(tmp_path):
    rows = [dict(id=f'{g}-{i:03}', y=int(g[0]), group=g, split=1)
            for g in ('00', '01', '10', '11') for i in range(17)]
    parts = ex.partition(rows)
    assert parts == ex.partition(rows)
    ids = [{r['id'] for r in parts[k]} for k in 'ABC']
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set.union(*ids) == {r['id'] for r in rows}
    assert all({r['group'] for r in part} == set(('00', '01', '10', '11')) for part in parts.values())
    x = np.random.RandomState(0).normal(size=(len(rows), 8))
    path = tmp_path / 'features.npz'
    for bad_ids in ([r['id'] for r in rows][::-1], [rows[0]['id']] * len(rows), [r['id'] for r in rows[:-1]]):
        np.savez(path, ids=bad_ids, features=x)
        with pytest.raises(ValueError):
            ex.feature_file(path, rows, 8)
    for bad_x in (np.ones((2, 8)), np.full((2, 8), np.nan)):
        with pytest.raises(ValueError):
            e.standardize(bad_x)
    with pytest.raises(ValueError, match='zero optimizer'):
        train(*grouped(1), small_settings(), 0, {'fixture': 'synthetic'})


def test_config_schema():
    c = ex.config(ROOT / 'configs/dpe_head_reweighting/waterbirds_standard.yaml', 2)
    assert c['seed'] == 2 and 'seed2' in c['val_features']
    with pytest.raises(ValueError):
        ex.check_keys(dict(c, typo=1), ex.SCHEMA)
    with pytest.raises(ValueError):
        ex.check_keys(dict(c, seed=True), ex.SCHEMA)


def test_official_metadata_and_track_summary(tmp_path, monkeypatch):
    path = tmp_path / 'metadata.csv'
    path.write_text('img_filename,y,place,split\nbird.jpg,1,0,1\n')
    assert ex.metadata(path) == [dict(id='bird.jpg', y=1, group='10', split=1)]
    records = []
    for seed in range(3):
        records.append(dict(track='standard', seed=seed, metadata_hash='same', lock={'seed': seed},
                            uniform=dict(overall=.9, wga=.7), selected=dict(overall=.9, wga=.7 + seed * .01),
                            delta_pp=dict(overall=0, wga=seed)))
    monkeypatch.setattr(ex, 'config', lambda path, seed: dict(track='standard', seed=seed, results_dir=str(seed)))
    monkeypatch.setattr(ex, 'read_json', lambda path: records[int(path.parent.name)])
    monkeypatch.setattr(ex, 'lock', lambda c: {'seed': c['seed']})
    saved = []
    monkeypatch.setattr(ex, 'write_json', lambda path, value: saved.append(value))
    summary = ex.summarize('unused')
    assert summary['metrics']['delta_pp_wga'] == dict(mean=1., sample_std=1., values=[0, 1, 2])
    records[2]['track'] = 'enhanced'
    with pytest.raises(ValueError, match='mismatched'):
        ex.summarize('unused')


def test_synthetic_lifecycle_without_test_files_during_selection(tmp_path, monkeypatch):
    # Small heads/features for speed; CLI configs enforce 15 heads and dim=2048.
    c = ex.config(ROOT / 'configs/dpe_head_reweighting/waterbirds_standard.yaml')
    c['dim'], c['training'] = 8, small_settings()
    for key in ('metadata', 'val_features', 'test_features', 'backbone', 'provenance',
                'experiment_dir', 'ensemble_path', 'results_dir'):
        c[key] = str(tmp_path / key)
    Path(c['backbone']).write_bytes(b'synthetic-backbone-fixture')
    rows = []
    for split, count in enumerate([4795, 1199, 5794]):
        for i in range(count):
            g = ('00', '01', '10', '11')[i % 4]
            rows.append(dict(filename=f'{split}/{i}.jpg', y=int(g[0]), place=int(g[1]), split=split))
    with open(c['metadata'], 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    val = [r for r in rows if r['split'] == 1]
    rng = np.random.RandomState(3)
    with open(c['val_features'], 'wb') as f:
        np.savez(f, ids=[r['filename'] for r in val], features=rng.normal(size=(1199, 8)).astype('float32'))
    provenance = dict(track='standard', seed=0, checkpoint_sha256=ex.digest(c['backbone']),
                      val_features_sha256=ex.digest(c['val_features']), model='resnet50',
                      pretrained_weights='synthetic', transforms='synthetic', extraction_order='metadata_split_order',
                      feature_format='raw', checkpoint_selection='final_epoch_no_validation_selection', source_commit='synthetic')
    ex.write_json(c['provenance'], provenance)
    assert ex.prepare(c)['status'] == 'prepared'
    # Spy verifies both fitting and epoch selection see exactly A.
    original_train = ex.train
    def spy(x, y, g, *args):
        manifest = ex.read_json(Path(c['experiment_dir']) / 'manifest.json')
        with np.load(c['val_features']) as f:
            lookup = {key: i for i, key in enumerate(f['ids'])}
            expected = f['features'][[lookup[r['id']] for r in manifest['parts']['A']]]
        np.testing.assert_array_equal(x, expected)
        np.testing.assert_array_equal(y, [r['y'] for r in manifest['parts']['A']])
        return original_train(x, y, g, *args)
    monkeypatch.setattr(ex, 'train', spy)
    ex.train_action(c)
    # Selection works with no metadata or test features available.
    saved_metadata = Path(c['metadata']).read_bytes()
    Path(c['metadata']).unlink()
    selected = ex.select_action(c)
    assert len(selected['weights']) == 3
    frozen = (Path(c['experiment_dir']) / 'selection.json').read_bytes()
    Path(c['metadata']).write_bytes(saved_metadata)
    test = [r for r in rows if r['split'] == 2]
    with open(c['test_features'], 'wb') as f:
        np.savez(f, ids=[r['filename'] for r in test], features=rng.normal(size=(5794, 8)).astype('float32'))
    # Output caches must stay local to the test.
    monkeypatch.setattr(ex, 'cache', lambda *args: None)
    result = ex.evaluate_action(c)
    assert (Path(c['experiment_dir']) / 'selection.json').read_bytes() == frozen
    assert result['track'] == 'standard' and sum(result['uniform']['counts'].values()) == 5794
    assert (Path(c['results_dir']) / 'head_diagnostics.png').exists()
    with pytest.raises(ValueError, match='already exists'):
        ex.evaluate_action(c)
    # Mutating any locked input invalidates evaluation before reading test data.
    for key in ('ensemble_path', 'val_features'):
        path = Path(c[key])
        before = path.read_bytes()
        path.write_bytes(before + b'changed')
        with pytest.raises(ValueError, match='selection inputs changed'):
            ex.evaluate_action(c)
        path.write_bytes(before)
    c2 = copy.deepcopy(c)
    c2['seed'] = 1
    with pytest.raises(ValueError, match='selection inputs changed'):
        ex.evaluate_action(c2)
    path = Path(c['experiment_dir']) / 'manifest.json'
    m = ex.read_json(path)
    m['parts']['B'] = m['parts']['A']
    ex.write_json(path, m)
    with pytest.raises(ValueError, match='selection inputs changed'):
        ex.evaluate_action(c)
