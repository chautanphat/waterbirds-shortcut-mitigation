"""Saved pretest heads/features must survive a weighting-only source change."""
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from src.methods.dpe_head_reweighting import ensemble, experiment as ex, recovery, study, rule_suite
from src.methods.dpe_head_reweighting.notebook_setup import find_sources

spec = importlib.util.spec_from_file_location('recovery_fixtures',Path(__file__).with_name('test_weighting_rules.py'))
fixtures = importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
CONFIG = Path(__file__).resolve().parents[3] / 'configs/dpe_head_reweighting/weighting_rules.yaml'


def failed_output(tmp_path):
    old = tmp_path / 'failed-version'
    source, _, _ = fixtures.make_bundle(old / 'h2_sources')
    feature = old / 'features/seed1'
    feature.mkdir(parents=True)
    m = ex.read_json(source / 'experiment/manifest.json')
    m['provenance'] = dict(track='standard',seed=1)
    for entries in [m['validation'],*m['parts'].values()]:
        for entry in entries:
            entry['split']=1
    rows = [dict(filename=r['id'],y=r['y'],a=int(r['group'][1]),split=1) for r in m['validation']]
    p,y,g=fixtures.arrays()
    test_rows=[dict(filename=f'test-{i}',y=int(label),a=int(group[1]),split=2) for i,(label,group) in enumerate(zip(y,g))]
    with (feature/'metadata.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows+test_rows)
    random=np.random.default_rng(12)
    for role, items in [('val',rows),('test',test_rows)]:
        np.savez(feature/f'{role}_raw.npz',ids=[r['filename'] for r in items],features=random.normal(size=(len(items),2048)).astype('float32'))
    ex.write_json(feature/'provenance.json',m['provenance'])
    m.update(backbone_hash='frozen-completed-backbone',metadata_hash=ex.digest(feature/'metadata.csv'),val_features_hash=ex.digest(feature/'val_raw.npz'))
    ex.write_json(source/'experiment/manifest.json',m)
    a=dict(version=1,prototypes=torch.randn(15,2,2048),distance_scales=torch.ones(15),temperatures=torch.ones(15),
           head_ids=[f'head-{i:02d}' for i in range(15)],selected_epochs=[20]*15,classes=[0,1],groups=['00','01','10','11'],
           preprocessing=ensemble.PREPROCESSING,provenance=dict(manifest=ex.digest(source/'experiment/manifest.json')))
    ensemble.save(source/'checkpoints/ensemble.pt',a)
    s=ex.read_json(source/'experiment/selection.json')
    s['lock'].update(ensemble=ex.digest(source/'checkpoints/ensemble.pt'),manifest=ex.digest(source/'experiment/manifest.json'))
    ex.write_json(source/'experiment/selection.json',s)
    for role in ('B','C'):
        file=source/f'experiment/predictions/{role}.npz'
        with np.load(file) as f:data=dict(f)
        data['ensemble_hash']=s['lock']['ensemble'];np.savez(file,**data)
    ex.write_json(feature/'handoff.json',dict(identity=dict(checkpoint_sha256=m['backbone_hash']),
        artifacts={name:ex.digest(feature/name) for name in ('metadata.csv','val_raw.npz','test_raw.npz','provenance.json')}))
    return old,source,feature,a


def test_recovery_uses_frozen_heads_and_never_trains(tmp_path,monkeypatch):
    old,source,feature,a=failed_output(tmp_path)
    assert find_sources([old]) == []  # No test cache exists yet.
    assert find_sources([old],require_test=False) == [source]
    before={str(p.relative_to(old)):p.read_bytes() for p in old.rglob('*') if p.is_file()}
    monkeypatch.setattr(study,'ensure_dataset',lambda *a,**kw:pytest.fail('Must not download/retrain'))
    work=tmp_path/'new-working-session'
    result=study.run_study(work,seeds=[1],search_roots=[old],device='cpu')
    assert result['summary']['status']=='complete'
    recovered,=list((work/'recovered_sources').iterdir())
    assert ex.digest(recovered/'checkpoints/ensemble.pt') == ex.digest(source/'checkpoints/ensemble.pt')
    assert (recovered/'test_cache_recovery.json').is_file()
    with np.load(feature/'test_raw.npz') as f:expected=ensemble.predict(a,f['features'])
    with np.load(recovered/'experiment/predictions/test.npz') as f:np.testing.assert_array_equal(f['probabilities'],expected)
    for name,content in before.items():assert (old/name).read_bytes()==content
    # Second run reuses source data and its deterministic test cache, even if
    # every training/feature/prediction function is unavailable.
    monkeypatch.setattr(ensemble,'predict',lambda *a,**kw:pytest.fail('Test inference should be cached'))
    assert study.run_study(work,seeds=[1],search_roots=[old],device='cpu')['summary']==result['summary']


def test_recovery_rejects_changed_feature_artifacts(tmp_path):
    old,source,feature,_=failed_output(tmp_path)
    (feature/'test_raw.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='Recovery feature artifact changed'):
        recovery.recover_test_cache(source,[old],tmp_path/'new')
