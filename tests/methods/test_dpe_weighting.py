"""Safety, math, resume and full two-phase workflow checks on synthetic fixtures."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.weighting.cache import validate, save_predictions, load_predictions
from experiments.weighting.data import assert_disjoint, partition, fit_heads, predict_heads, prepare_meta
from experiments.weighting.gates import Gate, features
from experiments.weighting.losses import signed_scores, objective
from experiments.weighting.methods import IDS, Predictor, candidates, normalize, oracle_coverage
from experiments.weighting.run import Study, ablations


@pytest.fixture
def data():
    rng=np.random.default_rng(42)
    p=rng.dirichlet([2.,2.],size=(90,15)).astype('float32')
    return dict(probs=p,y=np.tile([0,1],45),sample_id=np.array([f'id-{i}' for i in range(90)]),
                distances=rng.random((90,15,2)).astype('float32'),embedding=rng.normal(size=(90,16)).astype('float32'))


def divided(data):
    return ({k:v[:60] for k,v in data.items()},{k:v[60:] for k,v in data.items()})


def test_split_leakage_and_determinism(data):
    a=partition(data['y'],data['sample_id'],0)
    b=partition(data['y'],data['sample_id'],0)
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    assert sorted(np.concatenate(a))==list(range(90))
    assert_disjoint(*(data['sample_id'][i] for i in a))
    with pytest.raises(ValueError,match='overlapping'):assert_disjoint(['a'],['a'])
    with pytest.raises(ValueError,match='Duplicate'):assert_disjoint(['a','a'])


def test_cache_dimension_normalization_and_identity(data,tmp_path):
    path=tmp_path/'p.npz'
    prov={'checkpoint':'sha256','fit_ids':['independent']}
    save_predictions(path,probs=data['probs'],sample_id=data['sample_id'],y=data['y'],split='oof',provenance=prov)
    p=load_predictions(path,prov)['probs'];np.testing.assert_array_equal(p,data['probs'])
    with pytest.raises(ValueError,match='provenance'):load_predictions(path,{'checkpoint':'other'})
    with pytest.raises(ValueError):validate(data['probs'].transpose(0,2,1))
    bad=data['probs'].copy();bad[0,0,0]=np.nan
    with pytest.raises(ValueError):validate(bad)


def test_uniform_and_fallback(data):
    a,b=divided(data);model=Predictor('U0').fit(a,b,0)
    np.testing.assert_allclose(model.predict(b),b['probs'].mean(1),atol=1e-7)
    np.testing.assert_allclose(normalize(np.zeros((3,15))),1/15)
    np.testing.assert_allclose(normalize(np.array([[1.,-1.,3.]])),[[.25,0.,.75]])


def test_teacher_polarity_and_oracle():
    p=torch.tensor([[[.99,.01],[.01,.99]]])
    scores=signed_scores(p,torch.tensor([0]))
    assert scores[0,0]>.9 and scores[0,1]<-.9
    assert oracle_coverage(p.numpy(),[0])==1
    with pytest.raises(ValueError,match='prohibited'):Predictor('X0')
    multiclass=torch.tensor([[[.1,.8,.1],[.8,.1,.1]]])
    np.testing.assert_allclose(signed_scores(multiclass,torch.tensor([1])).numpy(),[[.7,-.7]],atol=1e-6)


def test_uniform_initialization_and_gradient(data):
    f,_=features(data);gate=Gate(15,f.shape[-1]);p=torch.tensor(data['probs'])
    w=gate(torch.tensor(f));np.testing.assert_allclose(w.detach().numpy(),1/15,atol=1e-7)
    loss=objective((p*w[:,:,None]).sum(1),w,p,torch.tensor(data['y']),kl=.01,eta=.1,tail=.2)
    loss.backward()
    assert gate.net[-1].weight.grad.abs().sum()>0
    assert all(torch.isfinite(v.grad).all() for v in gate.parameters() if v.grad is not None)


def test_inference_excludes_labels_groups(data):
    a,b=divided(data);model=Predictor('D2').fit(a,b,0,epochs=4)
    before=model.predict(b)
    tampered=copy.deepcopy(b);tampered['y']=1-b['y'];tampered['group']=np.arange(len(b['y']))
    np.testing.assert_array_equal(before,model.predict(tampered))
    a['group']=np.zeros(60)
    with pytest.raises(ValueError,match='Group'):Predictor('D2').fit(a,b,0)
    a.pop('group');a['sample_id'][0]=b['sample_id'][0]
    with pytest.raises(ValueError,match='overlapping'):Predictor('S1').fit(a,b,0)


@pytest.mark.parametrize('method',[m for m in IDS if m!='X0'])
def test_all_families_executable(method,data):
    a,b=divided(data)
    model=Predictor(method,candidates(method)[0]).fit(a,b,0,epochs=3,patience=2)
    p=model.predict(b)
    assert p.shape==(30,2) and np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(1),1,atol=2e-6)
    w=model.weights(b)
    if w is not None:
        assert (w>=0).all();np.testing.assert_allclose(w.sum(1),1,atol=2e-6)


def test_gate_resume_and_determinism(data,tmp_path):
    a,b=divided(data);path=tmp_path/'latest.pt'
    first=Predictor('D2').fit(a,b,1,epochs=4,patience=20,progress_path=path)
    resumed=Predictor('D2').fit(a,b,1,epochs=8,patience=20,progress_path=path)
    direct=Predictor('D2').fit(a,b,1,epochs=8,patience=20)
    np.testing.assert_array_equal(resumed.predict(b),direct.predict(b))
    same=Predictor('D2').fit(a,b,1,epochs=4,patience=20)
    np.testing.assert_array_equal(first.predict(b),same.predict(b))


def test_head_crossfit_checkpoint_scope(data,tmp_path):
    c=dict(output_dir=str(tmp_path),device='cpu',protocol='crossfit',folds=3,
           head_training=dict(members=3,epochs=2,batch_size=32,lr=.001,cov_reg=1.,wd_weight=1.))
    a,b,indices=prepare_meta(data,c,0)
    assert 'group' not in a and 'group' not in b
    audit=json.loads((tmp_path/'seed0/split_audit.json').read_text())
    held=[]
    for fold in audit['folds']:
        assert_disjoint(fold['fit_ids'],fold['held_ids']);held+=fold['held_ids']
    assert sorted(held)==sorted(data['sample_id'])
    a2,b2,_=prepare_meta(data,c,0)
    np.testing.assert_array_equal(a['probs'],a2['probs'])
    path=tmp_path/'seed0/heads_fold0.pt';stamp=path.stat().st_mtime_ns
    prepare_meta(data,c,0);assert stamp==path.stat().st_mtime_ns


def test_ablation_coverage():
    assert {s['ablation'] for s in ablations('D2',{'kl':.01})}=={f'A{i}' for i in range(21)}
    assert next(s for s in ablations('H1',{}) if s['ablation']=='A13')['status']=='N/A'


def test_full_automatic_two_phase_resume(tmp_path,monkeypatch,data):
    import experiments.weighting.run as run
    # Synthetic fixture exercises orchestration only; never written to benchmark outputs.
    backbone=tmp_path/'backbone.pt';torch.save({},backbone)
    original=tmp_path/'original.pt'
    torch.save(dict(prototypes=torch.randn(15,2,16),distance_scales=torch.ones(15)),original)
    c=dict(output_dir=str(tmp_path/'study'),backbone=str(backbone),original_heads=str(original),dataset_dir='synthetic_only',
           device='cpu',protocol='crossfit',folds=3,seeds=[0,1,2],screen_seeds=[0,1],epochs=3,patience=2,
           max_wall_seconds=500,bootstrap_repeats=20,method_whitelist=['U0','S1'],
           head_training=dict(members=15,epochs=1,batch_size=32,lr=.001,cov_reg=1.,wd_weight=1.))
    def fake_extract(cfg,split):
        d={k:v.copy() for k,v in data.items() if k in ('embedding','y','sample_id')}
        d['group']=np.tile([0,1,2,3,0,1],15)
        if split=='test':d['sample_id']=np.array(['test-'+s for s in d['sample_id']])
        root=Path(cfg['output_dir'])/'features';root.mkdir(exist_ok=True)
        if not (root/f'{split}.npz').exists():np.savez_compressed(root/f'{split}.npz',**d)
        return d
    monkeypatch.setattr(run,'extract',fake_extract)
    monkeypatch.setattr(Study,'export',lambda *a,**kw:None)
    study=Study(c,True);study.run('all')
    root=Path(c['output_dir'])
    assert json.loads((root/'progress.json').read_text())['stage']=='complete'
    assert (root/'REPORT.md').exists() and (root/'phase2_ablation.csv').exists()
    assert (root/'selection_lock.json').exists() and (root/'gate_best.pt').exists()
    stamps={p:p.stat().st_mtime_ns for p in (root/'fits').rglob('model.joblib')}
    Study(c,True).run('all')
    assert stamps=={p:p.stat().st_mtime_ns for p in stamps}
