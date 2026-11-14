import inspect
import numpy as np
import pytest
from scipy.special import logsumexp,softmax
from experiments.geometry_aggregation_v5.model import aggregate,configs,fit_normalizer
from experiments.geometry_aggregation_v5.audit import pair_values
from experiments.geometry_aggregation_v5.journal import Journal
from experiments.discrete_weighting.store import atomic


def data(n=40):
    rng=np.random.default_rng(51);d=rng.uniform(.5,1.5,(n,15,2));scales=np.linspace(.8,1.2,15);l=-d*scales[None,:,None];return softmax(l,axis=-1),l,d


def test_fixed_budget_and_inference_no_labels():
    assert len(configs())==25
    assert not {'y','group','correct'}&set(inspect.signature(aggregate).parameters)
    p,l,d=data()
    for k in ['y','group','correct']:
        with pytest.raises(TypeError):aggregate(p,l,d,configs()[0],**{k:np.zeros(len(p))})


def test_all_probability_axes_and_normalization():
    p,l,d=data();fold=np.zeros(len(p),int);ident={'0':[str(j) for j in range(15)]};norm=fit_normalizer(d,fold,np.arange(len(p)).astype(str),ident)
    for cfg in configs():
        out=aggregate(p,l,d,cfg,fold,norm,ident);np.testing.assert_allclose(out['probs'].sum(1),1,atol=1e-6)
        if cfg['method'].startswith('S'):
            np.testing.assert_allclose(out['contributions'].sum(1),1,atol=1e-6);assert out['weights'].shape[1]==0
        else:np.testing.assert_allclose(out['weights'].sum(1),1);assert (out['weights']>=0).all()
    np.testing.assert_array_equal(aggregate(p,l,d,configs()[0])['probs'],p.mean(1))


def test_softmin_exact_axes_limits_actual_logit_control():
    p,l,d=data(2);cfg=dict(method='S1',alpha=1.,tau=.1,T_class=1.)
    out=aggregate(p,l,d,cfg);expected=softmax(.1*(logsumexp(-d/.1,axis=1)-np.log(15)),axis=1)
    np.testing.assert_allclose(out['probs'],expected)
    tiny=aggregate(p,l,d,dict(cfg,tau=1e-7));np.testing.assert_allclose(tiny['probs'],softmax(-d.min(1),axis=1),atol=1e-6)
    large=aggregate(p,l,d,dict(cfg,tau=1e6));np.testing.assert_allclose(large['probs'],softmax(-d.mean(1),axis=1),atol=1e-6)
    logit=aggregate(p,-d,d,dict(cfg,method='S2'));np.testing.assert_allclose(logit['probs'],out['probs'])
    actual=aggregate(p,l,d,dict(cfg,method='S2'));assert np.max(abs(actual['probs']-out['probs']))>1e-5
    assert np.max(abs(out['contributions'][...,0]-out['contributions'][...,1]))>1e-3


def test_centered_distance_offset_invariance_and_residual():
    p,l,d=data();cfg=dict(method='S3',alpha=1.,tau=.1,T_class=1.)
    offset=np.arange(15)[None,:,None]/3
    np.testing.assert_allclose(aggregate(p,l,d,cfg)['probs'],aggregate(p,l,d+offset,cfg)['probs'],atol=1e-12)
    a=.25;cfg=dict(method='S4',alpha=a,tau=.1,T_class=1.)
    base=aggregate(p,l,d,dict(cfg,method='S1'))['probs'];mix=aggregate(p,l,d,cfg)['probs'];np.testing.assert_allclose(mix,(1-a)*p.mean(1)+a*base)


def test_normalizer_actual_head_identity_and_train_only():
    p,l,d=data(80);fold=np.repeat([0,1],40);d[fold==1]+=10;ident={str(f):[f'{f}:{j}' for j in range(15)] for f in [0,1]};ids=np.arange(80).astype(str)
    norm=fit_normalizer(d,fold,ids,ident)
    np.testing.assert_allclose(norm['ensembles']['0']['mean'],d[:40].min(-1).mean(0))
    assert set(norm['ensembles']['0']['fit_ids']).isdisjoint(norm['ensembles']['1']['fit_ids'])
    with pytest.raises(ValueError):aggregate(p,l,d,dict(method='G2',alpha=.25,gamma=1),fold,norm,{'0':ident['1'],'1':ident['0']})


def test_top_m_deterministic_ties_and_pair_ranking():
    p,l,d=data(2);d[:]=1;out=aggregate(p,l,d,dict(method='G3',alpha=1.,M=3));np.testing.assert_allclose(out['weights'][:,:3],1/3);assert not out['weights'][:,3:].any()
    correct=np.array([[True,False,False],[True,True,False],[False,False,False]])
    score=np.array([[1.,1.,1.],[3.,1.,2.],[1.,2.,3.]])
    v=pair_values(score,correct);np.testing.assert_allclose(v[:2],[.5,.5]);assert np.isnan(v[2])


def test_journal_hash_reuse_and_source_guard(tmp_path):
    m={'config':'fixed'};j=Journal(tmp_path,m)
    def fit():
        p=tmp_path/'artifact';atomic(p,'saved');return {},[p]
    j.task('one',[],fit);stamp=(tmp_path/'artifact').stat().st_mtime_ns
    j=Journal(tmp_path,m);j.task('one',[],lambda:pytest.fail('must reuse'));assert (tmp_path/'artifact').stat().st_mtime_ns==stamp
    with pytest.raises(ValueError):Journal(tmp_path,{'config':'changed'})
    atomic(tmp_path/'artifact','corrupt')
    with pytest.raises(ValueError):Journal(tmp_path,m)


def test_rename_retry_keeps_atomic_bytes(tmp_path,monkeypatch):
    from experiments.geometry_aggregation_v5 import persistence
    replace=persistence.os.replace;attempts=[]
    def locked(source,target):
        attempts.append(1)
        if len(attempts)<=2:raise PermissionError('simulated OneDrive transient lock')
        return replace(source,target)
    monkeypatch.setattr(persistence.os,'replace',locked);monkeypatch.setattr(persistence.time,'sleep',lambda _:None)
    path=tmp_path/'retry.bin';persistence.atomic(path,b'valid bytes');assert path.read_bytes()==b'valid bytes' and len(attempts)==3


def test_epoch_adapter_math_matches_historical_source():
    import ast
    from pathlib import Path
    a=ast.parse(Path('experiments/wga_verification/nested_heads.py').read_text());b=ast.parse(Path('experiments/geometry_aggregation_v5/heads.py').read_text())
    functions=lambda tree:{node.name:ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,ast.FunctionDef)}
    assert functions(a)==functions(b)
