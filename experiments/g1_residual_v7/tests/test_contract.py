import copy
import inspect
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from experiments.geometry_aggregation_v5.io import load,single_writer
from experiments.geometry_aggregation_v5.persistence import atomic
from experiments.ccw_v6.model import infer as original
from experiments.g1_residual_v7.model import fit_base,advantage,attach_scales,infer,features,target,scaled,center,CONFIG,METHODS
from experiments.g1_residual_v7.store import Journal,jsonwrite
from experiments.g1_residual_v7.metrics import shared_draws,metric
from experiments.g1_residual_v7.run import B,C,outer_splits,secondary_splits,train_indices,nest


def toy(n=80):
    rng=np.random.default_rng(701);p=rng.uniform(.15,.85,(n,15,2));p/=p.sum(-1,keepdims=True);d=rng.uniform(.6,1.4,p.shape)
    return p,d,np.arange(n)%2,np.array([f'synthetic_{i}' for i in range(n)]),np.array([f'head_{j}' for j in range(15)])


def fitted():
    p,d,y,ids,h=toy();m=fit_base(p,d,y,ids,h);a=advantage(p,d,m,h);m=attach_scales(m,a['pooled'],a['head'],p,d,y)
    return p,d,y,ids,h,m


def test_lambda_zero_bit_exact_all_nine_real_C_folds():
    for s in range(3):
        for f in range(3):
            c=C(s,f);old=load(nest(s,f)/'G1_g50/predictions.npz');out=infer(c['probs'],c['distances'],None,c['head_sha256'],lambda_value=0)
            np.testing.assert_array_equal(out['probs'],old['probs']);np.testing.assert_array_equal(out['weights'],old['weights'])


def test_inference_rejects_labels_groups_ID_lookup():
    for fn in [infer,advantage,features]:assert not {'y','group','correctness','sample_id'}&set(inspect.signature(fn).parameters)
    p,d,y,ids,h,m=fitted()
    for kw in ['y','group','correctness','sample_id']:
        with pytest.raises(TypeError):infer(p,d,m,h,**{kw:y})


def test_train_only_normalizers_ridge_and_secondary_ownership():
    p,d,y,ids,h=toy();idx=np.arange(60);m=fit_base(p[idx],d[idx],y[idx],ids[idx],h)
    assert set(m['fit_ids']).isdisjoint(ids[60:]);np.testing.assert_array_equal(m['normalizer']['mu'],d[idx].min(-1).mean(0))
    np.testing.assert_array_equal(m['normalizer']['sigma'],d[idx].min(-1).std(0))
    assert m['q20']==np.quantile(abs(p[idx].mean(1)[:,1]-.5),.2)
    X,q,z,delta,u,pred=features(p[idx],d[idx],m['normalizer']);X=X/np.array(m['column_scale']);omega=len(idx)/(2*np.bincount(y[idx])[y[idx]])*(1+(abs(u[:,1]-.5)<=m['q20']));omega/=omega.mean();w=np.repeat(omega/15,15);xf=X.reshape(-1,6)
    expected=np.linalg.solve(xf.T@(w[:,None]*xf)+10*np.eye(6),xf.T@(w*target(p[idx],y[idx]).ravel()));np.testing.assert_array_equal(expected,m['theta'])
    b=B(0,0)
    for part in range(5):
        train=train_indices(b,0,0,part);allvalid=[]
        for a,v in secondary_splits(b,0,0,part):assert not set(a)&set(v) and set(a)|set(v)==set(train);allvalid.extend(v)
        assert sorted(allvalid)==sorted(train)


def test_full_B_provenance_and_head_hashes():
    for s in range(3):
        for f in range(3):
            b=B(s,f);c=C(s,f);assert not set(b['sample_id'])&set(c['sample_id']);np.testing.assert_array_equal(b['head_sha256'],c['head_sha256']);assert 'group' not in b


def test_head_mismatch_rejected():
    p,d,y,ids,h,m=fitted()
    with pytest.raises(ValueError):infer(p,d,m,h[::-1])


def test_delta_zero_target_zero_and_common_offset_cancellation():
    p,d,y,ids,h=toy();p[:]=.5;m=fit_base(p,d,y,ids,h);a=advantage(p,d,m,h)
    np.testing.assert_array_equal(a['pooled'],0);np.testing.assert_array_equal(a['head'],0);np.testing.assert_array_equal(target(p,y),0)
    x=np.tile(np.arange(-7,8)/128,(3,1));s=dict(enabled=True,s_R=.02,s_G=1.)
    np.testing.assert_array_equal(scaled(x,s),scaled(x+8,s));np.testing.assert_allclose(target(toy()[0],y).sum(1),0,atol=2e-6)


def test_scale_guard_nonfinite_and_degenerate_fallback():
    p,d,y,ids,h,m=fitted();m['scales']['head']=dict(enabled=False,s_R=.001,s_G=1.,off_reason='LOW')
    out=infer(p,d,m,h);base=original(p,d,None,h,'G1_frozen');np.testing.assert_array_equal(out['probs'],base['probs']);np.testing.assert_array_equal(out['weights'],base['weights'])
    np.testing.assert_array_equal(scaled(np.full((3,15),np.nan),dict(enabled=True,s_R=.01,s_G=1.)),0)
    z=attach_scales(copy.deepcopy(m),np.zeros((len(p),15)),np.zeros((len(p),15)),p,d,y);assert not z['scales']['head']['enabled']


def test_active_path_changes_weights_and_gate_is_separate():
    p,d,y,ids,h,m=fitted();m['scales']={k:dict(enabled=True,s_R=.01,s_G=1.,off_reason=None) for k in ['head','pooled']}
    base=infer(p,d,m,h,'G1_frozen');rh=infer(p,d,m,h);rhm=infer(p,d,m,h,'G1_RHM')
    assert abs(rh['weights']-base['weights']).max()>1e-8;assert abs(rhm['weights']-rh['weights']).max()>1e-8
    for out in [rh,rhm]:
        assert np.isfinite(out['scores']).all() and (out['weights']>=.05).all();np.testing.assert_allclose(out['r'].sum(1),1);np.testing.assert_allclose(out['weights'].sum(1),1)
    assert (rhm['gate']>0).all() and (rhm['gate']<=1).all()


def test_C_ID_alignment_and_group_coverage():
    for s in range(3):
        parts=[C(s,f,True) for f in range(3)];ids=np.concatenate([v['sample_id'] for v in parts]);g=np.concatenate([v['group'] for v in parts])
        assert len(ids)==len(set(ids))==1199;assert [int((g==k).sum()) for k in range(4)]==[467,466,133,133]


def test_shared_image_bootstrap_and_resume_RNG():
    strata=np.repeat(np.arange(4),[5,6,3,3]);rng=np.random.default_rng(730001);first=shared_draws(strata,rng,20);state=rng.bit_generator.state;second=shared_draws(strata,rng,20)
    resumed=np.random.default_rng();resumed.bit_generator.state=state;np.testing.assert_array_equal(second,shared_draws(strata,resumed,20))
    values=np.tile(np.arange(len(strata)),(5,3,1));means=values[:,:,first].mean(-1);np.testing.assert_array_equal(means[0,0],means[4,2])


def test_atomic_retry_single_writer_and_receipt_integrity(tmp_path):
    import experiments.geometry_aggregation_v5.persistence as persistence
    old=persistence.os.replace;attempts=[]
    def transient(a,b):
        attempts.append(1)
        if len(attempts)<3:raise PermissionError('simulated Windows transient lock')
        old(a,b)
    with patch.object(persistence.os,'replace',side_effect=transient):atomic(tmp_path/'atomic','valid')
    assert len(attempts)==3
    with single_writer(tmp_path):
        with pytest.raises(RuntimeError):
            with single_writer(tmp_path):pass
    m=dict(test='receipt');j=Journal(tmp_path/'journal',m)
    def work():
        p=tmp_path/'journal/data';atomic(p,'valid');return {},[p]
    j.task('first',[],work);stamp=(tmp_path/'journal/data').stat().st_mtime_ns;atomic(tmp_path/'journal/state.json','truncated')
    j=Journal(tmp_path/'journal',m);j.task('first',[],lambda:pytest.fail('recomputed'));assert (tmp_path/'journal/data').stat().st_mtime_ns==stamp
    atomic(tmp_path/'journal/data','corrupt')
    with pytest.raises(ValueError):Journal(tmp_path/'journal',m)


def test_margin_and_disagreement_are_not_labels():
    p,d,y,ids,h=toy();a=target(p,np.zeros(len(p),int));b=target(p,np.ones(len(p),int));np.testing.assert_array_equal(a,-b)
    # Identical label-free inputs can require opposite target directions.
    model=fit_base(p,d,y,ids,h);first=advantage(p,d,model,h);second=advantage(p.copy(),d.copy(),model,h)
    np.testing.assert_array_equal(first['head'],second['head'])


def test_strict_AA_floor_and_WGA_definition():
    assert not (-1.> -1.) and (-.99999> -1.)
    y=np.array([0,0,0,0,1,1,1,1]);g=np.repeat(np.arange(4),2);pred=y.copy();pred[4]=0;r=metric(np.eye(2)[pred],y,g)
    assert r['AA']==.875 and r['WGA']==.5 and r['WCA']==.75
