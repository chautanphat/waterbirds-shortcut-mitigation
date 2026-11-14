import inspect
import json
import numpy as np
import pytest
from scipy.special import softmax
from experiments.score_audit_v4.model import (MAPS,fit_knots,magnitude,teacher,observable,fit_lookup,infer,fit_representation,representation)
from experiments.score_audit_v4.journal import Journal
from experiments.discrete_weighting.store import atomic


def toy():
    rng=np.random.default_rng(41);logits=rng.normal(0,.08,(50,15,2));p=softmax(logits,axis=-1);y=rng.integers(2,size=50)
    return logits,p,y


def test_teacher_knots_sign_clipping_continuity():
    for mid in ['R1','R2','R3','R4','R6']:
        for c,h in MAPS[mid]:
            p=np.tile([c,1-c],(3,15,1));knots=fit_knots(mid,p)
            np.testing.assert_allclose(magnitude(p,knots),h)
            np.testing.assert_allclose(teacher(p,np.zeros(3,int),knots),h)
            np.testing.assert_allclose(teacher(p,np.ones(3,int),knots),-h)
            if .5<c<1:
                pp=np.tile([c+1e-8,1-c-1e-8],(3,15,1))
                np.testing.assert_allclose(magnitude(pp,knots),h,atol=1e-6)
        p=np.tile([1.,0.],(3,15,1));np.testing.assert_allclose(magnitude(p,fit_knots(mid,p)),1)


def test_agreement_self_exclusion():
    p=np.tile([.6,.4],(1,15,1));p[0,0]=[.4,.6]
    pred,conf,loo=observable(p);assert loo[0,0]==0;assert loo[0,1]==13/14
    p[:]=[.6,.4];assert np.all(observable(p)[2]==1)


def test_inference_api_rejects_labels_and_groups():
    assert set(inspect.signature(infer).parameters)=={'p','lookup','alpha','beta'}
    _,p,y=toy();lk=fit_lookup(p,y,np.arange(len(y)).astype(str),fit_knots('R1',p))
    for name in ['y','correct','group']:
        with pytest.raises(TypeError):infer(p,lk,**{name:y})
    for a in [0,.1,.25]:
        mix,w,s=infer(p,lk,a,2);assert (w>0).all();np.testing.assert_allclose(w.sum(1),1);np.testing.assert_allclose(mix.sum(1),1)
        if a==0:np.testing.assert_allclose(mix,p.mean(1),atol=1e-15)


def test_temperature_original_logits_and_training_only():
    logits,p,y=toy();r=fit_representation('R3',logits,y)
    np.testing.assert_allclose(representation(r,p,logits),softmax(logits/.25,axis=-1))
    r=fit_representation('R4',logits,y);assert .001<=r['temperature']<=10
    assert (p.max(-1)>=.75).sum()==0
    assert set(inspect.signature(fit_representation).parameters)=={'map_id','logits','y'}
    knots=fit_knots('R5',np.tile([.5,.5],(5,15,1)));assert len(knots['fallback'])==2


def test_atomic_crc_source_guard_and_reuse(tmp_path):
    manifest={'source':'same'};j=Journal(tmp_path,manifest)
    def work():
        p=tmp_path/'done.txt';atomic(p,'complete');return {},[p]
    j.task('one',[],work);stamp=(tmp_path/'done.txt').stat().st_mtime_ns
    j=Journal(tmp_path,manifest);j.task('one',[],lambda:(_ for _ in ()).throw(RuntimeError('must reuse')))
    assert (tmp_path/'done.txt').stat().st_mtime_ns==stamp
    with pytest.raises(ValueError):Journal(tmp_path,{'source':'changed'})
    atomic(tmp_path/'done.txt','corrupt')
    with pytest.raises(ValueError):Journal(tmp_path,manifest)


def test_selection_functions_no_test_or_group_read():
    from experiments.score_audit_v4.run import Workflow
    for fn in [Workflow.candidate,Workflow.fit]:
        source=inspect.getsource(fn)
        assert "['group']" not in source and 'historical' not in source


def test_quantitative_descriptive_stats_boolean_correctness():
    from experiments.score_audit_v4.evaluate import expert_stats, summary
    _,p,y=toy();weights=np.full((len(y),15),1/15)
    r=expert_stats(p[...,1],weights,p,y);assert r['expert_Spearman'] is not None
    r=summary(p.mean(1)[y==0],y[y==0]);assert r['class1_accuracy'] is None
