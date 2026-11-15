import ast
import copy
import importlib.util
import inspect
from pathlib import Path
import numpy as np
import pytest
from scipy.special import softmax
from experiments.ccw_v6.model import fit,infer,teacher,ccw_mix,features
from experiments.ccw_v6.metrics import aa_eligible
from experiments.ccw_v6.retrospective import promotion_allowed
from experiments.ccw_v6.store import Journal
from experiments.geometry_aggregation_v5.persistence import atomic


def toy(n=80):
    rng=np.random.default_rng(66);d=rng.uniform(.7,1.6,(n,15,2));p=softmax(-d,axis=-1);y=np.tile([0,1],n//2);ids=np.array([f'image{i}' for i in range(n)]);heads=[f'sha{j}' for j in range(15)]
    return p,d,y,ids,heads


def test_g1_exact_and_Uniform_mass():
    p,d,y,ids,heads=toy();p=p.astype(np.float32);d=d.astype(np.float32)
    spec=importlib.util.spec_from_file_location('original_v5_G1','baselines/dpe4subpop/src/dpe/geometry.py');g1=importlib.util.module_from_spec(spec);spec.loader.exec_module(g1)
    original,w=g1.distance_mix(p,d);out=ccw_mix(p,d,None,heads,rho=0,beta=0)
    np.testing.assert_array_equal(out['probs'],original);np.testing.assert_array_equal(out['weights'],w)
    u=ccw_mix(p,d,None,heads,alpha=0);np.testing.assert_array_equal(u['probs'],p.mean(1))


def test_inference_has_no_label_or_group_arguments_and_schema_head_guard():
    for fn in [infer,ccw_mix,features]:
        assert not {'y','correct','correctness','group','C','is_C'}&set(inspect.signature(fn).parameters)
    p,d,y,ids,heads=toy();model=fit(p,d,y,ids,heads)
    for forbidden in ['y','correct','group']:
        with pytest.raises(TypeError):infer(p,d,model,heads,'CCW_FULL',**{forbidden:y})
    with pytest.raises(ValueError):infer(p,d,model,heads[::-1],'CCW_FULL')
    wrong=copy.deepcopy(model);wrong['schema_version']=2
    with pytest.raises(ValueError):infer(p,d,wrong,heads,'CCW_FULL')
    a=infer(p,d,model,heads,'CCW_FULL');b=infer(p.copy(),d.copy(),model,heads,'CCW_FULL');np.testing.assert_array_equal(a['probs'],b['probs'])


def test_image_weight_support_hierarchical_parent_and_train_boundary():
    p,d,y,ids,heads=toy();train=np.arange(60);valid=np.arange(60,80)
    m=fit(p[train],d[train],y[train],ids[train],heads)
    np.testing.assert_allclose(m['normalizer']['mu'],d[train].min(-1).mean(0));np.testing.assert_allclose(m['normalizer']['sigma'],d[train].min(-1).std(0,ddof=0))
    assert set(m['fit_ids']).isdisjoint(ids[valid]);assert abs(sum(c['effective_image_support'] for c in m['cells'])-60)<1e-10
    assert abs(m['global_target_mean'])<1e-14
    frozen=copy.deepcopy(m);infer(p[valid],d[valid]*1.1,m,heads,'CCW_FULL');assert m==frozen
    np.testing.assert_allclose(teacher(p,y).mean(1),0,atol=1e-15)
    source=Path('experiments/ccw_v6/model.py').read_text();assert 'np.load' not in source and 'Path(' not in source


def test_zero_variance_ties_empty_cells_and_unanimity():
    p,d,y,ids,heads=toy();p[:]=.5;d[:]=1.
    m=fit(p,d,y,ids,heads);assert m['tie_backoff'] and m['z_edges']==[] and m['sigma_A']==.02
    assert any(c['expert_observations']==0 for c in m['cells'])
    for method in ['U0','G1_frozen','G2_z','CCW_A','CCW_FULL']:
        out=infer(p,d,m,heads,method);np.testing.assert_allclose(out['weights'],1/15);np.testing.assert_allclose(out['probs'],.5)


def test_signed_advantage_monotonic_and_probability_extremes():
    p,d,y,ids,heads=toy();p[:]=[.6,.4];p[:,0]=[.4,.6];m=fit(p,d,y,ids,heads);m['table']=np.zeros((2,3,2)).tolist()
    a=infer(p,d,m,heads,'CCW_FULL');m['table'][1]=np.full((3,2),.05).tolist();b=infer(p,d,m,heads,'CCW_FULL')
    assert np.all(b['scores'][:,0]>a['scores'][:,0]) and np.all(b['weights'][:,0]>=a['weights'][:,0])
    p[:]=[1-1e-12,1e-12];p[:,0]=[1e-12,1-1e-12];d[:]=[0,2];d[:,0]=[2,0];m=fit(p,d,y,ids,heads)
    for method in ['U0','G1_frozen','G2_z','CCW_A','CCW_FULL']:
        out=infer(p,d,m,heads,method);assert np.isfinite(out['weights']).all();assert (out['weights']>=0).all();np.testing.assert_allclose(out['weights'].sum(1),1);np.testing.assert_allclose(out['probs'].sum(1),1)


def test_class_swap_and_maximal_valid_disagreement():
    p,d,y,ids,heads=toy();a=fit(p,d,y,ids,heads);b=fit(p[...,::-1],d[...,::-1],1-y,ids,heads)
    out1=infer(p,d,a,heads,'CCW_FULL');out2=infer(p[...,::-1],d[...,::-1],b,heads,'CCW_FULL')
    np.testing.assert_allclose(out1['probs'],out2['probs'][...,::-1],atol=1e-12)
    p[:]=[.51,.49];p[:,-1]=[.01,.99];m=fit(p,d,y,ids,heads);_,_,pred,_,disagree=features(p,d,m['normalizer'],m['z_edges']);assert np.all(disagree.sum(1)==14)
    # All15 binary heads disagreeing with their probability mean is impossible.


def test_strict_floor_freshness_gate_and_original_production_unchanged():
    assert not aa_eligible(-1.) and not aa_eligible(-1.0008) and aa_eligible(-.9999)
    rows=[dict(delta_WGA_U0_pp=.5,delta_WGA_G1_pp=.4,delta_AA_U0_pp=0.) for _ in range(3)]
    assert promotion_allowed(False,rows)['status']=='BLOCKED';assert promotion_allowed(True,rows)['status']=='PASS'
    rows[0]['delta_AA_U0_pp']=-1.;assert promotion_allowed(True,rows)['status']=='FAIL'
    tree=ast.parse(Path('baselines/dpe4subpop/src/dpe/core.py').read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='predict_proba')
    default_map=dict(zip([a.arg for a in fn.args.args][-len(fn.args.defaults):],fn.args.defaults));assert default_map['aggregation'].value=='uniform'
    metadata=__import__('json').loads(Path('baselines/dpe4subpop/src/dpe/verified_distance_v1.json').read_text());assert metadata['gamma']==50 and metadata['alpha']==.25


def test_receipt_hash_reuse_and_incompatible_source(tmp_path):
    j=Journal(tmp_path,dict(source='fixed',planned_DAG={'first':[]}))
    def work():
        p=tmp_path/'saved';atomic(p,'valid');return {},[p]
    j.task('first',[],work);stamp=(tmp_path/'saved').stat().st_mtime_ns
    j=Journal(tmp_path,dict(source='fixed',planned_DAG={'first':[]}));j.task('first',[],lambda:pytest.fail('must not recompute'));assert (tmp_path/'saved').stat().st_mtime_ns==stamp
    with pytest.raises(ValueError):Journal(tmp_path,dict(source='changed'))
    atomic(tmp_path/'saved','corrupt')
    with pytest.raises(ValueError):Journal(tmp_path,dict(source='fixed',planned_DAG={'first':[]}))


def test_absent_fresh_data_serializes_blocked_decision(tmp_path):
    import json
    from experiments.ccw_v6.retrospective import confirmation
    result,paths=confirmation(tmp_path,dict(status='NO_GO'),tmp_path/'not_supplied.json')
    assert result['confirmation']=='BLOCKED' and all(p.is_file() for p in paths)
    decision=json.loads((tmp_path/'reports/promotion_decision.json').read_text())
    assert decision['production_integration'] is False and decision['reason']=='NO FRESH DATA'
