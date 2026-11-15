"""Explicit research-only C replay; cannot enter the original promotion DAG."""
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.ccw_v6.model import CONFIG,METHODS,fit,infer
from experiments.ccw_v6.inputs import V5,canonical
from experiments.geometry_aggregation_v5.io import load,save_npz,single_writer
from experiments.geometry_aggregation_v5.persistence import atomic,dump
from experiments.geometry_aggregation_v5.journal import csvwrite
from experiments.discrete_weighting.store import sha
from .journal import BASE,ROOT,BANNER,COMMAND,Journal

PRODUCTION=[Path('baselines/dpe4subpop/src/dpe')/p for p in ['core.py','geometry.py','verified_distance_v1.json']]


def bpath(s,f):return BASE/f'inputs/B/seed{s}/fold{f}.npz'
def cpath(s,f):return BASE/f'inputs/v5_analysis/caches/seed{s}/fold{f}/C_predictions.npz'
def nest(s,f):return V5/f'nested/seed{s}/fold{f}'
def predpath(s,f,m):return ROOT/f'predictions/seed{s}/fold{f}/{m}.npz'
def modelpath(s,f):return ROOT/f'fits/seed{s}/fold{f}/model.json'


def dag():
    d={'input_audit':[]}
    for s in range(3):
        for f in range(3):d[f'baseline:{s}:{f}']=['input_audit']
    d['baseline_gate']=[f'baseline:{s}:{f}' for s in range(3) for f in range(3)]
    d['recovery']=['baseline_gate']
    for s in range(3):
        for f in range(3):
            d[f'fit:{s}:{f}']=['recovery']
            for m in METHODS:d[f'infer:{s}:{f}:{m}']=[f'fit:{s}:{f}']
    predictions=[f'infer:{s}:{f}:{m}' for s in range(3) for f in range(3) for m in METHODS]
    d['inference_complete']=predictions
    for s in range(3):
        for m in METHODS:d[f'eval:{s}:{m}']=['inference_complete']
    d['evaluation_complete']=[f'eval:{s}:{m}' for s in range(3) for m in METHODS]
    d['prediction_export']=['evaluation_complete'];d['uncertainty']=['prediction_export']
    for name in ['RESULTS','PER_GROUP','FLIPS']:d[f'table:{name}']=['uncertainty']
    d['report']=[f'table:{name}' for name in ['RESULTS','PER_GROUP','FLIPS']]
    d['preservation_audit']=['report'];d['complete']=['preservation_audit']
    return d


def make_manifest():
    paths=[BASE/'model.py',BASE/'run.py',BASE/'store.py',Path('experiments/geometry_aggregation_v5')/'io.py',
           Path('experiments/geometry_aggregation_v5/journal.py'),Path('experiments/geometry_aggregation_v5/persistence.py'),
           Path('experiments/discrete_weighting/store.py'),Path('tests/methods/test_ccw_exploratory_c.py')]+list((BASE/'exploration').glob('*.py'))
    source={str(p):sha(p) for p in sorted(paths)}
    inputs={str(p):sha(p) for p in PRODUCTION+[V5/'promotion_decisions.json',BASE/'inputs/v5_analysis/provenance/development_index.csv']}
    for s in range(3):
        for f in range(3):
            for p in [bpath(s,f),cpath(s,f)]+[nest(s,f)/n for n in ['ABC_manifest.json','head_manifest.json','member_predictions.npz','G1_g50/predictions.npz']]:inputs[str(p)]=sha(p)
            h=json.loads((nest(s,f)/'head_manifest.json').read_text())
            for row in h['ordered_heads']:
                p=Path(row['path']);inputs[str(p)]=sha(p)
    protected={}
    for receipt in (BASE/'completed').glob('*.json'):
        r=json.loads(receipt.read_text());protected[str(receipt)]=sha(receipt)
        for p,a in r['artifacts'].items():protected[str(BASE/p)]=a['sha256']
    for p in [BASE/'LOCKED_CANDIDATE.json',BASE/'PROTOCOL_DECISIONS.md']+PRODUCTION:protected[str(p)]=sha(p)
    return dict(version=1,role=BANNER,original_NO_GO_sha256=sha(BASE/'LOCKED_CANDIDATE.json'),
        config=CONFIG,methods=METHODS,source=source,inputs=inputs,protected_original_artifacts=protected,
        planned_DAG=dag(),bootstrap=dict(repeats=2000,seed=20261115,unit='same unique image resampled across all seeds, paired, group-stratified; fixed fitted pipelines'),
        production_promotion=False,original_gate_modified=False,runtime=dict(numpy=np.__version__,python=sys.version.split()[0]))


def input_audit():
    canonical_rows=canonical(BASE);index={r['sample_id']:r for r in canonical_rows};rows=[]
    for s in range(3):
        pooled=[]
        for f in range(3):
            b=load(bpath(s,f));c=load(cpath(s,f));raw=load(nest(s,f)/'member_predictions.npz')
            abc=json.loads((nest(s,f)/'ABC_manifest.json').read_text());h=json.loads((nest(s,f)/'head_manifest.json').read_text())
            for split,z in [('B',b),('C',c)]:
                assert z['probs'].shape==z['distances'].shape==(len(z['sample_id']),15,2)
                assert np.isfinite(z['probs']).all() and np.isfinite(z['distances']).all()
                np.testing.assert_allclose(z['probs'].sum(-1),1,atol=2e-7,rtol=0)
                assert z['sample_id'].tolist()==abc[split]
                assert len(set(z['sample_id']))==len(z['sample_id'])
                np.testing.assert_array_equal(z['probs'],raw[split+'_probs']);np.testing.assert_array_equal(z['distances'],raw[split+'_distances'])
                np.testing.assert_array_equal(z['head_sha256'],raw['head_hashes'])
                np.testing.assert_array_equal(z['y'],[int(index[i]['y']) for i in z['sample_id']])
            np.testing.assert_array_equal(c['group'],[int(index[i]['group']) for i in c['sample_id']])
            assert not (set(abc['A'])&set(abc['B']) or set(abc['A'])&set(abc['C']) or set(abc['B'])&set(abc['C']))
            assert set(abc['A'])|set(abc['B'])|set(abc['C'])==set(index)
            for j,head in enumerate(h['ordered_heads']):
                assert head['slot']==j and sha(Path(head['path']))==head['sha256']==str(b['head_sha256'][j])
            pooled.extend(c['sample_id'].tolist());rows.append(dict(seed=s,fold=f,B=len(b['y']),C=len(c['y']),heads=15,role=BANNER))
        assert len(pooled)==len(set(pooled))==len(index) and set(pooled)==set(index)
    p=ROOT/'INPUT_AUDIT.json';dump(p,dict(status='PASS',role=BANNER,rows=rows,unique_images=len(index),heads_verified=135,
        B_C_disjoint=True,same_A_only_heads=True,labels='Original official labels; documented species-label caveat retained',C_for_fit=False))
    return dict(status='PASS',caches=9,role=BANNER),[p]


def baseline(s,f):
    c=load(cpath(s,f),['probs','distances','sample_id','head_sha256']);reference=load(nest(s,f)/'G1_g50/predictions.npz')
    np.testing.assert_array_equal(c['sample_id'],reference['sample_id']);np.testing.assert_array_equal(c['head_sha256'],reference['head_hashes'])
    outpaths=[];errors={}
    for m,key in [('U0','uniform'),('G1_frozen','probs')]:
        out=infer(c['probs'],c['distances'],None,c['head_sha256'],m)
        np.testing.assert_allclose(out['probs'],reference[key],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(out['probs'].argmax(1),reference[key].argmax(1))
        if m=='G1_frozen':np.testing.assert_allclose(out['weights'],reference['weights'],rtol=0,atol=1e-12)
        p=ROOT/f'baseline/seed{s}/fold{f}/{m}.npz';save_npz(p,**out,sample_id=c['sample_id'],head_sha256=c['head_sha256'],role=np.array(BANNER));outpaths.append(p)
        errors[m]=float(np.max(np.abs(out['probs']-reference[key])))
    return dict(seed=s,fold=f,status='PASS',max_probability_errors=errors,role=BANNER),outpaths


def baseline_gate():
    from .evaluate import metric
    records=json.loads((V5/'promotion_decisions.json').read_text())['decisions'][0]['seeds'];rows=[]
    for s in range(3):
        c=[load(cpath(s,f),['y','group','sample_id']) for f in range(3)]
        y=np.concatenate([v['y'] for v in c]);group=np.concatenate([v['group'] for v in c]);assert len(set(np.concatenate([v['sample_id'] for v in c])))==1199
        for m in ['U0','G1_frozen']:
            p=np.concatenate([load(ROOT/f'baseline/seed{s}/fold{f}/{m}.npz',['probs'])['probs'] for f in range(3)])
            r=metric(p,y,group);old=next(x for x in records if x['seed']==s)
            for k in ['AA','WGA']:np.testing.assert_allclose(r[k],old[('uniform_' if m=='U0' else '')+k],atol=1e-12,rtol=0)
            rows.append(dict(seed=s,method=m,**r,role=BANNER))
    p=ROOT/'BASELINE_REPRODUCTION.json';dump(p,dict(status='PASS',role=BANNER,predictions_exact=True,tolerance=1e-12,rows=rows))
    return dict(status='PASS',AA_WGA_match=True,predictions_exact=True,role=BANNER),[p]


def fit_full(s,f):
    b=load(bpath(s,f));reuse=ROOT/f'acceptance/interrupted/fold{f}/model.json'
    if s==0 and f<2:model=json.loads(reuse.read_text())
    else:model=fit(b['probs'],b['distances'],b['y'],b['sample_id'],b['head_sha256'])
    assert model['fit_ids']==b['sample_id'].tolist() and model['head_sha256']==b['head_sha256'].tolist() and model['config']==CONFIG
    p=modelpath(s,f);dump(p,model);cells=p.with_name('cells.csv');csvwrite(cells,[dict(**row,role=BANNER) for row in model['cells']])
    prov=p.with_name('provenance.json');dump(prov,dict(role=BANNER,seed=s,fold=f,source_manifest_sha256=sha(ROOT/'manifest.json'),
        B_source_sha256=sha(bpath(s,f)),head_sha256=model['head_sha256'],fit_ids=model['fit_ids'],C_labels_or_groups_used=False,
        reused_recovery_fit=s==0 and f<2,rng='Deterministic fit; no stochastic model state'))
    return dict(seed=s,fold=f,fit_images=len(model['fit_ids']),sigma_A=model['sigma_A'],role=BANNER),[p,cells,prov]


def predict(s,f,m):
    c=load(cpath(s,f),['probs','distances','sample_id','head_sha256'])
    if m in ['U0','G1_frozen']:out=load(ROOT/f'baseline/seed{s}/fold{f}/{m}.npz')
    else:
        fitted=json.loads(modelpath(s,f).read_text());out=infer(c['probs'],c['distances'],fitted,c['head_sha256'],m)
        out.update(sample_id=c['sample_id'],head_sha256=c['head_sha256'],role=np.array(BANNER))
    p=predpath(s,f,m);save_npz(p,**out)
    return dict(seed=s,fold=f,method=m,samples=len(c['sample_id']),labels_or_groups_loaded=False,role=BANNER),[p]


def marker(name):
    p=ROOT/f'phases/{name}.json';dump(p,dict(status='COMPLETE',role=BANNER,production_promotion=False));return dict(status='COMPLETE',role=BANNER),[p]


def preservation(manifest):
    for path,h in manifest['protected_original_artifacts'].items():assert sha(Path(path))==h,'Original artifact changed: '+path
    assert json.loads((BASE/'LOCKED_CANDIDATE.json').read_text())['status']=='NO_GO'
    p=ROOT/'ORIGINAL_PRESERVATION_AUDIT.json';dump(p,dict(status='PASS',role=BANNER,files_verified=len(manifest['protected_original_artifacts']),
        original_NO_GO_unchanged=True,production_unchanged=True,original_results_unchanged=True))
    return dict(status='PASS',original_NO_GO_unchanged=True,role=BANNER),[p]


def main(args):
    if args.status:
        print((ROOT/'state.json').read_text());return
    with single_writer(BASE),threadpool_limits(limits=1):
        manifest=make_manifest();j=Journal(ROOT,manifest,args.max_tasks)
        if args.verify:
            preservation(manifest);print(json.dumps(dict(status='PASS',completed=len(j.records),role=BANNER)));return
        for path,h in manifest['source'].items():
            p=ROOT/'source_snapshot'/path
            if not p.exists():atomic(p,Path(path).read_bytes())
            assert sha(p)==h
        protocol=ROOT/'PROTOCOL_DECISIONS.md'
        if not protocol.exists():atomic(protocol,'# Explicitly authorized separate research-only diagnostic\n\n'+BANNER+'\n\n'+json.dumps({k:manifest[k] for k in ['config','methods','bootstrap','production_promotion','original_gate_modified']},indent=2)+'\n\nAll B examples fit each frozen full-B model. The original recipe uses distance-z bins, predicted class and disagreement with Uniform; its B-only boundary confidence quantile weights fitting examples. No extra confidence binning or search is introduced. C labels/groups are evaluation-only. The original NO_GO remains authoritative for promotion.\n')
        try:
            j.task('input_audit',[],input_audit)
            for s in range(3):
                for f in range(3):j.task(f'baseline:{s}:{f}',['input_audit'],lambda s=s,f=f:baseline(s,f))
            j.task('baseline_gate',manifest['planned_DAG']['baseline_gate'],baseline_gate)
            from .recovery import acceptance
            j.task('recovery',['baseline_gate'],acceptance)
            for s in range(3):
                for f in range(3):
                    j.task(f'fit:{s}:{f}',['recovery'],lambda s=s,f=f:fit_full(s,f))
                    for m in METHODS:j.task(f'infer:{s}:{f}:{m}',[f'fit:{s}:{f}'],lambda s=s,f=f,m=m:predict(s,f,m))
            j.task('inference_complete',manifest['planned_DAG']['inference_complete'],lambda:marker('inference_complete'))
            from .evaluate import evaluate,prediction_export,uncertainty,table,report
            for s in range(3):
                for m in METHODS:j.task(f'eval:{s}:{m}',['inference_complete'],lambda s=s,m=m:evaluate(s,m))
            j.task('evaluation_complete',manifest['planned_DAG']['evaluation_complete'],lambda:marker('evaluation_complete'))
            j.task('prediction_export',['evaluation_complete'],prediction_export)
            j.task('uncertainty',['prediction_export'],uncertainty)
            for name in ['RESULTS','PER_GROUP','FLIPS']:j.task('table:'+name,['uncertainty'],lambda name=name:table(name))
            j.task('report',manifest['planned_DAG']['report'],report)
            j.task('preservation_audit',['report'],lambda:preservation(manifest))
            j.task('complete',['preservation_audit'],lambda:marker('complete'))
            j.status('EXPLORATORY_COMPLETE','No tuning or promotion. Read EXPLORATORY_C_REPORT.md.',report=str(ROOT/'EXPLORATORY_C_REPORT.md'),active_task=None,active_phase='complete')
            print(json.dumps(dict(status='EXPLORATORY_COMPLETE',new_tasks=j.count,completed=len(j.records),original_B_decision='NO_GO')))
        except InterruptedError:print('Stopped at durable boundary; resume: '+COMMAND)
