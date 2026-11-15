"""CCW v6: fixed B-only crossfit/selection; old C never certifies improvement."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from experiments.discrete_weighting.store import sha,key
from experiments.geometry_aggregation_v5.persistence import atomic,dump
from experiments.geometry_aggregation_v5.io import load,save_npz,single_writer
from experiments.score_audit_v4.evaluate import summary
from .store import Journal,csvwrite
from .model import METHODS,CONFIG,fit,infer
from .metrics import basic,diagnostics,metric_ci,aa_eligible,advantage_signal,paired_image_ci
from . import inputs

ROOT=Path('experiments/ccw_v6')
CMD='.venv/Scripts/python.exe -m experiments.ccw_v6.run --phase all --resume'
PROTOCOL=dict(model=CONFIG,methods=METHODS,inner_folds=4,inner_seed='61000+100*seed+outer_fold',
    normalizer='B-inner-train per actual head; ddof0; sigma floor1e-6; reference medians',
    bins='2 unweighted pooled z quantiles across heads within each B-inner-train actual ensemble; non-strict edges -> single-bin backoff',
    parent='predicted-class weighted marginal shrunk to global with kappa25; global raw weighted mean then0; no self-reference',
    selection='Mean worst-class NLL over9 outer B-OOF evaluations, then mean worst-class accuracy, seed-pooled AA, lower concentration, stable method order. Strict per-seed AA delta_U0>-1pp.',
    seed_pool='For repeated B image IDs across outer folds, average predictions per image before seed metrics/AA floor; no independent-row assumption.',
    proposal_gate='Primary FULL must be AA-eligible and outrank G1. If best eligible overall is CCW_A or CCW_FULL, lock that one; otherwise NO_GO. No G1/G2/U0 promotion and no CCW_A rescue if FULL loses to G1.',
    retrospective='Replay only after a GO lock; if NO_GO, retain only Gate0 U0/G1 diagnostic metrics and skip new CCW C replay.',
    optional_random_controls=False,head_retraining=False,historical_test='Never read',fresh_required_for_integration=True)


def dag():
    d={'recovery_store':[],'inventory':['recovery_store'],'package_verify':['inventory']}
    for s in range(3):
        for f in range(3):d[f'B_export:{s}:{f}']=['package_verify'];d[f'G1_reproduce:{s}:{f}']=['package_verify']
    d['gate0']=[f'{p}:{s}:{f}' for s in range(3) for f in range(3) for p in ['B_export','G1_reproduce']]
    d['real_recovery']=['gate0']
    for s in range(3):
        for f in range(3):
            d[f'B_split:{s}:{f}']=['real_recovery']
            for i in range(4):
                fitid=f'fit:{s}:{f}:{i}';d[fitid]=[f'B_split:{s}:{f}']
                for m in METHODS:
                    p=f'B_predict:{s}:{f}:{i}:{m}';e=f'B_evaluate:{s}:{f}:{i}:{m}';d[p]=[fitid];d[e]=[p]
            for m in METHODS:d[f'B_outer:{s}:{f}:{m}']=[f'B_evaluate:{s}:{f}:{i}:{m}' for i in range(4)]
        for m in METHODS:d[f'B_seed:{s}:{m}']=[f'B_outer:{s}:{f}:{m}' for f in range(3)]
    d['selection_lock']=[f'B_seed:{s}:{m}' for s in range(3) for m in METHODS]
    for s in range(3):
        for f in range(3):d[f'full_fit:{s}:{f}']=['selection_lock']
    d['phase1_report']=['selection_lock']
    for s in range(3):
        for f in range(3):
            for m in METHODS:
                p=f'C_predict:{s}:{f}:{m}';e=f'C_evaluate:{s}:{f}:{m}';d[p]=['phase1_report',f'full_fit:{s}:{f}'];d[e]=[p]
        for m in METHODS:d[f'C_pool:{s}:{m}']=[f'C_evaluate:{s}:{f}:{m}' for f in range(3)]
    d['retrospective_report']=['phase1_report'];d['confirmation_decision']=['retrospective_report'];d['final_report']=['confirmation_decision']
    return d


def inner_splits(B,seed,fold):return list(StratifiedKFold(4,shuffle=True,random_state=61000+100*seed+fold).split(B['probs'],B['y']))


def child_worker(folder,pool=None,pause=False):
    """Recovery exercise in a separate process; deterministic immutable task receipts."""
    folder=Path(folder)
    if pool:
        B=load(pool);manifest=dict(test='real B inner-fold recovery',pool_sha256=sha(pool),source_sha256=sha(Path(__file__)))
    else:
        rng=np.random.default_rng(662);dist=rng.uniform(.7,1.6,(80,15,2)).astype(np.float32);logits=-dist;exp=np.exp(logits-logits.max(-1,keepdims=True))
        B=dict(probs=exp/exp.sum(-1,keepdims=True),distances=dist,y=np.tile([0,1],40),sample_id=np.array([f'synthetic_{i}' for i in range(80)]),head_sha256=np.array([f'head{j}' for j in range(15)]));manifest=dict(test='synthetic durable store recovery',source_sha256=sha(Path(__file__)))
    j=Journal(folder,manifest)
    for i,(train,valid) in enumerate(inner_splits(B,0,0)[:2]):
        def work(i=i,train=train,valid=valid):
            model=fit(B['probs'][train],B['distances'][train],B['y'][train],B['sample_id'][train],B['head_sha256'])
            modelpath=folder/f'inner{i}/model.json';dump(modelpath,model)
            if pause and i==1:
                atomic(folder/'pause.ready','one inner-fold committed; second fitted but not committed');time.sleep(120)
            out=infer(B['probs'][valid],B['distances'][valid],model,B['head_sha256'],'CCW_FULL');p=folder/f'inner{i}/predictions.npz';save_npz(p,**out,sample_id=B['sample_id'][valid])
            return dict(inner_fold=i,AA=float((out['probs'].argmax(1)==B['y'][valid]).mean())),[modelpath,p]
        j.task(f'inner:{i}',[] if i==0 else ['inner:0'],work)


def recovery(root,real=False):
    folder=root/'acceptance'/('real_inner' if real else 'store');control=folder.with_name(folder.name+'_uninterrupted');pool=root/'inputs/B/seed0/fold0.npz' if real else None
    cmd=[sys.executable,'-m','experiments.ccw_v6.run','--recovery-worker',str(folder)]+(['--pool',str(pool)] if real else [])
    env=os.environ|dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1')
    log=folder.with_suffix('.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as stream:
        if len(list((folder/'completed').glob('*.json')))<2:
            proc=subprocess.Popen(cmd+['--pause-worker'],stdout=stream,stderr=subprocess.STDOUT,env=env)
            try:
                for _ in range(1200):
                    if (folder/'pause.ready').exists():break
                    if proc.poll() is not None:raise RuntimeError('Recovery worker failed: '+str(log))
                    time.sleep(.05)
                assert (folder/'pause.ready').exists()
            finally:
                if proc.poll() is None:proc.kill()
                proc.wait(timeout=15)
            assert len(list((folder/'completed').glob('*.json')))==1
            first=folder/'inner0/model.json';stamp=first.stat().st_mtime_ns;digest=sha(first)
            subprocess.run(cmd,check=True,env=env,stdout=stream,stderr=subprocess.STDOUT)
            assert first.stat().st_mtime_ns==stamp and sha(first)==digest
        direct=[sys.executable,'-m','experiments.ccw_v6.run','--recovery-worker',str(control)]+(['--pool',str(pool)] if real else [])
        subprocess.run(direct,check=True,env=env,stdout=stream,stderr=subprocess.STDOUT)
    paths=[log]
    for i in range(2):
        a=folder/f'inner{i}/model.json';b=control/f'inner{i}/model.json';assert json.loads(a.read_text())==json.loads(b.read_text())
        x=load(folder/f'inner{i}/predictions.npz');y=load(control/f'inner{i}/predictions.npz')
        for k in x:np.testing.assert_array_equal(x[k],y[k])
        paths.extend([a,folder/f'inner{i}/predictions.npz'])
    record=folder/'recovery_receipt.json';dump(record,dict(status='PASS',forced_kill_mid_inner1=True,inner0_completed_before_kill=True,
        completed_inner0_reused_without_rewrite=True,resumed_models_predictions_equal_uninterrupted=True,unfinished_tasks_replayed=1,real_B=real))
    return dict(recovery='PASS',real_B=real),paths+[record]


class Workflow:
    def __init__(self,args):
        self.args=args;self.discovery=inputs.discover();source=list(ROOT.glob('*.py'))+[
            Path('experiments/geometry_aggregation_v5/journal.py'),Path('experiments/geometry_aggregation_v5/persistence.py'),Path('experiments/geometry_aggregation_v5/io.py'),
            Path('experiments/discrete_weighting/store.py'),Path('experiments/score_audit_v4/evaluate.py'),Path('tests/methods/test_ccw_v6.py')]
        parent=ROOT/'history/manifest.r2.json'
        self.manifest=dict(version=3,parent_manifest_sha256=sha(parent) if parent.exists() else None,
            revision_reason='Add explicitly authorized isolated exploratory_c CLI; original formulas, inputs, splits, selection/gate and completed results unchanged.',
            source={str(p):sha(p) for p in sorted(source)},inputs=self.discovery['base_inputs'],config=PROTOCOL,planned_DAG=dag(),runtime=dict(numpy=np.__version__,python=sys.version.split()[0]))
        self.j=Journal(ROOT,self.manifest,args.max_tasks)
        for p,h in self.manifest['source'].items():
            dest=ROOT/'source_snapshot_r3'/p
            if not dest.exists():atomic(dest,Path(p).read_bytes())
            assert sha(dest)==h
        if not (ROOT/'PROTOCOL_DECISIONS.md').exists():atomic(ROOT/'PROTOCOL_DECISIONS.md','# Frozen CCW v6 protocol\n\n'+json.dumps(PROTOCOL,indent=2)+'\n\n'
            'No retraining or new architecture. Only matching v5 B caches may fit/select. All B-inner statistics, labels, bins, boundary quantile and table support use inner-training only. B group fields are never exported or read by the fitter/selector.\n'
            'Quantile edges are pooled across normalized head observations within each actual ensemble, not pooled head identities across outer folds. kappa25 support is effective image-weight units: each image weight divided by15.\n'
            'Seed B pooling averages duplicate image predictions across outer ensembles before AA floor/seed metrics. Mean proxy ranks nine outer B-OOF evaluations; uncertainty clusters on unique IDs. Exactly the five prescribed methods; optional random permutation controls omitted.\n'
            'A CCW_A best-proxy result can be locked only if primary FULL itself outranks G1 and satisfies AA floor; otherwise NO_GO. This prevents rescuing a failed primary by changing beta/rho. No G1/G2/U0 control is a new promoted method.\n'
            'If NO_GO, no new CCW C replay or full-B deployment refit; retain Gate0 U0/G1 retrospective results and all B failures. If GO, full-B refit and all frozen control C diagnostics remain exploratory. No fresh data means no integration regardless of C metrics.\n'
            'Original official labels are retained, including the documented species-label caveat. C and historical test labels never enter fit/selection. A new filename or different backbone on the same previously examined images is not fresh confirmation.\n')

    def task(self,*args):return self.j.task(*args)

    def B(self,seed,fold):return load(ROOT/f'inputs/B/seed{seed}/fold{fold}.npz')

    def phase0(self):
        self.task('recovery_store',[],lambda:recovery(ROOT))
        self.task('inventory',['recovery_store'],lambda:inputs.inventory(ROOT,self.discovery))
        self.task('package_verify',['inventory'],lambda:inputs.extract_and_verify(ROOT,self.discovery))
        for s in range(3):
            for f in range(3):
                self.task(f'B_export:{s}:{f}',['package_verify'],lambda s=s,f=f:inputs.export_B(ROOT,s,f))
                def reproduce(s=s,f=f):
                    path=ROOT/f'inputs/v5_analysis/caches/seed{s}/fold{f}/C_predictions.npz';C=load(path);module=inputs.frozen_geometry(ROOT)
                    p,w=module.distance_mix(C['probs'],C['distances'],ROOT/'inputs/v5_analysis/source/verified_distance_v1.json');uniform=C['probs'].mean(1)
                    out=ROOT/f'results/gate0/seed{s}/fold{f}/baselines.npz';save_npz(out,U0=uniform,G1_frozen=p,weights=w,sample_id=C['sample_id'])
                    rows=[]
                    for method,value in [('U0',uniform),('G1_frozen',p)]:rows.append(dict(seed=s,outer_fold=f,method=method,role='PREEXISTING C DIAGNOSTIC ONLY',**summary(value,C['y'],C['group'])))
                    csv=out.with_suffix('.csv');csvwrite(csv,rows);return dict(seed=s,outer_fold=f,G1='PASS',samples=len(p)),[out,csv]
                self.task(f'G1_reproduce:{s}:{f}',['package_verify'],reproduce)
        def gate():
            rows=[];expected=json.loads((inputs.V5/'promotion_decisions.json').read_text(encoding='utf-8'))['decisions'][0]['seeds'];index={r['sample_id']:int(r['development_index']) for r in inputs.canonical(ROOT)}
            for s in range(3):
                predictions=[load(ROOT/f'results/gate0/seed{s}/fold{f}/baselines.npz') for f in range(3)];labels=[load(ROOT/f'inputs/v5_analysis/caches/seed{s}/fold{f}/C_predictions.npz',['y','group','sample_id']) for f in range(3)]
                ids=np.concatenate([p['sample_id'] for p in predictions]);order=np.argsort([index[str(v)] for v in ids]);y=np.concatenate([p['y'] for p in labels])[order];g=np.concatenate([p['group'] for p in labels])[order]
                for m in ['U0','G1_frozen']:
                    p=np.concatenate([v[m] for v in predictions])[order];r=summary(p,y,g);ref=expected[s]
                    assert abs(r['AA']-ref['uniform_AA' if m=='U0' else 'AA'])<1e-12 and abs(r['WGA']-ref['uniform_WGA' if m=='U0' else 'WGA'])<1e-12
                    rows.append(dict(seed=s,method=m,**r))
            csv=ROOT/'results/g1_reproduction.csv';csvwrite(csv,rows)
            status=ROOT/'B_INPUT_STATUS.json';dump(status,dict(status='PASS',matching_B_caches=9,images_per_fold=240,group_fields_in_B=False,head_order_and_disjointness=True,new_head_training=False))
            p=ROOT/'inputs/manifest.json';dump(p,dict(status='PASS',source_input_hashes=self.manifest['inputs'],package_manifest_sha256=sha(ROOT/'inputs/v5_analysis/manifest.json'),B_caches={str(p):sha(p) for p in (ROOT/'inputs/B').glob('seed*/fold*.npz')}))
            report=ROOT/'reports/PHASE0.md';atomic(report,'# Gate0 PASS\n\nAll9 C caches/package manifests and exact frozen G1 reproduce. Per-seed AA/WGA match v5 to numerical precision. Matching B caches all240 images and15 ordered heads verified. B-only exports contain no group fields. No model training, unrelated OOF substitution or historical test loading.\n')
            return dict(gate0='PASS',B_caches=9),[csv,status,p,report]
        self.task('gate0',self.manifest['planned_DAG']['gate0'],gate)

    def phase1(self):
        self.task('real_recovery',['gate0'],lambda:recovery(ROOT,True))
        for s in range(3):
            for f in range(3):
                B=self.B(s,f);pairs=inner_splits(B,s,f)
                def plan(s=s,f=f,B=B,pairs=pairs):
                    rows=[dict(inner_fold=i,train_indices=a.tolist(),valid_indices=b.tolist(),train_ids=B['sample_id'][a].tolist(),valid_ids=B['sample_id'][b].tolist()) for i,(a,b) in enumerate(pairs)]
                    for r in rows:assert not set(r['train_ids'])&set(r['valid_ids'])
                    path=ROOT/f'checkpoints/seed{s}/fold{f}/inner_splits.json';dump(path,dict(seed=s,outer_fold=f,split_seed=61000+100*s+f,folds=rows));return dict(seed=s,outer_fold=f,inner_folds=4),[path]
                self.task(f'B_split:{s}:{f}',['real_recovery'],plan)
                for i,(a,b) in enumerate(pairs):
                    folder=ROOT/f'checkpoints/seed{s}/fold{f}/inner{i}'
                    def train(s=s,f=f,i=i,a=a,B=B,folder=folder):
                        cached=ROOT/f'acceptance/real_inner/inner{i}/model.json'
                        if s==0 and f==0 and i<2:
                            model=json.loads(cached.read_text());assert model['fit_ids']==B['sample_id'][a].tolist();assert model['head_sha256']==B['head_sha256'].tolist()
                        else:model=fit(B['probs'][a],B['distances'][a],B['y'][a],B['sample_id'][a],B['head_sha256'])
                        path=folder/'model.json';dump(path,model);cells=folder/'cells.csv';csvwrite(cells,model['cells']);provenance=folder/'fit_provenance.json';dump(provenance,dict(code_manifest_sha256=sha(ROOT/'manifest.json'),B_source_sha256=sha(ROOT/f'inputs/B/seed{s}/fold{f}.npz'),fit_ids=model['fit_ids'],inner_validation_not_used=True,reused_recovery_fit=s==0 and f==0 and i<2))
                        return dict(seed=s,outer_fold=f,inner_fold=i,fit_images=len(a),sigma_A=model['sigma_A']),[path,cells,provenance]
                    fitid=f'fit:{s}:{f}:{i}';self.task(fitid,[f'B_split:{s}:{f}'],train);model=json.loads((folder/'model.json').read_text())
                    for m in METHODS:
                        outfolder=folder/m;predid=f'B_predict:{s}:{f}:{i}:{m}'
                        def predict(m=m,b=b,B=B,model=model,outfolder=outfolder,folder=folder):
                            out=infer(B['probs'][b],B['distances'][b],model,B['head_sha256'],m);g1=infer(B['probs'][b],B['distances'][b],None,B['head_sha256'],'G1_frozen')['probs']
                            path=outfolder/'predictions.npz';save_npz(path,**out,g1_probs=g1,sample_id=B['sample_id'][b],q20=np.full(len(b),model['q20']))
                            desc=outfolder/'fitted_method.json';dump(desc,dict(method=m,common_fit_sha256=sha(folder/'model.json') if m not in ['U0','G1_frozen'] else None,normalizer='B-inner-train only' if m not in ['U0','G1_frozen'] else None,table_used=m=='CCW_FULL',head_sha256=B['head_sha256'].tolist()))
                            return dict(method=m,validation_images=len(b)),[path,desc]
                        self.task(predid,[fitid],predict)
                        def evaluate(s=s,f=f,i=i,m=m,b=b,B=B,outfolder=outfolder):
                            out=load(outfolder/'predictions.npz');y=B['y'][b];r=basic(out['probs'],y,out['uniform'],out['g1_probs']);d=diagnostics(out,B['probs'][b],y,out['q20']);r.update(seed=s,outer_fold=f,inner_fold=i,method=m,level='inner',effective_members=float(d['effective_members'].mean()),concentration=float((out['weights']**2).sum(1).mean()))
                            path=outfolder/'evaluation.json';dump(path,r);table=outfolder/'per_image.csv';csvwrite(table,[dict(seed=s,outer_fold=f,inner_fold=i,method=m,sample_id=str(out['sample_id'][j]),y=int(y[j]),p0=float(out['probs'][j,0]),p1=float(out['probs'][j,1]),uniform_p1=float(out['uniform'][j,1]),G1_p1=float(out['g1_probs'][j,1]),prediction=int(out['probs'][j].argmax()),uniform_wrong=bool(d['uniform_wrong'][j]),low_margin=bool(d['low_margin'][j]),signed_advantage_gain=float(d['gain'][j]),corrected_U0=bool(d['corrected'][j]),broken_U0=bool(d['broken'][j]),corrected_G1=bool(out['probs'][j].argmax()==y[j] and out['g1_probs'][j].argmax()!=y[j]),broken_G1=bool(out['probs'][j].argmax()!=y[j] and out['g1_probs'][j].argmax()==y[j]),entropy=float(d['entropy'][j]),effective_members=float(d['effective_members'][j]),top_head=int(d['top_head'][j]),unique_top=bool(d['unique_top'][j]),top_ties=int(d['top_ties'][j]),geometry_range=float(d['geometry_range'][j]),advantage_range=float(d['advantage_range'][j]),advantage_dominates=bool(d['advantage_dominates'][j]),true_probability_direction=float(d['true_probability_direction'][j])) for j in range(len(y))])
                            return r,[path,table]
                        self.task(f'B_evaluate:{s}:{f}:{i}:{m}',[predid],evaluate)
                for m in METHODS:self.task(f'B_outer:{s}:{f}:{m}',[f'B_evaluate:{s}:{f}:{i}:{m}' for i in range(4)],lambda s=s,f=f,m=m:self.pool_outer(s,f,m))
            for m in METHODS:self.task(f'B_seed:{s}:{m}',[f'B_outer:{s}:{f}:{m}' for f in range(3)],lambda s=s,m=m:self.pool_seed(s,m))
        lock=self.task('selection_lock',self.manifest['planned_DAG']['selection_lock'],self.select)
        if lock['status']=='GO':
            for s in range(3):
                for f in range(3):
                    def full(s=s,f=f):
                        B=self.B(s,f);model=fit(B['probs'],B['distances'],B['y'],B['sample_id'],B['head_sha256']);model['selection_lock_sha256']=sha(ROOT/'LOCKED_CANDIDATE.json');p=ROOT/f'checkpoints/full_B/seed{s}/fold{f}/model.json';dump(p,model);cells=p.with_name('cells.csv');csvwrite(cells,model['cells']);return dict(seed=s,outer_fold=f,fit_images=len(B['y'])),[p,cells]
                    self.task(f'full_fit:{s}:{f}',['selection_lock'],full)
        else:self.j.current['skipped_tasks']=[t for t in self.manifest['planned_DAG'] if t.startswith(('full_fit:','C_'))]
        from .report import phase1_report
        self.task('phase1_report',['selection_lock'],lambda:phase1_report(ROOT,self.j,lock))
        return lock

    def pool_outer(self,s,f,m):
        B=self.B(s,f);parts=[load(ROOT/f'checkpoints/seed{s}/fold{f}/inner{i}/{m}/predictions.npz') for i in range(4)];ids=np.concatenate([p['sample_id'] for p in parts]);idx={str(v):i for i,v in enumerate(B['sample_id'])};order=np.argsort([idx[str(v)] for v in ids]);assert len(set(ids))==len(ids)==len(B['y'])
        z={k:np.concatenate([p[k] for p in parts])[order] for k in parts[0]};np.testing.assert_array_equal(z['sample_id'],B['sample_id'])
        r=basic(z['probs'],B['y'],z['uniform'],z['g1_probs']);r.update(metric_ci(z['probs'],z['uniform'],z['g1_probs'],B['y'],B['sample_id']));r.update(seed=s,outer_fold=f,inner_fold=-1,method=m,level='outer_B',effective_members=float((1/(z['weights']**2).sum(1)).mean()),concentration=float((z['weights']**2).sum(1).mean()))
        folder=ROOT/f'results/B_outer/seed{s}/fold{f}/{m}';p=folder/'predictions.npz';save_npz(p,**z,y=B['y']);ep=folder/'evaluation.json';dump(ep,r);return r,[p,ep]

    def pool_seed(self,s,m):
        parts=[load(ROOT/f'results/B_outer/seed{s}/fold{f}/{m}/predictions.npz') for f in range(3)];ids=np.concatenate([p['sample_id'] for p in parts]);unique,inverse=np.unique(ids,return_inverse=True);counts=np.bincount(inverse);out={}
        for k in ['probs','uniform','g1_probs']:
            value=np.concatenate([p[k] for p in parts]);sums=np.zeros((len(unique),2));np.add.at(sums,inverse,value);out[k]=sums/counts[:,None]
        raw_y=np.concatenate([p['y'] for p in parts]);y=np.zeros(len(unique),int)
        for i in range(len(unique)):
            vals=raw_y[inverse==i];assert np.unique(vals).size==1;y[i]=vals[0]
        r=basic(out['probs'],y,out['uniform'],out['g1_probs']);r.update(metric_ci(out['probs'],out['uniform'],out['g1_probs'],y,unique));r.update(seed=s,outer_fold=-1,inner_fold=-1,method=m,level='seed_B',raw_B_rows=len(ids),unique_images=len(unique))
        folder=ROOT/f'results/B_seed/seed{s}/{m}';p=folder/'predictions.npz';save_npz(p,**out,sample_id=unique,y=y,multiplicity=counts);ep=folder/'evaluation.json';dump(ep,r);return r,[p,ep]

    def select(self):
        rankings=[]
        for m in METHODS:
            outer=[self.j.records[f'B_outer:{s}:{f}:{m}']['result'] for s in range(3) for f in range(3)];seeds=[self.j.records[f'B_seed:{s}:{m}']['result'] for s in range(3)]
            rankings.append(dict(method=m,eligible=all(aa_eligible(r['delta_AA_U0_pp']) for r in seeds),proxy=float(np.mean([r['worst_class_NLL'] for r in outer])),worst_class_accuracy=float(np.mean([r['worst_class_accuracy'] for r in outer])),AA=float(np.mean([r['AA'] for r in seeds])),concentration=float(np.mean([r['concentration'] for r in outer])),delta_AA_U0_pp_by_seed=[r['delta_AA_U0_pp'] for r in seeds]))
        rankings.sort(key=lambda r:(not r['eligible'],r['proxy'],-r['worst_class_accuracy'],-r['AA'],r['concentration'],METHODS.index(r['method'])))
        order={r['method']:i for i,r in enumerate(rankings)};full=next(r for r in rankings if r['method']=='CCW_FULL');best=rankings[0]
        go=full['eligible'] and order['CCW_FULL']<order['G1_frozen'] and best['method'] in ['CCW_A','CCW_FULL'] and best['eligible']
        result=dict(status='GO' if go else 'NO_GO',selected_method=best['method'] if go else None,primary='CCW_FULL',ranking=rankings,
            primary_beats_G1=order['CCW_FULL']<order['G1_frozen'],group_blind=True,C_used=False,config=CONFIG,
            reason='Best eligible CCW variant locked; primary also beats G1' if go else 'Primary loses to G1 / violates AA floor, or a fixed control ranks best; no rescue grid',
            code_manifest_sha256=sha(ROOT/'manifest.json'),B_data_hashes={str(p):sha(p) for p in (ROOT/'inputs/B').glob('seed*/fold*.npz')})
        p=ROOT/'LOCKED_CANDIDATE.json';dump(p,result);csv=ROOT/'results/B_selection_proxy.csv';csvwrite(csv,rankings);return dict(status=result['status'],selected_method=result['selected_method'],primary_beats_G1=result['primary_beats_G1']),[p,csv]

    def phase2(self,lock):
        from .retrospective import run_retrospective,confirmation
        run_retrospective(ROOT,self.j,lock)
        self.task('confirmation_decision',['retrospective_report'],lambda:confirmation(ROOT,lock,self.discovery['fresh']))
        from .report import final_report
        self.task('final_report',['confirmation_decision'],lambda:final_report(ROOT,self.j,lock))
        self.j.status('FEASIBLE_COMPLETE_CONFIRMATION_BLOCKED','Read INPUT_REQUEST.md; no CCW production integration; preserve the frozen B decision',
            B_decision=lock['status'],selected_method=lock['selected_method'],fresh_confirmation='BLOCKED_NO_FRESH_COHORT',production_integration='PROHIBITED',active_task=None,active_phase='complete')
        print(json.dumps(dict(completed=len(self.j.records),B_decision=lock['status'],confirmation='BLOCKED_NO_FRESH_COHORT',production='NOT_MODIFIED')))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['all','0','1','2','exploratory_c'],default='all');parser.add_argument('--resume',action='store_true');parser.add_argument('--status',action='store_true');parser.add_argument('--verify',action='store_true');parser.add_argument('--max-tasks',type=int);parser.add_argument('--recovery-worker');parser.add_argument('--pool');parser.add_argument('--pause-worker',action='store_true');args=parser.parse_args()
    if args.phase=='exploratory_c':
        from .exploration.run import main as exploratory_main
        return exploratory_main(args)
    if args.recovery_worker:return child_worker(args.recovery_worker,args.pool,args.pause_worker)
    if args.status:
        p=ROOT/'state.json';s=json.loads(p.read_text());print(json.dumps({k:v for k,v in s.items() if k not in ['task_DAG','completed_fits','skipped_tasks']},indent=2));return
    with single_writer(ROOT),threadpool_limits(limits=1):
        w=Workflow(args)
        if args.verify:print(json.dumps(dict(status='PASS',completed=len(w.j.records),all_artifact_hashes_verified=True)));return
        try:
            w.phase0()
            if args.phase=='0':w.j.status('GATE0_PASS','real_recovery / B crossfit');return
            lock=w.phase1()
            if args.phase=='1':w.j.status('PHASE1_COMPLETE','Frozen retrospective diagnostics / fresh-input audit',B_decision=lock['status']);return
            w.phase2(lock)
        except InterruptedError:print('Stopped at durable boundary; resume: '+CMD)
        except FileNotFoundError as exc:
            request=ROOT/'INPUT_REQUEST.md';atomic(request,'# Missing CCW input\n\n'+str(exc)+'\n\nProvide the exact matching B predictions (15-head probabilities/distances/IDs/y/head hashes and original ABC/head manifests), or those original heads and frozen features for inference only. Never substitute unrelated OOF heads or fit on C.\n')
            w.j.status('INPUT_BLOCKED',str(exc),blocker=str(exc),production_integration='PROHIBITED');raise


if __name__=='__main__':main()
