"""Nested B-only continuous precision weighting; all locked C methods are exploratory."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from experiments.geometry_aggregation_v5.io import load,save_npz,single_writer
from experiments.geometry_aggregation_v5.persistence import atomic
from experiments.geometry_aggregation_v5.journal import csvwrite
from experiments.discrete_weighting.store import sha
from experiments.ccw_v6.exploration.run import bpath,cpath,nest,BASE as V6
from .model import CONFIG,METHODS,fit_base,advantage,attach_scales,infer
from .store import ROOT,ROLE,COMMAND,Journal,digest,jsonwrite

HANDOFF=Path('baselines/dpe4subpop/DPE_G1_Precision_Weighting_v7_Codex_Handoff.md')
PRODUCTION=[Path('baselines/dpe4subpop/src/dpe')/n for n in ['core.py','geometry.py','verified_distance_v1.json']]


def B(s,f):return load(bpath(s,f))
def C(s,f,labels=False):return load(cpath(s,f),['sample_id','y','group'] if labels else ['probs','distances','sample_id','head_sha256'])
def outer_splits(b,s,f):return list(StratifiedKFold(4,shuffle=True,random_state=71000+100*s+f).split(b['probs'],b['y']))
def train_indices(b,s,f,part):return np.arange(len(b['y'])) if part==4 else outer_splits(b,s,f)[part][0]
def secondary_splits(b,s,f,part):
    train=train_indices(b,s,f,part);pairs=StratifiedKFold(3,shuffle=True,random_state=72000+1000*s+100*f+10*part).split(b['probs'][train],b['y'][train])
    return [(train[a],train[v]) for a,v in pairs]
def folder(s,f,part):return ROOT/f'checkpoints/seed{s}/fold{f}'/('full_B' if part==4 else f'b_inner{part}')
def pred_path(s,f,part,m):return folder(s,f,part)/('C_inference' if part==4 else 'B_inference')/f'{m}.npz'
def subname(s,f,part,k):return f'scale_fit:{s}:{f}:{part}:{k}'
def fitname(s,f,part):return f'fit:{s}:{f}:{part}'
def predname(s,f,part,m):return f'{"C" if part==4 else "B"}_infer:{s}:{f}:{part}:{m}'


def graph():
    d={'recovery_store':[],'inventory':['recovery_store']};specs={}
    for s in range(3):
        for f in range(3):d[f'baseline:{s}:{f}']=['inventory']
    d['baseline_gate']=[f'baseline:{s}:{f}' for s in range(3) for f in range(3)];d['recovery_secondary']=['baseline_gate']
    for s in range(3):
        for f in range(3):
            b=B(s,f);d[f'B_splits:{s}:{f}']=['recovery_secondary']
            for part in range(5):
                train=train_indices(b,s,f,part)
                for k,(a,v) in enumerate(secondary_splits(b,s,f,part)):
                    name=subname(s,f,part,k);d[name]=['B_gate'] if part==4 else [f'B_splits:{s}:{f}']
                    specs[name]=dict(seed=s,fold=f,part=part,secondary=k,train_ids=b['sample_id'][a].tolist(),validation_ids=b['sample_id'][v].tolist(),
                        head_sha256=b['head_sha256'].tolist(),rng_seed=72000+1000*s+100*f+10*part)
                name=fitname(s,f,part);d[name]=[subname(s,f,part,k) for k in range(3)]
                specs[name]=dict(seed=s,fold=f,part=part,train_ids=b['sample_id'][train].tolist(),head_sha256=b['head_sha256'].tolist())
                for m in METHODS:
                    name=predname(s,f,part,m);d[name]=[fitname(s,f,part)] if part<4 else ['recovery_C']
                    specs[name]=dict(seed=s,fold=f,part=part,method=m,head_sha256=b['head_sha256'].tolist(),
                        validation_ids=outer_splits(b,s,f)[part][1].tolist() if part<4 else json.loads((nest(s,f)/'ABC_manifest.json').read_text())['C'])
                if part<4:d[f'B_eval:{s}:{f}:{part}']=[predname(s,f,part,m) for m in METHODS]
    d['B_tables']=[f'B_eval:{s}:{f}:{part}' for s in range(3) for f in range(3) for part in range(4)]
    for batch in range(20):d[f'B_bootstrap:{batch}']=['B_tables'] if batch==0 else [f'B_bootstrap:{batch-1}']
    d['B_gate']=[f'B_bootstrap:{b}' for b in range(20)]
    d['recovery_C']=[fitname(s,f,4) for s in range(3) for f in range(3)]
    d['C_inference_complete']=[predname(s,f,4,m) for s in range(3) for f in range(3) for m in METHODS]
    for s in range(3):
        for m in METHODS:d[f'C_eval:{s}:{m}']=['C_inference_complete']
    d['C_export']=[f'C_eval:{s}:{m}' for s in range(3) for m in METHODS]
    for batch in range(20):d[f'C_bootstrap:{batch}']=['C_export'] if batch==0 else [f'C_bootstrap:{batch-1}']
    for batch in range(60):d[f'C_permutation:{batch}']=['C_export'] if batch==0 else [f'C_permutation:{batch-1}']
    d['C_tables']=[f'C_bootstrap:{b}' for b in range(20)]+[f'C_permutation:{b}' for b in range(60)]
    d['oracle']=['C_export'];d['report']=['C_tables','oracle','B_gate'];d['preservation']=['report'];d['complete']=['preservation']
    return d,specs


def manifest():
    source=list(ROOT.glob('*.py'))+list((ROOT/'tests').glob('*.py'))+[Path(p) for p in [
        'experiments/ccw_v6/model.py','experiments/geometry_aggregation_v5/journal.py','experiments/geometry_aggregation_v5/persistence.py',
        'experiments/geometry_aggregation_v5/io.py','experiments/discrete_weighting/store.py','experiments/ccw_v6/exploration/run.py']]
    paths=[HANDOFF]+PRODUCTION+[V6/'inputs/v5_analysis/provenance/development_index.csv',Path('outputs/geometry_aggregation_v5/run1/promotion_decisions.json'),V6/'exploratory_c/EXPLORATORY_C_RESULTS.csv']
    for s in range(3):
        for f in range(3):
            paths.extend([bpath(s,f),cpath(s,f)]+[nest(s,f)/n for n in ['ABC_manifest.json','head_manifest.json','member_predictions.npz','G1_g50/predictions.npz']])
            paths.extend(Path(r['path']) for r in json.loads((nest(s,f)/'head_manifest.json').read_text())['ordered_heads'])
    protected={}
    for base in [V6,V6/'exploratory_c']:
        for receipt in (base/'completed').glob('*.json'):
            protected[str(receipt)]=sha(receipt)
            for p,a in json.loads(receipt.read_text())['artifacts'].items():protected[str(base/p)]=a['sha256']
        for n in ['state.json','HANDOFF_STATUS.md','NEXT_AGENT_PROMPT.md','manifest.json']:
            if (base/n).exists():protected[str(base/n)]=sha(base/n)
    for p in PRODUCTION+[V6/'LOCKED_CANDIDATE.json']:protected[str(p)]=sha(p)
    d,specs=graph();locked=ROOT/'EXPERIMENT_LOCK.json'
    source_commit=json.loads(locked.read_text())['source_commit'] if locked.exists() else subprocess.run(['git','rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    return dict(version=1,role=ROLE,config=CONFIG,methods=METHODS,primary='G1_RH',source_commit=source_commit,
        source={str(p):sha(p) for p in sorted(source)},inputs={str(p):sha(p) for p in sorted(set(paths))},
        protected_original_artifacts=protected,planned_DAG=d,task_specs=specs,
        decision_rules=dict(direction='mean of per-seed unique-image mean gain differences on pooled-Uniform-wrong B images >0 and >=2 positive seeds',
            proxy='mean of nine ensemble B-OOF worst-class NLL; RH-G1 <=0.003',AA='strict per-seed RH-U0 >-1pp',
            scaling='separate pooled/head s_R from 3 secondary OOF partitions; threshold 0.002 only; finite/support integrity fallback; no quality-based posthoc cutoff',
            random_controls='3000 primary RH within-image residual permutations and G1 within-image weight permutations; retain per-image residual multiset, not guaranteed final mixture concentration',
            B_uncertainty='2000 class-stratified shared union-image cluster draws across seeds, seed-specific observed ID/subset masks',
            promotion=False,independent_confirmation='NO_FRESH_DATA'),runtime=dict(numpy=np.__version__,python=sys.version.split()[0]))


def inventory(m):
    rows=[]
    for path,h in m['inputs'].items():
        p=Path(path);schema=None
        if p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as f:schema={k:dict(shape=list(f[k].shape),dtype=str(f[k].dtype)) for k in f.files}
        rows.append(dict(path=p.as_posix(),absolute_path=str(p.resolve()),bytes=p.stat().st_size,sha256=h,access='READABLE',schema=schema))
    p=ROOT/'INPUT_INVENTORY.json';jsonwrite(p,dict(source_commit=m['source_commit'],role=ROLE,files=rows))
    out=ROOT/'DATA_PROVENANCE.md';atomic(out,'# G1 residual v7 provenance\n\n'+ROLE+'\n\n'
        'Reuse exact v6 B caches and v5 C caches with original ordered prototype SHA-256. No new head/backbone/IPS training. '
        'The three head seeds reuse 1199 images and a shared frozen ERM* backbone. Original official labels and known species-label caveat retained. '
        'B groups are never loaded. All B normalizers/ridge/head effects/scales respect nested training IDs. '
        'C labels/groups are evaluation-only; old C cannot select settings or confirm generalization. '
        'Inventory schemas are read only after EXPERIMENT_LOCK.json freezes all methods, sources and rules.\n')
    return dict(files=len(rows),status='PASS'),[p,out]


def baseline(s,f):
    b=B(s,f);c=C(s,f);abc=json.loads((nest(s,f)/'ABC_manifest.json').read_text());h=json.loads((nest(s,f)/'head_manifest.json').read_text());raw=load(nest(s,f)/'member_predictions.npz');old=load(nest(s,f)/'G1_g50/predictions.npz')
    assert 'group' not in b
    for split,z in [('B',b),('C',c)]:
        assert z['sample_id'].tolist()==abc[split] and z['probs'].shape==z['distances'].shape==(len(z['sample_id']),15,2)
        np.testing.assert_array_equal(z['probs'],raw[split+'_probs']);np.testing.assert_array_equal(z['distances'],raw[split+'_distances']);np.testing.assert_array_equal(z['head_sha256'],raw['head_hashes'])
    assert not(set(abc['A'])&set(abc['B']) or set(abc['A'])&set(abc['C']) or set(abc['B'])&set(abc['C']))
    for j,row in enumerate(h['ordered_heads']):assert row['slot']==j and sha(Path(row['path']))==row['sha256']==str(b['head_sha256'][j])
    rows=[];paths=[]
    for method,key in [('U0','uniform'),('G1_frozen','probs')]:
        out=infer(c['probs'],c['distances'],None,c['head_sha256'],method)
        np.testing.assert_array_equal(out['probs'],old[key]);np.testing.assert_array_equal(out['probs'].argmax(1),old[key].argmax(1))
        if method=='G1_frozen':np.testing.assert_array_equal(out['weights'],old['weights'])
        zero=infer(c['probs'],c['distances'],None,c['head_sha256'],'G1_RH',lambda_value=0)
        np.testing.assert_array_equal(zero['probs'],old['probs']);np.testing.assert_array_equal(zero['weights'],old['weights'])
        p=ROOT/f'checkpoints/baseline/seed{s}/fold{f}/{method}.npz';save_npz(p,**out,sample_id=c['sample_id'],head_sha256=c['head_sha256'],role=np.array(ROLE));paths.append(p)
        rows.append(dict(seed=s,fold=f,method=method,predictions_bit_exact=True,max_error=0.,role=ROLE))
    report=ROOT/f'checkpoints/baseline/seed{s}/fold{f}/audit.json';jsonwrite(report,rows)
    return dict(seed=s,fold=f,baseline='BIT_EXACT',lambda0='BIT_EXACT'),paths+[report]


def baseline_gate():
    from .metrics import metric
    original=json.loads(Path('outputs/geometry_aggregation_v5/run1/promotion_decisions.json').read_text())['decisions'][0]['seeds'];rows=[]
    for s in range(3):
        labels=[C(s,f,True) for f in range(3)];ids=np.concatenate([x['sample_id'] for x in labels]);assert len(set(ids))==len(ids)==1199
        y=np.concatenate([x['y'] for x in labels]);group=np.concatenate([x['group'] for x in labels]);assert [int((group==g).sum()) for g in range(4)]==[467,466,133,133]
        for method in METHODS[:2]:
            probs=np.concatenate([load(ROOT/f'checkpoints/baseline/seed{s}/fold{f}/{method}.npz',['probs'])['probs'] for f in range(3)])
            r=metric(probs,y,group);old=next(v for v in original if v['seed']==s)
            for key in ['AA','WGA']:np.testing.assert_allclose(r[key],old[('uniform_' if method=='U0' else '')+key],atol=1e-12,rtol=0)
            rows.append(dict(seed=s,method=method,**r,predictions_bit_exact=True,role=ROLE))
    p=ROOT/'BASELINE_REPRODUCTION.csv';csvwrite(p,rows);return dict(status='PASS',seeds=3,ensembles=9),[p]


def splits(s,f):
    b=B(s,f);rows=[]
    for i,(a,v) in enumerate(outer_splits(b,s,f)):
        assert not set(b['sample_id'][a])&set(b['sample_id'][v]);rows.append(dict(inner=i,train_ids=b['sample_id'][a].tolist(),validation_ids=b['sample_id'][v].tolist(),train_indices=a.tolist(),validation_indices=v.tolist()))
    p=ROOT/f'checkpoints/seed{s}/fold{f}/B_INNER_SPLITS.json';jsonwrite(p,dict(seed=71000+100*s+f,stratification='class only',folds=rows));return dict(seed=s,fold=f,images=len(b['y'])),[p]


def secondary(s,f,part,k,dest=None):
    b=B(s,f);a,v=secondary_splits(b,s,f,part)[k];train=train_indices(b,s,f,part)
    assert set(a)|set(v)==set(train) and not set(a)&set(v)
    model=fit_base(b['probs'][a],b['distances'][a],b['y'][a],b['sample_id'][a],b['head_sha256']);out=advantage(b['probs'][v],b['distances'][v],model,b['head_sha256'])
    dest=Path(dest) if dest is not None else folder(s,f,part)/f'scale_subcv/sub{k}'
    p=dest/'model.json';jsonwrite(p,model);npz=dest/'OOF_predictions.npz';save_npz(npz,pooled=out['pooled'],head=out['head'],sample_id=b['sample_id'][v],head_sha256=b['head_sha256'],role=np.array(ROLE))
    prov=dest/'split.json';jsonwrite(prov,dict(role=ROLE,seed=s,fold=f,part=part,secondary=k,train_ids=model['fit_ids'],validation_ids=b['sample_id'][v].tolist(),
        train_ID_sha256=digest(model['fit_ids']),validation_ID_sha256=digest(b['sample_id'][v].tolist()),B_source_sha256=sha(bpath(s,f)),class_only_seed=72000+1000*s+100*f+10*part))
    return dict(seed=s,fold=f,part=part,secondary=k,train_images=len(a),validation_images=len(v)),[p,npz,prov]


def model_fit(s,f,part):
    b=B(s,f);train=train_indices(b,s,f,part);model=fit_base(b['probs'][train],b['distances'][train],b['y'][train],b['sample_id'][train],b['head_sha256'])
    pieces=[load(folder(s,f,part)/f'scale_subcv/sub{k}/OOF_predictions.npz') for k in range(3)];ids=np.concatenate([x['sample_id'] for x in pieces]);idx={v:i for i,v in enumerate(ids)}
    assert len(ids)==len(set(ids))==len(train) and set(ids)==set(b['sample_id'][train]);order=[idx[v] for v in b['sample_id'][train]]
    estimates={k:np.concatenate([x[k] for x in pieces])[order] for k in ['pooled','head']}
    model=attach_scales(model,estimates['pooled'],estimates['head'],b['probs'][train],b['distances'][train],b['y'][train])
    dest=folder(s,f,part);p=dest/'model.json';jsonwrite(p,model);out=dest/'B_scale_OOF.npz';save_npz(out,**estimates,sample_id=b['sample_id'][train],role=np.array(ROLE))
    cells=dest/'head_effects.csv';csvwrite(cells,[dict(**c,role=ROLE) for c in model['cells']])
    prov=dest/'model_provenance.json';jsonwrite(prov,dict(role=ROLE,seed=s,fold=f,part=part,train_ids=model['fit_ids'],head_sha256=model['head_sha256'],
        B_source_sha256=sha(bpath(s,f)),model_source_sha256=sha(ROOT/'model.py'),config_sha256=digest(CONFIG),
        validation_ids=[] if part==4 else b['sample_id'][outer_splits(b,s,f)[part][1]].tolist(),C_labels_used=False,
        scales_from_nested_secondary_OOF=True,secondary_model_sha256=[sha(dest/f'scale_subcv/sub{k}/model.json') for k in range(3)]))
    return dict(seed=s,fold=f,part=part,fit_images=len(train),scales=model['scales']),[p,out,cells,prov]


def prediction(s,f,part,m,dest=None):
    if part==4:data=C(s,f)
    else:
        b=B(s,f);valid=outer_splits(b,s,f)[part][1];data={k:b[k][valid] for k in ['probs','distances','sample_id']};data['head_sha256']=b['head_sha256']
    fitted=json.loads((folder(s,f,part)/'model.json').read_text());assert not set(fitted['fit_ids'])&set(data['sample_id'])
    out=infer(data['probs'],data['distances'],fitted,data['head_sha256'],m)
    p=Path(dest) if dest is not None else pred_path(s,f,part,m);save_npz(p,**out,sample_id=data['sample_id'],head_sha256=data['head_sha256'],role=np.array(ROLE))
    return dict(seed=s,fold=f,part=part,method=m,samples=len(data['sample_id']),labels_in_inference=False),[p]


def preserved(m):
    for path,h in m['protected_original_artifacts'].items():assert sha(Path(path))==h,'Protected old artifact changed: '+path
    for path,h in m['inputs'].items():assert sha(Path(path))==h,'Input changed: '+path
    p=ROOT/'metrics/PRESERVATION_AUDIT.json';jsonwrite(p,dict(role=ROLE,status='PASS',old_artifacts=len(m['protected_original_artifacts']),all_inputs_unchanged=True,production_unchanged=True,v6_NO_GO_unchanged=True))
    return dict(status='PASS',old_artifacts=len(m['protected_original_artifacts'])),[p]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['inventory','B_crossfit','exploratory_c','all'],default='all');parser.add_argument('--resume',action='store_true');parser.add_argument('--verify',action='store_true');parser.add_argument('--status',action='store_true');parser.add_argument('--max-tasks',type=int);args=parser.parse_args()
    if args.status:print((ROOT/'state.json').read_text());return
    with single_writer(ROOT),threadpool_limits(limits=1):
        try:m=manifest()
        except FileNotFoundError as exc:
            atomic(ROOT/'INPUT_REQUEST.md','# Missing exact v7 input\n\n'+str(exc)+'\n\nNo head training is authorized. Provide matching B/C caches and ordered original heads/manifests.\n');jsonwrite(ROOT/'state.json',dict(status='BLOCKED_INPUTS',blocker=str(exc),resume_command=COMMAND));raise
        for name in ['config.lock.json','EXPERIMENT_LOCK.json']:
            p=ROOT/name;value=CONFIG if name=='config.lock.json' else {k:m[k] for k in ['version','role','config','methods','primary','source_commit','source','inputs','decision_rules']}
            if p.exists():assert json.loads(p.read_text())==value,'Frozen lock mismatch: '+name
            else:jsonwrite(p,value)
        j=Journal(ROOT,m,args.max_tasks)
        if args.verify:
            for path,h in m['protected_original_artifacts'].items():assert sha(Path(path))==h
            print(json.dumps(dict(status='PASS',completed=len(j.records),new_scientific_tasks=0,all_receipt_and_input_hashes_verified=True,role=ROLE)));return
        for path,h in m['source'].items():
            p=ROOT/'source_snapshot'/path
            if not p.exists():atomic(p,Path(path).read_bytes())
            assert sha(p)==h
        try:
            from .recovery import store_acceptance,acceptance,reuse_secondary,reuse_C
            j.task('recovery_store',[],store_acceptance);j.task('inventory',['recovery_store'],lambda:inventory(m))
            for s in range(3):
                for f in range(3):j.task(f'baseline:{s}:{f}',['inventory'],lambda s=s,f=f:baseline(s,f))
            j.task('baseline_gate',m['planned_DAG']['baseline_gate'],baseline_gate)
            if args.phase=='inventory':j.status('BASELINE_READY','recovery_secondary / nested B crossfit');return
            j.task('recovery_secondary',['baseline_gate'],lambda:acceptance('secondary'))
            from .metrics import B_evaluate,B_tables,B_bootstrap,B_gate,C_evaluate,C_export,C_bootstrap,C_permutation,C_tables,oracle,report,complete
            for s in range(3):
                for f in range(3):
                    j.task(f'B_splits:{s}:{f}',['recovery_secondary'],lambda s=s,f=f:splits(s,f))
                    for part in range(4):
                        for k in range(3):
                            fn=lambda s=s,f=f,part=part,k=k:reuse_secondary(s,f,part,k)
                            name=subname(s,f,part,k);j.task(name,m['planned_DAG'][name],fn)
                        name=fitname(s,f,part);j.task(name,m['planned_DAG'][name],lambda s=s,f=f,part=part:model_fit(s,f,part))
                        for method in METHODS:
                            name=predname(s,f,part,method);j.task(name,m['planned_DAG'][name],lambda s=s,f=f,part=part,method=method:prediction(s,f,part,method))
                        name=f'B_eval:{s}:{f}:{part}';j.task(name,m['planned_DAG'][name],lambda s=s,f=f,part=part:B_evaluate(s,f,part))
            j.task('B_tables',m['planned_DAG']['B_tables'],B_tables)
            for batch in range(20):
                name=f'B_bootstrap:{batch}';j.task(name,m['planned_DAG'][name],lambda batch=batch:B_bootstrap(batch))
            j.task('B_gate',m['planned_DAG']['B_gate'],B_gate);decision=j.result('B_gate')
            j.current['B_decision']=decision;j.status('B_CROSSFIT_COMPLETE','Full B / predeclared exploratory C regardless of B gate',best_B_only_observation=decision)
            if args.phase=='B_crossfit':return
            for s in range(3):
                for f in range(3):
                    for k in range(3):
                        name=subname(s,f,4,k);j.task(name,m['planned_DAG'][name],lambda s=s,f=f,k=k:secondary(s,f,4,k))
                    name=fitname(s,f,4);j.task(name,m['planned_DAG'][name],lambda s=s,f=f:model_fit(s,f,4))
            j.task('recovery_C',m['planned_DAG']['recovery_C'],lambda:acceptance('C'))
            for s in range(3):
                for f in range(3):
                    for method in METHODS:
                        name=predname(s,f,4,method);j.task(name,m['planned_DAG'][name],lambda s=s,f=f,method=method:reuse_C(s,f,method))
            def inference_marker():
                p=ROOT/'phases/C_inference_complete.json';jsonwrite(p,dict(role=ROLE,status='COMPLETE',labels_or_groups_used=False));return dict(status='COMPLETE'),[p]
            j.task('C_inference_complete',m['planned_DAG']['C_inference_complete'],inference_marker)
            for s in range(3):
                for method in METHODS:
                    name=f'C_eval:{s}:{method}';j.task(name,m['planned_DAG'][name],lambda s=s,method=method:C_evaluate(s,method))
            j.task('C_export',m['planned_DAG']['C_export'],C_export)
            for batch in range(20):
                name=f'C_bootstrap:{batch}';j.task(name,m['planned_DAG'][name],lambda batch=batch:C_bootstrap(batch))
            for batch in range(60):
                name=f'C_permutation:{batch}';j.task(name,m['planned_DAG'][name],lambda batch=batch:C_permutation(batch))
            j.task('C_tables',m['planned_DAG']['C_tables'],C_tables);j.task('oracle',['C_export'],oracle)
            j.task('report',m['planned_DAG']['report'],report);j.task('preservation',['report'],lambda:preserved(m));j.task('complete',['preservation'],complete)
            j.status('DONE_EXPLORATORY','Read EXPLORATORY_C_REPORT.md; no tuning/promotion; independent confirmation blocked',active_task=None,active_phase='complete',B_decision=decision,report=str(ROOT/'EXPLORATORY_C_REPORT.md'))
            print(json.dumps(dict(status='DONE_EXPLORATORY',completed=len(j.records),new_scientific_tasks=j.count,B_decision=decision['status'],independent_confirmation='INDEPENDENT_CONFIRMATION_NOT_PERFORMED')))
        except InterruptedError:print('Stopped at durable boundary; resume: '+COMMAND)
        except (AssertionError,ValueError) as exc:
            atomic(ROOT/'BLOCKERS.md','# Integrity or invariant failure\n\n'+str(exc)+'\n\nPreserve every receipt; do not change fixed methods or fabricate results. Investigate the failed task.\n');j.status('INTEGRITY_FAILURE',j.current.get('next_pending'),error=str(exc));raise


if __name__=='__main__':main()
