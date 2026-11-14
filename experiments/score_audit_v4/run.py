"""Crash-safe bounded score audit. python -m experiments.score_audit_v4.run --phase all --resume"""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import numpy as np
from scipy.special import softmax
from experiments.discrete_weighting.store import atomic,dump,sha
from .journal import Journal,csvwrite
from .model import MAPS,fit_representation,representation,fit_knots,fit_lookup,infer,observable,magnitude
from .evaluate import summary,evaluate

ROOT=Path('outputs/score_audit_v4/run1')
PRIOR=Path('outputs/wga_verification_v3/ce66e4448807')
COMMAND='.venv/Scripts/python.exe -m experiments.score_audit_v4.run --phase all --resume'
PROTOCOL=dict(maps=['R1','R2','R3','R4','R5','R6'],knots=MAPS,alphas=[.1,.25],betas=[.5,1.,2.],kappa=50,
    screen=dict(alpha=.1,beta=1.),selection='mean across seeds of worst-class NLL, then mean NLL, then map ID/alpha/beta; each seed delta AA strictly > -1pp',
    shortlist='top two eligible maps, full six alpha/beta grid, lock ONE globally best recipe before groups/outer evaluation',
    bootstrap=dict(per_output=500,pooled_gate=2000,seed=24019),outer_seeds=[0,1,2],outer_folds=3,
    promotion=dict(mean_delta_WGA_positive=True,positive_full_seeds_at_least=2,each_delta_AA_gt_minus1=True),
    R6_representation='raw',optional_ablations=False,outer_reuse='identical frozen v3 heads and A/B/C splits; previously inspected development, not untouched confirmation')


def save_npz(path,**arrays):
    b=io.BytesIO();np.savez_compressed(b,**arrays);atomic(path,b.getvalue())
    with np.load(path,allow_pickle=False) as f:
        for k,v in arrays.items(): np.testing.assert_array_equal(f[k],v)


def load(path,keys=None):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in (f.files if keys is None else keys)}


def locate(name):
    matches=[]
    for base,dirs,files in os.walk('.'):
        dirs[:]=[d for d in dirs if d not in ('.git','.venv','waterbird_complete95_forest2water2','node_modules','source_snapshot')]
        if name in files:matches.append(Path(base)/name)
    # Explicitly include handoff/verification inputs, which must not be pruned.
    for base in [Path('verification_package/inputs'),Path('baselines/dpe4subpop/inputs')]:
        if base.exists():matches.extend(base.rglob(name))
    matches=list(dict.fromkeys(matches))
    if not matches:raise FileNotFoundError('Missing source '+name+' (recursive workspace/verification_package/inputs search)')
    if len({sha(p) for p in matches})!=1:raise ValueError('Ambiguous different source copies: '+str(matches))
    return sorted(matches,key=lambda p:(len(p.parts),str(p)))[0]


def worker(root,pause):
    """Forced-kill acceptance worker, isolated from research output markers."""
    j=Journal(root,dict(acceptance='v4 kill/restart stable manifest'))
    def first():
        p=Path(root)/'first.txt';atomic(p,'durable completed output');return {},[p]
    j.task('first',[],first)
    def second():
        p=Path(root)/'second.txt';atomic(p,'atomic unfinished stage')
        if pause:
            atomic(Path(root)/'pause.ready','ready');time.sleep(120)
        return {},[p]
    j.task('second',['first'],second)


class Workflow:
    def __init__(self,args):
        self.args=args;self.root=ROOT
        self.devzip=locate('development_oof_bundle.zip');self.histzip=locate('prediction_bundle.zip')
        self.handoff=locate('DPE_WGA_Score_Audit_Piecewise_Codex_Handoff_v4.md')
        guarded=list(Path('experiments/score_audit_v4').glob('*.py'))+[Path('tests/methods/test_score_audit_v4.py'),Path('experiments/weighting/data.py'),Path('experiments/weighting/methods.py'),Path('experiments/weighting/gates.py'),Path('experiments/discrete_weighting/store.py'),Path('experiments/wga_verification/nested_heads.py')]
        source={p.as_posix():sha(p) for p in sorted(guarded)}
        inputs={str(p):sha(p) for p in [self.devzip,self.histzip,self.handoff,Path('outputs/weighting/features/val.npz'),PRIOR/'continuation_manifest.r2.json']}
        for seed in range(3):
            # D19 checkpoint, source experts/distances and final-head identity are guarded too.
            historical=Path(f'outputs/weighting/prediction_bundle/prediction_cache_seed{seed}.npz')
            with np.load(historical,allow_pickle=False) as f:prov=json.loads(str(f['provenance']))
            for p in [historical,Path(prov['original_expert_cache']),Path('outputs/weighting/fits')/prov['fit_key']/'model.joblib',Path(f'outputs/weighting/seed{seed}/heads_final.pt')]:inputs[str(p)]=sha(p)
            for fold in range(3):
                folder=PRIOR/f'nested/seed{seed}/fold{fold}'
                for p in [folder/'split.json',folder/'predictions.npz']+list(folder.glob('head*/head.pt')):inputs[str(p)]=sha(p)
        parent=ROOT/'history/manifest.r2.json'
        self.j=Journal(ROOT,dict(protocol=PROTOCOL,source=source,inputs=inputs,version=3,parent_manifest_sha256=sha(parent) if parent.exists() else None,
            revision_reason='Allow fresh initialization without a pre-existing lineage history; scoring, selection, data, splits and all experiment results unchanged.'),args.max_tasks)
        for p,h in source.items():
            dest=ROOT/'source_snapshot_r3'/p
            if not dest.exists():atomic(dest,Path(p).read_bytes())
            assert sha(dest)==h
        if not (ROOT/'PROTOCOL_DECISIONS.md').exists():
            atomic(ROOT/'PROTOCOL_DECISIONS.md', '# Preregistered v4 decisions\n\n'+json.dumps(PROTOCOL,indent=2)+'\n\n'
                '1. Reuse all 135 actual v3 heads only after checking per-head fit IDs, hashes, final epoch 20, independent deterministic splits and prediction identity. No LCMSW weights or models are reused. Epoch checkpoints remain in their original location. This saves training and preserves evidence.\n'
                '2. Original nested C outcomes were inspected for v3. These are nested engineering comparisons on reused development data, not an untouched confirmation. Method maps come verbatim from user v4; no adaptation to C outcomes.\n'
                '3. Fit R4 shared NLL temperature by deterministic bounded minimization in [0.001,10]; R6 uses raw probabilities. Lookup pools expert slots across folds and does not identify heads. No optional ablations.\n'
                '4. Select exactly one primary recipe by the group-blind proxy, with per-seed AA floor. No outer WGA selection between recipes. Diagnostic oracle never ranks. Test data only enters Phase 0 module.\n'
                '5. Fixed raw/calibrated Phase 0 edges and 15-bin ensemble ECE; uncertainty resamples unique images within true group after lock. Across seeds use identical image resamples, not 3597 independent rows.\n'
                '6. Phase 0 uses exact saved gate forward logits, verified against cached direct-softmax weights; missing OOF fold uses -1, meaning final development ensemble. No D19 retraining.\n'
                '7. Every task has a checksum commit; model/scaler/table and each inference export are separate tasks. Kill/restart of a v4 task and the prior real epoch-1 head recovery are verified before research evaluation.\n')

    def acceptance(self):
        def work():
            folder=ROOT/'acceptance/kill_test';cmd=[sys.executable,'-m','experiments.score_audit_v4.run','--acceptance-worker',str(folder)]
            if not (folder/'completed').exists() or len(list((folder/'completed').glob('*.json')))<2:
                env=os.environ|{'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
                proc=subprocess.Popen(cmd+['--pause-worker'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=env)
                try:
                    for _ in range(600):
                        if (folder/'pause.ready').exists():break
                        if proc.poll() is not None:raise RuntimeError('Acceptance subprocess exited unexpectedly')
                        time.sleep(.05)
                    assert (folder/'pause.ready').exists()
                finally:
                    if proc.poll() is None:proc.kill()
                    proc.wait(timeout=10)
                assert len(list((folder/'completed').glob('*.json')))==1
                stamp=(folder/'first.txt').stat().st_mtime_ns;digest=sha(folder/'first.txt')
                subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,env=env)
                assert (folder/'first.txt').stat().st_mtime_ns==stamp and sha(folder/'first.txt')==digest
            # Original epoch-by-epoch real head recovery evidence is immutable and reused.
            old=PRIOR/'acceptance/real_head_recovery.json';record=json.loads(old.read_text());assert record['status']=='PASS' and record['identical_to_uninterrupted']
            path=ROOT/'acceptance/recovery.json';dump(path,dict(status='PASS',forced_process_kill=True,partial_task_resumed=True,completed_task_reused_without_rewrite=True,
                real_head_epoch_recovery_source=str(old),real_head_epoch_recovery_sha256=sha(old),real_head_resumed_epoch=1))
            return dict(recovery='PASS'),[path]
        self.j.task('acceptance',[],work)

    def inputs(self):
        def work():
            files=[];inventory=[]
            for archive,dest in [(self.devzip,ROOT/'inputs/development'),(self.histzip,ROOT/'inputs/historical')]:
                with zipfile.ZipFile(archive) as z:
                    assert z.testzip() is None
                    for info in z.infolist():
                        name=Path(info.filename)
                        if name.is_absolute() or '..' in name.parts or ':' in info.filename:raise ValueError('Unsafe zip path')
                        if info.is_dir():continue
                        relative=Path(*name.parts[1:]) if name.parts[0] in ('development_bundle','prediction_bundle') else name
                        p=dest/relative;atomic(p,z.read(info));files.append(p);inventory.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
            dev=ROOT/'inputs/development';manifest=json.loads((dev/'manifest.json').read_text())
            for r in manifest['files']:assert sha(dev/r['path'])==r['sha256']
            for p,h in manifest['sources'].items():assert sha(p)==h
            checks=[]
            for seed in range(3):
                train=load(dev/f'seed{seed}/meta_train.npz');select=load(dev/f'seed{seed}/meta_select.npz')
                assert not set(train['sample_id'])&set(select['sample_id'])
                prov=json.loads((dev/f'seed{seed}/fold_provenance.json').read_text())
                for d in (train,select):
                    from .model import validate
                    validate(d['probs']);assert d['embedding'].shape==(len(d['y']),2048);np.testing.assert_array_equal(d['classes'],[0,1]);assert len(set(d['sample_id']))==len(d['y'])
                    np.testing.assert_allclose(softmax(d['logits'],axis=-1),d['probs'],atol=2e-6)
                    assert np.all(d['group']//2==d['y'])
                    for f in prov['folds']:
                        ids=set(d['sample_id'][d['fold']==f['fold']]);assert ids<=set(f['held_ids']) and not ids&set(f['fit_ids'])
                        assert sha(f"outputs/weighting/seed{seed}/heads_fold{f['fold']}.pt")==f['heads_sha256']
                checks.append(dict(seed=seed,train=len(train['y']),select=len(select['y']),raw_train_max=float(train['probs'].max()),raw_ge075=int((train['probs'].max(-1)>=.75).sum()),provenance='PASS'))
            p=ROOT/'inputs/verification.json';dump(p,dict(status='PASS',inventory=inventory,seeds=checks,group_access='Input integrity only; fitting/selecting load named arrays excluding group.'))
            return dict(inputs='PASS',raw075_empty=all(r['raw_ge075']==0 for r in checks)),files+[p]
        self.j.task('inputs',['acceptance'],work)

    def phase0(self):
        from . import audit
        tasks=[]
        for seed in range(3):
            t=f'phase0:expert_export:{seed}';self.j.task(t,['inputs'],lambda seed=seed:audit.export_seed(ROOT,seed,ROOT/'inputs/historical'));tasks.append(t)
            for rep in ['raw','calibrated']:
                a=f'phase0:aggregate:{seed}:{rep}';self.j.task(a,[t],lambda seed=seed,rep=rep:audit.aggregate(ROOT,seed,rep));tasks.append(a)
            a=f'phase0:image_plots:{seed}';self.j.task(a,[t],lambda seed=seed:audit.image_plots(ROOT,seed));tasks.append(a)
        self.j.task('phase0:report',tasks,lambda:audit.report(ROOT))

    def dev(self,seed,split):
        return load(ROOT/f'inputs/development/seed{seed}/{split}.npz',['probs','logits','y','sample_id','fold'])

    def fit(self,prefix,map_id,train,deps):
        folder=ROOT/prefix.replace(':','/')
        def transform():
            rep=fit_representation(map_id,train['logits'],train['y']);p=representation(rep,train['probs'],train['logits'])
            path=folder/'representation.json';dump(path,rep);cache=folder/'fit_representation.npz';save_npz(cache,probs=p,sample_id=train['sample_id'])
            return dict(map_id=map_id,temperature=rep['temperature']),[path,cache]
        self.j.task(prefix+':representation',deps,transform)
        rep=json.loads((folder/'representation.json').read_text());p=load(folder/'fit_representation.npz')['probs']
        def bins():
            knots=fit_knots(map_id,p);lookup=fit_lookup(p,train['y'],train['sample_id'],knots)
            path=folder/'lookup.json';dump(path,lookup);table=folder/'lookup_cells.csv';csvwrite(table,lookup['cells'])
            coverage=[];pred,conf,_=observable(p);correct=pred==train['y'][:,None]
            for k in (0,1):
                for lo,hi in zip([.5,.515,.53,.54,.55,.6,.75,.95],[.515,.53,.54,.55,.6,.75,.95,1.000001]):
                    mask=(pred==k)&(conf>=lo)&(conf<hi);n=int(mask.sum())
                    coverage.append(dict(predicted_class=k,lower=lo,upper=min(1,hi),observations=n,distinct_images=int(mask.any(1).sum()),
                        correctness=float(correct[mask].mean()) if n else None,wrong_observations=int((mask&~correct).sum()),
                        distinct_wrong_images=int((mask&~correct).any(1).sum()),low_support=int(mask.any(1).sum())<30,
                        temperature=rep['temperature'],mean_confidence=float(conf[pred==k].mean()) if (pred==k).any() else None))
            cpath=folder/'confidence_coverage.csv';csvwrite(cpath,coverage)
            return dict(map_id=map_id,training_images=len(p)),[path,table,cpath]
        self.j.task(prefix+':lookup',[prefix+':representation'],bins)
        return folder,rep,json.loads((folder/'lookup.json').read_text())

    def candidate(self,stage,map_id,alpha,beta,seed,folder,rep,lookup):
        name=f'{stage}:{map_id}:a{alpha:g}:b{beta:g}:s{seed}';out=ROOT/'phase1/candidates'/name.replace(':','_')
        def predict():
            d=self.dev(seed,'meta_select');p=representation(rep,d['probs'],d['logits']);mix,w,s=infer(p,lookup,alpha,beta)
            path=out/'predictions.npz';save_npz(path,mixture=mix,uniform=p.mean(1),raw_uniform=d['probs'].mean(1),members=p,weights=w,scores=s,sample_id=d['sample_id'])
            r=summary(mix,d['y']);u=summary(p.mean(1),d['y']);delta=(r['AA']-u['AA'])*100
            r.update(stage=stage,map_id=map_id,alpha=alpha,beta=beta,seed=seed,delta_AA_pp=delta,eligible=delta>-1.,uniform_AA=u['AA'])
            table=out/'selection_metrics.json';dump(table,r);return r,[path,table]
        r=self.j.task(name,[f'phase1fit:{map_id}:{seed}:lookup'],predict)
        return dict(r,task=name,folder=str(out))

    def phase1(self):
        fits={};screen=[]
        for map_id in PROTOCOL['maps']:
            for seed in range(3):
                fits[map_id,seed]=self.fit(f'phase1fit:{map_id}:{seed}',map_id,self.dev(seed,'meta_train'),['phase0:report'])
                folder,rep,lookup=fits[map_id,seed];screen.append(self.candidate('screen',map_id,.1,1.,seed,folder,rep,lookup))
        def rank(rows):
            keys=sorted({(r['map_id'],r['alpha'],r['beta']) for r in rows});ranked=[]
            for mid,a,b in keys:
                part=[r for r in rows if (r['map_id'],r['alpha'],r['beta'])==(mid,a,b)]
                ranked.append(dict(map_id=mid,alpha=a,beta=b,eligible=all(r['eligible'] for r in part),
                    proxy=float(np.mean([r['worst_class_NLL'] for r in part])),mean_NLL=float(np.mean([r['NLL'] for r in part])),
                    delta_AA_pp_by_seed=[r['delta_AA_pp'] for r in sorted(part,key=lambda r:r['seed'])]))
            return sorted(ranked,key=lambda r:(not r['eligible'],r['proxy'],r['mean_NLL'],r['map_id'],r['alpha'],r['beta']))
        def shortlist():
            ranked=rank(screen);maps=[r['map_id'] for r in ranked if r['eligible']][:2]
            path=ROOT/'phase1_screen_shortlist.json';dump(path,dict(maps=maps,ranking=ranked,groups_accessed=False));return dict(maps=maps),[path]
        short=self.j.task('phase1:shortlist',[r['task'] for r in screen],shortlist)['maps']
        grid=[]
        for mid in short:
            for a in PROTOCOL['alphas']:
                for b in PROTOCOL['betas']:
                    for seed in range(3):
                        folder,rep,lookup=fits[mid,seed];grid.append(self.candidate('grid',mid,a,b,seed,folder,rep,lookup))
        def lock():
            ranked=rank(grid);eligible=[r for r in ranked if r['eligible']];winner=eligible[0] if eligible else None
            path=ROOT/'phase1_locked_candidates.json';dump(path,dict(primary=winner,all_ranked=ranked,group_blind=True,groups_accessed=False,
                config_hash=sha(ROOT/'manifest.json'),selection=PROTOCOL['selection'],test_used=False,one_primary_only=True))
            table=ROOT/'phase1_ranked_by_predeclared_proxy.csv';csvwrite(table,ranked)
            return dict(primary=winner),[path,table]
        locked=self.j.task('phase1:lock',['phase1:shortlist']+[r['task'] for r in grid],lock)
        # All group metrics are first accessed AFTER the lock marker above.
        diag=[]
        for row in screen+grid:
            task='phase1diagnostic:'+row['task'];diag.append(task)
            def diagnostic(row=row):
                folder=Path(row['folder']);d=self.dev(row['seed'],'meta_select');group=load(ROOT/f"inputs/development/seed{row['seed']}/meta_select.npz",['group'])['group'];z=load(folder/'predictions.npz')
                metrics,flips=evaluate(z['mixture'],z['uniform'],z['raw_uniform'],z['members'],z['weights'],z['scores'],d['y'],group,d['sample_id'])
                metrics.update({k:row[k] for k in ['stage','map_id','alpha','beta','seed']})
                path=folder/'post_lock_diagnostics.json';dump(path,metrics);fp=folder/'corrected_broken.csv';csvwrite(fp,flips)
                from .visuals import score_cells
                cells=score_cells(z['members'],z['scores'],d['y'],d['sample_id'])
                cp=folder/'heldout_score_correctness.csv';csvwrite(cp,cells)
                return metrics,[path,fp,cp]
            self.j.task(task,['phase1:lock',row['task']],diagnostic)
        def report():
            rows=[self.j.records[t]['result'] for t in diag]
            # R0 represented separately for each map/seed temperature, plus raw baseline.
            controls=[]
            for mid in PROTOCOL['maps']:
                for seed in range(3):
                    d=self.dev(seed,'meta_select');g=load(ROOT/f'inputs/development/seed{seed}/meta_select.npz',['group'])['group'];rep=fits[mid,seed][1]
                    controls.append(dict(stage='control',map_id='R0_matched_'+mid,alpha=0,beta=0,seed=seed,**summary(representation(rep,d['probs'],d['logits']).mean(1),d['y'],g)))
            path=ROOT/'phase1_all.csv';csvwrite(path,rows+controls)
            winner=locked['primary'];text='# Phase 1 — exploratory group-blind selection\n\n'
            text+='Six maps screened at alpha=.10/beta=1; two maps shortlisted by mean worst-class NLL, then six alpha/beta combinations per map. Per-seed AA decline must be strictly below 1 pp. No group WGA enters ranking.\n\nLocked primary: '+json.dumps(winner,indent=2)+'\n\n'
            text+='All losing candidates and matched representation-specific uniform controls appear in phase1_all.csv. Raw confidence >=.75 is empty in all training seeds; coverage tables give distinct image counts, wrong-tail counts and all transformed scales. R4 fits shared temperature on original logits once, using meta_train member NLL only. R5 quantiles and all score bins/tables fit meta_train only.\n\n'
            text+='Group metrics were computed only after phase1:lock. meta_select has prior exploratory inspection and OOF heads differ across folds; pooled lookup has no expert identity. Confidence intervals resample distinct images within subgroup, conditional on fitted pipelines; no 15-fold sample-size claim. Seed splits are not pooled for fitting or significance.\n'
            rpath=ROOT/'phase1_report.md';atomic(rpath,text)
            from .visuals import phase1_figures
            plots=phase1_figures(ROOT,winner) if winner else []
            return dict(phase1='COMPLETE',primary=winner,candidate_seed_evaluations=len(rows)),[path,rpath]+plots
        self.j.task('phase1:report',diag,report)
        return locked['primary']

    def phase2(self,primary):
        if primary is None:
            def no_winner():
                p=ROOT/'promotion_gate.json';dump(p,dict(status='FAIL',reason='No eligible Phase 1 recipe',production_integration=False));return dict(gate='FAIL'),[p]
            return self.j.task('phase2:gate',['phase1:report'],no_winner)
        import torch
        from sklearn.model_selection import StratifiedKFold,train_test_split
        from experiments.weighting.data import predict_heads
        from experiments.wga_verification.nested_heads import SETTINGS
        val=load('outputs/weighting/features/val.npz',['embedding','y','sample_id'])
        mid,a,b=primary['map_id'],primary['alpha'],primary['beta'];tasks=[]
        for seed in range(3):
            outer=StratifiedKFold(3,shuffle=True,random_state=seed)
            for fold,(ab,c) in enumerate(outer.split(val['embedding'],val['y'])):
                ai,bi=train_test_split(ab,train_size=.7,stratify=val['y'][ab],random_state=7000+seed*31+fold)
                source=PRIOR/f'nested/seed{seed}/fold{fold}';folder=ROOT/f'phase2/seed{seed}/fold{fold}';prefix=f'outer:{seed}:{fold}'
                def reuse(source=source,folder=folder,ai=ai,bi=bi,c=c,seed=seed,fold=fold):
                    old=json.loads((source/'split.json').read_text());assert old['head_settings']==SETTINGS
                    for label,indices in [('head_indices',ai),('weight_indices',bi),('hold_indices',c)]:np.testing.assert_array_equal(old[label],indices)
                    sets=[set(val['sample_id'][idx]) for idx in (ai,bi,c)];assert all(not sets[x]&sets[y] for x in range(3) for y in range(x+1,3))
                    heads=[];refs=[]
                    for j in range(15):
                        path=source/f'head{j:02d}/head.pt';h=torch.load(path,weights_only=True);assert set(h['fit_ids'])==sets[0] and h['selected_epoch']==20 and h['member_id']==f'head-{j:02d}'
                        assert (path.parent/'latest.pt').exists();heads.append(h);refs.append(dict(expert=j,path=str(path),sha256=sha(path),epoch_checkpoint=str(path.parent/'latest.pt'),epoch_sha256=sha(path.parent/'latest.pt')))
                    oldp=load(source/'predictions.npz');np.testing.assert_array_equal(oldp['member_hashes'],[r['sha256'] for r in refs]);np.testing.assert_array_equal(oldp['weight_ids'],val['sample_id'][bi]);np.testing.assert_array_equal(oldp['holdout_ids'],val['sample_id'][c])
                    plan=folder/'ABC_manifest.json';dump(plan,dict(seed=seed,fold=fold,head_settings=SETTINGS,heads=refs,A=val['sample_id'][ai].tolist(),B=val['sample_id'][bi].tolist(),C=val['sample_id'][c].tolist(),
                        head_indices=ai.tolist(),weight_indices=bi.tolist(),hold_indices=c.tolist(),groups_used_for_fit=False,previously_inspected=True,source_cache_sha256=sha(source/'predictions.npz')))
                    return dict(seed=seed,fold=fold,heads_reused=15,provenance='PASS'),[plan]
                self.j.task(prefix+':reuse',['phase1:report'],reuse)
                def cache(source=source,folder=folder,bi=bi,c=c):
                    h=[torch.load(source/f'head{j:02d}/head.pt',weights_only=True) for j in range(15)]
                    heads=dict(prototypes=torch.stack([x['prototypes'] for x in h]),scales=torch.stack([x['scale'] for x in h]))
                    bp=predict_heads(heads,val['embedding'][bi]);cp=predict_heads(heads,val['embedding'][c]);old=load(source/'predictions.npz')
                    np.testing.assert_array_equal(bp['probs'],old['weight_probs']);np.testing.assert_array_equal(cp['probs'],old['holdout_probs'])
                    path=folder/'member_predictions.npz';save_npz(path,B_probs=bp['probs'],B_logits=bp['logits'],C_probs=cp['probs'],C_logits=cp['logits'],B_ids=val['sample_id'][bi],C_ids=val['sample_id'][c],member_hashes=old['member_hashes'])
                    return dict(cached=True,uniform_exact_match=True),[path]
                self.j.task(prefix+':cache',[prefix+':reuse'],cache)
                z=load(folder/'member_predictions.npz');train=dict(probs=z['B_probs'],logits=z['B_logits'],y=val['y'][bi],sample_id=z['B_ids'])
                fitfolder,rep,lookup=self.fit(prefix+':fit',mid,train,[prefix+':cache'])
                def predict(folder=folder,z=z,rep=rep,lookup=lookup):
                    p=representation(rep,z['C_probs'],z['C_logits']);mix,w,s=infer(p,lookup,a,b)
                    path=folder/'outer_predictions.npz';save_npz(path,mixture=mix,uniform=p.mean(1),raw_uniform=z['C_probs'].mean(1),members=p,weights=w,scores=s,sample_id=z['C_ids'],member_hashes=z['member_hashes'])
                    return dict(predictions_committed_before_group_evaluation=True),[path]
                self.j.task(prefix+':predict',[prefix+':fit:lookup'],predict)
                def evaluate_fold(folder=folder,c=c,seed=seed,fold=fold):
                    d=load(folder/'outer_predictions.npz');g=load('outputs/weighting/features/val.npz',['group'])['group'][c]
                    r,flips=evaluate(d['mixture'],d['uniform'],d['raw_uniform'],d['members'],d['weights'],d['scores'],val['y'][c],g,d['sample_id'])
                    r.update(seed=seed,fold=fold,map_id=mid,alpha=a,beta=b,level='fold');path=folder/'evaluation.json';dump(path,r);fp=folder/'corrected_broken.csv';csvwrite(fp,flips)
                    # Non-deployable oracle only after predictions; never used in selection/promotion.
                    fitfolder=ROOT/(f'outer:{seed}:{fold}:fit'.replace(':','/'));lk=json.loads((fitfolder/'lookup.json').read_text())
                    signed=magnitude(d['members'],lk['knots'])*np.where(d['members'].argmax(-1)==val['y'][c,None],1.,-1.)
                    ow=(1-a)/15+a*softmax(b*signed,axis=1);oracle=(d['members']*ow[...,None]).sum(1);op=folder/'ORACLE_NON_DEPLOYABLE.json';dump(op,dict(label='ORACLE NON-DEPLOYABLE: held-out true labels used, excluded from ranking',**summary(oracle,val['y'][c],g)))
                    return r,[path,fp,op]
                task=prefix+':evaluate';self.j.task(task,[prefix+':predict'],evaluate_fold);tasks.append(task)
        pooled=[]
        for seed in range(3):
            def pool(seed=seed):
                idx={str(s):i for i,s in enumerate(val['sample_id'])};collected=[load(ROOT/f'phase2/seed{seed}/fold{f}/outer_predictions.npz') for f in range(3)]
                ids=np.concatenate([d['sample_id'] for d in collected]);assert len(set(ids))==len(ids)==len(val['y']);order=np.argsort([idx[str(s)] for s in ids])
                d={k:np.concatenate([v[k] for v in collected])[order] for k in ['mixture','uniform','raw_uniform','members','weights','scores','sample_id']}
                np.testing.assert_array_equal(d['sample_id'],val['sample_id']);g=load('outputs/weighting/features/val.npz',['group'])['group']
                path=ROOT/f'phase2/seed{seed}/pooled_predictions.npz';save_npz(path,**d)
                r,flips=evaluate(d['mixture'],d['uniform'],d['raw_uniform'],d['members'],d['weights'],d['scores'],val['y'],g,d['sample_id'],2000)
                r.update(seed=seed,fold=-1,map_id=mid,alpha=a,beta=b,level='seed');out=path.with_name('evaluation.json');dump(out,r);fp=path.with_name('corrected_broken.csv');csvwrite(fp,flips)
                return r,[path,out,fp]
            task=f'phase2:pool:{seed}';pooled.append(self.j.task(task,[t for t in tasks if t.startswith(f'outer:{seed}:')],pool))
        def gate():
            wga=float(np.mean([r['delta_WGA_pp'] for r in pooled]));aa=float(np.mean([r['delta_AA_pp'] for r in pooled]));positive=sum(r['delta_WGA_pp']>0 for r in pooled)
            checks=dict(mean_WGA_positive=wga>0,positive_WGA_at_least_2_seeds=positive>=2,AA_floor_every_seed=all(r['delta_AA_pp']>-1 for r in pooled),provenance=True)
            # Resample same unique image IDs across seeds; preserve subgroup sizes.
            g=load('outputs/weighting/features/val.npz',['group'])['group'];rng=np.random.default_rng(24019)
            cc=np.stack([load(ROOT/f'phase2/seed{s}/pooled_predictions.npz',['mixture'])['mixture'].argmax(1)==val['y'] for s in range(3)])
            uu=np.stack([load(ROOT/f'phase2/seed{s}/pooled_predictions.npz',['uniform'])['uniform'].argmax(1)==val['y'] for s in range(3)])
            pools=[np.flatnonzero(g==k) for k in range(4)];ds=[];das=[]
            for _ in range(2000):
                takes=[rng.choice(t,len(t),replace=True) for t in pools];ix=np.concatenate(takes)
                ds.append(float((np.stack([cc[:,t].mean(1) for t in takes]).min(0)-np.stack([uu[:,t].mean(1) for t in takes]).min(0)).mean()*100));das.append(float((cc[:,ix].mean(1)-uu[:,ix].mean(1)).mean()*100))
            r=dict(status='PASS' if all(checks.values()) else 'FAIL',criteria=checks,primary=primary,mean_delta_WGA_pp=wga,mean_delta_AA_pp=aa,positive_seeds=positive,
                WGA_CI95_pp=np.quantile(ds,[.025,.975]).tolist(),AA_CI95_pp=np.quantile(das,[.025,.975]).tolist(),
                qualification='Nested fixed-head engineering verification on previously inspected development; not untouched confirmation',seeds=pooled)
            path=ROOT/'promotion_gate.json';dump(path,r);table=ROOT/'phase2_nested_verification.csv';csvwrite(table,[self.j.records[t]['result'] for t in tasks]+pooled)
            return r,[path,table]
        return self.j.task('phase2:gate',[f'phase2:pool:{s}' for s in range(3)],gate)

    def finish(self,gate):
        def report():
            import pandas as pd
            q=pd.read_csv(ROOT/'phase1_all.csv');primary=json.loads((ROOT/'phase1_locked_candidates.json').read_text())['primary']
            lines=['# DPE v4 — score audit and piecewise reliability','',f'**Promotion gate: {gate["status"]}.**','',
                'WGA is primary; every seed must have ΔAA strictly greater than −1.00 pp. Selection is by predeclared class-only worst-class NLL on the exploratory meta_select, with a strict per-seed AA floor. Exactly one primary was locked before group/outer evaluation.', '',
                '## Nested results','', '| Seed | Uniform WGA % | Lookup WGA % | ΔWGA pp | Uniform AA % | Lookup AA % | ΔAA pp |','|---|---:|---:|---:|---:|---:|---:|']
            for r in gate.get('seeds',[]):lines.append(f'| {r["seed"]} | {r["uniform_WGA"]*100:.4f} | {r["WGA"]*100:.4f} | {r["delta_WGA_pp"]:+.4f} | {r["uniform_AA"]*100:.4f} | {r["AA"]*100:.4f} | {r["delta_AA_pp"]:+.4f} |')
            lines+=['',json.dumps({k:v for k,v in gate.items() if k not in ('seeds','primary')},indent=2),'',
                'Locked primary: '+json.dumps(primary), '',
                '## Interpretation and limits','',
                'All three seeds share one verified augmented 400-epoch ERM* backbone; these are not standard ERM results or independent backbone replicates. A/B/C are disjoint, bins/calibration/teacher lookup fit only B, exact same ordered actual heads predict B and C. All 135 heads were reused after per-head fit-ID and checkpoint-hash verification; original per-epoch optimizer/RNG checkpoints remain intact. Nine prediction stages recover logits from frozen heads and exactly reproduce cached raw probabilities.', '',
                'The v3 nested C outcomes and development pool had prior inspection. No C outcome entered this run’s selection, but this reuse cannot establish a scientifically untouched confirmation or remove researcher-level selection bias. Bootstrap intervals condition on fixed trained pipelines and exclude backbone/training/selection uncertainty. Same-image resamples align across seeds. No historical test is used to train or select.', '',
                'Raw confidence >=.75 has zero meta_train support. Confidence .75 is representation-dependent: R3 uses original logits at fixed T=.25, R4 fits shared member-NLL temperature. Coverage and sparse confident-wrong support are logged by predicted class in each fit folder. No temperature is applied twice. The lookup estimates E[signed h(confidence)|predicted class, confidence bin, LOO agreement bin] with kappa50 shrinkage; correctness sign is training-only. R6 isolates correctness without confidence magnitude.', '',
                'D19 Phase 0 reports exact relative-score ranking, wrong-expert weight and concentration with calibrated/raw uniform controls; see phase0_mechanism_report.md. It separates score misalignment from softmax concentration descriptively, without causal attribution. Piecewise scoring is deliberately mild (alpha .10/.25); actual pairwise ranking, wrong-head mass, entropy, effective members, corrected/broken directions and subgroup accuracies are saved for every candidate/seed. Oracle results are labeled non-deployable and excluded from all rankings.', '',
                '## Historical context (different protocols, not confirmation)','',
                'Historical test matched raw uniform: AA95.708%, WGA92.761%; continuous D19: AA95.961%, WGA83.645%; discrete D5: AA96.076%, WGA90.343%. Separate original group-aware DPE checkpoint: AA96.048%, WGA94.393%, one seed. None of these is a new v4 test result. LCMSW v3 source reproduction passed, but nested promotion failed: mean WGA+.2915pp, AA−.8340pp, only1/3 WGA-positive and seed1 AA−1.000834pp. LCMSW is not retuned here.', '',
                '## Exactly one next step','',
                'Evaluate the frozen primary on a newly reserved development cohort with subgroup annotations kept evaluation-only, using the same A/B/C protocol; preserve these results as exploratory evidence and do not retune from the old test.', '',
                '## Reproduce and resume','',f'`{COMMAND}`','',
                'Read state.json, HANDOFF_STATUS.md, NEXT_AGENT_PROMPT.md and completed/ checksums. Source/input/protocol hashes are guarded. A v4 task process was forcibly killed and resumed without rewriting a completed artifact; the prior real DPE head epoch-1 kill/resume tensor-equality test is also preserved. Every evaluation, fit representation, lookup and report has an atomic commit. No completed v1/v2/v3 tasks were retrained or overwritten.']
            lines.insert(4,'Production integration: '+('gate passed; opt-in implementation required before completion.' if gate['status']=='PASS' else 'NOT PERFORMED because the frozen gate failed.'))
            path=ROOT/'FINAL_REPORT.md';atomic(path,'\n'.join(lines)+'\n')
            return dict(gate=gate['status'],primary=primary),[path]
        self.j.task('final:report',['phase2:gate'],report)
        if gate['status']=='PASS':
            self.j.status('INTEGRATION_REQUIRED','Implement opt-in serialized score lookup under frozen primary',promotion_gate='PASS')
        else:self.j.status('COMPLETE_GATE_FAILED','None; preserve results, no production integration',promotion_gate='FAIL',production_integration='NOT_PERFORMED')
        print(json.dumps(dict(output=str(ROOT),gate=gate['status'],completed=len(self.j.records))),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all','0','1','2'],default='all');p.add_argument('--resume',action='store_true');p.add_argument('--status',action='store_true');p.add_argument('--max-tasks',type=int);p.add_argument('--acceptance-worker');p.add_argument('--pause-worker',action='store_true');a=p.parse_args()
    if a.acceptance_worker:return worker(a.acceptance_worker,a.pause_worker)
    if a.status:
        path=ROOT/'state.json';print(path.read_text() if path.exists() else 'NOT_STARTED');return
    from threadpoolctl import threadpool_limits
    import torch
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        w=Workflow(a)
        try:
            w.acceptance();w.inputs();w.phase0()
            if a.phase=='0':w.j.status('PHASE0_COMPLETE','phase1');return
            primary=w.phase1()
            if a.phase=='1':w.j.status('PHASE1_COMPLETE','phase2');return
            gate=w.phase2(primary);w.finish(gate)
        except InterruptedError:print('Stopped at durable boundary; resume using '+COMMAND)


if __name__=='__main__':main()
