"""Preregistered, crash-safe geometry aggregation workflow; no historical test reads."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import numpy as np
import torch
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold,train_test_split
from threadpoolctl import threadpool_limits
from .persistence import atomic,dump
from experiments.discrete_weighting.store import sha,key
from experiments.weighting.data import predict_heads
from .heads import fit_member,SETTINGS
from experiments.score_audit_v4.evaluate import summary
from .journal import Journal,csvwrite
from .io import load,save_npz,single_writer
from .model import configs,fit_normalizer,aggregate,validate
from .evaluate import assess,joint_ci

ROOT=Path('outputs/geometry_aggregation_v5/run1')
CMD='.venv/Scripts/python.exe -m experiments.geometry_aggregation_v5.run --phase all --resume'
PROTOCOL=dict(candidates=configs(),seeds=[0,1,2],outer_folds=3,head_fraction=.7,head_seed='1000+100*seed+fold',
    AB_split_seed='7000+31*seed+fold',head_training=SETTINGS,
    selection='Exclude any seed delta_AA<=-1pp; mean worst-class NLL ascending, mean worst-class accuracy descending, mean AA descending, AA SD ascending, stable config id.',
    locking='Primary best eligible non-U0/non-diagnostic; secondary best eligible of opposite shared-weight/class-wise family, or next distinct config if opposite family has none. Maximum2.',
    screening_budget='Specific narrow grid from repository detailed v5: shared methods alpha=.25; S4 tau=.1 and alpha .1/.25/.5; T_class1. No expansion.',
    promotion=dict(mean_delta_WGA_gt0=True,positive_seeds_at_least2=True,every_delta_AA_gt_minus1=True,provenance=True),
    integration='Newest pasted request governs: integrate first locked passing candidate, if any; never select an outer-WGA maximum.',
    historical_test='Never read',backbone='one frozen verified augmented400ep ERMstar',head_identity='fold plus actual checkpoint hashes, never slot alone',
    bootstrap=dict(audit=2000,screen=500,fold=500,pooled=2000,gate=2000))


def locate(name):
    matches=[]
    for base,dirs,files in os.walk('.'):
        dirs[:]=[d for d in dirs if d not in ('.git','.venv','waterbird_complete95_forest2water2','node_modules','source_snapshot')]
        if name in files:matches.append(Path(base)/name)
    if not matches:raise FileNotFoundError(name+' missing in recursive workspace search')
    if len({sha(p) for p in matches})!=1:raise ValueError('Ambiguous different source copies for '+name)
    return sorted(matches,key=lambda p:(len(p.parts),str(p)))[0]


def splits(val):
    for seed in range(3):
        for fold,(ab,c) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed).split(val['embedding'],val['y'])):
            a,b=train_test_split(ab,train_size=.7,stratify=val['y'][ab],random_state=7000+31*seed+fold)
            yield seed,fold,a,b,c


def acceptance_worker(pool,folder,pause):
    val=load(pool);torch.set_num_threads(1)
    with threadpool_limits(limits=1):fit_member(val['embedding'],val['y'],val['sample_id'],[],1000,0,folder,pause)


class Workflow:
    def __init__(self,args):
        self.args=args;self.root=ROOT;self.dev=ROOT/'inputs/development_bundle';self.archive=locate('development_oof_bundle.zip')
        source=list(Path('experiments/geometry_aggregation_v5').glob('*.py'))+[
            Path('experiments/discrete_weighting/store.py'),Path('experiments/weighting/data.py'),Path('experiments/wga_verification/nested_heads.py'),
            Path('experiments/score_audit_v4/evaluate.py'),Path('experiments/score_audit_v4/model.py'),Path('tests/methods/test_geometry_aggregation_v5.py'),
            Path('tests/methods/test_dpe_geometry_inference.py'),Path('baselines/dpe4subpop/src/dpe/core.py'),Path('baselines/dpe4subpop/src/dpe/geometry.py'),
            Path('baselines/dpe4subpop/src/dpe/verified_distance_v1.json')]
        hashes={str(p):sha(p) for p in sorted(source)}
        inputs={str(p):sha(p) for p in [self.archive,Path('experiments/geometry_aggregation_v5/USER_HANDOFF_v5.md'),Path('baselines/dpe4subpop/DPE_Geometry_Aggregation_Codex_Handoff_v5.md'),Path('outputs/weighting/features/val.npz')]}
        for seed in range(3):
            for fold in range(3):
                p=Path(f'outputs/weighting/seed{seed}/heads_fold{fold}.pt');inputs[str(p)]=sha(p)
        parent=ROOT/'history/manifest.r2.json'
        self.j=Journal(ROOT,dict(version=3,parent_manifest_sha256=sha(parent) if parent.exists() else None,
            revision_reason='Integrate only the passing frozen G1 recipe as opt-in distance mode; production acceptance added after completed gates; no research mathematics or completed results changed.',
            source=hashes,inputs=inputs,protocol=PROTOCOL,runtime=dict(python=sys.version.split()[0],numpy=np.__version__,torch=torch.__version__)),args.max_tasks)
        for p,h in hashes.items():
            dest=ROOT/'source_snapshot_r3'/p
            if not dest.exists():atomic(dest,Path(p).read_bytes())
            assert sha(dest)==h
        if not (ROOT/'PROTOCOL_DECISIONS.md').exists():atomic(ROOT/'PROTOCOL_DECISIONS.md', '# Preregistered decisions — DPE geometry v5\n\n'+json.dumps(PROTOCOL,indent=2)+'\n\n'
            'The user-pasted v5 and repository v5 are both preserved/hashed. The repository gives a more specific, smaller screening grid; it is used to interpret the pasted “where specified” alpha values. The pasted latest instruction governs promotion of any passing candidate rather than requiring both to pass. No group WGA enters ranking or choice between passing candidates.\n\n'
            'All Phase0 score definitions and audit bins are fixed before group diagnostics. Group-blind fitting/selection code never receives group arrays. G2 uses a separate scaler for each actual OOF fold ensemble; within a fixed new nested ensemble fit on B only. A minimum30 fit observations per ensemble is required; no shared OOF-slot scaling.\n\n'
            'Train135 new nested A-only heads with the exact prior no-group20-epoch schedule; do not copy v3/v4 heads. The forced epoch1-kill/recovery head is reused as v5 seed0/fold0/head00. Seeds/splits match prior verification for comparability, so prior development inspection remains a limitation; no claim of fresh untouched confirmation.\n\n'
            'The two-family shortlist is fixed to assess both shared expert mixtures and class-wise aggregation in nesting when eligible, without selecting by subgroup results. No class-score temperature fitting or extra sensitivity is permitted. S4 residual base is S1. Top1 diagnostic cannot be shortlisted.\n\n'
            'Bootstrap intervals condition on fixed fitted pipelines; image IDs, not expert rows, are resampled. Gate resamples align the same IDs across all seeds. WGA is computed on pooled unique C images within each seed, never the average of fold minima.\n')

    def acceptance(self):
        def work():
            val=load('outputs/weighting/features/val.npz',['embedding','y','sample_id']);seed,fold,a,b,c=next(splits(val))
            pool=ROOT/'acceptance/head_fit_pool.npz';save_npz(pool,embedding=val['embedding'][a],y=val['y'][a],sample_id=val['sample_id'][a])
            sentinel=ROOT/'acceptance/completed_sentinel.npz';sentinel_record=ROOT/'acceptance/sentinel_record.json'
            if not sentinel_record.exists():
                save_npz(sentinel,arr=np.arange(12));dump(sentinel_record,dict(path=str(sentinel),sha256=sha(sentinel),mtime_ns=sentinel.stat().st_mtime_ns))
            folder=ROOT/'nested/seed0/fold0/head00';cmd=[sys.executable,'-m','experiments.geometry_aggregation_v5.run','--acceptance-pool',str(pool),'--acceptance-folder',str(folder)]
            log=ROOT/'acceptance/kill_recovery.log';env=os.environ|dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            with log.open('ab') as stream:
                if not (folder/'head.pt').exists():
                    if not (folder/'pause.ready').exists():
                        proc=subprocess.Popen(cmd+['--pause-worker'],stdout=stream,stderr=subprocess.STDOUT,env=env)
                        try:
                            for _ in range(1200):
                                if (folder/'pause.ready').exists():break
                                if proc.poll() is not None:raise RuntimeError('Recovery child failed; inspect '+str(log))
                                time.sleep(.05)
                            assert (folder/'pause.ready').exists()
                        finally:
                            if proc.poll() is None:proc.kill()
                            proc.wait(timeout=15)
                        epoch=torch.load(folder/'latest.pt',weights_only=True);assert epoch['epoch']==1
                    subprocess.run(cmd,check=True,stdout=stream,stderr=subprocess.STDOUT,env=env)
            resumed=torch.load(folder/'head.pt',weights_only=True);assert resumed['resumed_from_epoch']==1
            direct=fit_member(val['embedding'][a],val['y'][a],val['sample_id'][a],[],1000,0,ROOT/'acceptance/uninterrupted')
            assert torch.equal(resumed['prototypes'],direct['prototypes']) and torch.equal(resumed['scale'],direct['scale'])
            sentinel=json.loads((ROOT/'acceptance/sentinel_record.json').read_text());sp=Path(sentinel['path']);assert sha(sp)==sentinel['sha256'] and sp.stat().st_mtime_ns==sentinel['mtime_ns']
            path=ROOT/'acceptance/recovery.json';dump(path,dict(status='PASS',forced_process_kill_after_epoch=1,resumed_from_epoch=1,
                final_tensors_equal_uninterrupted=True,completed_sentinel_unchanged=True,head00_reused_by_nested=True))
            return dict(crash_recovery='PASS'),[path,pool,folder/'head.pt',ROOT/'acceptance/uninterrupted/head.pt',sp,ROOT/'acceptance/sentinel_record.json',log]
        self.j.task('acceptance',[],work)

    def verify(self):
        def work():
            files=[]
            with zipfile.ZipFile(self.archive) as z:
                assert z.testzip() is None
                for info in z.infolist():
                    name=Path(info.filename)
                    if name.is_absolute() or '..' in name.parts or ':' in info.filename:raise ValueError('Unsafe archive')
                    if info.is_dir():continue
                    relative=Path(*name.parts[1:]) if name.parts[0]=='development_bundle' else name
                    p=self.dev/relative;atomic(p,z.read(info));files.append(p)
            manifest=json.loads((self.dev/'manifest.json').read_text())
            for r in manifest['files']:assert sha(self.dev/r['path'])==r['sha256']
            checks=[]
            for seed in range(3):
                train=load(self.dev/f'seed{seed}/meta_train.npz');select=load(self.dev/f'seed{seed}/meta_select.npz')
                assert not set(train['sample_id'])&set(select['sample_id']);prov=json.loads((self.dev/f'seed{seed}/fold_provenance.json').read_text())
                for d in [train,select]:
                    validate(d['probs'],d['logits'],d['distances']);np.testing.assert_allclose(softmax(d['logits'],axis=-1),d['probs'],atol=2e-6)
                    np.testing.assert_array_equal(d['classes'],[0,1]);assert d['embedding'].shape==(len(d['y']),2048) and len(set(d['sample_id']))==len(d['y'])
                    assert str(d['backbone_sha256'])=='bce28dc72c3737ae19ffa057ba056e03d3f531180e7ad55e0b3c93bf4ea4bcaa'
                    for f in prov['folds']:
                        held=set(d['sample_id'][d['fold']==f['fold']]);assert held<=set(f['held_ids']) and not held&set(f['fit_ids'])
                        head=Path(f"outputs/weighting/seed{seed}/heads_fold{f['fold']}.pt");assert sha(head)==f['heads_sha256']
                checks.append(dict(seed=seed,meta_train=len(train['y']),meta_select=len(select['y']),OOF_provenance='PASS',scaler_fit_counts_by_fold={str(f):int((train['fold']==f).sum()) for f in range(3)}))
            path=ROOT/'inputs/verification.json';dump(path,dict(status='PASS',seeds=checks,archive_sha256=sha(self.archive),historical_test_read=False,group_check='not used for selection'))
            return dict(inputs='PASS'),files+[path]
        self.j.task('inputs',['acceptance'],work)

    def phase0(self):
        from . import audit
        tasks=[]
        for seed in range(3):
            task=f'audit:scales:{seed}';self.j.task(task,['inputs'],lambda seed=seed:audit.scales(ROOT,self.dev,seed));tasks.append(task)
            for split in ['meta_train','meta_select']:
                task=f'audit:export:{seed}:{split}';self.j.task(task,['inputs'],lambda seed=seed,split=split:audit.export(ROOT,self.dev,seed,split));tasks.append(task)
                for score in audit.SCORES:
                    rid=f'audit:rank:{seed}:{split}:{score}';self.j.task(rid,[task],lambda seed=seed,split=split,score=score:audit.rank(ROOT,seed,split,score));tasks.append(rid)
                pid=f'audit:plots:{seed}:{split}';deps=[task,f'audit:export:{seed}:meta_train'];self.j.task(pid,list(dict.fromkeys(deps)),lambda seed=seed,split=split:audit.plots(ROOT,self.dev,seed,split));tasks.append(pid)
        return self.j.task('audit:report',tasks,lambda:audit.report(ROOT))

    def data(self,seed,split):return load(self.dev/f'seed{seed}/{split}.npz',['probs','logits','distances','sample_id','y','fold'])

    def identity(self,seed):
        prov=json.loads((self.dev/f'seed{seed}/fold_provenance.json').read_text())
        return {str(f['fold']):[f['heads_sha256']+':slot'+str(j) for j in range(15)] for f in prov['folds']}

    def preprocess(self,seed):
        path=ROOT/f'screen/seed{seed}/distance_normalizer.json'
        def work():
            train=self.data(seed,'meta_train');n=fit_normalizer(train['distances'],train['fold'],train['sample_id'],self.identity(seed));dump(path,n)
            return dict(seed=seed,fit_images=len(train['y']),actual_ensembles=3),[path]
        self.j.task(f'screen:normalize:{seed}',['audit:report'],work);return json.loads(path.read_text())

    def phase1(self):
        normalizers={s:self.preprocess(s) for s in range(3)};records=[]
        for cfg in configs():
            for seed in range(3):
                task=f'screen:{cfg["id"]}:{seed}';folder=ROOT/f'screen/seed{seed}/{cfg["id"]}'
                def predict(cfg=cfg,seed=seed,folder=folder):
                    d=self.data(seed,'meta_select');start=time.perf_counter();out=aggregate(d['probs'],d['logits'],d['distances'],cfg,d['fold'],normalizers[seed],self.identity(seed));runtime=time.perf_counter()-start
                    path=folder/'predictions.npz';save_npz(path,**out,uniform=d['probs'].mean(1),sample_id=d['sample_id'])
                    r=summary(out['probs'],d['y']);u=summary(d['probs'].mean(1),d['y']);delta=(r['AA']-u['AA'])*100
                    r.update(candidate=cfg['id'],method=cfg['method'],seed=seed,delta_AA_pp=delta,uniform_AA=u['AA'],eligible=delta>-1.,diagnostic_only=cfg.get('diagnostic_only',False),runtime_seconds=runtime)
                    p=folder/'selection_metrics.json';dump(p,r);return r,[path,p]
                r=self.j.task(task,[f'screen:normalize:{seed}'],predict);records.append(dict(r,task=task))
        def lock():
            ranked=[]
            for cfg in configs():
                rows=[r for r in records if r['candidate']==cfg['id']]
                if cfg['method']=='U0' or cfg.get('diagnostic_only'):continue
                ranked.append(dict(config=cfg,eligible=all(r['eligible'] for r in rows),proxy=float(np.mean([r['worst_class_NLL'] for r in rows])),
                    worst_class_accuracy=float(np.mean([r['worst_class_accuracy'] for r in rows])),AA=float(np.mean([r['AA'] for r in rows])),
                    AA_SD=float(np.std([r['AA'] for r in rows],ddof=1)),delta_AA_pp_by_seed=[r['delta_AA_pp'] for r in rows]))
            ranked.sort(key=lambda r:(not r['eligible'],r['proxy'],-r['worst_class_accuracy'],-r['AA'],r['AA_SD'],r['config']['id']))
            eligible=[r for r in ranked if r['eligible']];locked=[]
            if eligible:
                locked.append(eligible[0]['config']);primary_soft=locked[0]['method'].startswith('S')
                opposite=[r for r in eligible[1:] if r['config']['method'].startswith('S')!=primary_soft]
                if opposite:locked.append(opposite[0]['config'])
                elif len(eligible)>1:locked.append(eligible[1]['config'])
            path=ROOT/'phase1_locked_candidates.json';dump(path,dict(locked=locked,rule=PROTOCOL['selection'],shortlist_rule=PROTOCOL['locking'],group_labels_accessed_by_selector=False,historical_test_used=False,ranking=ranked))
            csv=ROOT/'phase1_group_blind_ranking.csv';csvwrite(csv,ranked);return dict(locked=locked),[path,csv]
        locked=self.j.task('screen:lock',[r['task'] for r in records],lock)['locked'];tasks=[]
        # Descriptive group results are computed only after the frozen lock.
        for cfg in configs():
            for seed in range(3):
                name=f'screen:{cfg["id"]}:{seed}';task='diagnostic:'+name;folder=ROOT/f'screen/seed{seed}/{cfg["id"]}'
                def evaluate(cfg=cfg,seed=seed,folder=folder):
                    d=self.data(seed,'meta_select');g=load(self.dev/f'seed{seed}/meta_select.npz',['group'])['group'];z=load(folder/'predictions.npz')
                    r,flips=assess(z,z['uniform'],d['probs'],d['y'],g,d['sample_id']);r.update(candidate=cfg['id'],method=cfg['method'],seed=seed,diagnostic_only=cfg.get('diagnostic_only',False))
                    p=folder/'post_lock_evaluation.json';dump(p,r);fp=folder/'corrected_broken.csv';csvwrite(fp,flips)
                    return r,[p,fp]
                self.j.task(task,['screen:lock',name],evaluate);tasks.append(task)
        def report():
            rows=[self.j.records[t]['result'] for t in tasks];path=ROOT/'phase1_screen.csv';csvwrite(path,rows)
            report=ROOT/'phase1_report.md';atomic(report,'# Phase1 — frozen exploratory screening\n\n25 total configs including uniform and diagnostic Top1; 75 candidate/seed evaluations. All losing results retained. G2 scalers fit actual fold-specific meta_train heads only.\n\nLocked before subgroup evaluation: '+json.dumps(locked)+'\n\nThe proxy is mean worst-class NLL under strict per-seed AA floor; it is not WGA. Class-wise S methods use separate class-specific prototype contributions, not a common expert mixture. No extra class-temperature tuning, neural model or historical test rows are used. The two-family shortlist rule was preregistered. OOF slot transfer is not used in nesting.\n')
            return dict(phase1='COMPLETE',locked=locked),[path,report]
        self.j.task('screen:report',tasks,report);return locked

    def nested(self,locked):
        if not locked:
            def no_candidate():
                p=ROOT/'promotion_decisions.json';dump(p,dict(status='FAIL',reason='No eligible group-blind recipe',decisions=[],integration=None));csv=ROOT/'phase2_nested_results.csv';csvwrite(csv,[]);return dict(status='FAIL',decisions=[],integration=None),[p,csv]
            return self.j.task('nested:gate',['screen:report'],no_candidate)
        val=load('outputs/weighting/features/val.npz',['embedding','y','sample_id']);evaltasks=[];pools={cfg['id']:[] for cfg in locked}
        for seed,fold,a,b,c in splits(val):
            prefix=f'outer:{seed}:{fold}';folder=ROOT/f'nested/seed{seed}/fold{fold}'
            def plan(seed=seed,fold=fold,a=a,b=b,c=c,folder=folder):
                sets=[set(val['sample_id'][i]) for i in (a,b,c)];assert all(not sets[i]&sets[k] for i in range(3) for k in range(i+1,3))
                p=folder/'ABC_manifest.json';dump(p,dict(seed=seed,fold=fold,A=val['sample_id'][a].tolist(),B=val['sample_id'][b].tolist(),C=val['sample_id'][c].tolist(),
                    A_indices=a.tolist(),B_indices=b.tolist(),C_indices=c.tolist(),head_settings=SETTINGS,new_v5_training=True,groups_used_for_fit=False))
                return dict(seed=seed,fold=fold,A=len(a),B=len(b),C=len(c)),[p]
            self.j.task(prefix+':plan',['screen:report'],plan);previous=[];headtasks=[]
            for member in range(15):
                hfolder=folder/f'head{member:02d}';task=prefix+f':head{member:02d}'
                def fit(seed=seed,fold=fold,a=a,hfolder=hfolder,member=member,previous=previous.copy()):
                    h=fit_member(val['embedding'][a],val['y'][a],val['sample_id'][a],previous,1000+100*seed+fold,member,hfolder)
                    assert set(h['fit_ids'])==set(val['sample_id'][a]);epoch=torch.load(hfolder/'latest.pt',weights_only=True);assert epoch['epoch']==20
                    return dict(seed=seed,fold=fold,head=member,epochs=20,epoch_resume=h['resumed_from_epoch']),[hfolder/'head.pt',hfolder/'latest.pt']
                deps=[headtasks[-1]] if headtasks else [prefix+':plan','acceptance'];self.j.task(task,deps,fit);headtasks.append(task);previous.append(torch.load(hfolder/'head.pt',weights_only=True))
            def predict(previous=previous,folder=folder,b=b,c=c):
                heads=dict(prototypes=torch.stack([h['prototypes'] for h in previous]),scales=torch.stack([h['scale'] for h in previous]));bp=predict_heads(heads,val['embedding'][b]);cp=predict_heads(heads,val['embedding'][c])
                hashes=np.array([sha(folder/f'head{j:02d}/head.pt') for j in range(15)])
                p=folder/'member_predictions.npz';save_npz(p,**{'B_'+k:bp[k] for k in ['probs','logits','distances']},**{'C_'+k:cp[k] for k in ['probs','logits','distances']},B_ids=val['sample_id'][b],C_ids=val['sample_id'][c],head_hashes=hashes)
                manifest=folder/'head_manifest.json';dump(manifest,dict(ordered_heads=[dict(slot=j,path=str(folder/f'head{j:02d}/head.pt'),sha256=str(hashes[j]),fit_ids_sha256=key(previous[j]['fit_ids'])) for j in range(15)],B_C_same_order=True,cache_sha256=sha(p)))
                return dict(predictions_cached=True,new_heads=15),[p,manifest]
            self.j.task(prefix+':predict',headtasks,predict);z=load(folder/'member_predictions.npz');ident={'0':z['head_hashes'].tolist()}
            def normalize(folder=folder,z=z,ident=ident):
                normalizer=fit_normalizer(z['B_distances'],np.zeros(len(z['B_ids']),int),z['B_ids'],ident);path=folder/'B_only_normalizer.json';dump(path,normalizer)
                return dict(fit_split='B',fit_images=len(z['B_ids'])),[path]
            self.j.task(prefix+':normalize',[prefix+':predict'],normalize);normalizer=json.loads((folder/'B_only_normalizer.json').read_text())
            for cfg in locked:
                predtask=prefix+':aggregate:'+cfg['id'];outfolder=folder/cfg['id']
                def aggregate_c(cfg=cfg,outfolder=outfolder,z=z,ident=ident,normalizer=normalizer):
                    start=time.perf_counter();output=aggregate(z['C_probs'],z['C_logits'],z['C_distances'],cfg,np.zeros(len(z['C_ids']),int),normalizer,ident);runtime=time.perf_counter()-start
                    p=outfolder/'predictions.npz';save_npz(p,**output,uniform=z['C_probs'].mean(1),sample_id=z['C_ids'],head_hashes=z['head_hashes'])
                    return dict(candidate=cfg['id'],inference_seconds=runtime,group_read=False),[p]
                self.j.task(predtask,[prefix+':normalize','screen:lock'],aggregate_c)
                def evaluate_c(cfg=cfg,outfolder=outfolder,c=c,z=z,seed=seed,fold=fold):
                    saved=load(outfolder/'predictions.npz');g=load('outputs/weighting/features/val.npz',['group'])['group'][c]
                    r,flips=assess(saved,saved['uniform'],z['C_probs'],val['y'][c],g,saved['sample_id']);r.update(candidate=cfg['id'],method=cfg['method'],seed=seed,fold=fold,level='fold')
                    p=outfolder/'evaluation.json';dump(p,r);fp=outfolder/'flips.csv';csvwrite(fp,flips);return r,[p,fp]
                ev=prefix+':evaluate:'+cfg['id'];self.j.task(ev,[predtask],evaluate_c);evaltasks.append(ev)
            def complete_fold(folder=folder,seed=seed,fold=fold):
                p=folder/'fold_complete.json';dump(p,dict(seed=seed,fold=fold,all_locked_candidates_complete=True,head_hashes=z['head_hashes'].tolist()));return dict(seed=seed,fold=fold,complete=True),[p]
            self.j.task(prefix+':complete',[prefix+':evaluate:'+cfg['id'] for cfg in locked],complete_fold)
        for cfg in locked:
            for seed in range(3):
                def pool(cfg=cfg,seed=seed):
                    parts=[load(ROOT/f'nested/seed{seed}/fold{f}/{cfg["id"]}/predictions.npz') for f in range(3)];ids=np.concatenate([p['sample_id'] for p in parts]);lookup={str(s):i for i,s in enumerate(val['sample_id'])}
                    assert len(set(ids))==len(ids)==len(val['y']);order=np.argsort([lookup[str(s)] for s in ids]);saved={k:np.concatenate([p[k] for p in parts])[order] for k in ['probs','uniform','weights','contributions','scores','sample_id']}
                    np.testing.assert_array_equal(saved['sample_id'],val['sample_id']);memberparts=[load(ROOT/f'nested/seed{seed}/fold{f}/member_predictions.npz',['C_probs'])['C_probs'] for f in range(3)];members=np.concatenate(memberparts)[order]
                    g=load('outputs/weighting/features/val.npz',['group'])['group'];r,flips=assess(saved,saved['uniform'],members,val['y'],g,saved['sample_id'],2000)
                    r.update(candidate=cfg['id'],method=cfg['method'],seed=seed,fold=-1,level='pooled_seed');folder=ROOT/f'pooled/{cfg["id"]}/seed{seed}';p=folder/'predictions.npz';save_npz(p,**saved);ep=folder/'evaluation.json';dump(ep,r);fp=folder/'flips.csv';csvwrite(fp,flips)
                    return r,[p,ep,fp]
                task=f'pool:{cfg["id"]}:{seed}';r=self.j.task(task,[f'outer:{seed}:{f}:complete' for f in range(3)],pool);pools[cfg['id']].append(r)
        def gate():
            decisions=[];group=load('outputs/weighting/features/val.npz',['group'])['group']
            for cfg in locked:
                rows=pools[cfg['id']];wa=float(np.mean([r['delta_WGA_pp'] for r in rows]));aa=float(np.mean([r['delta_AA_pp'] for r in rows]));positive=sum(r['delta_WGA_pp']>0 for r in rows)
                criteria=dict(mean_WGA_positive=wa>0,positive_WGA_seeds_at_least2=positive>=2,AA_floor_every_seed=all(r['delta_AA_pp']>-1 for r in rows),provenance=True)
                pred=[load(ROOT/f'pooled/{cfg["id"]}/seed{s}/predictions.npz',['probs','uniform']) for s in range(3)]
                decisions.append(dict(candidate=cfg['id'],config=cfg,status='PASS' if all(criteria.values()) else 'FAIL',criteria=criteria,mean_delta_WGA_pp=wa,mean_delta_AA_pp=aa,positive_seeds=positive,seeds=rows,
                    **joint_ci([p['probs'] for p in pred],[p['uniform'] for p in pred],val['y'],group)))
            passing=[d for d in decisions if d['status']=='PASS'];result=dict(status='PASS' if passing else 'FAIL',decisions=decisions,integration=passing[0]['config'] if passing else None,
                winner_rule='First passing recipe in prelocked order, never highest outer WGA',data_qualification='Previously inspected development; single shared frozen ERMstar backbone')
            p=ROOT/'promotion_decisions.json';dump(p,result);csv=ROOT/'phase2_nested_results.csv';csvwrite(csv,[self.j.records[t]['result'] for t in evaltasks]+[r for cfg in locked for r in pools[cfg['id']]])
            return result,[p,csv]
        return self.j.task('nested:gate',[f'pool:{cfg["id"]}:{s}' for cfg in locked for s in range(3)],gate)

    def finish(self,gate):
        from .report import write_report
        self.j.task('final:report',['nested:gate'],lambda:write_report(ROOT,gate))
        if gate['status']=='PASS':
            from .integration import integrate
            integrate(self.j,gate)
        else:self.j.status('COMPLETE_GATE_FAILED','None; preserve results, no production integration',promotion='FAIL',integration='NOT_PERFORMED')
        print(json.dumps(dict(output=str(ROOT),gate=gate['status'],completed=len(self.j.records))),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['all','0','1','2','audit','screen','nested'],default='all');parser.add_argument('--resume',action='store_true');parser.add_argument('--status',action='store_true');parser.add_argument('--max-tasks',type=int)
    parser.add_argument('--acceptance-pool');parser.add_argument('--acceptance-folder');parser.add_argument('--pause-worker',action='store_true');args=parser.parse_args()
    if args.acceptance_pool:return acceptance_worker(args.acceptance_pool,args.acceptance_folder,args.pause_worker)
    if args.status:
        path=ROOT/'state.json'
        if not path.exists():print('NOT_STARTED');return
        state=json.loads(path.read_text());print(json.dumps({k:v for k,v in state.items() if k not in ['task_DAG','done_configs']},indent=2));return
    torch.set_num_threads(1)
    with threadpool_limits(limits=1),single_writer(ROOT):
        w=Workflow(args)
        try:
            w.acceptance();w.verify();w.phase0()
            if args.phase in ['0','audit']:w.j.status('AUDIT_COMPLETE','screen');return
            locked=w.phase1()
            if args.phase in ['1','screen']:w.j.status('SCREEN_COMPLETE','nested');return
            gate=w.nested(locked);w.finish(gate)
        except InterruptedError:print('Stopped at durable boundary; resume: '+CMD)


if __name__=='__main__':main()
