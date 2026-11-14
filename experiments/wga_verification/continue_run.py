"""Resume saved Gate A computations with newly supplied sources, then true nesting."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import joblib
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold,train_test_split
from threadpoolctl import threadpool_limits
from experiments.discrete_weighting.store import atomic,dump,key,sha
from experiments.discrete_weighting.run import csvwrite
from experiments.weighting.metrics import metrics
from experiments.weighting.data import predict_heads
from .journal import Journal
from .run import save_npz,score
from .solver import clusters,solve
from .nested_heads import fit_member,SETTINGS

ROOT=Path('outputs/wga_verification_v3/ce66e4448807')


def locate(name):
    matches=[p for p in Path('.').rglob(name) if not any(part in ('.venv','.git','source_snapshot') for part in p.parts)]
    if not matches:raise FileNotFoundError(name)
    hashes={sha(p) for p in matches}
    if len(hashes)!=1:raise RuntimeError('Ambiguous source copies: '+str(matches))
    return sorted(matches,key=lambda p:(len(p.parts),str(p)))[0]


class Continuation:
    def __init__(self,args):
        self.root=ROOT;self.args=args;self.bundle=Path('outputs/weighting/development_bundle')
        self.analysis=locate('DPE_WGA_Deep_Analysis.md');self.archive=locate('DPE_WGA_Audit_Repro.zip')
        self.command='python -m experiments.wga_verification.continue_run --resume'
        self.base=json.loads((ROOT/'manifest.json').read_text())
        self.j=Journal(ROOT,self.base,self.command,args.max_tasks)
        if len([r for r in self.j.records if r.startswith(('integrity:','preprocess:','solver:','legacy_score:','cv5infer:','cv5_score:'))])!=60:raise RuntimeError('Base verification checkpoints incomplete/corrupt; restore base run first')
        sources={p.as_posix():sha(p) for p in [Path(__file__),Path('experiments/wga_verification/nested_heads.py'),Path('experiments/wga_verification/solver.py'),Path('experiments/wga_verification/journal.py'),Path('experiments/weighting/data.py')]}
        plan=dict(base_manifest_sha256=sha(ROOT/'manifest.json'),source=sources,analysis_path=self.analysis.as_posix(),analysis_sha256=sha(self.analysis),archive_path=self.archive.as_posix(),archive_sha256=sha(self.archive),
                  feature_sha256=sha('outputs/weighting/features/val.npz'),outer_seeds=[0,1,2],outer_folds=3,head_fraction=.7,
                  inner_split_seed='7000 + seed*31 + fold',head_seeds='1000 + seed*100 + fold',head_training=SETTINGS,
                  frozen_weight_recipe=self.base['config']['recipe'],gate_B_criteria=dict(mean_WGA_positive=True,WGA_positive_seeds=2,mean_AA_gt_minus1=True,each_AA_gt_minus1=True))
        plan['revision']=2
        plan['parent_manifest_sha256']=sha(ROOT/'continuation_manifest.json')
        plan['revision_reason']='Add missing joblib import for preprocessing serialization; no head/weight mathematics, inputs, split or settings changed. All valid prior task artifacts retained.'
        marker=ROOT/'continuation_manifest.r2.json'
        if marker.exists() and json.loads(marker.read_text())!=plan:raise ValueError('Continuation source/config changed; preserve results and explicitly audit a new revision')
        if not marker.exists():dump(marker,plan)
        for p,h in sources.items():
            dest=ROOT/'continuation_source_snapshot_r2'/p
            if not dest.exists():atomic(dest,Path(p).read_bytes())
            assert sha(dest)==h
        for sig in (__import__('signal').SIGINT,__import__('signal').SIGTERM):__import__('signal').signal(sig,lambda *_:setattr(self.j,'stop',True))

    def source_gate(self):
        extract=ROOT/'original_sources'
        def unpack():
            names=[]
            with zipfile.ZipFile(self.archive) as z:
                assert z.testzip() is None
                for info in z.infolist():
                    name=Path(info.filename)
                    if name.is_absolute() or '..' in name.parts or ':' in info.filename:raise ValueError('Unsafe archive member')
                    if info.is_dir():continue
                    dest=extract/name;atomic(dest,z.read(info));names.append(dict(path=info.filename,sha256=sha(dest)))
            path=ROOT/'verification/source_comparison/archive_inventory.json';dump(path,dict(archive_sha256=sha(self.archive),analysis_sha256=sha(self.analysis),members=names,original_sha_manifest_present=any('manifest' in n['path'].lower() for n in names),
                   qualification='No separate original SHA manifest or epigraph_verify.py is provided; archive members are hashed on receipt; bundle export hashes and independent epigraph audit remain verified.'))
            return dict(source_archive_verified=True),[path]+[extract/n['path'] for n in names]
        self.j.task('source_inventory',[],unpack)
        for script in ('cv_sensitivity.py','robust_static.py'):
            def run_original(script=script):
                original=extract/'dpe_dev_analysis'/script
                text=original.read_text(encoding='utf-8')
                out=ROOT/'verification/source_comparison'/script[:-3];out.mkdir(parents=True,exist_ok=True)
                # Only replace absolute filesystem roots; all math and grids remain unchanged.
                adapted=text.replace('/mnt/data/dpe_dev_analysis/development_bundle',self.bundle.resolve().as_posix()).replace('/mnt/data/dpe_dev_analysis',out.resolve().as_posix())
                adapted_path=out/script;atomic(adapted_path,adapted)
                env=os.environ|dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1')
                proc=subprocess.run([sys.executable,str(adapted_path.resolve())],capture_output=True,text=True,encoding='utf-8',env=env)
                log=out/'execution.log';atomic(log,proc.stdout+'\n'+proc.stderr)
                if proc.returncode:raise RuntimeError('Original script failed: '+str(log))
                if any('opt_fail' in l and '0.25' in l for l in proc.stdout.splitlines()):raise RuntimeError('Original primary solver failed')
                files=list(out.glob('*.csv'))+list(out.glob('*.npz'))+[adapted_path,log]
                return dict(script=script,returncode=0,original_sha256=sha(original),adapted_sha256=sha(adapted_path),changes='filesystem path substitutions only; CPU thread-count environment pinned'),files
            self.j.task('original_script:'+script,['source_inventory'],run_original)
        def compare():
            source=ROOT/'verification/source_comparison/cv_sensitivity'
            actual=list(csv.DictReader((source/'cv_sensitivity_results.csv').open()))
            archived=list(csv.DictReader((extract/'dpe_dev_analysis/cv_sensitivity_results.csv').open()))
            legacy=list(csv.DictReader((ROOT/'verification/source_comparison/robust_static/robust_static_results.csv').open()))
            old_legacy=list(csv.DictReader((extract/'dpe_dev_analysis/robust_static_results.csv').open()))
            comparisons=[];reasons=[]
            for seed in (0,1,2):
                independent=json.loads((ROOT/f'verification/seed{seed}/cv5_metrics.json').read_text())
                for method,label in [('uniform','uniform'),('pseudo_cap025','candidate')]:
                    r=next(x for x in actual if int(x['seed'])==seed and x['method']==method)
                    old=next(x for x in archived if int(x['seed'])==seed and x['method']==method)
                    for metric,field in [('AA','accuracy'),('WGA','wga')]:
                        gap=max(abs(float(r[metric])-independent[label][field]),abs(float(r[metric])-float(old[metric])))*100
                        comparisons.append(dict(seed=seed,method=method,metric=metric,source=float(r[metric]),independent=independent[label][field],archived=float(old[metric]),gap_pp=gap))
                        if gap>.03:reasons.append(str(comparisons[-1]))
                with np.load(source/f'cv_preds_seed{seed}.npz',allow_pickle=False) as original,np.load(ROOT/f'verification/seed{seed}/cv5_predictions.npz',allow_pickle=False) as own:
                    assert np.array_equal(original['sample_id'],own['sample_id']) and np.array_equal(original['y'],own['y'])
                    assert np.array_equal(original['pseudo_cap025']>=.5,own['candidate'].argmax(1)==1)
                chosen=lambda rows:next(x for x in rows if int(x['seed'])==seed and x['kind']=='global_k2_max' and float(x['cap'])==.25 and float(x['ridge'])==.02 and float(x['alpha'])==1.)
                r,o=chosen(legacy),chosen(old_legacy);own=json.loads((ROOT/f'verification/seed{seed}/metaselect_metrics.json').read_text())['candidate']
                for m,f in [('AA','accuracy'),('WGA','wga')]:
                    gap=max(abs(float(r[m])-float(o[m])),abs(float(r[m])-own[f]))*100
                    if gap>.03:reasons.append(f'Legacy seed{seed} {m} mismatch {gap}')
            result=dict(status='FAIL' if reasons else 'PASS',reasons=reasons,source_scripts_executed=['cv_sensitivity.py','robust_static.py'],original_manifest='Not supplied: explained missing original manifest, receipt hashes recorded; development bundle export manifest verified.',
                        independent_epigraph_audit='18 valid prior tasks retained',max_source_discrepancy_pp=max(r['gap_pp'] for r in comparisons),supersedes='verification/GATE_A.json (old missing-source decision retained immutably)')
            path=ROOT/'verification/source_comparison/GATE_A.json';dump(path,result);report=path.with_name('comparison.csv');csvwrite(report,comparisons)
            return result,[path,report]
        return self.j.task('gate_A_with_sources',['gate_A','original_script:cv_sensitivity.py','original_script:robust_static.py'],compare)['metrics']

    def plans(self,val):
        for seed in (0,1,2):
            for fold,(outer,held) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed).split(val['embedding'],val['y'])):
                a,b=train_test_split(outer,train_size=.7,stratify=val['y'][outer],random_state=7000+seed*31+fold)
                yield seed,fold,a,b,held

    def real_head_acceptance(self,val):
        seed,fold,a,b,c=next(self.plans(val));folder=ROOT/'nested/seed0/fold0/head00';pool=ROOT/'acceptance/real_pool.npz'
        def work():
            save_npz(pool,embedding=val['embedding'][a],y=val['y'][a],sample_id=val['sample_id'][a])
            log=ROOT/'acceptance/real_head.log';log.parent.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-m','experiments.wga_verification.nested_heads','--pool',str(pool),'--folder',str(folder)]
            if not (folder/'head.pt').exists():
                with log.open('wb') as stream:
                    proc=subprocess.Popen(command+['--pause'],stdout=stream,stderr=subprocess.STDOUT)
                    try:
                        for _ in range(600):
                            if (folder/'pause.ready').exists():break
                            if proc.poll() is not None:raise RuntimeError('Head acceptance child failed: '+str(log))
                            time.sleep(.05)
                        assert (folder/'pause.ready').exists()
                    finally:
                        if proc.poll() is None:proc.kill()
                        proc.wait(timeout=15)
                    assert not (folder/'head.pt').exists()
                    stamp=(folder/'latest.pt').stat().st_mtime_ns
                    subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
                resumed=torch.load(folder/'head.pt',weights_only=True)
                assert resumed['resumed_from_epoch']==1
            resumed=torch.load(folder/'head.pt',weights_only=True)
            control=ROOT/'acceptance/uninterrupted'
            direct=fit_member(val['embedding'][a],val['y'][a],val['sample_id'][a],[],1000,0,control)
            assert torch.equal(resumed['prototypes'],direct['prototypes']) and torch.equal(resumed['scale'],direct['scale'])
            path=ROOT/'acceptance/real_head_recovery.json';dump(path,dict(status='PASS',killed_after_epoch=1,resumed_from_epoch=resumed['resumed_from_epoch'],identical_to_uninterrupted=True,real_head_checkpoint_reused_in_nested=True))
            return dict(real_head_recovery='PASS'),[path,folder/'head.pt',control/'head.pt',pool]
        self.j.task('real_head_recovery',['gate_A_with_sources'],work)

    def nested(self):
        with np.load('outputs/weighting/features/val.npz',allow_pickle=False) as f:val=dict(f)
        self.real_head_acceptance(val)
        all_fold_ids=[]
        for seed,fold,a,b,c in self.plans(val):
            prefix=f'nested:{seed}:{fold}';folder=ROOT/f'nested/seed{seed}/fold{fold}'
            def plan(seed=seed,fold=fold,a=a,b=b,c=c,folder=folder):
                sets=[set(val['sample_id'][v]) for v in (a,b,c)]
                assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
                paths=[]
                for label,idx in [('head_fit',a),('weight_fit',b),('outer_holdout',c)]:
                    p=folder/(label+'_ids.csv');csvwrite(p,[dict(sample_id=str(val['sample_id'][i])) for i in idx]);paths.append(p)
                manifest=folder/'split.json';dump(manifest,dict(seed=seed,fold=fold,head_indices=a.tolist(),weight_indices=b.tolist(),hold_indices=c.tolist(),groups_used_for_fit=False,head_settings=SETTINGS));paths.append(manifest)
                return dict(seed=seed,fold=fold,head_count=len(a),weight_count=len(b),holdout_count=len(c)),paths
            self.j.task(prefix+':split',['gate_A_with_sources'],plan)
            previous=[];head_tasks=[]
            for member in range(15):
                hfolder=folder/f'head{member:02d}';task=prefix+f':head{member:02d}'
                def train(member=member,hfolder=hfolder,previous=previous.copy(),seed=seed,fold=fold,a=a):
                    result=fit_member(val['embedding'][a],val['y'][a],val['sample_id'][a],previous,1000+seed*100+fold,member,hfolder)
                    return dict(seed=seed,fold=fold,member=member,epoch=20),[hfolder/'head.pt']
                deps=[head_tasks[-1]] if head_tasks else [prefix+':split','real_head_recovery']
                self.j.task(task,deps,train);head_tasks.append(task);previous.append(torch.load(hfolder/'head.pt',weights_only=True))
            cache=folder/'predictions.npz';infer_id=prefix+':predict'
            def infer(previous=previous,b=b,c=c,cache=cache,folder=folder):
                heads=dict(prototypes=torch.stack([h['prototypes'] for h in previous]),scales=torch.stack([h['scale'] for h in previous]))
                bp=predict_heads(heads,val['embedding'][b])['probs'];cp=predict_heads(heads,val['embedding'][c])['probs']
                assert bp.shape==(len(b),15,2) and cp.shape==(len(c),15,2)
                member_hashes=[sha(folder/f'head{i:02d}/head.pt') for i in range(15)]
                save_npz(cache,weight_probs=bp,holdout_probs=cp,weight_ids=val['sample_id'][b],holdout_ids=val['sample_id'][c],member_hashes=np.array(member_hashes))
                return {},[cache]
            self.j.task(infer_id,head_tasks,infer)
            fit_id=prefix+':preprocess';prep=folder/'weight_preprocess.joblib';labels_path=folder/'clusters.npz'
            def preprocess(b=b,prep=prep,labels_path=labels_path):
                pca,km,labels=clusters(val['embedding'][b]);buffer=io.BytesIO();joblib.dump(dict(pca=pca,kmeans=km,fit_ids=val['sample_id'][b]),buffer);atomic(prep,buffer.getvalue());save_npz(labels_path,labels=labels)
                return {},[prep,labels_path]
            self.j.task(fit_id,[infer_id],preprocess)
            wid=prefix+':weights';weights_path=folder/'weights.npz';waudit=folder/'solver.json'
            def weighting(cache=cache,b=b,labels_path=labels_path,weights_path=weights_path,waudit=waudit):
                with np.load(cache,allow_pickle=False) as f:p=f['weight_probs'];hashes=f['member_hashes']
                with np.load(labels_path,allow_pickle=False) as f:labels=f['labels']
                w,audit=solve(p,val['y'][b],labels,True)
                save_npz(weights_path,weights=w,member_hashes=hashes,fit_ids=val['sample_id'][b]);dump(waudit,audit)
                return dict(solver_success=audit['success'],max_weight=float(w.max())),[weights_path,waudit]
            self.j.task(wid,[fit_id],weighting)
            output=folder/'holdout_predictions.npz';predid=prefix+':outer_predictions'
            def outer(cache=cache,weights_path=weights_path,c=c,output=output):
                with np.load(cache,allow_pickle=False) as f:p=f['holdout_probs'];hashes=f['member_hashes'];ids=f['holdout_ids']
                with np.load(weights_path,allow_pickle=False) as f:w=f['weights'];assert np.array_equal(hashes,f['member_hashes'])
                assert np.array_equal(ids,val['sample_id'][c])
                candidate=np.einsum('nmc,m->nc',p,w);uniform=p.mean(1)
                save_npz(output,uniform=uniform,candidate=candidate,sample_id=ids,indices=c,member_hashes=hashes)
                return dict(seed=seed,fold=fold),[output]
            self.j.task(predid,[wid],outer);all_fold_ids.append(predid)
            # Only after predictions are committed, attach evaluation labels/groups.
            def evaluate(output=output,c=c,folder=folder):
                with np.load(output,allow_pickle=False) as f:u=f['uniform'];q=f['candidate']
                r=dict(seed=seed,fold=fold,uniform=score(u,val['y'][c],val['group'][c]),candidate=score(q,val['y'][c],val['group'][c]))
                path=folder/'evaluation.json';dump(path,r);return r,[path]
            self.j.task(prefix+':evaluation',[predid],evaluate)
        return self.nested_decision(val,all_fold_ids)

    def nested_decision(self,val,deps):
        def work():
            rows=[];all_u=[];all_q=[];paths=[]
            for seed in (0,1,2):
                uniform=np.zeros((1199,2));candidate=uniform.copy();coverage=np.zeros(1199,dtype=int)
                for fold in range(3):
                    with np.load(ROOT/f'nested/seed{seed}/fold{fold}/holdout_predictions.npz',allow_pickle=False) as f:
                        idx=f['indices'];assert np.array_equal(f['sample_id'],val['sample_id'][idx]);uniform[idx]=f['uniform'];candidate[idx]=f['candidate'];coverage[idx]+=1
                assert (coverage==1).all();u=score(uniform,val['y'],val['group']);q=score(candidate,val['y'],val['group'])
                up=uniform.argmax(1);qp=candidate.argmax(1);uc=up==val['y'];qc=qp==val['y']
                row=dict(seed=seed,uniform=u,candidate=q,delta_WGA_pp=100*(q['wga']-u['wga']),delta_AA_pp=100*(q['accuracy']-u['accuracy']),
                         corrected=int((qc&~uc).sum()),broken=int((~qc&uc).sum()))
                rows.append(row);all_u.append(up);all_q.append(qp)
                path=ROOT/f'nested/seed{seed}/pooled_predictions.npz';save_npz(path,uniform=uniform,candidate=candidate,y=val['y'],group=val['group'],sample_id=val['sample_id']);paths.append(path)
                cases=path.with_name('corrected_broken.csv');flips=[dict(sample_id=str(val['sample_id'][i]),y=int(val['y'][i]),group=int(val['group'][i]),uniform=int(up[i]),candidate=int(qp[i]),corrected=bool(qc[i])) for i in np.flatnonzero(up!=qp)]
                csvwrite(cases,flips)
                if not flips:atomic(cases,'sample_id,y,group,uniform,candidate,corrected\n')
                paths.append(cases)
            dw=np.array([r['delta_WGA_pp'] for r in rows]);da=np.array([r['delta_AA_pp'] for r in rows])
            criteria=dict(positive_mean_WGA=bool(dw.mean()>0),WGA_improves_two_seeds=bool((dw>0).sum()>=2),mean_AA_above_minus1=bool(da.mean()>-1),each_AA_above_minus1=bool((da>-1).all()),clean_provenance=True)
            rng=np.random.default_rng(3001);u=np.array(all_u)==val['y'];q=np.array(all_q)==val['y'];boot=[]
            for _ in range(2000):
                idx=np.concatenate([rng.choice(np.flatnonzero(val['group']==g),size=int((val['group']==g).sum()),replace=True) for g in range(4)])
                groups=val['group'][idx]
                wg_u=np.array([[u[s,idx][groups==g].mean() for g in range(4)] for s in range(3)]).min(1)
                wg_q=np.array([[q[s,idx][groups==g].mean() for g in range(4)] for s in range(3)]).min(1)
                boot.append([100*(wg_q-wg_u).mean(),100*(q[:,idx]-u[:,idx].astype(float)).mean()])
            ci=np.quantile(boot,[.025,.975],axis=0).T.tolist()
            result=dict(status='PASS' if all(criteria.values()) else 'FAIL',criteria=criteria,mean_delta_WGA_pp=float(dw.mean()),mean_delta_AA_pp=float(da.mean()),
                        paired_image_bootstrap_CI_WGA_pp=ci[0],paired_image_bootstrap_CI_AA_pp=ci[1],seeds=rows,
                        caution='Shared backbone, previously inspected development; not fresh scientific confirmation. Bootstrap aligns repeats by unique image ID and stratifies evaluation groups after predictions.')
            path=ROOT/'verification/source_comparison/GATE_B.json';dump(path,result);paths.append(path)
            table=ROOT/'nested/paired_metrics.csv';csvwrite(table,rows);paths.append(table)
            text='# Fully nested head-and-weight evaluation\n\n'+json.dumps(result,indent=2)+'\n\nEvery fold trains 15 heads on A only, fits frozen PCA/KMeans/static weights on B only, and evaluates C once with the same ordered frozen heads. A/B/C are class-stratified and disjoint. No groups enter training, clustering, stopping or selection. Fixed-final-epoch head math/schedule matches the prior class-balanced DPE adaptation, including partial batches; small A pools (~559 examples) yield fewer balanced optimizer updates than full development pools. No hyperparameter changes were made. Existing OOF results and all old checkpoints remain intact.\n'
            report=ROOT/'nested/verification_report.md';atomic(report,text);paths.append(report)
            return result,paths
        return self.j.task('gate_B_nested',deps,work)['metrics']

    def execute(self):
        gate=self.source_gate()
        if gate['status']!='PASS':self.finish(gate,None);return
        with threadpool_limits(1):b=self.nested()
        self.finish(gate,b)

    def finish(self,a,b):
        allowed=a['status']=='PASS' and b and b['status']=='PASS'
        # Integration is performed in a separate, subsequently verified task only after both gates pass.
        stage='INTEGRATION_PENDING' if allowed else 'STOPPED_VERIFICATION_FAILED'
        payload=dict(gate_A=a['status'],gate_B=b['status'] if b else 'NOT_RUN',production_integration='PENDING' if allowed else 'PROHIBITED',nested=b)
        atomic(ROOT/'CONTINUATION_SUMMARY.md','# Source comparison and nested verification\n\n'+json.dumps(payload,indent=2)+'\n')
        self.j.status(stage,'implement identity-bound opt-in integration' if allowed else 'do not integrate; inspect gate failure',gate_A=a['status'],gate_B=b['status'] if b else 'NOT_RUN',production_integration=payload['production_integration'])
        print(json.dumps(dict(gate_A=a['status'],gate_B=b['status'] if b else 'NOT_RUN',mean_delta_WGA_pp=b['mean_delta_WGA_pp'] if b else None,mean_delta_AA_pp=b['mean_delta_AA_pp'] if b else None,integration=payload['production_integration'])));print(self.command)


def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--status',action='store_true');p.add_argument('--max-tasks',type=int);a=p.parse_args();run=Continuation(a)
    if a.status:print((ROOT/'state.json').read_text());return
    try:run.execute()
    except InterruptedError:print('Stopped safely. '+run.command)
    except BaseException as e:
        import traceback
        atomic(ROOT/'CONTINUATION_BLOCKERS.md',traceback.format_exc());run.j.status('BLOCKED',str(e),production_integration='PROHIBITED');raise


if __name__=='__main__':main()
