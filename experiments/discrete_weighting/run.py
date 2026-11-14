import argparse
import copy
import csv
import io
import json
from pathlib import Path
import signal
import shutil
import subprocess
import time
import traceback
import joblib
import numpy as np
import yaml
from experiments.weighting.metrics import metrics,paired_ci
from .store import Store,atomic,dump,key,sha
from .reliability import Rule,recipes,ablations,fit_temperature,calibrate


def npz(path,**data):
    b=io.BytesIO();np.savez_compressed(b,**data);atomic(path,b.getvalue())
    with np.load(path,allow_pickle=False) as f:
        for k,v in data.items():
            if np.asarray(v).dtype.kind in 'fc' and not np.isfinite(f[k]).all():raise ValueError('Nonfinite artifact')


def csvwrite(path,rows):
    if not rows:return
    s=io.StringIO(newline='');cols=sorted(set(k for r in rows for k in r));w=csv.DictWriter(s,fieldnames=cols);w.writeheader()
    for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    atomic(path,s.getvalue())


class Run:
    def __init__(self,c,args):
        self.c=c;self.args=args;self.prior=Path(c['prior']);self.memory={}
        paths=list(Path('experiments/discrete_weighting').glob('*.py'))+[Path('experiments/weighting/metrics.py')]
        sources={str(p).replace('\\','/'):sha(p) for p in paths}
        inputs={}
        for seed in c['seeds']:
            for base in (self.prior,self.prior/'crossfit5'):
                for p in sorted((base/f'seed{seed}').glob('predictions_fold*.npz')):inputs[p.as_posix()]=sha(p)
                for p in sorted((base/f'seed{seed}').glob('heads_fold*.pt')):inputs[p.as_posix()]=sha(p)
            p=self.prior/f'seed{seed}/split_audit.json';inputs[p.as_posix()]=sha(p)
        # No test arrays/bundle are opened or used to construct recipes/selection.
        self.fingerprint=dict(config=c,source=sources,development_inputs=inputs,recipes=recipes())
        run_id=key(self.fingerprint)[:16];self.root=Path(c['output_root'])/run_id
        command=f'python -m experiments.discrete_weighting.run --config {args.config} --phase all --resume'
        manifest=self.fingerprint|dict(run_id=run_id,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
               planned=dict(phase1=3*len(recipes()),phase2='locked-family predeclared matrix; three seeds',seed_count=3))
        if args.resume_from and not self.root.exists():
            old=Path(args.resume_from);prior_manifest=json.loads((old/'manifest.json').read_text())
            assert not (old/'SELECTION_LOCK.json').exists(), 'Migration only allowed before selection lock'
            assert prior_manifest['config']==c and prior_manifest['development_inputs']==inputs
            assert all(prior_manifest['source'].get(k)==v for k,v in sources.items() if not k.endswith('/run.py'))
            self.root.mkdir(parents=True)
            for p in old.iterdir():
                if p.name in ('manifest.json','state.json','HANDOFF_STATUS.md','NEXT_AGENT_PROMPT.md','REPORT.md','BLOCKERS.md','failed'):continue
                if p.is_dir():shutil.copytree(p,self.root/p.name)
                else:shutil.copyfile(p,self.root/p.name)
            dump(self.root/'migration.json',dict(source_run=old.as_posix(),source_manifest_sha256=sha(old/'manifest.json'),reason='Control lookup restricted to full 15-member models; prior reliability/calibration computations unchanged. All migrated markers revalidated.'))
        self.store=Store(self.root,manifest,command,args.max_tasks,args.max_minutes)
        if not (self.root/'config.lock.yaml').exists():atomic(self.root/'config.lock.yaml',yaml.safe_dump(c))
        if yaml.safe_load((self.root/'config.lock.yaml').read_text())!=c:raise ValueError('Config lock mismatch')
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:setattr(self.store,'stop',True))
        atomic(Path(c['output_root'])/'LATEST_RUN.txt',self.root.as_posix())

    def task(self,t,phase,deps,fn):return self.store.task(t,phase,deps,fn)

    def prepare(self,seed,folds=3):
        tag=f'oof:{seed}:{folds}'
        if tag in self.memory:return self.memory[tag]
        base=self.prior if folds==3 else self.prior/'crossfit5'
        for fold in range(folds):
            def verify(fold=fold):
                source=base/f'seed{seed}/predictions_fold{fold}.npz'
                with np.load(source,allow_pickle=False) as f:
                    prov=json.loads(str(f['provenance']));p=f['probs'];ids=f['sample_id']
                    assert p.shape==(len(ids),15,2) and len(set(ids))==len(ids)
                    assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(-1),1,atol=2e-6)
                    assert ids.tolist()==prov['held_ids'] and not set(ids)&set(prov['fit_ids'])
                    assert sha(base/f'seed{seed}/heads_fold{fold}.pt')==prov['heads_sha256']
                path=self.root/f'cache/oof/seed{seed}/folds{folds}/fold{fold}.json';dump(path,dict(source=source.as_posix(),sha256=sha(source),provenance=prov,classes=[0,1]));return {},[path]
            self.task(f'{tag}:fold{fold}','audit',[],verify)
        path=self.root/f'cache/oof/seed{seed}/folds{folds}/assembled.npz'
        def assemble():
            blocks=[]
            for fold in range(folds):
                with np.load(base/f'seed{seed}/predictions_fold{fold}.npz',allow_pickle=False) as f:blocks.append({k:f[k] for k in ('probs','y','sample_id')})
            data={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]};assert len(set(data['sample_id']))==len(data['y'])==1199
            assert set(np.unique(data['y']))=={0,1};npz(path,**data);return {},[path]
        self.task(tag,'audit',[f'{tag}:fold{f}' for f in range(folds)],assemble)
        with np.load(path,allow_pickle=False) as f:data=dict(f)
        self.memory[tag]=data;return data

    def split(self,data,seed,cfg):
        if 'split_seed' in cfg:
            from sklearn.model_selection import train_test_split
            a,b=train_test_split(np.arange(len(data['y'])),test_size=.2,stratify=data['y'],random_state=cfg['split_seed'])
        else:
            audit=json.loads((self.prior/f'seed{seed}/split_audit.json').read_text());index={s:i for i,s in enumerate(data['sample_id'])}
            a=np.array([index[s] for s in audit['gate_train_ids']]);b=np.array([index[s] for s in audit['gate_select_ids']])
        assert not set(data['sample_id'][a])&set(data['sample_id'][b]);return a,b

    def fit(self,cfg,seed,phase='phase1',label=''):
        folds=cfg.get('folds',3);data=self.prepare(seed,folds);a,b=self.split(data,seed,cfg);m=cfg.get('members',15)
        p=data['probs'][:,:m];y=data['y'];ck=key(cfg)[:16];tid=f'fit:{phase}:{seed}:{ck}'
        folder=self.root/f'fits/{phase}/seed{seed}/{ck}';model_path=folder/'model.joblib';temp_path=folder/'temperature.json'
        def calibration():
            raw=p[a];mode=cfg.get('calibration','raw')
            t=fit_temperature(raw,y[a],mode=='shared') if mode!='raw' else np.ones(m)
            dump(temp_path,dict(temperature=t.tolist(),mode=mode,fit_ids=data['sample_id'][a].tolist(),grid=[.03,8,41]));return {},[temp_path]
        calid='calibrate:'+tid
        self.task(calid,'calibrate',[f'oof:{seed}:{folds}'],calibration)
        def fitting():
            start=time.perf_counter();t=np.array(json.loads(temp_path.read_text())['temperature'])
            model=Rule(cfg).fit(p[a],y[a],t);buff=io.BytesIO();joblib.dump(model,buff);atomic(model_path,buff.getvalue())
            artifact=joblib.load(model_path);mix,w=artifact.predict(p[b]);assert np.allclose(w.sum(1),1) and np.allclose(mix.sum(1),1)
            table=folder/'table.npz';npz(table,pooled=model.pool,head=model.table,counts=model.counts,edges=model.edges,temperature=t)
            info=folder/'fit.json';dump(info,dict(train_seconds=time.perf_counter()-start,train_count=len(a),select_count=len(b),class_counts=np.bincount(y[a]).tolist(),threshold=model.threshold))
            return {},[model_path,table,info]
        self.task(tid,'fit',[calid],fitting)
        sid=f'select:{phase}:{seed}:{ck}'
        def score():
            model=joblib.load(model_path);mix,w=model.predict(p[b]);met=self.measure(mix,w,p[b],y[b],model.threshold,calibrate(p[b],model.temperature).mean(1))
            result=dict(seed=seed,config=cfg,method=cfg['family'],ablation=label,eligible=True,**met,
                        **json.loads((folder/'fit.json').read_text()))
            output=folder/'select.npz';npz(output,probs=mix,weights=w,sample_id=data['sample_id'][b],y=y[b])
            return result,[output]
        rec=self.task(sid,phase,[tid],score)
        return rec,model_path

    def measure(self,mix,w,p,y,threshold,representation_uniform,group=None):
        met=metrics(mix,y,group,threshold);pred=(mix[:,1]>threshold).astype(int);base=p.mean(1);bp=base.argmax(1);correct=pred==y;bc=bp==y
        met.update(delta_raw_pp=100*(met['accuracy']-bc.mean()),delta_representation_uniform_pp=100*(met['accuracy']-(representation_uniform.argmax(1)==y).mean()),
                   effective_members=float(np.exp(-(w*np.log(w.clip(1e-12))).sum(1)).mean()),largest_weight_mean=float(w.max(1).mean()),largest_weight_max=float(w.max()),
                   wrong_expert_weight=float((w*(p.argmax(-1)!=y[:,None])).sum(1).mean()),flips_0_to_1=int(((bp==0)&(pred==1)).sum()),flips_1_to_0=int(((bp==1)&(pred==0)).sum()),
                   corrections=int((correct&~bc).sum()),breaks=int((~correct&bc).sum()),class_confusion=[[int(((y==a)&(pred==b)).sum()) for b in (0,1)] for a in (0,1)])
        return met

    def execute(self):
        def audit():
            path=self.root/'DATA_AUDIT.md';atomic(path,'# Data audit\n\nReuses 3/5-fold OOF caches. Per-fold tasks verify head SHA, held IDs disjoint from fit IDs, class order [0,1], finite normalized [samples,15,2] arrays. Fixed 80/20 split IDs are inherited from the prior class-stratified protocol. No groups enter fitting or selection. Test is previously explored and is only read after the immutable selection lock. Shared frozen backbone historical training provenance is unavailable; all results are conditional on its supplied training-only origin. No untouched external dataset was found/claimed. Calibration searches .03–8 using development NLL only. The tiny 240-sample select pool and prespecified search imply selection optimism.\n');return {},[path]
        self.task('environment_audit','audit',[],audit)
        phase1=[]
        for cfg in recipes():
            for seed in self.c['seeds']:
                try:phase1.append(self.fit(cfg,seed)[0])
                except InterruptedError:raise
                except Exception:continue
        for seed in self.c['seeds']:
            data=self.prepare(seed);a,b=self.split(data,seed,{})
            def diagnostic(seed=seed,data=data,b=b):
                p=data['probs'][b];y=data['y'][b];wrong=p.argmax(-1)!=y[:,None]
                path=self.root/f'diagnostics/seed{seed}.npz';npz(path,error_overlap=wrong.astype('int64').T@wrong.astype('int64'),sample_id=data['sample_id'][b])
                return dict(seed=seed,method='X0',eligible=False,oracle_coverage=float((~wrong).any(1).mean()),all_wrong=int(wrong.all(1).sum()),unanimous=int((p.argmax(-1)==p.argmax(-1)[:,:1]).all(1).sum())),[path]
            self.task(f'diagnostic:{seed}','diagnostic',[f'oof:{seed}:3'],diagnostic)
            for family in ('D19','D1'):
                matches=[]
                for result_path in (self.prior/'fits').glob('*/result.json'):
                    row=json.loads(result_path.read_text())
                    if row['method']==family and row['seed']==seed and row.get('members')==15 and row['config']==({'kl':0.,'lr':.001} if family=='D19' else {'kl':.01,'lr':.001}):matches.append(result_path.parent/'model.joblib')
                if not matches:continue
                def control(seed=seed,data=data,b=b,family=family,path=matches[0]):
                    model=joblib.load(path);mix=model.predict({'probs':data['probs'][b]})
                    met=metrics(mix,data['y'][b]);out=self.root/f'diagnostics/{family}_seed{seed}.npz';npz(out,probs=mix,sample_id=data['sample_id'][b],y=data['y'][b])
                    return dict(seed=seed,method='C1_'+family,eligible=False,source_sha256=sha(path),**met),[out]
                self.task(f'control:{family}:{seed}','phase1',[f'oof:{seed}:3'],control)
        grouped={}
        for r in phase1:grouped.setdefault(key(r['result']['config']),[]).append(r['result'])
        ranked=[]
        for rs in grouped.values():
            if len(rs)!=len(self.c['seeds']):continue
            ranked.append(dict(config=rs[0]['config'],accuracy=float(np.mean([r['accuracy'] for r in rs])),balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in rs])),nll=float(np.mean([r['nll'] for r in rs]))))
        ranked.sort(key=lambda r:(-r['accuracy'],-r['balanced_accuracy'],r['nll']));csvwrite(self.root/'results/phase1_ranked.csv',ranked)
        if not ranked:raise RuntimeError('No complete candidate family available')
        lockpath=self.root/'SELECTION_LOCK.json'
        def locking():
            dump(lockpath,dict(winner=ranked[0],seeds=self.c['seeds'],manifest_sha256=sha(self.root/'manifest.json'),test_read=False,ablations=ablations(ranked[0]['config'])));return {},[lockpath]
        self.task('lock_selection','lock',[r['task_id'] for r in phase1],locking)
        lock=json.loads(lockpath.read_text());winner=lock['winner']['config']
        if self.args.phase=='1':return
        specs=lock['ablations'];jobs=[]
        for spec in specs:
            inactive=winner['family'] in ('D0a','D0b','D1') and spec['ablation'] not in ('A0','A1','A12','A16','A18','A19')
            if inactive:
                def na(spec=spec):
                    path=self.root/f'not_applicable/{key(spec)[:16]}.json';dump(path,dict(status='N/A',reason='Locked winner is a uniform/threshold control; reliability component is absent',**spec));return dict(status='N/A',**spec),[path]
                self.task('na:'+key(spec)[:16],'phase2', ['lock_selection'],na);continue
            for seed in self.c['seeds']:
                try:
                    rec,path=self.fit(spec['config'],seed,'phase2',spec['ablation']+':'+spec['variant']);jobs.append((spec,seed,rec,path))
                except InterruptedError:raise
                except Exception:continue
        # Both the recipe and entire ablation matrix are frozen before this point.
        for spec,seed,rec,path in jobs:
            cfg=spec['config'];ck=key(cfg)[:16];tid=f'test:{seed}:{ck}'
            def test_score(spec=spec,seed=seed,rec=rec,path=path,cfg=cfg,ck=ck):
                with np.load(self.prior/f'seed{seed}/predictions_test.npz',allow_pickle=False) as f:
                    p=f['probs'][:,:cfg.get('members',15)];y=f['y'];ids=f['sample_id']
                with np.load(self.prior/'features/test.npz',allow_pickle=False) as f:
                    assert np.array_equal(ids,f['sample_id']);group=f['group']
                assert not set(ids)&set(self.prepare(seed,cfg.get('folds',3))['sample_id'])
                model=joblib.load(path);mix,w=model.predict(p);cal=calibrate(p,model.temperature)
                met=self.measure(mix,w,p,y,model.threshold,cal.mean(1),group)
                met['paired_ci_raw_pp']=paired_ci(mix,p.mean(1),y,self.c['bootstrap'],seed,model.threshold)
                base=p.mean(1).argmax(1);pred=(mix[:,1]>model.threshold).astype(int)
                met['majority_overturns']=int(((pred!=base)&((p.argmax(-1)==base[:,None]).sum(1)>=np.ceil(.8*p.shape[1]))).sum())
                output=self.root/f'test_results/seed{seed}/{ck}.npz';npz(output,probs=p,calibrated_probs=cal,weights=w,gate_probs=mix,y=y,sample_id=ids,group=group)
                cases=self.root/f'test_results/seed{seed}/{ck}_flips.csv';csvwrite(cases,[dict(sample_id=str(ids[i]),y=int(y[i]),group=int(group[i]),old=int(base[i]),new=int(pred[i]),corrected=bool(pred[i]==y[i])) for i in np.flatnonzero(pred!=base)])
                if not cases.exists():atomic(cases,'sample_id,y,group,old,new,corrected\n')
                return dict(seed=seed,config=cfg,method=cfg['family'],ablation=spec['ablation']+':'+spec['variant'],input_cache_sha256=sha(self.prior/f'seed{seed}/predictions_test.npz'),**met),[output,cases]
            try:self.task(tid,'test',['lock_selection',rec['task_id']],test_score)
            except InterruptedError:raise
            except Exception:continue
        self.final_report(lock)

    def final_report(self,lock):
        def reporting():
            rows=[r['result'] for r in self.store.valid.values() if r['phase']=='test']
            selected=[r for r in rows if r['config']==lock['winner']['config']]
            if len(selected)!=3:raise RuntimeError('Winner test tasks incomplete; inspect failed/')
            lines=['# Discrete reliability v2 research report','', 'Exploratory test, previously inspected. Three head seeds share one frozen backbone. Accuracy selection uses only the fixed OOF selection pools; groups are post-hoc only.', '',
                   'Locked winner: '+json.dumps(lock['winner'])+'','', '| Seed | Accuracy % | Delta raw pp | BA % | WGA % |','|---|---:|---:|---:|---:|']
            for r in selected:lines.append(f'| {r["seed"]} | {r["accuracy"]*100:.3f} | {r["delta_raw_pp"]:+.3f} | {r["balanced_accuracy"]*100:.3f} | {r["wga"]*100:.3f} |')
            for metric in ('accuracy','balanced_accuracy','wga','delta_raw_pp'):
                v=np.array([r[metric] for r in selected]);scale=1 if metric.endswith('pp') else 100
                lines+=['',f'{metric}: {v.mean()*scale:.4f} ± {v.std(ddof=1)*scale:.4f} (sample SD).']
            lines+=['','Reliability is a heuristic, not a proof of optimal mixture weighting. Small selection pools and repeated historical test use limit claims. No independent backbone seeds or untouched confirmatory dataset. Full candidate/per-seed metrics and ablations are in results/*.csv; test NPZ files retain [samples,members,classes], weights, labels, IDs and groups. Count-matched raw and calibrated uniform controls are computed per task. Temperature tables record the expanded .03–8 training-only grid. No CNN/head retraining. Phase-1 configs were frozen in manifest; Phase-2 configs in SELECTION_LOCK.json.', '',
                    'Ablations that coincide with an existing recipe are deduplicated by config hash; requested mappings are retained in SELECTION_LOCK.json. Every fitted table, calibration and selection score is an independent committed task. Current split sensitivity A19 uses predeclared class-only splits; A18 reuses verified prior five-fold caches. C1 and X0 comparisons/diagnostics are recorded separately if available. See DATA_AUDIT.md for provenance limitations.']
            report=self.root/'final_report.md';atomic(report,'\n'.join(lines)+'\n')
            user=self.root/'RESULTS_FOR_USER.md';atomic(user,'# Kết quả discrete reliability v2\n\nWinner khóa từ OOF selection: '+json.dumps(lock['winner']['config'])+'\n\nAccuracy test trung bình: '+str(round(np.mean([r['accuracy'] for r in selected])*100,4))+'%. Chênh lệch với uniform cùng heads: '+str(round(np.mean([r['delta_raw_pp'] for r in selected]),4))+' điểm %.\n\nBa seed dùng chung backbone; test đã được khảo sát trước nên kết quả chỉ mang tính thăm dò. Đọc final_report.md và các bảng đầy đủ để xem WGA và độ biến thiên.\n')
            return {},[report,user]
        self.task('report','report',[t for t,r in self.store.valid.items() if r['phase']=='test'],reporting)
        failed=list((self.root/'failed').glob('*.json'))
        unresolved=[p for p in failed if json.loads(p.read_text())['task_id'] not in self.store.valid]
        self.store.refresh('BLOCKED' if unresolved else 'COMPLETE','inspect failed tasks' if unresolved else 'none')
        atomic(self.root/'BLOCKERS.md','Unresolved candidate failures: '+str([str(p) for p in unresolved]) if unresolved else 'No missing data/checkpoints. Backbone historical training provenance remains unverified.\n')
        atomic(self.root/'REPORT.md',(self.root/'final_report.md').read_text())
        print('Finished. Output: '+str(self.root));print(self.store.command)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='experiments/discrete_weighting/config.yaml');p.add_argument('--resume',action='store_true');p.add_argument('--resume-from');p.add_argument('--phase',default='all',choices=['all','1','2']);p.add_argument('--status',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--max-tasks',type=int);p.add_argument('--max-minutes',type=float)
    args=p.parse_args();c=yaml.safe_load(Path(args.config).read_text());run=Run(c,args)
    if args.status or args.dry_run:
        print(json.dumps(json.loads((run.root/'state.json').read_text()),indent=2));return
    try:run.execute()
    except InterruptedError:print('Stopped safely; '+run.store.command)
    except BaseException as error:
        run.store.refresh('BLOCKED',str(error));atomic(run.root/'BLOCKERS.md',traceback.format_exc());raise


if __name__=='__main__':main()
