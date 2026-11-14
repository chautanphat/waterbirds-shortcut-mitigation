import argparse
import io
import json
from pathlib import Path
import signal
import subprocess
import importlib.metadata
import joblib
import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from experiments.discrete_weighting.store import atomic,dump,key,sha
from experiments.discrete_weighting.run import csvwrite
from experiments.weighting.metrics import metrics
from .journal import Journal
from .solver import clusters,solve


def save_npz(path,**data):
    b=io.BytesIO();np.savez_compressed(b,**data);atomic(path,b.getvalue())
    with np.load(path,allow_pickle=False) as f:
        for k,v in data.items():np.testing.assert_array_equal(f[k],v)


def score(p,y,g):
    if set(np.unique(g))!={0,1,2,3}:raise ValueError('Missing evaluation group')
    result=metrics(p,y,g);result['worst_class_accuracy']=min(result['per_class'].values());return result


class Verification:
    def __init__(self,c,args):
        self.c=c;self.args=args;self.bundle=Path(c['development_bundle']);self.prior=Path(c['prior'])
        inputs={}
        for p in sorted(self.bundle.rglob('*')):
            if p.is_file():inputs[p.as_posix()]=sha(p)
        sources={p.as_posix():sha(p) for p in Path('experiments/wga_verification').glob('*.py')}
        for p in ('experiments/discrete_weighting/store.py','experiments/discrete_weighting/run.py','experiments/weighting/metrics.py'):sources[p]=sha(p)
        optional={}
        for name in ('source_analysis','source_archive'):
            p=Path(c[name]);optional[name]=dict(path=p.as_posix(),available=p.is_file(),sha256=sha(p) if p.is_file() else None)
        manifest=dict(config=c,inputs=inputs,source=sources,source_reference=optional,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                      versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','torch')})
        run_id=key(manifest)[:12];self.root=Path(c['output_root'])/run_id
        self.command=f'python -m experiments.wga_verification.run --config {args.config} --phase all --resume'
        self.j=Journal(self.root,manifest,self.command,args.max_tasks)
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:setattr(self.j,'stop',True))
        if not (self.root/'frozen.yaml').exists():atomic(self.root/'frozen.yaml',yaml.safe_dump(c))
        for p,h in sources.items():
            assert sha(p)==h
            dest=self.root/'source_snapshot'/p
            if not dest.exists():atomic(dest,Path(p).read_bytes())
        atomic(Path(c['output_root'])/'LATEST_RUN.txt',self.root.as_posix())

    def integrity(self,seed):
        path=self.root/f'verification/seed{seed}/integrity.json'
        def work():
            with np.load(self.bundle/f'seed{seed}/meta_train.npz',allow_pickle=False) as f:a=dict(f)
            with np.load(self.bundle/f'seed{seed}/meta_select.npz',allow_pickle=False) as f:b=dict(f)
            assert len(a['y'])==959 and len(b['y'])==240
            assert not set(a['sample_id'])&set(b['sample_id'])
            all_ids=np.concatenate([a['sample_id'],b['sample_id']]);assert len(set(all_ids))==1199
            prov=json.loads((self.bundle/f'seed{seed}/fold_provenance.json').read_text())
            for d in (a,b):
                assert d['probs'].shape==(len(d['y']),15,2)
                assert np.isfinite(d['probs']).all() and (d['probs']>=0).all() and np.allclose(d['probs'].sum(-1),1,atol=2e-6)
                assert d['embedding'].shape==(len(d['y']),2048) and np.isfinite(d['embedding']).all()
                assert d['classes'].tolist()==[0,1] and np.all(d['group']//2==d['y'])
                for fold in range(3):
                    pr=prov['folds'][fold];ids=set(d['sample_id'][d['fold']==fold]);assert ids<=set(pr['held_ids']) and not ids&set(pr['fit_ids'])
                    assert sha(self.prior/f'seed{seed}/heads_fold{fold}.pt')==pr['heads_sha256']
            bundle_manifest=json.loads((self.bundle/'manifest.json').read_text())
            for entry in bundle_manifest['files']:
                if entry['seed']==seed:assert sha(self.bundle/entry['path'])==entry['sha256']
            dump(path,dict(status='PASS',seed=seed,unique_ids=1199,groups=[0,1,2,3],fold_slot_mismatch='Documented: each head slot corresponds to different parameters across predicting OOF folds; no transfer compatibility is assumed.'))
            return dict(seed=seed,integrity='PASS'),[path]
        self.j.task(f'integrity:{seed}',[],work)

    def fit_pool(self,seed,tag,p,y,z,ids):
        folder=self.root/f'verification/seed{seed}/{tag}'
        prep_id=f'preprocess:{seed}:{tag}';model_path=folder/'preprocess.joblib';cluster_path=folder/'cluster.npz'
        def preprocessing():
            pca,km,labels=clusters(z)
            b=io.BytesIO();joblib.dump(dict(pca=pca,kmeans=km,fit_ids=ids.tolist()),b);atomic(model_path,b.getvalue())
            save_npz(cluster_path,cluster=labels,fit_ids=ids)
            return dict(seed=seed,cluster_counts=np.bincount(labels,minlength=2).tolist()),[model_path,cluster_path]
        self.j.task(prep_id,[f'integrity:{seed}'],preprocessing)
        weight_path=folder/'weights.npz';audit_path=folder/'solver.json';solver_id=f'solver:{seed}:{tag}'
        def fitting():
            with np.load(cluster_path,allow_pickle=False) as f:labels=f['cluster']
            w,audit=solve(p,y,labels,True)
            try:
                direct,da=solve(p,y,labels,False)
                difference=abs(audit['objective']-da['objective']);da['objective_gap_vs_epigraph']=difference
                direct_valid=difference<=1e-6
            except RuntimeError as error:direct=np.full(15,np.nan);da=dict(success=False,message=str(error));direct_valid=False
            save_npz(weight_path,weights=w,direct_weights=direct)
            dump(audit_path,dict(epigraph=audit,direct=da,direct_objective_agreement=direct_valid))
            return dict(seed=seed,solver_success=audit['success'],direct_agreement=direct_valid,objective=audit['objective'],weight_min=float(w.min()),weight_max=float(w.max()),effective_members=float(np.exp(-(w*np.log(w.clip(1e-15))).sum())),top_members=np.argsort(-w)[:5].tolist()),[weight_path,audit_path]
        rec=self.j.task(solver_id,[prep_id],fitting)
        with np.load(weight_path,allow_pickle=False) as f:w=f['weights']
        return w,solver_id,rec['metrics']

    def legacy(self,seed):
        with np.load(self.bundle/f'seed{seed}/meta_train.npz',allow_pickle=False) as f:a=dict(f)
        with np.load(self.bundle/f'seed{seed}/meta_select.npz',allow_pickle=False) as f:b=dict(f)
        w,dep,solver=self.fit_pool(seed,'legacy',a['probs'],a['y'],a['embedding'],a['sample_id'])
        path=self.root/f'verification/seed{seed}/metaselect_metrics.json'
        def scoring():
            uniform=b['probs'].mean(1);weighted=np.einsum('nmc,m->nc',b['probs'],w)
            result=dict(seed=seed,uniform=score(uniform,b['y'],b['group']),candidate=score(weighted,b['y'],b['group']),solver=solver)
            dump(path,result);pred=self.root/f'verification/seed{seed}/metaselect_predictions.npz'
            save_npz(pred,uniform=uniform,candidate=weighted,y=b['y'],group=b['group'],sample_id=b['sample_id'])
            return result,[path,pred]
        return self.j.task(f'legacy_score:{seed}',[dep],scoring)['metrics']

    def cv5(self,seed):
        blocks=[]
        for split in ('meta_train','meta_select'):
            with np.load(self.bundle/f'seed{seed}/{split}.npz',allow_pickle=False) as f:blocks.append(dict(f))
        data={k:np.concatenate([d[k] for d in blocks]) for k in ('probs','y','group','sample_id','embedding')}
        p,y,z=data['probs'],data['y'],data['embedding'];uniform=p.mean(1);predictions=np.zeros_like(uniform,dtype=float);weights=[];deps=[]
        cv=StratifiedKFold(5,shuffle=True,random_state=3001)
        for fold,(train,held) in enumerate(cv.split(z,y)):
            assert not set(data['sample_id'][train])&set(data['sample_id'][held])
            w,dep,solver=self.fit_pool(seed,f'cv5fold{fold}',p[train],y[train],z[train],data['sample_id'][train])
            path=self.root/f'verification/seed{seed}/cv5fold{fold}/predictions.npz'
            def infer(path=path,w=w,held=held):
                q=np.einsum('nmc,m->nc',p[held],w);save_npz(path,probabilities=q,sample_id=data['sample_id'][held],held_indices=held)
                return dict(seed=seed,fold=fold,count=len(held)),[path]
            task=f'cv5infer:{seed}:{fold}';self.j.task(task,[dep],infer);deps.append(task)
            with np.load(path,allow_pickle=False) as f:predictions[f['held_indices']]=f['probabilities']
            weights.append(dict(fold=fold,**{f'head{j:02d}':float(v) for j,v in enumerate(w)},**solver))
        path=self.root/f'verification/seed{seed}/cv5_metrics.json'
        def pooling():
            u=uniform.argmax(1);q=predictions.argmax(1);correct_u=u==y;correct_q=q==y
            result=dict(seed=seed,uniform=score(uniform,y,data['group']),candidate=score(predictions,y,data['group']),
                        flips=int((u!=q).sum()),corrected=int((~correct_u&correct_q).sum()),broken=int((correct_u&~correct_q).sum()))
            dump(path,result);npz_path=path.with_name('cv5_predictions.npz');save_npz(npz_path,uniform=uniform,candidate=predictions,y=y,group=data['group'],sample_id=data['sample_id'])
            csv_path=path.with_name('weights_by_fold.csv');csvwrite(csv_path,weights)
            return result,[path,npz_path,csv_path]
        return self.j.task(f'cv5_score:{seed}',deps,pooling)['metrics']

    def execute(self):
        with threadpool_limits(1):
            results=[];legacy=[]
            for seed in self.c['seeds']:
                self.integrity(seed);legacy.append(self.legacy(seed));results.append(self.cv5(seed))
        def decide():
            expected=np.array([[94.7456,95.0792,91.7293,93.9850],[94.8290,95.2460,92.2747,93.1330],[95.1626,95.7465,92.4812,94.7368]])
            comparisons=[];reasons=[]
            for row,target in zip(results,expected):
                actual=100*np.array([row['uniform']['accuracy'],row['candidate']['accuracy'],row['uniform']['wga'],row['candidate']['wga']]);difference=float(abs(actual-target).max())
                comparisons.append(dict(seed=row['seed'],uniform_AA=actual[0],candidate_AA=actual[1],uniform_WGA=actual[2],candidate_WGA=actual[3],max_discrepancy_pp=difference))
                if difference>self.c['tolerance_pp']:reasons.append(f'CV5 seed {row["seed"]}: discrepancy {difference:.6f} pp > 0.03')
            for label,field,target in [('uniform','accuracy',95.00),('uniform','wga',90.72),('candidate','accuracy',95.97),('candidate','wga',92.76)]:
                actual=100*np.mean([r[label][field] for r in legacy])
                if abs(actual-target)>.03:reasons.append(f'Legacy {label} {field}: {actual:.6f} versus {target:.2f}')
            flips={k:sum(r[k] for r in results) for k in ('flips','corrected','broken')}
            if flips!=dict(flips=48,corrected=32,broken=16):reasons.append('Flip counts do not match: '+str(flips))
            solver_rows=[]
            for p in sorted((self.root/'verification').glob('seed*/*/solver.json')):
                d=json.loads(p.read_text());solver_rows.append(dict(path=p.relative_to(self.root).as_posix(),**d['epigraph'],direct_agreement=d['direct_objective_agreement']))
                if not d['direct_objective_agreement']:reasons.append('Direct/epigraph solver check failed: '+p.as_posix())
            missing=[self.c[n] for n in ('source_analysis','source_archive') if not Path(self.c[n]).is_file()]
            if missing:reasons.append('Required original analysis/reproduction source unavailable: '+', '.join(missing)+'; original script and original SHA manifest cannot be independently executed/compared.')
            deltaAA=float(100*np.mean([r['candidate']['accuracy']-r['uniform']['accuracy'] for r in results]));deltaWGA=float(100*np.mean([r['candidate']['wga']-r['uniform']['wga'] for r in results]))
            gate=dict(status='FAIL' if reasons else 'PASS',reasons=reasons,mean_delta_AA_pp=deltaAA,mean_delta_WGA_pp=deltaWGA,flips=flips,source_comparison_completed=not missing)
            path=self.root/'verification/GATE_A.json';dump(path,gate)
            comp=self.root/'verification/reference_comparison.csv';csvwrite(comp,comparisons)
            sol=self.root/'verification/solver_audit.csv';csvwrite(sol,solver_rows)
            text='# Gate A independent reproduction\n\n'+json.dumps(gate,indent=2)+'\n\n'+json.dumps(comparisons,indent=2)+'\n\nLegacy metrics: '+json.dumps(legacy,indent=2)+'\n\nThis reproduces a non-nested sensitivity analysis of existing 3-fold OOF predictions. Head slots differ across original folds; static slot weights are not assumed compatible with final refits. Prior method nomination used development group metrics; no untouched confirmation is claimed. No historical test arrays were opened. Original source scripts were not available, and no replacement script is represented as the original.\n'
            report=self.root/'verification/REFERENCE_REPRODUCTION.md';atomic(report,text)
            integrity=self.root/'verification/integrity_report.md';atomic(integrity,'# Integrity audit\n\nAll 3 seed caches verified: normalized raw [samples,15,2], class order, y/group mapping, exact IDs, 1,199 unique validation examples, meta split disjointness, per-fold fit/held exclusion and checkpoint hashes. Original export manifest hashes verified. Head slots across folds are different parameter sets, as documented. Original external analysis/archive absent; their original hashes cannot be checked.\n')
            return gate,[path,comp,sol,report,integrity]
        gate=self.j.task('gate_A',[f'cv5_score:{s}' for s in self.c['seeds']]+[f'legacy_score:{s}' for s in self.c['seeds']],decide)['metrics']
        blocked=gate['status']!='PASS'
        # Deliberately fail closed: missing reference evidence forbids stage B/integration.
        if blocked or self.args.phase=='audit':
            reason='Gate A failed; nested evaluation was not run and production integration is prohibited.' if blocked else 'Audit-only invocation; nested verification remains pending.'
            dump(self.root/'verification/GATE_B.json',dict(status='NOT_RUN_GATE_A_FAIL' if blocked else 'PENDING',reason=reason))
            atomic(self.root/'nested/NESTED_EVALUATION.md','# Nested evaluation\n\n'+reason+'\n\nNo head refits or production changes were performed. The requested real-head interruption acceptance test belongs before nested fitting and was not reached.\n')
            atomic(self.root/'BLOCKERS.md','\n'.join(gate['reasons']) if blocked else 'Nested stage pending.')
            atomic(self.root/'FINAL_SUMMARY.md','# Verify-first terminal summary\n\n'+json.dumps(gate,indent=2)+'\n\n'+reason+'\n\nResume command: '+self.command+'\n\nProvide original DPE_WGA_Deep_Analysis.md and DPE_WGA_Audit_Repro.zip to enable source-script comparison. Adding them changes the fingerprint and creates a new run; preserve this run as evidence.\n')
            self.j.status('GATE_A_FAILED' if blocked else 'AUDIT_COMPLETE',reason,gate_A=gate['status'],gate_B='NOT_RUN',production_integration='PROHIBITED',measured_delta_WGA_pp=gate['mean_delta_WGA_pp'],measured_delta_AA_pp=gate['mean_delta_AA_pp'])
            print(json.dumps(gate));print(self.root);return
        raise RuntimeError('Gate B not yet executed: production remains prohibited until genuine nested verification and interruption acceptance pass')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='experiments/wga_verification/frozen.yaml');p.add_argument('--phase',choices=['all','audit','nested','integrate'],default='all');p.add_argument('--resume',action='store_true');p.add_argument('--status',action='store_true');p.add_argument('--max-tasks',type=int)
    args=p.parse_args();run=Verification(yaml.safe_load(Path(args.config).read_text()),args)
    if args.status:print((run.root/'state.json').read_text());return
    try:run.execute()
    except InterruptedError:print('Stopped safely. '+run.command)
    except Exception as e:
        import traceback
        atomic(run.root/'BLOCKERS.md',traceback.format_exc());run.j.status('BLOCKED',str(e),production_integration='PROHIBITED');raise


if __name__=='__main__':main()
