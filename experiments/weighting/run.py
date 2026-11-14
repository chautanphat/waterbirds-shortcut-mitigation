"""Checkpointed Phase 1 -> locked winner -> Phase 2 -> one final test stage."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import joblib
import numpy as np
import torch
import yaml
from .cache import digest, identity, write_json, save_predictions, load_predictions
from .data import (seed_all, extract, prepare_meta, fit_heads, predict_heads, assert_disjoint)
from .methods import IDS, NEURAL, Predictor, candidates, oracle_coverage
from .metrics import metrics, paired_ci, weight_diagnostics


def write_csv(path, rows):
    if not rows:
        rows=[]
    columns=list(dict.fromkeys(k for r in rows for k in r)) or ['status','reason']
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    tmp.replace(path)


def sub(data,members=None):
    if members is None:return data
    return {k:v[:,:members] if k in ('probs','logits','distances') else v for k,v in data.items()}


def ablations(method,cfg):
    """Prespecified diagnostic variants. Never selected using test results."""
    neural=method in NEURAL
    mixture=method not in ('S6','S7','R4','R8')
    dynamic=neural or method in ('H5','R5','R6','R7','D5','D6','D7','D8','D9','D10','R1','R2','R3')
    result=[]
    def add(id_,label,change=None,reason=None,new_method=None,members=None):
        result.append(dict(ablation=id_,variant=label,method=new_method or method,
                           config=cfg|dict(change or {}),members=members,status='N/A' if reason else 'pending',reason=reason))
    add('A0','full_winner')
    add('A1','remove_calibration',{'calibrate':False},None if cfg.get('calibrate') or method=='D19' else 'Winner has no temperature calibration',new_method='D2' if method=='D19' else None)
    add('A2','uniform',new_method='U0')
    add('A3','remove_KL',{'kl':0.},None if cfg.get('kl',0)>0 else 'Winner KL coefficient already zero')
    add('A4','remove_teacher',{'eta':0.},None if method=='D13' else 'No label-conditioned teacher in winner')
    for id_,feature,used in [('A5','confidence',neural),('A6','margin',neural),('A7','distance',method=='D3'),
                             ('A8','context',method=='D4'),('A9','embedding',method=='D12')]:
        add(id_,'remove_'+feature,{'remove':[feature]},None if used else 'Feature not used by winning family')
    add('A10','average_dynamic_weights',{'static_average':True},None if dynamic else 'Winner has sample-independent weights or no mixture weights')
    add('A11','discrete_weight_mapping',{'weight_binning':True},None if dynamic else 'Winner has no continuous dynamic mapping')
    for count in (3,5,10,15):add('A12',f'members_{count}',members=count)
    for capacity in ('linear','16','32x16','64x32'):
        add('A13','capacity_'+capacity,{'capacity':capacity},None if neural else 'Winner is not a neural gate')
    for k in (1,3,5,None):
        add('A14','topk_'+str(k or 'all'),{'top_k':k},None if mixture else 'Winner departs from convex soft averaging')
    for loss in ('brier','balanced'):
        add('A15','loss_'+loss,{'loss':loss},None if neural or method in ('S1','S3','S4','S5','S6') else 'Winner has no compatible differentiable training objective')
    add('A16','toggle_CVaR',{'tail':0. if method=='D14' else .2},None if neural else 'Winner is not a neural gate')
    add('A17','toggle_temperature',{'calibrate':not (cfg.get('calibrate') or method=='D19')},
        None if mixture else 'Calibration ablation implemented only for convex mixtures',new_method='D2' if method=='D19' else None)
    add('A18','alternative_crossfit_count',reason='Not run: repeats would require a separate refitted head/cache protocol; bounded compute budget')
    add('A19','random_weight_control',{'random_weights':True},None if mixture else 'Winner has no mixture weights')
    add('A20','diverse_seeds',reason='Covered by the three source/head/gate seed reruns in A0; not an additional independent experiment')
    return result


class Study:
    def __init__(self,config,resume):
        self.c=config;self.root=Path(config['output_dir']);self.root.mkdir(parents=True,exist_ok=True)
        self.start=time.monotonic();self.resume=resume
        self.rows=[];self.phase2=[];self.final=[];self.meta={};self.fits={}
        self.command='python -m experiments.weighting.run --config experiments/weighting/config.yaml --phase all --resume'
        self.source={str(p).replace('\\','/'):digest(p) for p in Path('experiments/weighting').glob('*.py')}
        self.run_identity=dict(config=config,source=self.source,backbone_sha256=digest(config['backbone']))
        marker=self.root/'run_identity.json'
        if marker.exists() and json.loads(marker.read_text())!=self.run_identity:
            raise ValueError('Configuration/source changed: use a new output_dir; existing checkpoints remain intact')
        if marker.exists() and not resume:
            raise ValueError('Output exists; use --resume or a new output_dir')
        write_json(marker,self.run_identity)
        self.handoff('initializing')

    def handoff(self,stage,**extra):
        state=dict(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),rerun=self.command,
                   completed_fits=sum(1 for _ in (self.root/'fits').glob('*/result.json')),
                   identity_sha256=digest(self.root/'run_identity.json'),**extra)
        write_json(self.root/'progress.json',state)
        text=('# Checkpoint handoff\n\n'+json.dumps(state,indent=2,ensure_ascii=False)+'\n\n'
              'Run from the repository root with the same configuration and source.\n'
              'Completed candidate models/results are immutable and reused. Neural fits retain each epoch; '
              'heads retain each completed member and RNG state; features retain each batch. '
              'An interrupted current head is retried, at most 20 epochs. '
              'Test evaluation is prohibited before selection_lock.json exists.\n\n'
              'On Windows use `.venv/Scripts/python.exe` in place of `python`.\n')
        (self.root/'HANDOFF.md').write_text(text,encoding='utf-8')
        if time.monotonic()-self.start>self.c.get('max_wall_seconds',14400):
            raise TimeoutError('Wall-time budget reached. Checkpoints saved; resume with HANDOFF.md')

    def get_meta(self,seed):
        if seed not in self.meta:
            self.handoff('head_crossfit',seed=seed)
            self.meta[seed]=prepare_meta(self.val,self.c,seed)
        return self.meta[seed]

    def fit(self,method,cfg,seed,members=None):
        train,select,_=self.get_meta(seed);train,select=sub(train,members),sub(select,members)
        key=identity(dict(method=method,config=cfg,seed=seed,members=members))[:20]
        root=self.root/'fits'/key;root.mkdir(parents=True,exist_ok=True)
        marker=root/'result.json';model_path=root/'model.joblib'
        self.handoff('weighting_fit',method=method,seed=seed,fit_key=key)
        if marker.exists() and model_path.exists():
            row=json.loads(marker.read_text());model=joblib.load(model_path)
        else:
            model=Predictor(method,cfg).fit(train,select,seed,epochs=self.c['epochs'],patience=self.c['patience'],
                                          progress_path=root/'training_latest.pt')
            if cfg.get('static_average'):
                model.state['static_average']=model.weights(train).mean(0)
            pp=model.predict(select)
            start=time.perf_counter()
            for _ in range(5):model.predict(select)
            latency=(time.perf_counter()-start)/5/len(pp)*1000
            met=metrics(pp,select['y'],threshold=model.threshold)
            row=dict(rank=IDS.index(method)+1,method=method,config=cfg,seed=seed,members=members or train['probs'].shape[1],
                     split='meta_select',**met,train_seconds=model.train_seconds,
                     latency_ms_per_sample=latency,params=model.params,fit_key=key,status='completed',
                     wga=None,wga_reason='Group labels isolated from selection')
            temp=model_path.with_suffix('.tmp');joblib.dump(model,temp);temp.replace(model_path)
            np.savez_compressed(root/'selection_predictions.npz',probs=pp,sample_id=select['sample_id'],y=select['y'])
            write_json(marker,row)
        self.fits[key]=row
        return model,row

    def summarize(self,rows):
        grouped={}
        for r in rows:
            if r.get('status')!='completed':continue
            key=(r['method'],json.dumps(r['config'],sort_keys=True))
            grouped.setdefault(key,[]).append(r)
        result=[]
        for (method,cfg),values in grouped.items():
            row=dict(rank=IDS.index(method)+1,method=method,config=json.loads(cfg),seeds=[r['seed'] for r in values],status='completed')
            for k in ('accuracy','balanced_accuracy','nll','brier','ece','train_seconds','latency_ms_per_sample','params'):
                v=[r[k] for r in values];row[k]=float(np.mean(v));row[k+'_sd']=float(np.std(v,ddof=1)) if len(v)>1 else None
            row['wga']=None
            result.append(row)
        result.sort(key=lambda r:(-r['accuracy'],-r['balanced_accuracy'],r['nll'],r['rank']))
        base=next((r['accuracy'] for r in result if r['method']=='U0'),None)
        for i,r in enumerate(result):r['selection_position']=i+1;r['delta_pp_vs_U0']=(r['accuracy']-base)*100 if base is not None else None
        return result

    def screen(self):
        allowed=self.c.get('method_whitelist') or IDS
        # Hypothesis-led cheap-first priority, then remaining methods; all logged.
        first=['U0','S1','S2','H1','H2','H3','S3','H4','R1','R2','R3','R4','S5','H6','R7','D19','S7','R8','D1','D2']
        order=first+[m for m in IDS if m not in first]
        for method in order:
            if method=='X0':
                for seed in self.c['screen_seeds']:
                    _,select,_=self.get_meta(seed)
                    self.rows.append(dict(rank=42,method='X0',seed=seed,status='diagnostic_only',
                                          oracle_coverage=oracle_coverage(select['probs'],select['y']),reason='Excluded from selection and inference'))
            elif method not in allowed:
                self.rows.append(dict(rank=IDS.index(method)+1,method=method,status='skipped',reason='Method whitelist'))
            else:
                for cfg in candidates(method):
                    for seed in self.c['screen_seeds']:
                        _,row=self.fit(method,cfg,seed);self.rows.append(row)
                        write_csv(self.root/'phase1_all.csv',self.rows)
            write_csv(self.root/'phase1_all.csv',self.rows)
            write_csv(self.root/'phase1_ranked.csv',self.summarize(self.rows))
        ranked=self.summarize(self.rows)
        # Top five method families plus U0, keeping each family's best config.
        families=[];seen=set()
        for r in ranked:
            if r['method'] not in seen:
                families.append(r);seen.add(r['method'])
            if len(families)==5:break
        if 'U0' not in seen:families.append(next(r for r in ranked if r['method']=='U0'))
        finalists=[]
        for r in families:
            for seed in self.c['seeds']:
                _,row=self.fit(r['method'],r['config'],seed);finalists.append(row)
        write_csv(self.root/'phase1_finalists.csv',finalists)
        winner=self.summarize(finalists)[0]
        lock=dict(winner=winner,finalists=[dict(method=r['method'],config=r['config']) for r in families],
                  rule='mean held-out accuracy; then balanced accuracy; then lowest NLL; stable rank tie',
                  test_inspected=False,identity_sha256=digest(self.root/'run_identity.json'))
        write_json(self.root/'selection_lock.json',lock)
        self.handoff('phase1_locked',winner=winner['method'])
        return lock

    def phase_two(self,lock):
        specs=ablations(lock['winner']['method'],lock['winner']['config'])
        for spec in specs:
            if spec['status']=='N/A':
                self.phase2.append(spec);continue
            for seed in self.c['seeds']:
                _,row=self.fit(spec['method'],spec['config'],seed,spec['members'])
                self.phase2.append(row|dict(ablation=spec['ablation'],variant=spec['variant']))
                write_csv(self.root/'phase2_ablation.csv',self.phase2)
        write_csv(self.root/'phase2_ablation.csv',self.phase2)
        write_json(self.root/'ablation_lock.json',dict(specs=specs,selection_lock_sha256=digest(self.root/'selection_lock.json')))
        # Only after both locks: attach group metrics to existing held-out outputs.
        # This cannot affect candidate ordering or winner choice.
        if not self.rows and (self.root/'phase1_all.csv').exists():
            with (self.root/'phase1_all.csv').open(encoding='utf-8') as stream:
                for old in csv.DictReader(stream):
                    if old.get('status')=='completed':
                        self.rows.append(json.loads((self.root/'fits'/old['fit_key']/'result.json').read_text()))
                    else:
                        self.rows.append(old)
        group_by_id=dict(zip(self.val['sample_id'],self.val['group']))
        for row in self.rows:
            if row.get('status')!='completed':continue
            with np.load(self.root/'fits'/row['fit_key']/'selection_predictions.npz',allow_pickle=False) as f:
                groups=np.array([group_by_id[s] for s in f['sample_id']])
                model=joblib.load(self.root/'fits'/row['fit_key']/'model.joblib')
                met=metrics(f['probs'],f['y'],groups,threshold=model.threshold)
            row['wga']=met['wga'];row['per_group']=met['per_group'];row['wga_reason']='Post-hoc, after selection and ablation locks'
        write_csv(self.root/'phase1_all.csv',self.rows)
        ranked=self.summarize(self.rows)
        for row in ranked:
            values=[r['wga'] for r in self.rows if r.get('status')=='completed' and r['method']==row['method'] and r['config']==row['config']]
            row['wga']=float(np.mean(values));row['wga_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        write_csv(self.root/'phase1_ranked.csv',ranked)
        self.handoff('phase2_locked',winner=lock['winner']['method'])
        return specs

    def final_test(self,lock,specs):
        # This is the sole entry point that extracts/reads test features and labels.
        if not (self.root/'selection_lock.json').exists() or not (self.root/'ablation_lock.json').exists():
            raise ValueError('Test evaluation before frozen selection is forbidden')
        self.handoff('final_test_feature_extraction')
        test=extract(self.c,'test')
        assert_disjoint(self.val['sample_id'],test['sample_id'])
        original=torch.load(self.c['original_heads'],map_location='cpu',weights_only=True)
        # Original group-aware full-validation checkpoint is a separate reference.
        original_head=dict(prototypes=original['prototypes'],scales=original['distance_scales'])
        original_cache=self.root/'original_test_predictions.npz'
        original_prov=dict(checkpoint_sha256=digest(self.c['original_heads']),features_sha256=digest(self.root/'features/test.npz'))
        if original_cache.exists():orig=load_predictions(original_cache,original_prov)['probs']
        else:
            orig=predict_heads(original_head,test['embedding'])['probs']
            save_predictions(original_cache,probs=orig,sample_id=test['sample_id'],y=test['y'],split='test',provenance=original_prov)
        original_met=metrics(orig.mean(1),test['y'],test['group'])
        self.final=[dict(method='original_DPE_group_aware_fullval',seed=0,split='test',status='reference',**original_met)]
        requested=[dict(ablation='phase1',variant=r['method'],**r,members=None,status='pending') for r in lock['finalists']]
        requested+= [s for s in specs if s['status']!='N/A']
        all_predictions={}
        for seed in self.c['seeds']:
            _,_,proto=self.get_meta(seed)
            path=self.root/f'seed{seed}'/'heads_final.pt'
            self.handoff('final_head_refit',seed=seed)
            heads=fit_heads(self.val['embedding'][proto],self.val['y'][proto],self.val['sample_id'][proto],self.c,seed*10+9,path)
            cache=self.root/f'seed{seed}'/'predictions_test.npz'
            provenance=dict(heads_sha256=digest(path),features_sha256=digest(self.root/'features/test.npz'))
            if cache.exists():data=load_predictions(cache,provenance)
            else:
                data=predict_heads(heads,test['embedding'])
                save_predictions(cache,**data,sample_id=test['sample_id'],y=test['y'],split='test',provenance=provenance)
            data={k:v for k,v in data.items() if k in ('probs','logits','distances','embedding')}
            baseline=data['probs'].mean(1)
            base_met=metrics(baseline,test['y'],test['group'])
            for spec in requested:
                model,row=self.fit(spec['method'],spec['config'],seed,spec['members'])
                key=row['fit_key'];model_input=sub(data,spec['members'])
                marker=self.root/'test_results'/f'{key}.json'
                pp_path=marker.with_suffix('.npz')
                if marker.exists():
                    final=json.loads(marker.read_text());pp=np.load(pp_path,allow_pickle=False)['probs']
                else:
                    pp=model.predict(model_input)
                    met=metrics(pp,test['y'],test['group'],threshold=model.threshold)
                    w=model.weights(model_input)
                    if w is not None and self.c.get('bootstrap_repeats',0):
                        ci=paired_ci(pp,baseline,test['y'],self.c['bootstrap_repeats'],seed,model.threshold)
                    else:ci=paired_ci(pp,baseline,test['y'],self.c.get('bootstrap_repeats',2000),seed,model.threshold)
                    pred=(pp[:,1]>model.threshold).astype(int)
                    base_correct=baseline.argmax(1)==test['y'];correct=pred==test['y']
                    final=dict(method=spec['method'],config=spec['config'],seed=seed,fit_key=key,split='test',status='completed',
                               members=spec['members'] or 15,**met,
                               delta_pp_vs_U0=(met['accuracy']-base_met['accuracy'])*100,
                               wga_delta_pp_vs_U0=(met['wga']-base_met['wga'])*100,
                               delta_pp_vs_original_DPE=(met['accuracy']-original_met['accuracy'])*100,
                               paired_bootstrap_ci_pp=ci,corrected=int((correct&~base_correct).sum()),broken=int((~correct&base_correct).sum()),
                               paired_bootstrap_ci_pp_vs_original=paired_ci(pp,orig.mean(1),test['y'],self.c['bootstrap_repeats'],seed,model.threshold),
                               train_seconds=row['train_seconds'],latency_ms_per_sample=row['latency_ms_per_sample'],params=row['params'],
                               diagnostics=weight_diagnostics(w,model_input['probs'],test['y'],test['group']))
                    marker.parent.mkdir(exist_ok=True)
                    np.savez_compressed(pp_path,probs=pp,y=test['y'],sample_id=test['sample_id'],group=test['group'],
                                        weights=w if w is not None else np.empty((0,)),pred=pred)
                    write_json(marker,final)
                self.final.append(final|dict(ablation=spec['ablation'],variant=spec['variant']))
                all_predictions[(seed,key)]=pp
                write_csv(self.root/'final_test.csv',self.final)
            winner_model,winner_row=self.fit(lock['winner']['method'],lock['winner']['config'],seed)
            pp=all_predictions[(seed,winner_row['fit_key'])]
            self.export(seed,winner_model,winner_row,pp,test,path)
        # Export native gate checkpoint of seed0; full frozen model also retained in fits/.
        model,row=self.fit(lock['winner']['method'],lock['winner']['config'],self.c['seeds'][0])
        native=dict(method=model.method,config=model.cfg,threshold=model.threshold,feature_state=model.state,
                    model_state=model.model.state_dict() if isinstance(model.model,torch.nn.Module) else {},fit_key=row['fit_key'])
        # NumPy arrays are converted for safe weights_only loading.
        def tensors(v):
            if isinstance(v,np.ndarray):return torch.from_numpy(v.copy())
            if isinstance(v,dict):return {k:tensors(x) for k,x in v.items()}
            if isinstance(v,list):return [tensors(x) for x in v]
            if hasattr(v,'predict_proba'):return {'external_estimator':'Retained in the corresponding model.joblib'}
            return v
        torch.save(tensors(native),self.root/'gate_best.pt')

    def export(self,seed,model,row,pp,test,head_path):
        from src.evaluation.prediction_export import write_predictions
        from src.evaluation.prediction_protocol import read_metadata
        experiment_id=f'exp-dpe-weighting-{row["method"].lower()}-seed{seed}-{row["fit_key"][:8]}'
        root=Path('experiments')/experiment_id
        rows=read_metadata(Path(self.c['dataset_dir'])/'metadata.csv',split='test')
        config_path=root/'resolved_config.json';write_json(config_path,self.c|dict(winner=row))
        artifacts=[dict(id='backbone',role='frozen_backbone',location=self.c['backbone'],sha256=digest(self.c['backbone']),format='torch_state_dict',loader='src.models.resnet.load_resnet_checkpoint'),
                   dict(id='heads',role='final_heads',location=head_path.as_posix(),sha256=digest(head_path),format='torch_dict',loader='experiments.weighting.data.predict_heads'),
                   dict(id='weighting',role='fitted_weighting',location=(self.root/'fits'/row['fit_key']/'model.joblib').as_posix(),sha256=digest(self.root/'fits'/row['fit_key']/'model.joblib'),format='joblib_trusted_local',loader='joblib.load')]
        model_record=dict(model_contract_version=1,experiment_id=experiment_id,method='no-group crossfit DPE weighting adaptation',seed=seed,
                          availability='complete',implementation=dict(source=self.source),artifacts=artifacts,
                          input=dict(dataset=self.c['dataset_dir'],preprocessing='square256;center224;ImageNet;row-zscore-ddof0;L2'),
                          inference=dict(classes=[0,1],aggregation=row['method'],config=row['config'],threshold=model.threshold),
                          selection=dict(lock=(self.root/'selection_lock.json').as_posix(),rule='meta-select accuracy, BA, NLL'),
                          verification=dict(status='not_checked',scope='cached feature/head inference; full image replay not independently repeated'),
                          limitations=['Backbone training history not embedded in supplied bare checkpoint; training-only origin assumed from handoff.',
                                       'OOF-to-final-head distribution shift; single shared backbone; exploratory test reused historically.'])
        write_json(root/'model.json',model_record)
        if not (root/'predictions/test/predictions.csv').exists():
            write_predictions(root/'predictions/test',rows,pp,run_label=experiment_id,seed=seed,
                              provenance=dict(model_description=(root/'model.json').as_posix(),selection_lock_sha256=digest(self.root/'selection_lock.json')),
                              ids=test['sample_id'],predicted_labels=None)
        write_json(root/'run.json',dict(record_contract_version=1,experiment_id=experiment_id,method=model_record['method'],seed=seed,
                   purpose='Accuracy-prioritized weighting versus matched uniform and original DPE',
                   execution=dict(status='completed',invocation=self.command,execution_date=datetime.now(timezone.utc).isoformat()),
                   source=dict(commit=self.commit,source_hashes=self.source),configuration=dict(path=config_path.as_posix(),sha256=digest(config_path)),
                   dataset=dict(root=self.c['dataset_dir'],protocol=self.c['protocol'],folds=self.c['folds'],groups='evaluation only'),
                   randomness=dict(head_and_gate_seed=seed,shared_backbone=True),environment=self.environment,
                   selection=model_record['selection'],artifacts=dict(model=(root/'model.json').as_posix(),predictions=(root/'predictions/test').as_posix()),
                   results=dict(table=(self.root/'final_test.csv').as_posix(),units='fractions; deltas pp'),
                   assessment=dict(reproduction_status='exploratory adaptation',limitations=model_record['limitations']),relationships=dict(study=self.c['output_dir'])))

    def report(self,lock):
        def aggregate(rows,key):
            v=[r[key] for r in rows];return f'{100*np.mean(v):.3f} ± {100*np.std(v,ddof=1):.3f}' if len(v)>1 else f'{100*v[0]:.3f}'
        winner=lock['winner'];selected=[r for r in self.final if r.get('ablation')=='phase1' and r['method']==winner['method']]
        baseline=[r for r in self.final if r.get('ablation')=='phase1' and r['method']=='U0']
        lines=['# DPE weighting research report','',
               'Exploratory empirical study. All percentages below are mean ± sample SD across three refitted head/gate seeds sharing one frozen backbone.', '',
               '## Protocol and leakage audit','',
               f'Original validation: {len(self.val["y"])} samples. {self.c["folds"]}-fold class-stratified cross-fitting; '
               '15 independently resampled class-balanced prototype heads fitted on other folds only. '
               'Each head uses the fixed final epoch (20), with no group-based checkpoint selection. '
               'OOF predictions are split 80/20 by class into gate train/select. All scaler/calibrator/reliability fitting uses gate train only. '
               'Early stopping and configuration selection use select accuracy, then balanced accuracy, then NLL. '
               'Group annotations are absent from every gate fit and used only after selection. '
               'Final heads are refitted on the whole validation pool after both locks. Test is evaluated in one frozen reporting stage; no test-driven changes.', '',
               'The old checkpoint records all 1,199 validation IDs as its head-training pool. It cannot supply held-out validation gate features. '
               'It remains unchanged and is scored only as a separate group-aware full-validation reference. '
               'The pure aggregation effect uses matched uniform with the same new heads. Differences from original DPE also include allocation/training protocol.', '',
               'Baseline parity: U0 is exactly mean(member probabilities). Actual upstream predict_proba(avg=False) shape is [members,samples,classes], '
               'as shown by core.py:_predict; new caches explicitly transpose to [samples,members,classes]. '
               'Training entropic scale 30 is excluded at inference. Distances are real normalized prototype distances.', '',
               'Backbone limitation: supplied bare state_dict has no embedded training IDs/epoch record. Its training-only origin is assumed from the handoff, '
               'not independently proved. No CNN retraining or validation-based backbone selection was performed here. '
               'The old checkpoint uses group annotations, whereas the new heads/gates do not. '
               'The same test set has a history of earlier exploration in this repository; this is not a fresh confirmatory test.', '',
               '## Phase 1','',f'Locked winner: **{winner["method"]}**, config `{json.dumps(winner["config"],sort_keys=True)}`. '
               f'Mean select accuracy {winner["accuracy"]*100:.3f}%. '
               'All 42 ranks are represented in phase1_all.csv, including the oracle diagnostic excluded from selection. '
               'All declared configurations receive two screening seeds; top five families plus U0 receive three seeds. '
               'Large candidate search on a small select pool produces optimistic selection bias; no adaptive extension of grids was used.', '',
               'Full tables: phase1_all.csv, phase1_ranked.csv (two-seed screen), phase1_finalists.csv (three-seed confirmation).', '',
               '## Final test comparison','', '| Predictor | Accuracy % | Balanced accuracy % | WGA % |','|---|---:|---:|---:|',
               f'| Matched U0 | {aggregate(baseline,"accuracy")} | {aggregate(baseline,"balanced_accuracy")} | {aggregate(baseline,"wga")} |',
               f'| {winner["method"]} | {aggregate(selected,"accuracy")} | {aggregate(selected,"balanced_accuracy")} | {aggregate(selected,"wga")} |', '',
               f'Mean accuracy delta vs matched U0: {np.mean([r["delta_pp_vs_U0"] for r in selected]):+.3f} pp. '
               f'Mean WGA delta: {np.mean([r["wga_delta_pp_vs_U0"] for r in selected]):+.3f} pp. '
               f'Mean accuracy delta vs original group-aware DPE: {np.mean([r["delta_pp_vs_original_DPE"] for r in selected]):+.3f} pp.', '',
               'Per-seed 95% paired bootstrap CIs condition on fitted models and test examples; they do not establish independence across backbones. '
               'The 3 seeds rerun splits, heads and gates, not the backbone. Small seed count limits inference.', '',
               '## Phase 2 and failure analysis','',
               'phase2_ablation.csv records every requested A0–A20 category, including explicit N/A reasons. '
               'final_test.csv contains every frozen finalist and feasible ablation with per-class/per-group accuracy/counts, NLL, Brier, ECE, '
               'accuracy and WGA deltas, paired intervals, corrected/broken counts, mean weights, gate entropy, effective member count, '
               'weight mass on correct/incorrect experts, and post-hoc group weight means. '
               'Member-count ablations truncate the same ordered heads and refit weighting only. '
               'Feature removal can leave redundant information in full probability vectors; this limits causal interpretation.', '',
               'Latency measures cached-output weighting only, excluding CNN and head inference. Param counts cover trainable weighting coefficients; '
               'tree/neighbor model sizes are not represented as neural parameter counts. '
               'Head final refitting induces a possible OOF/final-head distribution gap. CVaR uses class tails, never groups.', '',
               '## Checkpoints and rerun','',f'`{self.command}`', '',
               'See HANDOFF.md and progress.json. Each feature batch, completed head (with RNG), neural epoch (optimizer/RNG), '
               'candidate and evaluation is saved. Source/config/input hashes reject incompatible resume. '
               'An interruption can replay the current unfinished head; completed heads and candidate outputs are reused. '
               'Native winner: gate_best.pt plus fits/<key>/model.joblib and seed*/heads_final.pt. '
               'Joblib models are trusted local artifacts and must not be loaded from untrusted sources.']
        if any(r['wga_delta_pp_vs_U0']<0 for r in selected):
            lines+=['','**Robustness trade-off:** WGA decreases for at least one seed; an accuracy gain must not be described as a universal robustness gain.']
        if np.mean([r['delta_pp_vs_U0'] for r in selected])<=0:
            lines+=['','**No held-out test accuracy improvement over matched uniform is established by this run.**']
        lines+=['','## All ranked families: best screening configuration','',
                '| Rank | Family | Select accuracy % | Delta vs U0 pp | BA % | Post-hoc WGA % |',
                '|---:|---|---:|---:|---:|---:|']
        ranked=self.summarize(self.rows)
        for method in IDS:
            values=[r for r in ranked if r['method']==method]
            if not values:
                lines.append(f'| {IDS.index(method)+1} | {method} | Diagnostic only / skipped | — | — | — |');continue
            r=values[0]
            group_values=[a['wga'] for a in self.rows if a.get('status')=='completed' and a['method']==method and a['config']==r['config']]
            lines.append(f'| {r["rank"]} | {method} | {r["accuracy"]*100:.3f} ± {r["accuracy_sd"]*100:.3f} | '
                         f'{r["delta_pp_vs_U0"]:+.3f} | {r["balanced_accuracy"]*100:.3f} | {np.mean(group_values)*100:.3f} |')
        lines+=['','## Ablation test summary','',
                '| Ablation | Variant | Accuracy % | Delta vs U0 pp | WGA % |','|---|---|---:|---:|---:|']
        grouped={}
        for r in self.final:
            if r.get('ablation','phase1')=='phase1':continue
            grouped.setdefault((r['ablation'],r['variant']),[]).append(r)
        for (id_,variant),rs in grouped.items():
            lines.append(f'| {id_} | {variant} | {aggregate(rs,"accuracy")} | {np.mean([r["delta_pp_vs_U0"] for r in rs]):+.3f} | {aggregate(rs,"wga")} |')
        for r in self.phase2:
            if r.get('status')=='N/A':lines.append(f'| {r["ablation"]} | {r["variant"]}: {r["reason"]} | N/A | — | — |')
        lines+=['','## Winner per-group accuracy and failure changes','',
                '| Group (2*y+background) | Count | U0 accuracy % | Winner accuracy % |','|---|---:|---:|---:|']
        for g in selected[0]['per_group']:
            lines.append(f'| {g} | {selected[0]["per_group"][g]["count"]} | '
                         f'{np.mean([r["per_group"][g]["accuracy"] for r in baseline])*100:.3f} | '
                         f'{np.mean([r["per_group"][g]["accuracy"] for r in selected])*100:.3f} |')
        lines+=['',f'Corrected/broken per seed: {[(r["seed"],r["corrected"],r["broken"]) for r in selected]}.',
                f'Paired 95% accuracy-delta CIs (pp): {[(r["seed"],r["paired_bootstrap_ci_pp"]) for r in selected]}.']
        (self.root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        summary=(f'# Kết quả nghiên cứu DPE weighting\n\nWinner chọn trên validation: {winner["method"]}.\n\n'
                 f'Test accuracy: {aggregate(selected,"accuracy")}% so với U0 {aggregate(baseline,"accuracy")}%.\n'
                 f'Chênh lệch accuracy trung bình: {np.mean([r["delta_pp_vs_U0"] for r in selected]):+.3f} điểm %.\n'
                 f'Chênh lệch WGA trung bình: {np.mean([r["wga_delta_pp_vs_U0"] for r in selected]):+.3f} điểm %.\n\n'
                 'Đã tách dữ liệu huấn luyện heads và gate bằng cross-fitting. Không dùng group để train/chọn mô hình mới. '
                 'Ba seed dùng chung backbone; lịch sử train backbone không được lưu trong checkpoint. '
                 'Đây là kết quả thăm dò, không phải kiểm chứng trên test hoàn toàn mới. Xem REPORT.md và toàn bộ CSV.\n')
        (self.root/'RESULTS_FOR_USER.md').write_text(summary,encoding='utf-8')
        self.plot(selected,baseline)

    def plot(self,selected,baseline):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        root=self.root/'plots';root.mkdir(exist_ok=True)
        fig,ax=plt.subplots(figsize=(7,4))
        x=np.arange(len(selected))
        ax.bar(x-.18,[r['delta_pp_vs_U0'] for r in selected],.36,label='Accuracy delta')
        ax.bar(x+.18,[r['wga_delta_pp_vs_U0'] for r in selected],.36,label='WGA delta')
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,[f'Seed {r["seed"]}' for r in selected])
        ax.set_ylabel('Percentage points vs matched U0');ax.legend();fig.tight_layout()
        fig.savefig(root/'winner_tradeoffs.png',dpi=160);plt.close(fig)

    def run(self,phase):
        seed_all(0)
        self.commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        self.environment=dict(python=sys.version,platform=platform.platform(),device=self.c['device'],
                              packages={p:importlib.metadata.version(p) for p in ('torch','torchvision','numpy','scikit-learn','scipy')})
        write_json(self.root/'manifest.json',dict(config=self.c,commit=self.commit,source=self.source,environment=self.environment,
                   command=self.command,backbone_provenance='bare checkpoint; training-only origin assumed, not independently proved'))
        for source,sha in self.source.items():
            target=self.root/'source_snapshot'/Path(source).name
            target.parent.mkdir(exist_ok=True)
            if not target.exists():shutil.copyfile(source,target)
            elif digest(target)!=sha:raise ValueError('Saved source snapshot changed')
        (self.root/'run.log').open('a',encoding='utf-8').write(datetime.now(timezone.utc).isoformat()+' '+phase+'\n')
        self.val=extract(self.c,'val')
        self.handoff('validation_features_ready')
        lock_path=self.root/'selection_lock.json'
        lock=json.loads(lock_path.read_text()) if lock_path.exists() else self.screen()
        if phase=='1':return
        specs=self.phase_two(lock)
        self.final_test(lock,specs)
        self.report(lock)
        cache_index={p.relative_to(self.root).as_posix():digest(p) for p in self.root.rglob('*.npz')}
        write_json(self.root/'cache_index.json',cache_index)
        self.handoff('complete',winner=lock['winner']['method'])
        print(f'Complete: {lock["winner"]["method"]}; outputs: {self.root}',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='experiments/weighting/config.yaml')
    parser.add_argument('--phase',choices=['1','2','all'],default='all');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();c=yaml.safe_load(Path(args.config).read_text())
    study=Study(c,args.resume)
    try:study.run(args.phase)
    except BaseException as error:
        write_json(study.root/'last_error.json',dict(type=type(error).__name__,message=str(error),rerun=study.command))
        raise


if __name__=='__main__':main()
