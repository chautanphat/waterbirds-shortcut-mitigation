"""CPU-only TEST ORACLE threshold diagnostic on cached Waterbirds scores.

Uses test labels deliberately; never loads checkpoints, images, torch, or GPU.
No validation-selected threshold or previous experiment file is changed.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from src.counterfactual.common import sha256,write_json,write_csv
from src.counterfactual.threshold_probe import metrics,group_codes,predict

MODELS=['POTER','ERM_enhanced','DPE_legacy']
SIGMAS=[0.,.5,1.,2.,4.]
PRIMARY=[0.,.5,1.,2.]
LABEL='TEST ORACLE / IN-SAMPLE UPPER BOUND'
REQUIRED={'source_split','control','cohort','model_name','model_seed','image_id','source_image_id',
    'foreground_id','species_id','true_y','original_group','background_id','bg_type','severity','sigma',
    'p_landbird','p_waterbird','logit_landbird','logit_waterbird','score_type','pred_y','is_correct'}


def score(frame,threshold):
    """Reuse existing four-group metric logic; expose explicit percent columns."""
    r=metrics(frame,float(threshold));g=group_codes(frame)
    if r['wga'] is None:raise ValueError('Four-group WGA requires all four true-class/background groups')
    result=dict(threshold=float(threshold),count=r['n'],n_foregrounds=frame.source_image_id.nunique(),
        aa_pct=100*r['accuracy'],wga_pct=100*r['wga'],
        landbird_accuracy_pct=100*r['landbird_accuracy'],waterbird_accuracy_pct=100*r['waterbird_accuracy'],
        landbird_to_waterbird_count=r['fp'],waterbird_to_landbird_count=r['fn'],
        landbird_to_waterbird_rate_pct=100*r['false_positive_rate'],waterbird_to_landbird_rate_pct=100*r['false_negative_rate'],
        fraction_predicted_waterbird=r['fraction_predicted_waterbird'],
        predicted_waterbird_pct=100*r['fraction_predicted_waterbird'],
        class_gap_water_minus_land_pp=100*(r['waterbird_accuracy']-r['landbird_accuracy']))
    for group in range(4):
        result[f'group{group}_n']=int((g==group).sum())
        result[f'group{group}_accuracy_pct']=100*r[f'group{group}_accuracy']
    return result


def probability_audit(frame):
    p=frame[['p_landbird','p_waterbird']].to_numpy(float)
    z=frame[['logit_landbird','logit_waterbird']].to_numpy(float)
    if not np.isfinite(p).all() or not np.isfinite(z).all():raise ValueError('Missing/nonfinite score')
    if (p<0).any() or (p>1).any():raise ValueError('Probability bounds violated')
    error=float(abs(p.sum(1)-1).max())
    if error>1e-6:raise ValueError('Unnormalized probabilities')
    shifted=z-z.max(1,keepdims=True);soft=np.exp(shifted);soft/=soft.sum(1,keepdims=True)
    logit_error=float(abs(soft-p).max())
    if logit_error>2e-6:raise ValueError('Logits/log-mean-probabilities disagree with saved probabilities')
    recomputed=predict(p[:,1],.5)
    return dict(count=len(frame),max_probability_sum_error=error,max_probability_vs_logit_error=logit_error,
        pred05_vs_saved_disagreements=int((recomputed!=frame.pred_y.to_numpy(int)).sum()),
        saved_correctness_disagreements=int(((recomputed==frame.true_y.to_numpy(int))!=frame.is_correct.astype(str).str.lower().eq('true').to_numpy()).sum()),
        prediction_rule='Waterbird iff p_waterbird>=threshold; recomputed, never trusts cached hard label')


def validate_natural(raw,metadata):
    if not REQUIRED.issubset(raw.columns):raise ValueError('Missing columns: '+str(sorted(REQUIRED-set(raw.columns))))
    if 'POTER' not in set(raw.model_name):raise ValueError('POTER cache required')
    natural=raw[(raw.source_split=='test')&(raw.control=='natural')].copy()
    if natural.empty:raise ValueError('No test natural composites')
    if natural[list(REQUIRED)].isna().any().any():raise ValueError('Natural rows contain missing required labels/scores/IDs')
    if not set(natural.model_name).issubset(MODELS):raise ValueError('Unrecognized model')
    if set(natural.true_y)!={0,1} or not set(natural.bg_type).issubset({'land','water'}):raise ValueError('Invalid binary label/background coding')
    canonical=metadata.set_index('img_id')
    joined=canonical.loc[natural.source_image_id.astype(int)]
    if not np.array_equal(joined.y.to_numpy(int),natural.true_y.to_numpy(int)):raise ValueError('Labels differ from canonical metadata')
    if not (joined.split==2).all():raise ValueError('Non-test source IDs')
    if not np.array_equal((2*joined.y+joined.place).to_numpy(int),natural.original_group.to_numpy(int)):
        raise ValueError('Original group/source join mismatch')
    if not np.array_equal(joined.img_filename.str.split('/').str[0].to_numpy(),natural.species_id.to_numpy()):
        raise ValueError('Species/source join mismatch')
    if natural.duplicated(['model_name','model_seed','source_image_id','sigma','background_id']).any():raise ValueError('Duplicate natural prediction key')
    if natural.duplicated(['model_name','image_id']).any():raise ValueError('Duplicate model/image IDs')
    reference_ids=None;coverage=[];audits=[]
    for model,part in natural.groupby('model_name',sort=False):
        if part.model_seed.nunique()!=1:raise ValueError('Mixed model seeds; analyze each independent seed explicitly')
        if set(part.sigma)!=set(SIGMAS):raise ValueError('Expected exactly sigma0/.5/1/2/4')
        ids=set(part.source_image_id)
        if reference_ids is None:reference_ids=ids
        if ids!=reference_ids:raise ValueError('Models have different foreground cohort')
        for sigma,data in part.groupby('sigma',sort=True):
            if len(data)!=400 or data.source_image_id.nunique()!=100:raise ValueError('Expected400 composites/100 foregrounds per model/sigma')
            if set(data.source_image_id)!=reference_ids:raise ValueError('Severity changes foreground IDs')
            if not (data.groupby('source_image_id').size()==4).all():raise ValueError('Incomplete per-foreground backgrounds')
            if not (data.groupby(['source_image_id','bg_type']).size()==2).all():raise ValueError('Missing two background replicates per type')
            if data.background_id.nunique()!=4:raise ValueError('Expected same four background IDs')
            if set(data.groupby('source_image_id').true_y.first().value_counts())!={50}:raise ValueError('Expected50 foregrounds/class')
            groups=group_codes(data);counts=[int((groups==g).sum()) for g in range(4)]
            if counts!=[100]*4:raise ValueError('Expected100 rows in each class/background group')
            record=dict(model_name=model,model_seed=int(data.model_seed.iloc[0]),sigma=float(sigma),
                count=len(data),n_foregrounds=100,group_denominators=counts,
                backgrounds=sorted(data.background_id.unique()),duplicates=0,missing_scores=0,all_four_groups=True)
            coverage.append(record);audits.append(dict(model_name=model,sigma=float(sigma),**probability_audit(data)))
    if 'normalized_tensor_sha256' in natural:
        if (natural.groupby('image_id').normalized_tensor_sha256.nunique()!=1).any():raise ValueError('Models saw different tensors')
    return natural,dict(raw_rows=len(raw),included_natural_rows=len(natural),
        excluded_by_control=raw[~raw.index.isin(natural.index)].control.value_counts().to_dict(),
        actual_cohort_values=sorted(raw.cohort.unique()),actual_control_values=sorted(raw.control.unique()),
        label_mapping={'0':'Landbird','1':'Waterbird'},group_rule='2*true_y+(bg_type==water), never original_group for composites',
        coverage=coverage,probability_checks=audits)


def winner(records):
    return min(records,key=lambda r:(-round(r['wga_pct'],12),-round(r['aa_pct'],12),round(abs(r['threshold']-.5),14),r['threshold']))


def exact_candidates(probabilities):
    """All decisions attainable by a float64 threshold in[0,1], including ties.

    >= keeps observations exactly on a boundary positive; nextafter probes the
    adjacent open interval. Adding0.5 gives the nearest-to-baseline candidate
    when that value lies inside an optimal interval.
    """
    values=np.unique(np.asarray(probabilities,dtype=float))
    adjacent=np.nextafter(values,np.inf)
    return np.unique(np.r_[0.,.5,1.,values,adjacent[adjacent<=1.]])


def delta_record(selected,baseline):
    return {field+'_delta_vs_fixed05_pp':selected[field]-baseline[field] for field in
        ['wga_pct','aa_pct','landbird_accuracy_pct','waterbird_accuracy_pct']}


def original_data(path,models,metadata):
    status={};groups={}
    if path is None or not path.is_file():
        return {},{m:dict(status='NOT AVAILABLE',reason='Full original-test prediction cache missing') for m in models}
    frame=pd.read_csv(path,float_precision='round_trip')
    canonical=metadata[metadata.split==2].set_index('img_id')
    for model in models:
        data=frame[frame.model_name==model].copy()
        if len(data)!=5794 or data.source_image_id.nunique()!=5794 or set(data.source_image_id)!=set(canonical.index):
            status[model]=dict(status='NOT AVAILABLE',reason='Complete5794 unique canonical test originals not present',rows=len(data));continue
        if set(data.control)!={'original'} or set(data.source_split)!={'test'}:raise ValueError('Original cache includes other controls/splits')
        labels=canonical.loc[data.source_image_id]
        if not np.array_equal(labels.y.to_numpy(int),data.true_y.to_numpy(int)):raise ValueError('Original cache labels disagree')
        if not np.array_equal((2*labels.y+labels.place).to_numpy(int),data.original_group.to_numpy(int)):raise ValueError('Original groups disagree')
        probability_audit(data);groups[model]=data
        status[model]=dict(status='AVAILABLE; full5794 original images',count=5794,retuned_on_original=False)
    return groups,status


def run(input_path,out,original_path,metadata_path,check_exact=True):
    started=time.perf_counter();out.mkdir(parents=True,exist_ok=True)
    if (out/'audit.json').exists():
        saved=json.loads((out/'audit.json').read_text())
        if saved['source_sha256']!=sha256(input_path):raise ValueError('Source changed: use a new output directory')
    if not check_exact and (out/'exact_best_thresholds.csv').exists():
        raise ValueError('Existing exact results preserved; use a new output directory with --skip-exact')
    raw=pd.read_csv(input_path,float_precision='round_trip');metadata=pd.read_csv(metadata_path)
    natural,audit=validate_natural(raw,metadata)
    models=[m for m in MODELS if m in set(natural.model_name)]
    grid=np.arange(5,96,dtype=int)/100
    sweep=[];best=[];exact=[];condition={};best_by={}
    expected={0.:86.,.5:83.,1.:74.,2.:58.,4.:23.}
    for model in models:
        for sigma in SIGMAS:
            data=natural[(natural.model_name==model)&(natural.sigma==sigma)]
            base=dict(model_name=model,model_seed=int(data.model_seed.iloc[0]),sigma=sigma,
                supplementary=sigma==4.,dataset='natural_composites100',source_split='test',
                selection_warning=LABEL,training_seed_count=1)
            records=[dict(base,**score(data,t)) for t in grid]
            baseline=next(r for r in records if r['threshold']==.5)
            if model=='POTER' and not np.isclose(baseline['wga_pct'],expected[sigma],atol=1e-10):
                raise ValueError('Fixed0.50 POTER WGA sanity check failed; do not adjust inputs to fit expected results')
            selected=winner(records);best_by[model,sigma]=selected
            tied=[r for r in records if np.isclose(r['wga_pct'],selected['wga_pct'],atol=1e-12,rtol=0)]
            tied_aa=[r for r in tied if np.isclose(r['aa_pct'],selected['aa_pct'],atol=1e-12,rtol=0)]
            best.append(dict(selected,**delta_record(selected,baseline),grid_max_wga_tie_count=len(tied),
                grid_max_wga_thresholds=json.dumps([r['threshold'] for r in tied]),
                after_aa_tie_count=len(tied_aa),after_aa_thresholds=json.dumps([r['threshold'] for r in tied_aa]),
                optimum_scope='0.01 grid only; not asserted as exact global optimum'))
            if check_exact:
                candidates=exact_candidates(data.p_waterbird)
                optimal=winner([score(data,t) for t in candidates])
                exact.append(dict(base,**optimal,exact_candidate_count=len(candidates),
                    grid_oracle_threshold=selected['threshold'],grid_oracle_wga_pct=selected['wga_pct'],
                    extra_wga_vs_grid_pp=optimal['wga_pct']-selected['wga_pct'],
                    scope='all attainable threshold decisions on fixed finite scores, float64 t in[0,1]; in-sample oracle only'))
            condition[model,sigma]=data;sweep.extend(records)
        print(model+': cached-score grid and exact-boundary analysis complete',flush=True)
    sweep_frame=pd.DataFrame(sweep);best_frame=pd.DataFrame(best)
    originals,original_status=original_data(original_path,models,metadata)
    strategies=[];global_choices={};joint_candidates=[]
    for model in models:
        clean_threshold=best_by[model,0.]['threshold']
        joint=[]
        for t in grid:
            primary=sweep_frame[(sweep_frame.model_name==model)&(sweep_frame.sigma.isin(PRIMARY))&(sweep_frame.threshold==t)]
            clean=primary[primary.sigma==0].iloc[0]
            joint.append(dict(model_name=model,threshold=float(t),mean_primary_wga_pct=float(primary.wga_pct.mean()),
                clean_wga_pct=float(clean.wga_pct),mean_primary_aa_pct=float(primary.aa_pct.mean()),selection_warning=LABEL))
        chosen=min(joint,key=lambda r:(-round(r['mean_primary_wga_pct'],12),-round(r['clean_wga_pct'],12),
            -round(r['mean_primary_aa_pct'],12),round(abs(r['threshold']-.5),14),r['threshold']))
        fixed={'fixed_05':.5,'fixed_065':.65,'oracle_clean_transfer':clean_threshold,'oracle_joint':chosen['threshold']}
        global_choices[model]=dict(thresholds=fixed,joint_objective='unweighted mean WGA over composite test sigma0/.5/1/2',
            joint_tie_rule='clean WGA desc, mean AA over four primary sigmas desc, abs(t-.5) asc, t asc',
            selection_warning=LABEL,validation_selected=False)
        joint_candidates.extend(joint)
        clean_fixed=score(condition[model,0.],.5)['wga_pct']
        for sigma in SIGMAS:
            for strategy,t in {**fixed,'oracle_per_sigma':best_by[model,sigma]['threshold']}.items():
                result=score(condition[model,sigma],t)
                baseline=score(condition[model,sigma],.5)
                clean_drop=clean_fixed-score(condition[model,0.],t)['wga_pct'] if strategy!='oracle_per_sigma' else None
                strategies.append(dict(model_name=model,model_seed=int(condition[model,sigma].model_seed.iloc[0]),
                    dataset='natural_composites100',sigma=sigma,supplementary=sigma==4.,strategy=strategy,
                    **result,**delta_record(result,baseline),clean_WGA_drop_vs_fixed05=clean_drop,
                    clean_drop_gt2pp=bool(clean_drop>2+1e-10) if clean_drop is not None else None,
                    clean_drop_reference='natural composite sigma0; same100 foregrounds',
                    selection_warning=LABEL if strategy.startswith('oracle') else 'fixed reference; already-inspected exploratory test',
                    evaluation_is_untouched=False,validation_selected=False))
        if model in originals:
            data=originals[model];baseline=score(data,.5)
            for strategy,t in fixed.items():
                result=score(data,t);clean_drop=baseline['wga_pct']-result['wga_pct']
                strategies.append(dict(model_name=model,model_seed=int(data.model_seed.iloc[0]),
                    dataset='full_original_test5794',sigma=0.,supplementary=False,strategy=strategy,
                    **result,**delta_record(result,baseline),clean_WGA_drop_vs_fixed05=clean_drop,
                    clean_drop_gt2pp=bool(clean_drop>2+1e-10),clean_drop_reference='full original unmodified test5794',
                    selection_warning='Composite-test-adapted threshold; applied to previously inspected original test without retuning' if strategy.startswith('oracle') else 'fixed reference; previously inspected original test',
                    evaluation_is_untouched=False,validation_selected=False))
    strategy_frame=pd.DataFrame(strategies)
    sweep_frame.to_csv(out/'sweep_metrics.csv',index=False);best_frame.to_csv(out/'best_thresholds.csv',index=False)
    strategy_frame.to_csv(out/'strategy_comparison.csv',index=False)
    pd.DataFrame(joint_candidates).to_csv(out/'oracle_joint_candidates.csv',index=False)
    if check_exact:pd.DataFrame(exact).to_csv(out/'exact_best_thresholds.csv',index=False)
    write_json(out/'oracle_threshold_choices.json',global_choices)
    write_json(out/'original_test_status.json',original_status)
    audit.update(source_path=str(input_path.resolve()),source_sha256=sha256(input_path),metadata_sha256=sha256(metadata_path),
        original_cache_path=str(original_path) if original_path else None,
        original_cache_sha256=sha256(original_path) if original_path and original_path.is_file() else None,
        test_labels_used_for_threshold_selection=True,selection_scope=LABEL,model_loading=False,inference=False,
        training=False,ot_updates=False,images_generated=False,bootstrap_or_significance_claims=False)
    write_json(out/'audit.json',audit)
    write_json(out/'experiment_config.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        input=str(input_path),output=str(out),models=models,sigmas=SIGMAS,primary_sigmas=PRIMARY,
        threshold_grid=[.05,.95,.01],grid_tie_rule='WGA desc, AA desc, abs(t-.5) asc, t asc',
        exact_boundary_check=check_exact,threshold_rule='p_waterbird>=t',oracle_only=True,
        elapsed_seconds=time.perf_counter()-started,python=sys.version,numpy=np.__version__,pandas=pd.__version__,
        source_code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/counterfactual/threshold_probe.py']}))
    plots(out,sweep_frame,best_frame,strategy_frame)
    write_json(out/'status.json',dict(analysis_complete=True,selection_scope=LABEL,training=False,
        model_loading=False,model_inference=False,gpu_used=False,test_labels_used_for_selection=True,
        validation_selected_method=False,unbiased_generalization_gain_claimed=False,
        elapsed_seconds=time.perf_counter()-started))
    report(out,audit,sweep_frame,best_frame,strategy_frame,pd.DataFrame(exact),global_choices,original_status)
    write_csv(out/'files_manifest.csv',[dict(path=p.name,bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ['integrity.json','files_manifest.csv']])
    write_json(out/'integrity.json',dict(sha256={p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='integrity.json'}))
    print(json.dumps(dict(report=str(out/'report.md'),rows_sweep=len(sweep_frame),oracle=True,training=False,
        model_inference=False,elapsed_seconds=time.perf_counter()-started)),flush=True)


def plots(out,sweep,best,strategies):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={0.:'#4b6a9b',.5:'#39806b',1.:'#bc8a2a',2.:'#b64b39',4.:'#775398'}
    fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained')
    for sigma in SIGMAS:
        rows=sweep[(sweep.model_name=='POTER')&(sweep.sigma==sigma)]
        selected=best[(best.model_name=='POTER')&(best.sigma==sigma)].iloc[0]
        ax.plot(rows.threshold,rows.wga_pct,label=f'sigma{sigma:g}'+(' supplementary' if sigma==4 else ''),color=colors[sigma])
        ax.scatter(selected.threshold,selected.wga_pct,color=colors[sigma],marker='*',s=110)
    ax.axvline(.5,color='gray',ls=':',label='fixed0.50');ax.axvline(.65,color='black',ls='--',label='fixed0.65')
    ax.set(title='POTER — TEST ORACLE / IN-SAMPLE UPPER BOUND',xlabel='Waterbird decision threshold',ylabel='Four-group WGA (%)',ylim=(-2,103))
    ax.legend(fontsize=8,ncol=2);fig.savefig(out/'threshold_vs_wga.png',dpi=170,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for strategy,style in [('fixed_05','-o'),('fixed_065','-s'),('oracle_per_sigma','--*'),('oracle_clean_transfer','--^'),('oracle_joint',':x')]:
        rows=strategies[(strategies.model_name=='POTER')&(strategies.dataset=='natural_composites100')&(strategies.strategy==strategy)].sort_values('sigma')
        ax.plot(rows.sigma,rows.wga_pct,style,label=strategy)
    ax.set(title='POTER composite-test threshold adaptation — exploratory oracle only',xlabel='Sigma (4 supplementary)',ylabel='Four-group WGA (%)',ylim=(0,103),xticks=SIGMAS)
    ax.legend(fontsize=8);fig.savefig(out/'wga_vs_sigma.png',dpi=170,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for sigma,ax in zip([1.,2.],axs):
        rows=sweep[(sweep.model_name=='POTER')&(sweep.sigma==sigma)]
        ax.plot(rows.threshold,rows.landbird_accuracy_pct,label='Landbird');ax.plot(rows.threshold,rows.waterbird_accuracy_pct,label='Waterbird')
        ax.plot(rows.threshold,rows.wga_pct,':',label='Four-group WGA',color='gray')
        ax.axvline(.5,color='gray',ls=':');ax.axvline(.65,color='black',ls='--')
        selected=best[(best.model_name=='POTER')&(best.sigma==sigma)].iloc[0]
        ax.axvline(selected.threshold,color='#b64b39',ls='-.',label=f'oracle t={selected.threshold:.2f}')
        ax.set(title=f'POTER sigma{sigma:g}',xlabel='Waterbird threshold',ylabel='Accuracy (%)',ylim=(0,103));ax.legend(fontsize=8)
    fig.suptitle('Class trade-offs — TEST ORACLE / IN-SAMPLE UPPER BOUND; no calibration claim')
    fig.savefig(out/'class_tradeoff.png',dpi=170,bbox_inches='tight');plt.close(fig)


def report(out,audit,sweep,best,strategies,exact,choices,original_status):
    # Filled below from actual executed results, never from expected targets.
    rows=[]
    for sigma in SIGMAS:
        baseline=sweep[(sweep.model_name=='POTER')&(sweep.sigma==sigma)&(sweep.threshold==.5)].iloc[0]
        reference=sweep[(sweep.model_name=='POTER')&(sweep.sigma==sigma)&(sweep.threshold==.65)].iloc[0]
        selected=best[(best.model_name=='POTER')&(best.sigma==sigma)].iloc[0]
        rows.append(f'|{sigma:g}|{baseline.wga_pct:.2f}|{reference.wga_pct:.2f}|{selected.threshold:.2f}|{selected.wga_pct:.2f}|{selected.wga_pct-baseline.wga_pct:+.2f}|{selected.aa_pct-baseline.aa_pct:+.2f}|{selected.landbird_accuracy_pct-baseline.landbird_accuracy_pct:+.2f}|{selected.waterbird_accuracy_pct-baseline.waterbird_accuracy_pct:+.2f}|')
    selection=choices['POTER']['thresholds']
    compare=strategies[(strategies.model_name=='POTER')&(strategies.dataset=='natural_composites100')]
    strategy_rows=[]
    for strategy in ['fixed_05','fixed_065','oracle_per_sigma','oracle_clean_transfer','oracle_joint']:
        part=compare[compare.strategy==strategy].set_index('sigma')
        strategy_rows.append('|'+strategy+'|'+('|'.join(f'{part.loc[s,"threshold"]:.2f}/{part.loc[s,"wga_pct"]:.2f}' for s in SIGMAS))+'|')
    full=strategies[(strategies.model_name=='POTER')&(strategies.dataset=='full_original_test5794')]
    original_rows=[]
    for r in full.itertuples():
        original_rows.append(f'|{r.strategy}|{r.threshold:.2f}|{r.wga_pct:.2f}|{r.aa_pct:.2f}|{r.landbird_accuracy_pct:.2f}|{r.waterbird_accuracy_pct:.2f}|{r.clean_WGA_drop_vs_fixed05:+.2f}|{r.clean_drop_gt2pp}|')
    selected1=best[(best.model_name=='POTER')&(best.sigma==1.)].iloc[0]
    selected2=best[(best.model_name=='POTER')&(best.sigma==2.)].iloc[0]
    summary=(f'**Yes, threshold changes substantially recover POTER composite WGA in-sample:** '
        f'sigma1 recovers {selected1.wga_pct_delta_vs_fixed05_pp:+.0f} pp to{selected1.wga_pct:.0f}% '
        f'(grid t={selected1.threshold:.2f}); sigma2 recovers {selected2.wga_pct_delta_vs_fixed05_pp:+.0f} pp '
        f'to{selected2.wga_pct:.0f}% (t={selected2.threshold:.2f}). '
        'This is a test-label-fitted feasibility result, not evidence of a deployable correction.\n\n')
    joint_rows=compare[compare.strategy=='oracle_joint'].set_index('sigma')
    per_sigma_rows=compare[compare.strategy=='oracle_per_sigma'].set_index('sigma')
    transfer_rows=compare[compare.strategy=='oracle_clean_transfer'].set_index('sigma')
    joint_values='/'.join(f'{joint_rows.loc[s,"wga_pct"]:g}' for s in PRIMARY)
    per_values='/'.join(f'{per_sigma_rows.loc[s,"wga_pct"]:g}' for s in PRIMARY)
    largest_gap=max(per_sigma_rows.loc[s,'wga_pct']-joint_rows.loc[s,'wga_pct'] for s in PRIMARY)
    summary+=(f'A single test-joint threshold{selection["oracle_joint"]:.2f} gives composite WGAs{joint_values}% '
        f'at sigma0/.5/1/2, versus per-severity{per_values}%. '
        f'Per-severity selection adds at most{largest_gap:.2f} pp over that single global oracle on the primary levels. '
        f'The clean-transfer threshold{selection["oracle_clean_transfer"]:.2f} reaches '
        f'{transfer_rows.loc[2.,"wga_pct"]:.2f}% at sigma2. '
        'Thus the transfer table, rather than a claim of one universally suitable threshold, determines consistency.\n\n')
    if len(full):
        b=full[full.strategy=='fixed_05'].iloc[0];shift=full[full.strategy=='fixed_065'].iloc[0]
        c0=compare[(compare.strategy=='fixed_05')&(compare.sigma==0.)].iloc[0]
        c65=compare[(compare.strategy=='fixed_065')&(compare.sigma==0.)].iloc[0]
        b2=compare[(compare.strategy=='fixed_05')&(compare.sigma==2.)].iloc[0]
        s2=compare[(compare.strategy=='fixed_065')&(compare.sigma==2.)].iloc[0]
        summary+=(f'**Original-clean trade-off:** applying0.65 unchanged to full5794 originals changes WGA '
            f'{b.wga_pct:.2f}→{shift.wga_pct:.2f}% ({shift.wga_pct-b.wga_pct:+.2f} pp), '
            f'while AA changes{b.aa_pct:.2f}→{shift.aa_pct:.2f}% and Waterbird accuracy changes '
            f'{b.waterbird_accuracy_pct:.2f}→{shift.waterbird_accuracy_pct:.2f}%. '
            f'Composite-clean WGA changes{c0.wga_pct:.2f}→{c65.wga_pct:.2f}%; those two clean conditions must not be conflated. '
            f'At composite sigma2,0.65 changes Landbird{b2.landbird_accuracy_pct:.2f}→{s2.landbird_accuracy_pct:.2f}% '
            f'and Waterbird{b2.waterbird_accuracy_pct:.2f}→{s2.waterbird_accuracy_pct:.2f}%.\n\n')
    exact_text='Exact-boundary analysis disabled; grid maximum is not claimed to be a global optimum.'
    if len(exact):
        exact_text='\n'.join(f'- {r.model_name}, sigma{r.sigma:g}: grid t={r.grid_oracle_threshold:.2f}, WGA={r.grid_oracle_wga_pct:.2f}%; exact finite-score boundary t={r.threshold:.12g}, WGA={r.wga_pct:.2f}% (extra{r.extra_wga_vs_grid_pp:+.2f} pp).' for r in exact.itertuples())
    text=f'''# Fast test-oracle threshold sweep

**TEST ORACLE / IN-SAMPLE UPPER BOUND. Test labels intentionally used to select thresholds. These are not validation-selected methods, deployable policies, or unbiased generalization gains.** No checkpoint loading, model inference, training, images or OT updates.

{summary}

## POTER primary results

All values are percentages or percentage-point changes. Natural composites only:400 images/sigma,100 source foregrounds, two Land and two Water backgrounds/foreground; four groups each contain100 rows. Sigma4 is supplementary.

|Sigma|WGA@0.50|WGA@0.65|Oracle threshold|Oracle WGA|Gain vs0.50 pp|AA change pp|Landbird change pp|Waterbird change pp|
|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

## Threshold transfer across severities

Each cell below is threshold/WGA(%). `oracle_per_sigma` fits labels at each severity separately. `oracle_clean_transfer` fits composite test sigma0 only. `oracle_joint` fits the unweighted mean of four test WGAs at sigma0/.5/1/2; tie-break clean WGA, mean primary AA, proximity to0.5, lower threshold. Both global oracle strategies are **test-adapted**, not held-out validation calibration. Threshold0.65 is a user-mentioned reference recomputed here, not validation-selected.

|Strategy|Sigma0|Sigma0.5|Sigma1|Sigma2|Sigma4 supplementary|
|---|---|---|---|---|---|
{chr(10).join(strategy_rows)}

POTER clean-transfer threshold={selection['oracle_clean_transfer']:.2f}; joint threshold={selection['oracle_joint']:.2f}. `clean_WGA_drop_vs_fixed05` is positive for a loss of clean WGA; >2pp flags are reporting flags, never selection constraints. Per-sigma strategy has no single clean-transfer threshold, so that column is undefined for it.

## Original unmodified full test — no retuning

These5794 cached original images are a different input condition from clean composites. The100 composite foregrounds overlap this full test; this is **not independent confirmation**. Composite-selected global thresholds are applied unchanged using `original_group`. Status: {original_status['POTER']['status']}.

|Strategy|Threshold|Original WGA %|AA %|Landbird %|Waterbird %|Clean WGA drop vs0.50 pp|Drop >2pp|
|---|---|---|---|---|---|---|---|
{chr(10).join(original_rows) if original_rows else '|NOT AVAILABLE||||||||'}

## Exact-boundary check and other models

Primary strategy comparison uses the declared0.01 grid. Separately, exact finite-score decisions are enumerated at distinct probabilities, float64 `nextafter` points in adjacent open intervals, and endpoints0/1 (plus0.5 for the proximity tie-break). This covers all attainable >=threshold classifications for t in[0,1], including ties. It does not establish an optimum on a new score distribution or meaningful deployment precision.

{exact_text}

ERM_enhanced/DPE_legacy are included because their cached probabilities were already present. Their full metrics, ties, global thresholds, clean trade-offs and full-original results are in the same CSVs. DPE's probabilities are narrowly distributed, so a0.01 grid can miss useful boundaries; never conflate the grid maximum with the exact finite-score maximum.

## Audit and scope

Source `{audit['source_path']}`, SHA256 `{audit['source_sha256']}`. The requested browser name predictions(5).csv was not supplied; canonical audited9000-row mild-blur CSV is used. Actual cohort values:{audit['actual_cohort_values']}; control values:{audit['actual_control_values']}. Included natural rows={audit['included_natural_rows']}; excluded controls={audit['excluded_by_control']}. No excluded controls contribute to composite WGA.

Canonical labels0=Landbird/1=Waterbird, source IDs/species/splits and original-group joins checked. Composite WGA uses **2*true_y+natural background type**, not the source photo's original background. Every model×sigma has400 rows,100 foregrounds, all four groups with100 rows each, four fixed background IDs and two replicas/type. Duplicate/missing/invalid rows cause explicit errors. Probabilities are finite/bounded/normalized and checked against logits (DPE uses log mean head probabilities); >=0.50 predictions are recomputed. POTER fixed WGA86/83/74/58/23 reproduced as checks, never used as optimization targets. Seed coverage remains POTER1, ERM0, DPE0; no multi-seed claim. Official labels are unchanged; known source-label issues and legacy DPE head/backbone provenance limitations from previous reports remain.

Grid91 candidates0.05…0.95. Per-sigma tie-break: WGA, AA, proximity to0.5, lower threshold. `best_thresholds.csv` includes all WGA-tied grid thresholds and remaining AA ties. Probabilities are unrounded; percentages are rounded only for Markdown display. `audit.json` retains denominators and hard-label/logit checks. Repeated background images and severities are dependent observations; no naive confidence interval or statistical-significance claim is made after oracle selection.

## Interpretation and next-step decision

Threshold changes can reveal a favorable decision boundary in these cached scores. Even large in-sample WGA recovery would show only that **this cohort's scores contain useful class-discriminative ordering**; it would not identify calibration, learned priors, representation mechanisms, or the cause of POTER training behavior. Inspect the original-clean, composite-clean, AA and Waterbird trade-offs before calling any recovery useful.

The previous independent-development limitation remains: POTER validation overlaps OT reference; DPE validation fitted its heads. This oracle exercise does not repair it and does not replace the validation-only report or its locked JSON. **Prioritize genuinely independent four-group threshold selection/calibration feasibility before expensive VA-POTER retraining.** No new training algorithm or mitigation is justified solely by an oracle gain. A future genuine method needs separate labeled development data (or a separately specified label-free adaptation rule) and untouched final evaluation.

## Reproduction and exports

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_test_oracle_threshold_sweep.py --input outputs/mild_blur_v2/predictions.csv --output outputs/test_oracle_threshold_sweep --original-test results/poter_failure/original_test_predictions.csv
```

Use `--skip-exact` to omit the supplemental exact-boundary pass. `--original-test` may point to a missing file; that condition is marked NOT AVAILABLE, with no substitute inference. The reusable script accepts only the specified audited100-foreground factorial design. Required CSVs/three PNGs are present; additional exact optima, joint candidates, audit/config/status and hashes support review. Existing predictions/checkpoints and earlier threshold locks are untouched.
'''
    if (out/'tests.log').exists():text+='\nAutomated checks: '+(out/'tests.log').read_text().strip().splitlines()[-1]+'\n'
    (out/'report.md').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'outputs/mild_blur_v2/predictions.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/test_oracle_threshold_sweep')
    parser.add_argument('--original-test',type=Path,default=ROOT/'results/poter_failure/original_test_predictions.csv')
    parser.add_argument('--metadata',type=Path,default=ROOT/'data/waterbirds/metadata.csv')
    parser.add_argument('--skip-exact',action='store_true')
    args=parser.parse_args()
    if args.output.resolve() in [args.input.parent.resolve(),args.original_test.parent.resolve()]:
        parser.error('Use a separate output directory to preserve existing predictions')
    run(args.input,args.output,args.original_test,args.metadata,not args.skip_exact)


if __name__=='__main__':main()
