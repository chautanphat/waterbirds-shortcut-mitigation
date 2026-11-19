"""Cached paired-score prototype and retrospective species-blocked transfer."""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.counterfactual.common import write_json
from src.counterfactual.fdat_feasibility import (fdat,prediction_metrics,join_pairs,species_folds,
    choose_fdat,constrained_candidates,choose_constrained)
from src.counterfactual.threshold_probe import group_codes

ROOT=Path(__file__).resolve().parents[1]
MODELS=['POTER','ERM_enhanced','DPE_legacy']


def original_pairs(raw,eligible):
    originals=raw[(raw.control=='original')&raw.source_image_id.isin(eligible)].copy()
    gray=raw[(raw.control=='foreground_only')&(raw.sigma==0.)]
    keys=['model_name','model_seed','source_image_id','foreground_id']
    joined=originals.merge(gray[keys+['p_waterbird']].rename(columns={'p_waterbird':'p_gray'}),on=keys,how='left',validate='one_to_one')
    if len(joined) and joined.p_gray.isna().any():raise ValueError('Missing verified original-gray counterpart')
    joined['p_full']=joined.p_waterbird
    return joined


def prototype(config,out,raw,chosen,identity_audit):
    paired=join_pairs(raw)
    keep=['model_name','model_seed','image_id','source_image_id','foreground_id','species_id','sigma','control',
          'background_id','bg_type','true_y','p_full','p_gray']
    paired[keep+['gray_image_id','layer_sha256','alpha_sha256','mask_sha256','source_sha256',
                 'model_input_sha256','normalized_tensor_sha256']].to_csv(out/'pair_identity_audit.csv',index=False)
    eligible,coverage=identity_audit(config,out,raw)
    original=original_pairs(raw,eligible)
    ids=np.array(sorted(paired.source_image_id.unique()));rng=np.random.default_rng(config['negative_control_seed'])
    order=rng.permutation(ids);permutation=dict(zip(order,np.roll(order,1)))
    pd.DataFrame(dict(source_image_id=list(permutation),gray_donor_source_image_id=list(permutation.values()))).to_csv(out/'negative_control_permutation.csv',index=False)
    if any(a==b for a,b in permutation.items()):raise ValueError('Negative control must break every foreground match')
    gray_lookup=paired.drop_duplicates(['model_name','model_seed','source_image_id','sigma']).set_index(
        ['model_name','model_seed','source_image_id','sigma']).p_gray
    donor=np.array([gray_lookup.loc[(r.model_name,r.model_seed,permutation[r.source_image_id],r.sigma)] for r in paired.itertuples()])
    parameters=list(itertools.product(config['lambdas'],config['caps']))
    grid=[];references=[];identity_checks=0;first=True
    with (out/'fdat_paired_predictions.csv').open('w',encoding='utf-8',newline='') as handle:
        for variant in ['matched_gray','shuffled_gray_negative_control']:
            scores=paired.p_gray.to_numpy() if variant=='matched_gray' else donor
            for lam,cap in parameters:
                result=fdat(paired.p_full,scores,lam,cap)
                baseline=(paired.p_full.to_numpy()>=.5).astype(int)
                if lam==0:
                    if not np.array_equal(result['pred'],baseline):raise ValueError('Identity FDAT differs from fixed0.5')
                    identity_checks+=len(baseline)
                data=paired.assign(_prediction=result['pred'])
                if variant=='matched_gray':
                    export=paired[keep].copy();export['lambda_']=lam;export['t_max']=cap
                    export['positive_discrepancy']=result['discrepancy'];export['per_image_threshold']=result['threshold']
                    export['p_adjusted']=result['adjusted_score'];export['fdat_pred_y']=result['pred'];export['fixed05_pred_y']=baseline
                    export['label_role']='evaluation only';export['parameter_role']='oracle candidate; not independently selected'
                    export.to_csv(handle,index=False,header=first);first=False
                for (model,seed,sigma),part in data.groupby(['model_name','model_seed','sigma'],sort=False):
                    op=original[original.model_name==model]
                    om=prediction_metrics(op,fdat(op.p_full,op.p_gray,lam,cap)['pred']) if len(op) and variant=='matched_gray' else {}
                    grid.append(dict(model_name=model,model_seed=int(seed),sigma=float(sigma),supplementary=sigma==4.,
                        variant=variant,lambda_=lam,t_max=cap,uncapped_ablation=cap==1.,
                        **prediction_metrics(part,part._prediction),
                        matched_original_count=len(op) if variant=='matched_gray' else 0,
                        matched_original_wga_pct=om.get('wga_pct'),matched_original_aa_pct=om.get('aa_pct'),
                        full_original_wga_status='BLOCKED: no full5794 exact gray pairs',selection_scope='TEST ORACLE / IN-SAMPLE'))
    # Full/gray prediction evidence is shared across all backgrounds; never independent row resampling.
    prior_best=pd.read_csv(ROOT/'outputs/test_oracle_threshold_sweep/best_thresholds.csv')
    prior_choices=json.loads((ROOT/'outputs/test_oracle_threshold_sweep/oracle_threshold_choices.json').read_text())
    for (model,seed,sigma),part in paired.groupby(['model_name','model_seed','sigma'],sort=False):
        methods={'fixed05':.5,'fixed065':.65,
            'prior_joint_oracle_includes_sigma0':prior_choices[model]['thresholds']['oracle_joint'],
            'per_sigma_oracle_fullcohort':float(prior_best[(prior_best.model_name==model)&(prior_best.sigma==sigma)].threshold.iloc[0])}
        selected=chosen[(chosen.model_name==model)&(chosen.search==('exact_finite_scores' if model=='POTER' else 'grid_0.001'))]
        for row in selected.itertuples():methods[f'global_constrained_eps{row.epsilon_pp:g}']=row.threshold
        for name,t in methods.items():
            references.append(dict(model_name=model,model_seed=seed,sigma=sigma,method=name,threshold=t,
                **prediction_metrics(part,(part.p_full>=t).astype(int)),selection_scope='TEST ORACLE / IN-SAMPLE'))
        references.append(dict(model_name=model,model_seed=seed,sigma=sigma,method='foreground_only_classifier',threshold=.5,
            **prediction_metrics(part,(part.p_gray>=.5).astype(int)),selection_scope='paired oracle-mask diagnostic'))
    pd.DataFrame(grid).to_csv(out/'fdat_grid.csv',index=False)
    pd.DataFrame(references).to_csv(out/'fdat_reference_baselines.csv',index=False)
    equivalence=[]
    for (model,sigma),part in paired.groupby(['model_name','sigma'],sort=False):
        candidate=fdat(part.p_full,part.p_gray,1.,1.)['pred']
        and_rule=((part.p_full>=.5)&(part.p_gray>=.5)).to_numpy(int)
        equivalence.append(dict(model_name=model,sigma=sigma,disagreements_with_binary_AND=int((candidate!=and_rule).sum()),
            full_equals_one_and_gray_below_half=int(((part.p_full==1.)&(part.p_gray<.5)).sum()),
            note='cap1 is an ablation, not literally infinite cap: full=1 can remain Waterbird by >=tie'))
    pd.DataFrame(equivalence).to_csv(out/'and_rule_equivalence.csv',index=False)
    write_json(out/'identity_control.json',dict(lambda_zero_predictions_checked=identity_checks,
        exactly_matches_fixed05=True,natural_gray_pairs=len(paired),negative_control_fixed_points=0,
        negative_control_uses_class_or_species=False,inference_inputs=['p_full','p_gray','lambda_','t_max'],
        excluded_from_inference=['true_y','species','background_group','sigma','original_composite_flag','split']))
    print(f'Stage B complete:25 parameters × matched/shuffled controls ×3 models; exact original pairs={len(eligible)}/100',flush=True)
    return paired


def species_bootstrap(frame,alternative,reference,samples,seed):
    """Conditional OOF prediction contrasts; species clusters keep all source repeats."""
    if samples<2000:raise ValueError('At least2000 cluster-bootstrap draws required')
    species=frame[['species_id','true_y']].drop_duplicates().sort_values('species_id')
    if species.species_id.duplicated().any():raise ValueError('Species has inconsistent labels')
    labels=species.true_y.to_numpy(int);names=species.species_id.tolist();position={s:i for i,s in enumerate(names)}
    index=np.array([position[s] for s in frame.species_id]);groups=group_codes(frame);y=frame.true_y.to_numpy(int)
    den=np.zeros((len(names),4));a=np.zeros_like(den);b=np.zeros_like(den)
    np.add.at(den,(index,groups),1)
    np.add.at(a,(index,groups),(np.asarray(alternative)==y).astype(int))
    np.add.at(b,(index,groups),(np.asarray(reference)==y).astype(int))
    def reduce(c,d):
        return np.column_stack([100*c.sum(1)/d.sum(1),(100*c/d).min(1),
            100*c[:,:2].sum(1)/d[:,:2].sum(1),100*c[:,2:].sum(1)/d[:,2:].sum(1)])
    rng=np.random.default_rng(seed)
    draws=np.concatenate([rng.choice(np.flatnonzero(labels==v),size=(samples,int((labels==v).sum())),replace=True) for v in [0,1]],axis=1)
    d=den[draws].sum(1);distribution=reduce(a[draws].sum(1),d)-reduce(b[draws].sum(1),d)
    point=reduce(a.sum(0)[None,:],den.sum(0)[None,:])-reduce(b.sum(0)[None,:],den.sum(0)[None,:])
    rows=[]
    for i,metric in enumerate(['aa_pct','wga_pct','landbird_accuracy_pct','waterbird_accuracy_pct']):
        lo,hi=np.quantile(distribution[:,i],[.025,.975])
        rows.append(dict(metric=metric,delta_pp=float(point[0,i]),ci95_low_pp=float(lo),ci95_high_pp=float(hi),
            bootstrap_samples=samples,bootstrap_seed=seed,n_species=len(names),n_foregrounds=frame.source_image_id.nunique(),
            bootstrap_unit='species; class-stratified; every foreground/background within a species travels together',
            threshold_reselected=False,independent_generalization_ci=False,
            caveat='Conditional on retrospective folds/tuning; folds share training species and hypothesis used this test cohort'))
    return rows


def transfer(config,out,raw,originals,chosen):
    paired=join_pairs(raw)
    membership=species_folds(paired,config['folds'],config['fold_seed'])
    membership.to_csv(out/'fold_membership.csv',index=False)
    paired=paired.merge(membership[['source_image_id','fold']],on='source_image_id',validate='many_to_one')
    parameters=list(itertools.product(config['lambdas'],config['caps']))
    fold_rows=[];predictions=[];tuning=[]
    grid=np.arange(1001)/1000
    coverage=json.loads((out/'original_coverage.json').read_text())
    exact_original=original_pairs(raw,set(coverage['exact_original_source_ids']))
    for model in MODELS:
        data=paired[paired.model_name==model]
        for fold in range(config['folds']):
            train=data[data.fold!=fold];held=data[data.fold==fold]
            train_species=set(train.species_id);held_species=set(held.species_id)
            if train_species & held_species or set(train.source_image_id)&set(held.source_image_id):raise ValueError('Fold leakage')
            selection=choose_fdat(train[train.sigma.isin(config['primary_sigmas'])],parameters,config['blur_objective_sigmas'])
            # Global comparator also excludes held-out species from original-clean fitting pool.
            full_train=originals[model][~originals[model].species_id.isin(held_species)]
            if set(full_train.species_id)&held_species:raise ValueError('Original constraint fitting leaks heldout species')
            curves=constrained_candidates(train,full_train,grid,config['blur_objective_sigmas'])
            globals_={eps:choose_constrained(curves,eps)['threshold'] for eps in config['epsilon_pp']}
            unconstrained=choose_constrained(curves,100.)['threshold']
            tuning.append(dict(model_name=model,fold=fold,**selection,training_foregrounds=train.source_image_id.nunique(),
                heldout_foregrounds=held.source_image_id.nunique(),training_species=json.dumps(sorted(train_species)),
                heldout_species=json.dumps(sorted(held_species)),original_constraint_training_images=len(full_train),
                original_constraint_excludes_heldout_species=True,
                global_eps0_threshold=globals_[0],global_eps1_threshold=globals_[1],global_eps2_threshold=globals_[2],
                global_unconstrained_threshold=unconstrained,
                clean_FDAT_tuning_constraint='not imputed: exact original-gray pairs unavailable' if len(exact_original)==0 else 'original pairing audit available; see coverage'))
            for sigma,part in held.groupby('sigma',sort=True):
                adaptive=fdat(part.p_full,part.p_gray,selection['lambda_'],selection['t_max'])
                methods={'fdat':adaptive['pred'],'fixed05':(part.p_full>=.5).to_numpy(int),
                    'fixed065':(part.p_full>=.65).to_numpy(int),'foreground_only_classifier':(part.p_gray>=.5).to_numpy(int),
                    'global_unconstrained_transfer':(part.p_full>=unconstrained).to_numpy(int)}
                for eps,t in globals_.items():methods[f'global_constrained_transfer_eps{eps}']=(part.p_full>=t).to_numpy(int)
                baseline=prediction_metrics(part,methods['fixed05'])
                export=part[['model_name','model_seed','source_image_id','foreground_id','species_id','sigma','control',
                             'image_id','true_y','bg_type','background_id','original_group','p_full','p_gray','fold']].copy()
                export['p_waterbird']=export.p_full
                export['selected_lambda']=selection['lambda_'];export['selected_t_max']=selection['t_max']
                for method,pred in methods.items():
                    export[method+'_pred']=pred
                    result=prediction_metrics(part,pred)
                    op=exact_original[(exact_original.model_name==model)&exact_original.species_id.isin(held_species)]
                    om={};ob={}
                    if len(op):
                        opred=fdat(op.p_full,op.p_gray,selection['lambda_'],selection['t_max'])['pred'] if method=='fdat' else None
                        if opred is not None:om=prediction_metrics(op,opred);ob=prediction_metrics(op,(op.p_full>=.5).astype(int))
                    orig_delta=om.get('wga_pct')-ob.get('wga_pct') if om.get('wga_pct') is not None and ob.get('wga_pct') is not None else None
                    fold_rows.append(dict(model_name=model,model_seed=int(part.model_seed.iloc[0]),fold=fold,method=method,sigma=float(sigma),
                        supplementary=sigma==4.,selected_lambda=selection['lambda_'] if method=='fdat' else None,
                        selected_t_max=selection['t_max'] if method=='fdat' else None,
                        heldout_species=json.dumps(sorted(held_species)),**result,
                        wga_gain_vs_fixed05_pp=result['wga_pct']-baseline['wga_pct'] if result['wga_pct'] is not None else None,
                        matched_original_count=len(op) if method=='fdat' else None,
                        matched_original_wga_pct=om.get('wga_pct'),matched_original_wga_change_pp=orig_delta,
                        meets_sigma2_gain8pp=result['wga_pct']-baseline['wga_pct']>=config['fdat_transfer_required_gain_pp']-1e-10 if sigma==2. else None,
                        meets_matched_original_loss2pp=orig_delta>=-config['matched_original_max_loss_pp'] if orig_delta is not None else None,
                        transfer_scope='retrospective species-blocked; not independent benchmark'))
                predictions.append(export)
        print(f'Stage C {model}:5 species-blocked folds complete; no row-random splits',flush=True)
    fold_frame=pd.DataFrame(fold_rows);fold_frame.to_csv(out/'fdat_transfer_folds.csv',index=False)
    pd.DataFrame(tuning).to_csv(out/'fdat_transfer_tuning.csv',index=False)
    oof=pd.concat(predictions,ignore_index=True);oof.to_csv(out/'fdat_transfer_predictions.csv',index=False)
    summaries=[];intervals=[]
    for (model,seed,sigma),part in oof.groupby(['model_name','model_seed','sigma'],sort=False):
        baseline=prediction_metrics(part,part.fixed05_pred)
        for method in ['fdat','fixed05','fixed065','foreground_only_classifier','global_unconstrained_transfer',
                       'global_constrained_transfer_eps0','global_constrained_transfer_eps1','global_constrained_transfer_eps2']:
            result=prediction_metrics(part,part[method+'_pred'])
            summaries.append(dict(model_name=model,model_seed=seed,sigma=sigma,method=method,**result,
                wga_gain_vs_fixed05_pp=result['wga_pct']-baseline['wga_pct'],original_FDAT_clean_protection='BLOCKED',
                scope='pooled OOF predictions; retrospective, not untouched'))
        for reference in ['fixed05','foreground_only_classifier','global_constrained_transfer_eps2']:
            for row in species_bootstrap(part,part.fdat_pred,part[reference+'_pred'],config['bootstrap_samples'],config['bootstrap_seed']):
                intervals.append(dict(model_name=model,model_seed=seed,sigma=sigma,comparison='FDAT_minus_'+reference,**row))
    pd.DataFrame(summaries).to_csv(out/'fdat_transfer_summary.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'fdat_transfer_bootstrap_ci.csv',index=False)
    species=[]
    for (model,sigma,name),part in oof.groupby(['model_name','sigma','species_id'],sort=False):
        y=part.true_y.to_numpy(int);a=part.fdat_pred.to_numpy()==y;b=part.fixed05_pred.to_numpy()==y
        species.append(dict(model_name=model,sigma=sigma,species_id=name,true_y=int(y[0]),
            n_foregrounds=part.source_image_id.nunique(),n_rows=len(part),fixed05_accuracy_pct=100*b.mean(),
            fdat_accuracy_pct=100*a.mean(),accuracy_gain_pp=100*(a.mean()-b.mean()),
            net_corrected_rows=int(a.sum()-b.sum())))
    pd.DataFrame(species).to_csv(out/'fdat_transfer_species.csv',index=False)
    return pd.DataFrame(summaries)
