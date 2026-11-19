"""CPU score-only FDAT candidate and constrained threshold feasibility primitives."""
import numpy as np
import pandas as pd

from .threshold_probe import group_codes

METRICS=['aa_pct','wga_pct','landbird_accuracy_pct','waterbird_accuracy_pct']


def fdat(p_full,p_gray,lambda_,t_max):
    """Inference receives scores and fixed parameters only; no labels/metadata."""
    full,gray=np.asarray(p_full,dtype=float),np.asarray(p_gray,dtype=float)
    if full.shape!=gray.shape:raise ValueError('Full/gray score shapes differ')
    if not np.isfinite(full).all() or not np.isfinite(gray).all() or ((full<0)|(full>1)|(gray<0)|(gray>1)).any():
        raise ValueError('Invalid probability')
    if not 0<=lambda_<=1 or not .5<=t_max<=1:raise ValueError('Invalid fixed FDAT parameters')
    discrepancy=np.maximum(0.,full-gray)
    threshold=np.minimum(t_max,.5+lambda_*discrepancy)
    pred=(full>=threshold).astype(int)
    return dict(discrepancy=discrepancy,threshold=threshold,pred=pred,
                adjusted_score=full-np.minimum(lambda_*discrepancy,t_max-.5))


def prediction_metrics(frame,pred):
    y=frame.true_y.to_numpy(int);pred=np.asarray(pred,dtype=int)
    if pred.shape!=y.shape or not set(pred).issubset({0,1}):raise ValueError('Invalid prediction vector')
    g=group_codes(frame);correct=pred==y
    counts=[int((g==i).sum()) for i in range(4)]
    groups=[100*float(correct[g==i].mean()) if counts[i] else None for i in range(4)]
    fp=int(((y==0)&(pred==1)).sum());fn=int(((y==1)&(pred==0)).sum())
    baseline=(frame.p_waterbird.to_numpy()>=.5).astype(int)
    result=dict(count=len(y),n_foregrounds=int(frame.source_image_id.nunique()),
        aa_pct=100*float(correct.mean()),wga_pct=min(groups) if all(x is not None for x in groups) else None,
        landbird_accuracy_pct=100*float(correct[y==0].mean()) if (y==0).any() else None,
        waterbird_accuracy_pct=100*float(correct[y==1].mean()) if (y==1).any() else None,
        landbird_to_waterbird_count=fp,waterbird_to_landbird_count=fn,
        cfps_pred_waterbird=fp/(fp+fn) if fp+fn else None,
        cfps_pred_landbird=fn/(fp+fn) if fp+fn else None,
        fraction_predicted_waterbird=float(pred.mean()),
        prediction_flips_vs05=int((pred!=baseline).sum()),
        wrong_to_correct=int(((baseline!=y)&correct).sum()),
        correct_to_wrong=int(((baseline==y)&~correct).sum()),all_four_groups=all(counts))
    for i in range(4):result[f'group{i}_n']=counts[i];result[f'group{i}_accuracy_pct']=groups[i]
    return result


def threshold_curve(frame,thresholds):
    """Exact empirical group counts by sorted scores, >= boundary semantics."""
    ts=np.asarray(thresholds,dtype=float)
    if not np.isfinite(ts).all() or ((ts<0)|(ts>1)).any():raise ValueError('Invalid threshold grid')
    g=group_codes(frame);p=frame.p_waterbird.to_numpy();den=[];correct=[]
    for i in range(4):
        scores=np.sort(p[g==i]);den.append(len(scores))
        less=np.searchsorted(scores,ts,side='left')
        correct.append(less if i<2 else len(scores)-less)
    if not all(den):raise ValueError('Threshold objective needs all four groups')
    c=np.column_stack(correct);den=np.asarray(den)
    group=100*c/den
    fp=den[:2].sum()-c[:,:2].sum(1);fn=den[2:].sum()-c[:,2:].sum(1);errors=fp+fn
    result=dict(threshold=ts,aa_pct=100*c.sum(1)/den.sum(),wga_pct=group.min(1),
        landbird_accuracy_pct=100*c[:,:2].sum(1)/den[:2].sum(),
        waterbird_accuracy_pct=100*c[:,2:].sum(1)/den[2:].sum(),
        landbird_to_waterbird_count=fp,waterbird_to_landbird_count=fn,
        cfps_pred_waterbird=np.divide(fp,errors,out=np.full(len(ts),np.nan),where=errors!=0),
        cfps_pred_landbird=np.divide(fn,errors,out=np.full(len(ts),np.nan),where=errors!=0),
        fraction_predicted_waterbird=(fp+c[:,2:].sum(1))/den.sum(),count=np.repeat(den.sum(),len(ts)))
    for i in range(4):result[f'group{i}_accuracy_pct']=group[:,i];result[f'group{i}_n']=np.repeat(den[i],len(ts))
    return pd.DataFrame(result)


def constrained_candidates(natural,original,thresholds,blur_sigmas):
    ts=np.asarray(thresholds)
    curve=threshold_curve(original,ts).add_prefix('original_')
    curve=curve.rename(columns={'original_threshold':'threshold'})
    objective=[];aa=[]
    for sigma in sorted(natural.sigma.unique()):
        part=threshold_curve(natural[natural.sigma==sigma],ts)
        prefix=f'sigma{sigma:g}_'
        for name in part.columns:
            if name!='threshold':curve[prefix+name]=part[name].to_numpy()
        if sigma in blur_sigmas:objective.append(part.wga_pct.to_numpy());aa.append(part.aa_pct.to_numpy())
    curve['mean_blur_wga_pct']=np.mean(objective,axis=0);curve['mean_blur_aa_pct']=np.mean(aa,axis=0)
    baseline=float(threshold_curve(original,[.5]).wga_pct.iloc[0])
    curve['original_baseline05_wga_pct']=baseline
    curve['original_clean_drop_pp']=baseline-curve.original_wga_pct
    return curve


def choose_constrained(curve,epsilon):
    eligible=curve[curve.original_clean_drop_pp<=epsilon+1e-10]
    if eligible.empty:raise ValueError('No feasible threshold (identity0.5 should always be feasible)')
    winner=min(eligible.to_dict('records'),key=lambda r:(-round(r['mean_blur_wga_pct'],12),
        -round(r['original_wga_pct'],12),-round(r['mean_blur_aa_pct'],12),round(abs(r['threshold']-.5),14),r['threshold']))
    return winner


def join_pairs(frame):
    """Only natural/gray pairs with identical source, layer, alpha and checkpoint seed."""
    natural=frame[(frame.control=='natural')&(frame.source_split=='test')].copy()
    gray=frame[(frame.control=='foreground_only')&(frame.source_split=='test')].copy()
    keys=['model_name','model_seed','source_image_id','foreground_id','sigma']
    if gray.duplicated(keys).any():raise ValueError('Duplicate gray prediction')
    if not (natural.groupby(keys,dropna=False).size()==4).all():raise ValueError('Exactly four natural replicas required')
    selected=gray[keys+['p_waterbird','image_id','layer_sha256','alpha_sha256','mask_sha256','source_sha256','true_y']].rename(
        columns={k:'gray_'+k for k in ['p_waterbird','image_id','layer_sha256','alpha_sha256','mask_sha256','source_sha256','true_y']})
    joined=natural.merge(selected,on=keys,how='left',validate='many_to_one',indicator=True)
    if (joined._merge!='both').any():raise ValueError('Missing gray counterpart')
    for name in ['layer_sha256','alpha_sha256','mask_sha256','source_sha256','true_y']:
        if not (joined[name]==joined['gray_'+name]).all():raise ValueError('Full/gray foreground/mask identity differs: '+name)
    if not (joined.groupby(keys).bg_type.nunique()==2).all():raise ValueError('Both backgrounds required')
    joined['p_full']=joined.p_waterbird
    joined['p_gray']=joined.gray_p_waterbird
    return joined.drop(columns=['_merge'])


def species_folds(cohort,k,seed):
    meta=cohort[['source_image_id','species_id','true_y']].drop_duplicates().sort_values('source_image_id')
    if meta.source_image_id.duplicated().any() or (meta.groupby('species_id').true_y.nunique()!=1).any():
        raise ValueError('Inconsistent foreground/species labels')
    rng=np.random.default_rng(seed);assignment={}
    for y in [0,1]:
        names=np.array(sorted(meta[meta.true_y==y].species_id.unique()))
        if len(names)<k:raise ValueError('Insufficient species for stratified species folds')
        for i,name in enumerate(rng.permutation(names)):assignment[name]=i%k
    meta['fold']=meta.species_id.map(assignment)
    return meta


def choose_fdat(training,parameters,blur_sigmas):
    scores=[]
    for lam,cap in parameters:
        result=fdat(training.p_full,training.p_gray,lam,cap)
        labeled=training.assign(_prediction=result['pred']);rows={}
        for sigma,part in labeled.groupby('sigma'):
            rows[float(sigma)]=prediction_metrics(part,part._prediction)
        objective=float(np.mean([rows[s]['wga_pct'] for s in blur_sigmas]))
        aa=float(np.mean([rows[s]['aa_pct'] for s in blur_sigmas]))
        scores.append(dict(lambda_=lam,t_max=cap,mean_blur_wga_pct=objective,
            mean_blur_aa_pct=aa,composite_clean_wga_pct=rows[0.]['wga_pct']))
    return min(scores,key=lambda r:(-round(r['mean_blur_wga_pct'],12),-round(r['composite_clean_wga_pct'],12),
        -round(r['mean_blur_aa_pct'],12),r['lambda_'],r['t_max']))
