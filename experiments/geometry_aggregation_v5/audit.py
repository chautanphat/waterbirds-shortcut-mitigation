"""Fixed geometry diagnostics on development only; never supplies selection scores."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import softmax
from .persistence import atomic,dump
from experiments.discrete_weighting.store import sha
from experiments.score_audit_v4.model import observable
from experiments.weighting.data import predict_heads
from .io import load,save_npz,figure
from .journal import csvwrite

SCORES=['negative_min_distance','negative_entropy','confidence','distance_margin']


def export(root,dev,seed,split):
    d=load(dev/f'seed{seed}/{split}.npz');p=d['probs'];dist=d['distances'];pred,conf,loo=observable(p)
    ent=-(p*np.log(p.clip(1e-15))).sum(-1);dm=dist.min(-1);margin=np.ptp(dist,axis=-1);correct=pred==d['y'][:,None]
    uniform_wrong=p.mean(1).argmax(1)!=d['y'];n=len(p)
    f=pd.DataFrame(dict(seed=seed,split=split,sample_id=np.repeat(d['sample_id'],15),fold=np.repeat(d['fold'],15),expert_slot=np.tile(np.arange(15),n),
        predicted_class=pred.ravel(),confidence=conf.ravel(),entropy=ent.ravel(),min_distance=dm.ravel(),distance_margin=margin.ravel(),loo_agreement=loo.ravel(),
        expert_correct_AUDIT_ONLY=correct.ravel(),y_AUDIT_ONLY=np.repeat(d['y'],15),group_AUDIT_ONLY=np.repeat(d['group'],15),uniform_wrong_AUDIT_ONLY=np.repeat(uniform_wrong,15)))
    folder=root/f'phase0/seed{seed}/{split}';csv=folder/'expert_geometry.csv';atomic(csv,f.to_csv(index=False))
    cache=folder/'ranking_inputs.npz';save_npz(cache,negative_min_distance=-dm,negative_entropy=-ent,confidence=conf,distance_margin=margin,
        correct=correct,sample_id=d['sample_id'],y=d['y'],group=d['group'],uniform_wrong=uniform_wrong)
    return dict(seed=seed,split=split,images=n,uniform_wrong_images=int(uniform_wrong.sum())),[csv,cache]


def pair_values(score,correct):
    values=np.full(len(score),np.nan)
    for i,(s,c) in enumerate(zip(score,correct)):
        if c.any() and (~c).any():
            gap=s[c][:,None]-s[~c][None];values[i]=((gap>0)+.5*(gap==0)).mean()
    return values


def mean_ci(v,repeats=2000):
    v=np.asarray(v);v=v[np.isfinite(v)]
    if not len(v):return None,None,None
    rng=np.random.default_rng(51015);means=np.array([v[rng.integers(len(v),size=len(v))].mean() for _ in range(repeats)])
    lo,hi=np.quantile(means,[.025,.975]);return float(v.mean()),float(lo),float(hi)


def rank(root,seed,split,score):
    folder=root/f'phase0/seed{seed}/{split}';z=load(folder/'ranking_inputs.npz');v=pair_values(z[score],z['correct']);conf=pair_values(z['confidence'],z['correct']);rows=[]
    for subset in ['all','uniform_wrong']:
        selected=np.ones(len(v),bool) if subset=='all' else z['uniform_wrong']
        for group in ['all',0,1,2,3]:
            m=selected if group=='all' else selected&(z['group']==group);usable=m&np.isfinite(v)
            mean,lo,hi=mean_ci(v[usable]);delta,dlo,dhi=mean_ci((v-conf)[usable])
            rows.append(dict(seed=seed,split=split,score=score,subset=subset,group_AUDIT_ONLY=str(group),images=int(m.sum()),mixed_images=int(usable.sum()),
                ranking_accuracy=mean,CI95_low=lo,CI95_high=hi,paired_delta_vs_confidence=delta,paired_delta_CI95_low=dlo,paired_delta_CI95_high=dhi,
                oracle_any_correct=float(z['correct'][m].any(1).mean()) if m.any() else None,
                mean_correct_experts=float(z['correct'][m].sum(1).mean()) if m.any() else None,
                minority_correct_images=int(((z['correct'].sum(1)>0)&(z['correct'].sum(1)<7.5)&m).sum())))
    csv=folder/f'ranking_{score}.csv';csvwrite(csv,rows);p=folder/f'image_ranking_{score}.npz';save_npz(p,values=v,sample_id=z['sample_id'])
    return dict(seed=seed,split=split,score=score,overall=rows[0]['ranking_accuracy'],uniform_wrong=rows[5]['ranking_accuracy']),[csv,p]


def scales(root,dev,seed):
    rows=[];source=[]
    for fold in range(3):
        path=Path(f'outputs/weighting/seed{seed}/heads_fold{fold}.pt');heads=torch.load(path,weights_only=True)
        parts=[load(dev/f'seed{seed}/{s}.npz') for s in ['meta_train','meta_select']]
        dist=np.concatenate([d['distances'][d['fold']==fold] for d in parts]);logits=np.concatenate([d['logits'][d['fold']==fold] for d in parts])
        scale=np.abs(heads['scales'].numpy());residual=float(np.abs(logits+scale[None,:,None]*dist).max());assert residual<2e-6
        # Source normalization and learned scale checked against real parameter inference.
        d=parts[0];idx=np.flatnonzero(d['fold']==fold)[:3];own=predict_heads(heads,d['embedding'][idx])
        np.testing.assert_allclose(own['distances'],d['distances'][idx],atol=2e-6,rtol=0)
        np.testing.assert_allclose(own['probs'],d['probs'][idx],atol=2e-6,rtol=0)
        for j in range(15):rows.append(dict(seed=seed,fold=fold,head_slot=j,actual_ensemble_sha256=sha(path),learned_abs_scale=float(scale[j]),
            dmin_mean=float(dist[:,j].min(-1).mean()),dmin_std=float(dist[:,j].min(-1).std()),dmin_min=float(dist[:,j].min(-1).min()),dmin_max=float(dist[:,j].min(-1).max()),
            max_logit_distance_residual=residual,prototype_raw_norm_0=float(heads['prototypes'][j,0].norm()),prototype_raw_norm_1=float(heads['prototypes'][j,1].norm())))
        source.append(dict(fold=fold,sha256=sha(path),formula='logit=-abs(learned_scale)*normalized_euclidean_distance',max_abs_residual=residual))
    path=root/f'phase0/seed{seed}/head_scale_audit.csv';csvwrite(path,rows);prov=path.with_suffix('.json');dump(prov,dict(source=source,feature_and_prototype_vectors_L2_normalized=True,
        feature_preprocessing='standardize each image embedding across coordinates, then L2 normalize',training_entropic_scale_30_not_in_inference=True))
    fig,axes=plt.subplots(1,2,figsize=(12,4));frame=pd.DataFrame(rows)
    for f in range(3):
        t=frame[frame.fold==f];axes[0].errorbar(t.head_slot,t.dmin_mean,yerr=t.dmin_std,fmt='.-',label=f'actual fold{f} ensemble');axes[1].plot(t.head_slot,t.learned_abs_scale,'.-',label=f'fold{f}')
    axes[0].set(title='dmin mean ± image SD',xlabel='Head slot (identity differs by fold)',ylabel='Normalized Euclidean distance');axes[0].legend(fontsize=7)
    axes[1].set(title='Learned logit scale',xlabel='Head slot',ylabel='abs(scale)');fig.suptitle(f'Development geometry: seed{seed}');pic=path.with_suffix('.png');figure(pic,fig)
    return dict(seed=seed,geometry_source='PASS',heads=45),[path,prov,pic]


def plots(root,dev,seed,split):
    folder=root/f'phase0/seed{seed}/{split}';f=pd.read_csv(folder/'expert_geometry.csv');train=pd.read_csv(root/f'phase0/seed{seed}/meta_train/expert_geometry.csv')
    f['distance_bin']=0;f['confidence_bin']=0
    edges=[]
    for k in (0,1):
        tr=train[train.predicted_class==k];mask=f.predicted_class==k
        de=np.quantile(tr.min_distance,[.25,.5,.75]);ce=np.quantile(tr.confidence,[.25,.5,.75]);edges.append(dict(predicted_class=k,distance_edges=de.tolist(),confidence_edges=ce.tolist()))
        f.loc[mask,'distance_bin']=np.searchsorted(de,f.loc[mask,'min_distance'],side='right');f.loc[mask,'confidence_bin']=np.searchsorted(ce,f.loc[mask,'confidence'],side='right')
    f['agreement_bin']=np.searchsorted([1/3,2/3],f.loo_agreement,side='right')
    agg=dict(correctness=('expert_correct_AUDIT_ONLY','mean'),observations=('sample_id','size'),distinct_images=('sample_id','nunique'),
        mean_distance=('min_distance','mean'),mean_confidence=('confidence','mean'),mean_agreement=('loo_agreement','mean'))
    tables=[('distance_confidence',['predicted_class','distance_bin','confidence_bin']),('distance_agreement',['predicted_class','distance_bin','agreement_bin'])]
    paths=[]
    for name,keys in tables:
        t=f.groupby(keys,observed=True).agg(**agg).reset_index();t['low_support']=t.distinct_images<30
        p=folder/f'{name}_source.csv';atomic(p,t.to_csv(index=False));paths.append(p)
        fig,axes=plt.subplots(2,3,figsize=(13,8));other=keys[-1]
        for k in (0,1):
            sub=t[t.predicted_class==k]
            for ax,value in zip(axes[k],['correctness','mean_distance','mean_agreement']):
                mat=sub.pivot(index='distance_bin',columns=other,values=value);cnt=sub.pivot(index='distance_bin',columns=other,values='distinct_images')
                ax.imshow(mat,aspect='auto',vmin=0 if value=='correctness' else None,vmax=1 if value=='correctness' else None)
                ax.set(title=f'class{k}: {value}',xlabel=other,ylabel='Distance quantile bin');ax.set_xticks(range(len(mat.columns)),labels=mat.columns);ax.set_yticks(range(len(mat.index)),labels=mat.index)
                for i in range(len(mat.index)):
                    for j in range(len(mat.columns)):
                        if pd.notna(mat.iloc[i,j]):ax.text(j,i,f'{mat.iloc[i,j]:.3f}\nn={int(cnt.iloc[i,j])}'+('*' if cnt.iloc[i,j]<30 else ''),ha='center',fontsize=7,color='white')
        fig.suptitle(f'EXPLORATORY DEVELOPMENT: seed{seed}, {split}; * <30 distinct images; bins fit meta_train')
        p=folder/f'{name}.png';figure(p,fig);paths.append(p)
    # Distance × predicted class panels, counts and score in numeric source.
    one=f.groupby(['predicted_class','distance_bin'],observed=True).agg(**agg).reset_index();p=folder/'class_distance_source.csv';atomic(p,one.to_csv(index=False));paths.append(p)
    distribution=f.groupby(['predicted_class','expert_correct_AUDIT_ONLY']).agg(images=('sample_id','nunique'),expert_observations=('sample_id','size'),
        min_distance_mean=('min_distance','mean'),min_distance_sd=('min_distance','std'),margin_mean=('distance_margin','mean'),confidence_mean=('confidence','mean'),entropy_mean=('entropy','mean')).reset_index()
    p=folder/'class_correctness_distributions.csv';atomic(p,distribution.to_csv(index=False));paths.append(p)
    sample=f.sample(n=min(2500,len(f)),random_state=51);fig,ax=plt.subplots(figsize=(8,5))
    for value,label in [(True,'correct expert'),(False,'wrong expert')]:
        t=sample[sample.expert_correct_AUDIT_ONLY==value];ax.scatter(-t.min_distance,t.confidence,s=5,alpha=.2,label=label)
    ax.legend();ax.set(xlabel='Negative minimum prototype distance',ylabel='Maximum raw member probability',title=f'Development seed{seed} {split}; deterministic plotting subset')
    p=folder/'distance_vs_confidence.png';figure(p,fig);paths.append(p)
    p=folder/'scatter_source.csv';atomic(p,sample.to_csv(index=False));paths.append(p)
    p=folder/'train_only_bin_edges.json';dump(p,dict(edges=edges,fit_split='meta_train'));paths.append(p)
    # Conditional margin strata: does dmin offer ranking beyond separation?
    z=load(folder/'ranking_inputs.npz');v=pair_values(z['negative_min_distance'],z['correct']);margin=z['distance_margin'].mean(1)
    with np.load(root/f'phase0/seed{seed}/meta_train/ranking_inputs.npz',allow_pickle=False) as tr:cut=np.quantile(tr['distance_margin'].mean(1),[1/3,2/3])
    strata=np.searchsorted(cut,margin,side='right');conditional=[]
    for b in range(3):
        m=(strata==b)&z['uniform_wrong'];value,lo,hi=mean_ci(v[m]);conditional.append(dict(margin_stratum=b,uniform_wrong_images=int(m.sum()),mixed_images=int(np.isfinite(v[m]).sum()),dmin_rank=value,lo=lo,hi=hi))
    p=folder/'distance_rank_conditional_on_margin.csv';csvwrite(p,conditional);paths.append(p)
    return dict(seed=seed,split=split,figures=3),paths


def report(root):
    rows=[]
    for seed in range(3):
        for split in ['meta_train','meta_select']:
            for score in SCORES:rows.extend(pd.read_csv(root/f'phase0/seed{seed}/{split}/ranking_{score}.csv').to_dict('records'))
    f=pd.DataFrame(rows);table=root/'phase0_expert_ranking.csv';atomic(table,f.to_csv(index=False))
    focus=f[(f.split=='meta_train')&(f.subset=='uniform_wrong')&(f.group_AUDIT_ONLY.astype(str)=='all')]
    dmin=focus[focus.score=='negative_min_distance'];support=sum(dmin.ranking_accuracy>.5)>=2 and sum(dmin.paired_delta_vs_confidence>0)>=2
    result=dict(supports_distance_routing='yes' if support else 'no',criterion='On meta_train uniform-wrong mixed-expert images, -dmin rank >.5 and better than confidence in at least2/3 seeds; diagnostic only.',
        historical_test_used=False,group_selection=False,shared_ERMstar_backbone=True,
        formula='logit_jk=-abs(scale_j)*D_jk; D uses L2-normalized standardized embedding and L2-normalized prototype')
    decision=root/'phase0_audit.json';dump(decision,result)
    lines=['# Phase 0 — geometry/reliability audit','',
        '**Exploratory development only. No historical test rows are read. Groups are post-hoc diagnostics of the four fixed scores; never a selection input.**','',
        '## Uniform-wrong images: the critical diagnostic','',
        '| Seed | Split | Score | Wrong images | Mixed-expert images | Image-mean ranking | 95% CI | Paired Δ versus confidence |','|---|---|---|---:|---:|---:|---|---:|']
    for r in f[(f.subset=='uniform_wrong')&(f.group_AUDIT_ONLY.astype(str)=='all')].to_dict('records'):
        lines.append(f'| {r["seed"]} | {r["split"]} | {r["score"]} | {r["images"]} | {r["mixed_images"]} | {r["ranking_accuracy"]:.4f} | [{r["CI95_low"]:.4f}, {r["CI95_high"]:.4f}] | {r["paired_delta_vs_confidence"]:+.4f} |')
    lines+=['', 'Distance-routing diagnostic: **'+result['supports_distance_routing'].upper()+'** under the rule frozen above. The full table contains all-image results, paired difference intervals and subgroup rows; images without both expert outcomes have no pair-ranking estimate and are counted separately. The oracle-any-correct fraction is upper-bound headroom, not evidence of an attainable label-free router.', '',
        '## Geometry and scaling','',
        'Source inspection and actual checkpoint inference verify normalized Euclidean distances after per-image embedding standardization/L2 normalization and prototype L2 normalization. The learnable positive scale differs across actual heads: logits are −abs(scale_j)×distance, not automatically −distance. Entropic scale30 belongs to training only. Per-fold head means, SDs, scale and raw prototype norms appear in head_scale_audit.csv/PNG. Normalized distances share a Euclidean unit but have different head distributions; G2 therefore fits separate location/scale for every actual fold/head using meta_train only.', '',
        'The distance-margin conditional table checks -dmin ranking within train-derived margin strata on uniform-wrong images. It is descriptive and low support may preclude a strong conclusion about independent information. Binary confidence and negative entropy have theoretically identical ordering apart from finite-precision ties. Probability reconstruction is checked against original logits, not negative distances alone.', '',
        '## Plots and uncertainty','',
        'Every seed/split has distance×confidence and distance×agreement correctness/distance/agreement heatmaps with distinct-image counts, class×distance source tables, correctness/predicted-class distributions and negative-distance/confidence scatter. Quantile edges are fitted on meta_train; asterisks mark <30 distinct images. Group-specific oracle headroom and minority-correct counts are in the ranking table. Bootstrap intervals resample image-level pair statistics, never individual expert rows; repeated images across seeds are not independent backbone replicates.', '',
        'Continue the bounded preregistered screen regardless of this diagnostic; do not widen the grid or infer robustness from nearest-prototype intuition.']
    report=root/'phase0_geometry_audit.md';atomic(report,'\n'.join(lines)+'\n')
    return result,[table,decision,report]
