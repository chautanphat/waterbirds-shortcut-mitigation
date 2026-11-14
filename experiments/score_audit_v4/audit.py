"""Phase 0 ONLY: historically inspected test is diagnostic, never selection data."""
import io
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .model import observable
from .evaluate import summary, expert_stats
from .journal import csvwrite
from experiments.discrete_weighting.store import atomic, dump, sha

LABEL = 'HISTORICAL TEST / DIAGNOSTIC ONLY'
EDGES = [.5,.515,.530,.540,.550,.6,.75,1.000001]
CAL_EDGES = [.5,.55,.6,.65,.7,.75,.85,1.000001]


def png(path, fig):
    b=io.BytesIO(); fig.savefig(b,format='png',dpi=140,bbox_inches='tight'); plt.close(fig); atomic(path,b.getvalue())


def export_seed(root, seed, historical):
    root=Path(root); folder=root/f'phase0/seed{seed}'
    with np.load(historical/f'prediction_cache_seed{seed}.npz',allow_pickle=False) as f: d=dict(f)
    raw=d['probs']; calibrated=d['calibrated_probs']; w=d['weights']; y=d['y']; group=d['group']
    assert np.all(w>0) and np.allclose(w.sum(1),1)
    np.testing.assert_allclose((calibrated*w[...,None]).sum(1),d['gate_probs'],atol=1e-12,rtol=0)
    np.testing.assert_array_equal(d['gate_probs'].argmax(1),d['gate_predictions'])
    prov=json.loads(str(d['provenance'])); model_path=Path('outputs/weighting/fits')/prov['fit_key']/'model.joblib'
    model=joblib.load(model_path); assert model.method=='D19'
    # Gate.forward is direct softmax(v+bias), and D19 uses no residual transform.
    with torch.no_grad():
        x=torch.tensor(model.feature_data({'probs':calibrated}))
        gate=model.model; logits=(gate.net(x.flatten(1)) if gate.joint else gate.net(x).squeeze(-1))+gate.bias
        np.testing.assert_allclose(logits.softmax(1).numpy(),w,atol=2e-7,rtol=2e-6)
    exact=logits.numpy(); centered=exact-exact.mean(1,keepdims=True)
    # dist is already normalized Euclidean prototype distance, as stored by predict_heads.
    source=Path(prov['original_expert_cache'])
    with np.load(source,allow_pickle=False) as f: distances=f['distances']; ids=f['sample_id']; np.testing.assert_array_equal(ids,d['sample_id'])
    correct=d['gate_probs'].argmax(1)==y; uniform=raw.mean(1).argmax(1); basecorrect=uniform==y
    flip=np.where(correct & ~basecorrect,'corrected',np.where(~correct & basecorrect,'broken','unchanged'))
    rows=[]
    for name,p in [('raw',raw),('calibrated',calibrated)]:
        pred,conf,loo=observable(p); n=len(y)
        frame=pd.DataFrame(dict(seed=np.repeat(seed,n*15),split=LABEL,representation=name,
            sample_id=np.repeat(ids,15),fold=-1,expert_id=np.tile(np.arange(15),n),
            y_true_AUDIT_ONLY=np.repeat(y,15),group_AUDIT_ONLY=np.repeat(group,15),
            predicted_class=pred.ravel(),confidence=conf.ravel(),signed_margin=(p[...,1]-p[...,0]).ravel(),
            normalized_distance_0=distances[...,0].ravel(),normalized_distance_1=distances[...,1].ravel(),
            loo_agreement=loo.ravel(),gate_raw_logit=exact.ravel(),gate_centered_logit=centered.ravel(),
            gate_final_weight=w.ravel(),expert_correct_AUDIT_ONLY=(pred==y[:,None]).ravel(),
            uniform_prediction=np.repeat(uniform,15),gate_prediction=np.repeat(d['gate_predictions'],15),flip_type=np.repeat(flip,15)))
        path=folder/f'expert_audit_{name}.csv'; atomic(path,frame.to_csv(index=False)); rows.append(path)
    image=pd.DataFrame(dict(seed=seed,sample_id=ids,y=y,group=group,uniform_prediction=uniform,
        calibrated_uniform_prediction=calibrated.mean(1).argmax(1),gate_prediction=d['gate_predictions'],flip_type=flip,
        wrong_weight=(w*(raw.argmax(-1)!=y[:,None])).sum(1),effective_experts=1/(w*w).sum(1),
        entropy=-(w*np.log(w)).sum(1),max_weight=w.max(1),consensus=np.maximum((raw.argmax(-1)==0).mean(1),(raw.argmax(-1)==1).mean(1)),
        minority_wrong_overturn=(basecorrect & ~correct & ((raw.argmax(-1)!=y[:,None]).sum(1)<7.5) & (raw.argmax(-1)[np.arange(len(y)),w.argmax(1)]!=y))))
    path=folder/'image_diagnostics.csv'; atomic(path,image.to_csv(index=False)); rows.append(path)
    quantitative=[]
    for rep,p in [('raw',raw),('calibrated',calibrated)]:
        for kind, masks in [('overall',[('all',np.ones(len(y),bool))]),('true_group',[(str(g),group==g) for g in range(4)]),
                            ('true_class',[(str(k),y==k) for k in (0,1)])]:
            for value,mask in masks:
                quantitative.append(dict(seed=seed,representation=rep,stratum=kind,value=value,
                    **expert_stats(centered[mask],w[mask],p[mask],y[mask]),
                    **summary(d['gate_probs'][mask],y[mask])))
        # Predicted-class stratum expert summaries; ranking remains image-averaged within class.
        for k in (0,1):
            ec=p.argmax(-1)==k; corr=p.argmax(-1)==y[:,None]; vals=[]
            for ss,cc,mm in zip(centered,corr,ec):
                yes=ss[mm&cc]; no=ss[mm&~cc]
                if len(yes) and len(no):
                    gap=yes[:,None]-no[None];vals.append(float(((gap>0)+.5*(gap==0)).mean()))
            from scipy.stats import spearmanr
            from sklearn.metrics import roc_auc_score
            quantitative.append(dict(seed=seed,representation=rep,stratum='predicted_class',value=str(k),
                count=int(ec.sum()),distinct_images=int(ec.any(1).sum()),pairwise_image_ranking=float(np.mean(vals)) if vals else None,
                expert_AUROC=float(roc_auc_score(corr[ec],centered[ec])) if len(np.unique(corr[ec]))==2 else None,
                expert_Spearman=float(spearmanr(corr[ec],centered[ec]).statistic),
                mean_weight_correct=float(w[ec&corr].mean()),mean_weight_wrong=float(w[ec&~corr].mean())))
    path=folder/'quantitative_checks.csv'; csvwrite(path,quantitative); rows.append(path)
    for label,p in [('raw_uniform',raw.mean(1)),('calibrated_uniform',calibrated.mean(1)),('D19',d['gate_probs'])]:
        path=folder/f'{label}_metrics.json';dump(path,summary(p,y,group));rows.append(path)
    note=folder/'provenance.json';dump(note,dict(label=LABEL,gate_checkpoint_sha256=sha(model_path),
        exact_raw_logits_recovered=True,forward='net(features)+bias, direct softmax; exact stored weights validated',
        fold=-1,fold_note='final all-development ensemble, no OOF fold',distance_source=str(source),
        head_hash=prov['final_heads_sha256'],raw_uniform_vs_calibrated_uniform='Both controls reported; historical headline uses raw uniform.'))
    rows.append(note); return dict(seed=seed,exact_logits=True,images=len(y)),rows


def aggregate(root,seed,rep):
    folder=Path(root)/f'phase0/seed{seed}'; frame=pd.read_csv(folder/f'expert_audit_{rep}.csv')
    edges=EDGES if rep=='raw' else CAL_EDGES
    labels=[f'{a:.3f}–{min(b,1):.3f}' for a,b in zip(edges[:-1],edges[1:])]
    frame['confidence_bin']=pd.cut(frame.confidence,edges,right=False,labels=labels)
    frame['agreement_bin']=pd.cut(frame.loo_agreement,[-.001,1/3,2/3,1.001],right=False,labels=['<1/3','1/3–2/3','>=2/3'])
    agg=dict(score=('gate_centered_logit','mean'),correctness=('expert_correct_AUDIT_ONLY','mean'),
             weight=('gate_final_weight','mean'),observations=('sample_id','size'),distinct_images=('sample_id','nunique'))
    first=frame.groupby(['predicted_class','confidence_bin'],observed=True).agg(**agg).reset_index()
    second=frame.groupby(['predicted_class','confidence_bin','agreement_bin'],observed=True).agg(**agg).reset_index()
    paths=[]
    for name,table in [('class_confidence',first),('class_confidence_agreement',second)]:
        table['low_support']=table.distinct_images<30;path=folder/f'{rep}_{name}.csv';atomic(path,table.to_csv(index=False));paths.append(path)
    # Each heatmap annotates value plus distinct image count, and marks n<30.
    def heat(ax,table,index,columns,value,title):
        mat=table.pivot(index=index,columns=columns,values=value)
        cnt=table.pivot(index=index,columns=columns,values='distinct_images').reindex(index=mat.index,columns=mat.columns)
        ax.imshow(mat.to_numpy(dtype=float),aspect='auto',cmap='viridis')
        ax.set_xticks(range(len(mat.columns)),labels=[str(v) for v in mat.columns],rotation=45,ha='right');ax.set_yticks(range(len(mat.index)),labels=[str(v) for v in mat.index]);ax.set_title(title);ax.set_ylabel(index)
        for i in range(len(mat.index)):
            for j in range(len(mat.columns)):
                v=mat.iloc[i,j];n=cnt.iloc[i,j]
                if pd.notna(v):ax.text(j,i,f'{v:.3f}\nn={int(n)}'+('*' if n<30 else ''),ha='center',va='center',fontsize=7,color='white')
    fig,axes=plt.subplots(4,1,figsize=(12,11))
    for ax,value in zip(axes,['score','correctness','weight','distinct_images']):heat(ax,first,'predicted_class','confidence_bin',value,value)
    fig.suptitle(f'{LABEL} | seed {seed} | {rep} | * <30 distinct images');fig.tight_layout();p=folder/f'{rep}_class_confidence.png';png(p,fig);paths.append(p)
    fig,axes=plt.subplots(2,3,figsize=(15,11))
    for k in (0,1):
        t=second[second.predicted_class==k]
        for ax,value in zip(axes[k],['score','correctness','distinct_images']):heat(ax,t,'confidence_bin','agreement_bin',value,f'predicted class {k}: {value}')
    fig.suptitle(f'{LABEL} | seed {seed} | {rep} | * <30 distinct images');fig.tight_layout();p=folder/f'{rep}_confidence_agreement.png';png(p,fig);paths.append(p)
    # Fixed unlabelled score deciles; uncertainty resamples distinct images, not experts.
    curve=[];rng=np.random.default_rng(521)
    for k in (0,1):
        sub=frame[frame.predicted_class==k].copy();sub['score_decile']=pd.qcut(sub.gate_centered_logit,10,duplicates='drop')
        for interval,t in sub.groupby('score_decile',observed=True):
            im=t.groupby('sample_id').expert_correct_AUDIT_ONLY.agg(['sum','count']);arr=im.to_numpy();v=[]
            for _ in range(300):
                z=arr[rng.integers(len(arr),size=len(arr))];v.append(z[:,0].sum()/z[:,1].sum())
            ci=np.quantile(v,[.025,.975]);curve.append(dict(predicted_class=k,score_interval=str(interval),score=t.gate_centered_logit.mean(),correctness=t.expert_correct_AUDIT_ONLY.mean(),
                observations=len(t),distinct_images=len(arr),lo=ci[0],hi=ci[1]))
    p=folder/f'{rep}_score_deciles.csv';csvwrite(p,curve);paths.append(p)
    fig,ax=plt.subplots(figsize=(9,5))
    for k in (0,1):
        t=pd.DataFrame([r for r in curve if r['predicted_class']==k]);ax.plot(t.score,t.correctness,'o-',label=f'pred class {k}')
        ax.fill_between(t.score,t.lo,t.hi,alpha=.2)
        for r in t.itertuples():ax.annotate(f'n={r.distinct_images}',(r.score,r.correctness),fontsize=6)
    ax.set(xlabel='Centered gate logit (relative preference)',ylabel='Empirical expert correctness',title=f'{LABEL}\nseed {seed}, {rep}; image bootstrap 95% CI');ax.legend();p=folder/f'{rep}_score_ranking.png';png(p,fig);paths.append(p)
    return dict(seed=seed,representation=rep),paths


def image_plots(root,seed):
    folder=Path(root)/f'phase0/seed{seed}';f=pd.read_csv(folder/'image_diagnostics.csv')
    g=f.groupby('group').agg(images=('sample_id','nunique'),wrong_weight=('wrong_weight','mean'),effective_experts=('effective_experts','mean'),entropy=('entropy','mean'),minority_overturns=('minority_wrong_overturn','sum')).reset_index()
    for kind in ['corrected','broken','unchanged']:g[kind]=[int(((f.group==k)&(f.flip_type==kind)).sum()) for k in g.group]
    p=folder/'group_mechanisms.csv';atomic(p,g.to_csv(index=False));paths=[p]
    flips=f.groupby(['group','y','uniform_prediction','gate_prediction','flip_type']).size().reset_index(name='images');p=folder/'flip_directions.csv';atomic(p,flips.to_csv(index=False));paths.append(p)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key in zip(axes,['wrong_weight','effective_experts','entropy']):
        ax.bar(g.group,g[key]);ax.set(title=key,xlabel='True subgroup (audit only)');ax.set_xticks(g.group,labels=[f'{k}\nn={n}' for k,n in zip(g.group,g.images)])
    fig.suptitle(f'{LABEL}, seed {seed}');p=folder/'group_mechanisms.png';png(p,fig);paths.append(p)
    fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].scatter(f.consensus,f.max_weight,s=3,alpha=.12)
    m=f.minority_wrong_overturn;axes[0].scatter(f.consensus[m],f.max_weight[m],s=15,c='red',label=f'minority wrong overturn ({m.sum()})');axes[0].legend(fontsize=7);axes[0].set(xlabel='Expert consensus',ylabel='Maximum final weight')
    axes[1].hist(f.max_weight,bins=20);axes[1].set(xlabel='Maximum final weight',ylabel='Distinct images');fig.suptitle(f'{LABEL}, seed {seed}')
    p=folder/'concentration_vs_consensus.png';png(p,fig);paths.append(p)
    return dict(seed=seed,minority_wrong_overturns=int(m.sum())),paths


def report(root):
    root=Path(root);rows=[];summary_rows=[]
    for seed in range(3):
        q=pd.read_csv(root/f'phase0/seed{seed}/quantitative_checks.csv');r=q[(q.stratum=='overall')&(q.representation=='calibrated')].iloc[0]
        g=pd.read_csv(root/f'phase0/seed{seed}/group_mechanisms.csv');image=pd.read_csv(root/f'phase0/seed{seed}/image_diagnostics.csv')
        raw=json.loads((root/f'phase0/seed{seed}/raw_uniform_metrics.json').read_text());cal=json.loads((root/f'phase0/seed{seed}/calibrated_uniform_metrics.json').read_text());gate=json.loads((root/f'phase0/seed{seed}/D19_metrics.json').read_text())
        summary_rows.append(dict(seed=seed,raw_uniform_AA=raw['AA'],raw_uniform_WGA=raw['WGA'],calibrated_uniform_AA=cal['AA'],calibrated_uniform_WGA=cal['WGA'],D19_AA=gate['AA'],D19_WGA=gate['WGA']))
        rows.append(f'| {seed} | {r.pairwise_image_ranking:.4f} | {r.mean_weight_correct:.5f} | {r.mean_weight_wrong:.5f} | {r.effective_members:.2f} | {int(g.corrected.sum())} | {int(g.broken.sum())} | {int(image.minority_wrong_overturn.sum())} |')
    flips=pd.concat([pd.read_csv(root/f'phase0/seed{s}/flip_directions.csv').assign(seed=s) for s in range(3)])
    changed=flips[flips.uniform_prediction!=flips.gate_prediction]
    down=int(changed[changed.uniform_prediction==1].images.sum());total=int(changed.images.sum())
    table=root/'phase0/historical_summary.csv';csvwrite(table,summary_rows)
    text='\n'.join(['# Phase 0 — D19 mechanism audit','',f'**{LABEL}. No fitting or candidate selection uses these rows.**','',
        'Exact raw logits were recovered from the saved direct-softmax D19 checkpoints and the saved calibrated probabilities; forward weights match cached weights. Centered logits remove the common within-image offset. Raw and calibrated confidence views have separately predeclared fixed edges. All expert rows, counts, source tables, PNGs and checkpoint hashes are under phase0/seed*/.', '',
        '| Seed | Image-mean pair ranking | Mean w correct | Mean w wrong | Effective experts | Corrected | Broken | Minority wrong overturn |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']+rows+['',
        f'Changed labels Waterbird→Landbird: {down}/{total} across seed/image rows. Seeds share images and backbone; this is not an independent sample count.', '',
        '| Hypothesis | Conclusion | Evidence / limit |','|---|---|---|',
        '| H1 predicted-class bias | Supported descriptively | Flip direction table and class/confidence score panels show asymmetric preference. This is not a causal intervention. |',
        '| H2 preference/correctness misalignment | Assessed by ranking and reliability panels | The exact per-seed image-mean ranking, AUROC, Spearman and score-decile correctness tables quantify misalignment; relative preference is not a calibrated correctness probability. A nonmonotonic confidence curve alone is not a bug. |',
        '| H3 softmax over-concentration | Supported as a mechanism; causality undetermined | Effective expert count, incorrect-head mass and red minority-overturn cases show where concentrated final weights reverse a correct majority. Calibration and preference are separate contributors; raw and calibrated uniform controls are both saved. |','',
        'All 8 visualization families are saved per seed: class/confidence score, correctness, weight and support; class/confidence/agreement score/correctness/support; score-decile correctness with 95% distinct-image bootstrap intervals; subgroup wrong-weight/effective-count/entropy and flip tables; maximum-weight/consensus scatter and histogram. An asterisk marks fewer than 30 distinct images. Expert AUROC/Spearman are descriptive correlated-observation summaries, with no independent-expert significance claim. ECE uses 15 confidence bins weighted by images; NLL averages one ensemble true-class log probability per image.','',
        'Phase 1 code has no import or read path into these diagnostic rows. D19 is not rescored on development because its final heads differ from fold-specific OOF heads; those would not be an honest matched weighting comparison. D5 historical aggregate remains a diagnostic control only.'])+'\n'
    path=root/'phase0_mechanism_report.md';atomic(path,text);return dict(phase0='COMPLETE',historical_test_diagnostic_only=True),[path,table]
