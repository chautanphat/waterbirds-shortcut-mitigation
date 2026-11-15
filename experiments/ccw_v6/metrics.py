"""Group-free B evaluation and distinct-image uncertainty; no training imports."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from experiments.score_audit_v4.evaluate import summary
from .model import teacher


def aa_eligible(delta_pp):return float(delta_pp)>-1.


def basic(p,y,uniform,g1):
    r=summary(p,y);u=summary(uniform,y);g=summary(g1,y)
    r.update(delta_AA_U0_pp=(r['AA']-u['AA'])*100,delta_AA_G1_pp=(r['AA']-g['AA'])*100,
        loss_class0=float(-np.log(p[y==0,0].clip(1e-15)).mean()),loss_class1=float(-np.log(p[y==1,1].clip(1e-15)).mean()))
    return r


def paired_image_ci(values,ids,repeats=500):
    values=np.asarray(values,float);ids=np.asarray(ids).astype(str);unique=np.unique(ids)
    per=np.array([np.nanmean(values[ids==sid]) if np.isfinite(values[ids==sid]).any() else np.nan for sid in unique]);per=per[np.isfinite(per)]
    if not len(per):return dict(mean=None,CI95=[None,None],distinct_images=0)
    rng=np.random.default_rng(66123);sample=[per[rng.integers(len(per),size=len(per))].mean() for _ in range(repeats)]
    return dict(mean=float(per.mean()),CI95=np.quantile(sample,[.025,.975]).tolist(),distinct_images=len(per))


def metric_ci(p,uniform,g1,y,ids,repeats=500):
    assert len(ids)==len(set(ids))
    rng=np.random.default_rng(66124);pools=[np.flatnonzero(y==k) for k in (0,1)];values=[]
    for _ in range(repeats):
        take=np.concatenate([rng.choice(pool,len(pool),replace=True) for pool in pools]);r=basic(p[take],y[take],uniform[take],g1[take]);values.append([r['delta_AA_U0_pp'],r['delta_AA_G1_pp'],r['worst_class_NLL']])
    q=np.quantile(values,[.025,.975],axis=0)
    return dict(delta_AA_U0_CI95_pp=q[:,0].tolist(),delta_AA_G1_CI95_pp=q[:,1].tolist(),worst_class_NLL_CI95=q[:,2].tolist(),
        bootstrap='500 paired distinct-image class-stratified resamples; conditional on fitted heads/tables')


def diagnostics(out,member_p,y,q20):
    p=out['probs'];u=out['uniform'];w=out['weights'];r=out['r'];n=len(y)
    t=teacher(member_p,y);gain=(r*t).sum(1);margin=np.abs(u[:,1]-.5);correct=p.argmax(1)==y;ucorrect=u.argmax(1)==y
    geom=np.ptp(out['geometry_scores'],axis=1);adv=np.ptp(out['advantage_scores'],axis=1)
    maxw=w.max(1);ties=np.isclose(w,maxw[:,None],atol=1e-15,rtol=0).sum(1)
    return dict(gain=gain,low_margin=margin<=q20,uniform_wrong=~ucorrect,corrected=correct&~ucorrect,broken=~correct&ucorrect,
        effective_members=1/(w*w).sum(1),entropy=-(w*np.log(w.clip(1e-15))).sum(1),
        top_head=w.argmax(1),unique_top=ties==1,top_ties=ties,geometry_range=geom,advantage_range=adv,advantage_dominates=adv>geom,
        true_probability_direction=p[np.arange(n),y]-u[np.arange(n),y])


def ranking_values(estimate,target,mask):
    values=np.full(len(estimate),np.nan)
    for i,(s,t,m) in enumerate(zip(estimate,target,mask)):
        a=s[m];b=t[m]
        if len(a)<2:continue
        gap=b[:,None]-b[None,:];usable=gap>1e-15
        if usable.any():
            delta=a[:,None]-a[None,:];values[i]=((delta[usable]>0)+.5*(delta[usable]==0)).mean()
    return values


def advantage_signal(estimate,member_p,y,ids,low_margin):
    t=teacher(member_p,y);pred=member_p.argmax(-1);uniform=member_p.mean(1).argmax(1);rows=[]
    selections=[('overall',np.ones(t.shape,bool)),('predicted_class0',pred==0),('predicted_class1',pred==1),
        ('low_uniform_margin',np.broadcast_to(low_margin[:,None],t.shape)),('disagree_with_uniform',pred!=uniform[:,None])]
    for name,m in selections:
        vals=ranking_values(estimate,t,m);ci=paired_image_ci(vals,ids)
        a=estimate[m];b=t[m];sign=b>0
        rho=float(spearmanr(a,b).statistic) if len(a)>1 and np.ptp(a)>0 and np.ptp(b)>0 else None
        auc=float(roc_auc_score(sign,a)) if len(np.unique(sign))==2 else None
        rows.append(dict(stratum=name,pair_ranking=ci['mean'],pair_ranking_CI95=ci['CI95'],distinct_mixed_images=ci['distinct_images'],
            distinct_supported_images=len(set(np.asarray(ids)[m.any(1)])),expert_observations=int(m.sum()),Spearman_descriptive=rho,AUROC_positive_t_descriptive=auc,
            uncertainty='Expert rows correlated; ranking CIs cluster by image; no iid-expert significance claim'))
    return rows
