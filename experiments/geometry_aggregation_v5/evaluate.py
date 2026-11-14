"""Evaluation-only group access; shared mixtures and class-wise scores stay distinct."""
import numpy as np
from experiments.score_audit_v4.evaluate import summary,paired_ci,expert_stats


def assess(output,uniform,member_p,y,group,ids,repeats=500):
    p=output['probs'];r=summary(p,y,group);u=summary(uniform,y,group)
    r.update(uniform_AA=u['AA'],uniform_WGA=u['WGA'],delta_AA_pp=(r['AA']-u['AA'])*100,delta_WGA_pp=(r['WGA']-u['WGA'])*100,
        **paired_ci(p,uniform,y,group,ids,repeats))
    if output['weights'].shape[1]:
        s=output['scores'] if output['scores'].shape[1]==15 else np.zeros_like(output['weights'])
        r.update(expert_stats(s,output['weights'],member_p,y));r['aggregation_type']='shared expert mixture'
    else:
        r.update(aggregation_type='class-wise score softmax; not shared expert mixture',wrong_head_weight=None,effective_members=None)
        for k in (0,1):r[f'effective_prototypes_class{k}']=float((1/(output['contributions'][...,k]**2).sum(1)).mean())
    pred=p.argmax(1);base=uniform.argmax(1);correct=pred==y;old=base==y
    r['corrected']=int((correct&~old).sum());r['broken']=int((~correct&old).sum());flips=[]
    for kind,values,keys in [('group',group,range(4)),('class',y,range(2))]:
        for k in keys:
            m=values==k;flips.append(dict(kind=kind,value=k,count=int(m.sum()),corrected=int((m&correct&~old).sum()),broken=int((m&~correct&old).sum())))
    for true in (0,1):
        for pr in (0,1):r[f'confusion_true{true}_pred{pr}']=int(((y==true)&(pred==pr)).sum())
    for a in (0,1):
        for b in (0,1):r[f'flip_{a}_to_{b}']=int(((base==a)&(pred==b)&(a!=b)).sum())
    return r,flips


def joint_ci(outputs,uniform,y,group,repeats=2000):
    c=np.stack([p.argmax(1)==y for p in outputs]);u=np.stack([p.argmax(1)==y for p in uniform]);pools=[np.flatnonzero(group==k) for k in range(4)]
    rng=np.random.default_rng(51015);wa=[];aa=[]
    for _ in range(repeats):
        takes=[rng.choice(v,len(v),replace=True) for v in pools];idx=np.concatenate(takes)
        wa.append(float((np.stack([c[:,t].mean(1) for t in takes]).min(0)-np.stack([u[:,t].mean(1) for t in takes]).min(0)).mean()*100))
        aa.append(float((c[:,idx].mean(1)-u[:,idx].mean(1)).mean()*100))
    return dict(WGA_CI95_pp=np.quantile(wa,[.025,.975]).tolist(),AA_CI95_pp=np.quantile(aa,[.025,.975]).tolist(),
        bootstrap='2000 paired, group-stratified unique-image resamples aligned across head seeds; conditional on fixed trained pipelines')
