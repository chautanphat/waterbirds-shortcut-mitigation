"""Six-column weighted ridge, exact-head partial pooling, nested OOF scaling."""
import numpy as np
from experiments.ccw_v6.model import validate,infer as frozen_infer

METHODS=['U0','G1_frozen','G1_R','G1_RH','G1_RHM']
CONFIG=dict(version='v7.0-research-spec',lambda_value=.5,gamma=50.,alpha=.25,ridge=10.,kappa=25.,
    effect_clip=.1,sigma_floor=1e-6,delta_rms_floor=1e-4,column_rms_floor=1e-3,
    scale_floor=.002,scale_clip=2.,distance_z_clip=3.,q_boundary=.2,outer_B_folds=4,scale_folds=3,
    worst_class_NLL_tolerance=.003,AA_delta_U0_strict_min_pp=-1.,bootstrap_repeats=2000,
    bootstrap_batch=100,bootstrap_seed=730001,B_bootstrap_seed=735001,permutation_repeats=3000,permutation_batch=50,
    permutation_seed=740001,lambda1_sensitivity=False,
    fit_integrity_guard='finite coefficients/condition/OOF MSE; both classes; train counts >=3 for scale CV; no added data-driven quality cutoff')


def features(member_probs,member_distances,normalizer):
    p,d=validate(member_probs,member_distances);u=p.mean(1)
    delta=p[...,1].astype(float)-u[:,1,None]
    dm=d.astype(float).min(-1);z=np.clip((dm-np.array(normalizer['mu']))/np.array(normalizer['sigma_safe']),-3,3)
    c=np.broadcast_to(2*u[:,1,None].astype(float)-1,delta.shape);h=abs(c)
    disagree=p.argmax(-1)!=u.argmax(1)[:,None];v=p.argmax(-1)==1
    q=delta/normalizer['delta_rms_safe'];base=np.stack([np.ones_like(z),z,c,h,disagree,v],-1)
    return q[...,None]*base,q,z,delta,u,v.astype(int)


def target(member_probs,y):
    p=np.asarray(member_probs);u=p.mean(1);return (2*np.asarray(y)-1)[:,None]*(p[...,1].astype(float)-u[:,1,None])


def fit_base(member_probs,member_distances,y,sample_id,head_sha256):
    p,d=validate(member_probs,member_distances);y=np.asarray(y,int);ids=np.asarray(sample_id).astype(str)
    if len(set(ids))!=len(p) or len(y)!=len(p) or len(head_sha256)!=15:raise ValueError('Fit IDs / head count')
    if not np.isin(y,[0,1]).all() or np.unique(y).size!=2:raise ValueError('Both training classes required')
    dm=d.astype(float).min(-1);u=p.mean(1);delta=p[...,1].astype(float)-u[:,1,None]
    sigma=dm.std(0,ddof=0);drms=float(np.sqrt(np.mean(delta**2)))
    norm=dict(mu=dm.mean(0).tolist(),sigma=sigma.tolist(),sigma_safe=np.maximum(sigma,1e-6).tolist(),
        ddof=0,delta_rms=drms,delta_rms_safe=max(drms,1e-4))
    X,q,z,delta,u,pred=features(p,d,norm);rms=np.sqrt(np.mean(X**2,axis=(0,1)));scales=np.maximum(rms,1e-3);X=X/scales
    counts=np.bincount(y,minlength=2);margin=abs(u[:,1]-.5);q20=float(np.quantile(margin,.2))
    omega=len(p)/(2*counts[y])*(1+(margin<=q20).astype(int));omega/=omega.mean()
    t=target(p,y);xf=X.reshape(-1,6);wf=np.repeat(omega/15,15);tf=t.ravel()
    gram=xf.T@(wf[:,None]*xf);system=gram+10*np.eye(6);theta=np.linalg.solve(system,xf.T@(wf*tf))
    pooled=X@theta;e=t-pooled;b=np.zeros((15,2));cells=[]
    for j in range(15):
        for k in range(2):
            mask=pred[:,j]==k;n=float(omega[mask].sum());den=float((omega[mask]*q[mask,j]**2).sum());num=float((omega[mask]*q[mask,j]*e[mask,j]).sum())
            raw=num/(den+1e-8);shrunk=float(np.clip(raw,-.1,.1)*n/(n+25)) if n else 0.;b[j,k]=shrunk
            cells.append(dict(head=j,head_sha256=str(head_sha256[j]),predicted_class=k,distinct_images=int(mask.sum()),effective_images=n,
                denominator=den,numerator=num,raw=raw,clipped=float(np.clip(raw,-.1,.1)),shrinkage=n/(n+25),b=shrunk))
    full=pooled+q*b[np.arange(15)[None,:],pred]
    diag=dict(condition_number=float(np.linalg.cond(system)),weighted_MSE_pooled=float(np.sum(wf*(tf-pooled.ravel())**2)/wf.sum()),
        weighted_MSE_head=float(np.sum(wf*(tf-full.ravel())**2)/wf.sum()),weighted_null_MSE=float(np.sum(wf*tf**2)/wf.sum()),
        target_RMS=float(np.sqrt(np.mean(t*t))),max_target_sum_error=float(abs(t.sum(1)).max()),ridge_penalty=10.,
        effective_image_weight_sum=float(omega.sum()),expert_weight_sum=float(wf.sum()),penalty_per_image=10/float(omega.sum()),
        gram_trace=float(np.trace(gram)),feature_zero_delta_max=0.)
    if not np.isfinite(theta).all() or not np.isfinite(diag['condition_number']):raise ValueError('Nonfinite ridge fit')
    return dict(schema_version=1,config=CONFIG,normalizer=norm,column_rms=rms.tolist(),column_scale=scales.tolist(),theta=theta.tolist(),
        head_effects=b.tolist(),cells=cells,fit_ids=ids.tolist(),head_sha256=list(map(str,head_sha256)),q20=q20,tau=max(q20,1e-4),
        margin_quantiles=np.quantile(margin,[.2,.5,.8]).tolist(),class_counts=counts.tolist(),diagnostics=diag,
        feature_order=['q','q*z','q*c','q*abs(c)','q*disagree','q*predclass1'])


def advantage(member_probs,member_distances,fitted_B_model,head_sha256):
    if fitted_B_model['schema_version']!=1 or list(map(str,head_sha256))!=fitted_B_model['head_sha256']:raise ValueError('Fitted head order mismatch')
    X,q,z,delta,u,pred=features(member_probs,member_distances,fitted_B_model['normalizer'])
    pooled=(X/np.array(fitted_B_model['column_scale']))@np.array(fitted_B_model['theta'])
    head=pooled+q*np.array(fitted_B_model['head_effects'])[np.arange(15)[None,:],pred]
    return dict(pooled=pooled,head=head,q=q,z=z,delta=delta,uniform=u)


def center(a):return np.asarray(a,float)-np.asarray(a,float).mean(-1,keepdims=True)


def attach_scales(model,OOF_pooled,OOF_head,member_probs,member_distances,y):
    p,d=validate(member_probs,member_distances);geom=-50*d.min(-1);sG=float(np.sqrt(np.mean(center(geom)**2)))
    t=target(p,y);out={}
    for key,a in [('pooled',OOF_pooled),('head',OOF_head)]:
        sR=float(np.sqrt(np.mean(center(a)**2)));mse=float(np.mean((a-t)**2));reason=None
        if not np.isfinite(a).all() or not np.isfinite(sR) or not np.isfinite(sG) or not np.isfinite(mse):reason='NONFINITE_SCALE_OR_OOF_PREDICTION'
        elif sR<.002:reason='PREDICTED_OOF_SCALE_BELOW_0.002'
        out[key]=dict(s_R=sR if np.isfinite(sR) else None,s_G=sG if np.isfinite(sG) else None,enabled=reason is None,
            off_reason=reason,OOF_MSE=mse if np.isfinite(mse) else None,null_MSE=float(np.mean(t*t)),
            scale_ratio=sG/sR if reason is None else 0.,OOF_images=len(p))
    model['scales']=out;return model


def scaled(a,scale):
    if not scale['enabled'] or scale['s_R'] is None or scale['s_R']<.002 or not np.isfinite(a).all():return np.zeros_like(a,float)
    sR=scale['s_R'];return scale['s_G']/sR*np.clip(center(a),-2*sR,2*sR)


def infer(member_probs,member_distances,fitted_B_model,head_sha256,method='G1_RH',*,lambda_value=.5):
    if method not in METHODS:raise ValueError('Unregistered method')
    p,d=validate(member_probs,member_distances)
    if fitted_B_model is not None and list(map(str,head_sha256))!=fitted_B_model['head_sha256']:raise ValueError('Fitted head order mismatch')
    base=frozen_infer(p,d,None,head_sha256,'U0' if method=='U0' else 'G1_frozen');n=len(p);zero=np.zeros((n,15))
    a=dict(pooled=zero,head=zero,delta=p[...,1].astype(float)-base['uniform'][:,1,None],z=zero)
    gate=np.ones(n);raw=zero;residual=zero;ahat=zero;reason='BASELINE_OR_LAMBDA_ZERO'
    if method not in ['U0','G1_frozen'] and lambda_value!=0:
        if fitted_B_model is None:raise ValueError('B-fitted model required')
        a=advantage(p,d,fitted_B_model,head_sha256);key='pooled' if method=='G1_R' else 'head';ahat=a[key]
        if method=='G1_RHM':gate=np.exp(-abs(base['uniform'][:,1]-.5)/fitted_B_model['tau'])
        raw=scaled(ahat,fitted_B_model['scales'][key]);residual=lambda_value*gate[:,None]*raw
        reason=fitted_B_model['scales'][key]['off_reason'] or 'ENABLED'
    geom=base['geometry_scores'];scores=geom if not np.any(residual) else geom+residual
    if np.any(residual):
        e=np.exp(scores-scores.max(1,keepdims=True));r=e/e.sum(1,keepdims=True);w=.05+.25*r;probs=(p*w[...,None]).sum(1)
    else:r=base['r'];w=base['weights'];probs=base['probs']
    return dict(probs=probs,uniform=base['uniform'],r=r,weights=w,Ahat=ahat,pooled_Ahat=a['pooled'],
        geom_scores=geom,residual_scores=residual,scores=scores,delta=a['delta'],z=a['z'],gate=gate,
        fallback_reason=np.array(reason))
