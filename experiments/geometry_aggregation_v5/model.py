"""Label-free aggregation of frozen DPE probabilities, logits and distances."""
import numpy as np
from scipy.special import logsumexp,softmax


def configs():
    rows=[dict(id='U0',method='U0',alpha=0.)]
    for method,values in [('E1',[1.,5.]),('G1',[5.,20.,50.]),('G2',[.5,1.,2.])]:
        for g in values:rows.append(dict(id=f'{method}_g{g:g}',method=method,gamma=g,alpha=.25))
    for m in [3,5,10]:rows.append(dict(id=f'G3_m{m}',method='G3',M=m,alpha=.25))
    rows.append(dict(id='G3_m1_DIAGNOSTIC',method='G3',M=1,alpha=1.,diagnostic_only=True))
    for method in ['S1','S2','S3']:
        for tau in [.025,.1,.5]:rows.append(dict(id=f'{method}_t{tau:g}',method=method,tau=tau,T_class=1.,alpha=1.))
    for a in [.10,.25,.50]:rows.append(dict(id=f'S4_a{a:g}',method='S4',tau=.1,T_class=1.,alpha=a))
    return rows


def validate(probs,logits,distances):
    p=np.asarray(probs);l=np.asarray(logits);d=np.asarray(distances)
    assert p.shape==l.shape==d.shape and p.ndim==3 and p.shape[1:]==(15,2)
    assert all(np.isfinite(a).all() for a in (p,l,d)) and (p>=0).all() and (d>=0).all()
    assert np.allclose(p.sum(-1),1,atol=2e-6)


def fit_normalizer(distances,fold,sample_id,head_hashes):
    """Each fold has actual distinct heads: never pool a slot across folds."""
    dmin=np.asarray(distances).min(-1);out={}
    for f in np.unique(fold):
        mask=fold==f
        if mask.sum()<30:raise ValueError('Insufficient training-only observations for actual head ensemble')
        out[str(int(f))]=dict(mean=dmin[mask].mean(0).tolist(),std=np.maximum(dmin[mask].std(0),1e-6).tolist(),
            count=int(mask.sum()),fit_ids=np.asarray(sample_id)[mask].astype(str).tolist(),head_hashes=list(head_hashes[str(int(f))]))
    return dict(ensembles=out,features='minimum normalized Euclidean distance, z-score per actual fold/head',group_blind=True)


def normalized(distances,fold,normalizer,head_hashes):
    z=np.empty(distances.shape[:2],float);dmin=distances.min(-1)
    for f in np.unique(fold):
        key=str(int(f));entry=normalizer['ensembles'][key]
        if list(head_hashes[key])!=entry['head_hashes']:raise ValueError('Normalizer/head identity mismatch')
        mask=fold==f;z[mask]=(dmin[mask]-entry['mean'])/entry['std']
    return z


def aggregate(probs,logits,distances,config,fold=None,normalizer=None,head_hashes=None):
    """Inference API. Labels, correctness and subgroup annotations are forbidden."""
    validate(probs,logits,distances);p=np.asarray(probs);d=np.asarray(distances);l=np.asarray(logits)
    method=config['method'];n=len(p);uniform=p.mean(1);contrib=np.empty((n,0,2));scores=np.empty((n,0));alpha=config['alpha']
    if method=='U0':return dict(probs=uniform,weights=np.full((n,15),1/15),contributions=contrib,scores=scores)
    if method in ['S1','S2','S3','S4']:
        v=l if method=='S2' else -(d-d.mean(-1,keepdims=True)) if method=='S3' else -d
        tau=config['tau'];class_scores=tau*(logsumexp(v/tau,axis=1)-np.log(15))
        # Class-specific derivative/contribution weights, NOT shared mixture weights.
        contrib=softmax(v/tau,axis=1);mix=softmax(class_scores/config['T_class'],axis=-1)
        if method=='S4':mix=(1-alpha)*uniform+alpha*mix
        w=np.empty((n,0));scores=class_scores
    else:
        if method=='E1':scores=config['gamma']*(p*np.log(p.clip(1e-15))).sum(-1)
        elif method=='G1':scores=-config['gamma']*d.min(-1)
        elif method=='G2':scores=-config['gamma']*normalized(d,fold,normalizer,head_hashes)
        elif method=='G3':scores=-d.min(-1)
        else:raise ValueError(method)
        if method=='G3':
            order=np.argsort(d.min(-1),axis=1,kind='stable')[:,:config['M']];r=np.zeros((n,15));np.put_along_axis(r,order,1/config['M'],axis=1)
        else:r=softmax(scores,axis=1)
        w=(1-alpha)/15+alpha*r;mix=(p*w[...,None]).sum(1)
        assert (w>=0).all() and np.allclose(w.sum(1),1)
    assert np.isfinite(mix).all() and np.allclose(mix.sum(-1),1,atol=2e-6)
    return dict(probs=mix,weights=w,contributions=contrib,scores=scores)
