"""Frozen CCW mathematics. Fitter receives B labels; inference never does."""
import hashlib
import numpy as np

METHODS=['U0','G1_frozen','G2_z','CCW_A','CCW_FULL']
CONFIG=dict(schema_version=1,rho=.5,beta=1.,gamma=50.,alpha=.25,kappa=25.,sigma_floor=.02,
            std_ddof=0,z_quantiles=[1/3,2/3],boundary_quantile=.2)


def array_hash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def validate(probs,distances):
    p=np.asarray(probs);d=np.asarray(distances)
    if p.ndim!=3 or p.shape[1:]!=(15,2) or d.shape!=p.shape:raise ValueError('Expected matching [images,15,2] arrays')
    if not np.isfinite(p).all() or not np.isfinite(d).all() or (p<0).any() or (d<0).any():raise ValueError('Invalid probabilities/distances')
    if not np.allclose(p.sum(-1),1,atol=2e-6):raise ValueError('Probability mass')
    return p,d


def softmax(score):
    a=np.asarray(score);e=np.exp(a-a.max(1,keepdims=True));return e/e.sum(1,keepdims=True)


def teacher(probs,y):
    raw=np.asarray(probs);p=raw.astype(np.float64);labels=np.asarray(y,dtype=int)
    if labels.shape!=(len(p),) or not np.isin(labels,[0,1]).all():raise ValueError('Invalid B labels')
    true=p[np.arange(len(p))[:,None],np.arange(15),labels[:,None]]
    # Use the exact frozen raw-probability Uniform reference (including its dtype
    # reduction), then accumulate target statistics in float64.
    uniform=raw.mean(1)
    return true-uniform[np.arange(len(p)),labels,None]


def features(probs,distances,norm,edges):
    p,d=validate(probs,distances);dm=np.asarray(d,dtype=np.float64).min(-1)
    z=(dm-np.asarray(norm['mu']))/np.asarray(norm['sigma_safe'])
    pred=p.argmax(-1);uniform_pred=p.mean(1).argmax(1)
    bins=np.searchsorted(np.asarray(edges),z,side='right')
    disagree=(pred!=uniform_pred[:,None]).astype(int)
    return dm,z,pred,bins,disagree


def fit(probs,distances,y,sample_id,head_sha256):
    p,d=validate(probs,distances);p64=p.astype(np.float64);labels=np.asarray(y,dtype=int);ids=np.asarray(sample_id).astype(str)
    if len(set(ids))!=len(ids) or len(ids)!=len(p) or len(head_sha256)!=15:raise ValueError('Fit IDs/head order')
    counts=np.bincount(labels,minlength=2)
    if len(counts)!=2 or (counts==0).any():raise ValueError('Both B-training classes are required')
    dm=d.astype(np.float64).min(-1);mu=dm.mean(0);sigma=dm.std(0,ddof=0)
    norm=dict(mu=mu.tolist(),sigma=sigma.tolist(),sigma_safe=np.maximum(sigma,1e-6).tolist(),mu_ref=float(np.median(mu)),sigma_ref=float(np.median(sigma)),ddof=0)
    z=(dm-mu)/np.maximum(sigma,1e-6);raw_edges=np.quantile(z,[1/3,2/3])
    edges=raw_edges.tolist() if raw_edges[0]<raw_edges[1] else []
    margin=np.abs(p.mean(1)[:,1]-.5);q20=float(np.quantile(margin,.2))
    omega=len(p)/(2*counts[labels])*(1+(margin<=q20).astype(int));omega=omega/omega.mean()
    weights=np.broadcast_to(omega[:,None]/15,(len(p),15));t=teacher(p,labels)
    global_mean=float((weights*t).sum()/weights.sum());weighted_std=float(np.sqrt((weights*(t-global_mean)**2).sum()/weights.sum()))
    sigma_A=max(weighted_std,.02);_,_,pred,bins,disagree=features(p,d,norm,edges)
    kappa=25.;parents=[]
    for k in (0,1):
        m=pred==k;support=float(weights[m].sum());total=float((weights*t)[m].sum())
        parents.append(float((total+kappa*global_mean)/(support+kappa)))
    table=np.zeros((2,3,2));cells=[]
    for k in (0,1):
        for b in range(3):
            for a in (0,1):
                m=(pred==k)&(bins==b)&(disagree==a);support=float(weights[m].sum());total=float((weights*t)[m].sum())
                value=float(np.clip((total+kappa*parents[k])/(support+kappa),-1,1));table[k,b,a]=value
                cells.append(dict(predicted_class=k,z_bin=b,disagree_with_uniform=a,effective_image_support=support,
                    distinct_images=int(m.any(1).sum()),expert_observations=int(m.sum()),weighted_target_sum=total,
                    target_mean=total/support if support else None,parent=parents[k],A_hat=value,
                    backoff='predicted_class' if not support else 'smoothed_cell',low_support=int(m.any(1).sum())<30))
    return dict(schema_version=1,config=CONFIG,normalizer=norm,z_edges=edges,raw_z_edges=raw_edges.tolist(),
        bin_policy='unweighted pooled quantiles within actual ensemble; one-bin backoff for non-strict edges',
        tie_backoff=not bool(edges),q20=q20,class_counts=counts.tolist(),image_weight_sum=float(omega.sum()),
        sigma_A=sigma_A,weighted_target_std=weighted_std,global_target_mean=global_mean,predicted_class_parents=parents,
        parent_policy='predicted-class weighted target mean shrunk with kappa25 to global weighted mean; global falls back0 only without support; no self-reference',
        table=table.tolist(),cells=cells,fit_ids=ids.tolist(),head_sha256=list(map(str,head_sha256)),
        fit_array_hashes=dict(probs=array_hash(p),distances=array_hash(d),y=array_hash(labels),sample_id=array_hash(ids)))


def ccw_mix(probs,distances,fitted,head_sha256,*,rho=.5,beta=1.,alpha=.25,gamma=50.,z_control=False):
    """Inference-only API. No y/correctness/group or C-specific flags."""
    p,d=validate(probs,distances);u=p.mean(1)
    if fitted is not None:
        if fitted['schema_version']!=1 or list(map(str,head_sha256))!=fitted['head_sha256']:raise ValueError('Fitted schema/head order mismatch')
    n=len(p);dm=d.min(-1);zero=np.zeros((n,15));adv=zero.copy();g=zero.copy();z=zero.copy();dcal=np.asarray(dm,float)
    if alpha==0:
        weights=np.full((n,15),1/15);r=weights.copy();scores=zero.copy();geom=zero.copy();adjustment=zero.copy();dnew=np.asarray(dm,float)
        mixture=u.copy()
    elif rho==0 and beta==0 and not z_control:
        # Preserve original float32 G1 operation order, not just its algebra.
        scores=-gamma*dm;r=softmax(scores);weights=(1-alpha)/15+alpha*r;mixture=(p*weights[...,None]).sum(1)
        geom=scores.copy();adjustment=zero.copy();dnew=dm.copy()
    else:
        if fitted is None:raise ValueError('B-fitted metadata required')
        dm,z,pred,bins,disagree=features(p,d,fitted['normalizer'],fitted['z_edges']);norm=fitted['normalizer']
        dcal=norm['mu_ref']+norm['sigma_ref']*z;dnew=(1-rho)*dm+rho*dcal
        if beta:
            adv=np.asarray(fitted['table'])[pred,bins,disagree];g=np.clip(adv/fitted['sigma_A'],-2,2)
        geom=-z if z_control else -gamma*dnew;adjustment=beta*g;scores=geom+adjustment
        r=softmax(scores);weights=(1-alpha)/15+alpha*r;mixture=(p*weights[...,None]).sum(1)
    if not np.isfinite(weights).all() or (weights<0).any() or not np.allclose(weights.sum(1),1,atol=2e-6):raise ValueError('Invalid simplex')
    if not np.isfinite(mixture).all() or not np.allclose(mixture.sum(1),1,atol=2e-6):raise ValueError('Invalid mixture')
    return dict(probs=mixture,uniform=u,weights=weights,r=r,scores=scores,advhat=adv,g=g,dmin=dm,z=z,dcal=dcal,dnew=dnew,
                geometry_scores=geom,advantage_scores=adjustment)


def infer(probs,distances,fitted,head_sha256,method):
    if method=='U0':return ccw_mix(probs,distances,fitted,head_sha256,alpha=0)
    if method=='G1_frozen':return ccw_mix(probs,distances,fitted,head_sha256,rho=0,beta=0)
    if method=='G2_z':return ccw_mix(probs,distances,fitted,head_sha256,rho=1,beta=0,z_control=True)
    if method=='CCW_A':return ccw_mix(probs,distances,fitted,head_sha256,rho=.5,beta=0)
    if method=='CCW_FULL':return ccw_mix(probs,distances,fitted,head_sha256,rho=.5,beta=1)
    raise ValueError('Variant outside frozen protocol')
