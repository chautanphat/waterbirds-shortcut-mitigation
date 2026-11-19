"""Threshold diagnostics with explicit split guards and foreground clustering."""
import numpy as np
import pandas as pd

METRICS=['accuracy','wga','landbird_accuracy','waterbird_accuracy',
         'landbird_to_waterbird_rate','waterbird_to_landbird_rate','fraction_predicted_waterbird']


def predict(probability,threshold):
    p=np.asarray(probability,dtype=float)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid Waterbird probabilities')
    if not 0<=threshold<=1:raise ValueError('Invalid threshold')
    return (p>=threshold).astype(int)


def group_codes(frame):
    if frame.control.isin(['original','original_validation']).all():return frame.original_group.to_numpy(int)
    if (frame.control=='natural').all():return 2*frame.true_y.to_numpy(int)+(frame.bg_type=='water').to_numpy(int)
    if (frame.control=='foreground_only').all():return 2*frame.true_y.to_numpy(int)
    raise ValueError('Unsupported/mixed control for semantic accuracy')


def metrics(frame,threshold):
    y=frame.true_y.to_numpy(int);p=predict(frame.p_waterbird,threshold);g=group_codes(frame)
    tn,fp,fn,tp=[int(((y==a)&(p==b)).sum()) for a,b in [(0,0),(0,1),(1,0),(1,1)]]
    ratio=lambda a,b:float(a/b) if b else None
    group_acc=[float((p[g==i]==y[g==i]).mean()) if (g==i).any() else None for i in range(4)]
    # Neutral gray has no Land/Water group: no four-group WGA.
    has_groups=not (frame.control=='foreground_only').all() and all(v is not None for v in group_acc)
    result=dict(n=len(y),accuracy=float((p==y).mean()),wga=min(group_acc) if has_groups else None,
        landbird_accuracy=ratio(tn,tn+fp),waterbird_accuracy=ratio(tp,tp+fn),
        landbird_to_waterbird_count=fp,waterbird_to_landbird_count=fn,
        landbird_to_waterbird_rate=ratio(fp,tn+fp),waterbird_to_landbird_rate=ratio(fn,tp+fn),
        false_positive_rate=ratio(fp,tn+fp),false_negative_rate=ratio(fn,tp+fn),
        tn=tn,fp=fp,fn=fn,tp=tp,fraction_predicted_waterbird=float(p.mean()),
        cfps_predicted_landbird=ratio(fn,fp+fn),cfps_predicted_waterbird=ratio(fp,fp+fn),
        mean_p_waterbird=float(frame.p_waterbird.mean()),median_p_waterbird=float(frame.p_waterbird.median()))
    for i,value in enumerate(group_acc):result[f'group{i}_accuracy']=value if not (frame.control=='foreground_only').all() else None
    return result


def select_clean_threshold(validation,grid):
    """Never accept test rows, blur or fitting/reference-pool development rows."""
    if set(validation.source_split)!={'val'}:raise ValueError('Threshold selection requires validation only')
    if not validation.selection_eligible.all():raise ValueError('Development rows overlap model fitting/reference pool')
    if set(validation.control)!={'original_validation'} or set(validation.sigma)!={0.}:
        raise ValueError('Clean original validation only')
    if set(group_codes(validation))!={0,1,2,3}:raise ValueError('All four independent development groups required')
    records=[dict(threshold=float(t),**metrics(validation,float(t))) for t in grid]
    # Exact integer confusion counts generate metrics; rounding only resolves float tie noise.
    winner=min(records,key=lambda r:(-round(r['wga'],14),-round(r['accuracy'],14),round(abs(r['threshold']-.5),14),r['threshold']))
    return winner,records


def bootstrap_delta(frame,threshold,*,samples=2000,seed=20261119):
    """Paired fixed-vs-locked threshold changes; resample source foregrounds by class.

    Threshold stays locked in every resample. CIs on selection data are conditional
    descriptive intervals, not selection-adjusted or out-of-sample inference.
    """
    if samples<2000:raise ValueError('At least2000 bootstrap draws required')
    identities=frame[['source_image_id','true_y']].drop_duplicates().sort_values('source_image_id')
    if identities.source_image_id.duplicated().any():raise ValueError('Conflicting class labels for foreground')
    ids=identities.source_image_id.to_numpy();labels=identities.true_y.to_numpy(int)
    position={v:i for i,v in enumerate(ids)};index=np.array([position[v] for v in frame.source_image_id])
    y=frame.true_y.to_numpy(int);g=group_codes(frame);den=np.zeros((len(ids),4))
    baseline=np.zeros_like(den);selected=np.zeros_like(den)
    np.add.at(den,(index,g),1)
    np.add.at(baseline,(index,g),(predict(frame.p_waterbird,.5)==y).astype(float))
    np.add.at(selected,(index,g),(predict(frame.p_waterbird,threshold)==y).astype(float))
    groups=not (frame.control=='foreground_only').all() and set(g)=={0,1,2,3}
    def reduce(c,d):
        total=d.sum(1);c0=c[:,:2].sum(1);c1=c[:,2:].sum(1)
        n0=d[:,:2].sum(1);n1=d[:,2:].sum(1)
        r=np.divide(c,d,out=np.full_like(c,np.nan),where=d!=0)
        wga=np.nanmin(r,axis=1) if groups else np.full(len(c),np.nan)
        # Require every group in a resample; never silently reduce to fewer groups.
        if groups:wga[(d==0).any(1)]=np.nan
        return np.column_stack([c.sum(1)/total,wga,c0/n0,c1/n1,
                                1-c0/n0,1-c1/n1,(n0-c0+c1)/total])
    rng=np.random.default_rng(seed);distribution=[]
    # Chunk draws so full5794-image resampling stays memory bounded.
    for start in range(0,samples,100):
        size=min(100,samples-start)
        draws=np.concatenate([rng.choice(np.flatnonzero(labels==a),size=(size,int((labels==a).sum())),replace=True) for a in [0,1]],axis=1)
        d=den[draws].sum(1)
        distribution.append(reduce(selected[draws].sum(1),d)-reduce(baseline[draws].sum(1),d))
    distribution=np.concatenate(distribution)
    before,after=metrics(frame,.5),metrics(frame,threshold);output=[]
    for j,metric in enumerate(METRICS):
        valid=distribution[:,j][np.isfinite(distribution[:,j])]
        if metric=='wga' and not groups:continue
        if len(valid):
            lo,hi=np.quantile(valid,[.025,.975]);estimate=after[metric]-before[metric]
            output.append(dict(metric=metric,baseline=before[metric],selected=after[metric],
                delta=estimate,ci95_low=float(lo),ci95_high=float(hi),bootstrap_samples=samples,
                valid_bootstrap_samples=len(valid),bootstrap_seed=seed,n_foregrounds=len(ids),
                resampling_unit='source_image_id; class-stratified; all background repeats within foreground',
                training_seed_ci=False,threshold_reselected_in_bootstrap=False))
    return output


def scope_metrics(frame,threshold):
    """Aggregate, class and class×background rows; denominators remain explicit."""
    output=[dict(scope='all',true_class=None,true_group=None,background='all',**metrics(frame,threshold))]
    for y,part in frame.groupby('true_y',sort=True):
        output.append(dict(scope='true_class',true_class=int(y),true_group=None,background='all',**metrics(part,threshold)))
    if not (frame.control=='foreground_only').all():
        data=frame.assign(_g=group_codes(frame))
        for g,part in data.groupby('_g',sort=True):
            output.append(dict(scope='class_background',true_class=int(g)//2,true_group=int(g),
                background=['land','water'][int(g)%2],**metrics(part,threshold)))
    return output
