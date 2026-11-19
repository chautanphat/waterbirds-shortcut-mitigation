"""Class-stratified, source-foreground clustered paired inference statistics."""
import numpy as np
import pandas as pd

from .investigation import MODELS, LEVELS
from .common import KNOWN_WATERBIRDS_LABEL_ERROR_SPECIES


def validate_predictions(frame, foregrounds, *, original=False, severity_count=4):
    expected = len(foregrounds) * (1 if original else 5*severity_count+5) * 3
    if len(frame) != expected or frame.duplicated(['model_name','image_id']).any():
        raise ValueError('Missing or duplicated prediction keys')
    if set(frame.model_name) != set(MODELS):
        raise ValueError('Missing model')
    if set(frame.source_image_id.astype(str)) != set(foregrounds.bird_source_id.astype(str)):
        raise ValueError('Prediction cohort coverage differs')
    reference=foregrounds.set_index('bird_source_id')
    for field,source in [('source_y','y'),('original_group','original_group'),('species_id','species'),
                         ('mask_sha256','mask_sha256'),('source_sha256','source_sha256')]:
        expected_values=frame.source_image_id.map(reference[source])
        if not np.array_equal(frame[field].astype(str).to_numpy(),expected_values.astype(str).to_numpy()):
            raise ValueError(f'Canonical foreground label/provenance mismatch: {field}')
    known=frame.true_y.notna()
    if not (frame.loc[known,'true_y']==frame.loc[known,'source_y']).all():
        raise ValueError('Prediction target disagrees with canonical label')
    if not frame.invariance_passed.all():
        raise ValueError('Compositor invariance failed')
    if not np.isfinite(frame[['p_landbird','p_waterbird','logit_landbird','logit_waterbird']]).all().all():
        raise ValueError('Nonfinite scores')
    if not np.allclose(frame.p_landbird+frame.p_waterbird,1,atol=1e-6,rtol=0):
        raise ValueError('Unnormalized probabilities')
    if ((frame.p_waterbird<0)|(frame.p_waterbird>1)).any():
        raise ValueError('Invalid probabilities')
    if not np.array_equal(frame.pred_y.to_numpy(),(frame.p_waterbird>frame.p_landbird).astype(int).to_numpy()):
        raise ValueError('Predicted class disagrees with saved probabilities')
    by_image=frame.groupby('image_id')
    for field in ['normalized_tensor_sha256','model_input_sha256','raw_input_sha256','true_y','source_y',
                  'species_id','mask_sha256','background_id','severity','sigma']:
        if (by_image[field].nunique(dropna=False)!=1).any():
            raise ValueError(f'Model input or label mismatch: {field}')
    for model in MODELS:
        records=frame[frame.model_name==model]
        if original:
            if not (records.control=='original').all():
                raise ValueError('Original test contains composites')
        else:
            expected_controls={'natural':4*severity_count,'background_only':4,'foreground_only':severity_count,'original':1}
            counts=records.groupby(['source_image_id','control']).size().unstack(fill_value=0)
            if any(not (counts[k]==v).all() for k,v in expected_controls.items()):
                raise ValueError('Incomplete per-bird factorial design')
            natural=records[records.control=='natural']
            if not (natural.groupby(['source_image_id','severity','bg_type']).size()==2).all():
                raise ValueError('Natural group quotas disagree')
            if not (natural.groupby(['source_image_id','severity']).layer_sha256.nunique()==1).all():
                raise ValueError('Foreground changed on background swap')
            if not (natural.groupby('source_image_id').alpha_sha256.nunique()==1).all():
                raise ValueError('Mask changed across severity')
    bg=frame[frame.control=='background_only']
    if len(bg) and (bg.true_y.notna().any() or bg.is_correct.notna().any()):
        raise ValueError('Background-only cannot carry semantic accuracy')
    return dict(rows=len(frame),unique_images=frame.image_id.nunique(),
        unique_raw_images=frame.raw_input_sha256.nunique(),models=list(MODELS),
        exact_model_input_match=True,complete_factorial_design=True)


def metric_tables(frame, *, label_policy='official'):
    data=frame.copy()
    if label_policy=='corrected':
        affected=data.species_id.isin(KNOWN_WATERBIRDS_LABEL_ERROR_SPECIES)
        data.loc[affected & data.true_y.notna(),'true_y']=0
    elif label_policy=='exclude_known_errors':
        data=data[~data.species_id.isin(KNOWN_WATERBIRDS_LABEL_ERROR_SPECIES)]
    elif label_policy!='official':
        raise ValueError('Unsupported label policy')
    group_rows,class_rows,summary=[],[],[]
    for (model,control,severity),rows in data.groupby(['model_name','control','severity'],sort=False):
        known=rows[rows.true_y.notna()].copy()
        known['correct']=known.pred_y==known.true_y
        groups=[]
        if control in ('natural','original'):
            for (y,bg),r in known.groupby(['true_y','bg_type']):
                value=float(r.correct.mean());groups.append(value)
                group_rows.append(dict(model_name=model,control=control,severity=severity,label_policy=label_policy,
                    true_y=int(y),bg_type=bg,n=len(r),correct=int(r.correct.sum()),accuracy=value,
                    source_foregrounds=r.source_image_id.nunique()))
        class_accuracy={}
        for y,r in known.groupby('true_y'):
            acc=float(r.correct.mean());class_accuracy[int(y)]=acc
            class_rows.append(dict(model_name=model,control=control,severity=severity,label_policy=label_policy,
                true_y=int(y),n=len(r),source_foregrounds=r.source_image_id.nunique(),accuracy=acc,
                mean_p_waterbird=float(r.p_waterbird.mean()),predicted_waterbird_fraction=float((r.pred_y==1).mean()),
                mean_waterbird_score_margin=float((r.logit_waterbird-r.logit_landbird).mean()),
                score_type=r.score_type.iloc[0],
                count_pred_landbird=int((r.pred_y==0).sum()),count_pred_waterbird=int((r.pred_y==1).sum())))
        summary.append(dict(model_name=model,control=control,severity=severity,label_policy=label_policy,
            n=len(rows),source_foregrounds=rows.source_image_id.nunique(),
            semantic_accuracy=float(known.correct.mean()) if len(known) else None,
            wga=min(groups) if len(groups)==4 else None,
            landbird_accuracy=class_accuracy.get(0),waterbird_accuracy=class_accuracy.get(1),
            mean_p_waterbird=float(rows.p_waterbird.mean()),predicted_waterbird_fraction=float((rows.pred_y==1).mean()),
            mean_waterbird_score_margin=float((rows.logit_waterbird-rows.logit_landbird).mean()),
            score_type=rows.score_type.iloc[0],
            unique_raw_images=rows.raw_input_sha256.nunique()))
    return pd.DataFrame(summary),pd.DataFrame(group_rows),pd.DataFrame(class_rows)


def paired_flips(frame):
    rows=[]
    natural=frame[frame.control=='natural']
    clean=natural[natural.severity=='clean'].set_index(['model_name','source_image_id','background_id'])
    for r in natural.itertuples():
        c=clean.loc[(r.model_name,r.source_image_id,r.background_id)]
        rows.append(dict(model_name=r.model_name,model_seed=r.model_seed,source_image_id=r.source_image_id,
            species_id=r.species_id,true_y=int(r.true_y),severity=r.severity,
            pair_type='clean_to_degraded',background_id=r.background_id,pair_index=r.pair_index,
            from_pred=int(c.pred_y),to_pred=int(r.pred_y),flip=int(c.pred_y!=r.pred_y),
            land_to_water=int(c.pred_y==0 and r.pred_y==1),water_to_land=int(c.pred_y==1 and r.pred_y==0),
            delta_p_waterbird=float(r.p_waterbird-c.p_waterbird)))
    for (model,bird,severity,pair),records in natural.groupby(['model_name','source_image_id','severity','pair_index']):
        land=records[records.bg_type=='land'].iloc[0];water=records[records.bg_type=='water'].iloc[0]
        rows.append(dict(model_name=model,model_seed=land.model_seed,source_image_id=bird,species_id=land.species_id,
            true_y=int(land.true_y),severity=severity,pair_type='land_background_to_water_background',
            background_id=land.background_id+'__'+water.background_id,pair_index=int(pair),
            from_pred=int(land.pred_y),to_pred=int(water.pred_y),flip=int(land.pred_y!=water.pred_y),
            land_to_water=int(land.pred_y==0 and water.pred_y==1),water_to_land=int(land.pred_y==1 and water.pred_y==0),
            delta_p_waterbird=float(water.p_waterbird-land.p_waterbird)))
    return pd.DataFrame(rows)


def stratified_draws(labels, samples, seed):
    labels=np.asarray(labels)
    if set(labels)!={0,1} or samples<100:
        raise ValueError('Need both classes and at least 100 bootstrap resamples')
    rng=np.random.default_rng(seed)
    indices=[np.flatnonzero(labels==y) for y in (0,1)]
    return [rng.choice(i,size=(samples,len(i)),replace=True) for i in indices]


def bootstrap_comparison(frame, *, samples=2000, seed=0, levels=LEVELS, include_extra=False):
    """Paired across methods/severities; all backgrounds travel with foreground.

    Balanced replication has exactly two backgrounds/category. WGA is recomputed
    inside each draw rather than bootstrapping previously aggregated WGAs.
    """
    natural=frame[frame.control=='natural'].copy()
    ids=sorted(natural.source_image_id.unique())
    meta=natural.drop_duplicates('source_image_id').set_index('source_image_id').loc[ids]
    labels=meta.true_y.to_numpy().astype(int)
    draws=stratified_draws(labels,samples,seed)
    direct=[np.flatnonzero(labels==y)[None,:] for y in (0,1)]
    # Per-bird arrays are the only bootstrap observations.
    arrays={}
    clean_predictions=natural[natural.severity=='clean'].set_index(['model_name','source_image_id','background_id']).pred_y
    for model in MODELS:
        for severity in levels:
            selected=natural[(natural.model_name==model)&(natural.severity==severity)].copy()
            if len(selected)!=len(ids)*4:
                raise ValueError('Missing bird/background prediction')
            selected['correct']=(selected.true_y==selected.pred_y).astype(float)
            selected['pred_water']=(selected.pred_y==1).astype(float)
            selected['margin']=selected.logit_waterbird-selected.logit_landbird
            selected['flip']=[int(r.pred_y!=clean_predictions.loc[(model,r.source_image_id,r.background_id)])
                              for r in selected.itertuples()]
            if include_extra:
                baseline=np.array([clean_predictions.loc[(model,r.source_image_id,r.background_id)] for r in selected.itertuples()])
                selected['land_to_water']=((baseline==0)&(selected.pred_y.to_numpy()==1)).astype(float)
                selected['water_to_land']=((baseline==1)&(selected.pred_y.to_numpy()==0)).astype(float)
            grouped=selected.groupby('source_image_id')
            core=grouped[['correct','p_waterbird','pred_water','margin','flip']].mean().loc[ids].to_numpy()
            bg=selected.groupby(['source_image_id','bg_type']).correct.mean().unstack().loc[ids,['land','water']].to_numpy()
            arrays[(model,severity)]=np.column_stack([core,bg])
            if include_extra:
                direction=grouped[['land_to_water','water_to_land']].mean().loc[ids].to_numpy()
                matched=selected.pivot(index=['source_image_id','pair_index'],columns='bg_type',values=['pred_y','p_waterbird'])
                pair_stats=pd.DataFrame(dict(flip=(matched['pred_y']['land']!=matched['pred_y']['water']).astype(float),
                    absolute=(matched['p_waterbird']['water']-matched['p_waterbird']['land']).abs(),
                    signed=matched['p_waterbird']['water']-matched['p_waterbird']['land']))
                sensitivity=pair_stats.groupby('source_image_id').mean().loc[ids].to_numpy()
                arrays[(model,severity)]=np.column_stack([core,bg,direction,sensitivity])
    def stats(array,indices):
        class_values=[array[ix].mean(axis=1) for ix in indices]
        c0,c1=class_values
        result=dict(accuracy=(c0[:,0]+c1[:,0])/2,
            landbird_accuracy=c0[:,0],waterbird_accuracy=c1[:,0],
            class_accuracy_gap=c1[:,0]-c0[:,0],
            wga=np.minimum(np.minimum(c0[:,5],c0[:,6]),np.minimum(c1[:,5],c1[:,6])),
            p_waterbird_landbird=c0[:,1],p_waterbird_waterbird=c1[:,1],
            predicted_waterbird_given_landbird=c0[:,2],predicted_waterbird_given_waterbird=c1[:,2],
            predicted_waterbird_fraction=(c0[:,2]+c1[:,2])/2,
            mean_waterbird_probability=(c0[:,1]+c1[:,1])/2,
            waterbird_score_margin_landbird=c0[:,3],waterbird_score_margin_waterbird=c1[:,3],
            clean_to_degraded_flip_rate=(c0[:,4]+c1[:,4])/2)
        if include_extra:
            result.update(accuracy_land_background=(c0[:,5]+c1[:,5])/2,
                accuracy_water_background=(c0[:,6]+c1[:,6])/2,
                landbird_on_land_accuracy=c0[:,5],landbird_on_water_accuracy=c0[:,6],
                waterbird_on_land_accuracy=c1[:,5],waterbird_on_water_accuracy=c1[:,6],
                clean_to_degraded_flip_rate_landbird=c0[:,4],clean_to_degraded_flip_rate_waterbird=c1[:,4],
                clean_land_to_water_flip_rate=(c0[:,7]+c1[:,7])/2,
                clean_water_to_land_flip_rate=(c0[:,8]+c1[:,8])/2,
                clean_land_to_water_flip_rate_landbird=c0[:,7],clean_land_to_water_flip_rate_waterbird=c1[:,7],
                clean_water_to_land_flip_rate_landbird=c0[:,8],clean_water_to_land_flip_rate_waterbird=c1[:,8],
                background_swap_flip_rate=(c0[:,9]+c1[:,9])/2,
                background_swap_flip_rate_landbird=c0[:,9],background_swap_flip_rate_waterbird=c1[:,9],
                mean_abs_delta_p_waterbird_background_swap=(c0[:,10]+c1[:,10])/2,
                mean_delta_p_waterbird_background_swap=(c0[:,11]+c1[:,11])/2)
        return result
    boot={k:stats(a,draws) for k,a in arrays.items()}
    point={k:stats(a,direct) for k,a in arrays.items()}
    for model in MODELS:
        for severity in levels:
            for values in (boot,point):
                current,clean=values[(model,severity)],values[(model,'clean')]
                current['wga_drop']=clean['wga']-current['wga']
                current['landbird_accuracy_drop']=clean['landbird_accuracy']-current['landbird_accuracy']
                current['waterbird_accuracy_drop']=clean['waterbird_accuracy']-current['waterbird_accuracy']
                current['class_asymmetric_drop']=current['landbird_accuracy_drop']-current['waterbird_accuracy_drop']
                if include_extra:current['accuracy_drop']=clean['accuracy']-current['accuracy']
    result=[]
    def append(model,comparison,severity,metric,estimate,distribution):
        lo,hi=np.quantile(distribution,[.025,.975])
        result.append(dict(model_name=model,comparison=comparison,severity=severity,metric=metric,
            estimate=float(estimate[0]),ci95_low=float(lo),ci95_high=float(hi),
            bootstrap_samples=samples,bootstrap_seed=seed,source_foregrounds=len(ids),
            unit='source_image_id; class-stratified; all four backgrounds jointly resampled; paired methods',
            seed_level_std=None,seed_status='one image checkpoint per method; cannot estimate training-seed variation'))
    for (model,severity),values in boot.items():
        for metric,distribution in values.items():
            append(model,'within_model',severity,metric,point[(model,severity)][metric],distribution)
    for other in MODELS[1:]:
        for severity in levels:
            for metric in ['wga_drop','landbird_accuracy_drop','waterbird_accuracy_drop','class_asymmetric_drop',
                           'class_accuracy_gap','predicted_waterbird_fraction','mean_waterbird_probability']:
                append('POTER','POTER_minus_'+other,severity,metric,
                    point[('POTER',severity)][metric]-point[(other,severity)][metric],
                    boot[('POTER',severity)][metric]-boot[(other,severity)][metric])
    return pd.DataFrame(result)
