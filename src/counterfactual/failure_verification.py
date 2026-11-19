"""Paired input intervention statistics, without training or model inference."""
from collections import defaultdict
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from torchvision import transforms

from .common import ROOT,pixel_hash,write_csv,sha256
from .data import ForegroundRepository
from .investigation import MODELS,intervention_records
from .failure_statistics import stratified_draws
from src.datasets.waterbirds import waterbirds_transform

LEVELS=['clean','mild','moderate','strong']
SIGMAS=dict(zip(LEVELS,[0.,.5,1.,2.]))
SCOPES=['all','landbird','waterbird']
CONTRASTS=['gray_to_land','gray_to_water','land_to_water']
NUMERIC=['accuracy_gray','accuracy_land','accuracy_water','accuracy_natural',
    'p_gray','p_land','p_water','delta_background','interference','amplification','natural_accuracy_drop']
for contrast in CONTRASTS:
    NUMERIC += [contrast+'_'+metric for metric in ['flip_rate','signed_probability_shift',
        'absolute_probability_shift','margin_shift','accuracy_change']]


def audit_images(frame,cohort,config,data_root,out):
    """Re-render primary conditions and compare actual unrounded tensor bytes."""
    cfg=dict(config,severities={l:SIGMAS[l] for l in LEVELS})
    source=frame[frame.model_name=='POTER'].set_index('image_id')
    t=waterbirds_transform().transforms;preview,tensor=transforms.Compose(t[:2]),transforms.Compose(t[2:])
    records=[]
    for images,rows in intervention_records(cohort,cfg,out,save_examples=False):
        for image,r in zip(images,rows):
            previous=source.loc[r['image_id']]
            crop=preview(image)
            tensor_hash=hashlib.sha256(tensor(crop).numpy().tobytes()).hexdigest()
            if (pixel_hash(image)!=previous.raw_input_sha256 or pixel_hash(crop)!=previous.model_input_sha256
                    or tensor_hash!=previous.normalized_tensor_sha256 or not r['invariance_passed']):
                raise ValueError('Reconstructed intervention differs from saved inference: '+r['image_id'])
            records.append(dict(image_id=r['image_id'],source_image_id=r['source_image_id'],sigma=r['sigma'],
                control=r['control'],raw_input_sha256=previous.raw_input_sha256,
                normalized_tensor_sha256=tensor_hash,invariance_passed=True))
    if len(records)!=2500:raise ValueError('Expected 2,500 primary/control image records')
    write_csv(out/'image_integrity_audit.csv',records)
    return dict(rendered_records_verified=len(records),raw_and_model_tensor_hashes_match=True,
        model_inference_executed=False,mask_and_opaque_foreground_invariance=True,
        unchanged_background_outside_foreground=True)


def paired_interventions(frame):
    data=frame[frame.control.isin(['natural','foreground_only']) & frame.severity.isin(LEVELS)]
    rows=[];foregrounds=[]
    for (model,seed,bird,level),group in data.groupby(['model_name','model_seed','source_image_id','severity'],sort=False):
        gray=group[group.control=='foreground_only']
        natural=group[group.control=='natural']
        if len(gray)!=1 or len(natural)!=4:raise ValueError('Missing gray/natural paired predictions')
        gray=gray.iloc[0];land=natural[natural.bg_type=='land'].sort_values('pair_index')
        water=natural[natural.bg_type=='water'].sort_values('pair_index')
        if len(land)!=2 or len(water)!=2:raise ValueError('Expected both background replicates')
        if group.layer_sha256.nunique()!=1 or group.alpha_sha256.nunique()!=1 or group.mask_sha256.nunique()!=1:
            raise ValueError('Background intervention also changed foreground/mask')
        base=dict(model_name=model,model_seed=int(seed),source_image_id=int(bird),foreground_id=gray.foreground_id,
            species_id=gray.species_id,true_y=int(gray.true_y),severity=level,sigma=float(gray.sigma),
            mask_sha256=gray.mask_sha256,layer_sha256=gray.layer_sha256,alpha_sha256=gray.alpha_sha256,
            score_type=gray.score_type)
        pairs={}
        for contrast,a,b in [('gray_to_land',[gray,gray],list(land.itertuples())),
            ('gray_to_water',[gray,gray],list(water.itertuples())),
            ('land_to_water',list(land.itertuples()),list(water.itertuples()))]:
            pairs[contrast]=[]
            for replicate,(source,target) in enumerate(zip(a,b)):
                shift=float(target.p_waterbird-source.p_waterbird)
                margin_from=float(source.logit_waterbird-source.logit_landbird)
                margin_to=float(target.logit_waterbird-target.logit_landbird)
                before,after=int(source.pred_y),int(target.pred_y)
                correct_from,correct_to=int(before==base['true_y']),int(after==base['true_y'])
                pair=dict(base,contrast=contrast,replicate=replicate,
                    from_image_id=source.image_id,to_image_id=target.image_id,
                    from_background_id=source.background_id,to_background_id=target.background_id,
                    p_waterbird_from=float(source.p_waterbird),p_waterbird_to=float(target.p_waterbird),
                    signed_probability_shift=shift,absolute_probability_shift=abs(shift),
                    margin_from=margin_from,margin_to=margin_to,margin_shift=margin_to-margin_from,
                    pred_from=before,pred_to=after,flip=int(before!=after),
                    land_to_water_prediction_flip=int(before==0 and after==1),
                    water_to_land_prediction_flip=int(before==1 and after==0),
                    correct_from=correct_from,correct_to=correct_to,accuracy_change=correct_to-correct_from,
                    raw_hash_from=source.raw_input_sha256,raw_hash_to=target.raw_input_sha256,
                    tensor_hash_from=source.normalized_tensor_sha256,tensor_hash_to=target.normalized_tensor_sha256)
                pairs[contrast].append(pair);rows.append(pair)
        f=dict(base,accuracy_gray=float(gray.pred_y==gray.true_y),
            accuracy_land=float((land.pred_y==land.true_y).mean()),
            accuracy_water=float((water.pred_y==water.true_y).mean()),
            accuracy_natural=float((natural.pred_y==natural.true_y).mean()),
            p_gray=float(gray.p_waterbird),p_land=float(land.p_waterbird.mean()),p_water=float(water.p_waterbird.mean()),
            foreground_count=1)
        f['delta_background']=f['p_water']-f['p_land']
        f['interference']=f['accuracy_natural']-f['accuracy_gray']
        for contrast,paired in pairs.items():
            for destination,field in [('flip_rate','flip'),('signed_probability_shift','signed_probability_shift'),
                ('absolute_probability_shift','absolute_probability_shift'),('margin_shift','margin_shift'),('accuracy_change','accuracy_change')]:
                f[contrast+'_'+destination]=float(np.mean([r[field] for r in paired]))
        foregrounds.append(f)
    f=pd.DataFrame(foregrounds)
    reference=f[f.severity=='clean'].set_index(['model_name','model_seed','source_image_id'])
    f['amplification']=[r.interference-reference.loc[(r.model_name,r.model_seed,r.source_image_id)].interference for r in f.itertuples()]
    f['natural_accuracy_drop']=[reference.loc[(r.model_name,r.model_seed,r.source_image_id)].accuracy_natural-r.accuracy_natural for r in f.itertuples()]
    if len(rows)!=7200 or len(f)!=1200:raise ValueError('Unexpected paired intervention coverage')
    return pd.DataFrame(rows),f


def summarize_bootstrap(foregrounds,*,samples,seed,unit='source_image_id'):
    """Paired class-stratified resampling; background replicates averaged first."""
    meta=foregrounds.drop_duplicates(unit).sort_values(unit)
    ids=meta[unit].to_numpy();labels=meta.true_y.to_numpy().astype(int)
    draws=stratified_draws(labels,samples,seed)
    point_draws=[np.flatnonzero(labels==y)[None,:] for y in (0,1)]
    boot={};points={};metadata={}
    for (model,model_seed,sigma),group in foregrounds.groupby(['model_name','model_seed','sigma'],sort=False):
        ordered=group.set_index(unit).loc[ids]
        a=ordered[NUMERIC].to_numpy()
        values=[]
        for indices in (draws,point_draws):
            class_means=[a[i].mean(axis=1) for i in indices]
            stats={}
            for scope,means in [('all',(class_means[0]+class_means[1])/2),('landbird',class_means[0]),('waterbird',class_means[1])]:
                stats[scope]={name:means[:,j] for j,name in enumerate(NUMERIC)}
            c0,c1=stats['landbird'],stats['waterbird']
            stats['all'].update(wga=np.minimum.reduce([c0['accuracy_land'],c0['accuracy_water'],c1['accuracy_land'],c1['accuracy_water']]),
                class_asymmetry=c0['natural_accuracy_drop']-c1['natural_accuracy_drop'],
                raw_class_gap=c1['accuracy_natural']-c0['accuracy_natural'],
                land_minus_water_interference=c0['interference']-c1['interference'],
                land_minus_water_amplification=c0['amplification']-c1['amplification'])
            values.append(stats)
        boot[(model,float(sigma))],points[(model,float(sigma))]=values
        metadata[model]=dict(model_seed=int(model_seed),score_type=ordered.score_type.iloc[0])
    for model in MODELS:
        for sigma in SIGMAS.values():
            for values in (boot,points):
                values[(model,sigma)]['all']['wga_drop']=values[(model,0.)]['all']['wga']-values[(model,sigma)]['all']['wga']
    output=[]
    def append(model,sigma,scope,metric,estimate,distribution,comparison='within_seed'):
        mask=np.ones(len(meta),dtype=bool) if scope=='all' else labels==(0 if scope=='landbird' else 1)
        lo,hi=np.quantile(distribution,[.025,.975])
        output.append(dict(model_name=model,model_seed=metadata[model]['model_seed'],sigma=sigma,scope=scope,
            comparison=comparison,metric=metric,estimate=float(estimate[0]),ci95_low=float(lo),ci95_high=float(hi),
            bootstrap_unit=unit+'; class-stratified; paired models/severities; all background replicates travel with cluster',
            bootstrap_samples=samples,bootstrap_seed=seed,n_clusters=int(mask.sum()),
            foreground_count=int(meta.loc[mask,'foreground_count'].sum()),species_count=int(meta.loc[mask,'species_id'].nunique()),
            training_seed_n=1,training_seed_std=None,seed_status='incomplete_independent_seed_validation',
            score_type=metadata[model]['score_type'] if metric.endswith('margin_shift') else 'probability_or_accuracy_scale'))
    for (model,sigma),scopes in boot.items():
        for scope,metrics in scopes.items():
            for metric,distribution in metrics.items():append(model,sigma,scope,metric,points[(model,sigma)][scope][metric],distribution)
    for baseline in MODELS[1:]:
        for sigma in SIGMAS.values():
            for scope in SCOPES:
                metrics=['interference','amplification','delta_background','gray_to_land_flip_rate','gray_to_water_flip_rate','land_to_water_flip_rate']
                if scope=='all':metrics+=['wga_drop','class_asymmetry']
                for metric in metrics:
                    append('POTER',sigma,scope,metric,
                        points[('POTER',sigma)][scope][metric]-points[(baseline,sigma)][scope][metric],
                        boot[('POTER',sigma)][scope][metric]-boot[(baseline,sigma)][scope][metric],
                        comparison='POTER_minus_'+baseline)
    return pd.DataFrame(output)


def species_summaries(foregrounds):
    columns=['model_name','model_seed','severity','sigma','true_y','species_id','score_type']
    result=foregrounds.groupby(columns,sort=False)[NUMERIC].mean().reset_index()
    count=foregrounds.groupby(columns,sort=False).source_image_id.nunique().reset_index(name='foreground_count')
    return result.merge(count,on=columns,validate='one_to_one')
