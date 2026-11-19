"""Validation-guarded threshold diagnostic. Inference only; never trains."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
import torch

from src.counterfactual.common import read_csv,write_csv,write_json,sha256,pixel_hash
from src.counterfactual.investigation import FrozenComparison,MODELS
from src.counterfactual.threshold_probe import metrics,scope_metrics,select_clean_threshold,bootstrap_delta
from src.evaluation.dpe_export import predict_heads


def settings():
    cfg=json.loads((ROOT/'configs/poter_threshold_probe.json').read_text())
    if any(cfg[k] for k in ['allow_training','allow_ot_recomputation','allow_new_corruptions','allow_test_selection']):
        raise ValueError('This entrypoint never trains/recomputes OT/generates corruptions/selects on test')
    return cfg


def audit(config,out):
    source=ROOT/'outputs/mild_blur_v2'
    for name,digest in json.loads((source/'integrity.json').read_text())['sha256'].items():
        if sha256(source/name)!=digest:raise ValueError('Saved experiment changed: '+name)
    old=ROOT/'results/poter_failure'
    for name in ['original_test_predictions.csv','checkpoint_audit.json']:
        if sha256(old/name)!=json.loads((old/'integrity.json').read_text())['sha256'][name]:
            raise ValueError('Original test cache changed')
    metadata=pd.read_csv(Path(config['validation_data_root'])/'metadata.csv')
    if sha256(Path(config['validation_data_root'])/'metadata.csv')!=sha256(ROOT/'data/waterbirds/metadata.csv'):
        raise ValueError('Noncanonical labels/groups')
    validation=metadata[metadata.split==1].copy()
    validation['group']=2*validation.y+validation.place
    counts=validation.group.value_counts().sort_index().to_dict()
    if counts!={0:467,1:466,2:133,3:133}:raise ValueError('Unexpected validation groups')
    # Historical runner requires all466 group1 +all133 group2 validation rows as OT reference.
    # This determines exact membership even though the historical feature-cache bytes are missing.
    reference_ids=set(validation.loc[validation.group.isin([1,2]),'img_filename'])
    info=json.loads((source/'checkpoint_audit.json').read_text())
    eligibility={
        'POTER':dict(eligible=False,status='BLOCKED: no independent four-group development data for threshold selection',
            reason='599 canonical validation rows used to construct OT training reference; remaining600 cover only groups0/3. All1199 also monitored for checkpoint selection.',
            fitting_or_reference_overlap=599,remaining_independent_groups=[0,3]),
        'ERM_enhanced':dict(eligible=True,status='clean-validation diagnostic allowed',
            reason='Backbone fitting uses canonical train4795; fixed last400 checkpoint. Validation1199 has four groups and no gradient-fitting overlap; historical validation monitoring disclosed.',
            fitting_or_reference_overlap=0,remaining_independent_groups=[0,1,2,3]),
        'DPE_legacy':dict(eligible=False,status='BLOCKED: no independent development data for threshold selection',
            reason='Prototype heads were fitted on all1199 canonical validation images; no held-out calibration rows remain.',
            fitting_or_reference_overlap=1199,remaining_independent_groups=[])}
    for model in ['POTER','ERM_enhanced']:
        if sha256(info[model]['checkpoint'])!=info[model]['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    if sha256(info['DPE_legacy']['heads_path'])!=info['DPE_legacy']['heads_sha256']:raise ValueError('Heads changed')
    record=dict(created_utc=datetime.now(timezone.utc).isoformat(),labels={'0':'landbird','1':'waterbird'},
        group_definition='2*true_y+place; place0=land,1=water',validation_count=len(validation),
        validation_group_counts=counts,reference_ids=sorted(reference_ids),eligibility=eligibility,
        no_untouched_evaluation_available=True,independence_policy=config['independence_policy'],
        source_files={str(p):sha256(p) for p in [ROOT/config['source_predictions'],ROOT/config['original_test_predictions'],
            Path(config['cached_validation_features']),Path(config['cached_validation_features_metadata'])]},
        checkpoint_info=info,requested_csv='predictions(5).csv',requested_csv_attached=False,
        canonical_substitution='outputs/mild_blur_v2/predictions.csv;9000 audited real prediction rows')
    write_json(out/'asset_audit.json',record)
    write_csv(out/'validation_membership.csv',[dict(source_image_id=int(r.img_id),filename=r.img_filename,
        true_y=int(r.y),group=int(r.group),poter_ot_reference=r.img_filename in reference_ids,
        dpe_head_fitting=True,erm_threshold_eligible=True,
        poter_threshold_eligible=False,dpe_threshold_eligible=False) for r in validation.itertuples()])
    return record,validation


def infer_validation(config,out,asset,validation):
    destination=out/'validation_predictions.csv';manifest=out/'validation_inference.json'
    if destination.exists():
        saved=json.loads(manifest.read_text())
        if sha256(destination)!=saved['prediction_sha256']:raise ValueError('Validation prediction cache changed')
        if saved['source_files']!=asset['source_files']:raise ValueError('Cached validation dependencies changed')
        print('Reusing3597 completed validation probability rows; no new inference.');return
    started=time.perf_counter();torch.set_num_threads(4)
    # Match the saved mild-blur comparison's FP32 numeric runtime.
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=json.loads((ROOT/config['source_config']).read_text());cfg.update(device=config['device'],batch_size=config['batch_size'])
    frozen=FrozenComparison(cfg,out)
    # Confirm actual head-fitting IDs, not just historical prose.
    pool=set(frozen.heads['provenance']['training_pool_ids'])
    if pool!=set(validation.img_filename):raise ValueError('DPE head fitting pool audit disagrees')
    sidecar=json.loads(Path(config['cached_validation_features_metadata']).read_text())
    if sidecar['backbone_sha256']!=cfg['expected_erm_sha256'] or sidecar['metadata_sha256']!=sha256(ROOT/'data/waterbirds/metadata.csv'):
        raise ValueError('Feature cache identity differs')
    if sidecar['preprocessing']!='square256;center224;ImageNet;raw2048' or sidecar['split']!='val':
        raise ValueError('Cached feature preprocessing/split differs')
    with np.load(config['cached_validation_features'],allow_pickle=False) as archive:
        ids=archive['sample_id'].astype(str);raw=archive['embedding'].copy()
        if len(set(ids))!=1199 or set(ids)!=set(validation.img_filename):raise ValueError('Cache IDs are not canonical validation')
        order={name:i for i,name in enumerate(ids)}
        index=[order[name] for name in validation.img_filename]
        if not np.array_equal(archive['y'][index],validation.y) or not np.array_equal(archive['group'][index],validation.group):
            raise ValueError('Cached labels/group mapping differs')
        raw=raw[index]
    if raw.shape!=(1199,2048) or not np.isfinite(raw).all():raise ValueError('Bad cached feature shape')
    data=Path(config['validation_data_root'])
    # Eight predeclared live checks (two per group), independent of correctness/confidence.
    checks=[i for g in range(4) for i in np.flatnonzero(validation.group.to_numpy()==g)[:2]]
    images=[]
    for i in checks:
        with Image.open(data/validation.iloc[i].img_filename) as image:images.append(ImageOps.exif_transpose(image).convert('RGB'))
    tensors=torch.stack([frozen.erm.tensor(frozen.erm.preview(im)) for im in images]).to(config['device'])
    with torch.inference_mode():actual=frozen.erm.model.extract_features(tensors).cpu().numpy()
    feature_error=float(abs(actual-raw[checks]).max())
    if feature_error>1e-3:raise ValueError(f'Cached ERM features fail live source/preprocess parity: {feature_error}')
    with torch.inference_mode():
        logits=torch.cat([frozen.erm.model[-1](torch.as_tensor(x,device=config['device'])) for x in np.array_split(raw,75)]).cpu()
        erm_probability=logits.softmax(1).numpy();erm_logits=logits.numpy()
    dpe_probability=predict_heads(frozen.converted,raw,device=config['device'])
    live=frozen.erm.predict_batch(images,config['batch_size'])
    probability_error=float(abs(erm_probability[checks]-np.array([r['probabilities'] for r in live])).max())
    legacy=frozen.heads
    from src.methods.dpe_head_reweighting import ensemble
    reference=ensemble.predict(legacy,raw[checks],device=config['device']).mean(0)
    dpe_error=float(abs(reference-dpe_probability[checks]).max())
    if probability_error>1e-4 or dpe_error>1e-4:raise ValueError('Cached head inference fails adapter/reference parity')
    predictions=[];batch=config['batch_size']
    for start in range(0,len(validation),batch):
        rows=validation.iloc[start:start+batch];images=[]
        for r in rows.itertuples():
            with Image.open(data/r.img_filename) as image:images.append(ImageOps.exif_transpose(image).convert('RGB'))
        poter=frozen.poter.predict_batch(images,batch)
        for offset,(r,image,pot) in enumerate(zip(rows.itertuples(),images,poter)):
            index=start+offset;preview=frozen.erm.preview(image);tensor=frozen.erm.tensor(preview)
            tensor_hash=hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
            if tensor_hash!=pot['normalized_tensor_sha256']:raise ValueError('CNN preprocess mismatch')
            for name,p,z,saved_pred in [
                ('POTER',pot['probabilities'],pot['logits'],pot['pred_y']),
                ('ERM_enhanced',erm_probability[index],erm_logits[index],int(erm_probability[index].argmax())),
                ('DPE_legacy',dpe_probability[index],np.log(dpe_probability[index].clip(np.finfo(np.float32).tiny)),int(dpe_probability[index].argmax()))]:
                info=asset['checkpoint_info'][name]
                predictions.append(dict(dataset='canonical_validation1199',source_split='val',model_name=name,
                    model_seed=1 if name=='POTER' else 0,source_image_id=int(r.img_id),foreground_id=f'canonical-{r.img_id}',
                    species_id=r.img_filename.split('/')[0],filename=r.img_filename,true_y=int(r.y),original_group=int(r.group),
                    bg_type=['land','water'][int(r.place)],background_id=f'original-{r.img_id}',sigma=0.,severity='original',
                    control='original_validation',pred_y=saved_pred,p_landbird=float(p[0]),p_waterbird=float(p[1]),
                    logit_landbird=float(z[0]),logit_waterbird=float(z[1]),selection_eligible=asset['eligibility'][name]['eligible'],
                    checkpoint_sha256=info.get('checkpoint_sha256',info.get('backbone_sha256')),
                    heads_sha256=info.get('heads_sha256'),raw_input_sha256=pixel_hash(image),
                    model_input_sha256=pixel_hash(preview),normalized_tensor_sha256=tensor_hash,
                    source_file_sha256=sha256(data/r.img_filename),untouched_evaluation=False))
        if start%160==0 or start+batch>=len(validation):
            print(json.dumps(dict(phase='validation inference only',images=min(start+batch,len(validation)),total=1199,
                elapsed_seconds=time.perf_counter()-started)),flush=True)
    frame=pd.DataFrame(predictions)
    if len(frame)!=3597 or frame.duplicated(['model_name','source_image_id']).any():raise ValueError('Incomplete/duplicate validation predictions')
    temporary=destination.with_suffix('.partial.csv');frame.to_csv(temporary,index=False);temporary.replace(destination)
    write_json(manifest,dict(prediction_sha256=sha256(destination),source_files=asset['source_files'],
        validation_rows=3597,poter_new_CNN_images=1199,erm_live_check_images=8,
        erm_live_probability_check_images=8,erm_cached_feature_rows_reused=1199,
        erm_new_linear_probability_rows=1199,dpe_new_ensemble_probability_rows=1199,
        maximum_cached_feature_error=feature_error,maximum_erm_probability_error=probability_error,
        maximum_legacy_dpe_probability_error=dpe_error,
        no_model_training=True,no_ot_recomputation=True,elapsed_seconds=time.perf_counter()-started))
    print('Validation inference complete; threshold selection has not used test labels.',flush=True)


def analyze(config,out,asset):
    from scripts.report_threshold_probe import execute_analysis
    execute_analysis(config,out,asset)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','infer','analysis'],default='all')
    args=parser.parse_args();config=settings();out=ROOT/config['output_root'];out.mkdir(parents=True,exist_ok=True)
    asset,validation=audit(config,out)
    if args.stage in ['all','infer']:infer_validation(config,out,asset,validation)
    if args.stage in ['all','analysis']:analyze(config,out,asset)


if __name__=='__main__':main()
