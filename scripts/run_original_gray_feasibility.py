"""Exact benchmark-RGB gray inference gate; frozen POTER, no search/training."""
import argparse
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
from PIL import Image,ImageDraw,ImageFont

from src.counterfactual.common import sha256,write_json
from src.counterfactual.original_gray import decode_pair,build_gray,contact_tiles
from src.counterfactual.fdat_feasibility import fdat


def config():
    c=json.loads((ROOT/'configs/original_gray_feasibility.json').read_text())
    if any(c[k] for k in ['allow_training','allow_threshold_search','allow_ot_changes','allow_mask_model']):
        raise ValueError('Only fixed-policy inference is authorized in this experiment')
    if (c['locked_threshold'],c['locked_fdat_lambda'],c['locked_fdat_cap'])!=(.5,1,1):
        raise ValueError('Locked policies must remain threshold0.5/lambda1/cap1')
    return c


def dependencies(c,out):
    paths={'config':ROOT/'configs/original_gray_feasibility.json',
        'builder':ROOT/'src/counterfactual/original_gray.py',
        'model_adapter':ROOT/'src/counterfactual/model.py',
        'model_loader':ROOT/'src/evaluation/prediction_export.py',
        'preprocess':ROOT/'src/datasets/waterbirds.py',
        'source_predictions':ROOT/c['original_predictions'],
        'foreground_manifest':ROOT/c['data_root']/'foregrounds_manifest.csv',
        'metadata':ROOT/c['data_root']/'metadata.csv',
        'dataset_audit':out/'dataset_audit.csv'}
    hashes={name:sha256(path) for name,path in paths.items()}
    hashes['checkpoint']=c['checkpoint_sha256']
    digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    return hashes,digest


def audit(c,out):
    started=time.perf_counter();data=ROOT/c['data_root']
    for folder in ['results/poter_failure','outputs/mild_blur_v2','outputs/threshold_algorithm_feasibility']:
        base=ROOT/folder
        for name,digest in json.loads((base/'integrity.json').read_text())['sha256'].items():
            if sha256(base/name)!=digest:raise ValueError('Earlier evidence changed: '+str(base/name))
    meta=pd.read_csv(data/'metadata.csv');meta=meta[meta.split==2].set_index('img_id')
    manifests=pd.read_csv(data/'foregrounds_manifest.csv').set_index('bird_source_id')
    cached=pd.read_csv(ROOT/c['original_predictions'],float_precision='round_trip')
    cached=cached[cached.model_name=='POTER'].copy().sort_values('source_image_id')
    info=json.loads((ROOT/c['checkpoint_audit']).read_text())['POTER']
    if info['checkpoint_sha256']!=c['checkpoint_sha256'] or sha256(ROOT/c['checkpoint'])!=c['checkpoint_sha256']:
        raise ValueError('POTER checkpoint identity differs')
    if info['class_index_to_name']!={'0':'landbird','1':'waterbird'}:raise ValueError('Model class order differs')
    if len(cached)!=5794 or cached.source_image_id.nunique()!=5794 or set(cached.source_image_id)!=set(meta.index):
        raise ValueError('Original cache is not one-to-one full5794 test')
    if cached.source_image_id.duplicated().any() or manifests.index.duplicated().any():raise ValueError('Duplicate source or mask key')
    if set(cached.control)!={'original'} or set(cached.source_split)!={'test'} or set(cached.model_seed)!={1}:
        raise ValueError('Wrong cached condition/split/seed')
    if not np.allclose(cached.p_landbird+cached.p_waterbird,1,atol=1e-6,rtol=0):raise ValueError('Unnormalized original scores')
    if not np.isfinite(cached[['p_waterbird','p_landbird','logit_waterbird','logit_landbird']]).all().all():raise ValueError('Invalid original scores')
    if not np.array_equal((cached.p_waterbird>=.5).to_numpy(int),cached.pred_y.to_numpy(int)):
        raise ValueError('Fixed0.5 prediction differs from original cache')
    prior=out/'run_config.json'
    if prior.exists():
        saved=json.loads(prior.read_text())
        if saved['config']!=c:raise ValueError('Configuration changed: use a new output directory')
    else:write_json(prior,dict(config=c,started_utc=datetime.now(timezone.utc).isoformat(),
        no_training=True,no_tuning=True,gray_builder_uses_only_benchmark_RGB=True))
    records=[]
    for number,r in enumerate(cached.itertuples(),1):
        sid=int(r.source_image_id);m=manifests.loc[sid];official=meta.loc[sid]
        if m.canonical_filename!=official.img_filename or m.species!=r.species_id:
            raise ValueError(f'Wrong image/mask canonical identity: {sid}')
        if int(r.true_y)!=int(official.y) or int(r.original_group)!=2*int(official.y)+int(official.place):
            raise ValueError(f'Wrong official label/group: {sid}')
        expected_mask_suffix=Path(official.img_filename).with_suffix('.png').as_posix()
        if not Path(m.original_mask_path).as_posix().endswith(expected_mask_suffix):
            raise ValueError(f'Original mask filename is not aligned: {sid}')
        image_path=data/m.source_image_path;mask_path=data/m.mask_path
        if not image_path.is_file() or not mask_path.is_file():raise ValueError(f'Missing benchmark image/mask: {sid}')
        image_hash=sha256(image_path);mask_hash=sha256(mask_path)
        if image_hash!=m.source_sha256 or image_hash!=r.source_sha256 or mask_hash!=m.mask_sha256:
            raise ValueError(f'Changed image/mask file: {sid}')
        if sha256(m.original_mask_path)!=m.original_mask_sha256:raise ValueError(f'Official mask changed: {sid}')
        rgb,mask,native=decode_pair(image_path,mask_path)
        with Image.open(m.original_mask_path) as original_mask:
            # Stored mask is the original L coverage; no inferred segmentation.
            if int(native['orientation'])==1 and not np.array_equal(np.asarray(mask),np.asarray(original_mask.convert('L'))):
                raise ValueError(f'Stored versus official mask differs: {sid}')
        _,full,gray,alpha,pixels=build_gray(rgb,mask,tuple(c['gray_rgb']))
        if native['source_decoded_pixel_sha256']!=r.raw_input_sha256:
            raise ValueError(f'Decoded original pixel identity differs: {sid}')
        if pixels['original_model_input_sha256']!=r.model_input_sha256:
            raise ValueError(f'Original spatial preprocessing differs from saved model input: {sid}')
        records.append(dict(source_image_id=sid,image_id=r.image_id,foreground_id=r.foreground_id,
            species_id=r.species_id,true_y=int(r.true_y),original_group=int(r.original_group),control='original',source_split='test',
            source_path=str(image_path.resolve()),source_file_sha256=image_hash,mask_path=str(mask_path.resolve()),mask_sha256=mask_hash,
            official_mask_path=str(m.original_mask_path),official_mask_sha256=m.original_mask_sha256,
            canonical_filename=official.img_filename,**native,**pixels,
            p_full=float(r.p_waterbird),p_full_landbird=float(r.p_landbird),
            full_logit_waterbird=float(r.logit_waterbird),full_logit_landbird=float(r.logit_landbird),
            full_normalized_tensor_sha256=r.normalized_tensor_sha256,
            checkpoint_sha256=c['checkpoint_sha256'],model_seed=1,
            preprocessing_fingerprint=info['preprocess_fingerprint']))
        if number%1000==0 or number==5794:print(json.dumps(dict(phase='source/mask/pixel audit',images=number,total=5794,elapsed_seconds=time.perf_counter()-started)),flush=True)
    frame=pd.DataFrame(records)
    if frame.original_group.nunique()!=4:raise ValueError('Official four groups missing')
    frame.to_csv(out/'dataset_audit.csv',index=False)
    write_json(out/'dataset_audit_summary.json',dict(images=len(frame),unique_sources=frame.source_image_id.nunique(),
        group_counts={int(g):int(n) for g,n in frame.original_group.value_counts().sort_index().items()},
        mask_hashes_valid=True,official_masks_exact=True,all_opaque_pixels_preserved=True,
        images_without_model_opaque=int((frame.opaque_pixels==0).sum()),images_without_eroded_core=int((~frame.eroded_core_available).sum()),
        all_model_input_hashes_match_cached_full=True,exclusions=0,soft_masks=int(frame.mask_is_soft.sum()),
        no_new_mask_generation=True,elapsed_seconds=time.perf_counter()-started,
        spatial_order='shared benchmark RGB square256 first; then attach bilinear alpha, neutral127; center224 common',
        soft_alpha_caveat='Soft masks can include broad interior regions; every nonopaque pixel blends by design. No mask threshold or morphology is used for inference.'))
    return frame


def choose_preflight(frame,c):
    selected=[]
    for group in range(4):
        rows=frame[frame.original_group==group].copy();used=set();species=set()
        for field,ascending,label in [('mask_support_fraction',True,'small bird'),
            ('mask_boundary_ratio',False,'thin/complex mask'),('background_texture_score',False,'complex background')]:
            ranked=rows.sort_values([field,'source_image_id'],ascending=[ascending,True])
            for _ in range(2):
                options=ranked[~ranked.source_image_id.isin(used)&~ranked.species_id.isin(species)]
                if options.empty:options=ranked[~ranked.source_image_id.isin(used)]
                if options.empty:raise ValueError('Insufficient diverse preflight examples')
                row=options.iloc[0].to_dict();row['preflight_reason']=label
                selected.append(row);used.add(row['source_image_id']);species.add(row['species_id'])
    result=pd.DataFrame(selected)
    if len(result)!=24 or result.source_image_id.nunique()!=24:raise ValueError('Exactly24 diverse preflight images required')
    return result


def sheets(c,out,selected):
    font=ImageFont.load_default(size=12)
    def render(rows,path):
        width=224;row_height=256
        canvas=Image.new('RGB',(4*width,44+row_height*len(rows)),(248,248,248));draw=ImageDraw.Draw(canvas)
        titles=['Original model224','New gray model224','Mask overlay (QC)','Abs RGB diff ×4 (QC)']
        for i,title in enumerate(titles):draw.text((i*width+5,5),title,fill='black',font=font)
        for i,r in enumerate(rows.itertuples()):
            rgb,mask,_=decode_pair(r.source_path,r.mask_path);_,full,gray,alpha,_=build_gray(rgb,mask,tuple(c['gray_rgb']))
            y=44+i*row_height
            caption=f'ID{r.source_image_id} G{r.original_group} {r.preflight_reason} | {r.species_id}'
            draw.text((5,y),caption,fill='black',font=font)
            for col,tile in enumerate(contact_tiles(full,gray,alpha)):canvas.paste(tile,(col*width,y+28))
        canvas.save(path)
    render(selected,out/'original_gray_contact_sheet.png')
    for page in range(3):render(selected.iloc[page*8:(page+1)*8],out/f'original_gray_contact_sheet_{page+1}.png')


def model(c):
    import torch
    torch.set_num_threads(c['cpu_threads']);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if c['device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable for bounded inference')
    from src.counterfactual.model import ModelAdapter
    return ModelAdapter(ROOT/c['checkpoint'],c['device'],metadata=ROOT/c['data_root']/'metadata.csv')


def gray_records(c,out,adapter,rows,dependency):
    images=[]
    for r in rows.itertuples():
        rgb,mask,_=decode_pair(r.source_path,r.mask_path)
        gray256,_,_,_,audit=build_gray(rgb,mask,tuple(c['gray_rgb']))
        if audit['gray_model_input_sha256']!=r.gray_model_input_sha256:raise ValueError('Gray reconstruction changed')
        images.append(gray256)
    predictions=adapter.predict_batch(images,c['batch_size'])
    result=[]
    for r,p in zip(rows.itertuples(),predictions):
        if p['model_input_sha256']!=r.gray_model_input_sha256:raise ValueError('Gray model preprocess hash differs')
        result.append(dict(source_image_id=int(r.source_image_id),p_gray=float(p['probabilities'][1]),
            p_gray_landbird=float(p['probabilities'][0]),gray_logit_waterbird=float(p['logits'][1]),
            gray_logit_landbird=float(p['logits'][0]),gray_normalized_tensor_sha256=p['normalized_tensor_sha256'],
            gray_model_input_sha256=p['model_input_sha256'],source_file_sha256=r.source_file_sha256,
            mask_sha256=r.mask_sha256,checkpoint_sha256=c['checkpoint_sha256'],cache_dependency_sha256=dependency))
    return result


def preflight(c,out,frame):
    selected=choose_preflight(frame,c);selected.to_csv(out/'preflight_examples.csv',index=False);sheets(c,out,selected)
    adapter=model(c);hashes,dependency=dependencies(c,out)
    originals=[]
    for r in selected.itertuples():
        rgb,_,_=decode_pair(r.source_path,r.mask_path);originals.append(rgb)
    started=time.perf_counter();fresh=adapter.predict_batch(originals,c['batch_size'])
    probability_error=[];logit_error=[];checks=[]
    for r,p in zip(selected.itertuples(),fresh):
        pe=float(np.max(abs(np.asarray(p['probabilities'])-[r.p_full_landbird,r.p_full])))
        le=float(np.max(abs(np.asarray(p['logits'])-[r.full_logit_landbird,r.full_logit_waterbird])))
        if p['model_input_sha256']!=r.original_model_input_sha256 or p['normalized_tensor_sha256']!=r.full_normalized_tensor_sha256:
            raise ValueError('Live original pixel/tensor input differs from cache')
        probability_error.append(pe);logit_error.append(le)
        checks.append(dict(source_image_id=r.source_image_id,probability_max_abs_error=pe,logit_max_abs_error=le,
            original_pixel_hash_matches=True,original_normalized_tensor_hash_matches=True,
            original_cached_pred=int(r.p_full>=.5),original_fresh_pred=int(p['probabilities'][1]>=.5)))
    if max(probability_error)>c['parity_probability_atol'] or max(logit_error)>c['parity_logit_atol']:
        raise ValueError('Cached/live original FP32 parity failed; tolerance is not relaxed')
    if any(r['original_cached_pred']!=r['original_fresh_pred'] for r in checks):raise ValueError('Original policy prediction parity failed')
    pd.DataFrame(checks).to_csv(out/'preflight_probability_parity.csv',index=False)
    predictions=gray_records(c,out,adapter,selected,dependency)
    pd.DataFrame(predictions).to_csv(out/'preflight_gray_predictions.csv',index=False)
    write_json(out/'model_runtime.json',adapter.info())
    write_json(out/'preflight.json',dict(status='PASS_NUMERIC_PIXEL_PARITY',images=24,groups=4,
        species=selected.species_id.nunique(),maximum_probability_error=max(probability_error),
        maximum_logit_error=max(logit_error),probability_tolerance=c['parity_probability_atol'],logit_tolerance=c['parity_logit_atol'],
        fresh_original_forwards=24,fresh_gray_forwards=24,cache_dependency_sha256=dependency,dependencies=hashes,
        preflight_seconds=time.perf_counter()-started,
        contact_sheet_sha256={p.name:sha256(p) for p in out.glob('original_gray_contact_sheet*.png')},
        next_action='Inspect all24 contact examples, record visual review, then automatically continue gray inference'))
    print(json.dumps(dict(preflight='numeric PASS',images=24,max_probability_error=max(probability_error),max_logit_error=max(logit_error))),flush=True)


def infer(c,out,frame):
    pre=json.loads((out/'preflight.json').read_text());review=json.loads((out/'preflight_visual_review.json').read_text())
    hashes,dependency=dependencies(c,out)
    if pre['cache_dependency_sha256']!=dependency or review['preflight_dependency_sha256']!=dependency:
        raise ValueError('Preflight inputs/code/config changed')
    if review['status']!='PASS' or review['images_reviewed']!=24:raise ValueError('24-example visual preflight incomplete')
    for name,digest in pre['contact_sheet_sha256'].items():
        if sha256(out/name)!=digest:raise ValueError('Contact sheet changed after preflight')
    cache=out/'gray_cache';cache.mkdir(exist_ok=True)
    chunks=[out/'preflight_gray_predictions.csv',*sorted(cache.glob('chunk_*.csv'))]
    prior=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in chunks],ignore_index=True)
    if prior.source_image_id.duplicated().any():raise ValueError('Duplicate gray cache IDs')
    if set(prior.cache_dependency_sha256)!={dependency}:raise ValueError('Stale gray inference cache')
    lookup=frame.set_index('source_image_id')
    for r in prior.itertuples():
        source=lookup.loc[r.source_image_id]
        if r.source_file_sha256!=source.source_file_sha256 or r.mask_sha256!=source.mask_sha256 or r.gray_model_input_sha256!=source.gray_model_input_sha256:
            raise ValueError('Gray cache/source identity differs')
    missing=frame[~frame.source_image_id.isin(prior.source_image_id)]
    adapter=model(c) if len(missing) else None;started=time.perf_counter();buffer=[];new_count=0
    next_chunk=len(list(cache.glob('chunk_*.csv')))
    for start in range(0,len(missing),c['batch_size']):
        batch=missing.iloc[start:start+c['batch_size']]
        buffer.extend(gray_records(c,out,adapter,batch,dependency));new_count+=len(batch)
        if len(buffer)>=256 or start+c['batch_size']>=len(missing):
            target=cache/f'chunk_{next_chunk:04d}.csv';temporary=target.with_suffix('.tmp')
            if target.exists():raise ValueError('Never overwrite completed gray cache shard')
            pd.DataFrame(buffer).to_csv(temporary,index=False);temporary.replace(target);buffer=[];next_chunk+=1
            progress=dict(gray_images_completed=len(prior)+new_count,total=5794,new_gray_forwards=new_count,
                elapsed_seconds=time.perf_counter()-started)
            write_json(out/'inference_progress.json',progress);print(json.dumps(progress),flush=True)
    all_gray=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in
        [out/'preflight_gray_predictions.csv',*sorted(cache.glob('chunk_*.csv'))]],ignore_index=True)
    if len(all_gray)!=5794 or all_gray.source_image_id.nunique()!=5794 or set(all_gray.source_image_id)!=set(frame.source_image_id):
        raise ValueError('Full5794 gray probability coverage incomplete')
    joined=frame.merge(all_gray,on='source_image_id',validate='one_to_one',suffixes=('','_inference'))
    if not np.allclose(joined.p_gray+joined.p_gray_landbird,1,atol=1e-6,rtol=0):raise ValueError('Invalid gray probabilities')
    if not np.isfinite(joined[['p_gray','gray_logit_landbird','gray_logit_waterbird']]).all().all():raise ValueError('Nonfinite gray output')
    fixed=fdat(joined.p_full,joined.p_gray,c['locked_fdat_lambda'],c['locked_fdat_cap'])
    joined['p_waterbird']=joined.p_full
    joined['model_name']='POTER'
    joined['original_input_id']=joined.image_id
    joined['gray_input_id']='original-gray-'+joined.source_image_id.astype(str)+'-shared-spatial-v1'
    joined['mask_native_width']=joined.native_width;joined['mask_native_height']=joined.native_height
    joined['mask_mode']='L; soft alpha coverage /255'
    joined['full_05_pred']=(joined.p_full>=.5).astype(int);joined['gray_05_pred']=(joined.p_gray>=.5).astype(int)
    joined['fdat_pred']=fixed['pred'];joined['fdat_positive_discrepancy']=fixed['discrepancy'];joined['fdat_per_image_threshold']=fixed['threshold']
    joined['and_pred']=((joined.p_full>=.5)&(joined.p_gray>=.5)).astype(int)
    joined['historical_full_065_pred']=(joined.p_full>=.65).astype(int)
    joined['evaluation_role']='previously-inspected original test; exploratory fixed-policy feasibility'
    joined.to_csv(out/'original_gray_pairs.csv',index=False)
    history=json.loads((out/'inference_run.json').read_text()) if (out/'inference_run.json').exists() else dict(runs=[])
    history['runs'].append(dict(utc=datetime.now(timezone.utc).isoformat(),new_gray_forwards=new_count,
        prior_gray_rows_reused=len(prior),model_loaded=adapter is not None,
        inference_loop_seconds=time.perf_counter()-started))
    history.update(total_gray_images=5794,preflight_original_forwards=24,preflight_gray_forwards=24,
        all_gray_cached_for_future_reuse=True,no_model_training=True,no_threshold_search=True,
        cache_dependency_sha256=dependency,dependencies=hashes)
    write_json(out/'inference_run.json',history)
    print('Original-gray5794 inference/pairing complete; fixed policies applied without fitting.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['audit','preflight','infer','report'],default='preflight')
    args=parser.parse_args();c=config();out=ROOT/c['output_root'];out.mkdir(parents=True,exist_ok=True)
    try:
        if args.stage in ['audit','preflight']:frame=audit(c,out)
        else:frame=pd.read_csv(out/'dataset_audit.csv',float_precision='round_trip')
        if args.stage=='preflight':preflight(c,out,frame)
        elif args.stage=='infer':infer(c,out,frame)
        elif args.stage=='report':
            from scripts.report_original_gray_feasibility import report
            report(c,out)
    except Exception as error:
        write_json(out/'blocking_status.json',dict(status='BLOCKED',stage=args.stage,error=str(error),
            original_gray_metrics_valid=False,training=False,threshold_search=False))
        raise


if __name__=='__main__':main()
