"""Paired mild-blur re-evaluation using frozen cohort, compositor and checkpoints."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
import torch
from torchvision import transforms
import yaml

from src.counterfactual.common import read_csv,write_csv,write_json,sha256,pixel_hash
from src.counterfactual.compositor import foreground_layer,composite
from src.counterfactual.data import ForegroundRepository,BackgroundRepository
from src.counterfactual.investigation import MODELS,FrozenComparison,intervention_records,evaluate_stream
from src.datasets.waterbirds import waterbirds_transform


def key(r,model):
    return (model,str(r['source_image_id']),r['background_id'],r['control'],float(r['sigma']))


def old_records(previous):
    rows=read_csv(previous/'predictions.csv')
    result={key(r,r['model_name']):r for r in rows}
    if len(result)!=len(rows):raise ValueError('Previous predictions have duplicate paired keys')
    return result


def image_hashes(image,preview,tensor):
    im=preview(ImageOps.exif_transpose(image).convert('RGB'))
    return dict(model_input_sha256=pixel_hash(im),
        normalized_tensor_sha256=hashlib.sha256(tensor(im).numpy().tobytes()).hexdigest())


def runtime():
    torch.set_num_threads(4);torch.manual_seed(20261118)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def contact_sheet(config,cohort,output):
    rng=random.Random(config['sampling_seed'])
    selected=[]
    for y in (0,1):
        species=sorted({r['species'] for r in cohort if int(r['y'])==y})
        if len(species)!=10:raise ValueError('Expected ten species per class')
        selected.extend(rng.choice([r for r in cohort if r['species']==s]) for s in species)
    fr,br=ForegroundRepository(ROOT/config['data_root']),BackgroundRepository(ROOT/config['data_root'])
    t=waterbirds_transform().transforms;preview=transforms.Compose(t[:2])
    size=224;left=220;top=100;row_h=262
    font_path=Path('C:/Windows/Fonts/arial.ttf')
    font=ImageFont.truetype(str(font_path),16) if font_path.exists() else ImageFont.load_default()
    small=ImageFont.truetype(str(font_path),13) if font_path.exists() else font
    canvas=Image.new('RGB',(left+5*(size+12),top+row_h*20),'white');draw=ImageDraw.Draw(canvas)
    draw.text((12,10),'Waterbirds mild blur v2: 20 fixed foregrounds, exact 224 x 224 model crops',font=font,fill='black')
    draw.text((12,34),'Candidate sigmas; independent human recognizability review is PENDING. Mask/background unchanged within each row.',font=small,fill='#445566')
    for col,(level,sigma) in enumerate(config['severities'].items()):
        draw.text((left+col*(size+12),65),f'{level}: sigma={sigma:g}'+(' (diagnostic)' if level=='extreme' else ''),font=font,fill='black')
    neutral_selected=selected[:3]+selected[10:13]
    neutral=Image.new('RGB',(left+5*(size+12),top+row_h*6),'white');nd=ImageDraw.Draw(neutral)
    nd.text((12,12),'Foreground-only examples: same neutral gray RGB(127,127,127)',font=font,fill='black')
    for col,(level,sigma) in enumerate(config['severities'].items()):
        nd.text((left+col*(size+12),65),f'{level}: sigma={sigma:g}',font=font,fill='black')
    chosen=[]
    for index,fg in enumerate(selected):
        rgba,_=fr.get(fg['foreground_id']);bg_id=rng.choice(config['background_ids']);bg,_=br.get(bg_id)
        reference_alpha=None
        y=top+row_h*index
        text=f"{'Landbird' if fg['y']=='0' else 'Waterbird'} / ID {fg['bird_source_id']}\n"+fg['species'].replace('_',' ').replace('.', '.\n',1)+f'\n{bg_id}'
        draw.multiline_text((10,y+25),text,font=small,fill='black',spacing=4)
        for col,(level,sigma) in enumerate(config['severities'].items()):
            layer=foreground_layer(rgba,float(sigma));alpha=pixel_hash(layer.getchannel('A'))
            if reference_alpha is None:reference_alpha=alpha
            if alpha!=reference_alpha:raise ValueError('QC mask changed across blur')
            image,audit=composite(layer,bg)
            if not audit['outside_support_exact'] or not audit['opaque_foreground_exact']:raise ValueError('QC invariance failed')
            view=preview(image)
            if view.size!=(224,224):raise ValueError('Inconsistent contact sheet image dimensions')
            canvas.paste(view,(left+col*(size+12),y))
            if fg in neutral_selected:
                ni=neutral_selected.index(fg);nv,_=composite(layer,Image.new('RGB',(256,256),(127,127,127)))
                neutral.paste(preview(nv),(left+col*(size+12),top+row_h*ni))
                nd.multiline_text((10,top+row_h*ni+25),text.split('\n'+bg_id)[0],font=small,fill='black',spacing=4)
        chosen.append(dict(source_image_id=fg['bird_source_id'],foreground_id=fg['foreground_id'],
            species_id=fg['species'],true_y=int(fg['y']),background_id=bg_id,mask_sha256=fg['mask_sha256']))
    canvas.save(output/'blur_contact_sheet.png');neutral.save(output/'foreground_only_contact_sheet.png')
    write_csv(output/'qc_foregrounds.csv',chosen)
    return chosen


def qc(settings):
    previous=ROOT/settings['previous_results'];output=ROOT/settings['output_root']
    output.mkdir(exist_ok=True,parents=True)
    if (output/'blur_config.json').exists():raise FileExistsError('Preserve existing v2 output; use --stage infer/report or choose a new output_root')
    integrity=json.loads((previous/'integrity.json').read_text(encoding='utf-8'))['sha256']
    for filename in ['predictions.csv','cohort.csv','checkpoint_audit.json','experiment_config.yaml']:
        if sha256(previous/filename)!=integrity[filename]:raise ValueError('Previous result integrity failure: '+filename)
    config=yaml.safe_load((previous/'experiment_config.yaml').read_text(encoding='utf-8'))
    old_source_hashes=config['source_code_sha256']
    config.update(settings)
    config.update(started_utc=datetime.now(timezone.utc).isoformat(),previous_predictions_sha256=sha256(previous/'predictions.csv'),
        previous_config_sha256=sha256(previous/'experiment_config.yaml'),previous_source_code_sha256=old_source_hashes,
        source_code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),
            ROOT/'src/counterfactual/investigation.py',ROOT/'src/counterfactual/compositor.py',ROOT/'src/counterfactual/model.py',
            ROOT/'src/datasets/waterbirds.py',ROOT/'src/evaluation/dpe_export.py',ROOT/'src/methods/dpe_head_reweighting/ensemble.py']},
        effective_sigma=dict(stage='After fixed premultiplied bilinear foreground resize to 256x256; before composition and center crop',
            canvas=[256,256],model_view=[224,224],scale='1 canvas pixel = 1 model-crop pixel; no later rescale',
            mask='Unchanged alpha; normalized Gaussian RGB convolution with masked numerator/denominator',
            interpolation='PIL bilinear; same as previous',neutral_rgb=[127,127,127]),
        requested_previous_filename='predictions(2).csv',requested_previous_file_supplied=False,
        previous_csv_resolution='Audited canonical saved results/poter_failure/predictions.csv (34 columns); no unprovided browser-renamed CSV is assumed identical')
    shutil.copyfile(previous/'cohort.csv',output/'cohort.csv')
    config['cohort_sha256']=sha256(output/'cohort.csv')
    cohort=read_csv(output/'cohort.csv')
    if len(cohort)!=100 or Counter(r['y'] for r in cohort)!={'0':50,'1':50} or len({r['species'] for r in cohort})!=20:
        raise ValueError('Previous cohort does not match the required 100-image, 20-species design')
    config['qc_selection']=contact_sheet(config,cohort,output)
    old=old_records(previous)
    t=waterbirds_transform().transforms;preview,tensor=transforms.Compose(t[:2]),transforms.Compose(t[2:])
    audits=[];matches=Counter();images_n=0
    for images,records in intervention_records(cohort,config,output,save_examples=False):
        for image,r in zip(images,records):
            hashes=image_hashes(image,preview,tensor)
            if pixel_hash(image)!=r['raw_input_sha256'] or not r['invariance_passed']:raise ValueError('Rendered image integrity failed')
            matched=0
            for model in MODELS:
                previous_row=old.get(key(r,model))
                if previous_row:
                    for field,value in dict(raw_input_sha256=r['raw_input_sha256'],mask_sha256=r['mask_sha256'],
                        foreground_id=r['foreground_id'],species_id=r['species_id'],source_sha256=r['source_sha256'],
                        source_split=r['source_split'],original_group=str(r['original_group']),**hashes).items():
                        if previous_row[field]!=value:raise ValueError(f'Paired previous/current mismatch {field}: {key(r,model)}')
                    matched+=1;matches[r['control']+'@'+str(r['sigma'])]+=1
                elif r['control'] in ('original','background_only') or r['sigma'] in (0,4):
                    raise ValueError('Required shared previous image is missing')
            audits.append(dict(image_id=r['image_id'],source_image_id=r['source_image_id'],control=r['control'],
                severity=r['severity'],sigma=r['sigma'],raw_input_sha256=r['raw_input_sha256'],
                mask_sha256=r['mask_sha256'],alpha_sha256=r['alpha_sha256'],
                invariance_passed=True,previous_model_rows_matched=matched,**hashes))
            images_n+=1
    if images_n!=3000 or sum(matches.values())!=4500:raise ValueError('Unexpected v2/shared image coverage')
    write_csv(output/'image_integrity.csv',audits)
    write_json(output/'blur_config.json',config)
    write_json(output/'qc_audit.json',dict(status='passed_before_model_inference',images=images_n,
        model_view=[224,224],all_masks_unchanged=True,all_outside_foreground_background_pixels_unchanged=True,
        shared_previous_prediction_rows=4500,shared_counts=dict(matches),clean_and_sigma4_hashes_identical=True,
        contact_foregrounds=20,classes={'landbird':10,'waterbird':10},species=20,
        human_recognizability='Not independently reviewed; automated invariance/dimension checks do not establish recognizability'))
    write_json(output/'status.json',dict(status='qc_passed_no_inference_yet'))
    print('QC passed: 3,000 inputs; 4,500 shared model-row hashes matched. Contact sheets saved.',flush=True)


class PairedPredictionCache:
    def __init__(self,model,previous,output):
        self.model=model;self.old=old_records(previous);self.audits={r['image_id']:r for r in read_csv(output/'image_integrity.csv')}
        self.counts=Counter()
        info=json.loads((previous/'checkpoint_audit.json').read_text(encoding='utf-8'))
        for name in ('POTER','ERM_enhanced'):
            for field in ('checkpoint_sha256','preprocess_fingerprint','numeric_runtime'):
                if info[name][field]!=model.info[name][field]:raise ValueError('Cache checkpoint/preprocess/runtime mismatch: '+field)
        for field in ('backbone_sha256','heads_sha256'):
            if info['DPE_legacy'][field]!=model.info['DPE_legacy'][field]:raise ValueError('DPE cache checkpoint changed')

    def __call__(self,images,records):
        output={name:[None]*len(images) for name in MODELS};fresh=[]
        for i,(image,r) in enumerate(zip(images,records)):
            if pixel_hash(image)!=self.audits[r['image_id']]['raw_input_sha256']:raise ValueError('Images changed after QC')
            cached=[self.old.get(key(r,m)) for m in MODELS]
            if all(cached):
                for model,old in zip(MODELS,cached):
                    if old['normalized_tensor_sha256']!=self.audits[r['image_id']]['normalized_tensor_sha256']:
                        raise ValueError('Cached tensor changed')
                    output[model][i]=dict(logits=[float(old['logit_landbird']),float(old['logit_waterbird'])],
                        probabilities=[float(old['p_landbird']),float(old['p_waterbird'])],pred_y=int(old['pred_y']),
                        model_input_sha256=old['model_input_sha256'],normalized_tensor_sha256=old['normalized_tensor_sha256'])
                    self.counts[(model,r['severity'],r['control'],'reused_exact_previous')]+=1
            else:fresh.append(i)
        if fresh:
            inferred=self.model.predict([images[i] for i in fresh])
            for model in MODELS:
                for i,pred in zip(fresh,inferred[model]):
                    if pred['normalized_tensor_sha256']!=self.audits[records[i]['image_id']]['normalized_tensor_sha256']:
                        raise ValueError('Real inference tensor differs from QC tensor')
                    output[model][i]=pred
                    self.counts[(model,records[i]['severity'],records[i]['control'],'new_real_inference')]+=1
        return output


def infer(settings):
    output=ROOT/settings['output_root'];config=json.loads((output/'blur_config.json').read_text(encoding='utf-8'))
    if (output/'predictions.csv').exists():raise FileExistsError('Preserve existing v2 predictions; use --stage report')
    previous=ROOT/config['previous_results'];cohort=read_csv(output/'cohort.csv')
    if sha256(previous/'predictions.csv')!=config['previous_predictions_sha256'] or sha256(output/'cohort.csv')!=config['cohort_sha256']:
        raise ValueError('Locked previous results/cohort changed since QC')
    model=FrozenComparison(config,output)
    cache=PairedPredictionCache(model,previous,output)
    # Smoke uses the selected QC images only, includes new sigma0.5 and shared0/4.
    fg_ids=[config['qc_selection'][i]['foreground_id'] for i in (0,10)]
    smoke_cohort=[next(r for r in cohort if r['foreground_id']==fg) for fg in fg_ids]
    images=[];records=[]
    for ims,rr in intervention_records(smoke_cohort,config,output,save_examples=False):
        for im,r in zip(ims,rr):
            if r['control']=='natural' and r['background_id']==config['background_ids'][0] and r['sigma'] in (0,.5,4):
                images.append(im);records.append(r)
    model.smoke(images,output)
    fresh=model.predict(images);differences=[]
    for i,r in enumerate(records):
        if r['sigma'] not in (0,4):continue
        for name in MODELS:
            old=cache.old[key(r,name)];pred=fresh[name][i]
            error=max(abs(pred['probabilities'][j]-float(old[['p_landbird','p_waterbird'][j]])) for j in (0,1))
            if error>1e-4 or pred['pred_y']!=int(old['pred_y']):raise ValueError('Shared sigma real-inference smoke disagrees with old predictions')
            differences.append(dict(model_name=name,source_image_id=r['source_image_id'],sigma=r['sigma'],
                max_probability_error=error,prediction_matches=True))
    write_csv(output/'shared_sigma_inference_smoke.csv',differences)
    result=evaluate_stream(model,intervention_records(cohort,config,output),output/'predictions.csv',
        total_sources=100,phase='mild_blur_v2',output=output,predictor=cache)
    reuse=[dict(model_name=m,severity=s,control=c,execution=e,rows=n) for (m,s,c,e),n in cache.counts.items()]
    write_csv(output/'prediction_reuse_audit.csv',reuse)
    result.update(reused_prediction_rows=sum(r['rows'] for r in reuse if r['execution']=='reused_exact_previous'),
        new_prediction_rows=sum(r['rows'] for r in reuse if r['execution']=='new_real_inference'),
        auxiliary_smoke_images=len(images),auxiliary_smoke_independent_from_cache=True)
    write_json(output/'timings.json',result)
    write_json(output/'status.json',dict(status='inference_completed',**result))
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/mild_blur_v2.yaml')
    parser.add_argument('--stage',choices=['qc','infer','report','all'],default='all')
    args=parser.parse_args();settings=yaml.safe_load((ROOT/args.config).read_text(encoding='utf-8'))
    runtime()
    if args.stage in ('qc','all'):qc(settings)
    if args.stage in ('infer','all'):infer(settings)
    if args.stage in ('report','all'):
        from report_mild_blur_v2 import build_report
        build_report(ROOT/settings['output_root'])


if __name__=='__main__':main()
