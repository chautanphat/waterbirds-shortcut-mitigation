"""Immutable, reproducible experiment storage and complete factorial inference."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import subprocess
import uuid

import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision

from .common import (DEFAULT_RUNS, ROOT, SEVERITIES, OPERATOR_STRENGTHS,contained, pixel_hash, read_csv,
                     sha256, write_csv, write_json)
from .compositor import composite, foreground_layer, mask_overlay, difference_heatmap
from .data import BackgroundRepository, ForegroundRepository
from .metrics import annotate_failures, paired_metrics, summarize, truthy


def selected_backgrounds(repository, config, split):
    explicit = config.get('background_ids')
    if explicit:
        selected = [repository.by_id[i] for i in explicit]
    else:
        quota = int(config.get('backgrounds_per_category', 1))
        if not 1 <= quota <= 100:
            raise ValueError('Choose 1–100 backgrounds per category')
        rng = random.Random(int(config.get('seed', 0)))
        selected = []
        for category in ('land', 'water'):
            candidates = sorted(repository.list(category, split), key=lambda r:r['background_id'])
            # Do not unknowingly count copies of the same background as independent candidates.
            unique = {r['pixel_sha256']: r for r in candidates}
            candidates = list(unique.values())
            if len(candidates) < quota:
                raise ValueError(f'Need {quota} unique {category} backgrounds in source split {split}')
            selected.extend(rng.sample(candidates, quota))
    if any(r['original_split'] != split for r in selected):
        raise ValueError('Background source splits must match the source bird split; cross-split borrowing is disabled')
    land = [r for r in selected if r['bg_type'] == 'land']
    water = [r for r in selected if r['bg_type'] == 'water']
    if not land or len(land) != len(water):
        raise ValueError('Natural interventions need balanced Land/Water quotas')
    if len({r['pixel_sha256'] for r in selected}) != len(selected):
        raise ValueError('Duplicate background pixels selected for one experiment')
    return land, water


def run_experiment(config, adapters, *, data_root, output_root=DEFAULT_RUNS, progress=None):
    config = json.loads(json.dumps(config))
    if int(config.get('seed',0))<0:
        raise ValueError('Sampling seed must be nonnegative')
    fg_repo, bg_repo = ForegroundRepository(data_root), BackgroundRepository(data_root)
    fg_ids = config.get('foreground_ids', [])
    if not fg_ids or len(fg_ids) != len(set(fg_ids)):
        raise ValueError('Choose unique canonical foreground IDs')
    if len(fg_ids) > 1000:
        raise ValueError('Interactive run is limited to 1,000 birds; partition larger runs in saved configs')
    if len({fg_repo.by_id[i]['bird_source_id'] for i in fg_ids}) != len(fg_ids):
        raise ValueError('Choose one mask version per source bird in a run')
    splits = {fg_repo.by_id[i]['original_split'] for i in fg_ids}
    if len(splits) != 1:
        raise ValueError('All source birds in a run must share their canonical split')
    split = next(iter(splits))
    background_split = config.get('external_background_split', 'test') if split == 'unknown' else split
    land, water = selected_backgrounds(bg_repo, config, background_split)
    operator=config.get('operator','gaussian')
    if operator not in ('gaussian','contrast','occlusion'):
        raise ValueError('One operator per run: gaussian, contrast, or occlusion')
    severities = config.get('severities', SEVERITIES if operator=='gaussian' else {name:0. for name in SEVERITIES})
    if not isinstance(severities, dict) or not severities:
        raise ValueError('Severities must map names to foreground Gaussian sigmas')
    for name, sigma in severities.items():
        if not name.replace('_','').isalnum() or not np.isfinite(sigma) or not 0 <= sigma <= 30:
            raise ValueError('Invalid severity name or sigma')
    strengths={}
    if operator!='gaussian':
        if any(float(sigma)!=0 for sigma in severities.values()):
            raise ValueError('Contrast and occlusion runs require zero Gaussian sigma; keep operators in separate runs')
        requested=config.get('operator_strengths',OPERATOR_STRENGTHS[operator])
        for name in severities:
            strength=requested.get(name)
            if strength is None or not np.isfinite(strength) or not 0<=strength<=1:
                raise ValueError('Every selected severity requires an explicit operator strength in [0,1]')
            strengths[name]=float(strength)
        severities={name:0. for name in severities}
    opacity = float(config.get('opacity', 1.))
    if not np.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError('Opacity must be between 0 and 1')
    if not config.get('allow_approximate', False) and any(fg_repo.by_id[i]['quality_status'] != 'verified_original_cub' for i in fg_ids):
        raise ValueError('Original CUB cutouts unavailable. Enable explicitly labeled approximate exploratory mode, or run setup --download-cub / --cub-root.')
    if len({a.model_id for a in adapters}) != len(adapters):
        raise ValueError('Model IDs must be unique')
    batch_size = int(config.get('batch_size', 16))
    if batch_size < 1 or batch_size > 256:
        raise ValueError('Batch size must be 1–256')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:10]
    out = Path(output_root) / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / 'images').mkdir()
    (out / 'sources').mkdir()
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = 'unavailable'
    config.update(run_id=run_id, foreground_ids=fg_ids, background_ids=[r['background_id'] for pair in zip(land,water) for r in pair],
        seed=int(config.get('seed', 0)), source_split=split, severities=severities, opacity=opacity,
        operator=operator,operator_strengths=strengths,
        analysis='retrospective_exploratory' if split == 'test' else 'exploratory_intervention',
        git_commit=commit,data_root=str(Path(data_root).resolve()),
        background_selection=[dict(r) for pair in zip(land,water) for r in pair],
        dataset_audit=json.loads((Path(data_root)/'dataset_audit.json').read_text(encoding='utf-8')),
        models=[a.info() for a in adapters], software=dict(python=platform.python_version(), torch=str(torch.__version__),
            torchvision=str(torchvision.__version__), numpy=np.__version__, pillow=Image.__version__),
        geometry=dict(canvas=[256,256], foreground='fixed square resample, premultiplied bilinear',
                      background_fit=config.get('background_fit','cover'), model_view='Resize((256,256));CenterCrop(224)',
                      sigma_unit='pixels in final model view', neutral_rgb=[127,127,127]),
        operator_parameters=dict(kind=operator,strengths=strengths,
            contrast='alpha-weighted foreground RGB mean; factor 1=clean',
            occlusion='source-seeded center; nearest alpha-support pixels; fraction counts support pixels; RGB fill 127; alpha unchanged',
            occlusion_seed='experiment seed XOR first 32 bits of SHA256(source bird ID)'),
        source_code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in sorted([
            *(ROOT/'src/counterfactual').glob('*.py'),*(ROOT/'src/counterfactual/web').glob('*'),
            ROOT/'src/datasets/waterbirds.py',ROOT/'src/evaluation/prediction_export.py',
            ROOT/'src/evaluation/dpe_export.py',ROOT/'src/models/resnet.py',ROOT/'scripts/mixer_core.js']) if p.is_file()},
        input_manifests_sha256={p:sha256(Path(data_root)/p) for p in ['foregrounds_manifest.csv','backgrounds_manifest.csv']})
    write_json(out / 'run_config.json', config)
    write_json(out / 'status.json', dict(status='running'))
    predictions, manifest, audits = [], [], []
    try:
        for bird_number, fg_id in enumerate(fg_ids):
            rgba, fg = fg_repo.get(fg_id)
            source_dir = out / 'sources' / fg_id
            source_dir.mkdir()
            rgba.save(source_dir / 'foreground.png')
            rgba.getchannel('A').save(source_dir / 'mask.png')
            mask_overlay(rgba).save(source_dir / 'mask_overlay.png')
            with Image.open(contained(data_root, fg['source_image_path'])) as im:
                original_image=im.convert('RGB')
                original_image.save(source_dir / 'source.png')
            write_json(source_dir / 'provenance.json', fg)
            images, records = [], []
            if config.get('include_original',True):
                original_bg=bg_repo.by_id.get('hf-bg-'+fg['bird_source_id'],dict(background_id='external-original',
                    bg_type='unknown',scene_category='unknown',source_dataset='external',sha256='',original_split=split))
                original_audit=dict(foreground_layer_sha256='not_applicable_original',alpha_sha256='not_applicable_original',
                    foreground_dimensions=list(original_image.size),aligned_background_sha256='',opaque_foreground_pixels=0,
                    opaque_foreground_exact=None,outside_support_exact=None)
                records.append(_asset_record(out,run_id,fg,original_bg,'original',0,1.,-1,'original',original_image,original_audit,config['seed']))
                images.append(original_image)
            for severity, sigma in severities.items():
                strength=strengths.get(severity)
                occlusion_seed=config['seed'] ^ int(hashlib.sha256(fg['bird_source_id'].encode()).hexdigest()[:8],16)
                layer = foreground_layer(rgba, float(sigma), opacity,operator=operator,strength=strength,seed=occlusion_seed)
                layer_path = source_dir / f'layer_{severity}.png'
                layer.save(layer_path)
                for pair_index, pair in enumerate(zip(land, water)):
                    for bg in pair:
                        background, bg_meta = bg_repo.get(bg['background_id'])
                        image, audit = composite(layer, background, fit=config.get('background_fit','cover'))
                        controls = [('natural',image,audit,bg_meta)]
                        if config.get('include_background_only', True) and severity == next(iter(severities)):
                            no_bird = layer.copy()
                            no_bird.putalpha(0)
                            bg_image, bg_audit = composite(no_bird, background, fit=config.get('background_fit','cover'))
                            controls.append(('background_only',bg_image,bg_audit,bg_meta))
                        for control, rendered, control_audit, meta in controls:
                            records.append(_asset_record(out, run_id, fg, meta, severity, sigma, opacity,
                                                         pair_index, control, rendered, control_audit, config['seed'],
                                                         operator=operator,strength=strength,occlusion_seed=occlusion_seed))
                            images.append(rendered)
                if config.get('include_neutral',True):
                    neutral, audit = composite(layer, Image.new('RGB',layer.size,(127,127,127)))
                    records.append(_asset_record(out,run_id,fg,dict(background_id='neutral-gray127',bg_type='neutral',
                        scene_category='uniform_gray',source_dataset='diagnostic_control',sha256='',original_split=split),
                        severity,sigma,opacity,-1,'foreground_only_on_neutral',neutral,audit,config['seed'],
                        operator=operator,strength=strength,occlusion_seed=occlusion_seed))
                    images.append(neutral)
            for r in records:
                manifest.append(r)
                audits.append({k:r[k] for k in ['asset_id','foreground_layer_sha256','alpha_sha256',
                    'opaque_foreground_exact','outside_support_exact','opaque_foreground_pixels']})
            if adapters:
                for adapter in adapters:
                    results = adapter.predict_batch(images, batch_size=batch_size)
                    for r, result in zip(records, results):
                        pred = dict(r, row_id=r['asset_id']+'__'+adapter.model_id, status='completed',
                            model_id=adapter.model_id, checkpoint_sha256=adapter.checkpoint_sha256,
                            preprocess_fingerprint=adapter.preprocess_fingerprint,
                            score_type=adapter.info()['score_type'], model_input_sha256=result['model_input_sha256'],
                            normalized_tensor_sha256=result['normalized_tensor_sha256'],
                            logit_landbird=result['logits'][0], logit_waterbird=result['logits'][1],
                            true_label_margin=result['logits'][int(r['true_y'])]-result['logits'][1-int(r['true_y'])] if r['valid_target'] else None,
                            p_landbird=result['probabilities'][0], p_waterbird=result['probabilities'][1],
                            pred_y=result['pred_y'], is_correct=result['pred_y']==int(r['true_y']) if r['valid_target'] else None)
                        preview_path = f"images/{r['asset_id']}_model.png"
                        result['preview'].save(out / preview_path)
                        pred['model_input_path'] = preview_path
                        predictions.append(pred)
            else:
                for r in records:
                    predictions.append(dict(r,row_id=r['asset_id']+'__not_run',model_id='not_configured',
                        status='not_run', checkpoint_sha256='',preprocess_fingerprint='',score_type='',
                        model_input_sha256='', normalized_tensor_sha256='',logit_landbird=None,logit_waterbird=None,
                        true_label_margin=None,
                        p_landbird=None,p_waterbird=None,pred_y=None,is_correct=None,model_input_path=''))
            write_csv(out / 'manifest.csv', manifest)
            write_csv(out / 'predictions.csv', predictions)
            if progress:
                progress(dict(completed_birds=bird_number+1,total_birds=len(fg_ids),run_id=run_id))
        pairs = paired_metrics(predictions)
        summary = summarize(predictions,pairs,seed=config['seed'])
        write_csv(out / 'paired_metrics.csv', pairs)
        write_json(out / 'summary.json', summary)
        write_csv(out/'summary.csv',[{k:json.dumps(v,allow_nan=False) if isinstance(v,(dict,list)) else v
                                     for k,v in row.items()} for row in summary])
        write_csv(out/'group_metrics.csv',[dict(model_id=s['model_id'],severity=s['severity'],control=s['control'],
            group_id=g,**values) for s in summary for g,values in s['group_stats'].items()])
        write_json(out / 'invariance_audit.json', audits)
        failures = annotate_failures(predictions, pairs)
        # Original-scene deltas persist with every row, independently of gallery filters.
        write_csv(out / 'predictions.csv', failures)
        write_csv(out / 'failure_index.csv', failures)
        write_json(out / 'background_reuse_audit.json',dict(background_ids=config['background_ids'],
            reuse_across_birds=len(fg_ids), explanation='Fixed selected backgrounds reused across birds; bootstrap unit is source bird. Background generalization is not estimated.'))
        export_case(out, fg_ids[0])
        write_json(out/'integrity.json',dict(version=1,
            core_sha256={name:sha256(out/name) for name in ['run_config.json','manifest.csv','predictions.csv',
                'paired_metrics.csv','summary.json','summary.csv','group_metrics.csv','invariance_audit.json','failure_index.csv']},
            assets_sha256={p.relative_to(out).as_posix():sha256(p) for folder in ('images','sources')
                           for p in sorted((out/folder).rglob('*')) if p.is_file()}))
        write_json(out / 'status.json', dict(status='completed',rows=len(predictions),source_birds=len(fg_ids)))
    except Exception as error:
        write_json(out / 'status.json', dict(status='failed',error=str(error),
            recovery='Reload run_config.json; resume creates a new immutable run with parent_run_id'))
        raise
    return dict(run_id=run_id, path=str(out), rows=len(predictions), summary=summary)


def _asset_record(out, run_id, fg, bg, severity, sigma, opacity, pair, control, image, audit, seed,
                  *,operator='gaussian',strength=None,occlusion_seed=None):
    asset_id = f"{fg['foreground_id']}_{severity}_{pair}_{bg['background_id']}_{control}"
    path = 'images/' + asset_id + '.png'
    image.save(out / path)
    source_known = str(fg['y']) in ('0','1')
    known = source_known and control != 'background_only'
    valid = known and opacity > 0 and control in ('natural','original')
    if control=='natural' and operator=='occlusion' and strength==1:
        valid=False
    return dict(run_id=run_id,asset_id=asset_id, source_bird_id=fg['bird_source_id'], foreground_id=fg['foreground_id'],
        species=fg['species'], background_id=bg['background_id'],bg_type=bg['bg_type'],scene_category=bg['scene_category'],
        background_source_dataset=bg['source_dataset'],background_source_split=bg['original_split'],
        background_source_path=bg.get('source_path',''),background_place_filename=bg.get('place_filename',''),
        background_source_revision=bg.get('source_revision',''),
        background_sha256=bg['sha256'], source_split=fg['original_split'], true_y=int(fg['y']) if known else None,
        source_y=int(fg['y']) if source_known else None,
        original_group=int(fg['original_group']) if source_known else None,
        group_id=2*int(fg['y'])+(bg['bg_type']=='water') if valid else None,
        severity=severity,sigma=float(sigma),opacity=opacity,pair_index=pair,seed=seed,control=control,
        operator=operator if control not in ('original','background_only') else 'none',
        operator_strength=strength if control not in ('original','background_only') else None,
        occlusion_seed=occlusion_seed if operator=='occlusion' else None,
        known_label=known,unknown_ground_truth=not source_known,valid_target=valid,
        exploratory_only=fg['quality_status']!='verified_original_cub',composite_quality_status=fg['quality_status'],
        provenance_status='canonical_metadata_joined' if fg.get('canonical_filename') else 'external_unknown_ground_truth',mask_version=fg['mask_version'],
        raw_input_sha256=pixel_hash(image),png_sha256=sha256(out/path),image_path=path,
        source_path=f"sources/{fg['foreground_id']}/source.png",
        mask_overlay_path=f"sources/{fg['foreground_id']}/mask_overlay.png",
        rgba_path=f"sources/{fg['foreground_id']}/foreground.png",
        **audit)


def load_run(path):
    path = Path(path)
    integrity_path=path/'integrity.json'
    if integrity_path.exists():
        integrity=json.loads(integrity_path.read_text(encoding='utf-8'))
        for name,digest in integrity['core_sha256'].items():
            if sha256(contained(path,name))!=digest:
                raise ValueError(f'Saved run integrity check failed: {name}')
    rows = read_csv(path/'predictions.csv') if (path/'predictions.csv').exists() else []
    ids = {r['row_id'] for r in rows}
    assets = {r['asset_id']:r for r in read_csv(path/'manifest.csv')} if (path/'manifest.csv').exists() else {}
    if len(ids) != len(rows) or any(r['asset_id'] not in assets for r in rows):
        raise ValueError('Prediction rows do not reconcile to manifest')
    pairs = paired_metrics(rows)
    return dict(config=json.loads((path/'run_config.json').read_text(encoding='utf-8')),
        integrity_status='verified_core_files' if integrity_path.exists() else 'legacy_or_incomplete_no_integrity_record',
        rows=annotate_failures(rows,pairs),pairs=pairs,
        summary=json.loads((path/'summary.json').read_text(encoding='utf-8')) if (path/'summary.json').exists() else [],
        status=json.loads((path/'status.json').read_text(encoding='utf-8')))


def export_case(run_dir, foreground_id, row_ids=None):
    """PNG cells, CSV and JSON carry the exact same selected row IDs."""
    run_dir = Path(run_dir)
    rows = [r for r in read_csv(run_dir/'predictions.csv') if r['foreground_id']==foreground_id
            and (row_ids is None or r['row_id'] in row_ids)]
    if not rows:
        raise ValueError('No matching rows to export')
    target = run_dir/'exports'/('case_'+foreground_id+'_'+uuid.uuid4().hex[:8])
    target.mkdir(parents=True,exist_ok=False)
    write_csv(target/'cases.csv', rows)
    write_json(target/'case_packet.json',dict(rows=rows, config=json.loads((run_dir/'run_config.json').read_text(encoding='utf-8')),
        export_provenance=dict(exporter_sha256=sha256(__file__),row_order=[r['row_id'] for r in rows])))
    # First four columns: source, mask, Land/Water natural input. Further rows
    # show the exact model previews and probabilities for every selected result.
    width, cell_w, cell_h = 1120, 280, 310
    height = 340 + ((len(rows)+3)//4)*cell_h
    canvas = Image.new('RGB',(width,height),'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((12,8),f"{foreground_id} | saved run {rows[0]['run_id']} | retrospective exploratory",fill='black')
    head_paths = [rows[0]['source_path'],rows[0]['mask_overlay_path']]
    natural = [r for r in rows if r['control']=='natural']
    head_paths += [next((r['image_path'] for r in natural if r['bg_type']==b),rows[0]['image_path']) for b in ('land','water')]
    for i,(label,p) in enumerate(zip(['Source','Mask overlay','Land composite','Water composite'],head_paths)):
        with Image.open(contained(run_dir,p)) as im:
            canvas.paste(ImageOpsContain(im,(256,256)),(i*cell_w+12,30))
        draw.text((i*cell_w+12,292),label,fill='black')
    for i,r in enumerate(rows):
        x,y = (i%4)*cell_w,340+(i//4)*cell_h
        with Image.open(contained(run_dir,r.get('model_input_path') or r['image_path'])) as im:
            canvas.paste(ImageOpsContain(im,(224,224)),(x+12,y))
        prob = f"P(water)={float(r['p_waterbird']):.5f}" if r['status']=='completed' else 'not_run'
        label=f"{r['model_id']} {r['severity']} {r['bg_type']}\n{r['control']}\n{prob} | target={r['true_y']} pred={r['pred_y']}\n{r.get('operator','gaussian')} strength={r.get('operator_strength','')} sigma={r['sigma']}\n{r['background_id']} pair={r['pair_index']}"
        draw.multiline_text((x+8,y+230),label,fill='black',spacing=3)
    from PIL.PngImagePlugin import PngInfo
    png_info=PngInfo()
    png_info.add_text('run_id',rows[0]['run_id'])
    png_info.add_text('row_ids',json.dumps([r['row_id'] for r in rows]))
    png_info.add_text('cell_order','row-major, after four source/context panes')
    canvas.save(target/'montage.png',pnginfo=png_info)
    operators=', '.join(sorted({r.get('operator','gaussian') for r in rows if r['control']=='natural'}))
    caption=f"Run {rows[0]['run_id']}; foreground {foreground_id}; {len(rows)} rows. Foreground operator: {operators}. Natural interventions and OOD controls labeled separately. Exact model previews; no generated saliency. Quality: {rows[0]['composite_quality_status']}. Checkpoint(s): {', '.join(sorted({r['checkpoint_sha256'] for r in rows}))}."
    (target/'caption.md').write_text(caption,encoding='utf-8')
    # PIL's PDF writer is present with Pillow; no external PDF renderer required.
    canvas.save(target/'montage.pdf','PDF',resolution=150.)
    return target


def ImageOpsContain(image,size):
    from PIL import ImageOps
    return ImageOps.contain(image.convert('RGB'),size,Image.Resampling.LANCZOS)
