"""Frozen three-method evaluation. Reuses Runner and mask compositor; never fits.

Legacy DPE has no historical backbone hash. Its frozen, explicitly configured
pair is usable for exploratory evaluation, with that limitation carried into
every result. It does not weaken the Lab's hash-bound DPE adapter.
"""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import random
import time

import numpy as np
import torch
from PIL import Image

from .common import ROOT, contained, pixel_hash, read_csv, sha256, write_csv, write_json
from .compositor import aligned, composite, foreground_layer, mask_overlay
from .data import ForegroundRepository, BackgroundRepository
from .model import ModelAdapter
from src.methods.dpe_head_reweighting import ensemble
from src.evaluation.dpe_export import predict_heads


MODELS = ('POTER', 'ERM_enhanced', 'DPE_legacy')
LEVELS = ('clean', 'mild', 'moderate', 'strong')


def select_cohort(rows, config, excluded=()):
    """Ten randomly chosen species/class, five random images/species; no scores."""
    rng = random.Random(config['sampling_seed'])
    count, species_n = config['foregrounds_per_class'], config['species_per_class']
    if count % species_n:
        raise ValueError('Equal images per species requires divisible class quota')
    excluded = set(excluded)
    suspect = set(config['known_mislabeled_species'])
    output = []
    for y in (0, 1):
        candidates = defaultdict(list)
        for r in sorted(rows, key=lambda r: int(r['bird_source_id'])):
            if (int(r['y']) == y and r['original_split'] == 'test'
                    and r['quality_status'] == 'verified_original_cub'
                    and r['foreground_id'] not in excluded
                    and r['species'] not in suspect):
                candidates[r['species']].append(r)
        eligible = sorted(s for s, group in candidates.items() if len(group) >= count // species_n)
        if len(eligible) < species_n:
            raise ValueError('Insufficient independently selected species')
        for species in sorted(rng.sample(eligible, species_n)):
            output += rng.sample(candidates[species], count // species_n)
    assert len({r['bird_source_id'] for r in output}) == 2 * count
    return output


def audit_preliminary(config, output):
    """Reconcile saved rows, not rounded summaries. Missing requested CSV explicit."""
    path = ROOT / config['preliminary_run']
    integrity = json.loads((path / 'integrity.json').read_text(encoding='utf-8'))
    for name, digest in integrity['core_sha256'].items():
        if sha256(path / name) != digest:
            raise ValueError(f'Preliminary integrity mismatch: {name}')
    rows = read_csv(path / 'predictions.csv')
    cfg = json.loads((path / 'run_config.json').read_text(encoding='utf-8'))
    if len({r['row_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate preliminary row IDs')
    natural = [r for r in rows if r['control'] == 'natural']
    records = []
    for severity in LEVELS:
        rr = [r for r in natural if r['severity'] == severity]
        groups = [[r for r in rr if int(r['group_id']) == g] for g in range(4)]
        records.append(dict(severity=severity, n=len(rr),
            accuracy=np.mean([r['true_y'] == r['pred_y'] for r in rr]).item(),
            wga=min(np.mean([r['true_y'] == r['pred_y'] for r in g]).item() for g in groups),
            landbird_accuracy=np.mean([r['true_y'] == r['pred_y'] for r in rr if r['true_y'] == '0']).item(),
            waterbird_accuracy=np.mean([r['true_y'] == r['pred_y'] for r in rr if r['true_y'] == '1']).item()))
    if len(rows) != 400 or len(natural) != 256:
        raise ValueError('Unexpected preliminary factorial coverage')
    ids = sorted({r['foreground_id'] for r in rows})
    repo = ForegroundRepository(ROOT / config['data_root'])
    mask_stats = []
    for fg_id in ids:
        rgba, fg = repo.get(fg_id)
        alpha = np.asarray(rgba.getchannel('A'))
        reference = foreground_layer(rgba, 0)
        for severity, sigma in config['severities'].items():
            layer = foreground_layer(rgba, sigma)
            assert layer.getchannel('A').tobytes() == reference.getchannel('A').tobytes()
            _, audit = composite(layer, Image.new('RGB', layer.size, (15, 40, 90)))
            assert audit['outside_support_exact'] and audit['opaque_foreground_exact']
        mask_stats.append(dict(foreground_id=fg_id, species=fg['species'],
            alpha_support_fraction=float((alpha > 0).mean()), alpha_mean=float(alpha.mean()/255),
            mask_sha256=fg['mask_sha256'], alpha_unchanged_all_severities=True))
    known = set(config['known_mislabeled_species'])
    metadata = read_csv(ROOT / config['data_root'] / 'metadata.csv')
    affected = [r for r in metadata if r['img_filename'].split('/')[0] in known]
    report = dict(source=str(path / 'predictions.csv'), csv_sha256=sha256(path / 'predictions.csv'),
        requested_filename='predictions(1).csv', requested_file_available=False,
        substitution='Existing immutable 400-row original preliminary run; no claim of auditing absent attachment',
        rows=len(rows), foregrounds=len(ids), species=len({r['species'] for r in rows}),
        background_ids=sorted({r['background_id'] for r in natural}),
        unique_raw_images=len({r['raw_input_sha256'] for r in rows}),
        unique_predictions=len({r['row_id'] for r in rows}),
        repeated_background_only_rows=sum(r['control'] == 'background_only' for r in rows),
        source_label_consistent=all(int(r['source_y']) == int(repo.by_id[r['foreground_id']]['y']) for r in rows),
        metrics=records, masks=mask_stats, severities=cfg['severities'],
        known_label_errors=dict(species=sorted(known),
            counts_by_split=dict(Counter(r['split'] for r in affected)),
            preliminary_affected=sum(r['species'] in known for r in rows),
            reference=str(Path(config['legacy_root']) / 'scripts/analysis/dpe_label_correction.py'),
            mechanism='Generator substring Tern matches Eastern / Western; official and corrected labels reported separately'))
    write_json(output / 'phase0_audit.json', report)
    write_csv(output / 'phase0_metrics.csv', records)
    text = ('# Phase 0 audit\n\nThe requested `predictions(1).csv` was not attached/found. '
        'Audited the preserved 400-row preliminary run instead, with core SHA-256 verification.\n\n'
        f"16 foreground IDs, 2 species, 4 natural backgrounds, {report['unique_raw_images']} unique raw images. "
        '256 natural composites + 64 background-only rows (4 unique images repeated across 16 birds) '
        '+ 64 neutral controls + 16 originals. No missing or duplicate prediction keys; repeated controls are not independent.\n\n'
        'Labels: 0=Landbird, 1=Waterbird; group=2*y+background (Land=0, Water=1). '
        'Gaussian sigma is 0, 1.5, 4, 9 pixels in the model crop. Premultiplied bilinear geometry is fixed; '
        'normalized Gaussian convolution operates on masked foreground RGB only. Alpha bytes are unchanged across '
        'severity/background; outside-support background and opaque foreground equality assertions passed for all 16 birds.\n\n'
        'Original CUB segmentation has been checksum verified. Pixel assertions do not establish semantic mask quality; '
        'segmentation artifacts remain possible. New representative masks/composites are saved for inspection.\n\n'
        'Known official label issue: 021.Eastern_Towhee, 088.Western_Meadowlark, '
        '102.Western_Wood_Pewee. The upstream Tern substring rule matches Eastern/Western. '
        f"Affected counts by split: {report['known_label_errors']['counts_by_split']}; preliminary affected rows=0. "
        'Official labels remain unchanged; corrected and species-excluded sensitivity analyses are secondary. '
        'The new 100-image replication excludes these known ambiguous species and the original 16 foregrounds, '
        'before any new predictions.\n\n'
        'Direct CSV recomputation:\n\n|Severity|n|Accuracy|WGA|Landbird|Waterbird|\n|---|---:|---:|---:|---:|---:|\n')
    for r in records:
        text += f"|{r['severity']}|{r['n']}|{r['accuracy']:.6f}|{r['wga']:.6f}|{r['landbird_accuracy']:.6f}|{r['waterbird_accuracy']:.6f}|\n"
    (output / 'phase0_audit.md').write_text(text, encoding='utf-8')
    return report, ids


class FrozenComparison:
    """Same input for all three; ERM and DPE share one exact feature extraction.

    Both CNNs stay loaded during the joint run. A single GPU batch evaluates
    POTER then the shared ERM backbone; this avoids redundant DPE CNN work.
    """
    def __init__(self, config, output):
        self.config = config
        self.device = config['device']
        self.poter = ModelAdapter(ROOT / config['poter_checkpoint'], self.device,
            metadata=ROOT / config['data_root'] / 'metadata.csv')
        legacy = Path(config['legacy_root'])
        checkpoint, heads_path = legacy / config['erm_checkpoint'], legacy / config['dpe_ensemble']
        if sha256(checkpoint) != config['expected_erm_sha256'] or sha256(heads_path) != config['expected_heads_sha256']:
            raise ValueError('Configured frozen legacy checkpoint hashes changed')
        self.erm = ModelAdapter(checkpoint, self.device, method='erm', model_id='ERM_enhanced')
        self.heads = ensemble.load(heads_path, dim=2048, count=15)
        metadata = read_csv(ROOT / config['data_root'] / 'metadata.csv')
        val = {r['img_filename'] for r in metadata if r['split'] == '1'}
        if set(self.heads['provenance']['training_pool_ids']) != val:
            raise ValueError('DPE head training pool is not exactly canonical validation; refusing possible test leakage')
        identity_path = legacy / 'outputs/backbone_provenance_audit/checkpoint_identity.json'
        identity = json.loads(identity_path.read_text(encoding='utf-8'))
        if identity['remote_sha256'] != sha256(checkpoint) or not identity['exact_match']:
            raise ValueError('ERM historical checkpoint identity audit disagrees')
        self.converted = dict(version=1, method='original-dpe', classes=[0,1], temperature=1.,
            prototypes=self.heads['prototypes'], distance_scales=self.heads['distance_scales'],
            normalize_embeddings=True)
        self.info = dict(POTER=self.poter.info(), ERM_enhanced=self.erm.info(),
            DPE_legacy=dict(model_id='DPE_legacy', seed=0, backbone_sha256=sha256(checkpoint),
                heads_sha256=sha256(heads_path), heads_path=str(heads_path), heads=15,
                aggregation='equal mean of 15 head probabilities; row-zscore ddof0; L2; IsoMax distance; temperature1',
                historical_pair_binding='Not hash-bound in legacy ensemble; explicitly frozen co-located pair, historical association not independently verified',
                training_pool='all 1199 canonical validation IDs; no test IDs',
                head_epochs=self.heads['selected_epochs'],
                training=self.heads['provenance']['settings']),
            comparison_limitations=[
                'ERM checkpoint is enhanced/class-balanced ERM, not a standard uniform ERM baseline.',
                'POTER seed1 versus ERM/DPE seed0; one available image backbone per method, no matched three-seed comparison.',
                'Legacy DPE head/backbone historical hash binding is absent; exploratory pair association only.',
                'Same ResNet50 architecture, evaluation images, preprocessing and FP32 device settings; training recipes differ.',
                'DPE scores are log mean probabilities, not raw CNN logits; cross-model logit magnitudes are not comparable.'])
        self.info['POTER']['model_seed'] = 1
        self.info['ERM_enhanced'].update(model_seed=0, training_recipe=json.loads(json.dumps(
            __import__('yaml').safe_load((legacy / 'outputs/backbone_provenance_audit/remote_config.yaml').read_text()))))
        write_json(output / 'checkpoint_audit.json', self.info)

    def predict(self, images):
        pot = self.poter.predict_batch(images, self.config['batch_size'])
        result = {'POTER': pot, 'ERM_enhanced': [], 'DPE_legacy': []}
        with torch.inference_mode():
            for start in range(0, len(images), self.config['batch_size']):
                previews = [self.erm.preview(im) for im in images[start:start+self.config['batch_size']]]
                tensors = torch.stack([self.erm.tensor(im) for im in previews])
                features = self.erm.model.extract_features(tensors.to(self.device))
                logits = self.erm.model[-1](features)
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite ERM logits')
                raw = features.cpu().numpy()
                probability = predict_heads(self.converted, raw, device=self.device)
                for name, z, probs in [('ERM_enhanced', logits.cpu().numpy(), logits.softmax(1).cpu().numpy()),
                    ('DPE_legacy', np.log(probability.clip(np.finfo(np.float32).tiny)), probability)]:
                    for preview, tensor, score, p in zip(previews, tensors, z, probs):
                        result[name].append(dict(logits=score.astype(float).tolist(),
                            probabilities=p.astype(float).tolist(), pred_y=int(p.argmax()),
                            model_input_sha256=pixel_hash(preview),
                            normalized_tensor_sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest()))
        for i in range(len(images)):
            if len({result[name][i]['normalized_tensor_sha256'] for name in MODELS}) != 1:
                raise ValueError('Models did not receive identical normalized input')
        return result

    def smoke(self, images, output):
        result = self.predict(images)
        raw = []
        with torch.inference_mode():
            for im in images:
                x = self.erm.tensor(self.erm.preview(im))[None].to(self.device)
                raw.append(self.erm.model.extract_features(x).cpu().numpy())
        raw = np.concatenate(raw)
        # Independent saved-ensemble reference, rather than testing only conversion.
        ref = ensemble.predict(self.heads, raw, device=self.device).mean(0)
        actual = np.array([r['probabilities'] for r in result['DPE_legacy']])
        max_error = float(abs(ref-actual).max())
        if max_error > 1e-4:
            raise ValueError('DPE adapter disagrees with legacy inference')
        repeat = self.predict(images)
        repeat_error = max(abs(np.array([r['probabilities'] for r in result[m]])-
            np.array([r['probabilities'] for r in repeat[m]])).max().item() for m in MODELS)
        if repeat_error > 1e-6:
            raise ValueError('Repeated inference changed')
        record = dict(images=len(images), models=list(MODELS), dpe_legacy_max_probability_error=max_error,
            repeat_max_probability_error=repeat_error, identical_input_tensors=True,
            poter_reference='Unmodified shared ModelAdapter / existing Runner; previous live HTTP parity verified',
            model_seeds={'POTER':1,'ERM_enhanced':0,'DPE_legacy':0})
        write_json(output / 'smoke_test.json', record)
        return record


def original_records(fg_rows, data_root, cohort):
    for fg in fg_rows:
        with Image.open(contained(data_root, fg['source_image_path'])) as im:
            image = im.convert('RGB')
        yield [image], [dict(cohort=cohort, image_id=fg['bird_source_id']+'_original',
            source_image_id=fg['bird_source_id'], foreground_id=fg['foreground_id'], species_id=fg['species'],
            true_y=int(fg['y']), original_group=int(fg['original_group']), background_id='original-'+fg['bird_source_id'],
            bg_type=('land','water')[int(fg['original_group'])%2], severity='original', sigma=0.,
            control='original', pair_index=-1, source_path=fg['source_image_path'],
            mask_sha256=fg['mask_sha256'], source_sha256=fg['source_sha256'],
            source_split=fg['original_split'], raw_input_sha256=pixel_hash(image),
            layer_sha256='not_applicable', alpha_sha256='not_applicable', invariance_passed=True)]


def intervention_records(cohort, config, output, *, save_examples=True):
    data_root = ROOT / config['data_root']
    fr, br = ForegroundRepository(data_root), BackgroundRepository(data_root)
    backgrounds = [(bg_id, *br.get(bg_id)) for bg_id in config['background_ids']]
    examples = {next(r['foreground_id'] for r in cohort if int(r['y']) == y) for y in (0,1)}
    for fg in cohort:
        rgba, _ = fr.get(fg['foreground_id'])
        original_images, original_metadata = next(original_records([fg],data_root,'replication100'))
        images, records = original_images, original_metadata
        reference_alpha = None
        for severity, sigma in config['severities'].items():
            layer = foreground_layer(rgba, sigma)
            alpha = pixel_hash(layer.getchannel('A'))
            if reference_alpha is None:
                reference_alpha = alpha
            if alpha != reference_alpha:
                raise ValueError('Gaussian changed alpha')
            base = dict(cohort='replication100', source_image_id=fg['bird_source_id'],
                foreground_id=fg['foreground_id'], species_id=fg['species'], true_y=int(fg['y']),
                original_group=int(fg['original_group']), severity=severity, sigma=sigma,
                source_path=fg['source_image_path'], mask_sha256=fg['mask_sha256'],
                source_sha256=fg['source_sha256'], source_split='test')
            for index, (bg_id, background, meta) in enumerate(backgrounds):
                image, audit = composite(layer, background)
                if meta['original_split'] != 'test':
                    raise ValueError('Cross-split background')
                records.append(dict(base,image_id=f"{fg['bird_source_id']}_{severity}_{bg_id}",
                    background_id=bg_id, bg_type=meta['bg_type'], control='natural', pair_index=index//2,
                    raw_input_sha256=pixel_hash(image), layer_sha256=audit['foreground_layer_sha256'],
                    alpha_sha256=alpha, invariance_passed=audit['opaque_foreground_exact'] and audit['outside_support_exact']))
                images.append(image)
                if severity == 'clean':
                    bg_image = aligned(background, layer.size)
                    records.append(dict(base,image_id=f"{fg['bird_source_id']}_background_{bg_id}",
                        true_y=None, background_id=bg_id, bg_type=meta['bg_type'], control='background_only',
                        pair_index=index//2, raw_input_sha256=pixel_hash(bg_image), layer_sha256='no_foreground',
                        alpha_sha256='no_foreground',invariance_passed=True))
                    images.append(bg_image)
            neutral, audit = composite(layer,Image.new('RGB',layer.size,(127,127,127)))
            images.append(neutral)
            records.append(dict(base,image_id=f"{fg['bird_source_id']}_{severity}_neutral",
                background_id='neutral-gray127',bg_type='neutral',control='foreground_only',pair_index=-1,
                raw_input_sha256=pixel_hash(neutral),layer_sha256=audit['foreground_layer_sha256'],
                alpha_sha256=alpha,invariance_passed=audit['opaque_foreground_exact'] and audit['outside_support_exact']))
        if save_examples and fg['foreground_id'] in examples:
            folder=output/'examples'/fg['foreground_id']
            folder.mkdir(parents=True,exist_ok=False)
            rgba.save(folder/'foreground.png')
            rgba.getchannel('A').save(folder/'mask.png')
            mask_overlay(rgba).save(folder/'mask_overlay.png')
            for image, record in zip(images, records):
                image.save(folder/(record['image_id']+'.png'))
        expected=1+len(config['severities'])*(len(backgrounds)+1)+len(backgrounds)
        assert len(images)==expected and len(records)==expected
        yield images, records


def evaluate_stream(model, stream, destination, *, total_sources, phase, output, predictor=None):
    start=time.perf_counter()
    writer=None
    rows=0
    buffer_images, buffer_records=[],[]
    def flush(handle):
        nonlocal writer,rows
        result=predictor(buffer_images,buffer_records) if predictor else model.predict(buffer_images)
        for name in MODELS:
            for record,pred in zip(buffer_records,result[name]):
                y=record['true_y']
                row=dict(record,source_y=int(record['original_group'])//2,
                    model_name=name,model_seed=1 if name=='POTER' else 0,
                    sampling_seed=model.config['sampling_seed'],pred_y=pred['pred_y'],
                    p_landbird=pred['probabilities'][0],p_waterbird=pred['probabilities'][1],
                    logit_landbird=pred['logits'][0],logit_waterbird=pred['logits'][1],
                    score_type='log_mean_probability' if name=='DPE_legacy' else 'cnn_logit',
                    is_correct=pred['pred_y']==y if y is not None else None,
                    model_input_sha256=pred['model_input_sha256'],
                    normalized_tensor_sha256=pred['normalized_tensor_sha256'])
                if writer is None:
                    writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader()
                writer.writerow(row); rows+=1
        handle.flush()
        buffer_images.clear();buffer_records.clear()
    with destination.open('x',encoding='utf-8',newline='') as handle:
        for count,(images,records) in enumerate(stream,1):
            buffer_images.extend(images);buffer_records.extend(records)
            if len(buffer_images)>=model.config['batch_size'] or count==total_sources:
                flush(handle)
            if count%50==0 or count==total_sources:
                status=dict(phase=phase,sources=count,total_sources=total_sources,rows=rows,
                    elapsed_seconds=time.perf_counter()-start)
                write_json(output/'progress.json',status)
                print(json.dumps(status),flush=True)
    return dict(sources=total_sources,predictions=rows,elapsed_seconds=time.perf_counter()-start,
        csv_sha256=sha256(destination))
