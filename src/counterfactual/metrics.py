"""Descriptive paired diagnostics; controls and unknown labels cannot be failures."""
from collections import defaultdict
import numpy as np


def truthy(value):
    return value in (True, 1, '1', 'True', 'true')


def eligible(r):
    return (r.get('status') == 'completed' and r.get('control') in ('natural','original')
            and truthy(r.get('known_label')) and truthy(r.get('valid_target')))


def paired_metrics(rows):
    groups = defaultdict(dict)
    for r in rows:
        if r.get('status') == 'completed' and r.get('control') == 'natural':
            key = (r['model_id'], r['source_bird_id'], r['severity'], str(r['pair_index']))
            if r['bg_type'] in groups[key]:
                raise ValueError('Ambiguous duplicate row in matched pair')
            groups[key][r['bg_type']] = r
    output = []
    for (model, bird, severity, pair), bg in groups.items():
        if set(bg) != {'land', 'water'}:
            continue
        land, water = bg['land'], bg['water']
        if (land['foreground_layer_sha256'] != water['foreground_layer_sha256']
                or land['source_split'] != water['source_split'] or land['true_y'] != water['true_y']):
            raise ValueError('Pair does not hold foreground/label/split fixed')
        known = eligible(land) and eligible(water)
        y = int(land['true_y']) if known else None
        margins = []
        for r in (land, water):
            scores = [float(r['logit_landbird']), float(r['logit_waterbird'])]
            margins.append(scores[y]-scores[1-y] if known else None)
        delta = (margins[y]-margins[1-y]) if known else None
        output.append(dict(model_id=model, source_bird_id=bird, severity=severity, pair_index=pair,
            operator=land.get('operator','gaussian'),operator_strength=land.get('operator_strength'),
            land_row_id=land['row_id'], water_row_id=water['row_id'],
            land_background_id=land['background_id'], water_background_id=water['background_id'],
            p_water_land=float(land['p_waterbird']), p_water_water=float(water['p_waterbird']),
            true_label_margin_land=margins[0], true_label_margin_water=margins[1],
            paired_flip=int(land['pred_y']) != int(water['pred_y']), delta_margin=delta,
            right_aligned_wrong_conflicting=bool(known and truthy(bg[['land','water'][y]]['is_correct'])
                                                and not truthy(bg[['land','water'][1-y]]['is_correct'])),
            seed=int(land['seed']), known_label=bool(known), score_type=land['score_type'],
            composite_quality_status=land['composite_quality_status']))
    clean = {(r['model_id'], r['source_bird_id'], r['pair_index']):r['delta_margin']
             for r in output if r['severity'] == 'clean'}
    for r in output:
        baseline = clean.get((r['model_id'], r['source_bird_id'], r['pair_index']))
        r['amplification'] = r['delta_margin']-baseline if baseline is not None and r['delta_margin'] is not None else None
    return output


def _cluster_ci(rows, field, seed, samples):
    clusters = defaultdict(list)
    for r in rows:
        if r.get(field) is not None:
            clusters[r['source_bird_id']].append(float(r[field]))
    values = np.array([np.mean(v) for v in clusters.values()])
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    estimates = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(samples)])
    return np.quantile(estimates, [.025, .975]).tolist()


def summarize(rows, pairs=None, *, seed=0, bootstrap_samples=1000):
    pairs = paired_metrics(rows) if pairs is None else pairs
    groups = defaultdict(list)
    for r in rows:
        groups[(r['model_id'], r['severity'], r['control'])].append(r)
    result = []
    clean = {(r['model_id'], r['source_bird_id'], r['background_id']): r
             for r in rows if r['severity'] == 'clean' and eligible(r)}
    for (model, severity, control), records in groups.items():
        valid = [r for r in records if eligible(r)]
        by_group = defaultdict(list)
        for r in valid:
            group = 2*int(r['true_y']) + (r['bg_type'] == 'water')
            by_group[group].append(truthy(r['is_correct']))
        group_stats = {str(g):dict(n=len(v), correct=sum(v), accuracy=float(np.mean(v))) for g, v in by_group.items()}
        p = [r for r in pairs if r['model_id'] == model and r['severity'] == severity] if control == 'natural' else []
        ds = [r['delta_margin'] for r in p if r['delta_margin'] is not None]
        amps = [r['amplification'] for r in p if r['amplification'] is not None]
        false_true = true_false = 0
        for r in valid:
            c = clean.get((model, r['source_bird_id'], r['background_id']))
            if c and control=='natural':
                false_true += not truthy(c['is_correct']) and truthy(r['is_correct'])
                true_false += truthy(c['is_correct']) and not truthy(r['is_correct'])
        result.append(dict(model_id=model, severity=severity, control=control, rows=len(records),
            score_type=records[0].get('score_type'),
            operator=records[0].get('operator','gaussian'),
            source_birds=len({r['source_bird_id'] for r in records}), valid_target_n=len(valid),
            accuracy=float(np.mean([truthy(r['is_correct']) for r in valid])) if valid else None,
            group_stats=group_stats, wga=min(v['accuracy'] for v in group_stats.values()) if len(by_group) == 4 else None,
            wga_status='complete' if len(by_group) == 4 else 'undefined/incomplete',
            D_s=float(np.mean(ds)) if ds else None, amplification=float(np.mean(amps)) if amps else None,
            D_s_ci95=_cluster_ci(p, 'delta_margin', seed, bootstrap_samples),
            amplification_ci95=_cluster_ci(p, 'amplification', seed, bootstrap_samples),
            bootstrap_unit='source_bird_id; pairs averaged within bird', bootstrap_samples=bootstrap_samples,
            paired_flips=sum(truthy(r['paired_flip']) for r in p),
            clean_correct_to_wrong=int(true_false), clean_wrong_to_correct=int(false_true),
            confusion_matrix=[[sum(int(r['true_y']) == y and int(r['pred_y']) == pred for r in valid)
                               for pred in (0, 1)] for y in (0, 1)],
            seed_level_std=None, seed_level_status='one fixed sampling seed; no independent model seed ensemble',
            analysis='retrospective_exploratory',
            controlled_claim_eligible=bool(valid and all(r['composite_quality_status'] == 'verified_original_cub' for r in valid))))
    return result


def annotate_failures(rows, pairs):
    p = {r[key]:r for r in pairs for key in ('land_row_id', 'water_row_id')}
    clean = {(r['model_id'], r['source_bird_id'], r['background_id']): r
             for r in rows if r['severity'] == 'clean' and eligible(r)}
    by_image = defaultdict(list)
    originals = {(r['model_id'],r['source_bird_id']):r for r in rows
                 if r.get('control')=='original' and r.get('status')=='completed'}
    for r in rows:
        by_image[(r['raw_input_sha256'], r['preprocess_fingerprint'])].append(r)
    output = []
    for original in rows:
        r = dict(original)
        pair = p.get(r['row_id'])
        c = clean.get((r['model_id'], r['source_bird_id'], r['background_id']))
        r.update(is_failure=eligible(r) and not truthy(r['is_correct']),
            paired_flip=bool(pair and pair['paired_flip']),
            aligned_right_conflicting_wrong=bool(pair and pair['right_aligned_wrong_conflicting']),
            paired_margin_gap=abs(pair['delta_margin']) if pair and pair['delta_margin'] is not None else None,
            clean_correct_degraded_wrong=bool(c and r['control']=='natural' and eligible(r) and truthy(c['is_correct']) and not truthy(r['is_correct'])),
            model_disagreement=len({v['model_id'] for v in by_image[(r['raw_input_sha256'], r['preprocess_fingerprint'])] if v['status']=='completed'})>1
                and len({int(v['pred_y']) for v in by_image[(r['raw_input_sha256'], r['preprocess_fingerprint'])]
                         if v['status'] == 'completed'}) > 1,
            confidence=max(float(r['p_landbird']), float(r['p_waterbird'])) if r['status'] == 'completed' else None)
        baseline=originals.get((r['model_id'],r['source_bird_id']))
        r['original_p_waterbird']=float(baseline['p_waterbird']) if baseline else None
        r['delta_p_waterbird_vs_original']=float(r['p_waterbird'])-float(baseline['p_waterbird']) if baseline and r['status']=='completed' else None
        r['original_to_counterfactual_flip']=bool(baseline and r.get('control')=='natural' and r['status']=='completed'
                                                  and int(baseline['pred_y'])!=int(r['pred_y']))
        output.append(r)
    return output
