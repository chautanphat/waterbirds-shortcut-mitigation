"""Constrained test-oracle threshold and FDAT feasibility. No model inference/training."""
import argparse
from datetime import datetime,timezone
import itertools
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from PIL import Image

from src.counterfactual.common import sha256,write_json,write_csv
from src.counterfactual.fdat_feasibility import (fdat,prediction_metrics,threshold_curve,constrained_candidates,
    choose_constrained,join_pairs,species_folds,choose_fdat)
from scripts.run_test_oracle_threshold_sweep import validate_natural,original_data,exact_candidates,MODELS


def load(config,out):
    for flag in ['allow_training','allow_new_model_inference','allow_segmentation_jobs']:
        if config[flag]:raise ValueError('Score-only study; training/inference/segmentation disabled')
    for name in ['outputs/mild_blur_v2','results/poter_failure','outputs/test_oracle_threshold_sweep','outputs/poter_threshold_probe']:
        folder=ROOT/name
        for filename,digest in json.loads((folder/'integrity.json').read_text())['sha256'].items():
            if sha256(folder/filename)!=digest:raise ValueError('Prior result changed: '+str(folder/filename))
    raw=pd.read_csv(ROOT/config['input'],float_precision='round_trip')
    metadata=pd.read_csv(ROOT/config['metadata'])
    natural,audit=validate_natural(raw,metadata)
    originals,status=original_data(ROOT/config['original_test'],MODELS,metadata)
    if set(originals)!=set(MODELS):raise ValueError('Stage A requires complete full-original scores for each method')
    audit.update(full_original_status=status,source_sha256=sha256(ROOT/config['input']),
        original_sha256=sha256(ROOT/config['original_test']),study_scope=config['study_scope'],
        checkpoint_metadata=json.loads((ROOT/'outputs/mild_blur_v2/checkpoint_audit.json').read_text()),
        no_training=True,no_model_loading=True,no_model_inference=True)
    if (out/'run_config.json').exists():
        prior=json.loads((out/'run_config.json').read_text())
        if prior['source_sha256']!=audit['source_sha256'] or prior['original_sha256']!=audit['original_sha256']:
            raise ValueError('Use new output directory for changed source data')
        if any(prior.get(key)!=value for key,value in config.items()):
            raise ValueError('Predeclared experiment configuration changed; create a new exploratory output directory')
    else:write_json(out/'run_config.json',dict(config,started_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=audit['source_sha256'],original_sha256=audit['original_sha256']))
    write_json(out/'input_audit.json',audit)
    # Recompute every historical anchor; discrepancies are reported, never forced.
    anchors=[]
    expected05={0.:86.,.5:83.,1.:74.,2.:58.};expected65={0.:91.,.5:89.,1.:87.,2.:83.}
    expected_oracle={0.:(.60,93.),.5:(.58,92.),1.:(.64,88.),2.:(.65,83.)}
    for sigma in config['primary_sigmas']:
        part=natural[(natural.model_name=='POTER')&(natural.sigma==sigma)]
        for t,expected in [(.5,expected05[sigma]),(.65,expected65[sigma])]:
            value=threshold_curve(part,[t]).iloc[0].to_dict()
            anchors.append(dict(dataset='natural_composites100',sigma=sigma,kind='fixed_reference',**value,
                historical_wga_pct=expected,match_to_reported_precision=round(value['wga_pct'],2)==expected))
        curve=threshold_curve(part,np.arange(5,96)/100)
        value=min(curve.to_dict('records'),key=lambda r:(-round(r['wga_pct'],12),-round(r['aa_pct'],12),round(abs(r['threshold']-.5),14),r['threshold']))
        threshold_expected,wga_expected=expected_oracle[sigma]
        anchors.append(dict(dataset='natural_composites100',sigma=sigma,kind='prior_grid_oracle',**value,
            historical_wga_pct=wga_expected,historical_threshold=threshold_expected,
            match_to_reported_precision=round(value['wga_pct'],2)==wga_expected and value['threshold']==threshold_expected))
    for t,wga_expected,aa_expected in [(.5,91.04,91.99),(.65,82.71,94.55)]:
        value=threshold_curve(originals['POTER'],[t]).iloc[0].to_dict()
        anchors.append(dict(dataset='full_original_test5794',sigma=0.,kind='fixed_reference',**value,
            historical_wga_pct=wga_expected,historical_aa_pct=aa_expected,
            match_to_reported_precision=round(value['wga_pct'],2)==wga_expected and round(value['aa_pct'],2)==aa_expected))
    pd.DataFrame(anchors).to_csv(out/'anchor_reproduction.csv',index=False)
    return raw,natural,originals,audit


def stage_a(config,out,natural,originals):
    grid=np.arange(1001,dtype=int)/1000;selected=[];all_candidates=[];curves={}
    for model in MODELS:
        full=originals[model];data=natural[natural.model_name==model]
        kinds={'grid_0.001':grid}
        if model in config['exact_boundary_models']:
            kinds['exact_finite_scores']=np.unique(np.r_[grid,exact_candidates(np.r_[full.p_waterbird,data.p_waterbird])])
        for kind,thresholds in kinds.items():
            curve=constrained_candidates(data,full,thresholds,config['blur_objective_sigmas'])
            baseline=float(curve.loc[curve.threshold==.5,'mean_blur_wga_pct'].iloc[0])
            unconstrained=choose_constrained(curve,100.)
            curves[model,kind]=curve
            for eps in config['epsilon_pp']:
                candidates=curve.copy();candidates.insert(0,'model_name',model)
                candidates.insert(1,'search',kind);candidates.insert(2,'epsilon_pp',eps)
                candidates['feasible']=candidates.original_clean_drop_pp<=eps+1e-10
                candidates['constraint_margin_pp']=eps-candidates.original_clean_drop_pp
                choice=choose_constrained(curve,eps)
                candidates['chosen']=candidates.threshold==choice['threshold']
                all_candidates.append(candidates)
                gain=choice['mean_blur_wga_pct']-baseline
                unconstrained_gain=unconstrained['mean_blur_wga_pct']-baseline
                selected.append(dict(model_name=model,search=kind,epsilon_pp=eps,**choice,
                    baseline_mean_blur_wga_pct=baseline,mean_blur_gain_pp=gain,
                    unconstrained_mean_blur_wga_pct=unconstrained['mean_blur_wga_pct'],
                    unconstrained_threshold=unconstrained['threshold'],
                    fraction_unconstrained_gain_recovered=gain/unconstrained_gain if unconstrained_gain>0 else None,
                    constraint_satisfied=choice['original_clean_drop_pp']<=eps+1e-10,
                    selection_scope='TEST ORACLE / IN-SAMPLE'))
            print(f'Stage A {model} {kind}: {len(curve)} score boundaries; no model loading',flush=True)
    pd.concat(all_candidates,ignore_index=True).to_csv(out/'constrained_global_thresholds.csv',index=False)
    chosen=pd.DataFrame(selected);chosen.to_csv(out/'constrained_global_selected.csv',index=False)
    # Nondominated frontier: original WGA and average blurred WGA, maximize both.
    frontier=[]
    for (model,kind),curve in curves.items():
        ordered=curve.sort_values(['original_wga_pct','mean_blur_wga_pct'],ascending=[False,False])
        maximum=-np.inf
        for _,r in ordered.iterrows():
            if r.mean_blur_wga_pct>maximum+1e-10:
                frontier.append(dict(model_name=model,search=kind,**r.to_dict()));maximum=r.mean_blur_wga_pct
    pd.DataFrame(frontier).to_csv(out/'global_pareto_frontier.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curve=curves['POTER','exact_finite_scores']
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    ax.scatter(curve.original_wga_pct,curve.mean_blur_wga_pct,s=8,alpha=.20,label='Finite-score thresholds')
    p=pd.DataFrame(frontier);p=p[(p.model_name=='POTER')&(p.search=='exact_finite_scores')].sort_values('original_wga_pct')
    ax.plot(p.original_wga_pct,p.mean_blur_wga_pct,'-',label='Pareto frontier')
    for t,label,color in [(.5,'fixed0.50','#437e4c'),(.65,'fixed0.65','#b64b39')]:
        r=curve[curve.threshold==t].iloc[0]
        ax.scatter(r.original_wga_pct,r.mean_blur_wga_pct,s=85,color=color,label=label)
    for r in chosen[(chosen.model_name=='POTER')&(chosen.search=='exact_finite_scores')].itertuples():
        ax.scatter(r.original_wga_pct,r.mean_blur_wga_pct,marker='x',s=100,label=f'epsilon{r.epsilon_pp:g}pp')
    ax.set(title='POTER constrained global threshold — TEST ORACLE / IN-SAMPLE',
        xlabel='Full original5794 WGA (%)',ylabel='Mean composite WGA at sigma0.5/1/2 (%)');ax.legend(fontsize=8)
    fig.savefig(out/'global_threshold_pareto.png',dpi=160,bbox_inches='tight');plt.close(fig)
    lines=['# Stage A — completed before FDAT','', '**TEST ORACLE / IN-SAMPLE; no held-out or safe-bound claim.**','',
        '|Epsilon pp|Search|Threshold|Original WGA %|Mean blur WGA %|Gain pp|Fraction unconstrained gain|',
        '|---|---|---|---|---|---|---|']
    for r in chosen[chosen.model_name=='POTER'].itertuples():
        lines.append(f'|{r.epsilon_pp:g}|{r.search}|{r.threshold:.12g}|{r.original_wga_pct:.4f}|{r.mean_blur_wga_pct:.4f}|{r.mean_blur_gain_pp:+.4f}|{r.fraction_unconstrained_gain_recovered:.4f}|')
    lines+=['','Objective uses only sigma0.5/1/2 with equal weight. Original full5794 WGA constrains selection. '
        'Tie: original WGA, mean blur AA, proximity to0.5, lower t. Epsilon is a deterministic inspected-test bound, not a statistical safety guarantee.',
        '',f'Predeclared "most" means >={100*config["global_most_gain_fraction"]:g}% of unconstrained global gain at epsilon<=2pp. '
        'This is a research triage definition, not a publication criterion. Stage B is still executed as requested.']
    (out/'stage_a_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return chosen


def audit_original_pairs(config,out,raw):
    """Pixel identity audit only; never generates a new prediction or saves an image."""
    data_root=ROOT/config['data_root']
    manifest=pd.read_csv(data_root/'foregrounds_manifest.csv').set_index('bird_source_id')
    # pandas type inference may use ints; normalize IDs explicitly.
    manifest.index=manifest.index.astype(int)
    gray=raw[(raw.model_name=='POTER')&(raw.control=='foreground_only')&(raw.sigma==0.)]
    originals=raw[(raw.model_name=='POTER')&(raw.control=='original')]
    if len(gray)!=100 or len(originals)!=100:raise ValueError('Expected100 candidate original/clean-gray controls')
    audits=[]
    for r in originals.itertuples():
        info=manifest.loc[int(r.source_image_id)]
        if sha256(data_root/info.rgba_path)!=info.rgba_sha256 or sha256(data_root/info.mask_path)!=info.mask_sha256:
            raise ValueError('Source foreground/mask asset changed')
        if sha256(data_root/info.source_image_path)!=r.source_sha256:raise ValueError('Original source changed')
        with Image.open(data_root/info.source_image_path) as im:full=im.convert('RGB')
        with Image.open(data_root/info.rgba_path) as im:rgba=im.convert('RGBA')
        # Exact geometry used by foreground_layer at sigma0 versus Runner original preview.
        layer=rgba.convert('RGBa').resize((256,256),Image.Resampling.BILINEAR).convert('RGBA').crop((16,16,240,240))
        original=full.resize((256,256),Image.Resampling.BILINEAR).crop((16,16,240,240))
        opaque=np.asarray(layer.getchannel('A'))==255
        a,b=np.asarray(layer.convert('RGB'))[opaque],np.asarray(original)[opaque]
        identical=bool(len(a)>0 and np.array_equal(a,b))
        source_mask=np.asarray(rgba.getchannel('A'))>0
        source_rgb_exact=bool(full.size==rgba.size and np.array_equal(
            np.asarray(full)[source_mask],np.asarray(rgba.convert('RGB'))[source_mask]))
        partner=gray[gray.source_image_id==r.source_image_id].iloc[0]
        audits.append(dict(source_image_id=int(r.source_image_id),foreground_id=r.foreground_id,
            species_id=r.species_id,mask_sha256=info.mask_sha256,
            same_original_size=full.size==rgba.size,opaque_pixels=int(len(a)),
            opaque_rgb_exact=identical,source_masked_rgb_exact=source_rgb_exact,
            opaque_pixel_equal_fraction=float(np.all(a==b,axis=1).mean()) if len(a) else None,
            opaque_max_absolute_difference=int(abs(a.astype(int)-b.astype(int)).max()) if len(a) else None,
            original_input_sha256=r.model_input_sha256,gray_input_sha256=partner.model_input_sha256,
            eligible_original_pair=identical and source_rgb_exact and full.size==rgba.size,
            reason='Original HF benchmark JPEG versus gray composited from original-CUB RGB; accept only exact opaque RGB identity'))
    pd.DataFrame(audits).to_csv(out/'original_pair_identity_audit.csv',index=False)
    valid={r['source_image_id'] for r in audits if r['eligible_original_pair']}
    # Count and checksum every existing official mask without segmentation or inference.
    mask_missing=[];mask_invalid=[]
    for source,info in manifest.iterrows():
        mask=data_root/info.mask_path
        if not mask.is_file():mask_missing.append(int(source))
        elif sha256(mask)!=info.mask_sha256:mask_invalid.append(int(source))
    coverage=dict(registered_foregrounds=len(manifest),mask_files_present=len(manifest)-len(mask_missing),
        mask_hash_failures=mask_invalid,missing_mask_ids=mask_missing,
        cached_gray_foregrounds=100,cached_original_pair_candidates=100,
        cached_exact_original_pairs=len(valid),exact_original_source_ids=sorted(valid),
        all5794_original_gray_predictions_available=False,
        full_original_FDAT_WGA='BLOCKED: no5794 exact original/gray prediction pairs',
        no_new_inference=True,no_masks_generated=True)
    write_json(out/'original_coverage.json',coverage)
    return valid,coverage


def stage_b(config,out,raw,chosen):
    from scripts.threshold_fdat_stages import prototype
    return prototype(config,out,raw,chosen,audit_original_pairs)


def stage_c(config,out,raw,originals,chosen):
    from scripts.threshold_fdat_stages import transfer
    return transfer(config,out,raw,originals,chosen)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','A','B','C','report'],default='all')
    args=parser.parse_args();config=json.loads((ROOT/'configs/threshold_algorithm_feasibility.json').read_text())
    out=ROOT/config['output_root'];out.mkdir(parents=True,exist_ok=True)
    raw,natural,originals,audit=load(config,out)
    if args.stage in ['all','A']:chosen=stage_a(config,out,natural,originals)
    else:chosen=pd.read_csv(out/'constrained_global_selected.csv',float_precision='round_trip')
    if args.stage in ['all','B']:stage_b(config,out,raw,chosen)
    if args.stage in ['all','C']:stage_c(config,out,raw,originals,chosen)
    if args.stage in ['all','report']:
        from scripts.report_threshold_algorithm_feasibility import report
        report(config,out,raw,natural,originals,audit,chosen)


if __name__=='__main__':main()
