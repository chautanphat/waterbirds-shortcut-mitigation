"""Honest oracle/retrospective feasibility reporting; no classifier execution."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.counterfactual.common import sha256,write_json,write_csv

ROOT=Path(__file__).resolve().parents[1]


def table(headers,rows):
    return '\n'.join(['|'+ '|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
        ['|'+'|'.join(str(x).replace('|','/') for x in row)+'|' for row in rows])


def figures(config,out,grid,summary,folds,baselines,selected,coverage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    if not coverage['cached_exact_original_pairs']:
        axs[0].text(.5,.5,'FDAT original-clean Pareto: BLOCKED\n0/100 cached original-gray candidates pass identity\n0/5794 full-original FDAT predictions\nNo imputed clean-WGA point',ha='center',va='center',transform=axs[0].transAxes)
        axs[0].set(title='True original-clean FDAT performance unavailable',xlabel='Matched-original WGA (%)',ylabel='Mean blur WGA (%)',xlim=(0,100),ylim=(0,100))
    else:
        p=grid[(grid.model_name=='POTER')&(grid.variant=='matched_gray')&grid.sigma.isin(config['blur_objective_sigmas'])]
        means=p.groupby(['lambda_','t_max']).agg(blur=('wga_pct','mean'),clean=('matched_original_wga_pct','first'))
        axs[0].scatter(means.clean,means.blur)
        axs[0].set(title=f'Matched-original subset n={coverage["cached_exact_original_pairs"]}; oracle only',xlabel='Matched-original subset WGA (%)',ylabel='Mean blur WGA (%)')
    m=grid[(grid.model_name=='POTER')&(grid.variant=='matched_gray')]
    points=[]
    for (lam,cap),part in m.groupby(['lambda_','t_max']):
        points.append(dict(lam=lam,cap=cap,clean=part[part.sigma==0.].wga_pct.iloc[0],
            blur=part[part.sigma.isin(config['blur_objective_sigmas'])].wga_pct.mean()))
    p=pd.DataFrame(points);axs[1].scatter(p.clean,p.blur,c=p.lam,cmap='viridis',s=45,label='FDAT oracle candidates')
    uncapped=p[(p.lam==1)&(p.cap==1)].iloc[0]
    axs[1].scatter(uncapped.clean,uncapped.blur,marker='*',s=170,color='#b64b39',label='lambda1/cap1 ablation')
    for name,marker in [('fixed05','o'),('fixed065','s'),('foreground_only_classifier','x'),('global_constrained_eps2','^')]:
        rows=baselines[(baselines.model_name=='POTER')&(baselines.method==name)]
        axs[1].scatter(rows[rows.sigma==0].wga_pct.iloc[0],rows[rows.sigma.isin(config['blur_objective_sigmas'])].wga_pct.mean(),marker=marker,s=85,label=name)
    axs[1].set(title='Composite-clean proxy: different from original-clean',xlabel='Composite sigma0 WGA (%)',ylabel='Mean composite blur WGA (%)');axs[1].legend(fontsize=7)
    fig.suptitle('POTER oracle-mask feasibility — TEST ORACLE / IN-SAMPLE; no clean-original imputation')
    fig.savefig(out/'fdat_pareto.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    colors={'fdat':'#b64b39','fixed05':'#2878a4','foreground_only_classifier':'#437e4c','global_constrained_transfer_eps2':'#8c6e35'}
    for method,color in colors.items():
        part=summary[(summary.model_name=='POTER')&(summary.method==method)&summary.sigma.isin(config['primary_sigmas'])].sort_values('sigma')
        axs[0].plot(part.sigma,part.landbird_accuracy_pct,'-o',label=method,color=color)
        axs[1].plot(part.sigma,part.waterbird_accuracy_pct,'-o',label=method,color=color)
    for ax,title in zip(axs[:2],['Landbird correctness','Waterbird correctness']):
        ax.set(title=title,xlabel='Sigma',ylabel='Accuracy (%)',ylim=(0,103),xticks=config['primary_sigmas']);ax.legend(fontsize=7)
    data=summary[(summary.model_name=='POTER')&(summary.method=='fdat')&summary.sigma.isin(config['primary_sigmas'])].sort_values('sigma')
    axs[2].plot(data.sigma,data.wrong_to_correct,'-o',label='wrong→correct')
    axs[2].plot(data.sigma,data.correct_to_wrong,'-s',label='correct→wrong')
    axs[2].plot(data.sigma,data.prediction_flips_vs05,':',label='all flips')
    axs[2].set(title='FDAT OOF flips vs fixed0.5',xlabel='Sigma',ylabel='Composite rows (400/sigma)',xticks=config['primary_sigmas']);axs[2].legend(fontsize=8)
    fig.suptitle('Retrospective species-blocked predictions; cap1 ablation allowed in tuning; not independent validation')
    fig.savefig(out/'fdat_classwise.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    part=folds[(folds.model_name=='POTER')&(folds.method=='fdat')&(folds.sigma==2.)]
    ax.bar(part.fold,part.wga_gain_vs_fixed05_pp);ax.axhline(8,color='gray',ls='--',label='8pp triage line')
    ax.set(title='POTER sigma2 gain on each held-out species fold (retrospective)',xlabel='Fold',ylabel='WGA change vs0.5 (pp)',xticks=range(config['folds']));ax.legend()
    fig.savefig(out/'fdat_fold_gains.png',dpi=160,bbox_inches='tight');plt.close(fig)


def report(config,out,raw,natural,originals,audit,chosen):
    grid=pd.read_csv(out/'fdat_grid.csv');summary=pd.read_csv(out/'fdat_transfer_summary.csv')
    folds=pd.read_csv(out/'fdat_transfer_folds.csv');tuning=pd.read_csv(out/'fdat_transfer_tuning.csv')
    bootstrap=pd.read_csv(out/'fdat_transfer_bootstrap_ci.csv');species=pd.read_csv(out/'fdat_transfer_species.csv')
    baselines=pd.read_csv(out/'fdat_reference_baselines.csv');oof=pd.read_csv(out/'fdat_transfer_predictions.csv')
    coverage=json.loads((out/'original_coverage.json').read_text());identity=json.loads((out/'identity_control.json').read_text())
    anchors=pd.read_csv(out/'anchor_reproduction.csv')
    anchor_status='All14 anchors match reported precision.' if anchors.match_to_reported_precision.all() else 'DISCREPANCIES PRESENT: inspect anchor_reproduction.csv; old numbers were not forced.'
    exact=chosen[(chosen.model_name=='POTER')&(chosen.search=='exact_finite_scores')].sort_values('epsilon_pp')
    point=summary[(summary.model_name=='POTER')&(summary.method=='fdat')&(summary.sigma==2.)].iloc[0]
    fgpoint=summary[(summary.model_name=='POTER')&(summary.method=='foreground_only_classifier')&(summary.sigma==2.)].iloc[0]
    potfolds=folds[(folds.model_name=='POTER')&(folds.method=='fdat')&(folds.sigma==2.)]
    gain=point.wga_gain_vs_fixed05_pp;passing=int((potfolds.wga_gain_vs_fixed05_pp>=config['fdat_transfer_required_gain_pp']-1e-10).sum())
    ci=bootstrap[(bootstrap.model_name=='POTER')&(bootstrap.sigma==2.)&(bootstrap.comparison=='FDAT_minus_fixed05')&(bootstrap.metric=='wga_pct')].iloc[0]
    pot_species=species[(species.model_name=='POTER')&(species.sigma==2.)]
    positives=pot_species[pot_species.net_corrected_rows>0]
    top2=float(positives.net_corrected_rows.nlargest(2).sum()/positives.net_corrected_rows.sum()) if len(positives) else None
    most=bool((exact.fraction_unconstrained_gain_recovered>=config['global_most_gain_fraction']).any())
    blur_pass=bool(gain>=config['fdat_transfer_required_gain_pp'] and passing>=config['transfer_required_folds'])
    full_protection_measured=coverage['all5794_original_gray_predictions_available']
    verdict='GLOBAL-THRESHOLD SUFFICIENT' if most else 'INCONCLUSIVE' if blur_pass and not full_protection_measured else 'FDAT NOT PROMISING'
    decision=dict(executive_verdict=verdict,stage_A=dict(global_sufficient=most,
        most_gain_definition=config['global_most_gain_fraction'],scope='TEST ORACLE / IN-SAMPLE'),
        stage_B=dict(verdict='INCONCLUSIVE' if blur_pass and not full_protection_measured else 'NOT PROMISING',
            sigma2_oof_wga_gain_pp=gain,folds_meeting8pp=passing,folds_total=config['folds'],
            clean_original_loss_criterion_measurable=full_protection_measured,
            clean_original_loss_criterion_passed=None,
            incremental_wga_vs_gray_only_pp=float(point.wga_pct-fgpoint.wga_pct),
            positive_gain_species=len(positives),top2_positive_species_share=top2,
            uncapped_ablation_included_in_transfer=True),
        stage_C_deployable_publishable='BLOCKED',training=False,new_model_inference=False,
        recommended_next_action='Stop FDAT-specific method/training expansion. If continuing, obtain genuinely paired original-gray scores and compare the already-fixed AND and gray-only rules before investing in algorithm changes.')
    write_json(out/'decision.json',decision)
    figures(config,out,grid,summary,folds,baselines,chosen,coverage)
    arows=[[f'{r.epsilon_pp:g}',f'{r.threshold:.12g}',f'{r.original_wga_pct:.4f}',f'{r.original_clean_drop_pp:.4f}',
            f'{r.mean_blur_wga_pct:.4f}',f'{r.mean_blur_gain_pp:+.4f}',f'{100*r.fraction_unconstrained_gain_recovered:.2f}%'] for r in exact.itertuples()]
    transfer_rows=[]
    for sigma in config['primary_sigmas']:
        for method in ['fixed05','fixed065','global_constrained_transfer_eps2','fdat','foreground_only_classifier']:
            r=summary[(summary.model_name=='POTER')&(summary.sigma==sigma)&(summary.method==method)].iloc[0]
            transfer_rows.append([sigma,method,f'{r.wga_pct:.2f}',f'{r.aa_pct:.2f}',f'{r.landbird_accuracy_pct:.2f}',f'{r.waterbird_accuracy_pct:.2f}',f'{r.wga_gain_vs_fixed05_pp:+.2f}'])
    foldrows=[[r.fold,f'{r.wga_pct:.2f}',f'{r.wga_gain_vs_fixed05_pp:+.2f}',f'{r.selected_lambda:g}',f'{r.selected_t_max:g}',r.matched_original_count] for r in potfolds.itertuples()]
    modelrows=[]
    for model in ['POTER','ERM_enhanced','DPE_legacy']:
        rows=summary[(summary.model_name==model)&(summary.sigma==2.)].set_index('method')
        modelrows.append([model,*[f'{rows.loc[m,"wga_pct"]:.2f}' for m in ['fixed05','fdat','foreground_only_classifier','global_constrained_transfer_eps2']]])
    mismatches={str(s):int((p.fdat_pred!=p.foreground_only_classifier_pred).sum()) for s,p in oof[oof.model_name=='POTER'].groupby('sigma')}
    pp=grid[(grid.model_name=='POTER')&(grid.variant=='matched_gray')]
    means=pp[pp.sigma.isin(config['blur_objective_sigmas'])].groupby(['lambda_','t_max']).wga_pct.mean()
    bestparam=means.idxmax();proto=pp[(pp.lambda_==bestparam[0])&(pp.t_max==bestparam[1])]
    primary_proto=pp[pp.t_max<1.]
    capmeans=primary_proto[primary_proto.sigma.isin(config['blur_objective_sigmas'])].groupby(['lambda_','t_max']).wga_pct.mean()
    capped=capmeans.idxmax();cap_sigma2=primary_proto[(primary_proto.lambda_==capped[0])&(primary_proto.t_max==capped[1])&(primary_proto.sigma==2.)].iloc[0]
    shuffled=grid[(grid.model_name=='POTER')&(grid.variant=='shuffled_gray_negative_control')&(grid.lambda_==bestparam[0])&(grid.t_max==bestparam[1])&(grid.sigma==2.)].iloc[0]
    # Use actual measured earlier inference timing only for a clearly labeled budget proxy.
    timing=json.loads((ROOT/'outputs/poter_threshold_probe/validation_inference.json').read_text())
    proxy_minutes=5794/1199*timing['elapsed_seconds']/60
    deployment=f'''# Deployment and compute audit — proposal only

FDAT needs a real segmentation mask and a second classifier input. Existing dataset masks are oracle CUB masks, not a tested deployable segmenter. All5794 mask files exist and hashes match; no mask model was run and bad-mask robustness is unmeasured. Natural photographs would need a separately validated segmentation system, with its own error and latency budget.

The inference formula accepts only p_full,p_gray,lambda,cap. Labels/species/background/sigma/condition/split are used only in retrospective evaluation and fold design, never in fdat(). Existing POTER weights/OT remain unchanged. This is a post-hoc proposal, not an OT-learning improvement or a novelty claim.

One additional foreground-only POTER forward is normally needed per input. With full scores already cached, completing a5794-original gray bank would require5794 new gray forwards, not redoing the full predictions. The earlier1199-original POTER validation run took{timing['elapsed_seconds']:.2f}s including I/O/head checks. Scaling that measurement gives **{proxy_minutes:.2f} GPU-minutes** as a throughput proxy, not a measured full-gray job or guaranteed end-to-end runtime. CPU composition, mask QA, hashing, model load, concurrency and export overhead remain unmeasured; no such job was launched. No GPU price estimate is asserted.

Online serial full+gray inference is roughly two network forwards, plus segmentation/compositing; latency must be benchmarked. The same model weights can be reused sequentially, avoiding a second backbone copy. Sequential execution can reuse activation storage; stacking both views doubles the image batch and can raise peak memory. No new latency/VRAM benchmark was run. FP32 input tensor storage is about0.574MiB/view (3×224×224×4 bytes), about9.19MiB for batch16, excluding model/activations/segmentation.

**Clean coverage:** cached gray scores exist for100 composite foregrounds only. All100 candidate original/gray pairs fail the opaque-foreground pixel equality test, despite matching source ID/mask geometry. Original full images are benchmark/HF JPEG pixels, while cached gray used original-CUB RGB. Reusing those scores would silently change the foreground. Full-original FDAT WGA is BLOCKED; paired-original FDAT count is0/100. To create valid pairs later, start from each exact original benchmark image's pixels and its aligned mask using the frozen preprocessing, verify foreground invariance and provenance, then infer gray once. Do not reuse CUB-RGB gray scores or approximate geometry.

The gray cue also changes scene statistics and removes context; failure can arise from segmentation artifacts or lost class evidence. No controlled mask perturbation was evaluated because it would require additional images/inference. Comparable cache-based tests on ERM/DPE show mixed gains and provide no POTER-specific novelty evidence. Legacy DPE historical backbone/head binding remains unverified.

Budget recommendation is conditional: before any expensive training or segmentation, discuss a bounded original-gray inference check using existing masks, compare the fixed lambda1/cap1 AND rule against gray-only and fixed0.5, and retain all original-WGA/class trade-offs. A valid contribution still requires independent development data, realistic masks, untouched final evaluation and eventually more checkpoints.
'''
    (out/'deployment_audit.md').write_text(deployment,encoding='utf-8')
    audit_text=f'''# Audit: constrained threshold and FDAT feasibility

Sources: `{config['input']}` (9000 rows, SHA{audit['source_sha256']}) and `{config['original_test']}` (17382 rows, SHA{audit['original_sha256']}). Prior outputs and the validation-only threshold lock were hash-verified and left unchanged. Full group/probability/logit audits are in input_audit.json. Canonical0=Landbird,1=Waterbird; natural group=2*y+assigned background, original group=official original_group. Official labels are retained, including known original-label limitations documented previously.

Natural6000 rows=3 checkpoints×5 sigmas×100 foregrounds×4 backgrounds. Gray1500 rows=3×5×100. Pair keys model/seed/source foreground/sigma, with exact layer/alpha/mask/source hashes checked. Each natural key has exactly4 backgrounds and exactly1 gray counterpart. Source sampling20 species/100 foregrounds is fixed; no new image/cohort selection. Background-only1200 and originals300 do not enter natural-composite WGA. Full-original5794/model is used for Stage A; original100 controls do not replace it.

Identity control: lambda0 exactly reproduces >=0.5 for{identity['lambda_zero_predictions_checked']} tested candidate rows. Negative control is a label/species-independent derangement of foreground IDs; same donor across backgrounds/severities, with no self matches. Shuffled cue is diagnostic only and never included in FDAT parameter selection. Inference function has only4 score/parameter inputs.

All5794 existing mask hashes pass, but exact original-gray pair coverage is0/100 cached candidates and0/5794 full originals. Original pixel audit rejects every candidate; source ID and image dimensions are insufficient. No full-original FDAT predictions are imputed. See original_pair_identity_audit.csv/original_coverage.json.

Stage A threshold grid0..1 step0.001, plus16583 finite-boundary candidates for POTER (secondary ERM/DPE grid only). Epsilon0/1/2pp on inspected originals is deterministic feasibility, not a confidence guarantee. Capped FDAT grid20 candidates plus5 optional cap1 ablations was declared before transfer. Extended25-candidate transfer allowed that ablation; all POTER folds selected it. Main capped oracle and uncapped ablation are reported separately. No claim that86% sigma2 is a validated capped-rule result.

Five folds have2 species/class and20 source foregrounds each. A species/foreground never crosses a fold. Global comparator fitting excludes all held-out species from full-original clean-constraint data too. Parameter selection uses K−1 fold composites; evaluation uses the remaining species. All tuning/held-out memberships/denominators are retained. This is retrospective parameter-transfer sanity checking on a repeatedly inspected test cohort, not an untouched benchmark or unseen-species backbone-training experiment.

Uncertainty:2000 class-stratified species-bootstrap draws of fixed pooled OOF predictions, all foreground/background repeats retained. Parameters/folds are not re-fit inside bootstrap; training folds overlap. CIs are descriptive conditional uncertainty, not unbiased generalization significance. No new checkpoints, CNN inference, segmentation, OT updates or generative images.
'''
    (out/'audit.md').write_text(audit_text,encoding='utf-8')
    text=f'''# Constrained threshold and FDAT feasibility

## Executive verdict: **{verdict}**

Global thresholds preserving original-clean WGA recover only a minority of the available blur oracle gain. Extra gray-view scores substantially improve retrospective composite performance. **FDAT does not yet add WGA beyond the simpler foreground-only classifier for POTER, and original-clean protection is unmeasurable.** Stop FDAT-specific algorithm/training expansion at this point; the next informative check is genuine original-gray pairs, not more hyperparameter searching or retraining.

{table(['Decision','Result','Reason'],[['A: constrained global','Not sufficient under declared80% triage','epsilon2 recovers '+f'{100*exact.fraction_unconstrained_gain_recovered.max():.2f}% of unconstrained blur gain'],['B: FDAT prototype','INCONCLUSIVE','blur gate passes; clean-original gate BLOCKED; no added POTER WGA over gray-only'],['C: deployable/publishable','BLOCKED','independent dev/final test, original gray pairs, realistic masks and checkpoint replication absent']])}

Every optimized result below is **TEST ORACLE / IN-SAMPLE** or **retrospective species-blocked transfer**. None is an unbiased gain on a fresh test set. POTER's original objective concerns subgroup/spurious-correlation robustness and label noise; foreground blur/composition is an additional conditional shift.

## Reproduced historical anchors

POTER fixed0.50 natural-composite WGA86/83/74/58% at sigma0/.5/1/2; fixed0.65 gives91/89/87/83%. Prior per-severity grid oracles0.60/0.58/0.64/0.65 give93/92/88/83%. On5794 originals,0.50 WGA91.04%/AA91.99%;0.65 WGA82.71%/AA94.55% (WGA−8.33pp). These checks were recomputed from unchanged CSVs, not forced by editing data. **{anchor_status}** Exact recomputation and reported-precision comparisons are in anchor_reproduction.csv. Clean composite sigma0 and original benchmark photographs remain distinct distributions with overlapping source birds.

## Stage A: constrained global threshold

Maximize mean composite WGA at sigma0.5/1/2 subject to full-original WGA loss<=epsilon. Tie original WGA, mean blur AA, proximity to0.5, lower threshold. Exact finite-score decisions use sorted boundaries plus adjacent float64 intervals; the0.001 grid is reported separately. POTER exact selections:

{table(['Eps pp','Threshold','Original WGA %','Original loss pp','Mean blur WGA %','Blur gain pp','Fraction unconstrained gain'],arows)}

The baseline mean blur WGA is{exact.baseline_mean_blur_wga_pct.iloc[0]:.4f}%. The unrestricted finite-score global optimum is{exact.unconstrained_mean_blur_wga_pct.iloc[0]:.4f}%; preserving original-clean WGA within2pp recovers{100*exact.fraction_unconstrained_gain_recovered.max():.2f}% of that gain. Our predeclared "most" criterion is80%; it is not met. This makes additional cue feasibility worth testing, without establishing a safe/validated constraint. `constrained_global_thresholds.csv` retains every epsilon/candidate and all per-sigma/class/group/AA/error/CFPS metrics. `stage_a_report.md` was written before Stage B.

## Stage B: FDAT candidate, ablations and oracle results

`d=max(0,p_full−p_gray); t(x)=min(t_max,0.5+lambda*d); yhat=int(p_full>=t(x))`.

Inference reads only scores and fixed parameters. True label/species/group/sigma/original/composite/split never enter the formula. A valid mask and one extra gray prediction are real input dependencies. Twenty capped candidates (lambda0/.25/.5/.75/1; cap.55/.60/.65/.70), plus5 optional cap1 ablations, are scored. Identity lambda0 is exact for every row. `fdat_paired_predictions.csv` contains150000 matched candidate decisions; `pair_identity_audit.csv` identifies the6000 actual natural-gray pairs. Shuffled-gray aggregate controls and the foreground-level permutation are retained.

POTER best extended-grid oracle is lambda{bestparam[0]:g}/cap{bestparam[1]:g}; mean blur WGA{means.max():.2f}%. Main capped-grid oracle is lambda{capped[0]:g}/cap{capped[1]:g}, with sigma2 WGA{cap_sigma2.wga_pct:.2f}%. The selected extended-grid sigma2 WGA is{proto[proto.sigma==2.].wga_pct.iloc[0]:.2f}%; using the same parameters with shuffled gray gives{shuffled.wga_pct:.2f}%. This supports the value of matched foreground scores in this cohort, while the shuffled rule can itself have threshold effects. Negative controls were never tuning candidates.

## Stage C: retrospective species-blocked transfer and uncertainty

Five deterministic folds,2 species/class/fold,20 foregrounds/fold. Each source's backgrounds/severities remain together. Tune on4 folds by mean WGA at sigma.5/1/2; tie composite-clean WGA, mean blur AA, smaller lambda/cap. Global comparators tune on the same training species and constrain original-clean scores only after removing every held-out species from the original fitting pool. Both use test-derived data retrospectively; hypothesis/design already saw this cohort.

All POTER folds chose lambda1/cap1 **with the optional cap1 ablation allowed**. On the observed POTER scores, this gives the binary AND of full>=0.5 and gray>=0.5. In general cap1 is not literally infinite: if p_full=1 and p_gray<0.5, clipping t to1 and the >=tie keep Waterbird, unlike AND. `and_rule_equivalence.csv` records this boundary check for every model/sigma. It is a simple two-view decision, not established adaptive calibration. The primary capped-only oracle remains separate; no primary-capped-only OOF performance is asserted.

{table(['Sigma','OOF strategy','WGA %','AA %','Landbird %','Waterbird %','WGA gain pp'],transfer_rows)}

At sigma2 the pooled OOF gain is{gain:+.2f}pp; descriptive species-bootstrap95% interval[{ci.ci95_low_pp:+.2f},{ci.ci95_high_pp:+.2f}]pp. {passing}/{config['folds']} held-out folds exceed8pp. Fold WGAs are not averaged to obtain pooled WGA; group counts are pooled first.

{table(['Fold','FDAT WGA sigma2 %','Gain vs0.5 pp','Lambda','Cap','Matched original count'],foldrows)}

Gain appears in{len(positives)} of20 species at sigma2; the top2 positive species contribute{100*top2:.2f}% of positive net corrections, so gains are not entirely one/two species. Waterbird loss at this severity is concentrated in one species; see per-species table. All uncertainty remains conditional on these folds and the already-inspected test cohort, with no parameter refitting inside bootstrap.

## Does FDAT beat foreground-only or global thresholds?

On POTER, foreground-only and FDAT tie WGA at all four primary sigmas. At sigma2 their predictions are identical on all400 composites; other primary severities differ on only2 rows each. Full per-sigma disagreement counts:{mismatches}. Thus a large improvement versus default0.5 does **not** establish incremental value for FDAT over the simpler foreground-only rule. The full image adds virtually no WGA here. OOF global epsilon2 WGA at sigma2 is60%, versus86% for both gray-based rules, but only the global rule has measurable original-clean score constraints; FDAT's clean trade-off is missing, so joint clean/blur dominance cannot be claimed.

Model-agnostic sigma2 OOF comparison:

{table(['Model','Fixed0.5 WGA','FDAT extended-grid WGA','Foreground-only WGA','Global eps2 transfer WGA'],modelrows)}

Other models also benefit in some conditions; ERM can lose Waterbird recall and be weaker than gray-only, while DPE gains from combining views. No POTER-specific novelty or isolated OT mechanism is established. One checkpoint/method, unmatched training seeds/recipes and legacy DPE historical binding limitations remain.

Only four fixed backgrounds were tested, including semantic distractors identified in earlier visual QC; compositing/segmentation artifacts and altered gray scene statistics may contribute. Independent human recognizability and realistic-mask robustness are not established. A cue that distinguishes these composites from originals is not evidence of adaptation to a naturally occurring shift.

## Original full5794 clean evaluation: **BLOCKED for FDAT**

Full original score cache is AVAILABLE for global thresholds. All5794 mask files are present and checksum-valid. But only100 composite-gray foreground scores exist, and **0/100** candidate original-gray pairs pass exact opaque-foreground pixel identity. Matching dimensions and source ID do not prove identical foreground RGB: gray used original-CUB pixels while originals use benchmark/HF JPEG pixels. No100-original FDAT WGA and no5794-original FDAT WGA is imputed. Therefore the<=2pp matched-original protection gate cannot be evaluated, regardless of strong blur results.

## Mask/runtime dependencies and contribution assessment

Masks are oracle dataset assets; realistic segmentation and mask-error robustness have not been tested. Serial inference typically adds one gray-view forward, approximately doubling network passes plus segmentation/composition; model weights can be shared. A later5794-gray job has an inference-throughput proxy of{proxy_minutes:.2f}GPU-minutes from earlier measured validation inference, plus unmeasured staging/composition/QA/latency overhead. No such job, segmentation or training was started. See deployment_audit.md for assumptions/memory and exact pairing requirements.

The [POTER paper](https://arxiv.org/abs/2610.01028) describes OT reweighting using limited validation-group reference information. [Medi et al.](https://arxiv.org/abs/2411.19853) motivates class-wise robustness/error-direction analysis. FDAT is the candidate supplied by this handoff; no claim is made that either paper introduces it, and no comprehensive novelty/prior-art review was performed. Binary CFPS restates error direction and is not a new objective. These tests manipulate outputs of frozen weights; they do not improve POTER's OT training algorithm.

## One next action / STOP

**STOP FDAT-specific algorithm expansion and expensive retraining.** If continuing the cheap diagnostic, the next gate is to obtain genuinely paired gray scores from the exact original benchmark pixels and compare the already-fixed AND rule and foreground-only against0.5, without more test-label hyperparameter search. Only a favorable original-clean result would justify planning independent development data and realistic-mask evaluation. The present run cannot support deployable, publishable or method-level robustness claims.

## Reproducibility and outputs

Run `.\\.venv\\Scripts\\python.exe scripts\\run_threshold_algorithm_feasibility.py`; `--stage A|B|C|report` regenerates individual supported stages. Configuration and all hypotheses/tie rules precede fold evaluation. No hyperparameter was changed after the held-out fold numbers were inspected; no new CNN predictions or generated experimental images were added. Parameter-grid and bootstrap code is NumPy/Pandas; Pillow is used only for existing-pixel identity audit. `files_manifest.csv` lists every generated artifact/hash, including all required CSVs/PNGs and the one-page ChatGPT handoff. Earlier experiment outputs and threshold locks are unchanged.
'''
    if (out/'tests.log').exists():text+='\nAutomated checks: '+(out/'tests.log').read_text().strip().splitlines()[-1]+'\n'
    (out/'algorithm_feasibility_report.md').write_text(text,encoding='utf-8')
    handoff=f'''# One-page handoff to ChatGPT

**Verdict: {verdict}. No training, CNN inference, OT changes or segmentation was run.**

Goal: improve foreground-blur WGA while preserving original Waterbirds WGA. Everything here is test-oracle or retrospective, never independent validation. Source CSV9000 rows,3 methods,100 foregrounds/20 species,4 backgrounds, sigmas0/.5/1/2/4. Full original5794/model scores are cached.

Stage A exact constrained global threshold: original loss epsilon0/1/2pp gives mean blurred-composite WGA gains+1.00/+2.33/+5.67pp. Epsilon2 reaches77.33% mean blur WGA while original WGA89.10%, recovering36.96% of unrestricted finite-score global gain. The predeclared80% "most" criterion fails. Composite-clean and original-clean are different.

FDAT uses d=max(0,p_full−p_gray),t=min(cap,.5+lambda*d),prediction=p_full>=t. Only scores/parameters enter inference.20 capped candidates plus5 cap1 ablations were predeclared; lambda0 matches0.5 exactly. True paired composite layer/alpha/mask hashes match. Shuffled foreground-gray negative controls were run.

Five class-stratified species folds keep every foreground/background/severity together; all POTER folds select lambda1/cap1 when that optional ablation is allowed. It agrees with binary AND on observed POTER scores; the full=1/clipped-threshold tie is an exception in general. Sigma2 pooled OOF WGA58→86% (+28pp; conditional species CI+19…+36pp); individual gains25/35/25/25/20pp. Gains span9 species. **Foreground-only also yields86%, with identical sigma2 decisions and tied WGA at every primary sigma.** No incremental POTER WGA is demonstrated for the extra FDAT rule. Primary capped oracle cap.7 yields83% sigma2, distinguished from extended-grid transfer. ERM/DPE tests show mixed generic benefits, not POTER-specific novelty.

Original clean protection is **BLOCKED**:5794 masks exist/hash-match, but no5794 exact-original gray prediction bank exists.100 cached gray controls are built from CUB RGB;0/100 pass original benchmark foreground-pixel identity. Do not copy them to originals, invent100-original FDAT WGA or treat those100 as5794. No deployable/publishable claim; independent dev/final test and realistic masks are absent.

Next: STOP algorithm/training expansion. If investigating further, obtain exact-original gray scores and compare fixed AND/foreground-only/0.5 before additional fitting. A5794-gray forward pass has a3.6GPU-minute throughput proxy, plus unmeasured composition/QA overhead; it was not launched. Read algorithm_feasibility_report.md, decision.json, fdat_transfer_summary.csv, original_coverage.json and deployment_audit.md before making claims.
'''
    (out/'handoff_to_chatgpt.md').write_text(handoff,encoding='utf-8')
    code=['configs/threshold_algorithm_feasibility.json','src/counterfactual/fdat_feasibility.py',
        'scripts/run_threshold_algorithm_feasibility.py','scripts/threshold_fdat_stages.py',
        'scripts/report_threshold_algorithm_feasibility.py','tests/test_fdat_feasibility.py']
    write_json(out/'implementation_provenance.json',dict(sha256={p:sha256(ROOT/p) for p in code if (ROOT/p).exists()},
        no_model_inference=True,no_training=True,hyperparameters_changed_after_fold_inspection=False))
    write_csv(out/'files_manifest.csv',[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha256(p))
        for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['files_manifest.csv','integrity.json']])
    write_json(out/'integrity.json',dict(sha256={p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='integrity.json'}))
    print(json.dumps(dict(report=str(out/'algorithm_feasibility_report.md'),verdict=verdict,
        full_original_FDAT='BLOCKED',new_training=False,new_CNN_inference=False)),flush=True)
