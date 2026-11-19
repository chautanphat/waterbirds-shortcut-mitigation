"""Analyze frozen validation predictions, lock thresholds, then inspect test caches."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.counterfactual.common import sha256,write_json,write_csv
from src.counterfactual.investigation import MODELS
from src.counterfactual.threshold_probe import metrics,scope_metrics,select_clean_threshold,bootstrap_delta,predict


GROUP=['model_name','model_seed','dataset','source_split','control','sigma']


def validate_probabilities(frame):
    probabilities=frame[['p_landbird','p_waterbird']].to_numpy(float)
    if not np.isfinite(probabilities).all() or (probabilities<0).any() or (probabilities>1).any():
        raise ValueError('Invalid saved probabilities')
    if not np.allclose(probabilities.sum(1),1,atol=1e-6,rtol=0):raise ValueError('Probabilities do not sum to1')
    argmax=(frame.p_waterbird>frame.p_landbird).to_numpy(int)
    if not np.array_equal(argmax,frame.pred_y.to_numpy(int)):raise ValueError('Saved hard predictions disagree')
    baseline=predict(frame.p_waterbird,.5)
    delta=baseline!=frame.pred_y.to_numpy(int)
    if (delta & (abs(frame.p_waterbird.to_numpy()-.5)>1e-6)).any():raise ValueError('Non-tie baseline discrepancy')
    return dict(rows=len(frame),saved_argmax_matches=True,threshold05_vs_saved_disagreements=int(delta.sum()),
        probability_half_ties=int((frame.p_waterbird==.5).sum()),maximum_sum_error=float(abs(probabilities.sum(1)-1).max()))


def lock_choices(validation,config,asset,out):
    grid=np.arange(5,96,dtype=int)/100
    thresholds={};sweep=[]
    for model in MODELS:
        data=validation[validation.model_name==model]
        thresholds[model]=dict(model_seed=1 if model=='POTER' else 0,fixed_threshold=.5,
            clean_wga_threshold=None,joint_threshold=None,status=asset['eligibility'][model]['status'],
            reason=asset['eligibility'][model]['reason'],selection_rule=config['selection_rule'],
            selection_split='val',selection_dataset='canonical_validation1199',
            selection_input_sha256=sha256(out/'validation_predictions.csv'),
            checkpoint_sha256=str(data.checkpoint_sha256.iloc[0]),training_seed_n=1)
        if asset['eligibility'][model]['eligible']:
            winner,candidates=select_clean_threshold(data,grid)
            thresholds[model].update(clean_wga_threshold=winner['threshold'],
                validation_selected_wga=winner['wga'],validation_selected_accuracy=winner['accuracy'])
            for row in candidates:
                for details in scope_metrics(data,row['threshold']):
                    sweep.append(dict(model_name=model,model_seed=0,source_split='val',sigma=0.,
                        control='original_validation',threshold=row['threshold'],**details))
    payload=dict(locked_utc=datetime.now(timezone.utc).isoformat(),models=thresholds,
        grid=dict(start=.05,stop=.95,step=.01,count=91),
        tie_rule=config['selection_rule'],selection_datasets=['canonical_validation1199'],
        test_labels_used_for_selection=False,training_or_ot_changes=False,
        joint_status=config['joint_threshold'],untouched_evaluation_available=False,
        evaluation_caveat='Previously inspected test only; no new unbiased final-test claim. Validation CIs conditional on selected threshold, not selection-adjusted.')
    destination=out/'selected_thresholds.json'
    if destination.exists():
        original=json.loads(destination.read_text())
        if original['models']!=payload['models'] or original['grid']!=payload['grid']:
            raise ValueError('Locked thresholds differ; use a new output directory, never silently retune')
        payload=original
    else:write_json(destination,payload)
    pd.DataFrame(sweep).to_csv(out/'threshold_sweep_validation.csv',index=False)
    return payload


def prepare_test(config,asset):
    original=pd.read_csv(ROOT/config['original_test_predictions'],float_precision='round_trip')
    original['dataset']='original_test5794'
    counter=pd.read_csv(ROOT/config['source_predictions'],float_precision='round_trip')
    counter=counter[counter.control!='original'].copy();counter['dataset']='counterfactual100'
    # Original100 is already a subset of full5794; never concatenate it again.
    frame=pd.concat([original,counter],ignore_index=True)
    frame['selection_eligible']=False;frame['untouched_evaluation']=False
    for model in MODELS:
        info=asset['checkpoint_info'][model]
        mask=frame.model_name==model
        frame.loc[mask,'checkpoint_sha256']=info.get('checkpoint_sha256',info.get('backbone_sha256'))
        frame.loc[mask,'heads_sha256']=info.get('heads_sha256')
    if set(frame.source_split)!={'test'}:raise ValueError('Unexpected evaluation split')
    canonical=pd.read_csv(ROOT/'data/waterbirds/metadata.csv').set_index('img_id')
    labels=canonical.loc[frame.source_image_id.astype(int)]
    known=frame.true_y.notna().to_numpy()
    if not np.array_equal(frame.loc[known,'true_y'].to_numpy(int),labels.y.to_numpy(int)[known]):
        raise ValueError('Cached targets differ from canonical source IDs')
    if not np.array_equal(frame.original_group.to_numpy(int),(2*labels.y+labels.place).to_numpy(int)):
        raise ValueError('Cached original groups differ from canonical metadata')
    if not np.array_equal(frame.species_id.to_numpy(),labels.img_filename.str.split('/').str[0].to_numpy()):
        raise ValueError('Species/source ID join differs')
    return frame


def paired_gray_shifts(test,config,out):
    data=test[test.control.isin(['natural','foreground_only'])]
    records=[];pairs=[]
    for (model,seed,source,sigma),part in data.groupby(['model_name','model_seed','source_image_id','sigma'],sort=False):
        gray=part[part.control=='foreground_only'];natural=part[part.control=='natural']
        if len(gray)!=1 or len(natural)!=4:raise ValueError('Incomplete paired background bank')
        gray=gray.iloc[0]
        if part.layer_sha256.nunique()!=1 or part.alpha_sha256.nunique()!=1:raise ValueError('Foreground intervention mismatch')
        gray_pred=int(gray.p_waterbird>=.5);natural_pred=predict(natural.p_waterbird,.5)
        base=dict(model_name=model,model_seed=seed,source_image_id=int(source),species_id=gray.species_id,
            true_y=int(gray.true_y),sigma=float(sigma),source_split='test',threshold=.5,exploratory=True)
        records.append(dict(base,gray_p_waterbird=float(gray.p_waterbird),
            natural_mean_p_waterbird=float(natural.p_waterbird.mean()),
            mean_signed_probability_shift=float(natural.p_waterbird.mean()-gray.p_waterbird),
            mean_absolute_probability_shift=float(abs(natural.p_waterbird-gray.p_waterbird).mean()),
            paired_flip_rate=float((natural_pred!=gray_pred).mean()),
            accuracy_change=float((natural_pred==base['true_y']).mean()-(gray_pred==base['true_y']))))
        for row in natural.itertuples():
            pairs.append(dict(base,natural_image_id=row.image_id,gray_image_id=gray.image_id,
                background_id=row.background_id,bg_type=row.bg_type,
                signed_probability_shift=float(row.p_waterbird-gray.p_waterbird),
                pred_gray=gray_pred,pred_natural=int(row.p_waterbird>=.5),
                layer_sha256=row.layer_sha256,alpha_sha256=row.alpha_sha256))
    frame=pd.DataFrame(records);frame.to_csv(out/'paired_gray_natural_shifts.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'paired_gray_natural_replicates.csv',index=False)
    intervals=[]
    for (model,seed,sigma,y),part in frame.groupby(['model_name','model_seed','sigma','true_y'],sort=False):
        rng=np.random.default_rng(config['bootstrap_seed'])
        draws=rng.integers(0,len(part),(config['bootstrap_samples'],len(part)))
        for metric in ['mean_signed_probability_shift','mean_absolute_probability_shift','paired_flip_rate','accuracy_change']:
            values=part[metric].to_numpy();lo,hi=np.quantile(values[draws].mean(1),[.025,.975])
            intervals.append(dict(model_name=model,model_seed=seed,sigma=sigma,true_y=y,metric=metric,
                estimate=float(values.mean()),ci95_low=float(lo),ci95_high=float(hi),
                bootstrap_samples=config['bootstrap_samples'],bootstrap_unit='foreground; within true class; average4 background replicates first',
                n_foregrounds=len(part),exploratory=True))
    pd.DataFrame(intervals).to_csv(out/'paired_gray_natural_ci.csv',index=False)


def figures(out,validation,test,sweep,evaluation,locked):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'plots';folder.mkdir(exist_ok=True)
    colors={'POTER':'#b64b39','ERM_enhanced':'#2878a4','DPE_legacy':'#437e4c'}
    def save(fig,name):fig.savefig(folder/name,dpi=150,bbox_inches='tight');plt.close(fig)
    for metric,title,name in [('wga','Clean validation WGA','wga_vs_threshold.png'),
                             ('class','Validation class accuracies','class_accuracy_vs_threshold.png')]:
        fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
        for model,ax in zip(MODELS,axs):
            rows=sweep[(sweep.model_name==model)&(sweep.scope=='all')]
            ax.set(title=model,xlabel='Waterbird threshold',ylabel='WGA (%)' if metric=='wga' else 'Accuracy (%)',ylim=(0,103))
            if rows.empty:
                ax.text(.5,.5,'Threshold sweep BLOCKED\nNo independent four-group\ndevelopment data',ha='center',va='center',transform=ax.transAxes)
                continue
            if metric=='class':
                for field,label in [('landbird_accuracy','Landbird'),('waterbird_accuracy','Waterbird')]:ax.plot(rows.threshold,100*rows[field],label=label)
                ax.legend()
            else:ax.plot(rows.threshold,100*rows.wga,color=colors[model])
            ax.axvline(.5,ls=':',color='gray');ax.axvline(locked['models'][model]['clean_wga_threshold'],ls='--',color='black')
            ax.text(.03,.04,f"locked t={locked['models'][model]['clean_wga_threshold']:.2f}; baseline t=0.50",transform=ax.transAxes,fontsize=8)
        fig.suptitle(title+' | selection uses original validation only; no test sweep')
        save(fig,name)
    fig,axs=plt.subplots(3,3,figsize=(10,9),layout='constrained')
    subsets=[('Validation1199; exposure audited',validation),
             ('Original test5794; exploratory',test[test.dataset=='original_test5794']),
             ('Natural test sigma2; exploratory',test[(test.control=='natural')&(test.sigma==2)])]
    for row,(title,data) in enumerate(subsets):
        for col,model in enumerate(MODELS):
            ax=axs[row,col];m=metrics(data[data.model_name==model],.5)
            matrix=np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]])
            ax.imshow(matrix,cmap='Blues')
            for (y,x),value in np.ndenumerate(matrix):ax.text(x,y,str(value),ha='center',va='center',color='white' if value>matrix.max()*.5 else 'black')
            ax.set(title=model+'\n'+title,xticks=[0,1],xticklabels=['Land','Water'],yticks=[0,1],yticklabels=['Land','Water'],xlabel='Predicted',ylabel='True')
    fig.suptitle('Confusion counts at fixed threshold0.5; rows have different sample sizes')
    save(fig,'confusion_matrices.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    original=evaluation[(evaluation.dataset=='original_test5794')]
    for i,model in enumerate(MODELS):
        fixed=original[(original.model_name==model)&(original.strategy=='fixed05')].iloc[0]
        axs[0].scatter(i,100*fixed.wga,color=colors[model],label=model)
        selected=original[(original.model_name==model)&(original.strategy=='clean_wga')]
        if len(selected):axs[0].scatter(i,100*selected.wga.iloc[0],marker='x',s=90,color='black',label='ERM validation-selected threshold')
    axs[0].set(title='Original unmodified test5794',xticks=range(3),xticklabels=MODELS,ylabel='WGA (%)',ylim=(0,103))
    axs[0].text(.03,.08,'Circle: fixed t=0.50\nCross: ERM locked t=0.13',transform=axs[0].transAxes,fontsize=9)
    for model in MODELS:
        for strategy in ['fixed05','clean_wga']:
            rows=evaluation[(evaluation.model_name==model)&(evaluation.control=='natural')&(evaluation.strategy==strategy)].sort_values('sigma')
            if len(rows):axs[1].plot(rows.sigma,100*rows.wga,'-o' if strategy=='fixed05' else '--x',color=colors[model],label=model+' / '+strategy)
    axs[1].set(title='Natural composites of100 test foregrounds',xlabel='Sigma (4 supplementary)',ylabel='WGA (%)',ylim=(0,103))
    axs[1].legend(fontsize=8);fig.suptitle('Clean/blur trade-offs — exploratory evaluation, never threshold-selection targets')
    save(fig,'clean_blur_wga_tradeoff.png')
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,(data,title) in enumerate([(validation,'Original validation'),(test[(test.control=='natural')&(test.sigma==2)],'Natural test sigma2; exploratory')]):
        for col,model in enumerate(MODELS):
            ax=axs[row,col];part=data[data.model_name==model]
            for y,label,color in [(0,'True Landbird','#b86e31'),(1,'True Waterbird','#247fa3')]:
                ax.hist(part[part.true_y==y].p_waterbird,bins=np.linspace(0,1,31),density=True,histtype='step',label=label,color=color)
            ax.axvline(.5,color='gray',ls=':');ax.set(title=model+' / '+title,xlabel='P(Waterbird)',ylabel='Density',xlim=(0,1));ax.legend(fontsize=8)
    fig.suptitle('Probabilities are uncalibrated model outputs; cross-model confidence scales differ')
    save(fig,'probability_distributions.png')


def write_reports(config,out,asset,locked,validation,test,evaluation,bootstrap,checks):
    eligible=locked['models']['ERM_enhanced'];t=eligible['clean_wga_threshold']
    def original(model,strategy):return evaluation[(evaluation.model_name==model)&(evaluation.dataset=='original_test5794')&(evaluation.strategy==strategy)].iloc[0]
    def cell(value):return f'{100*value:.2f}' if pd.notna(value) else 'undefined'
    rows=[]
    for model in MODELS:
        for strategy in ['fixed05','clean_wga']:
            subset=evaluation[(evaluation.model_name==model)&(evaluation.strategy==strategy)]
            for dataset,sigma in [('canonical_validation1199',0.),('original_test5794',0.),('counterfactual100',.5),('counterfactual100',1.),('counterfactual100',2.)]:
                selected=subset[(subset.dataset==dataset)&(subset.sigma==sigma)]
                if dataset=='counterfactual100':selected=selected[selected.control=='natural']
                if len(selected):
                    r=selected.iloc[0];rows.append(f'|{model}|{strategy}|{dataset}|{sigma:g}|{r.threshold:.2f}|{cell(r.accuracy)}|{cell(r.wga)}|{cell(r.landbird_accuracy)}|{cell(r.waterbird_accuracy)}|')
    inference=json.loads((out/'validation_inference.json').read_text())
    poter=evaluation[(evaluation.model_name=='POTER')&(evaluation.control=='natural')&evaluation.sigma.isin([0.,.5,1.,2.])].sort_values('sigma')
    poter_directions='\n'.join(f"|{r.sigma:g}|{r.fp}|{r.fn}|{100*r.false_positive_rate:.2f}|{100*r.false_negative_rate:.2f}|{100*r.fraction_predicted_waterbird:.2f}|{100*r.cfps_predicted_waterbird:.2f}|" for r in poter.itertuples())
    audit=f'''# Asset and leakage audit

Canonical metadata maps Landbird0/Waterbird1 and group=2*y+place, Land0/Water1. Validation1199 has groups467/466/133/133; all images/IDs/labels are checked. The requested browser-renamed predictions(5).csv was not attached; canonical audited9000-row mild-blur CSV is used. Full original test5794×3 is cached separately. Its100-image subset is not added twice.

## Selection eligibility is model specific

- POTER seed1: **BLOCKED: no independent four-group development data for threshold selection**. OT reference used all466 validation group1 and all133 group2 images; unused600 have only groups0/3. Validation also selected the best checkpoint. No threshold sweep or selected threshold is produced for POTER.
- ERM enhanced seed0: clean validation sweep allowed. Train4795 used for fitting; fixed last400 checkpoint, original validation1199 for this diagnostic. Historical monitoring and reuse for threshold selection mean these validation metrics are not unbiased validation of a selected threshold.
- DPE legacy seed0: **BLOCKED: no independent development data for threshold selection**. Actual saved head provenance lists exactly all1199 validation filenames as fitting data. Those probabilities are descriptive fitting-pool diagnostics, not held-out calibration data.
- No independent development blur bank with masks exists here; joint clean-constrained threshold not run. No untouched evaluation set was located. Previously inspected test is exploratory only.

See `asset_audit.json`, `validation_membership.csv`, `checkpoint_audit.json` and `validation_inference.json` for paths, SHA256, model/seed, group membership, fitting overlap and numeric parity. POTER seed1 and ERM/DPE seed0 are unmatched; no multi-seed claim. DPE historical head→backbone hash binding remains unverified, although explicit frozen pairing and inference parity are checked.

Official labels are retained throughout selection/evaluation. Known Waterbirds label-generation issues affect species021.Eastern_Towhee,088.Western_Meadowlark,102.Western_Wood_Pewee; the100-image counterfactual cohort had already excluded them before its predictions. Full original validation/test retain canonical labels, without post-hoc correction or selecting a favorable label policy.

## Prediction schema and reuse

`validation_predictions.csv` retains source split/model/seed/checkpoint hashes, canonical source foreground ID, species, true label/group/background, sigma/control, hard prediction, probabilities/logits, raw/model/tensor hashes and eligibility. Cached test columns retain the equivalent source fields; checkpoint hashes are attached from verified manifests during analysis. Background-only rows have no bird label and are excluded from all accuracy/error denominators.

Validation: {inference['validation_rows']} probability rows, {inference['poter_new_CNN_images']} new POTER image forwards, {inference['erm_cached_feature_rows_reused']} cached ERM feature rows reused;1199 ERM linear predictions and1199 legacy DPE ensemble probabilities newly computed. Additional live ERM checks are disclosed in inference manifest. This is inference, not training. Shared square256/bilinear/center224/ImageNet preprocessing matches Runner. No OT recomputation, generated experimental images, new blur or augmentation changes.

Probabilities are finite/in[0,1]/sum-to-one checked, p_waterbird means class1, saved argmax matches. Recomputing >=0.5 versus saved argmax differs on {checks['combined']['threshold05_vs_saved_disagreements']} rows (ties/roundoff counted explicitly). See `prediction_checks.json`. Diagnostic row groups are kept separate: original1199 validation, original5794 test and100-foreground repeated-composite test. Clean natural composite sigma0 is not the original unmodified photograph.
'''
    (out/'audit.md').write_text(audit,encoding='utf-8')
    bf=original('ERM_enhanced','fixed05');sf=original('ERM_enhanced','clean_wga')
    cis=bootstrap[(bootstrap.dataset=='original_test5794')&(bootstrap.metric=='wga')]
    wga_ci=cis.iloc[0]
    report=f'''# Fast POTER class-wise threshold diagnostic — no retraining

## Main conclusion

**The primary POTER threshold question remains BLOCKED by absence of independent four-group development data.** No POTER threshold was optimized, and fixed0.5 POTER results do not establish whether threshold tuning would fix its asymmetry. DPE is also blocked because all validation images fitted its heads. ERM provides a valid cheap clean-validation-selected threshold baseline, with limitations below. A new VA-POTER training method is **not justified by this run**; neither is a claim that threshold tuning has solved POTER's errors.

## Validation-only lock

ERM seed0 selected threshold **{t:.2f}**, using91 candidates0.05…0.95 step0.01; maximize original-clean validation WGA, then AA, then proximity to0.5, then lower threshold. Locked JSON was written **before loading test labels for the analysis/evaluation stage**. Test-source files were integrity-hashed during audit but no test metric or label entered selection. POTER/DPE clean-WGA thresholds are null with explicit BLOCKED reasons; fixed threshold0.5 is predeclared for all models. Optional joint threshold was skipped; no independent development blur data. Thresholds are not reselected during bootstrap or after test inspection.

Validation1199 is not a shared held-out development set for all three methods: POTER used599 as OT reference and DPE fitted all1199. The remaining600 POTER non-reference images lack two groups, so WGA over four groups is undefined there. Splitting them now cannot repair the missing independent groups. Do not select a favorable threshold using existing test labels.

## Main tables

All metrics below use official labels; percentages. Validation POTER/DPE rows are **reference/fitting-exposed descriptive** diagnostics. All test rows are **already-inspected exploratory evaluation**. Sigma4 is supplementary and excluded from selection. Clean original test and clean composites are distinct image populations.

|Model|Strategy|Dataset|Sigma|Threshold|AA %|WGA %|Landbird %|Waterbird %|
|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

ERM original test WGA changes **{100*bf.wga:.2f}%→{100*sf.wga:.2f}%**, paired Δ **{100*wga_ci.delta:+.2f} pp**, foreground-bootstrap CI95% **[{100*wga_ci.ci95_low:+.2f}, {100*wga_ci.ci95_high:+.2f}] pp**. Landbird Δ={100*(sf.landbird_accuracy-bf.landbird_accuracy):+.2f} pp; Waterbird Δ={100*(sf.waterbird_accuracy-bf.waterbird_accuracy):+.2f} pp; AA Δ={100*(sf.accuracy-bf.accuracy):+.2f} pp. These are conditional descriptive contrasts on an already-inspected test set, not an unbiased generalization gain. Read every severity/group in `locked_threshold_evaluation.csv` and `classwise_metrics.csv` rather than optimizing for blurred-test results.

## Class-wise errors and background shifts

POTER fixed0.5, natural composites of100 test foregrounds (200 observations per true class at each sigma):

|Sigma|Land→Water errors|Water→Land errors|FPR %|FNR %|Predicted Water %|CFPS predicted Water %|
|---|---|---|---|---|---|---|
{poter_directions}

On original unmodified5794 test images, POTER has similar class-conditioned error rates (Land→Water8.03%, Water→Land7.94%). Increasing error direction toward Waterbird is therefore a property of these degraded/composited inputs, not evidence that the original clean task has the same class asymmetry. Original-test class counts are imbalanced (4510 Landbird/1284 Waterbird), whereas the100-foreground cohort is balanced; raw error fractions and CFPS depend on that prevalence. Neither CFPS nor these fixed-threshold observations isolate calibration or an internal mechanism.

`classwise_metrics.csv` contains aggregate, true-class and class×background rows: AA/WGA when defined, four group accuracies, full confusion counts, FPR=Landbird→Waterbird rate, FNR=Waterbird→Landbird rate, fraction predicted Waterbird, mean/median P(Waterbird), and CFPS by predicted class. CFPS is wrong-to-class / all wrong; undefined with zero errors. In binary classification it restates error direction and is not an independent mechanism or proposed novel score. Class/group rows retain their own denominators; four-group WGA is undefined on a single class/group and on gray backgrounds.

`paired_gray_natural_replicates.csv` retains all background pairs; `paired_gray_natural_shifts.csv` averages four natural backgrounds within each source foreground. `paired_gray_natural_ci.csv` resamples foregrounds within true class (2000 draws), preserving all background repeats. These fixed0.5 probability/flip/accuracy shifts describe background dependence in the existing test cohort. They do not identify an internal mechanism or justify an OT adjustment. POTER's original goal concerns spurious correlations/label noise; corruption robustness is an additional diagnostic, not a guaranteed objective.

## Statistical interpretation

`paired_bootstrap_ci.csv` compares locked ERM threshold versus fixed0.5 on identical original/blurred/gray images. At least2000 class-stratified foreground draws per condition; all background repeats travel together. WGA is recomputed over all four groups per draw. Undefined group resamples are excluded and counted. Selection-data CIs are conditional/in-sample, not selection-adjusted; no unbiased validation gain is claimed. Test CIs remain exploratory. One checkpoint per method cannot estimate training-seed variability. We report no new multi-seed stability or untouched-test claim.

## Decision and remaining evidence

- **POTER/DPE: decision gate4 — insufficient independent development data for threshold selection.** Cannot distinguish cheap threshold correction from representation/background sensitivity with a tuned threshold yet.
- **ERM: clean-WGA-selected baseline executed**, with class/AA/WGA/blur trade-offs quantified. Its behavior cannot be transferred to POTER's probability scale or taken as evidence of a POTER-specific mechanism.
- Obtain genuinely independent labeled development images spanning all four groups, with foreground masks only if joint blur selection is desired; freeze threshold rules there, then evaluate on a separately untouched final set. Current original/blur test images have already been inspected and cannot become fresh by relabeling their split.
- Do not train a new method, tune OT references, or spend on additional POTER seeds merely to run this diagnostic. Ordinary calibration/cost-sensitive baselines and representation/logit analysis remain future options once the data separation is valid.

## Execution and files

Inference completed using actual checkpoints and1199 validation originals, reusing cached ERM embeddings only after8 predeclared live feature/probability parity checks. No training, OT recomputation or new experimental images. Validation inference elapsed {inference['elapsed_seconds']:.1f}s; phase counts/errors in `validation_inference.json`. Cached26082 test/control rows reused; background-only1200 have no classification target. No unchanged test CNN inference was rerun. See `files_manifest.csv` and `integrity.json` for exported file hashes.

An initial8-image ERM preflight using default cuDNN TF32 failed cache parity and produced no accepted prediction CSV. Matching the existing FP32 comparison flags resolved it (feature max error1.12e-5, ERM probability max error8.94e-7, DPE reference error8.94e-8). The successful validation run used those fixed flags; `preflight_attempts.json` records the rejected attempt. No tolerance was relaxed to accept it. Total additional ERM diagnostic CNN forwards are24 images across failed/successful checks (8+8+8), separate from the1199 cached embeddings and1199 new POTER forwards.

Required outputs: audit.md, classwise_metrics.csv, threshold_sweep_validation.csv, selected_thresholds.json, locked_threshold_evaluation.csv, paired_bootstrap_ci.csv, five PNG plots and this report. Additional validation probabilities, membership, per-pair shifts and source/code provenance are retained. Run `python scripts/run_threshold_probe.py --stage analysis` to regenerate reports from verified validation predictions; no model inference or training in that stage. The locked JSON is preserved and a changed choice causes an error rather than silent retuning.
'''
    if (out/'tests.log').exists():report+='\nAutomated checks: '+(out/'tests.log').read_text(encoding='utf-8').strip().splitlines()[-1]+'\n'
    (out/'report.md').write_text(report,encoding='utf-8')


def execute_analysis(config,out,asset):
    validation=pd.read_csv(out/'validation_predictions.csv',float_precision='round_trip')
    inference=json.loads((out/'validation_inference.json').read_text())
    if sha256(out/'validation_predictions.csv')!=inference['prediction_sha256']:raise ValueError('Validation predictions changed')
    validation_checks=validate_probabilities(validation)
    # This function cannot read test labels before the threshold lock is committed.
    locked=lock_choices(validation,config,asset,out)
    lock_sha=sha256(out/'selected_thresholds.json')
    test=prepare_test(config,asset);test_checks=validate_probabilities(test)
    if test[test.control=='background_only'].true_y.notna().any():raise ValueError('Background-only labels are invalid')
    known=test[test.true_y.notna()].copy();known['true_y']=known.true_y.astype(int)
    combined=pd.concat([validation,known],ignore_index=True)
    checks=dict(validation=validation_checks,test=test_checks,combined=dict(
        threshold05_vs_saved_disagreements=validation_checks['threshold05_vs_saved_disagreements']+test_checks['threshold05_vs_saved_disagreements']),
        selection_lock_sha256=lock_sha,validation_inference_sha256=sha256(out/'validation_predictions.csv'),
        test_semantic_rows=len(known),test_background_only_rows=int((test.control=='background_only').sum()),
        validation_rows=len(validation),test_rows_reused=len(test))
    write_json(out/'prediction_checks.json',checks)
    tables=[];evaluation=[];bootstrap=[]
    for key,data in combined.groupby(GROUP,sort=False):
        base=dict(zip(GROUP,key));model=base['model_name']
        strategies={'fixed05':.5}
        threshold=locked['models'][model]['clean_wga_threshold']
        if threshold is not None:strategies['clean_wga']=threshold
        for strategy,t in strategies.items():
            context=dict(base,strategy=strategy,threshold=t,untouched_evaluation=False,
                evaluation_role='selection-data; conditional/in-sample' if base['source_split']=='val' and asset['eligibility'][model]['eligible'] else
                    'fitting/reference-exposed descriptive' if base['source_split']=='val' else 'previously-inspected test; exploratory',
                checkpoint_sha256=str(data.checkpoint_sha256.iloc[0]))
            evaluation.append(dict(context,**metrics(data,t)))
            for row in scope_metrics(data,t):tables.append(dict(context,**row))
        if threshold is not None:
            for row in bootstrap_delta(data,threshold,samples=config['bootstrap_samples'],seed=config['bootstrap_seed']):
                bootstrap.append(dict(base,strategy='clean_wga_minus_fixed05',threshold=threshold,
                    evaluation_role='conditional/in-sample selection-data' if base['source_split']=='val' else 'exploratory previously-inspected test',**row))
    tables=pd.DataFrame(tables);evaluation=pd.DataFrame(evaluation);bootstrap=pd.DataFrame(bootstrap)
    tables.to_csv(out/'classwise_metrics.csv',index=False)
    evaluation.to_csv(out/'locked_threshold_evaluation.csv',index=False)
    bootstrap.to_csv(out/'paired_bootstrap_ci.csv',index=False)
    paired_gray_shifts(test,config,out)
    figures(out,validation,known,pd.read_csv(out/'threshold_sweep_validation.csv'),evaluation,locked)
    write_reports(config,out,asset,locked,validation,known,evaluation,bootstrap,checks)
    if sha256(out/'selected_thresholds.json')!=lock_sha:raise ValueError('Threshold lock changed during evaluation')
    write_json(out/'status.json',dict(training_started=False,ot_recomputed=False,new_experimental_images=0,
        untouched_evaluation_available=False,multi_seed_validation_complete=False,
        POTER='BLOCKED: no independent four-group development data',DPE_legacy='BLOCKED: validation used for head fitting',
        ERM_enhanced='validation threshold selected; exploratory test evaluation complete'))
    code=['scripts/run_threshold_probe.py','scripts/report_threshold_probe.py','src/counterfactual/threshold_probe.py','configs/poter_threshold_probe.json','tests/test_threshold_probe.py']
    write_json(out/'analysis_provenance.json',dict(sha256={name:sha256(ROOT/name) for name in code},
        threshold_lock_sha256=lock_sha,bootstrap_seed=config['bootstrap_seed'],bootstrap_samples=config['bootstrap_samples'],
        checkpoint_selection='fixed existing checkpoints; never selected using test',source_assets=asset['source_files']))
    write_csv(out/'files_manifest.csv',[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['files_manifest.csv','integrity.json']])
    write_json(out/'integrity.json',dict(sha256={p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='integrity.json'}))
    print(json.dumps(dict(report=str(out/'report.md'),selected_ERM_threshold=locked['models']['ERM_enhanced']['clean_wga_threshold'],POTER_threshold_selection='blocked',training=False)),flush=True)
