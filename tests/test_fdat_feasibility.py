"""Deterministic FDAT inference, identity/pairing and blocked-transfer checks."""
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.counterfactual.fdat_feasibility import (fdat,join_pairs,threshold_curve,prediction_metrics,
    choose_constrained,constrained_candidates,species_folds)
from scripts.threshold_fdat_stages import species_bootstrap

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/threshold_algorithm_feasibility'


@pytest.fixture(scope='module')
def raw():return pd.read_csv(ROOT/'outputs/mild_blur_v2/predictions.csv',float_precision='round_trip')


def test_inference_uses_only_scores_and_fixed_parameters_and_ge_ties():
    assert list(inspect.signature(fdat).parameters)==['p_full','p_gray','lambda_','t_max']
    result=fdat([.5,.4,.65,1.],[.5,.7,.5,0.],1.,1.)
    assert result['pred'].tolist()==[1,0,1,1]
    assert result['threshold'][-1]==1.  # >=tie; cap1 is not literally infinite/AND here.
    assert np.array_equal(result['discrepancy'],[0.,0.,.65-.5,1.])
    with pytest.raises(ValueError):fdat([.5],[np.nan],1.,.7)
    with pytest.raises(ValueError):fdat([.5],[.4,.2],1.,.7)


def test_identity_and_one_sided_decision_changes(raw):
    paired=join_pairs(raw)
    baseline=(paired.p_full>=.5).to_numpy(int)
    for cap in [.55,.6,.65,.7,1.]:
        identity=fdat(paired.p_full,paired.p_gray,0.,cap)
        assert np.array_equal(identity['pred'],baseline)
        alternative=fdat(paired.p_full,paired.p_gray,1.,cap)
        assert not ((baseline==0)&(alternative['pred']==1)).any()
        assert (alternative['threshold']>=.5).all() and (alternative['threshold']<=cap).all()


@pytest.mark.parametrize('fault',['missing_gray','duplicate_gray','mask','layer','target'])
def test_pairing_rejects_changed_or_incomplete_foreground(raw,fault):
    data=raw.copy();index=data.index[(data.model_name=='POTER')&(data.control=='foreground_only')][0]
    if fault=='missing_gray':data=data.drop(index)
    elif fault=='duplicate_gray':data=pd.concat([data,data.loc[[index]]],ignore_index=True)
    elif fault=='mask':data.loc[index,'mask_sha256']='different-mask'
    elif fault=='layer':data.loc[index,'layer_sha256']='different-layer'
    elif fault=='target':data.loc[index,'true_y']=1-int(data.loc[index,'true_y'])
    with pytest.raises(ValueError):join_pairs(data)


def test_sorted_score_curve_matches_independent_group_counts(raw):
    data=raw[(raw.model_name=='POTER')&(raw.control=='natural')&(raw.sigma==2.)]
    thresholds=[0.,.5,.65,1.];curve=threshold_curve(data,thresholds)
    for i,t in enumerate(thresholds):
        manual=prediction_metrics(data,(data.p_waterbird>=t).astype(int))
        for name in ['wga_pct','aa_pct','landbird_accuracy_pct','waterbird_accuracy_pct']:
            assert curve.iloc[i][name]==pytest.approx(manual[name])


def test_constrained_choices_satisfy_each_original_bound_and_maximize_objective():
    curves=pd.read_csv(OUT/'constrained_global_thresholds.csv',float_precision='round_trip')
    chosen=pd.read_csv(OUT/'constrained_global_selected.csv',float_precision='round_trip')
    for r in chosen.itertuples():
        rows=curves[(curves.model_name==r.model_name)&(curves.search==r.search)&(curves.epsilon_pp==r.epsilon_pp)]
        feasible=rows[rows.original_clean_drop_pp<=r.epsilon_pp+1e-10]
        assert r.original_clean_drop_pp<=r.epsilon_pp+1e-10
        assert r.mean_blur_wga_pct==pytest.approx(feasible.mean_blur_wga_pct.max())
        independent=choose_constrained(rows,r.epsilon_pp)
        assert independent['threshold']==r.threshold


def test_species_and_all_foreground_repeats_stay_in_one_fold(raw):
    paired=join_pairs(raw);membership=species_folds(paired,5,20261119)
    assert len(membership)==100 and membership.source_image_id.nunique()==100
    assert (membership.groupby('species_id').fold.nunique()==1).all()
    assert membership.groupby(['fold','true_y']).size().to_dict()=={(f,y):10 for f in range(5) for y in [0,1]}
    merged=paired.merge(membership[['source_image_id','fold']],on='source_image_id',validate='many_to_one')
    assert (merged.groupby('source_image_id').fold.nunique()==1).all()
    tuning=pd.read_csv(OUT/'fdat_transfer_tuning.csv')
    for r in tuning.itertuples():
        assert not set(json.loads(r.training_species))&set(json.loads(r.heldout_species))
        assert r.original_constraint_excludes_heldout_species


def test_species_bootstrap_keeps_repeated_backgrounds_together():
    data=pd.DataFrame(dict(source_image_id=list(range(8)),species_id=['a','a','b','b','c','c','d','d'],
        true_y=[0,0,0,0,1,1,1,1],bg_type=['land','water']*4,control=['natural']*8,
        p_waterbird=[.1,.6,.2,.3,.8,.9,.4,.8]))
    base=(data.p_waterbird>=.5).to_numpy(int);alternative=np.array([0,0,0,0,1,1,1,1])
    first=species_bootstrap(data,alternative,base,2000,1)
    duplicated=pd.concat([data]*4,ignore_index=True)
    second=species_bootstrap(duplicated,np.tile(alternative,4),np.tile(base,4),2000,1)
    for a,b in zip(first,second):
        assert a['delta_pp']==pytest.approx(b['delta_pp'])
        assert a['ci95_low_pp']==pytest.approx(b['ci95_low_pp'])
        assert a['ci95_high_pp']==pytest.approx(b['ci95_high_pp'])
    with pytest.raises(ValueError):species_bootstrap(data,alternative,base,100,1)


def test_full_original_missing_pairs_cannot_be_imputed_from_composites():
    coverage=json.loads((OUT/'original_coverage.json').read_text())
    identity=pd.read_csv(OUT/'original_pair_identity_audit.csv')
    assert coverage['mask_files_present']==5794 and not coverage['mask_hash_failures']
    assert coverage['cached_exact_original_pairs']==int(identity.eligible_original_pair.sum())==0
    assert coverage['full_original_FDAT_WGA'].startswith('BLOCKED')
    grid=pd.read_csv(OUT/'fdat_grid.csv')
    assert (grid.matched_original_count==0).all() and grid.matched_original_wga_pct.isna().all()
    folds=pd.read_csv(OUT/'fdat_transfer_folds.csv')
    assert folds.loc[folds.method=='fdat','meets_matched_original_loss2pp'].isna().all()


def test_negative_controls_have_no_label_conditioning_or_self_match():
    permutation=pd.read_csv(OUT/'negative_control_permutation.csv')
    assert set(permutation.source_image_id)==set(permutation.gray_donor_source_image_id)
    assert not (permutation.source_image_id==permutation.gray_donor_source_image_id).any()
    identity=json.loads((OUT/'identity_control.json').read_text())
    assert identity['lambda_zero_predictions_checked']==60000
    assert not identity['negative_control_uses_class_or_species']


def test_report_must_preserve_gray_only_baseline_and_clean_blocker():
    summary=pd.read_csv(OUT/'fdat_transfer_summary.csv')
    rows=summary[(summary.model_name=='POTER')&(summary.sigma==2.)].set_index('method')
    assert rows.loc['fdat','wga_pct']==rows.loc['foreground_only_classifier','wga_pct']==86
    oof=pd.read_csv(OUT/'fdat_transfer_predictions.csv')
    p=oof[(oof.model_name=='POTER')&(oof.sigma==2.)]
    assert (p.fdat_pred==p.foreground_only_classifier_pred).all()
    decision=json.loads((OUT/'decision.json').read_text())
    assert decision['executive_verdict']=='INCONCLUSIVE'
    assert decision['stage_C_deployable_publishable']=='BLOCKED'
    assert decision['stage_B']['clean_original_loss_criterion_passed'] is None


def test_analysis_imports_no_torch_or_classifier():
    code="import scripts.run_threshold_algorithm_feasibility; import sys; assert 'torch' not in sys.modules"
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
