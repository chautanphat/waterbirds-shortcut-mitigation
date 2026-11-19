"""Threshold selection leakage guards and clustered diagnostic statistics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.counterfactual.threshold_probe import predict,metrics,select_clean_threshold,bootstrap_delta

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def validation():
    return pd.read_csv(ROOT/'outputs/poter_threshold_probe/validation_predictions.csv',float_precision='round_trip')


def toy():
    # Unit-test confusion arithmetic only; never exported as experiment predictions.
    return pd.DataFrame(dict(source_image_id=[1,2,3,4],true_y=[0,0,1,1],original_group=[0,1,2,3],
        p_waterbird=[.1,.9,.8,.2],control=['original_validation']*4,source_split=['val']*4,
        sigma=[0.]*4,selection_eligible=[True]*4))


def test_confusion_error_direction_cfps_and_probability_semantics():
    r=metrics(toy(),.5)
    assert [r[k] for k in ['tn','fp','fn','tp']]==[1,1,1,1]
    assert r['accuracy']==.5 and r['wga']==0
    assert r['false_positive_rate']==r['false_negative_rate']==.5
    assert r['cfps_predicted_landbird']==r['cfps_predicted_waterbird']==.5
    assert predict([.499,.5,.501],.5).tolist()==[0,1,1]
    data=toy();data['p_waterbird']=[.1,.2,.8,.9]
    perfect=metrics(data,.5)
    assert perfect['accuracy']==perfect['wga']==1
    assert perfect['cfps_predicted_landbird'] is perfect['cfps_predicted_waterbird'] is None


def test_neutral_control_cannot_report_four_group_wga():
    data=toy();data['control']='foreground_only'
    assert metrics(data,.5)['wga'] is None


def test_only_independent_clean_validation_can_select(validation):
    grid=np.arange(5,96)/100
    for model in ['POTER','DPE_legacy']:
        with pytest.raises(ValueError,match='fitting/reference'):
            select_clean_threshold(validation[validation.model_name==model],grid)
    erm=validation[validation.model_name=='ERM_enhanced'].copy()
    contaminated=erm.copy();contaminated.loc[contaminated.index[0],'source_split']='test'
    with pytest.raises(ValueError,match='validation only'):select_clean_threshold(contaminated,grid)
    blurred=erm.copy();blurred['sigma']=.5
    with pytest.raises(ValueError,match='Clean original'):select_clean_threshold(blurred,grid)
    subset=erm[erm.original_group.isin([0,3])]
    with pytest.raises(ValueError,match='All four'):select_clean_threshold(subset,grid)


def test_locked_real_threshold_matches_independent_integer_count_sweep(validation):
    erm=validation[validation.model_name=='ERM_enhanced'];grid=np.arange(5,96)/100
    independent=[]
    for threshold in grid:
        pred=(erm.p_waterbird.to_numpy()>=threshold).astype(int)
        correct=pred==erm.true_y.to_numpy()
        groups=erm.original_group.to_numpy()
        wga=min(correct[groups==g].mean() for g in range(4))
        independent.append((float(wga),float(correct.mean()),float(threshold)))
    result=sorted(independent,key=lambda r:(-round(r[0],14),-round(r[1],14),round(abs(r[2]-.5),14),r[2]))[0]
    locked=json.loads((ROOT/'outputs/poter_threshold_probe/selected_thresholds.json').read_text())
    assert result[2]==locked['models']['ERM_enhanced']['clean_wga_threshold']==.13
    assert locked['models']['POTER']['clean_wga_threshold'] is None
    assert locked['models']['DPE_legacy']['clean_wga_threshold'] is None
    assert not locked['test_labels_used_for_selection']


def test_tie_break_is_deterministic_and_uses_lower_threshold_last():
    data=toy();data['p_waterbird']=[.1,.2,.8,.9]
    winner,_=select_clean_threshold(data,[.51,.49])
    assert winner['threshold']==.49


def test_bootstrap_keeps_background_replicates_in_foreground_clusters():
    data=toy();data['p_waterbird']=[.12,.3,.7,.4]
    first=bootstrap_delta(data,.25)
    repeated=pd.concat([data]*4,ignore_index=True)
    second=bootstrap_delta(repeated,.25)
    for a,b in zip(first,second):
        assert a['n_foregrounds']==b['n_foregrounds']==4
        assert a['delta']==pytest.approx(b['delta'])
        assert a['ci95_low']==pytest.approx(b['ci95_low'])
        assert a['ci95_high']==pytest.approx(b['ci95_high'])
    unchanged=bootstrap_delta(data,.5)
    assert all(r['delta']==r['ci95_low']==r['ci95_high']==0 for r in unchanged)
    with pytest.raises(ValueError,match='At least2000'):bootstrap_delta(data,.25,samples=100)


def test_model_specific_development_overlap_matches_saved_membership():
    pool=pd.read_csv(ROOT/'outputs/poter_threshold_probe/validation_membership.csv')
    assert len(pool)==1199 and pool.source_image_id.nunique()==1199
    assert pool.poter_ot_reference.sum()==599
    assert set(pool.loc[~pool.poter_ot_reference,'group'])=={0,3}
    assert pool.dpe_head_fitting.all()
    assert not pool.poter_threshold_eligible.any() and not pool.dpe_threshold_eligible.any()


def test_real_evaluation_has_no_unreported_threshold_or_untouched_claim():
    out=ROOT/'outputs/poter_threshold_probe'
    rows=pd.read_csv(out/'locked_threshold_evaluation.csv')
    assert set(rows[rows.strategy=='clean_wga'].model_name)=={'ERM_enhanced'}
    assert not rows.untouched_evaluation.any()
    assert set(rows[rows.strategy=='clean_wga'].threshold)=={.13}
    ci=pd.read_csv(out/'paired_bootstrap_ci.csv')
    assert ci.bootstrap_samples.min()>=2000 and not ci.threshold_reselected_in_bootstrap.any()
    assert not ci.training_seed_ci.any()
    checks=json.loads((out/'prediction_checks.json').read_text())
    assert checks['combined']['threshold05_vs_saved_disagreements']==0
    assert checks['test_background_only_rows']==1200 and checks['test_rows_reused']==26082
