"""Real-output regression checks for the paired milder Gaussian experiment."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.counterfactual.common import ROOT,sha256
from src.counterfactual.failure_statistics import validate_predictions
from scripts.run_mild_blur_v2 import PairedPredictionCache


@pytest.fixture(scope='module')
def saved():
    out=ROOT/'outputs/mild_blur_v2'
    if not (out/'bootstrap_ci.csv').exists():pytest.skip('Complete real mild-blur v2 outputs are required')
    return out,pd.read_csv(out/'predictions.csv'),pd.read_csv(out/'cohort.csv')


def test_cohort_and_prediction_schema_are_unchanged(saved):
    out,frame,cohort=saved
    assert sha256(out/'cohort.csv')==sha256(ROOT/'results/poter_failure/cohort.csv')
    previous=pd.read_csv(ROOT/'results/poter_failure/predictions.csv',nrows=1)
    assert list(frame.columns)==list(previous.columns) and len(frame.columns)==34
    assert validate_predictions(frame,cohort,severity_count=5)['rows']==9000
    assert frame.control.value_counts().to_dict()==dict(natural=6000,foreground_only=1500,background_only=1200,original=300)


def test_sigma_not_severity_name_defines_shared_previous_inputs(saved):
    out,frame,_=saved
    old=pd.read_csv(ROOT/'results/poter_failure/predictions.csv')
    keys=['model_name','source_image_id','background_id','control','sigma']
    common=frame.merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    assert len(common)==4500
    assert set(common[common.sigma==4].severity_new)=={'extreme'}
    assert set(common[common.sigma==4].severity_old)=={'moderate'}
    for field in ['normalized_tensor_sha256','raw_input_sha256','model_input_sha256','p_waterbird','pred_y','mask_sha256']:
        assert np.array_equal(common[field+'_new'],common[field+'_old'])


def test_cached_predictions_are_bound_to_the_same_checkpoints(saved):
    out,_,_=saved
    info=json.loads((out/'checkpoint_audit.json').read_text(encoding='utf-8'))
    changed=copy.deepcopy(info);changed['POTER']['checkpoint_sha256']='0'*64
    with pytest.raises(ValueError,match='checkpoint/preprocess/runtime mismatch'):
        PairedPredictionCache(SimpleNamespace(info=changed),ROOT/'results/poter_failure',out)
    changed=copy.deepcopy(info);changed['DPE_legacy']['heads_sha256']='0'*64
    with pytest.raises(ValueError,match='DPE cache checkpoint changed'):
        PairedPredictionCache(SimpleNamespace(info=changed),ROOT/'results/poter_failure',out)


def test_background_ci_point_is_mean_absolute_paired_difference(saved):
    out,frame,_=saved
    bg=pd.read_csv(out/'background_sensitivity.csv');ci=pd.read_csv(out/'bootstrap_ci.csv')
    for model in frame.model_name.unique():
        for level in ['clean','mild','moderate','strong','extreme']:
            point=bg[(bg.model_name==model)&(bg.severity==level)&(bg.scope=='all')].iloc[0]
            boot=ci[(ci.model_name==model)&(ci.severity==level)&(ci.comparison=='within_model')&
                    (ci.condition=='natural')&(ci.metric=='mean_abs_delta_p_waterbird_background_swap')].iloc[0]
            assert point.mean_abs_delta_p_waterbird==pytest.approx(boot.estimate)
            assert point.mean_abs_delta_p_waterbird>=abs(point.mean_signed_delta_p_waterbird)-1e-12
    assert (ci.bootstrap_samples>=2000).all() and (ci.ci95_low<=ci.ci95_high).all()


def test_flips_pair_same_foreground_and_background_with_clean(saved):
    out,frame,_=saved
    flips=pd.read_csv(out/'paired_flips.csv')
    assert len(flips)==9000
    assert (flips.flip==flips.land_to_water+flips.water_to_land).all()
    clean=flips[(flips.pair_type=='clean_to_degraded')&(flips.severity=='clean')]
    assert not clean.flip.any() and np.allclose(clean.delta_p_waterbird,0)
    r=flips[(flips.model_name=='POTER')&(flips.pair_type=='clean_to_degraded')&(flips.severity=='mild')].iloc[0]
    paired=frame[(frame.model_name=='POTER')&(frame.source_image_id==r.source_image_id)&
                 (frame.background_id==r.background_id)&(frame.control=='natural')]
    baseline=paired[paired.severity=='clean'].iloc[0];corrupted=paired[paired.severity=='mild'].iloc[0]
    assert r.flip==int(baseline.pred_y!=corrupted.pred_y)
    assert r.delta_p_waterbird==pytest.approx(corrupted.p_waterbird-baseline.p_waterbird)


def test_contact_sheet_cohort_and_extreme_diagnostic_role(saved):
    out,frame,_=saved
    qc=pd.read_csv(out/'qc_foregrounds.csv')
    assert len(qc)==20 and qc.species_id.nunique()==20
    assert qc.true_y.value_counts().to_dict()=={0:10,1:10}
    metrics=pd.read_csv(out/'metrics.csv')
    assert set(metrics[metrics.severity_role=='main'].sigma)=={0,.5,1,2}
    assert set(metrics[metrics.severity=='extreme'].severity_role)=={'supplementary_diagnostic'}
    controls=metrics[metrics.control=='background_only']
    assert controls.semantic_accuracy.isna().all() and controls.wga.isna().all()
    reuse=pd.read_csv(out/'prediction_reuse_audit.csv').groupby('execution').rows.sum().to_dict()
    assert reuse=={'new_real_inference':4500,'reused_exact_previous':4500}
