"""Paired-intervention and missing-seed integrity checks on saved real outputs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.counterfactual.common import ROOT,read_csv,sha256
from src.counterfactual.failure_verification import paired_interventions,summarize_bootstrap,MODELS


@pytest.fixture(scope='module')
def saved():
    out=ROOT/'outputs/failure_verification'
    if not (out/'statistical_analysis.csv').exists():pytest.skip('Completed cache-only verification outputs not present')
    return out,pd.read_csv(ROOT/'outputs/mild_blur_v2/predictions.csv',float_precision='round_trip')


def test_pair_coverage_keeps_both_replicates_and_foreground_fixed(saved):
    out,_=saved;pairs=pd.read_csv(out/'exp1_paired_interventions.csv')
    assert len(pairs)==7200
    assert set(pairs.replicate)=={0,1}
    assert (pairs.groupby(['model_name','source_image_id','sigma','contrast']).size()==2).all()
    assert not pairs.duplicated(['model_name','model_seed','source_image_id','sigma','contrast','replicate']).any()
    assert (pairs.groupby(['model_name','source_image_id','sigma']).layer_sha256.nunique()==1).all()
    assert (pairs.flip==pairs.land_to_water_prediction_flip+pairs.water_to_land_prediction_flip).all()
    assert np.allclose(pairs.signed_probability_shift,pairs.p_waterbird_to-pairs.p_waterbird_from)
    assert np.allclose(pairs.absolute_probability_shift,pairs.signed_probability_shift.abs())
    assert np.allclose(pairs.margin_shift,pairs.margin_to-pairs.margin_from)


def test_changed_mask_or_missing_pair_is_rejected(saved):
    _,frame=saved;changed=frame.copy()
    index=changed[(changed.model_name=='POTER')&(changed.control=='natural')].index[0]
    changed.loc[index,'layer_sha256']='0'*64
    with pytest.raises(ValueError,match='changed foreground/mask'):paired_interventions(changed)
    with pytest.raises(ValueError,match='Missing gray/natural'):paired_interventions(frame.drop(index))


def test_interference_and_amplification_reconcile_to_independent_csv_query(saved):
    out,frame=saved;effects=pd.read_csv(out/'exp1_foreground_effects.csv')
    source=frame[(frame.model_name=='POTER')&(frame.true_y==0)]
    values={}
    for sigma in (0.,2.):
        d=source[source.sigma==sigma]
        natural=d[d.control=='natural'];gray=d[d.control=='foreground_only']
        values[sigma]=(natural.pred_y==natural.true_y).mean()-(gray.pred_y==gray.true_y).mean()
    e=effects[(effects.model_name=='POTER')&(effects.true_y==0)&(effects.sigma==2)]
    assert e.interference.mean()==pytest.approx(values[2])
    assert e.amplification.mean()==pytest.approx(values[2]-values[0])
    assert values[2]==pytest.approx(-.21)
    assert np.allclose(e.delta_background,e.p_water-e.p_land)
    assert e.delta_background.mean()<0  # Do not reverse the observed direction to fit a water-cue narrative.


def test_paired_model_bootstrap_identical_observations_have_zero_difference(saved):
    out,_=saved;fg=pd.read_csv(out/'exp1_foreground_effects.csv')
    actual=fg[fg.model_name=='POTER']
    matched=pd.concat([actual.assign(model_name=model,model_seed=0) for model in MODELS],ignore_index=True)
    stats=summarize_bootstrap(matched,samples=200,seed=20261118)
    compared=stats[stats.comparison!='within_seed']
    assert np.allclose(compared[['estimate','ci95_low','ci95_high']],0)


def test_partial_multiseed_outputs_do_not_fake_missing_seeds_or_std(saved):
    out,_=saved
    manifest=json.loads((out/'checkpoint_manifest.json').read_text())
    assert manifest['available_method_checkpoints']==3 and manifest['target_checkpoints']==9
    assert sum(len(v) for v in manifest['missing_seeds'].values())==6
    assert manifest['training']['new_checkpoints_trained']==0
    across=pd.read_csv(out/'exp2_across_seed_summary.csv')
    assert (across.available_seeds==1).all() and across.training_seed_std.isna().all()
    rows=read_csv(out/'exp2_predictions.csv')
    old={ (r['model_name'],r['image_id']):r for r in read_csv(ROOT/'outputs/mild_blur_v2/predictions.csv') }
    assert len(rows)==6300
    assert all(row==old[(row['model_name'],row['image_id'])] for row in rows)
    assert all(r['control']!='background_only' and r['severity']!='extreme' for r in rows)


def test_foreground_and_species_uncertainty_is_not_training_seed_uncertainty(saved):
    out,_=saved;stats=pd.read_csv(out/'statistical_analysis.csv')
    assert set(stats.bootstrap_samples)=={2000}
    assert stats.training_seed_std.isna().all() and (stats.training_seed_n==1).all()
    assert stats.bootstrap_unit.str.startswith('species_id').any()
    assert stats.bootstrap_unit.str.startswith('source_image_id').any()
    assert (stats.ci95_low<=stats.ci95_high).all()
    h=pd.read_csv(out/'hypothesis_evaluation.csv').set_index('hypothesis')
    assert h.loc['H4','status']=='Inconclusive'
    assert json.loads((out/'decision.json').read_text())['decision']=='NO-GO / INCONCLUSIVE'
