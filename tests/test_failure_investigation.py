"""Statistics/design checks; saved real-output checks skip if assets unavailable."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.counterfactual.common import ROOT,read_csv
from src.counterfactual.investigation import select_cohort
from src.counterfactual.failure_statistics import (stratified_draws,metric_tables,
    validate_predictions,bootstrap_comparison,paired_flips)


def test_cluster_draws_keep_class_counts_and_whole_foregrounds():
    labels=np.array([0,0,0,1,1])
    a,b=stratified_draws(labels,200,42)
    assert a.shape==(200,3) and b.shape==(200,2)
    assert (labels[a]==0).all() and (labels[b]==1).all()
    assert np.array_equal(a,stratified_draws(labels,200,42)[0])
    assert (a[:,0]==a[:,1]).any()  # Genuine resampling, rather than permutations.
    with pytest.raises(ValueError):stratified_draws([0,0],200,42)


def test_species_sampling_is_balanced_deterministic_and_prediction_free():
    config=dict(sampling_seed=23,foregrounds_per_class=50,species_per_class=10,
                known_mislabeled_species=['bad'])
    rows=[dict(bird_source_id=str(1000*y+10*s+i),foreground_id=f'{y}-{s}-{i}',
               species=f'{y}/species{s}',y=str(y),original_split='test',quality_status='verified_original_cub')
          for y in (0,1) for s in range(12) for i in range(6)]
    selected=select_cohort(rows,config,['0-0-0'])
    assert selected==select_cohort(list(reversed(rows)),config,['0-0-0'])
    assert len(selected)==100 and '0-0-0' not in {r['foreground_id'] for r in selected}
    for y in (0,1):
        subset=[r for r in selected if r['y']==str(y)]
        assert len(subset)==50 and len({r['species'] for r in subset})==10
        assert set(pd.DataFrame(subset).groupby('species').size())=={5}


@pytest.fixture
def saved():
    root=ROOT/'results/poter_failure'
    if not (root/'timings.json').exists():pytest.skip('Saved real three-method run not present')
    return pd.read_csv(root/'predictions.csv'),pd.read_csv(root/'cohort.csv')


def test_real_outputs_reconcile_and_tampering_is_rejected(saved):
    frame,cohort=saved
    assert validate_predictions(frame,cohort)['rows']==7500
    with pytest.raises(ValueError,match='Missing or duplicated'):
        validate_predictions(pd.concat([frame,frame.iloc[:1]]),cohort)
    changed=frame.copy();changed.loc[0,'normalized_tensor_sha256']='tampered'
    with pytest.raises(ValueError,match='mismatch'):validate_predictions(changed,cohort)


def test_background_only_is_ood_and_neutral_has_no_wga(saved):
    frame,_=saved
    summary,_,_=metric_tables(frame)
    bg=summary[summary.control=='background_only']
    assert bg.semantic_accuracy.isna().all() and bg.wga.isna().all()
    assert summary[summary.control=='foreground_only'].wga.isna().all()
    assert set(bg.unique_raw_images)=={4}


def test_paired_bootstrap_identical_methods_gives_exact_zero(saved):
    frame,_=saved
    selected=frame[frame.model_name=='POTER']
    # Reuse actual predictions identically for each name to check paired draws.
    equal=pd.concat([selected.assign(model_name=m) for m in ['POTER','ERM_enhanced','DPE_legacy']],ignore_index=True)
    ci=bootstrap_comparison(equal,samples=200,seed=42)
    compared=ci[ci.comparison!='within_model']
    assert (compared[['estimate','ci95_low','ci95_high']]==0).all().all()
    point=ci[(ci.comparison=='within_model')&(ci.metric=='wga')]
    summary=metric_tables(equal)[0]
    for r in point.itertuples():
        expected=summary[(summary.model_name==r.model_name)&(summary.severity==r.severity)&(summary.control=='natural')].iloc[0].wga
        assert r.estimate==pytest.approx(expected)


def test_flip_directions_and_clean_reference(saved):
    frame,_=saved
    flips=paired_flips(frame)
    assert (flips.flip==flips.land_to_water+flips.water_to_land).all()
    clean=flips[(flips.pair_type=='clean_to_degraded')&(flips.severity=='clean')]
    assert (clean.flip==0).all() and np.allclose(clean.delta_p_waterbird,0)
    assert len(flips)==7200
