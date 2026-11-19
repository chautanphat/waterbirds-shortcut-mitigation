"""CPU-only oracle arithmetic, audit guards, tie rules and frozen-score transfer."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.run_test_oracle_threshold_sweep import (validate_natural,score,winner,exact_candidates,original_data)

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def source():
    return (pd.read_csv(ROOT/'outputs/mild_blur_v2/predictions.csv',float_precision='round_trip'),
            pd.read_csv(ROOT/'data/waterbirds/metadata.csv'))


def test_control_filtering_and_fixed05_parity(source):
    raw,metadata=source;natural,audit=validate_natural(raw,metadata)
    assert len(natural)==6000 and set(natural.control)=={'natural'}
    assert audit['excluded_by_control']=={'foreground_only':1500,'background_only':1200,'original':300}
    for sigma,expected in [(0.,86),(.5,83),(1.,74),(2.,58),(4.,23)]:
        rows=natural[(natural.model_name=='POTER')&(natural.sigma==sigma)]
        r=score(rows,.5)
        assert r['wga_pct']==pytest.approx(expected)
        assert [r[f'group{g}_n'] for g in range(4)]==[100]*4
        assert ((rows.p_waterbird>=.5).astype(int)==rows.pred_y).all()


@pytest.mark.parametrize('problem',['invalid_label','missing_score','duplicate','missing_pair','invalid_background'])
def test_invalid_records_are_not_silently_dropped(source,problem):
    frame,metadata=source;changed=frame.copy()
    first=changed.index[(changed.model_name=='POTER')&(changed.control=='natural')][0]
    if problem=='invalid_label':changed.loc[first,'true_y']=2
    elif problem=='missing_score':changed.loc[first,'p_waterbird']=np.nan
    elif problem=='duplicate':changed=pd.concat([changed,changed.loc[[first]]],ignore_index=True)
    elif problem=='missing_pair':changed=changed.drop(first)
    elif problem=='invalid_background':changed.loc[first,'bg_type']='neutral'
    with pytest.raises(ValueError):validate_natural(changed,metadata)


def test_wga_is_minimum_group_accuracy_not_minimum_image_correctness():
    data=pd.DataFrame(dict(source_image_id=range(8),true_y=[0,0,0,0,1,1,1,1],
        original_group=[0,0,1,1,2,2,3,3],control=['original']*8,
        p_waterbird=[.1,.6,.2,.3,.8,.9,.7,.8]))
    result=score(data,.5)
    assert result['wga_pct']==50 and result['aa_pct']==87.5


def test_stable_tie_break_and_greater_equal_rule():
    rows=[dict(wga_pct=80.,aa_pct=85.,threshold=t) for t in [.6,.4]]
    assert winner(rows)['threshold']==.4
    rows.append(dict(wga_pct=80.,aa_pct=86.,threshold=.7))
    assert winner(rows)['threshold']==.7
    values=np.array([0.,.5,.5,1.])
    candidates=exact_candidates(values)
    attainable={tuple((values>=t).astype(int)) for t in candidates}
    dense={tuple((values>=t).astype(int)) for t in np.linspace(0,1,1001)}
    assert dense<=attainable
    assert tuple((values>=.5).astype(int))==(0,1,1,1)
    assert tuple((values>=np.nextafter(.5,np.inf)).astype(int))==(0,0,0,1)


def test_exact_boundary_wga_independently_reconciles(source):
    raw,_=source;out=ROOT/'outputs/test_oracle_threshold_sweep'
    data=raw[(raw.model_name=='DPE_legacy')&(raw.control=='natural')&(raw.sigma==1.)]
    labels=data.true_y.to_numpy(int)
    groups=2*labels+(data.bg_type=='water').to_numpy(int)
    probabilities=data.p_waterbird.to_numpy()
    maximum=0.
    for threshold in exact_candidates(probabilities):
        correct=(probabilities>=threshold)==labels
        maximum=max(maximum,min(correct[groups==g].mean() for g in range(4)))
    exact=pd.read_csv(out/'exact_best_thresholds.csv')
    saved=exact[(exact.model_name=='DPE_legacy')&(exact.sigma==1.)].iloc[0]
    assert saved.wga_pct==pytest.approx(100*maximum)
    assert saved.wga_pct>=saved.grid_oracle_wga_pct


def test_global_oracles_are_frozen_and_joint_uses_only_primary_sigmas():
    out=ROOT/'outputs/test_oracle_threshold_sweep'
    sweep=pd.read_csv(out/'sweep_metrics.csv');comparison=pd.read_csv(out/'strategy_comparison.csv')
    for model,rows in comparison.groupby('model_name'):
        for strategy in ['fixed_05','fixed_065','oracle_clean_transfer','oracle_joint']:
            selected=rows[rows.strategy==strategy]
            assert selected.threshold.nunique()==1
        primary=sweep[(sweep.model_name==model)&sweep.sigma.isin([0.,.5,1.,2.])]
        means=primary.groupby('threshold').wga_pct.mean()
        chosen=rows[rows.strategy=='oracle_joint'].threshold.iloc[0]
        assert means.loc[chosen]==pytest.approx(means.max())
    assert len(comparison[comparison.dataset=='full_original_test5794'])==12
    assert not comparison.validation_selected.any() and not comparison.evaluation_is_untouched.any()


def test_full_original_transfer_no_substitution_or_retuning(source,tmp_path):
    _,metadata=source
    missing,status=original_data(tmp_path/'absent.csv',['POTER'],metadata)
    assert missing=={} and status['POTER']['status']=='NOT AVAILABLE'
    original=pd.read_csv(ROOT/'results/poter_failure/original_test_predictions.csv',float_precision='round_trip')
    original=original[original.model_name=='POTER']
    pred=(original.p_waterbird>=.65).to_numpy(int)
    correct=pred==original.true_y.to_numpy(int);groups=original.original_group.to_numpy(int)
    wga=min(correct[groups==g].mean() for g in range(4))
    comparison=pd.read_csv(ROOT/'outputs/test_oracle_threshold_sweep/strategy_comparison.csv')
    row=comparison[(comparison.model_name=='POTER')&(comparison.dataset=='full_original_test5794')&(comparison.strategy=='fixed_065')].iloc[0]
    assert row['count']==5794 and row.wga_pct==pytest.approx(100*wga)
    assert row.clean_drop_gt2pp and row.clean_WGA_drop_vs_fixed05>8


def test_reusable_script_import_never_loads_torch_or_model():
    result=subprocess.run([sys.executable,'-c',
        "import scripts.run_test_oracle_threshold_sweep; import sys; assert 'torch' not in sys.modules"],
        cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
