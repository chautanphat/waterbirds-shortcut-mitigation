"""Original-RGB pairing, soft-mask semantics and locked-policy evaluation."""
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.counterfactual.original_gray import build_gray,decode_pair
from src.counterfactual.fdat_feasibility import fdat
from src.counterfactual.common import sha256
from scripts.run_original_gray_feasibility import choose_preflight,dependencies
from scripts.report_original_gray_feasibility import summarize,paired_bootstrap

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/original_gray_feasibility'


def fixtures():
    # Numerical pixel fixtures only, never used as experimental images/predictions.
    yy,xx=np.mgrid[:256,:256]
    rgb=Image.fromarray(np.stack([xx,yy,(xx//2+yy//2)],axis=2).astype('uint8'))
    alpha=np.zeros((256,256),dtype='uint8');alpha[50:206,50:206]=128;alpha[75:181,75:181]=255
    return rgb,Image.fromarray(alpha)


def test_exact_opaque_foreground_and_gray_outside_with_soft_boundaries():
    rgb,mask=fixtures();_,full,gray,alpha,audit=build_gray(rgb,mask)
    a=np.asarray(alpha);x=np.asarray(full);g=np.asarray(gray)
    assert np.array_equal(x[a==255],g[a==255]) and (g[a==0]==127).all()
    assert (a==128).any() and audit['soft_boundary_pixels']>0
    assert audit['alpha_matches_historical_resize'] and audit['soft_blend_rounding_error_max']<=1
    assert rgb.size==mask.size==(256,256)  # Inputs not mutated.


def test_soft_only_masks_are_not_hardened_or_dropped():
    rgb,mask=fixtures();a=np.asarray(mask).copy();a[a>0]=204
    _,full,gray,alpha,audit=build_gray(rgb,Image.fromarray(a))
    assert audit['opaque_pixels']==audit['core_pixels']==0
    assert audit['soft_boundary_pixels']>0 and np.asarray(alpha).max()==204
    assert not np.array_equal(np.asarray(full),np.asarray(gray))


@pytest.mark.parametrize('mask_kind',['empty','full','wrong_size'])
def test_invalid_masks_fail_instead_of_silent_alignment(mask_kind):
    rgb,mask=fixtures()
    if mask_kind=='empty':mask=Image.new('L',rgb.size,0)
    elif mask_kind=='full':mask=Image.new('L',rgb.size,255)
    else:mask=Image.new('L',(255,256),0)
    with pytest.raises(ValueError):build_gray(rgb,mask)


def test_decode_does_not_guess_native_mask_resolution(tmp_path):
    rgb,mask=fixtures();rgb.save(tmp_path/'source.png');mask.resize((128,128)).save(tmp_path/'mask.png')
    with pytest.raises(ValueError,match='dimensions differ'):decode_pair(tmp_path/'source.png',tmp_path/'mask.png')


def test_generation_and_fixed_inference_do_not_accept_labels_or_groups():
    assert list(inspect.signature(build_gray).parameters)==['rgb','mask','gray_rgb']
    assert list(inspect.signature(fdat).parameters)==['p_full','p_gray','lambda_','t_max']
    result=fdat([.5,1.,.4],[.5,.1,.9],1.,1.)
    assert result['pred'].tolist()==[1,1,0]  # Saturated full=1 exception is retained.


@pytest.fixture(scope='module')
def pairs():
    return pd.read_csv(OUT/'original_gray_pairs.csv',float_precision='round_trip')


def test_full5794_one_to_one_and_original_group_denominators(pairs):
    assert len(pairs)==pairs.source_image_id.nunique()==5794
    assert pairs.original_group.value_counts().to_dict()=={0:2255,1:2255,2:642,3:642}
    assert pairs.opaque_rgb_exact.all() and pairs.core_rgb_exact.all() and pairs.outside_gray_exact.all()
    assert pairs.alpha_matches_historical_resize.all()
    assert pairs.source_file_sha256.eq(pairs.source_file_sha256_inference).all()
    assert pairs.mask_sha256.eq(pairs.mask_sha256_inference).all()
    assert pairs.gray_model_input_sha256.eq(pairs.gray_model_input_sha256_inference).all()
    assert np.isfinite(pairs[['p_full','p_gray']]).all().all()
    assert np.allclose(pairs.p_gray+pairs.p_gray_landbird,1,atol=1e-6,rtol=0)


def test_preflight_selection_ignores_probabilities_and_matches_cache():
    frame=pd.read_csv(OUT/'dataset_audit.csv',float_precision='round_trip')
    c=json.loads((OUT/'run_config.json').read_text())['config']
    selected=choose_preflight(frame,c)
    shuffled=frame.copy();shuffled['p_full']=1-shuffled.p_full;shuffled['full_logit_waterbird']=100.
    assert choose_preflight(shuffled,c).source_image_id.tolist()==selected.source_image_id.tolist()
    assert len(selected)==selected.source_image_id.nunique()==24
    assert (selected.groupby('original_group').size()==6).all()
    pre=json.loads((OUT/'preflight.json').read_text())
    assert pre['maximum_probability_error']<=pre['probability_tolerance']
    assert pre['maximum_logit_error']<=pre['logit_tolerance']
    _,digest=dependencies(c,OUT)
    assert digest==pre['cache_dependency_sha256']
    review=json.loads((OUT/'preflight_visual_review.json').read_text())
    assert review['status']=='PASS' and review['images_reviewed']==24


def test_full_baseline_and_locked_policies_independently_reconcile(pairs):
    original=pd.read_csv(ROOT/'results/poter_failure/original_test_predictions.csv',float_precision='round_trip')
    original=original[original.model_name=='POTER'].set_index('source_image_id').loc[pairs.source_image_id]
    assert np.array_equal(original.p_waterbird.to_numpy(),pairs.p_full.to_numpy())
    assert np.array_equal(original.model_input_sha256.to_numpy(),pairs.original_model_input_sha256.to_numpy())
    assert np.array_equal(pairs.gray_05_pred,(pairs.p_gray>=.5).astype(int))
    result=fdat(pairs.p_full,pairs.p_gray,1.,1.)
    assert np.array_equal(result['pred'],pairs.fdat_pred)
    differences=(pairs.fdat_pred!=pairs.and_pred)
    assert np.array_equal(differences,((pairs.p_full==1)&(pairs.p_gray<.5)))
    summary=summarize(pairs).set_index('policy')
    correct=pairs.full_05_pred==pairs.true_y
    wga=min(correct[pairs.original_group==g].mean() for g in range(4))*100
    assert wga==pytest.approx(summary.loc['full_05','wga_pct'])
    assert round(wga,2)==91.04 and round(100*correct.mean(),2)==91.99


def test_bootstrap_and_gate_use_paired_original_sources_without_retuning(pairs):
    ci=pd.read_csv(OUT/'bootstrap_ci.csv')
    assert ci.bootstrap_samples.min()>=2000 and (ci.n_sources==5794).all()
    assert not ci.parameter_reselected.any() and not ci.independent_final_test.any()
    baseline=ci[(ci.policy=='full_05')&(ci.comparison=='minus_full05')]
    assert (baseline[['estimate_pct_or_pp','ci95_low_pct_or_pp','ci95_high_pct_or_pp']]==0).all().all()
    metrics=pd.read_csv(OUT/'original_gray_metrics.csv').set_index('policy')
    for name in ['gray_05','locked_FDAT']:
        r=metrics.loc[name]
        assert r.gate==('PASS' if r.delta_wga_vs_full05_pp>=-2-1e-10 else 'FAIL')
    with pytest.raises(ValueError):paired_bootstrap(pairs,samples=100)


def test_inference_cache_has5794_new_gray_views_not_old_composite_scores(pairs):
    run=json.loads((OUT/'inference_run.json').read_text())
    assert run['total_gray_images']==5794 and run['preflight_original_forwards']==24
    assert run['preflight_gray_forwards']+sum(r['new_gray_forwards'] for r in run['runs'])==5794
    assert pairs.cache_dependency_sha256.nunique()==1
    assert not (OUT/'blocking_status.json').exists()
