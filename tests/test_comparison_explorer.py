"""Acceptance tests on saved real three-model predictions; never loads a model."""
import csv
import io
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import pytest
from PIL import Image
import torch

from src.counterfactual.common import ROOT,pixel_hash
from src.counterfactual.comparison_explorer import ComparisonStore,create_handler
from src.counterfactual.server import LocalHTTPServer


@pytest.fixture(scope='module')
def store():
    if not (ROOT/'results/poter_failure/integrity.json').exists():
        pytest.skip('Saved real three-model run is not present')
    torch.set_num_threads(4)
    return ComparisonStore(ROOT/'results/poter_failure',ROOT/'data/waterbirds')


def test_real_coverage_and_all_three_models(store):
    info=store.info()
    assert info['inference'] is False
    assert info['cohorts']['original']['prediction_rows']==17382
    assert info['cohorts']['replication']['prediction_rows']==7500
    assert info['cohorts']['original']['failures_by_model']==dict(POTER=464,ERM_enhanced=444,DPE_legacy=229)
    assert len(store.catalog['original'])==5794


def test_reconstructed_pixels_match_saved_real_images_and_crop_hash(store):
    bird='654'
    for suffix in ['original','clean_hf-bg-11699','strong_hf-bg-7032']:
        image_id=bird+'_'+suffix
        image=Image.open(io.BytesIO(store.asset('replication',image_id,'raw')))
        with Image.open(ROOT/'results/poter_failure/examples/hf-fg-654'/(image_id+'.png')) as reference:
            assert image.tobytes()==reference.tobytes()
        model=Image.open(io.BytesIO(store.asset('replication',image_id,'model')))
        assert model.size==(224,224)
        assert pixel_hash(model)==store.images['replication'][image_id]['model_input_sha256']


def test_no_forged_or_unknown_assets_and_csv_reconciliation(store):
    image_id='654_strong_hf-bg-11699'
    record=store.images['replication'][image_id]
    original=record['model_input_sha256']
    store.asset.cache_clear()
    try:
        record['model_input_sha256']='0'*64
        with pytest.raises(ValueError,match='224 pixels'):store.asset('replication',image_id,'model')
    finally:
        record['model_input_sha256']=original;store.asset.cache_clear()
    with pytest.raises(KeyError):store.asset('replication','../metadata.csv','raw')
    rows=list(csv.DictReader(io.StringIO(store.download('replication','654').decode('utf-8-sig'))))
    assert len(rows)==75 and {r['model_name'] for r in rows}=={'POTER','ERM_enhanced','DPE_legacy'}
    assert all(not r['true_y'] and not r['is_correct'] for r in rows if r['control']=='background_only')


def test_http_gallery_is_read_only_and_serves_real_images(store):
    server=LocalHTTPServer(('127.0.0.1',0),create_handler(store))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with urllib.request.urlopen(base) as response:assert b'comparison.js' in response.read()
        with urllib.request.urlopen(base+'/api/info') as response:assert json.load(response)['inference'] is False
        with urllib.request.urlopen(base+'/api/asset?cohort=replication&id=654_strong_hf-bg-11699&view=model') as response:
            assert response.headers['Content-Type']=='image/png'
            assert Image.open(io.BytesIO(response.read())).size==(224,224)
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(base+'/api/info',headers={'Origin':'https://example.com'}))
        assert error.value.code==403
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(base+'/api/run',data=b'{}'))
        assert error.value.code==405
    finally:server.shutdown();server.server_close();thread.join()


def test_mild_blur_saved_gallery_has_five_sigmas_and_original_subset():
    out=ROOT/'outputs/mild_blur_v2'
    if not (out/'integrity.json').exists():pytest.skip('Mild blur v2 results not present')
    gallery=ComparisonStore(out,ROOT/'data/waterbirds')
    info=gallery.info()
    assert info['cohorts']['replication']['prediction_rows']==9000
    assert info['cohorts']['original']['source_foregrounds']==100
    assert info['supplementary']==['extreme']
    row=next(r for r in gallery.catalog['replication'] if r['control']=='natural' and r['severity']=='mild')
    image=Image.open(io.BytesIO(gallery.asset('replication',row['image_id'],'model')))
    assert pixel_hash(image)==row['model_input_sha256'] and row['sigma']==.5
