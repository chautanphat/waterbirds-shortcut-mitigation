import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.special import softmax
import torch
from torch.utils.data import DataLoader,TensorDataset

sys.path.insert(0,str(Path('baselines/dpe4subpop/src').resolve()))
from dpe import DPE
from dpe.geometry import distance_mix,read_metadata


def test_frozen_metadata_invariants_and_no_label_api(tmp_path):
    metadata=read_metadata();p=tmp_path/'metadata.json';p.write_text(json.dumps(metadata),encoding='utf-8');assert read_metadata(p)==metadata
    assert set(inspect.signature(distance_mix).parameters)=={'probabilities','distances','metadata'}
    d=np.ones((3,15,2),np.float32);p=np.full_like(d,.5);mix,w=distance_mix(p,d)
    np.testing.assert_allclose(mix,.5);np.testing.assert_allclose(w,1/15);assert (w>0).all()
    with pytest.raises(ValueError):read_metadata(dict(metadata,gamma=20.))
    with pytest.raises(ValueError):distance_mix(p[:,:14],d[:,:14])
    with pytest.raises(ValueError):distance_mix(p,np.full_like(d,np.inf))
    with pytest.raises(TypeError):distance_mix(p,d,y=np.zeros(3))


def test_production_matches_all_nine_verified_C_caches():
    root=Path('outputs/geometry_aggregation_v5/run1/nested')
    if not root.exists():pytest.skip('Run the v5 nested study first')
    for folder in root.glob('seed*/fold*'):
        with np.load(folder/'member_predictions.npz',allow_pickle=False) as z,np.load(folder/'G1_g50/predictions.npz',allow_pickle=False) as r:
            p,w=distance_mix(z['C_probs'],z['C_distances']);np.testing.assert_allclose(p,r['probs'],atol=1e-12,rtol=0);np.testing.assert_allclose(w,r['weights'],atol=1e-12,rtol=0)
            np.testing.assert_array_equal(p.argmax(1),r['probs'].argmax(1))
            legacy=torch.from_numpy(z['C_probs']).transpose(0,1).mean(0).numpy()
            np.testing.assert_allclose(legacy,z['C_probs'].mean(1),atol=2e-7,rtol=0)
            np.testing.assert_array_equal(legacy.argmax(1),z['C_probs'].mean(1).argmax(1))


def test_DPE_real_frozen_head_API_and_uniform_unchanged():
    folder=Path('outputs/geometry_aggregation_v5/run1/nested/seed0/fold0')
    if not folder.exists():pytest.skip('Run the v5 nested study first')
    torch.set_num_threads(1)
    from experiments.weighting.data import standardized
    plan=json.loads((folder/'ABC_manifest.json').read_text());idx=np.array(plan['C_indices'][:5])
    with np.load('outputs/weighting/features/val.npz',allow_pickle=False) as z:features=standardized(z['embedding'][idx])
    ds=TensorDataset(features);ds.num_labels=2
    model=DPE.__new__(DPE);model.config=SimpleNamespace(emb_dim=2048,loss_name='isomax',device='cpu');model.datasets={'val':ds};model.loaders={'val':DataLoader(ds,batch_size=5)}
    heads=[torch.load(folder/f'head{j:02d}/head.pt',weights_only=True) for j in range(15)]
    model.ensemble=[(h['prototypes'][:,None,:],h['scale']) for h in heads]
    with np.load(folder/'member_predictions.npz',allow_pickle=False) as z,np.load(folder/'G1_g50/predictions.npz',allow_pickle=False) as saved:
        uniform=model.predict_proba(target='val');members=model.predict_proba(target='val',avg=False)
        np.testing.assert_array_equal(uniform.numpy(),members.mean(0).numpy())
        np.testing.assert_allclose(uniform.numpy(),z['C_probs'][:5].mean(1),atol=2e-7,rtol=0)
        np.testing.assert_array_equal(model.predict_proba(target='val',avg=False,raw_logits=True).numpy().transpose(1,0,2),z['C_logits'][:5])
        routed=model.predict_proba(target='val',aggregation='distance');np.testing.assert_allclose(routed.numpy(),saved['probs'][:5],atol=1e-12,rtol=0)
    with pytest.raises(ValueError):model.predict_proba(target='val',aggregation='distance',raw_logits=True)
    assert inspect.signature(DPE.predict_proba).parameters['aggregation'].default=='uniform'


def test_prediction_cache_CLI_defaults_uniform_and_opt_in(tmp_path):
    rng=np.random.default_rng(51);dist=rng.uniform(.5,1.5,(7,15,2)).astype(np.float32);p=softmax(-dist,axis=-1);cache=tmp_path/'cache.npz'
    np.savez_compressed(cache,probs=p,distances=dist,sample_id=np.arange(7).astype(str),group=np.full(7,-999))
    env=os.environ|dict(PYTHONPATH=str(Path('baselines/dpe4subpop/src').resolve()),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for mode in ['uniform','distance']:
        target=tmp_path/(mode+'.npz');cmd=[sys.executable,'-m','dpe.geometry','--input-cache',str(cache),'--output',str(target)]
        if mode=='distance':cmd+=['--aggregation','distance']
        subprocess.run(cmd,env=env,check=True,capture_output=True)
        with np.load(target,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['probs'],p.mean(1) if mode=='uniform' else distance_mix(p,dist)[0]);assert 'group' not in z and 'y' not in z
