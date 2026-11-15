"""Standalone verifier for the minimal DPE v5 analysis export (NumPy only)."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_record(value):
    value=np.asarray(value)
    return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())


def verify(root,reproduce_g1=False):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    expected=set(manifest['files'])|{'manifest.json'}
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert actual==expected,('Missing/unexpected files',expected-actual,actual-expected)
    for relative,record in manifest['files'].items():
        p=root/relative;assert p.stat().st_size==record['bytes'] and digest(p)==record['sha256'],relative
    with (root/'provenance/development_index.csv').open(encoding='utf-8',newline='') as f:index=list(csv.DictReader(f))
    assert len(index)==1199 and len({r['sample_id'] for r in index})==1199
    ids=np.array([r['sample_id'] for r in index]);y=np.array([int(r['y']) for r in index]);group=np.array([int(r['group']) for r in index])
    assert np.all(group//2==y)
    if reproduce_g1:
        spec=importlib.util.spec_from_file_location('frozen_v5_geometry',root/'source/geometry.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    counts=[];all_ids=[]
    for seed in range(3):
        seen=[]
        for fold in range(3):
            folder=root/f'caches/seed{seed}/fold{fold}'
            plan=json.loads((folder/'ABC_manifest.json').read_text(encoding='utf-8'));head=json.loads((folder/'head_manifest.json').read_text(encoding='utf-8'));prov=json.loads((folder/'export_provenance.json').read_text(encoding='utf-8'))
            assert plan['seed']==seed and plan['fold']==fold
            sets=[set(plan[k]) for k in ['A','B','C']];assert all(not sets[a]&sets[b] for a in range(3) for b in range(a+1,3))
            assert set.union(*sets)==set(ids)
            for key in ['A','B','C']:np.testing.assert_array_equal(ids[plan[key+'_indices']],plan[key])
            z=dict(np.load(folder/'C_predictions.npz',allow_pickle=False));n=[400,400,399][fold]
            assert z['probs'].shape==z['distances'].shape==z['logits'].shape==(n,15,2)
            for key in ['sample_id','y','group','seed','fold']:assert z[key].shape==(n,)
            assert z['head_sha256'].shape==z['head_id'].shape==z['head_abs_scale'].shape==(15,)
            assert np.all(z['seed']==seed) and np.all(z['fold']==fold)
            np.testing.assert_array_equal(z['classes'],[0,1]);np.testing.assert_array_equal(z['head_id'],[f'head-{j:02d}' for j in range(15)])
            np.testing.assert_array_equal(z['sample_id'],plan['C']);np.testing.assert_array_equal(z['y'],y[plan['C_indices']]);np.testing.assert_array_equal(z['group'],group[plan['C_indices']])
            assert [r['slot'] for r in head['ordered_heads']]==list(range(15)) and head['B_C_same_order']
            np.testing.assert_array_equal(z['head_sha256'],[r['sha256'] for r in head['ordered_heads']])
            assert head['cache_sha256']==prov['original_member_predictions']['sha256']
            assert all(np.isfinite(z[k]).all() for k in ['probs','distances','logits','head_abs_scale'])
            assert (z['probs']>=0).all() and (z['distances']>=0).all() and (z['distances']<=2+2e-6).all()
            np.testing.assert_allclose(z['probs'].sum(-1),1,atol=2e-6,rtol=0)
            exp=np.exp(z['logits']-z['logits'].max(-1,keepdims=True));reconstructed=exp/exp.sum(-1,keepdims=True)
            np.testing.assert_allclose(reconstructed,z['probs'],atol=2e-6,rtol=0)
            np.testing.assert_allclose(z['logits'],-z['head_abs_scale'][None,:,None]*z['distances'],atol=2e-6,rtol=0)
            for key,record in prov['exported_arrays'].items():assert array_record(z[key])==record,key
            if reproduce_g1:
                # Frozen inference checksum only. No metrics, search, labels or groups enter this call.
                mixture,weights=module.distance_mix(z['probs'],z['distances'],root/'source/verified_distance_v1.json')
                assert array_record(mixture)==prov['saved_g1_reference']['probs'],'G1 bitwise reproduction mismatch; check recorded NumPy/platform versions'
                assert array_record(weights)==prov['saved_g1_reference']['weights'],'G1 weight reproduction mismatch'
            seen.extend(z['sample_id'].tolist());counts.append(dict(seed=seed,fold=fold,samples=n));all_ids.extend(z['sample_id'].tolist())
        assert len(seen)==len(set(seen))==1199 and set(seen)==set(ids)
    assert len(list((root/'caches').glob('seed*/fold*/C_predictions.npz')))==9
    return dict(status='PASS',caches=9,seed_image_rows=len(all_ids),distinct_images=len(set(all_ids)),folds=counts,
        shapes_labels_splits_and_head_order='PASS',frozen_g1_reproduction='PASS' if reproduce_g1 else 'NOT_RUN',
        models_trained=0,experiments_rerun=0,heldout_labels_for_tuning=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);parser.add_argument('--reproduce-g1',action='store_true');args=parser.parse_args()
    print(json.dumps(verify(args.root,args.reproduce_g1),indent=2))
