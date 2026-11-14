"""Deliver existing development OOF outputs, embeddings and evaluation groups."""
import csv
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from experiments.discrete_weighting.store import atomic,dump,sha


def main():
    source=Path('outputs/weighting');out=source/'development_bundle';out.mkdir(exist_ok=True)
    with np.load(source/'features/val.npz',allow_pickle=False) as f:val=dict(f)
    lookup={s:i for i,s in enumerate(val['sample_id'])}
    assert len(lookup)==len(val['y'])==1199
    assert np.all((val['group']//2)==val['y'])
    entries=[];manifest_sources={};audit_rows=[]
    for seed in (0,1,2):
        audit_path=source/f'seed{seed}/split_audit.json';audit=json.loads(audit_path.read_text())
        manifest_sources[audit_path.as_posix()]=sha(audit_path)
        data={};fold_records=[]
        for fold in range(3):
            path=source/f'seed{seed}/predictions_fold{fold}.npz'
            with np.load(path,allow_pickle=False) as f:cache=dict(f)
            prov=json.loads(str(cache['provenance']));ids=cache['sample_id'];p=cache['probs']
            assert ids.tolist()==prov['held_ids']
            assert not set(ids)&set(prov['fit_ids'])
            assert prov==audit['folds'][fold]
            head_path=source/f'seed{seed}/heads_fold{fold}.pt'
            assert sha(head_path)==prov['heads_sha256']
            assert p.shape==(len(ids),15,2) and np.isfinite(p).all() and np.allclose(p.sum(-1),1,atol=2e-6)
            manifest_sources[path.as_posix()]=sha(path)
            fold_records.append(dict(fold=fold,fit_ids=prov['fit_ids'],held_ids=prov['held_ids'],heads_sha256=prov['heads_sha256'],cache_sha256=sha(path)))
            for i,sample in enumerate(ids):
                assert sample not in data
                j=lookup[sample]
                assert cache['y'][i]==val['y'][j]
                np.testing.assert_array_equal(cache['embedding'][i],val['embedding'][j])
                data[sample]=(fold,{k:cache[k][i] for k in ('probs','logits','distances','embedding','y')})
        assert set(data)==set(lookup)
        assert not set(audit['gate_train_ids'])&set(audit['gate_select_ids'])
        assert set(audit['gate_train_ids'])|set(audit['gate_select_ids'])==set(lookup)
        for split,ids in [('meta_train',audit['gate_train_ids']),('meta_select',audit['gate_select_ids'])]:
            arrays={k:np.stack([data[s][1][k] for s in ids]) for k in ('probs','logits','distances','embedding','y')}
            arrays.update(sample_id=np.asarray(ids),group=np.array([val['group'][lookup[s]] for s in ids]),
                          fold=np.array([data[s][0] for s in ids]),seed=np.full(len(ids),seed),
                          split=np.full(len(ids),split),classes=np.array([0,1]),
                          head_id=np.array([f'head-{j:02d}' for j in range(15)]))
            arrays['backbone_sha256']=np.array('bce28dc72c3737ae19ffa057ba056e03d3f531180e7ad55e0b3c93bf4ea4bcaa')
            dest=out/f'seed{seed}/{split}.npz';buff=io.BytesIO();np.savez_compressed(buff,**arrays);atomic(dest,buff.getvalue())
            with np.load(dest,allow_pickle=False) as loaded:
                for k in arrays:np.testing.assert_array_equal(loaded[k],arrays[k])
            entries.append(dict(seed=seed,split=split,path=dest.relative_to(out).as_posix(),samples=len(ids),
                                probs_shape=list(arrays['probs'].shape),embedding_shape=list(arrays['embedding'].shape),
                                class_counts=np.bincount(arrays['y']).tolist(),group_counts=np.bincount(arrays['group'],minlength=4).tolist(),sha256=sha(dest)))
            for i,s in enumerate(ids):audit_rows.append(dict(seed=seed,split=split,sample_id=s,fold=int(arrays['fold'][i]),y=int(arrays['y'][i]),group=int(arrays['group'][i])))
        dump(out/f'seed{seed}/fold_provenance.json',dict(seed=seed,folds=fold_records,meta_train_ids=audit['gate_train_ids'],meta_select_ids=audit['gate_select_ids']))
    s=io.StringIO(newline='');writer=csv.DictWriter(s,fieldnames=['seed','split','sample_id','fold','y','group']);writer.writeheader();writer.writerows(audit_rows);atomic(out/'development_index.csv',s.getvalue())
    readme='''# DPE development OOF bundle — no test data

Six NPZ files cover three head seeds (0,1,2), each split into:
- `seed*/meta_train.npz`: 959 development examples for fitting weighting.
- `seed*/meta_select.npz`: 240 development examples for configuration selection.

All 1,199 samples belong to the official Waterbirds validation/development pool.
No test samples, test labels, test predictions or test-derived calibration are included.
Each seed uses a fixed class-stratified 80/20 split, inherited from the prior study.
The same image may have a different meta role in another seed; use seed-specific
train/select pairs. Do not pool seeds and let the same image cross train/select.

## NPZ schema

```text
probs       [samples,15,2]     raw temperature-1 member probabilities
logits      [samples,15,2]     real member logits
distances   [samples,15,2]     normalized prototype distances
embedding   [samples,2048]    raw frozen backbone features f(x)
y           [samples]         true class, 0=landbird, 1=waterbird
sample_id   [samples]         stable relative image filename
group       [samples]         EVALUATION ONLY, 2*y+background (0..3)
fold        [samples]         OOF fold predicting that image (0,1,2)
seed        [samples]
split       [samples]
classes     [2]               [0,1]
head_id     [15]              head-00..head-14, slots within each fold ensemble
backbone_sha256 scalar        verified ERM* backbone identity
```

All arrays in one file have identical sample order. Per-fold head ensembles are
different models: head-00 in fold 0 is not the same parameter tensor as head-00
in fold 1. Each OOF sample was excluded from fitting its predicting ensemble;
the full fit/held ID lists and actual head checkpoint hashes are included in
`seed*/fold_provenance.json`. No in-sample final-head predictions were substituted.
The split is made after cross-fitting; it is not nested outer validation.

## Use boundaries

Fit reliability tables, bins, scalers, calibration, feature clustering/models and
weighting on `meta_train` only. Select a prespecified shortlist on `meta_select`.
Do not pass `y`, correctness or `group` into an inference-time gate. Groups are
provided to compute post-hoc WGA and inspect feature-space group structure.
Using groups to train/choose a model would be a separate group-aware experiment;
do not label it no-group. Even feature preprocessing should fit train statistics
only when reporting held-out selection performance.

These are the development pools already used in previous weighting searches,
so meta_select is not an untouched confirmation set. Repeated tuning can overfit
it. The three seeds share ONE verified **ERM*** backbone trained for 400 epochs
with augmentation; these files are NOT standard-ERM/no-augmentation results.
Creating that track requires new backbone features and new OOF heads.

## Example

```python
import numpy as np
train = np.load('seed0/meta_train.npz', allow_pickle=False)
select = np.load('seed0/meta_select.npz', allow_pickle=False)
assert not set(train['sample_id']) & set(select['sample_id'])
p, y, z = train['probs'], train['y'], train['embedding']
# Fit weighting using p/y (optionally z); do not use train['group'] for no-group fitting.
uniform_pred = select['probs'].mean(1).argmax(1)
correct = uniform_pred == select['y']
group = select['group']
wga = min(correct[group == g].mean() for g in np.unique(group))
```

Group mapping: 0=landbird/land, 1=landbird/water, 2=waterbird/land,
3=waterbird/water. `development_index.csv` provides all membership and labels.
`manifest.json` records source/export hashes and class/group counts.

Re-export: `.venv/Scripts/python.exe -m scripts.export_dpe_development_bundle`.
This export performs no training and reads no test data.
'''
    atomic(out/'README.md',readme)
    dump(out/'manifest.json',dict(status='verified',data_role='development_only',backbone_track='ERM_star_400_epochs_augmented_verified',
         backbone_provenance='results/dpe/backbone_provenance_20261114/README.md',
         exporter_sha256=sha(__file__),sources=manifest_sources,files=entries,contains_test=False,
         checks=['raw probability normalization','per-fold held/fit disjointness','head checkpoint hashes','exact feature alignment',
                 'meta_train/meta_select disjointness per seed','unique full validation coverage','export NPZ exact roundtrip']))
    archive=source/'development_oof_bundle.zip';tmp=archive.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,arcname='development_bundle/'+p.relative_to(out).as_posix())
    tmp.replace(archive)
    print(json.dumps(dict(path=archive.as_posix(),bytes=archive.stat().st_size,files=entries),indent=2))


if __name__=='__main__':main()
