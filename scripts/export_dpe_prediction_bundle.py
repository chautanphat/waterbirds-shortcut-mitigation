"""Export existing frozen expert/gate outputs for per-image error-overlap audits."""
import csv
import json
from pathlib import Path
import shutil
import zipfile
import joblib
import numpy as np
from experiments.weighting.cache import digest, write_json


def main():
    root=Path('outputs/weighting')
    out=root/'prediction_bundle'
    out.mkdir(exist_ok=True)
    lock=json.loads((root/'selection_lock.json').read_text())
    with (root/'final_test.csv').open(encoding='utf-8') as stream:
        results=list(csv.DictReader(stream))
    selected=[r for r in results if r.get('ablation')=='phase1' and r['method']==lock['winner']['method']]
    assert sorted(int(r['seed']) for r in selected)==[0,1,2]
    summary=[]
    table=out/'final_test.csv'
    columns=['seed','sample_id','sample_key','y','group','uniform_prediction','gate_prediction',
             'uniform_correct','gate_correct','uniform_p0','uniform_p1','gate_p0','gate_p1',
             'wrong_expert_count','weight_on_wrong_experts','weight_on_correct_experts',
             'highest_weight_expert','highest_weight_expert_wrong']
    for j in range(15):
        columns += [f'head_{j:02d}_{name}' for name in ('p0','p1','calibrated_p0','calibrated_p1','weight','prediction','correct')]
    temporary=table.with_suffix('.tmp')
    with temporary.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(columns)
        for row in sorted(selected,key=lambda r:int(r['seed'])):
            seed=int(row['seed']);key=row['fit_key']
            source=root/f'seed{seed}/predictions_test.npz'
            gate_source=root/'test_results'/f'{key}.npz'
            with np.load(source,allow_pickle=False) as f:
                p=f['probs'];ids=f['sample_id'];y=f['y'];logits=f['logits']
            with np.load(gate_source,allow_pickle=False) as f:
                np.testing.assert_array_equal(ids,f['sample_id']);np.testing.assert_array_equal(y,f['y'])
                w=f['weights'];gate=f['probs'];group=f['group'];gate_pred=f['pred']
            assert p.shape==(5794,15,2) and w.shape==(5794,15)
            assert len(set(ids))==5794
            assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(-1),1,atol=2e-6)
            assert np.isfinite(w).all() and (w>=0).all() and np.allclose(w.sum(1),1,atol=1e-12)
            model=joblib.load(root/'fits'/key/'model.joblib')
            calibrated=model.calibrated({'probs':p})['probs']
            reconstructed=(calibrated*w[:,:,None]).sum(1)
            difference=float(np.max(np.abs(reconstructed-gate)))
            np.testing.assert_allclose(reconstructed,gate,atol=1e-12,rtol=0)
            expert_pred=p.argmax(-1);expert_correct=expert_pred==y[:,None]
            np.testing.assert_array_equal(expert_pred,calibrated.argmax(-1))
            uniform=p.mean(1);uniform_pred=uniform.argmax(1)
            wrong_weight=(w*~expert_correct).sum(1);right_weight=(w*expert_correct).sum(1)
            top=w.argmax(1);top_wrong=~expert_correct[np.arange(len(y)),top]
            cache=out/f'prediction_cache_seed{seed}.npz'
            provenance=dict(seed=seed,method=row['method'],fit_key=key,
                            original_expert_cache=source.as_posix(),original_expert_cache_sha256=digest(source),
                            gate_cache=gate_source.as_posix(),gate_cache_sha256=digest(gate_source),
                            final_heads_sha256=digest(root/f'seed{seed}/heads_final.pt'),
                            selection_lock_sha256=digest(root/'selection_lock.json'),
                            gate_combines='calibrated_probs, not raw probs',shape_order='samples,members,classes')
            with cache.with_suffix('.tmp').open('wb') as target:
                np.savez_compressed(target,probs=p,calibrated_probs=calibrated,logits=logits,
                     weights=w,y=y,sample_id=ids,sample_key=np.array([f'{seed}:{s}' for s in ids]),
                     seed=np.full(len(y),seed),group=group,split=np.full(len(y),'test'),classes=np.array([0,1]),
                     head_id=np.array([f'head-{j:02d}' for j in range(15)]),
                     temperature=model.state.get('temperature',np.ones(15)),
                     expert_predictions=expert_pred,expert_correct=expert_correct,gate_probs=gate,
                     gate_predictions=gate_pred,uniform_probs=uniform,
                     weight_on_wrong_experts=wrong_weight,weight_on_correct_experts=right_weight,
                     provenance=np.array(json.dumps(provenance,sort_keys=True)))
            cache.with_suffix('.tmp').replace(cache)
            for i in range(len(y)):
                values=[seed,ids[i],f'{seed}:{ids[i]}',int(y[i]),int(group[i]),int(uniform_pred[i]),int(gate_pred[i]),
                        int(uniform_pred[i]==y[i]),int(gate_pred[i]==y[i]),*uniform[i].tolist(),*gate[i].tolist(),
                        int((~expert_correct[i]).sum()),float(wrong_weight[i]),float(right_weight[i]),int(top[i]),int(top_wrong[i])]
                for j in range(15):
                    values += [float(p[i,j,0]),float(p[i,j,1]),float(calibrated[i,j,0]),float(calibrated[i,j,1]),
                               float(w[i,j]),int(expert_pred[i,j]),int(expert_correct[i,j])]
                writer.writerow(values)
            errors=(~expert_correct).astype(np.int64)
            both_wrong=errors.T@errors
            union=errors.sum(0)[:,None]+errors.sum(0)[None,:]-both_wrong
            jaccard=np.divide(both_wrong,union,out=np.zeros_like(both_wrong,dtype=float),where=union>0)
            for name,values in [('both_wrong_count',both_wrong),('error_jaccard',jaccard)]:
                with (out/f'{name}_seed{seed}.csv').open('w',newline='',encoding='utf-8') as f:
                    cw=csv.writer(f);cw.writerow(['head_id']+[f'head-{j:02d}' for j in range(15)])
                    for j,v in enumerate(values):cw.writerow([f'head-{j:02d}',*v.tolist()])
            # Reload exported bytes to check the consumer-facing cache itself.
            with np.load(cache,allow_pickle=False) as f:
                np.testing.assert_array_equal(f['probs'],p)
                np.testing.assert_array_equal(f['sample_id'],ids)
                np.testing.assert_array_equal(f['y'],y)
                np.testing.assert_allclose((f['calibrated_probs']*f['weights'][:,:,None]).sum(1),gate,atol=1e-12,rtol=0)
            summary.append(dict(seed=seed,samples=len(y),shape=list(p.shape),cache=cache.name,sha256=digest(cache),
                      gate_reconstruction_max_abs_difference=difference,
                      mean_weight_on_wrong_experts=float(wrong_weight.mean()),
                      highest_weight_expert_wrong_fraction=float(top_wrong.mean()),
                      uniform_accuracy=float((uniform_pred==y).mean()),gate_accuracy=float((gate_pred==y).mean()),
                      sources=provenance))
    temporary.replace(table)
    shutil.copyfile(root/'final_test.csv',out/'final_test_metrics.csv')
    # The unchanged historical annotated reference has a distinct ensemble.
    with np.load(root/'original_test_predictions.npz',allow_pickle=False) as f:
        np.savez_compressed(out/'prediction_cache_original_DPE.npz',probs=f['probs'],y=f['y'],sample_id=f['sample_id'],
                            classes=np.array([0,1]),head_id=np.array([f'head-{j:02d}' for j in range(15)]),
                            weights=np.full((5794,15),1/15),provenance=f['provenance'])
    with table.open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==17382 and len({r['sample_key'] for r in rows})==17382
    readme='''# Per-image classifier and gate prediction bundle

`final_test.csv`: 17,382 rows = 5,794 test images x 3 seeds. One row is one
image/seed, with every head's raw/calibrated class probabilities, gate weight,
prediction and correctness. `sample_key` is unique across seeds; `sample_id`
is the original image path. Original aggregate results are `final_test_metrics.csv`.

Each `prediction_cache_seed{0,1,2}.npz` has `probs [5794,15,2]`, `y [5794]`,
`sample_id [5794]`, and `weights [5794,15]`, plus calibrated probabilities,
member predictions/correctness, group, class/head order, temperatures and frozen
gate/uniform outputs. Arrays are sample-major and share exactly the same row order.
Classes: 0=landbird, 1=waterbird. Group: 2*y+background, zero-based (0..3).
Heads are ordered head-00..head-14. Three seeds use distinct refitted ensembles.

IMPORTANT: D19 mixes CALIBRATED member probabilities:
    gate_probs = (calibrated_probs * weights[...,None]).sum(axis=1)
Do not mix raw `probs` with D19 weights when reconstructing gate output.
Positive temperature calibration leaves each member's top-1 decision unchanged.
Group labels/correctness are included only for post-hoc analysis.

`prediction_cache_original_DPE.npz` is the separate, unchanged group-aware
original reference; it uses uniform weights and has no learned gate.
`both_wrong_count_seed*.csv` and `error_jaccard_seed*.csv` provide pairwise
error-overlap counts/Jaccard for all 15 experts.

Example:
```python
import numpy as np
d = np.load('prediction_cache_seed0.npz', allow_pickle=False)
p, y, ids, w = d['probs'], d['y'], d['sample_id'], d['weights']
wrong = p.argmax(-1) != y[:, None]
weight_on_wrong = (w * wrong).sum(axis=1)
both_wrong = wrong.astype('int64').T @ wrong.astype('int64')
np.testing.assert_allclose(
    (d['calibrated_probs'] * w[..., None]).sum(1), d['gate_probs'], atol=1e-12)
```

Only previously saved outputs and frozen temperatures were used. No training,
selection or test-driven tuning was performed during this export. Source hashes
and exact reconstruction checks are in `manifest.json`.

Reproduce from repository root:
`.venv/Scripts/python.exe -m scripts.export_dpe_prediction_bundle`
'''
    (out/'README.md').write_text(readme,encoding='utf-8')
    write_json(out/'manifest.json',dict(status='verified',csv_rows=len(rows),csv_columns=len(columns),
               csv_sha256=digest(table),seeds=summary,exporter_sha256=digest(__file__)))
    archive=root/'prediction_bundle.zip'
    with zipfile.ZipFile(archive.with_suffix('.tmp'),mode='w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.iterdir()):
            if p.is_file():z.write(p,arcname='prediction_bundle/'+p.name)
    archive.with_suffix('.tmp').replace(archive)
    print(json.dumps(dict(bundle=archive.as_posix(),bytes=archive.stat().st_size,csv_rows=len(rows),
                          cache_shape=[5794,15,2],seeds=[0,1,2],verification='passed')))


if __name__=='__main__':main()
