"""Independent arithmetic/ID/checkpoint audit of completed weighting artifacts."""
import csv
import json
from pathlib import Path
import numpy as np
from experiments.weighting.cache import digest, write_json


def main():
    root=Path('outputs/weighting')
    assert json.loads((root/'progress.json').read_text())['stage']=='complete'
    assert (root/'selection_lock.json').exists() and (root/'ablation_lock.json').exists()
    pre=json.loads((root/'preflight_audit.json').read_text())
    assert digest('checkpoints/kaggle_exploration/seed0/ensemble.pt')==pre['original_heads_sha256']
    assert digest('checkpoints/kaggle_exploration/seed0/ckpt_last.pt')==pre['backbone_sha256']
    with np.load(root/'features/val.npz',allow_pickle=False) as f:val_ids=set(f['sample_id'])
    with np.load(root/'features/test.npz',allow_pickle=False) as f:test_ids=set(f['sample_id'])
    assert len(val_ids)==1199 and len(test_ids)==5794 and not val_ids&test_ids
    folds=0
    for path in root.glob('seed*/split_audit.json'):
        audit=json.loads(path.read_text());held=[]
        for fold in audit['folds']:
            fit_ids=set(fold['fit_ids']);held_ids=set(fold['held_ids'])
            assert not fit_ids&held_ids and not fit_ids&test_ids
            held+=fold['held_ids'];folds+=1
        assert len(held)==1199 and set(held)==val_ids
        assert not set(audit['gate_train_ids'])&set(audit['gate_select_ids'])
        assert audit['groups_used_for_training'] is False
    audited=0
    for path in (root/'test_results').glob('*.json'):
        row=json.loads(path.read_text())
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as f:
            p=f['probs'];y=f['y'];g=f['group'];pred=f['pred'];ids=f['sample_id']
            assert len(ids)==len(set(ids))==5794 and set(ids)==test_ids
            assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1)
            correct=pred==y
            assert abs(correct.mean()-row['accuracy'])<1e-12
            wga=min(correct[g==k].mean() for k in np.unique(g))
            assert abs(wga-row['wga'])<1e-12
            ba=np.mean([correct[y==k].mean() for k in np.unique(y)])
            assert abs(ba-row['balanced_accuracy'])<1e-12
            nll=-np.log(p[np.arange(len(y)),y].clip(1e-12)).mean()
            assert abs(nll-row['nll'])<1e-6
            w=f['weights']
            if w.size:assert np.isfinite(w).all() and (w>=0).all() and np.allclose(w.sum(1),1)
        audited+=1
    with (root/'phase1_all.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    ranks={int(r['rank']) for r in rows};assert ranks==set(range(1,43))
    oracles=[r for r in rows if r['method']=='X0'];assert all(r['status']=='diagnostic_only' for r in oracles)
    report=dict(status='passed',checkpoint_hashes_unchanged=True,validation_count=1199,test_count=5794,
                crossfit_fold_audits=folds,unique_final_predictors_audited=audited,all_42_ranks_present=True,
                native_adapter_probability_difference=pre['native_vs_adapter_max_probability_difference'],
                scope='ID separation, cached-output arithmetic, normalized weights, original checkpoint preservation; not independent CNN training provenance')
    write_json(root/'independent_audit.json',report)
    print(json.dumps(report))


if __name__=='__main__':main()
