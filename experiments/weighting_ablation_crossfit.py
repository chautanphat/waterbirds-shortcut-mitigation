"""Prespecified, checkpointed A18 diagnostic using the locked winner and 5 folds.

Run after the main workflow. Never selects a new method using test performance.
"""
import json
from pathlib import Path
import time
import joblib
import numpy as np
import yaml
from experiments.weighting.cache import digest, write_json
from experiments.weighting.data import seed_all, prepare_meta
from experiments.weighting.methods import Predictor
from experiments.weighting.metrics import metrics, paired_ci
from experiments.weighting.run import write_csv


def main():
    root=Path('outputs/weighting')
    plan=json.loads((root/'supplemental_plan.json').read_text())
    if plan['source_sha256']!=digest(__file__):raise ValueError('Supplement source changed after plan lock')
    lock=json.loads((root/'selection_lock.json').read_text())
    if plan['selection_lock_sha256']!=digest(root/'selection_lock.json'):raise ValueError('Winner changed')
    c=yaml.safe_load(Path('experiments/weighting/config.yaml').read_text())
    c['folds']=5;c['output_dir']=(root/'crossfit5').as_posix()
    with np.load(root/'features/val.npz',allow_pickle=False) as f:val=dict(f)
    with np.load(root/'features/test.npz',allow_pickle=False) as f:test=dict(f)
    results=[]
    for seed in c['seeds']:
        target=Path(c['output_dir'])/f'seed{seed}'
        target.mkdir(parents=True,exist_ok=True)
        marker=target/'result.json'
        if marker.exists():results.append(json.loads(marker.read_text()));continue
        write_json(root/'crossfit5/progress.json',dict(stage='head_crossfit',seed=seed,
                   rerun='python -m experiments.weighting_ablation_crossfit'))
        train,select,_=prepare_meta(val,c,seed)
        model_path=target/'model.joblib'
        if model_path.exists():model=joblib.load(model_path)
        else:
            model=Predictor(lock['winner']['method'],lock['winner']['config']).fit(train,select,seed,
                        epochs=c['epochs'],patience=c['patience'],progress_path=target/'training_latest.pt')
            tmp=model_path.with_suffix('.tmp');joblib.dump(model,tmp);tmp.replace(model_path)
        # Head final refit is identical across OOF fold-count settings: reuse it.
        with np.load(root/f'seed{seed}/predictions_test.npz',allow_pickle=False) as f:
            data={k:f[k] for k in ('probs','logits','distances','embedding')}
        baseline=data['probs'].mean(1);base=metrics(baseline,test['y'],test['group'])
        pp=model.predict(data);met=metrics(pp,test['y'],test['group'],threshold=model.threshold)
        row=dict(ablation='A18',variant='crossfit_5',method=model.method,config=model.cfg,seed=seed,
                 status='completed',split='test',**met,delta_pp_vs_U0=(met['accuracy']-base['accuracy'])*100,
                 wga_delta_pp_vs_U0=(met['wga']-base['wga'])*100,
                 paired_bootstrap_ci_pp=paired_ci(pp,baseline,test['y'],c['bootstrap_repeats'],seed,model.threshold),
                 train_seconds=model.train_seconds,params=model.params,final_heads_reused=True)
        np.savez_compressed(target/'test_predictions.npz',probs=pp,y=test['y'],sample_id=test['sample_id'],group=test['group'])
        write_json(marker,row);results.append(row)
        write_csv(root/'phase2_crossfit5.csv',results)
    write_csv(root/'phase2_crossfit5.csv',results)
    write_json(root/'crossfit5/progress.json',dict(stage='complete',seeds=c['seeds']))
    mean=lambda key:float(np.mean([r[key] for r in results]))
    sd=lambda key:float(np.std([r[key] for r in results],ddof=1))
    addition=('\n## Additional locked A18 diagnostic: 5-fold cross-fitting\n\n'
              'The original budget-reserved N/A entry is superseded by `phase2_crossfit5.csv`. '
              'This diagnostic was declared before test metrics were read, uses the exact locked winner/configuration, '
              'refits the OOF heads/gate for all three seeds, and reuses identical final full-validation heads. '
              'It does not select a new winner. Its separate plan/source hashes and per-head/epoch checkpoints are retained.\n\n'
              f'Accuracy: {mean("accuracy")*100:.3f} ± {sd("accuracy")*100:.3f}%; '
              f'WGA: {mean("wga")*100:.3f} ± {sd("wga")*100:.3f}%; '
              f'accuracy delta vs matched U0: {mean("delta_pp_vs_U0"):+.3f} pp; '
              f'WGA delta: {mean("wga_delta_pp_vs_U0"):+.3f} pp.\n')
    report=root/'REPORT.md';text=report.read_text(encoding='utf-8')
    if '## Additional locked A18 diagnostic' not in text:report.write_text(text+addition,encoding='utf-8')
    print('A18 complete: '+str(root/'phase2_crossfit5.csv'))


if __name__=='__main__':main()
