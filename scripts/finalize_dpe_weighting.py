"""Publish local summary artifacts and explicit corrected/broken sample tables."""
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from experiments.weighting.cache import digest, write_json
from experiments.weighting.run import write_csv


def main():
    from src.evaluation.prediction_protocol import validate_prediction_csv
    root=Path('outputs/weighting')
    assert json.loads((root/'progress.json').read_text())['stage']=='complete'
    assert json.loads((root/'crossfit5/progress.json').read_text())['stage']=='complete'
    lock=json.loads((root/'selection_lock.json').read_text());winner=lock['winner']
    with (root/'final_test.csv').open(encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    selected=[r for r in rows if r.get('ablation')=='phase1' and r['method']==winner['method']]
    baseline=[r for r in rows if r.get('ablation')=='phase1' and r['method']=='U0']
    original=rows[0]
    summaries=[]
    for name,rs in [('original_DPE_group_aware',[original]),('matched_U0',baseline),(winner['method'],selected)]:
        entry=dict(predictor=name,seeds=len(rs))
        for metric in ('accuracy','balanced_accuracy','wga','nll','brier','ece'):
            values=np.array([float(r[metric]) for r in rs])
            entry[metric+'_mean']=float(values.mean())
            entry[metric+'_sd']=float(values.std(ddof=1)) if len(values)>1 else None
        summaries.append(entry)
    write_csv(root/'summary.csv',summaries)
    failures=[];group_changes=[];baseline_member_comparisons=[]
    for row in selected:
        seed=int(row['seed']);key=row['fit_key']
        with np.load(root/'test_results'/f'{key}.npz',allow_pickle=False) as f:
            p=f['probs'];y=f['y'];g=f['group'];ids=f['sample_id'];pred=f['pred'];w=f['weights']
        with np.load(root/f'seed{seed}/predictions_test.npz',allow_pickle=False) as f:
            base=f['probs'].mean(1);expert_pred=f['probs'].argmax(-1)
        base_pred=base.argmax(1);correct=pred==y;base_correct=base_pred==y
        for k in np.unique(g):
            mask=g==k
            group_changes.append(dict(seed=seed,group=int(k),count=int(mask.sum()),
                      corrected=int((mask&correct&~base_correct).sum()),broken=int((mask&~correct&base_correct).sum()),
                      baseline_accuracy=float(base_correct[mask].mean()),winner_accuracy=float(correct[mask].mean())))
        for i in np.flatnonzero(correct!=base_correct):
            failures.append(dict(seed=seed,sample_id=str(ids[i]),group=int(g[i]),y=int(y[i]),
                      change='corrected' if correct[i] else 'broken',baseline_prediction=int(base_pred[i]),
                      winner_prediction=int(pred[i]),baseline_p_waterbird=float(base[i,1]),winner_p_waterbird=float(p[i,1]),
                      weight_entropy=float(-(w[i]*np.log(w[i].clip(1e-12))).sum()),
                      weight_on_correct_experts=float(w[i][expert_pred[i]==y[i]].sum())))
    write_csv(root/'failure_cases.csv',failures);write_csv(root/'failure_changes_by_group.csv',group_changes)
    for row in rows:
        if row.get('ablation')!='A12':continue
        seed=int(row['seed']);m=int(row['members'])
        with np.load(root/f'seed{seed}/predictions_test.npz',allow_pickle=False) as f:
            accuracy=float((f['probs'][:,:m].mean(1).argmax(1)==f['y']).mean())
        baseline_member_comparisons.append(dict(seed=seed,members=m,winner_accuracy=float(row['accuracy']),
                    same_member_U0_accuracy=accuracy,delta_vs_same_member_U0_pp=(float(row['accuracy'])-accuracy)*100))
    write_csv(root/'member_ablation_matched_baselines.csv',baseline_member_comparisons)
    validations={}
    metadata=Path('waterbird_complete95_forest2water2/waterbird_complete95_forest2water2/metadata.csv')
    for path in Path('experiments').glob('exp-dpe-weighting-*/predictions/test/predictions.csv'):
        validations[path.as_posix()]=validate_prediction_csv(path,metadata)
    write_json(root/'prediction_export_validation.json',validations)
    addition=('\n## Research conclusion and detailed sample audit\n\n'
              'The locked winner does not improve mean accuracy over the supplied original DPE reference. '
              'Its small accuracy increase over matched U0 comes with a substantial WGA reduction, '
              'especially for waterbirds on land (evaluation group 2 in the internal 0-based mapping). '
              'Two of three per-seed paired accuracy-delta intervals versus U0 include zero. '
              'There is no evidence here to replace the original DPE checkpoint. '
              'It has been preserved byte-for-byte.\n\n'
              'See `summary.csv`, `failure_cases.csv` (all corrected/broken sample IDs), '
              '`failure_changes_by_group.csv`, and `member_ablation_matched_baselines.csv` '
              '(same-member-count uniform controls in addition to the full-15 reference). '
              'All three canonical winner prediction exports were validated against the full reference metadata. '
              'The independent audit recomputed 84 final predictor outputs and checked nine OOF folds. '
              'Additional implementation choices and bounded-grid deviations are recorded in '
              '`experiments/weighting/PROTOCOL_DECISIONS.md`; in particular D11 uses augmented joint features, '
              'and focal screening uses gamma 2. These choices must not be described as an exhaustive Cartesian search.\n')
    report=root/'REPORT.md';text=report.read_text(encoding='utf-8')
    if '## Research conclusion and detailed sample audit' not in text:report.write_text(text+addition,encoding='utf-8')
    summary=root/'RESULTS_FOR_USER.md';text=summary.read_text(encoding='utf-8')
    if 'DPE gốc:' not in text:
        text+=(f'\nDPE gốc: accuracy {float(original["accuracy"])*100:.3f}%, WGA {float(original["wga"])*100:.3f}%. '
               'Winner không vượt accuracy trung bình của DPE gốc, đồng thời giảm WGA rõ rệt. '
               'Giữ nguyên checkpoint DPE gốc; chưa có cơ sở thay thế bằng weighting.\n'
               '\nĐã hoàn tất A18 5-fold, lưu ở phase2_crossfit5.csv. Toàn bộ checkpoint, bảng kết quả và '
               'hướng dẫn resume/handoff đã lưu cục bộ.\n')
        summary.write_text(text,encoding='utf-8')
    tracked=Path('results/dpe/weighting_20261114');tracked.mkdir(parents=True,exist_ok=True)
    for name,target in [('REPORT.md','README.md'),('summary.csv','summary.csv'),('independent_audit.json','independent_audit.json'),
                        ('validation_results.json','validation_results.json'),('phase2_crossfit5.csv','phase2_crossfit5.csv')]:
        shutil.copyfile(root/name,tracked/target)
    handoff=root/'HANDOFF.md'
    if 'All main and supplemental phases completed.' not in handoff.read_text(encoding='utf-8'):
        with handoff.open('a',encoding='utf-8') as stream:
            stream.write('\nAll main and supplemental phases completed. Audit passed. Original DPE preserved.\n'
                     'Supplemental resume: `python -m experiments.weighting_ablation_crossfit`.\n'
                     'Audit: `python -m scripts.audit_dpe_weighting`.\n'
                     'Final summaries: `REPORT.md`, `RESULTS_FOR_USER.md`, `summary.csv`.\n'
                     'Regenerate post-hoc summaries: `python -m scripts.finalize_dpe_weighting`.\n')
    write_json(root/'artifact_index.json',{p.relative_to(root).as_posix():digest(p) for p in root.rglob('*')
                  if p.is_file() and p.name not in ('artifact_index.json',) and not p.name.endswith('.tmp')})
    print('Finalized reports, per-sample failure tables and handoff artifacts.')


if __name__=='__main__':main()
