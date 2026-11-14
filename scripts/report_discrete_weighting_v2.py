"""Post-lock aggregation/audit, committed using the same durable task journal."""
import json
from pathlib import Path
import shutil
import numpy as np
from experiments.discrete_weighting.store import Store,atomic,dump,key,sha
from experiments.discrete_weighting.run import csvwrite
from experiments.weighting.metrics import metrics


def main():
    root=Path(Path('outputs/discrete_weighting_v2/LATEST_RUN.txt').read_text().strip())
    manifest=json.loads((root/'manifest.json').read_text());command='python -m scripts.report_discrete_weighting_v2'
    store=Store(root,manifest,command)
    assert 'report' in store.valid
    lock=json.loads((root/'SELECTION_LOCK.json').read_text());winner=lock['winner']['config']
    def work():
        tests={key(r['result']['config']):[] for r in store.valid.values() if r['phase']=='test'}
        for r in store.valid.values():
            if r['phase']=='test':tests[key(r['result']['config'])].append(r['result'])
        summary=[]
        for spec in lock['ablations']:
            rs=tests.get(key(spec['config']),[])
            if len(rs)!=3:continue
            out=dict(ablation=spec['ablation'],variant=spec['variant'],config=spec['config'],seeds=3)
            for metric in ('accuracy','balanced_accuracy','wga','delta_raw_pp','delta_representation_uniform_pp'):
                values=np.array([r[metric] for r in rs]);out[metric]=float(values.mean());out[metric+'_sd']=float(values.std(ddof=1))
            summary.append(out)
        path=root/'results/phase2_summary.csv';csvwrite(path,summary)
        rows=tests[key(winner)];matched=[];group_weights=[];deltas=[]
        for r in rows:
            cfg=r['config'];seed=r['seed'];pfile=root/f'test_results/seed{seed}/{key(cfg)[:16]}.npz'
            with np.load(pfile,allow_pickle=False) as f:
                p=f['probs'];cal=f['calibrated_probs'];w=f['weights'];y=f['y'];g=f['group'];mix=f['gate_probs']
                assert np.isfinite(p).all() and (w>=0).all() and np.allclose(w.sum(1),1)
                np.testing.assert_allclose((cal*w[:,:,None]).sum(1),mix,atol=1e-12,rtol=0)
                pred=mix.argmax(1);base=p.mean(1)
                for group in np.unique(g):
                    q=g==group;group_weights.append(dict(seed=seed,group=int(group),mean_weights=w[q].mean(0).tolist(),wrong_expert_mass=float((w[q]*(p[q].argmax(-1)!=y[q,None])).sum(1).mean())))
                for label,probs in [('raw_uniform',base),('representation_uniform',cal.mean(1))]:matched.append(dict(seed=seed,control=label,**metrics(probs,y,g)))
                deltas.append((pred==y).astype(float)-(base.argmax(1)==y))
        controls=root/'results/matched_controls.csv';csvwrite(controls,matched)
        groups=root/'results/winner_weights_by_group.csv';csvwrite(groups,group_weights)
        delta=np.mean(deltas,axis=0);rng=np.random.default_rng(1)
        ci=np.quantile([delta[rng.integers(0,len(delta),len(delta))].mean()*100 for _ in range(2000)],[.025,.975]).tolist()
        lines=['# Discrete reliability v2 complete research report','',
               'All test results are exploratory on a repeatedly inspected test set. Three refitted-head seeds share one backbone. No test labels or group annotations enter fitting, calibration or recipe selection. The 29 configurations were fixed before results; the single family comparison uses fixed OOF selection IDs. Calibration, bins, thresholds and tables use meta_train only. No CNN or head retraining was performed.', '',
               'Winner selected by mean held-out accuracy, then BA, then NLL: `'+json.dumps(winner)+'`. Selection accuracy: '+str(lock['winner']['accuracy']*100)+'%.', '',
               '## Winner and matched controls','', '| Method | Accuracy % | BA % | WGA % |','|---|---:|---:|---:|']
        for label,rs in [('winner',rows),('raw_uniform',[r for r in matched if r['control']=='raw_uniform']),('representation_uniform',[r for r in matched if r['control']=='representation_uniform'])]:
            values=[]
            for k in ('accuracy','balanced_accuracy','wga'):
                v=np.array([r[k] for r in rs]);values.append(f'{v.mean()*100:.3f} ± {v.std(ddof=1)*100:.3f}')
            lines.append('| '+label+' | '+' | '.join(values)+' |')
        gain=float(np.mean([r['delta_raw_pp'] for r in rows]));raw=[r for r in matched if r['control']=='raw_uniform'];wga=100*(np.mean([r['wga'] for r in rows])-np.mean([r['wga'] for r in raw]))
        lines+=['',f'Accuracy delta vs raw uniform: {gain:+.4f} pp; WGA delta {wga:+.4f} pp. Conditional paired-example bootstrap 95% interval for the mean across these fixed models: {ci} pp. This does not measure independent-backbone uncertainty.', '',
                'A positive micro-accuracy difference alone is not a robust win; inspect WGA, per-class confusion/recall, flips and seed variation in the full task CSVs. No winner was changed after test evaluation.', '',
                '## Phase 1 full configuration selection results','', '| Family | Config | Accuracy % | BA % | NLL |','|---|---|---:|---:|---:|']
        bycfg={}
        for rec in store.valid.values():
            r=rec['result']
            if rec['phase']=='phase1' and r.get('eligible'):bycfg.setdefault(key(r['config']),[]).append(r)
        for rs in bycfg.values():
            c=rs[0]['config'];lines.append(f'| {c["family"]} | {json.dumps(c)} | {np.mean([r["accuracy"] for r in rs])*100:.3f} | {np.mean([r["balanced_accuracy"] for r in rs])*100:.3f} | {np.mean([r["nll"] for r in rs]):.5f} |')
        lines+=['','## Phase 2 ablations','', '| ID | Variant | Accuracy % | Delta raw pp | WGA % |','|---|---|---:|---:|---:|']
        for r in summary:lines.append(f'| {r["ablation"]} | {r["variant"]} | {r["accuracy"]*100:.3f} ± {r["accuracy_sd"]*100:.3f} | {r["delta_raw_pp"]:+.3f} | {r["wga"]*100:.3f} ± {r["wga_sd"]*100:.3f} |')
        lines+=['','## Interpretation and limitations','',
                'Reliability conditions on predicted class, never unknown true class at inference. Pooled head observations are correlated; counts are descriptive, not independent samples. Confidence tertiles are estimated by predicted class on training only. Agreement is leave-one-out with fixed 1/3 and 2/3 boundaries. Sparse cells back off to confidence/class/global priors; head tables shrink toward pooled correctness. Ordinal bands .5/.65/.8/.95 are predeclared. True-class-balanced fitting is a separately flagged training-only contrast. Predicted-class-balanced fitting changes observation weights; it can coincide with empirical estimates within a conditioned cell.', '',
                'The threshold control is fitted on meta_train directly and assessed once on select. The search is bounded and predeclared, not exhaustive; select pools of 240 images create selection optimism. The expanded temperature grid .03–8 investigates the earlier .25 boundary without using test NLL for tuning. Empty cells use robust priors. A20 shuffled-bin control rotates expert lookup scores per image; random reliability is label-free and deterministic. D0/D1 controls and X0 diagnostics are not learned neural gates. C1 saved D19 was reused for three seeds; the saved linear control is available only for the original two screening seeds (seed 2 missing, not fabricated).', '',
                'Final full-validation heads differ from fold-trained OOF heads, so distribution mismatch remains. Historical backbone training IDs/recipe are not embedded in its bare checkpoint. There is no truly untouched external holdout. Prior source attempt and verified migration are preserved; its already-completed reliability work was reused after a comparison-model lookup fix. All fresh training here is pure table/calibrator fitting, persisted per atomic task.', '',
                'Per-task records in completed/ are authoritative. Results, weights, IDs, corrected/broken case CSVs and class confusion are retained. Resume regenerates tables from valid markers; corrupt/missing outputs and dependents are rebuilt. Tests include separate-process budgeted resumes, abandoned temporary files and output corruption. On abrupt termination at most one incomplete task is retried. `HANDOFF_STATUS.md`, `NEXT_AGENT_PROMPT.md` and `state.json` identify the next task. Running --status performs integrity verification.', '',
                'Main resume: `python -m experiments.discrete_weighting.run --config experiments/discrete_weighting/config.yaml --phase all --resume`. Detailed report regeneration: `python -m scripts.report_discrete_weighting_v2`.']
        report=root/'detailed_report.md';atomic(report,'\n'.join(lines)+'\n')
        audit=root/'final_audit.json';dump(audit,dict(status='passed',completed_task_count=len(store.valid),all_completion_hashes_verified=True,winner_reconstruction_verified=True,paired_mean_ci_pp=ci,accuracy_delta_pp=gain,wga_delta_pp=wga,report_script_sha256=sha(__file__)))
        return {},[path,controls,groups,report,audit]
    store.task('analysis_report','report',[t for t,r in store.valid.items() if r['phase']=='test'],work)
    store.command='python -m experiments.discrete_weighting.run --config experiments/discrete_weighting/config.yaml --phase all --resume'
    store.refresh('COMPLETE','none');atomic(root/'REPORT.md',(root/'detailed_report.md').read_text(encoding='utf-8'))
    for source,h in manifest['source'].items():
        assert sha(source)==h
        dest=root/'source_snapshot'/source;atomic(dest,Path(source).read_bytes())
    print((root/'final_audit.json').read_text());print(root)


if __name__=='__main__':main()
