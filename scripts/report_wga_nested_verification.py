"""Audit and report the completed continuation without changing either recipe."""
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from experiments.discrete_weighting.store import atomic,dump,sha
from experiments.discrete_weighting.run import csvwrite
from experiments.wga_verification.journal import Journal
from experiments.weighting.metrics import metrics

ROOT=Path('outputs/wga_verification_v3/ce66e4448807')


def main():
    command='python -m experiments.wga_verification.continue_run --resume'
    j=Journal(ROOT,json.loads((ROOT/'manifest.json').read_text()),command)
    a=json.loads((ROOT/'verification/source_comparison/GATE_A.json').read_text());b=json.loads((ROOT/'verification/source_comparison/GATE_B.json').read_text())
    assert a['status']=='PASS' and b['status']=='FAIL'
    def audit_report():
        with np.load('outputs/weighting/features/val.npz',allow_pickle=False) as f:val=dict(f)
        audited_heads=0;fold_rows=[];weights=[]
        for seed in (0,1,2):
            seen=[]
            for fold in range(3):
                folder=ROOT/f'nested/seed{seed}/fold{fold}';plan=json.loads((folder/'split.json').read_text());idx=[np.array(plan[k]) for k in ('head_indices','weight_indices','hold_indices')]
                sets=[set(val['sample_id'][v]) for v in idx]
                assert all(not sets[i]&sets[k] for i in range(3) for k in range(i+1,3));seen.extend(idx[2].tolist())
                for member in range(15):
                    h=torch.load(folder/f'head{member:02d}/head.pt',weights_only=True)
                    assert set(h['fit_ids'])==sets[0] and h['selected_epoch']==20 and h['member_id']==f'head-{member:02d}'
                    assert torch.isfinite(h['prototypes']).all() and torch.isfinite(h['scale']).all();audited_heads+=1
                with np.load(folder/'predictions.npz',allow_pickle=False) as p,np.load(folder/'weights.npz',allow_pickle=False) as w,np.load(folder/'holdout_predictions.npz',allow_pickle=False) as held:
                    hashes=np.array([sha(folder/f'head{member:02d}/head.pt') for member in range(15)])
                    assert np.array_equal(hashes,p['member_hashes']) and np.array_equal(hashes,w['member_hashes']) and np.array_equal(hashes,held['member_hashes'])
                    assert set(p['weight_ids'])==sets[1] and set(p['holdout_ids'])==sets[2] and set(w['fit_ids'])==sets[1]
                    ww=w['weights'];assert np.isfinite(ww).all() and (ww>=-1e-8).all() and ww.max()<=.25+1e-8 and abs(ww.sum()-1)<1e-7
                    np.testing.assert_allclose(np.einsum('nmc,m->nc',p['holdout_probs'],ww),held['candidate'],atol=1e-12,rtol=0)
                    assert np.isfinite(p['weight_probs']).all() and np.isfinite(p['holdout_probs']).all()
                    np.testing.assert_allclose(p['holdout_probs'].sum(-1),1,atol=2e-6)
                    weights.append(dict(seed=seed,fold=fold,**{f'head_{m:02d}':float(v) for m,v in enumerate(ww)}))
                solver=json.loads((folder/'solver.json').read_text());assert solver['success'] and solver['min_epigraph_slack']>=-1e-7
                fold_rows.append(dict(seed=seed,fold=fold,head_fit=len(idx[0]),weight_fit=len(idx[1]),outer_holdout=len(idx[2]),simplex_residual=solver['simplex_residual']))
            assert sorted(seen)==list(range(1199))
            with np.load(ROOT/f'nested/seed{seed}/pooled_predictions.npz',allow_pickle=False) as f:
                assert np.array_equal(f['sample_id'],val['sample_id']) and np.array_equal(f['y'],val['y'])
                r=next(r for r in b['seeds'] if r['seed']==seed)
                for label in ('uniform','candidate'):
                    m=metrics(f[label],f['y'],f['group']);assert abs(m['accuracy']-r[label]['accuracy'])<1e-12 and abs(m['wga']-r[label]['wga'])<1e-12
        assert audited_heads==135
        summary=[]
        for row in b['seeds']:
            summary.append(dict(seed=row['seed'],uniform_AA=100*row['uniform']['accuracy'],candidate_AA=100*row['candidate']['accuracy'],uniform_WGA=100*row['uniform']['wga'],candidate_WGA=100*row['candidate']['wga'],delta_AA_pp=row['delta_AA_pp'],delta_WGA_pp=row['delta_WGA_pp'],corrected=row['corrected'],broken=row['broken']))
        table=ROOT/'nested/per_seed_summary.csv';csvwrite(table,summary)
        folds=ROOT/'nested/split_and_solver_audit.csv';csvwrite(folds,fold_rows)
        weight_file=ROOT/'nested/weights_by_fold.csv';csvwrite(weight_file,weights)
        lines=['# LCMSW v3 — source reproduction passes, nested promotion fails','',
               '**Gate A: PASS. Gate B: FAIL. Production integration: PROHIBITED and not performed.**','',
               'The newly supplied original analysis/archive were recursively found under baselines/dpe4subpop. Original cv_sensitivity.py and robust_static.py executed with only filesystem path substitutions. Results match archived CSVs and the independent calculation with 0.0 pp discrepancy. No separate original SHA manifest or epigraph_verify.py was supplied; this is explicitly documented, all archive members were hashed on receipt, the bundle export manifest verified, and the original 18 independent epigraph/direct-max checks retained.', '',
               'Gate A reproduces +1.789882 pp WGA and +0.444815 pp AA, with 48 flips (32 corrected, 16 broken). These are non-nested development sensitivity results, not deployment/test improvements.', '',
               '## Truly held-out head-and-weight evaluation','',
               'For each of three seeds and three outer class-stratified folds, split the outer training pool 70/30 into A=head fit and B=weight fit; C is the held-out outer fold. Train 15 heads from scratch only on A, fit frozen PCA/KMeans/minimax weights only on B, and predict C with the exact same ordered heads. Every A/B/C ID set is disjoint. Groups are used only after outer predictions are committed. No historical test arrays were read and no hyperparameters were changed.', '',
               '| Seed | Uniform AA % | LCMSW AA % | ΔAA pp | Uniform WGA % | LCMSW WGA % | ΔWGA pp |','|---|---:|---:|---:|---:|---:|---:|']
        for r in summary:lines.append(f'| {r["seed"]} | {r["uniform_AA"]:.4f} | {r["candidate_AA"]:.4f} | {r["delta_AA_pp"]:+.4f} | {r["uniform_WGA"]:.4f} | {r["candidate_WGA"]:.4f} | {r["delta_WGA_pp"]:+.4f} |')
        means={k:float(np.mean([r[k] for r in summary])) for k in ('uniform_AA','candidate_AA','delta_AA_pp','uniform_WGA','candidate_WGA','delta_WGA_pp')}
        lines.append(f'| Mean | {means["uniform_AA"]:.4f} | {means["candidate_AA"]:.4f} | {means["delta_AA_pp"]:+.4f} | {means["uniform_WGA"]:.4f} | {means["candidate_WGA"]:.4f} | {means["delta_WGA_pp"]:+.4f} |')
        lines+=['','Sample SD across the three head/pipeline seeds:']
        for label in ('uniform','candidate'):
            lines.append('- '+label+': '+', '.join(f'{field} {np.std([r[label][field]*100 for r in b["seeds"]],ddof=1):.4f} pp' for field in ('accuracy','wga')))
        lines+=['','## Exact failure reasons','',
               '- Positive mean WGA: PASS (+0.2915 pp).',
               '- WGA improvement in at least 2/3 seeds: FAIL (only seed 0 improves; seeds 1 and 2 decline).',
               '- Mean ΔAA > −1.00 pp: PASS (−0.8340 pp).',
               '- Every seed ΔAA > −1.00 pp: FAIL (seed 1 is −1.0008340284 pp). This narrowly crosses the frozen boundary; no rounding tolerance or criterion relaxation was introduced. The WGA consistency requirement fails independently.',
               '- Provenance/leakage/identity audit: PASS; 135 trained heads, nine A/B/C manifests, exact paired-head hashes, capped weights and pooled metric arithmetic verified.', '',
               f'Paired group-stratified unique-image bootstrap 95% CI: ΔWGA {b["paired_image_bootstrap_CI_WGA_pp"]} pp; ΔAA {b["paired_image_bootstrap_CI_AA_pp"]} pp. The same sampled image IDs are aligned across all three seeds. WGA uncertainty crosses zero; AA indicates harm conditional on these fixed pipelines. These intervals exclude independent-backbone and method-selection uncertainty.', '',
               '## Failure analysis and scope','',
               'Across the three seeds, 13 predictions were corrected and 43 were broken relative to matched uniform. Landbird-on-water accuracy falls in all three seeds. In seed 0, improving waterbirds-on-land changes which group is worst; in seed 1, landbirds-on-water become the worst group. Real subgroup WGA therefore does not track the latent worst-cluster CE objective consistently.', '',
               'A contains about 559 examples and B about 240. Small class-balanced head subsets produce fewer updates per epoch than full-development training. The existing no-group head adapter retains the final partial batch, uses fixed epoch 20, SGD .001/momentum .9, entropic scale 30, prototype penalty 10 and inverse-decayed IPS starting at 5e5. This exact prior-adapter schedule was frozen and retained; no group-based early stopping or weighting occurs.', '',
               'Gate B removes the OOF-slot transfer problem by fitting weights on the actual heads used on C. It also changes head/weight sample sizes relative to the exploratory sensitivity check. The loss of the earlier +1.79 pp WGA effect cannot be attributed uniquely to one of those changes. It demonstrates that the stronger verification does not sustain the claimed promotion under this protocol.', '',
               'The shared backbone is the previously verified 400-epoch augmented ERM*, not standard ERM. Development data have prior researcher inspection; this is nested engineering verification, not a fresh confirmatory dataset or independent-backbone scientific claim. No production interfaces, flags or default changes were implemented.', '',
               '## Crash recovery and preserved work','',
               'All 61 earlier tasks were reused. New original-source runs and comparisons were added as tasks. A real head process was killed after epoch 1 and resumed from that epoch; final tensors exactly matched uninterrupted training, and the recovered head was reused. A missing joblib serialization import stopped the first fold after head/prediction checkpoints; revision r2 records the import-only correction, preserves r1 sources/manifest and resumes at preprocessing without retraining those heads. Every head has atomic epoch-level optimizer/RNG/model state and a committed final artifact.', '',
               'See state.json, HANDOFF_STATUS.md, NEXT_AGENT_PROMPT.md and completed/ markers. Original gate JSON files remain immutable historical decisions; verification/CURRENT_GATES.json points to the authoritative continuation decisions. Do not modify the frozen algorithm or integrate it to chase better metrics. A different method/protocol requires a new preregistered research task.', '',
               '`python -m experiments.wga_verification.continue_run --resume`']
        report=ROOT/'nested/NESTED_EVALUATION_CURRENT.md';atomic(report,'\n'.join(lines)+'\n')
        audit=ROOT/'nested/independent_integrity_audit.json';dump(audit,dict(status='PASS',heads=135,outer_folds=9,unique_heldout_ids_per_seed=1199,paired_head_identity_verified=True,weights_reconstructed=True,group_blind_training=True,report_source_sha256=sha(__file__)))
        return dict(gate_A='PASS',gate_B='FAIL',integration='PROHIBITED',**means),[table,folds,weight_file,report,audit]
    j.task('continuation_failure_report',['gate_A_with_sources','gate_B_nested'],audit_report)
    report=(ROOT/'nested/NESTED_EVALUATION_CURRENT.md').read_text(encoding='utf-8')
    history=ROOT/'history';history.mkdir(exist_ok=True)
    for name in ('FINAL_SUMMARY.md','BLOCKERS.md','CONTINUATION_BLOCKERS.md','nested/NESTED_EVALUATION.md'):
        p=ROOT/name
        if p.exists():
            target=history/('pre_continuation_'+name.replace('/','_'))
            if not target.exists():atomic(target,p.read_bytes())
    atomic(ROOT/'FINAL_SUMMARY.md',report);atomic(ROOT/'nested/NESTED_EVALUATION.md',report)
    atomic(ROOT/'BLOCKERS.md','No missing source files remain. Gate A PASS. Gate B FAIL: WGA improves in 1/3 seeds, and seed 1 ΔAA = -1.0008340284 pp. Production integration prohibited.\n')
    atomic(ROOT/'CONTINUATION_BLOCKERS.md','Serialization-import issue resolved by recorded revision r2. No runtime blocker remains. Gate B criteria failed.\n')
    dump(ROOT/'verification/CURRENT_GATES.json',dict(gate_A=dict(status='PASS',path='verification/source_comparison/GATE_A.json'),gate_B=dict(status='FAIL',path='verification/source_comparison/GATE_B.json'),production_integration='PROHIBITED'))
    j.status('STOPPED_GATE_B_FAILED','No integration; preserve results and await a new research instruction',gate_A='PASS',gate_B='FAIL',production_integration='PROHIBITED',mean_delta_WGA_pp=b['mean_delta_WGA_pp'],mean_delta_AA_pp=b['mean_delta_AA_pp'])
    atomic(ROOT/'RESULTS_FOR_USER.md','# Kết quả xác minh LCMSW v3\n\nGate A PASS: chạy source gốc và đối chiếu khớp kết quả +1,7899 điểm % WGA, +0,4448 điểm % AA.\n\nGate B FAIL: nested head/weight tách A/B/C cho 3 seed × 3 fold đạt trung bình ΔWGA +0,2915 điểm %, ΔAA −0,8340 điểm %. WGA chỉ tăng ở 1/3 seed; seed 1 giảm AA −1,000834 điểm %, vượt ngưỡng đã khóa. Không nới tiêu chí hoặc chỉnh thuật toán.\n\nKhông tích hợp production. Giữ nguyên checkpoint, source cũ và toàn bộ dự đoán. Backbone là ERM*, không phải ERM chuẩn. Nested evaluation vẫn trên development đã từng được khảo sát, chưa là xác nhận khoa học trên dữ liệu hoàn toàn mới.\n')
    tracked=Path('results/dpe/wga_verification_v3_continuation_20261114');tracked.mkdir(parents=True,exist_ok=True)
    for src,dest in [('nested/NESTED_EVALUATION_CURRENT.md','README.md'),('verification/source_comparison/GATE_A.json','GATE_A.json'),('verification/source_comparison/GATE_B.json','GATE_B.json'),('nested/per_seed_summary.csv','per_seed.csv'),('nested/independent_integrity_audit.json','integrity_audit.json'),('acceptance/real_head_recovery.json','real_head_recovery.json')]:shutil.copyfile(ROOT/src,tracked/dest)
    print(json.dumps(dict(gate_A='PASS',gate_B='FAIL',integration='PROHIBITED',completed=len(j.records),output=ROOT.as_posix())))


if __name__=='__main__':main()
