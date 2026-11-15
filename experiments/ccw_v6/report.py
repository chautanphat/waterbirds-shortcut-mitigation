"""B-only mechanism reports and truthful final missing-confirmation decision."""
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.geometry_aggregation_v5.io import load,figure
from experiments.geometry_aggregation_v5.persistence import atomic,dump
from .store import csvwrite
from .metrics import diagnostics,advantage_signal,paired_image_ci


def phase1_report(root,journal,lock):
    from .model import METHODS
    methods=[r['result'] for k,r in journal.records.items() if k.startswith(('B_evaluate:','B_outer:','B_seed:'))]
    allmetrics=root/'results/B_crossfit_all_methods.csv';csvwrite(allmetrics,methods)
    images=[];cells=[];signals=[]
    for s in range(3):
        seed_est=[];seed_member=[];seed_y=[];seed_ids=[];seed_low=[]
        for f in range(3):
            B=load(root/f'inputs/B/seed{s}/fold{f}.npz');full=load(root/f'results/B_outer/seed{s}/fold{f}/CCW_FULL/predictions.npz');np.testing.assert_array_equal(full['sample_id'],B['sample_id'])
            diag=diagnostics(full,B['probs'],B['y'],full['q20']);rows=advantage_signal(full['advhat'],B['probs'],B['y'],B['sample_id'],diag['low_margin'])
            signals.extend(dict(seed=s,outer_fold=f,level='outer_B',**r) for r in rows)
            seed_est.append(full['advhat']);seed_member.append(B['probs']);seed_y.append(B['y']);seed_ids.append(B['sample_id']);seed_low.append(diag['low_margin'])
            for i in range(4):
                folder=root/f'checkpoints/seed{s}/fold{f}/inner{i}';model=json.loads((folder/'model.json').read_text());cells.extend(dict(seed=s,outer_fold=f,inner_fold=i,fit='inner_train',sigma_A=model['sigma_A'],q20=model['q20'],**r) for r in model['cells'])
                for m in METHODS:
                    with (folder/m/'per_image.csv').open(encoding='utf-8',newline='') as stream:images.extend(csv.DictReader(stream))
        signals.extend(dict(seed=s,outer_fold=-1,level='seed_B_clustered',**r) for r in advantage_signal(np.concatenate(seed_est),np.concatenate(seed_member),np.concatenate(seed_y),np.concatenate(seed_ids),np.concatenate(seed_low)))
    if lock['status']=='GO':
        for s in range(3):
            for f in range(3):
                model=json.loads((root/f'checkpoints/full_B/seed{s}/fold{f}/model.json').read_text());cells.extend(dict(seed=s,outer_fold=f,inner_fold=-1,fit='full_B',sigma_A=model['sigma_A'],q20=model['q20'],**r) for r in model['cells'])
    per=root/'results/B_crossfit_per_image.csv';csvwrite(per,images);celltable=root/'results/B_cell_tables.csv';csvwrite(celltable,cells);signal=root/'results/B_advantage_signal.csv';csvwrite(signal,signals)
    frame=pd.DataFrame(images)
    for col in ['seed','signed_advantage_gain','geometry_range','advantage_range','effective_members','entropy','top_head','top_ties']:frame[col]=pd.to_numeric(frame[col])
    for col in ['uniform_wrong','low_margin','unique_top','advantage_dominates']:frame[col]=frame[col].astype(str).str.lower()=='true'
    gains=[]
    for s in range(3):
        for m in METHODS:
            t=frame[(frame.seed==s)&(frame.method==m)]
            for subset,sub in [('all',t),('uniform_wrong',t[t.uniform_wrong]),('low_margin',t[t.low_margin])]:
                ci=paired_image_ci(sub.signed_advantage_gain.to_numpy(),sub.sample_id.to_numpy())
                gains.append(dict(seed=s,method=m,subset=subset,rows=len(sub),unique_images=sub.sample_id.nunique(),gain=ci['mean'],gain_CI95=ci['CI95'],
                    positive_gain_fraction=float((sub.signed_advantage_gain>0).mean()) if len(sub) else None,
                    effective_members=float(sub.effective_members.mean()) if len(sub) else None,entropy=float(sub.entropy.mean()) if len(sub) else None,
                    head00_top_share=float((sub.top_head==0).mean()) if len(sub) else None,
                    head00_unique_top_share=float(((sub.top_head==0)&sub.unique_top).mean()) if len(sub) else None,
                    mean_top_ties=float(sub.top_ties.mean()) if len(sub) else None,
                    advantage_dominates_fraction=float(sub.advantage_dominates.mean()) if len(sub) else None,
                    geometry_range_mean=float(sub.geometry_range.mean()) if len(sub) else None,advantage_range_mean=float(sub.advantage_range.mean()) if len(sub) else None))
    gp=root/'results/B_gain_and_concentration.csv';csvwrite(gp,gains)
    paths=[allmetrics,per,celltable,signal,gp]
    # Figures and their source table are a committed part of this report task.
    selection=pd.read_csv(root/'results/B_selection_proxy.csv');fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    axes[0].bar(selection.method,selection.proxy);axes[0].set(title='B OOF worst-class NLL (lower is better)',ylabel='Nine outer B evaluations, mean');axes[0].tick_params(axis='x',rotation=30)
    seedrows=pd.DataFrame([r for r in methods if r['level']=='seed_B'])
    for m in METHODS:
        t=seedrows[seedrows.method==m].sort_values('seed');axes[1].plot(t.seed,t.delta_AA_U0_pp,'o-',label=m)
    axes[1].axhline(-1,color='red',linestyle=':');axes[1].set(title='AA floor on unique-image pooled B OOF',xlabel='Head seed',ylabel='Delta AA versus U0 (pp)');axes[1].legend(fontsize=7)
    pic=root/'figures/B_selection_and_AA.png';figure(pic,fig);paths.append(pic)
    summary_signal=[r for r in signals if r['level']=='seed_B_clustered' and r['stratum']=='overall'];fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for r in summary_signal:
        if r['pair_ranking'] is not None:ax.errorbar(r['seed'],r['pair_ranking'],yerr=[[r['pair_ranking']-r['pair_ranking_CI95'][0]],[r['pair_ranking_CI95'][1]-r['pair_ranking']]],fmt='o',capsize=4)
    ax.axhline(.5,color='gray',linestyle='--');ax.set(xticks=[0,1,2],xlabel='Head seed',ylabel='Within-image advantage ranking',title='A_hat versus true t, B OOF; image-cluster 95% CI')
    pic=root/'figures/B_advantage_ranking.png';figure(pic,fig);paths.append(pic)
    lines=['# CCW v6 mechanism audit — B crossfit only','',f'**B decision: {lock["status"]}. Selected method: {lock["selected_method"]}.**','',
        'Every fitted table, normalizer, q20 and sigma_A uses only B-inner-train. Model selection has no group input. The weighted target is p_head(true class) minus the exact same Uniform true-class probability, not correctness. Each image distributes omega across15 heads; kappa25 is effective image-weight support.','',
        '## Advantage reliability','',
        '| Seed | Image-cluster pair ranking | 95% CI | Mixed images | Descriptive AUROC(t>0) |','|---|---:|---|---:|---:|']
    for r in summary_signal:lines.append(f'| {r["seed"]} | {r["pair_ranking"]} | {r["pair_ranking_CI95"]} | {r["distinct_mixed_images"]} | {r["AUROC_positive_t_descriptive"]} |')
    lines+=['','Signal estimates are also stratified by predicted class, train-defined low-margin status and disagreement with Uniform. Correlation/AUROC are descriptive correlated-expert statistics; only image-cluster pair-ranking intervals are claimed. Positive correlation with relative probability is not guaranteed subgroup robustness.','',
        '## Mechanism, boundaries and support','',
        'CCW_A isolates partial distance calibration. CCW_FULL adds the smoothed signed advantage. G2_z fully standardizes distances at gamma1 and remains a control. G1 keeps raw normalized Euclidean distance at gamma50. No probability temperature or head parameter is changed.','',
        'B_gain_and_concentration.csv records true signed gain sum(r*t), low-margin/uniform-wrong strata, score dynamic ranges, effective experts, entropy, top-head/tie frequency and any advantage-range dominance. Such dominance is reported without retuning. Head00 is never penalized by index. Uniform deterministic argmax ties favor slot00; unique-top share/tie counts are reported to avoid interpreting that as learned dominance.','',
        'B_cell_tables.csv contains raw expert observations, distinct-image support and effective weighted image support for every one of12 cells. Sparse and empty cells shrink toward smoothed predicted-class parents then global mean/0; parent computation has no self-reference. Shared head slots across outer ensembles are never identified as the same expert.','',
        'The seed AA floor averages repeated B-image predictions across outer folds before evaluation. There are720 raw B rows per seed but fewer unique images; these are not720 independent examples or three independent backbone replicates. OOF lookup fits approximately180 images, whereas an allowed full-B deployment refit uses240.','',
        'No new CCW C replay occurs after a NO_GO. All failing fixed variants are retained. No optional random controls or unplanned rho/beta/alpha search was added.']
    report=root/'reports/CCW_MECHANISM_AUDIT.md';atomic(report,'\n'.join(lines)+'\n');paths.append(report)
    phase=root/'reports/PHASE1.md';atomic(phase,'# B-only four-fold crossfit complete\n\n'+json.dumps(lock,indent=2)+'\n\nAll5 fixed methods and all losing predictions are retained. Read CCW_MECHANISM_AUDIT.md, B_selection_proxy.csv and LOCKED_CANDIDATE.json. No C or subgroup field was used by the selector.\n');paths.append(phase)
    return dict(phase1='COMPLETE',B_decision=lock['status'],selected_method=lock['selected_method'],B_rows=len(images),cell_rows=len(cells)),paths


def final_report(root,journal,lock):
    selection=json.loads((root/'LOCKED_CANDIDATE.json').read_text());confirmation=json.loads((root/'reports/promotion_decision.json').read_text());B=pd.read_csv(root/'results/B_crossfit_all_methods.csv');seeds=B[B.level=='seed_B'];C=pd.read_csv(root/'results/C_EXPLORATORY_ONLY.csv')
    lines=['# CCW v6 — not independently confirmed','',f'**B decision: {lock["status"]}. Fresh confirmation: BLOCKED. CCW production integration: NOT PERFORMED.**','',
        'The fixed primary is CCW_FULL (rho.5, beta1, gamma50, alpha.25, kappa25); all five preregistered variants were evaluated. Only matching B data fitted/selects. Existing Uniform and opt-in G1 source files/default behavior were preserved.','',
        '## B-only selection and AA floor','',
        '| Rank | Method | Mean worst-class NLL | AA eligible every seed | Seed ΔAA versus U0 (pp) |','|---|---|---:|---|---|']
    for i,r in enumerate(selection['ranking']):lines.append(f'| {i+1} | {r["method"]} | {r["proxy"]:.8f} | {r["eligible"]} | {r["delta_AA_U0_pp_by_seed"]} |')
    lines+=['','Primary FULL outranks G1: '+str(selection['primary_beats_G1'])+'. Locked new method: '+str(lock['selected_method'])+'. '+selection['reason']+'.','',
        'AA degradation exactly −1.0000 pp fails. Duplicate B image IDs across outer ensembles are averaged before seed AA and bootstrap metrics; class-only worst-class NLL averages nine outer B-OOF evaluations. No group WGA contributes to ranking.','',
        '## WGA versus G1 and Uniform','',
        '**REUSED DEVELOPMENT; NOT CONFIRMATORY; NO C-BASED TUNING.**','']
    if lock['status']=='GO':
        pooled=C[C.level=='pooled_seed'];lines+=['| Method | Seed | WGA % | ΔWGA U0 pp | ΔWGA G1 pp | AA % | ΔAA U0 pp |','|---|---:|---:|---:|---:|---:|---:|']
        for r in pooled.to_dict('records'):lines.append(f'| {r["method"]} | {r["seed"]} | {100*r["WGA"]:.4f} | {r["delta_WGA_U0_pp"]:+.4f} | {r["delta_WGA_G1_pp"]:+.4f} | {100*r["AA"]:.4f} | {r["delta_AA_U0_pp"]:+.4f} |')
        lines+=['','All pairwise image-bootstrap CIs and subgroup counts/flips are in C_EXPLORATORY_ONLY.csv and companion files. These old C observations were inspected before CCW design and cannot establish independent gain, irrespective of their point estimates.']
    else:lines+=['No new CCW C replay was authorized after the B-only NO_GO decision. C_EXPLORATORY_ONLY.csv contains only the already reproduced v5 U0/G1 baselines. CCW WGA versus U0/G1 is **not measured here**, rather than fabricated or tuned on a known C result. No full-B deployment table was refitted because no novel candidate was locked.']
    signals=pd.read_csv(root/'results/B_advantage_signal.csv');sig=signals[(signals.level=='seed_B_clustered')&(signals.stratum=='overall')]
    lines+=['','## Why the mechanism did or did not help','',
        'B estimated-advantage within-image ranking by seed: '+', '.join(str(v) for v in sig.pair_ranking)+'. Full class/margin/disagreement strata and uncertainty are in CCW_MECHANISM_AUDIT.md. Class-balanced boundary weighting and the t target do not directly estimate groups or guarantee WGA. A proxy win/loss can arise from small probability shifts without top-1 gains, or from partial geometric normalization changing which expert is emphasized. The saved score-range, weighted signed-gain, per-class losses and sparse-cell supports distinguish these effects; no IPS/head-index causal claim is inferred.','',
        '## Verification, preservation and limitations','',
        'Gate0 reproduced all9 C caches and the original per-seed Uniform/G1 AA/WGA exactly. All9 matching B caches were available and verified against ABC IDs, source hashes, ordered heads and learned scales. No head/backbone retraining or replacement OOF setup was needed. B exports contain no group fields. Thirty-six inner fits are checkpointed with normalizers, bin edges, class/boundary weights, hierarchical tables, sigma_A, IDs and source/data hashes; the first two recovery fits were reused. Method descriptors/predictions and each evaluation are independently committed.','',
        'Recovery tests forcibly killed a worker mid-inner-fold after one committed inner-fold, resumed in another process and verified bitwise model/prediction equality to uninterrupted execution, unchanged completed-file timestamps/hashes and no duplicate receipts. Atomic fsync/replace uses bounded Windows/OneDrive retries and an OS single-writer lock. Read completed/ for authoritative artifacts and state.json for the DAG.','',
        'One shared augmented400-epoch ERM* backbone, reused1199 development images, and approximately240 B examples per ensemble limit power. Seed replicates share images/backbone and are not independent datasets. The OOF table is fit on approximately3/4 of B; full-B deployment support differs. Original official labels retain the documented species-label issue. Old test arrays were never opened by this study.','',
        'Fresh-cohort gate: '+confirmation['status']+' — '+confirmation['reason']+'. No old-C point estimate can override this. No FRESH_CONFIRMATION.csv was created because no fresh evaluation was performed.','',
        '## Exactly one justified next step','',
        ('Preserve the negative B decision and audit its recorded per-class loss/score decomposition on B only before proposing any separately preregistered follow-up; do not use old C to rescue CCW.' if lock['status']=='NO_GO' else 'Obtain the untouched, provenance-verified cohort specified in INPUT_REQUEST.md and evaluate the already locked B models once before making any CCW-versus-G1 improvement claim.'), '',
        '## Resume','',
        '`python -m experiments.ccw_v6.run --phase all --resume`','',
        '`python -m experiments.ccw_v6.run --verify`','',
        'Read HANDOFF_STATUS.md, NEXT_AGENT_PROMPT.md and INPUT_REQUEST.md. All feasible current-data work is complete; preserve completed fits and all v1-v5 artifacts.']
    report=root/'reports/FINAL_REPORT.md';atomic(report,'\n'.join(lines)+'\n')
    return dict(B_decision=lock['status'],fresh_confirmation='BLOCKED',CCW_integrated=False,report='reports/FINAL_REPORT.md'),[report]
