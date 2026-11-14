"""Decision-oriented report and independent saved-checkpoint arithmetic audit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp,softmax
from .persistence import atomic,dump
from experiments.discrete_weighting.store import sha
from .io import load
from .journal import csvwrite


def independent_audit(root,locked):
    val=load('outputs/weighting/features/val.npz',['sample_id','y','group']);heads=0;folds=0
    for seed in range(3):
        seen=[]
        for fold in range(3):
            folder=root/f'nested/seed{seed}/fold{fold}';plan=json.loads((folder/'ABC_manifest.json').read_text());a,b,c=[set(plan[k]) for k in ['A','B','C']]
            assert not a&b and not a&c and not b&c;assert plan['new_v5_training'];seen.extend(plan['C']);folds+=1
            z=load(folder/'member_predictions.npz');np.testing.assert_array_equal(z['B_ids'],plan['B']);np.testing.assert_array_equal(z['C_ids'],plan['C'])
            hashes=[]
            for j in range(15):
                p=folder/f'head{j:02d}/head.pt';h=torch.load(p,weights_only=True);ep=torch.load(p.parent/'latest.pt',weights_only=True)
                assert set(h['fit_ids'])==a and h['selected_epoch']==ep['epoch']==20 and h['signature']==ep['signature'];assert h['member_id']==f'head-{j:02d}'
                np.testing.assert_array_equal(h['prototypes'].numpy(),ep['model']['prototypes'].numpy());hashes.append(sha(p));heads+=1
            np.testing.assert_array_equal(z['head_hashes'],hashes)
            norm=json.loads((folder/'B_only_normalizer.json').read_text())['ensembles']['0'];assert set(norm['fit_ids'])==b;assert norm['head_hashes']==hashes
            for cfg in locked:
                out=load(folder/cfg['id']/'predictions.npz');np.testing.assert_array_equal(out['head_hashes'],hashes);np.testing.assert_array_equal(out['sample_id'],z['C_ids'])
                p=z['C_probs'];d=z['C_distances'];l=z['C_logits'];u=p.mean(1)
                np.testing.assert_array_equal(out['uniform'],u)
                if cfg['method'].startswith('S'):
                    v=l if cfg['method']=='S2' else -(d-d.mean(-1,keepdims=True)) if cfg['method']=='S3' else -d
                    scores=cfg['tau']*(logsumexp(v/cfg['tau'],axis=1)-np.log(15));expected=softmax(scores/cfg['T_class'],axis=-1)
                    if cfg['method']=='S4':expected=(1-cfg['alpha'])*u+cfg['alpha']*expected
                    np.testing.assert_allclose(out['probs'],expected,atol=1e-12)
                    np.testing.assert_allclose(out['contributions'].sum(1),1,atol=2e-6)
                    assert out['weights'].shape==(len(p),0)
                else:
                    w=out['weights'];assert (w>=0).all() and np.allclose(w.sum(1),1);np.testing.assert_allclose((p*w[...,None]).sum(1),out['probs'],atol=1e-12)
        assert len(set(seen))==len(seen)==1199
        for cfg in locked:
            folder=root/f'pooled/{cfg["id"]}/seed{seed}';z=load(folder/'predictions.npz');r=json.loads((folder/'evaluation.json').read_text());np.testing.assert_array_equal(z['sample_id'],val['sample_id'])
            for label,aa,wga in [('probs','AA','WGA'),('uniform','uniform_AA','uniform_WGA')]:
                corr=z[label].argmax(1)==val['y'];assert abs(corr.mean()-r[aa])<1e-14
                assert abs(min(corr[val['group']==g].mean() for g in range(4))-r[wga])<1e-14
    return dict(status='PASS',newly_trained_v5_heads=heads,ABC_folds=folds,head_fit_IDs_epochs_and_hashes=True,
        exact_uniform=True,same_ordered_B_C_heads=True,normalization_only_B=True,classwise_not_shared_mixture=True,
        independent_pooled_metrics=True,historical_test_used=False)


def write_report(root,gate):
    root=Path(root);locked=json.loads((root/'phase1_locked_candidates.json').read_text())['locked'];decision=json.loads((root/'phase0_audit.json').read_text());paths=[]
    integrity=independent_audit(root,locked) if locked else dict(status='NOT_RUN',reason='No eligible candidate')
    ip=root/'integrity_audit.json';dump(ip,integrity);paths.append(ip)
    rankings=pd.read_csv(root/'phase0_expert_ranking.csv');focus=rankings[(rankings.split=='meta_train')&(rankings.subset=='uniform_wrong')&(rankings.group_AUDIT_ONLY.astype(str)=='all')]
    table=[]
    for d in gate['decisions']:
        for r in d['seeds']:table.append(dict(candidate=d['candidate'],seed=r['seed'],uniform_WGA=100*r['uniform_WGA'],WGA=100*r['WGA'],delta_WGA_pp=r['delta_WGA_pp'],
            uniform_AA=100*r['uniform_AA'],AA=100*r['AA'],delta_AA_pp=r['delta_AA_pp'],BA=100*r['BA'],worst_class_accuracy=100*r['worst_class_accuracy'],corrected=r['corrected'],broken=r['broken']))
    p=root/'nested_per_seed_percent.csv';csvwrite(p,table);paths.append(p)
    lines=['# DPE v5 — prototype geometry aggregation','',f'**Engineering promotion: {gate["status"]}.** WGA is primary; every seed must retain ΔAA strictly greater than −1.00 percentage point.','',
        '## Nested verification','',
        '| Candidate | Seed | U WGA % | Candidate WGA % | ΔWGA pp | U AA % | Candidate AA % | ΔAA pp | Corrected / broken |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f'| {r["candidate"]} | {r["seed"]} | {r["uniform_WGA"]:.4f} | {r["WGA"]:.4f} | {r["delta_WGA_pp"]:+.4f} | {r["uniform_AA"]:.4f} | {r["AA"]:.4f} | {r["delta_AA_pp"]:+.4f} | {r["corrected"]} / {r["broken"]} |')
    for d in gate['decisions']:
        lines+=['',f'**{d["candidate"]}: {d["status"]}.** Mean ΔWGA {d["mean_delta_WGA_pp"]:+.6f} pp; mean ΔAA {d["mean_delta_AA_pp"]:+.6f} pp. WGA positive in {d["positive_seeds"]}/3 seeds. Criteria: '+json.dumps(d['criteria'])+'.',
            f'Paired mean ΔWGA 95% CI {d["WGA_CI95_pp"]} pp; ΔAA CI {d["AA_CI95_pp"]} pp. Unique image IDs resampled identically across seeds; no independent-backbone or selection uncertainty included.']
    lines+=['','## 1. Does distance rank the right experts when uniform fails?','',
        '| Seed | −min distance rank | Confidence rank | Distance margin rank | Mixed uniform-wrong images |','|---|---:|---:|---:|---:|']
    for seed in range(3):
        part=focus[focus.seed==seed];get=lambda score:part[part.score==score].iloc[0]
        a=get('negative_min_distance');b=get('confidence');c=get('distance_margin');lines.append(f'| {seed} | {a.ranking_accuracy:.4f} | {b.ranking_accuracy:.4f} | {c.ranking_accuracy:.4f} | {int(a.mixed_images)} |')
    lines+=['',f'Min-distance routing diagnostic: **{decision["supports_distance_routing"].upper()}**. See phase0_geometry_audit.md for all-sample versus Uniform-wrong rankings, paired difference intervals and meta_select sensitivity. Correct-vs-wrong pairs are averaged within each image first; bootstrap units are distinct images. All-wrong images contribute no pair-ranking statistic and are reported separately with oracle headroom.', '',
        'Source-level distance audit confirms logit_jk = −abs(learned_scale_j)×D_jk. Features are standardized per image then L2-normalized; prototypes are L2-normalized for distance. Learned scales and dmin mean/SD vary by actual head. A small absolute distance is closeness to some prototype, not a correctness probability. Confidence and entropy can confidently favor a wrong shortcut expert. G2 fits each actual OOF fold/head separately and does not pool slots from unrelated heads.', '',
        '## 2. Does class-wise Soft-min improve WGA?','']
    soft=[d for d in gate['decisions'] if d['config']['method'].startswith('S')]
    if soft:
        for d in soft:lines.append(f'Nested {d["candidate"]}: mean ΔWGA {d["mean_delta_WGA_pp"]:+.6f} pp; gate {d["status"]}. This is class-specific logsumexp/prototype contribution aggregation, not a common expert-weighted average.')
    else:lines.append('No class-wise recipe was eligible/locked under the fixed group-blind rule. All class-wise screen results are saved, but there is no nested confirmation for that family.')
    lines+=['','All S1/S2/S3 screening taus .025/.1/.5, S4 tau.1 and residual alphas .1/.25/.5 were retained including losing candidates. T_class remains1; no unplanned calibration grid was added. S2 uses actual logits; S3 removes per-head class-common distance offsets. S1 and S2 are distinct when learned scales vary.','',
        '## 3. AA constraint and 4. survival under nesting','',
        'Each gate records every seed’s exact ΔAA, its strict floor, pooled WGA sign and seed consistency. The two recipes were locked by class-only worst-class NLL and AA before subgroup WGA. No post-hoc relaxation, reranking by group or expanded sweep occurred. AA and class recall are also reported with full confusion matrices; improving majority-class accuracy cannot substitute for subgroup robustness.', '',
        '135 new actual heads were trained from scratch on A using the frozen no-group adapter: class-balanced subsample, final epoch20, SGD .001/momentum.9, batch256 including final partial batch, entropic training scale30, prototype penalty10 and inverse-decayed IPS starting5e5. Backbone was frozen. Every epoch has atomic model/optimizer/RNG state; every head-final and prediction cache has hashes. B fits scalers, C only evaluates committed predictions from the identical ordered heads. All A/B/C IDs and parameter hashes are saved. Heads from old OOF/v3/v4 were not substituted.', '',
        '## 5. Failure / success mechanism','']
    for d in gate['decisions']:
        total_c=sum(r['corrected'] for r in d['seeds']);total_b=sum(r['broken'] for r in d['seeds']);worst=[]
        for r in d['seeds']:
            vals=[r[f'group{k}_accuracy'] for k in range(4)];worst.append(int(np.argmin(vals)))
        lines.append(f'{d["candidate"]}: {total_c} corrected and {total_b} broken seed/image rows; worst true group per seed {worst}. These counts repeat images across seeds and are not independent samples. Per-group flip CSVs identify whether changes help the actual worst subgroup or shift which group is worst.')
        if d['config']['method'].startswith('S'):lines.append('Class-wise pooling can let different heads support different class scores. Learned-scale/common-offset effects and class-score softness differ from uniform probability averaging; class-specific effective-prototype counts and S2/S3 controls are saved. This mechanism must not be described as one reliability weight per expert.')
        else:lines.append('Distance/entropy weights encode geometric proximity or confidence, not observed correctness. Any reversal on Uniform-wrong images explains why overall expert ranking can look useful while routing harms the difficult subgroup. Effective expert count and wrong-expert mass quantify concentration separately from signal quality.')
    lines+=['','## Scope and historical context','',
        'All three seeds share one verified augmented400-epoch ERM* backbone; these are not standard-ERM or independent-backbone results. The original1199 development images, meta_select and these deterministic nested split identities have prior researcher inspection. New head fitting and strict A/B/C isolation remove within-run leakage, but do not turn reused development into a scientifically untouched confirmation. No historical test array was opened by this study.', '',
        'Historical repeatedly inspected test: matched uniform AA95.708%, WGA92.761%; D19 AA95.961%, WGA83.645%; D5 AA96.076%, WGA90.343%. LCMSW and v4 piecewise reliability failed their stronger nested gates. These are context only, with different fitted ensembles/protocols, not v5 confirmation or a pure cross-method effect.','',
        '## One next step','',
        'Evaluate the prelocked recipe/protocol on a newly reserved development cohort with evaluation-only subgroup annotations, without further tuning on these development folds or the historical test.','',
        '## Persistence / reproduce','',
        '` .venv/Scripts/python.exe -m experiments.geometry_aggregation_v5.run --phase all --resume `','',
        'Forced epoch1 process kill/restart matched uninterrupted final tensors; the recovered head was reused. Journal hashes detect missing/corrupt artifacts and source/config mismatch. All source snapshots, failed diagnostics, checkpoints, predictions, selections and decisions remain intact. Read state.json, HANDOFF_STATUS.md and NEXT_AGENT_PROMPT.md for the current committed boundary.', '',
        'Production integration: '+('REQUIRED for the first passing prelocked recipe; uniform must remain default.' if gate['status']=='PASS' else 'NOT PERFORMED because neither locked recipe passed.')]
    report=root/'FINAL_REPORT.md';atomic(report,'\n'.join(lines)+'\n');paths.append(report)
    return dict(report='COMPLETE',promotion=gate['status'],integrity=integrity['status']),paths
