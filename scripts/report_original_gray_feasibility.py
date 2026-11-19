"""Fixed original-gray policy metrics and paired original-group bootstrap."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.counterfactual.common import sha256,write_json,write_csv
from src.counterfactual.fdat_feasibility import prediction_metrics

ROOT=Path(__file__).resolve().parents[1]
POLICIES={'full_05':'full_05_pred','gray_05':'gray_05_pred','locked_FDAT':'fdat_pred','boolean_AND':'and_pred'}
METRICS=['aa_pct','wga_pct','landbird_accuracy_pct','waterbird_accuracy_pct',
         'group0_accuracy_pct','group1_accuracy_pct','group2_accuracy_pct','group3_accuracy_pct',
         'landbird_to_waterbird_rate_pct','waterbird_to_landbird_rate_pct']
GROUPS=['Landbird/Land','Landbird/Water','Waterbird/Land','Waterbird/Water']


def summarize(frame):
    rows=[];base=prediction_metrics(frame,frame.full_05_pred)
    policies={**POLICIES,'historical_full_065':'historical_full_065_pred'}
    for policy,column in policies.items():
        r=prediction_metrics(frame,frame[column]);y=frame.true_y.to_numpy(int);p=frame[column].to_numpy(int)
        r.update(policy=policy,tn=int(((y==0)&(p==0)).sum()),fp=int(((y==0)&(p==1)).sum()),
            fn=int(((y==1)&(p==0)).sum()),tp=int(((y==1)&(p==1)).sum()),
            landbird_to_waterbird_rate_pct=100*float((p[y==0]==1).mean()),
            waterbird_to_landbird_rate_pct=100*float((p[y==1]==0).mean()),
            delta_wga_vs_full05_pp=r['wga_pct']-base['wga_pct'],delta_aa_vs_full05_pp=r['aa_pct']-base['aa_pct'],
            role='historical diagnostic only; not a candidate' if policy=='historical_full_065' else 'locked candidate' if policy!='full_05' else 'reference',
            gate='HISTORICAL_DIAGNOSTIC' if policy=='historical_full_065' else 'PASS' if r['wga_pct']-base['wga_pct']>=-2-1e-10 else 'FAIL',
            worst_group=int(np.argmin([r[f'group{i}_accuracy_pct'] for i in range(4)])))
        group_changes=[r[f'group{i}_accuracy_pct']-base[f'group{i}_accuracy_pct'] for i in range(4)]
        r['worst_individual_group_change_pp']=min(group_changes);r['most_harmed_group']=int(np.argmin(group_changes))
        for i,change in enumerate(group_changes):r[f'group{i}_delta_vs_full05_pp']=change
        rows.append(r)
    return pd.DataFrame(rows)


def paired_bootstrap(frame,samples=2000,seed=20261119):
    if samples<2000:raise ValueError('At least2000 paired source bootstrap draws required')
    group=frame.original_group.to_numpy(int);y=frame.true_y.to_numpy(int)
    if len(frame)!=5794 or frame.source_image_id.nunique()!=5794 or set(group)!={0,1,2,3}:
        raise ValueError('Bootstrap requires exactly5794 unique paired original sources and four groups')
    correct=np.column_stack([(frame[column].to_numpy(int)==y) for column in POLICIES.values()]).astype(float)
    counts=np.array([(group==g).sum() for g in range(4)])
    distributions=[];rng=np.random.default_rng(seed)
    for start in range(0,samples,100):
        size=min(100,samples-start);fractions=[]
        for g in range(4):
            indices=np.flatnonzero(group==g)
            draws=rng.choice(indices,size=(size,len(indices)),replace=True)
            fractions.append(correct[draws].mean(1))
        groups=np.stack(fractions,axis=2) # draw,policy,group
        aa=100*(groups*counts[None,None,:]).sum(2)/counts.sum()
        wga=100*groups.min(2)
        land=100*(groups[:,:,:2]*counts[:2][None,None,:]).sum(2)/counts[:2].sum()
        water=100*(groups[:,:,2:]*counts[2:][None,None,:]).sum(2)/counts[2:].sum()
        distributions.append(np.stack([aa,wga,land,water,*[100*groups[:,:,g] for g in range(4)],100-land,100-water],axis=2))
    samples_array=np.concatenate(distributions)
    point=[]
    for column in POLICIES.values():
        r=prediction_metrics(frame,frame[column])
        point.append([r['aa_pct'],r['wga_pct'],r['landbird_accuracy_pct'],r['waterbird_accuracy_pct'],
            *[r[f'group{g}_accuracy_pct'] for g in range(4)],100-r['landbird_accuracy_pct'],100-r['waterbird_accuracy_pct']])
    point=np.asarray(point);output=[]
    for policy_index,policy in enumerate(POLICIES):
        for j,metric in enumerate(METRICS):
            for reference,label in [(None,'absolute'),(0,'minus_full05')]:
                values=samples_array[:,policy_index,j] if reference is None else samples_array[:,policy_index,j]-samples_array[:,reference,j]
                estimate=point[policy_index,j] if reference is None else point[policy_index,j]-point[reference,j]
                lo,hi=np.quantile(values,[.025,.975])
                output.append(dict(policy=policy,comparison=label,metric=metric,estimate_pct_or_pp=float(estimate),
                    ci95_low_pct_or_pp=float(lo),ci95_high_pct_or_pp=float(hi),bootstrap_samples=samples,bootstrap_seed=seed,
                    n_sources=5794,resampling='source image within each of four official original groups; all policies paired',
                    parameter_reselected=False,independent_final_test=False))
    for j,metric in enumerate(METRICS):
        values=samples_array[:,2,j]-samples_array[:,1,j];lo,hi=np.quantile(values,[.025,.975])
        output.append(dict(policy='locked_FDAT',comparison='minus_gray05',metric=metric,
            estimate_pct_or_pp=float(point[2,j]-point[1,j]),ci95_low_pct_or_pp=float(lo),ci95_high_pct_or_pp=float(hi),
            bootstrap_samples=samples,bootstrap_seed=seed,n_sources=5794,
            resampling='source image within four original groups; full/gray/FDAT paired',
            parameter_reselected=False,independent_final_test=False))
    return pd.DataFrame(output)


def transitions(frame):
    rows=[]
    scopes=[('all',None,frame),*[(f'class{y}',None,frame[frame.true_y==y]) for y in [0,1]],
            *[(f'group{g}',g,frame[frame.original_group==g]) for g in range(4)]]
    comparisons=[('full_05',p) for p in POLICIES]+[('gray_05','locked_FDAT'),('boolean_AND','locked_FDAT')]
    for reference,target in comparisons:
        for scope,group,part in scopes:
            a=part[POLICIES[reference]].to_numpy(int);b=part[POLICIES[target]].to_numpy(int);y=part.true_y.to_numpy(int)
            rows.append(dict(reference=reference,policy=target,scope=scope,official_group=group,count=len(part),
                agreement=int((a==b).sum()),prediction_changes=int((a!=b).sum()),
                reference_wrong_target_correct=int(((a!=y)&(b==y)).sum()),
                reference_correct_target_wrong=int(((a==y)&(b!=y)).sum()),
                both_correct=int(((a==y)&(b==y)).sum()),both_wrong=int(((a!=y)&(b!=y)).sum()),
                Land_to_Water_prediction_changes=int(((a==0)&(b==1)).sum()),
                Water_to_Land_prediction_changes=int(((a==1)&(b==0)).sum())))
    return pd.DataFrame(rows)


def report(c,out):
    pairs=pd.read_csv(out/'original_gray_pairs.csv',float_precision='round_trip')
    if len(pairs)!=5794 or pairs.source_image_id.nunique()!=5794:raise ValueError('Incomplete original-gray inference')
    if not pairs.opaque_rgb_exact.all() or not pairs.outside_gray_exact.all():raise ValueError('Pixel identity failed')
    metrics=summarize(pairs);ci=paired_bootstrap(pairs,c['bootstrap_samples'],c['bootstrap_seed'])
    flips=transitions(pairs)
    species=[]
    for (name,y),part in pairs.groupby(['species_id','true_y'],sort=True):
        for policy,column in POLICIES.items():
            species.append(dict(species_id=name,true_y=int(y),policy=policy,count=len(part),
                accuracy_pct=100*float((part[column]==part.true_y).mean()),
                delta_accuracy_vs_full05_pp=100*float((part[column]==part.true_y).mean()-(part.full_05_pred==part.true_y).mean())))
    pd.DataFrame(species).to_csv(out/'species_summary.csv',index=False)
    diagnostics=pairs.groupby('original_group').agg(count=('source_image_id','size'),mean_p_full=('p_full','mean'),
        mean_p_gray=('p_gray','mean'),median_p_full=('p_full','median'),median_p_gray=('p_gray','median'))
    diagnostics['mean_full_minus_gray']=diagnostics.mean_p_full-diagnostics.mean_p_gray
    diagnostics.to_csv(out/'group_score_diagnostics.csv')
    metrics.to_csv(out/'original_gray_metrics.csv',index=False);ci.to_csv(out/'bootstrap_ci.csv',index=False);flips.to_csv(out/'paired_flips.csv',index=False)
    indexed=metrics.set_index('policy');gray=indexed.loc['gray_05'];fd=indexed.loc['locked_FDAT'];full=indexed.loc['full_05']
    if gray.gate=='FAIL' and fd.gate=='FAIL':verdict='FAIL';action='NO-GO: both fixed policies fail clean protection; do not invent a new gate or start OT retraining.'
    elif gray.gate=='PASS' and gray.wga_pct>=fd.wga_pct-1e-10:
        verdict='PASS';action='STOP FDAT expansion: gray-only ties/exceeds FDAT WGA. Next validate foreground isolation with independent development/final data and realistic masks.'
    elif fd.gate=='PASS':
        verdict='PASS';action='GO only for independent-data investigation of the already-fixed two-view policy; no retuning or novelty claim.'
    else:verdict='PASS';action='STOP FDAT expansion; only gray-only passes this descriptive gate.'
    agreement=int((pairs.fdat_pred==pairs.gray_05_pred).sum());and_difference=int((pairs.fdat_pred!=pairs.and_pred).sum())
    fd_vs_gray=flips[(flips.reference=='gray_05')&(flips.policy=='locked_FDAT')]
    fd_vs_gray_all=fd_vs_gray[fd_vs_gray.scope=='all'].iloc[0]
    group_added='; '.join(f'G{g} {fd[f"group{g}_accuracy_pct"]-gray[f"group{g}_accuracy_pct"]:+.2f}pp' for g in range(4))
    decision=dict(executive_verdict=verdict,criterion='Observed original WGA(policy)-WGA(full05)>=-2.0pp; deterministic triage, not safety guarantee',
        policies={policy:dict(gate=r.gate,wga_pct=float(r.wga_pct),aa_pct=float(r.aa_pct),delta_wga_pp=float(r.delta_wga_vs_full05_pp),
            worst_group=int(r.worst_group),most_harmed_group=int(r.most_harmed_group),
            worst_individual_group_change_pp=float(r.worst_individual_group_change_pp)) for policy,r in indexed.iterrows()},
        paired_sources=5794,exclusions=0,fdAT_minus_gray_WGA_pp=float(fd.wga_pct-gray.wga_pct),
        FDAT_gray_agreement=agreement,FDAT_AND_disagreements=and_difference,one_next_action=action,
        training=False,threshold_search=False,ot_changed=False,already_inspected_test=True,independent_final_test=False)
    write_json(out/'decision.json',decision)
    plots(out,metrics)
    def interval(policy,metric,comparison):
        r=ci[(ci.policy==policy)&(ci.metric==metric)&(ci.comparison==comparison)].iloc[0]
        return f'[{r.ci95_low_pct_or_pp:+.2f}, {r.ci95_high_pct_or_pp:+.2f}]'
    table=['|Policy|AA %|WGA % [95% descriptive CI]|Landbird %|Waterbird %|G0 %|G1 %|G2 %|G3 %|ΔWGA pp [paired CI]|Gate|',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    for policy,r in indexed.iterrows():
        absolute=interval(policy,'wga_pct','absolute') if policy in POLICIES else 'historical comparator'
        paired=interval(policy,'wga_pct','minus_full05') if policy in POLICIES else 'not a candidate'
        table.append('|'+ '|'.join([policy,f'{r.aa_pct:.2f}',f'{r.wga_pct:.2f} {absolute}',f'{r.landbird_accuracy_pct:.2f}',
            f'{r.waterbird_accuracy_pct:.2f}',*[f'{r[f"group{g}_accuracy_pct"]:.2f}' for g in range(4)],
            f'{r.delta_wga_vs_full05_pp:+.2f} {paired}',r.gate])+'|')
    pre=json.loads((out/'preflight.json').read_text());audit=json.loads((out/'dataset_audit_summary.json').read_text())
    run=json.loads((out/'inference_run.json').read_text());runtime=json.loads((out/'model_runtime.json').read_text())
    seconds=sum(r['inference_loop_seconds'] for r in run['runs'] if r['new_gray_forwards']>0)
    audit_text=f'''# Original-gray audit

Exactly5794 unique benchmark test images, official group counts{audit['group_counts']}; exclusions0. Source file/hash, native dimensions/mode, decoded pixel hash, canonical filename, mask/hash/official provenance, soft-alpha statistics, full/gray input hashes and cached logits are in dataset_audit.csv/original_gray_pairs.csv. Every mask was joined by original image ID and canonical CUB filename, checked against the official mask and benchmark dimensions. No similarly named CUB RGB was used to build a gray input.

Gray builder: EXIF transpose RGB and mask together; shared bilinear square256 spatial transform; original resized benchmark RGB attached to bilinear mask alpha; alpha composite neutralRGB127; common center224/ImageNet normalization. Mask alpha is byte-identical to historical premultiplied RGBA resize. The shared RGB-first operation guarantees every opaque model pixel is unchanged; it does not claim the soft-edge RGB equals the previous CUB-RGB gray builder. No mask threshold, dilation, opacity hardening, blur, enhancement, JPEG re-encoding or learned segmentation. Lossless in-memory gray images were inferred.

Opaque/core RGB exact, outside-alpha0 exact127, soft blending equation verified to<=1 uint8 rounding level on every image. {audit['images_without_model_opaque']} images have no fully opaque model pixel; {audit['images_without_eroded_core']} have no5×5 eroded opaque core. Those soft masks are not hardened or dropped; RGB before blending stays identical, and soft regions blend by their supplied alpha. Alpha can be soft over broad interior regions as well as edges, so this fixed convention can reduce contrast there; no claim of all support pixels being opaque is made.

Preflight:24 unique images,6/group, {pre['species']} species, small-mask/thin-boundary/complex-background strata selected by mask/texture, never correctness/confidence. Contact sheets show actual224 model inputs, mask overlay and expected absolute RGB differences (not saliency). Numeric original/cached parity max probability error{pre['maximum_probability_error']:.3g} (declared tolerance{pre['probability_tolerance']:g}), logits{pre['maximum_logit_error']:.3g} (tolerance{pre['logit_tolerance']:g}); all input pixel and normalized-tensor hashes match. Visual review is saved before full inference in preflight_visual_review.json.

Frozen POTER seed1, validation-selected epoch258, checkpointSHA{c['checkpoint_sha256']}; ModelAdapter and Runner preprocessing reused. FP32, cuDNN/Matmul TF32 disabled, deterministic cuDNN, benchmark off; exact runtime in model_runtime.json. No training/OT/search.24 fresh original parity forwards plus5794 gray forwards total (including24 preflight gray inputs reused). Added gray inference loop time{seconds:.2f}s; preflight{pre['preflight_seconds']:.2f}s includes parity+gray checks. CPU dataset audit took{audit['elapsed_seconds']:.2f}s. These are wall times for this machine/run, not pure CUDA-kernel latency. Resumable atomic cache shards and dependency hashes prevent mismatched reuse.

Old composite-gray scores were never joined to originals. All legacy results/config/checkpoints are preserved. The optional original0.65 comparator uses existing full scores and is not fitted here. Only POTER was loaded; optional ERM/DPE evaluation is deferred to keep this gate bounded.
'''
    (out/'audit.md').write_text(audit_text,encoding='utf-8')
    worse=[]
    for policy in ['gray_05','locked_FDAT']:
        r=indexed.loc[policy]
        worse.append(f'- {policy}: most harmed group G{r.most_harmed_group} ({GROUPS[int(r.most_harmed_group)]}) changes{r.worst_individual_group_change_pp:+.2f}pp; its worst absolute group is G{r.worst_group}.')
    text=f'''# Original Waterbirds gray-only feasibility gate

## 1. Executive verdict: **{verdict}**

Gray-only clean-protection gate **{gray.gate}**, locked FDAT **{fd.gate}**. Observed FDAT minus gray-only WGA is{fd.wga_pct-gray.wga_pct:+.2f}pp; paired95% descriptive CI{interval('locked_FDAT','wga_pct','minus_gray05')}pp. This answers the one missing original-test check without another parameter search. **{action}**

## 2. Data/pixel audit and score parity

Exactly5794 benchmark/HF JPEGs with official masks, one-to-one IDs and all four official original groups; exclusions0. Gray uses the exact benchmark RGB, not the old CUB foreground RGB. Shared square256/bilinear and center224 geometry; neutral gray127; original grayscale mask alpha0..255; original ImageNet normalization. Foreground RGB before blending and every opaque/core model pixel is identical. Soft-alpha pixels blend by design, including broad soft mask interiors; those values are reported, not misrepresented as unchanged opaque foreground. No label/group/species/sigma enters the builder or model forward.

All original model-view pixel hashes match the5794 cached full inputs.24 diverse preflight examples passed source/mask/alignment/pixel checks and visual review; live full-score parity max probability error{pre['maximum_probability_error']:.3g}, with unchanged prespecified tolerances. See audit.md, preflight_probability_parity.csv and three eight-example contact-sheet pages for source details. Mask alignment is dataset-provenance/geometry plus visual QC, not a guarantee that every soft boundary is semantically perfect.

## 3. Original-test results

G0=Landbird/Land, G1=Landbird/Water, G2=Waterbird/Land, G3=Waterbird/Water. Official denominators:{audit['group_counts']}. AA uses all5794 images; WGA is minimum of the four group accuracies. CIs are paired source bootstrap within those groups (2000 draws), recomputing WGA every draw. Full class/group/error/AA intervals are in bootstrap_ci.csv.

{chr(10).join(table)}

The0.65 row is a historical diagnostic comparator from cached full scores; it was not optimized or introduced as another candidate. All policies in this task were locked before new gray inference.

## 4. Clean protection and group-specific harm

Gate uses the observed WGA change>=−2.0pp. It is a project triage bound, not a statistical safety guarantee; its point decision can differ from whether the descriptive CI crosses−2pp. Gray ΔWGA={gray.delta_wga_vs_full05_pp:+.2f}pp, paired CI{interval('gray_05','wga_pct','minus_full05')}pp; FDAT ΔWGA={fd.delta_wga_vs_full05_pp:+.2f}pp, paired CI{interval('locked_FDAT','wga_pct','minus_full05')}pp.

{chr(10).join(worse)}

Read all four group changes and class recalls rather than relying on AA or a PASS alone. paired_flips.csv retains full-correct→policy-wrong and full-wrong→policy-correct counts by class/group; group_score_diagnostics.csv reports score shifts, not a causal mechanism.

## 5. Relation to historical composites

The previous sigma2 natural-composite WGA58% full/86% gray/86% FDAT is fixed exploratory context, not a new independent finding. It involved100 foregrounds, four repeated backgrounds and old CUB-RGB gray views. This run evaluates5794 distinct original images with new benchmark-RGB gray inputs. Source birds overlap prior tests; results cannot be combined as independent replications. Composite-clean sigma0 is also different from unmodified original-clean. Historical blur gray inputs were not regenerated using this exact benchmark-RGB-first builder; this clean gate does not revalidate the historical+28pp under one production image-generation pipeline. No new blur experiment was run.

## 6. FDAT added value and boundary ablation

Locked formula d=max(0,p_full−p_gray),t=min(1,.5+d),predictWater iff p_full>=t. It agrees with gray-only on{agreement}/5794 images and changes{5794-agreement}. FDAT/AND disagreements:{and_difference}; full=1 with gray<.5 cases:{int(((pairs.p_full==1)&(pairs.p_gray<.5)).sum())}. The capped >=rule is not mathematically AND at every possible saturated input; exact differences are reported rather than assumed.

Relative to gray-only, FDAT corrects{int(fd_vs_gray_all.reference_wrong_target_correct)} mistakes and introduces{int(fd_vs_gray_all.reference_correct_target_wrong)} new mistakes across its{int(fd_vs_gray_all.prediction_changes)} changed decisions. AA changes{fd.aa_pct-gray.aa_pct:+.2f}pp; Landbird accuracy{fd.landbird_accuracy_pct-gray.landbird_accuracy_pct:+.2f}pp; Waterbird accuracy{fd.waterbird_accuracy_pct-gray.waterbird_accuracy_pct:+.2f}pp. Individual group changes are{group_added}. Thus it helps Landbird groups/AA while harming Waterbird groups and the primary WGA compared with gray-only; no added worst-group value is demonstrated here. Pairwise transitions by group are in paired_flips.csv (reference gray_05, policy locked_FDAT), and species_summary.csv exposes species concentration. No label-selected gates, per-class thresholds, new lambda/cap or model changes were evaluated.

## 7. Practical limitations and cost

Dataset masks are oracle assets. Quantized soft alpha sometimes covers foreground interiors; blending can change contrast there. A realistic segmentation pipeline, mask-error robustness and deployment latency are untested. This diagnostic added one gray forward per source while reusing cached full scores. Deployed gray-only could replace the full classifier with one gray-view forward plus segmentation/composition; FDAT needs both full and gray classifier views, normally two forwards plus segmentation. Added full-data gray inference took{seconds:.2f}s of this run's loop plus preflight/audit/export overhead. ModelAdapter runtime:{runtime['numeric_runtime']}. Only one POTER checkpoint was checked; no seed-stability or POTER-specific novelty claim. Official labels and known dataset-label limitations are unchanged.

Source-image bootstrap intervals describe this fixed, repeatedly inspected test set. They do not account for algorithm ideation on prior test analyses, new environments, segmentation-model errors, or independent training randomness; species can remain dependent. No untouched holdout or unbiased deployment gain is asserted.

## 8. One next action

**{action}** Do not start another search, fabricate original performance from old composite scores, invent a new adaptive branch, or automatically retrain OT/POTER. Any later contribution claim needs genuinely independent development/final data and realistic masks.

## Reproduce and artifacts

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_original_gray_feasibility.py --stage preflight
# Inspect the24 saved examples and record preflight_visual_review.json.
.\\.venv\\Scripts\\python.exe scripts\\run_original_gray_feasibility.py --stage infer
.\\.venv\\Scripts\\python.exe scripts\\run_original_gray_feasibility.py --stage report
```

The infer stage resumes verified gray-cache shards and never overwrites them. The report stage reads saved scores only. Required original_gray_pairs.csv, original_gray_metrics.csv, paired_flips.csv, bootstrap_ci.csv, decision.json, contact/group/tradeoff PNGs and manifests are present. Source/config/checkpoint/runtime/input/output hashes are retained. No earlier report/cache/checkpoint was overwritten.
'''
    if (out/'tests.log').exists():text+='\nAutomated checks: '+(out/'tests.log').read_text().strip().splitlines()[-1]+'\n'
    (out/'algorithm_feasibility_report.md').write_text(text,encoding='utf-8')
    codes=['configs/original_gray_feasibility.json','src/counterfactual/original_gray.py',
        'scripts/run_original_gray_feasibility.py','scripts/report_original_gray_feasibility.py','tests/test_original_gray.py']
    write_json(out/'implementation_provenance.json',dict(sha256={p:sha256(ROOT/p) for p in codes if (ROOT/p).is_file()},
        model_checkpoint_sha256=c['checkpoint_sha256'],no_training=True,no_threshold_selection=True))
    write_csv(out/'files_manifest.csv',[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha256(p))
        for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['files_manifest.csv','integrity.json']])
    write_json(out/'integrity.json',dict(sha256={p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='integrity.json'}))
    print(json.dumps(dict(report=str(out/'algorithm_feasibility_report.md'),verdict=verdict,
        gray_gate=gray.gate,FDAT_gate=fd.gate,gray_WGA=gray.wga_pct,FDAT_WGA=fd.wga_pct)),flush=True)


def plots(out,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=metrics.set_index('policy');xs=np.arange(4)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    for j,(name,color) in enumerate([('full_05','#2878a4'),('gray_05','#437e4c'),('locked_FDAT','#b64b39')]):
        values=[data.loc[name,f'group{g}_accuracy_pct'] for g in range(4)]
        bars=ax.bar(xs+(j-1)*.25,values,width=.25,label=name,color=color)
        ax.bar_label(bars,fmt='%.2f',fontsize=8)
    ax.set(title='Original5794 official groups — fixed-policy exploratory feasibility',xticks=xs,xticklabels=GROUPS,ylabel='Accuracy (%)',ylim=(0,106));ax.legend(loc='lower left')
    fig.savefig(out/'group_accuracy_comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for name,color,marker in [('full_05','#2878a4','o'),('gray_05','#437e4c','s'),('locked_FDAT','#b64b39','^'),('historical_full_065','gray','x')]:
        r=data.loc[name];ax.scatter(r.aa_pct,r.wga_pct,s=100,marker=marker,color=color,label=name)
        ax.annotate(f'{name}\nAA{r.aa_pct:.2f} WGA{r.wga_pct:.2f}',(r.aa_pct,r.wga_pct),xytext=(6,7),textcoords='offset points',fontsize=8)
    ax.axhline(data.loc['full_05','wga_pct']-2,color='gray',ls='--',label='−2pp observed gate')
    ax.set(title='Original clean WGA versus AA — no threshold fitting',xlabel='AA (%)',ylabel='Four-group WGA (%)')
    ax.margins(x=.25,y=.2);ax.legend(fontsize=8,loc='lower left')
    fig.savefig(out/'original_clean_tradeoff.png',dpi=160,bbox_inches='tight');plt.close(fig)
