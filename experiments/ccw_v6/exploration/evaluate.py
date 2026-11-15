"""Evaluation starts only after all label-free predictions have committed."""
import json
import numpy as np
from experiments.ccw_v6.model import METHODS
from experiments.ccw_v6.inputs import canonical
from experiments.geometry_aggregation_v5.io import load,save_npz
from experiments.geometry_aggregation_v5.persistence import atomic,dump
from experiments.geometry_aggregation_v5.journal import csvwrite
from experiments.discrete_weighting.store import sha
from .journal import BASE,ROOT,BANNER,COMMAND
from .run import cpath,predpath,modelpath


def metric(probs,y,group):
    correct=probs.argmax(-1)==y
    assert set(np.unique(group))=={0,1,2,3} and set(np.unique(y))=={0,1}
    groups=[float(correct[group==g].mean()) for g in range(4)]
    classes=[float(correct[y==c].mean()) for c in range(2)]
    return dict(n=len(y),AA=float(correct.mean()),WGA=min(groups),WCA=min(classes),BA=float(np.mean(classes)),
        group_accuracy=groups,group_count=[int((group==g).sum()) for g in range(4)],class_accuracy=classes)


def pool(s,m):
    parts=[load(predpath(s,f,m)) for f in range(3)]
    labels=[load(cpath(s,f),['sample_id','y','group']) for f in range(3)]
    for a,b in zip(parts,labels):np.testing.assert_array_equal(a['sample_id'],b['sample_id'])
    ids=np.concatenate([p['sample_id'] for p in parts]);assert len(ids)==len(set(ids))==1199
    expected=np.array([r['sample_id'] for r in canonical(BASE)]);lookup={v:i for i,v in enumerate(ids)}
    assert set(ids)==set(expected);order=np.array([lookup[v] for v in expected])
    out={k:np.concatenate([p[k] for p in parts])[order] for k in ['probs','weights','r','sample_id']}
    out.update({k:np.concatenate([p[k] for p in labels])[order] for k in ['y','group']})
    out['fold']=np.concatenate([np.full(len(p['sample_id']),f) for f,p in enumerate(parts)])[order]
    np.testing.assert_array_equal(out['sample_id'],expected)
    assert set(np.unique(out['group']))=={0,1,2,3}
    return out


def evaluate(s,m):
    out=pool(s,m);rows=[]
    for f in [-1,0,1,2]:
        mask=np.ones(len(out['y']),bool) if f==-1 else out['fold']==f
        r=metric(out['probs'][mask],out['y'][mask],out['group'][mask]);r.update(seed=s,fold=f,method=m,level='seed' if f==-1 else 'fold',role=BANNER)
        rows.append(r)
    p=ROOT/f'evaluations/seed{s}/{m}.json';dump(p,rows)
    return dict(seed=s,method=m,AA=rows[0]['AA'],WGA=rows[0]['WGA'],role=BANNER),[p]


def prediction_export():
    pooled=[[pool(s,m) for s in range(3)] for m in METHODS];ref=pooled[0][0]
    for group in pooled:
        for p in group:
            for k in ['y','group','sample_id']:np.testing.assert_array_equal(p[k],ref[k])
    arrays={k:np.stack([[p[k] for p in group] for group in pooled]) for k in ['probs','weights','r']}
    arrays.update(sample_id=ref['sample_id'],y=ref['y'],group=ref['group'],methods=np.array(METHODS),seeds=np.arange(3),classes=np.arange(2),
        fold=np.stack([p['fold'] for p in pooled[0]]),role=np.array(BANNER),
        original_NO_GO_sha256=np.array(sha(BASE/'LOCKED_CANDIDATE.json')))
    arrays['probability_shift_vs_uniform']=arrays['probs']-arrays['probs'][0:1]
    arrays['probability_shift_vs_g1']=arrays['probs']-arrays['probs'][1:2]
    arrays['weight_shift_vs_g1']=arrays['weights']-arrays['weights'][1:2]
    arrays['head_sha256']=np.array([[load(cpath(s,f),['head_sha256'])['head_sha256'] for f in range(3)] for s in range(3)])
    arrays['B_model_sha256']=np.array([[sha(modelpath(s,f)) for f in range(3)] for s in range(3)])
    # Every sample maps to the head order at [seed, fold[seed,sample], expert].
    for k in ['probs','distances']:
        members=[]
        for s in range(3):
            parts=[load(cpath(s,f),[k,'sample_id']) for f in range(3)];ids=np.concatenate([v['sample_id'] for v in parts]);idx={v:i for i,v in enumerate(ids)}
            members.append(np.concatenate([v[k] for v in parts])[[idx[v] for v in ref['sample_id']]])
        arrays['member_'+k]=np.stack(members)
    p=ROOT/'EXPLORATORY_C_PREDICTIONS.npz';save_npz(p,**arrays)
    schema=ROOT/'PREDICTION_SCHEMA.json';dump(schema,dict(role=BANNER,shapes={k:list(v.shape) for k,v in arrays.items()},
        axes=dict(probs='method, seed, canonical image, class',weights='method, seed, canonical image, expert',r='unmixed softmax router weights',
                  member_probs='seed, canonical image, expert, class',head_sha256='seed, outer fold, expert'),
        head_order='head_sha256[seed, fold[seed,image], expert]',probability_precision='Original float32 baselines preserved numerically when stacked with float64 CCW outputs',
        full_per_fold_internals='predictions/seedS/foldF/METHOD.npz',label_policy='Original official labels; evaluation only'))
    return dict(role=BANNER,samples=1199,seeds=3,methods=5,probs_shape=list(arrays['probs'].shape),weights_shape=list(arrays['weights'].shape)),[p,schema]


def uncertainty():
    z=load(ROOT/'EXPLORATORY_C_PREDICTIONS.npz');rng=np.random.default_rng(20261115);before=rng.bit_generator.state
    correct=z['probs'].argmax(-1)==z['y'];repeats=2000;group_stats=[];counts=[]
    for g in range(4):
        idx=np.flatnonzero(z['group']==g);draw=idx[rng.integers(0,len(idx),(repeats,len(idx)))];counts.append(len(idx))
        group_stats.append(correct[:,:,draw].mean(-1))
    ga=np.stack(group_stats,axis=-1);AA=(ga*np.array(counts)).sum(-1)/sum(counts);WGA=ga.min(-1)
    rows=[]
    for mi,m in enumerate(METHODS):
        for ri,reference in [(0,'U0'),(1,'G1_frozen')]:
            for seed in [0,1,2,-1]:
                r=dict(method=m,reference=reference,seed=seed,role=BANNER)
                for key,values in [('AA',AA),('WGA',WGA)]:
                    d=values[mi]-values[ri];d=d.mean(0) if seed==-1 else d[seed]
                    ci=np.quantile(100*d,[.025,.975]);r[key+'_CI95_low_pp']=float(ci[0]);r[key+'_CI95_high_pp']=float(ci[1])
                rows.append(r)
    p=ROOT/'uncertainty/bootstrap.npz';save_npz(p,AA=AA,WGA=WGA,group_accuracy=ga,role=np.array(BANNER))
    q=ROOT/'uncertainty/paired_CI.json';dump(q,rows)
    r=ROOT/'uncertainty/rng_state.json';dump(r,dict(role=BANNER,before=before,after=rng.bit_generator.state,repeats=repeats,
        unique_images=1199,independent_rows_not_3597=True,stratification='C groups for evaluation uncertainty only',
        sampling='Same sampled image IDs in every seed and every method; paired conditional fixed-pipeline image bootstrap',
        limitation='Does not estimate variance across newly trained backbones, new cohorts or head seeds beyond these three.'))
    return dict(status='PASS',role=BANNER,repeats=repeats,unique_image_clustered=True),[p,q,r]


def all_rows():
    return [r for s in range(3) for m in METHODS for r in json.loads((ROOT/f'evaluations/seed{s}/{m}.json').read_text())]


def table(name):
    z=load(ROOT/'EXPLORATORY_C_PREDICTIONS.npz');correct=z['probs'].argmax(-1)==z['y'];rows=[]
    raw=all_rows();lookup={(r['method'],r['seed'],r['fold']):r for r in raw}
    ci=json.loads((ROOT/'uncertainty/paired_CI.json').read_text());cis={(r['method'],r['reference'],r['seed']):r for r in ci}
    if name=='RESULTS':
        for r in raw:
            row={k:r[k] for k in ['method','seed','fold','level','n','role']}
            for k in ['AA','WGA','WCA','BA']:row[k+'_pct']=100*r[k]
            for reference in ['U0','G1_frozen']:
                other=lookup[reference,r['seed'],r['fold']]
                for k in ['AA','WGA','WCA','BA']:row[f'delta_{k}_vs_{reference}_pp']=100*(r[k]-other[k])
                row['AA_decrease_lt_1pp_vs_'+reference]=r['AA']-other['AA']>-.01
                if r['fold']==-1:
                    for k,v in cis[r['method'],reference,r['seed']].items():
                        if '_CI95_' in k:row[f'{k}_vs_{reference}']=v
            rows.append(row)
        for m in METHODS:
            seeds=[r for r in rows if r['method']==m and r['level']=='seed'];row=dict(method=m,seed=-1,fold=-1,level='mean_of_seeds',n=1199,role=BANNER)
            for k in seeds[0]:
                if k.endswith(('_pct','_pp')) and '_CI95_' not in k:row[k]=float(np.mean([r[k] for r in seeds]))
                if k.startswith('AA_decrease_lt'):row[k]=all(r[k] for r in seeds)
            for ref in ['U0','G1_frozen']:
                for k,v in cis[m,ref,-1].items():
                    if '_CI95_' in k:row[k+'_vs_'+ref]=v
            rows.append(row)
    elif name=='PER_GROUP':
        for s in range(3):
            for mi,m in enumerate(METHODS):
                for fold in [-1,0,1,2]:
                    for g in range(4):
                        mask=(z['group']==g)&(True if fold==-1 else z['fold'][s]==fold);acc=correct[mi,s,mask].mean()
                        row=dict(method=m,seed=s,fold=fold,level='seed' if fold==-1 else 'fold',group=g,n=int(mask.sum()),accuracy_pct=100*float(acc),role=BANNER)
                        for ri,ref in [(0,'U0'),(1,'G1_frozen')]:
                            a=correct[mi,s,mask];b=correct[ri,s,mask]
                            row['delta_vs_'+ref+'_pp']=100*float(a.mean()-b.mean());row['corrected_vs_'+ref]=int((a&~b).sum());row['broken_vs_'+ref]=int((~a&b).sum())
                        rows.append(row)
        for mi,m in enumerate(METHODS):
            for g in range(4):
                seeds=[r for r in rows if r['method']==m and r['group']==g and r['level']=='seed']
                row=dict(method=m,seed=-1,fold=-1,level='mean_of_seeds',group=g,n=int((z['group']==g).sum()),role=BANNER)
                for k in ['accuracy_pct','delta_vs_U0_pp','delta_vs_G1_frozen_pp','corrected_vs_U0','broken_vs_U0','corrected_vs_G1_frozen','broken_vs_G1_frozen']:row[k]=float(np.mean([r[k] for r in seeds]))
                rows.append(row)
    elif name=='FLIPS':
        predictions=z['probs'].argmax(-1)
        for mi,m in enumerate(METHODS):
            for ri,ref in [(0,'U0'),(1,'G1_frozen'),(3,'CCW_A')]:
                for s in range(3):
                    for i in np.flatnonzero(predictions[mi,s]!=predictions[ri,s]):
                        rows.append(dict(method=m,reference=ref,seed=s,fold=int(z['fold'][s,i]),sample_id=str(z['sample_id'][i]),y=int(z['y'][i]),group=int(z['group'][i]),
                            prediction=int(predictions[mi,s,i]),reference_prediction=int(predictions[ri,s,i]),corrected=bool(correct[mi,s,i] and not correct[ri,s,i]),broken=bool(not correct[mi,s,i] and correct[ri,s,i]),
                            p0=float(z['probs'][mi,s,i,0]),p1=float(z['probs'][mi,s,i,1]),reference_p0=float(z['probs'][ri,s,i,0]),reference_p1=float(z['probs'][ri,s,i,1]),
                            shift_p0=float(z['probs'][mi,s,i,0]-z['probs'][ri,s,i,0]),shift_p1=float(z['probs'][mi,s,i,1]-z['probs'][ri,s,i,1]),role=BANNER))
    else:raise ValueError(name)
    p=ROOT/f'EXPLORATORY_C_{name}.csv';csvwrite(p,rows)
    return dict(table=name,rows=len(rows),role=BANNER),[p]


def report():
    import csv
    def read(name):
        with (ROOT/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
    rows=read('EXPLORATORY_C_RESULTS.csv');groups=read('EXPLORATORY_C_PER_GROUP.csv');z=load(ROOT/'EXPLORATORY_C_PREDICTIONS.npz')
    means={r['method']:r for r in rows if r['level']=='mean_of_seeds'};seeds={(r['method'],int(r['seed'])):r for r in rows if r['level']=='seed'}
    full=means['CCW_FULL'];w=z['weights'];p=z['probs'];pred=p.argmax(-1)
    tv=.5*np.abs(w[4]-w[3]).sum(-1);flip=pred[4]!=pred[3]
    effect=dict(role=BANNER,weight_TV_mean=float(tv.mean()),weight_TV_max=float(tv.max()),max_absolute_weight_change=float(np.abs(w[4]-w[3]).max()),
        top_expert_changes=int((w[4].argmax(-1)!=w[3].argmax(-1)).sum()),seed_image_rows=3597,
        mean_absolute_probability_shift=float(np.abs(p[4]-p[3]).mean()),max_absolute_probability_shift=float(np.abs(p[4]-p[3]).max()),
        prediction_flips=int(flip.sum()),unique_images_flipped=int(flip.any(0).sum()),
        delta_AA_pp=float(full['AA_pct'])-float(means['CCW_A']['AA_pct']),delta_WGA_pp=float(full['WGA_pct'])-float(means['CCW_A']['WGA_pct']))
    worst=[]
    for s in range(3):
        for ref in ['U0','G1_frozen']:
            old=[r for r in groups if r['level']=='seed' and r['method']==ref and int(r['seed'])==s];minimum=min(float(r['accuracy_pct']) for r in old)
            for r in old:
                if abs(float(r['accuracy_pct'])-minimum)<1e-10:
                    a=next(v for v in groups if v['level']=='seed' and v['method']=='CCW_FULL' and int(v['seed'])==s and v['group']==r['group'])
                    worst.append(dict(seed=s,reference=ref,group=int(r['group']),n=int(a['n']),reference_accuracy_pct=minimum,
                        full_accuracy_pct=float(a['accuracy_pct']),corrected=int(a['corrected_vs_'+ref]),broken=int(a['broken_vs_'+ref]),delta_pp=float(a['delta_vs_'+ref+'_pp'])))
    evidence=ROOT/'ADVANTAGE_AND_WORST_GROUP.json';dump(evidence,dict(role=BANNER,advantage=effect,baseline_worst_groups=worst))
    text=['# CCW v6: requested exploratory held-out C evaluation','', '**'+BANNER+'**','',
        'The original B-selection NO_GO is unchanged. No production promotion or default inference change is authorized by these results. This is the requested measurement after the failed preregistered B proxy gate; it is not a replacement gate.','',
        'Nine matching v5 prototype ensembles (3 seeds × 3 outer folds); all 240 B images per ensemble fit the frozen recipe. C has 1,199 distinct images per seed, reused across all three seeds (3,597 prediction rows). A, B and C are disjoint within each ensemble. All normalization, pooled distance-z quantiles, B-only boundary-confidence quantile, shrinkage tables and advantage scale use B only. C labels and groups are used only for baseline verification and post-inference evaluation. No retraining, new configurations or parameter optimization.','',
        'Frozen parameters: rho=0.5, beta=1, gamma=50, alpha=0.25, kappa=25. The exact frozen G2 control uses z-score geometry (gamma=1); CCW-A sets beta=0. Uniform and G1 reproduce v5 probabilities, predictions and AA/WGA within 1e-12 (observed probability error zero). All 135 ordered head hashes and all B/C arrays match provenance.','',
        '| Method | Mean AA (%) | Mean WGA (%) | ΔAA vs Uniform (pp) | ΔWGA vs Uniform (pp) | ΔWGA vs G1 (pp) |',
        '|---|---:|---:|---:|---:|---:|']
    for m in METHODS:
        r=means[m];text.append(f"| {m} | {float(r['AA_pct']):.6f} | {float(r['WGA_pct']):.6f} | {float(r['delta_AA_vs_U0_pp']):+.6f} | {float(r['delta_WGA_vs_U0_pp']):+.6f} | {float(r['delta_WGA_vs_G1_frozen_pp']):+.6f} |")
    text+=['','| Seed | Method | AA (%) | WGA (%) | Worst-class accuracy (%) | ΔAA vs Uniform (pp) | ΔAA vs G1 (pp) |','|---:|---|---:|---:|---:|---:|---:|']
    for s in range(3):
        for m in METHODS:
            r=seeds[m,s];text.append(f"| {s} | {m} | {float(r['AA_pct']):.6f} | {float(r['WGA_pct']):.6f} | {float(r['WCA_pct']):.6f} | {float(r['delta_AA_vs_U0_pp']):+.6f} | {float(r['delta_AA_vs_G1_frozen_pp']):+.6f} |")
    text+=['','Requested questions:','']
    for i,(ref,name) in enumerate([('U0','Uniform'),('G1_frozen','G1')],1):
        d=float(full['delta_WGA_vs_'+ref+'_pp']);lo=float(full['WGA_CI95_low_pp_vs_'+ref]);hi=float(full['WGA_CI95_high_pp_vs_'+ref]);per=[float(seeds['CCW_FULL',s]['delta_WGA_vs_'+ref+'_pp']) for s in range(3)]
        text.append(f'{i}. Does CCW-Full outperform {name} in observed WGA? {"Yes" if d>0 else "No"}: mean difference {d:+.6f} pp; per seed {per}. Paired 95% image-bootstrap CI [{lo:+.6f}, {hi:+.6f}] pp. This is descriptive exploratory evidence only.')
    floors={ref:[float(seeds['CCW_FULL',s]['delta_AA_vs_'+ref+'_pp']) for s in range(3)] for ref in ['U0','G1_frozen']}
    text+=['3. AA decrease strictly below 1 pp per seed? '+str(all(d> -1 for v in floors.values() for d in v))+' versus both references; AA differences (pp): '+json.dumps(floors)+'.',
        '4. Actual corrections on the baseline worst group are reported below. WGA depends on the minimum across all four groups, so a net improvement in one group alone need not raise WGA.',
        '','| Seed | Baseline | Baseline worst group | n | Baseline acc (%) | Full acc (%) | Corrected | Broken | Net Δ (pp) |',
        '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in worst:text.append(f"| {r['seed']} | {r['reference']} | {r['group']} | {r['n']} | {r['reference_accuracy_pct']:.6f} | {r['full_accuracy_pct']:.6f} | {r['corrected']} | {r['broken']} | {r['delta_pp']:+.6f} |")
    text+=['',f"5. Advantage term versus CCW-A: mean weight total variation {effect['weight_TV_mean']:.8f}, maximum {effect['weight_TV_max']:.8f}; largest individual expert-weight change {effect['max_absolute_weight_change']:.8f}. Top expert changes on {effect['top_expert_changes']}/3597 seed-image rows; prediction changes on {effect['prediction_flips']} rows ({effect['unique_images_flipped']} unique images). Mean absolute probability shift {effect['mean_absolute_probability_shift']:.8f}; maximum {effect['max_absolute_probability_shift']:.8f}. Mean AA change {effect['delta_AA_pp']:+.6f} pp; mean WGA change {effect['delta_WGA_pp']:+.6f} pp. These quantities distinguish weight movement from actual classification effects; no post-hoc tuning follows.",
        '', 'Group definitions: 0=landbird/land, 1=landbird/water, 2=waterbird/land, 3=waterbird/water. Per-seed counts are 467, 466, 133, 133. All per-group accuracies, corrections and breakages are in EXPLORATORY_C_PER_GROUP.csv; individual changes versus Uniform, G1 and CCW-A are in EXPLORATORY_C_FLIPS.csv. All full probability shifts and all 15 expert weights are retained in the NPZ, including unchanged predictions.',
        '', 'Uncertainty uses 2,000 paired, group-stratified draws of unique image IDs; each draw is shared across all methods and all three seeds. Mean CIs average seed metrics within each draw. Treating the 3,597 repeated image rows as independent would be incorrect. Intervals are conditional on these fixed fitted pipelines; three head seeds on one shared frozen ERM* backbone do not establish variability across newly trained backbones or new cohorts.',
        '', 'Original official labels are retained, including the documented species-label caveat; no relabeling was performed. These are previously inspected development images, not a fresh untouched test cohort. Neither positive nor negative results confirm generalization. WGA remains primary; AA floor checks use strict > −1 pp per seed. Mean WGA is the average of three per-seed minima, not the minimum of seed-averaged group accuracies.',
        '', 'Crash safety: every full-B fit, fold inference, model evaluation and output table is committed atomically with source/input/config hashes. Real full-B forced termination verified reuse of the committed first fit and replay only of the uncommitted second fit, equal to an uninterrupted control. Acceptance fits are reused in the main run. Fits are deterministic; bootstrap RNG states are saved. All original completion artifacts and the original NO_GO are audited unchanged.',
        '', 'Resume / verify:', '', '```powershell', COMMAND, COMMAND+' --verify', '```','',
        'Main outputs: EXPLORATORY_C_RESULTS.csv (fold, seed and mean metrics, paired deltas and CIs); EXPLORATORY_C_PER_GROUP.csv; EXPLORATORY_C_FLIPS.csv; EXPLORATORY_C_PREDICTIONS.npz; PREDICTION_SCHEMA.json; ADVANTAGE_AND_WORST_GROUP.json. Fits and complete inference internals remain under fits/ and predictions/. Recovery receipts and RNG state remain under acceptance/ and uncertainty/.','']
    p=ROOT/'EXPLORATORY_C_REPORT.md';atomic(p,'\n'.join(text))
    return dict(role=BANNER,mean_AA_pct=float(full['AA_pct']),mean_WGA_pct=float(full['WGA_pct']),production_promotion=False),[p,evidence]
