"""Read-only C evaluators. Loaded after B locking; never imported by the fitter."""
import csv
import json
from pathlib import Path
import numpy as np
from experiments.geometry_aggregation_v5.io import load,save_npz
from experiments.geometry_aggregation_v5.persistence import atomic,dump
from experiments.discrete_weighting.store import sha
from experiments.score_audit_v4.evaluate import summary,paired_ci
from .model import METHODS,infer,teacher
from .store import csvwrite

BANNER='REUSED DEVELOPMENT; NOT CONFIRMATORY; NO C-BASED TUNING'


def promotion_allowed(fresh_attested,rows):
    if not fresh_attested:return dict(status='BLOCKED',reason='No independently attested untouched cohort',production_integration=False)
    checks=dict(mean_WGA_vs_U0_positive=np.mean([r['delta_WGA_U0_pp'] for r in rows])>0,
        WGA_vs_U0_positive_in_2_seeds=sum(r['delta_WGA_U0_pp']>0 for r in rows)>=2,
        AA_floor_every_seed=all(r['delta_AA_U0_pp']>-1 for r in rows),
        mean_WGA_vs_G1_positive=np.mean([r['delta_WGA_G1_pp'] for r in rows])>0,
        WGA_vs_G1_positive_in_2_seeds=sum(r['delta_WGA_G1_pp']>0 for r in rows)>=2)
    return dict(status='PASS' if all(checks.values()) else 'FAIL',criteria={k:bool(v) for k,v in checks.items()},production_integration=False)


def evaluate(out,y,group,ids,member_probs):
    p=out['probs'];u=out['uniform'];g=out['g1_probs'];r=summary(p,y,group);ur=summary(u,y,group);gr=summary(g,y,group)
    r.update(delta_AA_U0_pp=(r['AA']-ur['AA'])*100,delta_AA_G1_pp=(r['AA']-gr['AA'])*100,
        delta_WGA_U0_pp=(r['WGA']-ur['WGA'])*100,delta_WGA_G1_pp=(r['WGA']-gr['WGA'])*100,
        effective_members=float((1/(out['weights']**2).sum(1)).mean()),entropy=float(-(out['weights']*np.log(out['weights'].clip(1e-15))).sum(1).mean()),
        head00_top_share=float((out['weights'].argmax(1)==0).mean()),
        **{'vs_U0_'+k:v for k,v in paired_ci(p,u,y,group,ids,500).items()},
        **{'vs_G1_'+k:v for k,v in paired_ci(p,g,y,group,ids,500).items()},role=BANNER)
    corr=p.argmax(1)==y;flips=[]
    for name,base in [('U0',u),('G1',g)]:
        old=base.argmax(1)==y
        for k in range(4):
            m=group==k;flips.append(dict(reference=name,group=k,count=int(m.sum()),corrected=int((m&corr&~old).sum()),broken=int((m&~corr&old).sum())))
    t=teacher(member_probs,y);direction=(out['r']*t).sum(1)
    per=[dict(sample_id=str(ids[i]),y=int(y[i]),group_AUDIT_ONLY=int(group[i]),prediction=int(p[i].argmax()),uniform_prediction=int(u[i].argmax()),G1_prediction=int(g[i].argmax()),
        q_true_minus_uniform_true=float(direction[i]),mixture_true_minus_uniform_true=float(p[i,y[i]]-u[i,y[i]]),role=BANNER) for i in range(len(y))]
    return r,flips,per


def run_retrospective(root,journal,lock):
    if lock['status']!='GO':
        def skipped():
            with (root/'results/g1_reproduction.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
            path=root/'results/C_EXPLORATORY_ONLY.csv';csvwrite(path,[dict(**r,role=BANNER,CCW_replay='NOT_RUN_B_NO_GO') for r in rows])
            note=root/'reports/C_EXPLORATORY_ONLY.md';atomic(note,'# '+BANNER+'\n\nB selection was NO_GO. No new CCW C predictions were computed. The CSV contains only the already reproduced U0/G1 baselines; absent CCW results are not fabricated.\n')
            return dict(C_replay='SKIPPED_B_NO_GO'),[path,note]
        return journal.task('retrospective_report',['phase1_report'],skipped)
    task_ids=[]
    for s in range(3):
        for f in range(3):
            source=root/f'inputs/v5_analysis/caches/seed{s}/fold{f}/C_predictions.npz';modelpath=root/f'checkpoints/full_B/seed{s}/fold{f}/model.json';model=json.loads(modelpath.read_text())
            for m in METHODS:
                folder=root/f'results/C/seed{s}/fold{f}/{m}';pid=f'C_predict:{s}:{f}:{m}'
                def predict(source=source,model=model,modelpath=modelpath,folder=folder,m=m):
                    C=load(source,['probs','distances','sample_id','head_sha256']);out=infer(C['probs'],C['distances'],model,C['head_sha256'],m);g1=infer(C['probs'],C['distances'],None,C['head_sha256'],'G1_frozen')['probs']
                    path=folder/'predictions.npz';save_npz(path,**out,g1_probs=g1,sample_id=C['sample_id']);receipt=folder/'inference_receipt.json';dump(receipt,dict(B_model_sha256=sha(modelpath),C_cache_sha256=sha(source),selection_lock_sha256=sha(root/'LOCKED_CANDIDATE.json'),labels_or_groups_read=False,role=BANNER))
                    return dict(method=m,prediction_rows=len(C['sample_id']),C_labels_used_for_inference=False),[path,receipt]
                journal.task(pid,['phase1_report',f'full_fit:{s}:{f}'],predict)
                def score(source=source,folder=folder,s=s,f=f,m=m):
                    C=load(source,['probs','y','group','sample_id']);out=load(folder/'predictions.npz');np.testing.assert_array_equal(out['sample_id'],C['sample_id'])
                    r,flips,per=evaluate(out,C['y'],C['group'],C['sample_id'],C['probs']);r.update(seed=s,outer_fold=f,method=m,level='fold')
                    path=folder/'evaluation.json';dump(path,r);fp=folder/'corrected_broken.csv';csvwrite(fp,flips);pp=folder/'per_image.csv';csvwrite(pp,per);return r,[path,fp,pp]
                eid=f'C_evaluate:{s}:{f}:{m}';journal.task(eid,[pid],score);task_ids.append(eid)
        for m in METHODS:
            def pool(s=s,m=m):
                parts=[load(root/f'results/C/seed{s}/fold{f}/{m}/predictions.npz') for f in range(3)];Cparts=[load(root/f'inputs/v5_analysis/caches/seed{s}/fold{f}/C_predictions.npz',['probs','y','group','sample_id']) for f in range(3)]
                ids=np.concatenate([p['sample_id'] for p in parts]);order=np.argsort(ids);assert len(ids)==len(set(ids))==1199
                out={k:np.concatenate([p[k] for p in parts])[order] for k in parts[0]};C={k:np.concatenate([p[k] for p in Cparts])[order] for k in Cparts[0]}
                r,flips,per=evaluate(out,C['y'],C['group'],C['sample_id'],C['probs']);r.update(seed=s,outer_fold=-1,method=m,level='pooled_seed')
                folder=root/f'results/C_pooled/seed{s}/{m}';path=folder/'predictions.npz';save_npz(path,**out,y=C['y'],group=C['group']);ep=folder/'evaluation.json';dump(ep,r);fp=folder/'corrected_broken.csv';csvwrite(fp,flips);pp=folder/'per_image.csv';csvwrite(pp,per)
                return r,[path,ep,fp,pp]
            tid=f'C_pool:{s}:{m}';journal.task(tid,[f'C_evaluate:{s}:{f}:{m}' for f in range(3)],pool);task_ids.append(tid)
    def report():
        rows=[journal.records[t]['result'] for t in task_ids];path=root/'results/C_EXPLORATORY_ONLY.csv';csvwrite(path,rows)
        flips=[]
        for s in range(3):
            for m in METHODS:
                with (root/f'results/C_pooled/seed{s}/{m}/corrected_broken.csv').open(encoding='utf-8',newline='') as f:flips.extend(dict(seed=s,method=m,**r,role=BANNER) for r in csv.DictReader(f))
        fp=root/'results/C_corrected_broken_by_group.csv';csvwrite(fp,flips);note=root/'reports/C_EXPLORATORY_ONLY.md';atomic(note,'# '+BANNER+'\n\nThe method/configuration was locked solely from B before replay. No C result selects a recipe or changes the B decision. All point estimates are retrospective, never independent verification or promotion evidence.\n')
        return dict(C_replay='EXPLORATORY_ONLY',promotion_allowed=False),[path,fp,note]
    return journal.task('retrospective_report',task_ids,report)


def confirmation(root,lock,fresh_manifest):
    # A file is not evidence of freshness. Require explicit cohort provenance and a new
    # sealed evaluation stage before any labels could certify a scientific promotion.
    supplied=Path(fresh_manifest).is_file()
    reason='NO FRESH DATA' if not supplied else 'FRESH FILE REQUIRES PROVENANCE/HEAD/ROUTING VALIDATION BEFORE LABEL ACCESS'
    decision=promotion_allowed(False,[])
    path=root/'reports/CONFIRMATION_BLOCKED.md';atomic(path,'# CONFIRMATION BLOCKED — '+reason+'\n\nCCW is not confirmed. No CCW production inference changes are authorized by old C or B crossfit. B decision: '+lock['status']+'. Existing Uniform and v5 G1 remain intact.\n')
    request=root/'INPUT_REQUEST.md';atomic(request,'# Inputs required for independent CCW confirmation\n\n'
        'B inputs are complete: nine verified240-image matching-head B caches already exported. No B/model regeneration is needed.\n\n'
        'No verified untouched cohort is currently available. Provide `experiments/ccw_v6/fresh_inputs/CCW_FRESH_COHORT.json` describing the following assets:\n\n'
        '- Explicit attestation of no prior researcher/model-selection exposure, source/cohort identity, annotation policy and collection provenance; a new filename or reused1199 validation images is insufficient.\n'
        '- Nine fresh prediction caches identified by seed0/1/2 and actual outer-head ensemble fold0/1/2, with probabilities/distances/sample IDs/logits if available and the exact ordered15 head hashes recorded by the existing B models. Inference only, no retraining required.\n'
        '- Same frozen ERM* backbone SHA256 `bce28dc72c3737ae19ffa057ba056e03d3f531180e7ad55e0b3c93bf4ea4bcaa`; inference/normalization/prototype-distance provenance, source and file hashes.\n'
        '- A routing/assignment manifest fixed before labels: disjoint evaluation folds within seed, the same unique cohort across seeds, no reuse of original A/B/C, backbone-training or previously inspected test images.\n'
        '- Separately sealed class/subgroup annotations keyed by unique sample_id, classes[0,1], group=2*y+background, with nonempty support for every evaluated subgroup and label-source audit. Do not place labels into scoring/selection APIs.\n\n'
        'Continuation: preserve current B lock/source and all completed results. First validate and freeze new cohort file/ID/head hashes and input lineage before opening its annotations. Run the locked B models once, commit predictions, then evaluate the frozen fresh gate versus U0 and G1. Never use old C to rescue a NO_GO B decision. Adding a file does not automatically attest freshness or override the existing failure/lock.\n')
    p=root/'reports/promotion_decision.json';dump(p,{**decision,'B_decision':lock['status'],'fresh_cohort_supplied':supplied,'reason':reason,'C_metrics_are_confirmatory':False,'existing_G1':'UNCHANGED','default_Uniform':'UNCHANGED'})
    return dict(confirmation='BLOCKED',reason=reason,production_integration=False),[path,request,p]
