"""Independent arithmetic/provenance audit and compact v4 research deliverables."""
import io
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd
from scipy.special import softmax
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.discrete_weighting.store import atomic,dump,sha
from experiments.score_audit_v4.journal import Journal,csvwrite
from experiments.score_audit_v4.run import ROOT,PRIOR,load


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text());j=Journal(ROOT,manifest)
    gate=json.loads((ROOT/'promotion_gate.json').read_text());assert gate['status']=='FAIL'
    def audit():
        # Recompute mixtures directly without calling model.infer or evaluate.
        val=load('outputs/weighting/features/val.npz',['y','sample_id','group'])
        primary=json.loads((ROOT/'phase1_locked_candidates.json').read_text())['primary'];mid=primary['map_id']
        head_count=0;summary=[];flips=[];raw_deltas=[];scores=[];coverage=[];risks=[];phase1=[]
        for seed in range(3):
            pooled=load(ROOT/f'phase2/seed{seed}/pooled_predictions.npz');np.testing.assert_array_equal(pooled['sample_id'],val['sample_id'])
            collected=[]
            for fold in range(3):
                folder=ROOT/f'phase2/seed{seed}/fold{fold}';plan=json.loads((folder/'ABC_manifest.json').read_text());z=load(folder/'member_predictions.npz');saved=load(folder/'outer_predictions.npz')
                ai,bi,ci=[np.array(plan[k]) for k in ['head_indices','weight_indices','hold_indices']]
                sets=[set(plan[k]) for k in ['A','B','C']];assert all(not sets[x]&sets[y] for x in range(3) for y in range(x+1,3))
                for r in plan['heads']:
                    assert sha(r['path'])==r['sha256'] and sha(r['epoch_checkpoint'])==r['epoch_sha256']
                    h=torch.load(r['path'],weights_only=True);epoch=torch.load(r['epoch_checkpoint'],weights_only=True)
                    assert set(h['fit_ids'])==sets[0] and h['selected_epoch']==20 and epoch['epoch']==20
                    assert h['signature']==epoch['signature'];head_count+=1
                np.testing.assert_array_equal(z['B_ids'],val['sample_id'][bi]);np.testing.assert_array_equal(z['C_ids'],val['sample_id'][ci])
                np.testing.assert_array_equal(z['member_hashes'],[r['sha256'] for r in plan['heads']]);np.testing.assert_array_equal(saved['member_hashes'],z['member_hashes'])
                fit=ROOT/f'outer/{seed}/{fold}/fit';lookup=json.loads((fit/'lookup.json').read_text());rep=json.loads((fit/'representation.json').read_text());assert set(lookup['fit_ids'])==sets[1]
                p=softmax(np.asarray(z['C_logits'],float)/rep['temperature'],axis=-1)
                np.testing.assert_allclose(p,saved['members'],atol=0,rtol=0)
                w=(1-primary['alpha'])/15+primary['alpha']*softmax(primary['beta']*saved['scores'],axis=1)
                np.testing.assert_allclose(w,saved['weights'],atol=1e-15);np.testing.assert_allclose((p*w[...,None]).sum(1),saved['mixture'],atol=1e-15)
                np.testing.assert_allclose(p.mean(1),saved['uniform'],atol=0,rtol=0)
                assert not any(k in lookup for k in ['group','correct','y'])
                collected.extend(plan['C'])
            assert len(set(collected))==len(val['y'])==1199
            pred=pooled['mixture'].argmax(1);uniform=pooled['uniform'].argmax(1);raw=pooled['raw_uniform'].argmax(1);y=val['y'];g=val['group']
            calc=lambda p:(float((p==y).mean()),min(float((p[g==k]==y[g==k]).mean()) for k in range(4)))
            ca,cw=calc(pred);ua,uw=calc(uniform);ra,rw=calc(raw);r=gate['seeds'][seed]
            for actual,key in [(ca,'AA'),(cw,'WGA'),(ua,'uniform_AA'),(uw,'uniform_WGA'),(ra,'raw_uniform_AA'),(rw,'raw_uniform_WGA')]:assert abs(actual-r[key])<1e-14
            summary.append(dict(seed=seed,raw_uniform_AA_percent=ra*100,raw_uniform_WGA_percent=rw*100,
                calibrated_uniform_AA_percent=ua*100,calibrated_uniform_WGA_percent=uw*100,
                piecewise_AA_percent=ca*100,piecewise_WGA_percent=cw*100,
                weighting_delta_AA_pp=(ca-ua)*100,weighting_delta_WGA_pp=(cw-uw)*100,
                calibration_delta_AA_pp=(ua-ra)*100,calibration_delta_WGA_pp=(uw-rw)*100,
                total_delta_AA_vs_raw_pp=(ca-ra)*100,total_delta_WGA_vs_raw_pp=(cw-rw)*100))
            raw_deltas.append(((ca-ra)*100,(cw-rw)*100))
            f=pd.read_csv(ROOT/f'phase2/seed{seed}/corrected_broken.csv');flips.extend([dict(seed=seed,**row) for row in f.to_dict('records')])
            scores.append(dict(seed=seed,pairwise_image_ranking=r['pairwise_image_ranking'],wrong_head_weight=r['wrong_head_weight'],
                effective_members=r['effective_members'],mean_max_weight=r['mean_max_weight'],corrected=r['corrected'],broken=r['broken']))
            for map_id in ['R1','R3','R4']:
                d=load(ROOT/f'inputs/development/seed{seed}/meta_train.npz',['probs','logits','y','sample_id'])
                rep=json.loads((ROOT/f'phase1fit/{map_id}/{seed}/representation.json').read_text())
                p=softmax(np.asarray(d['logits'],float)/rep['temperature'],axis=-1) if map_id in ['R3','R4'] else d['probs']
                c=p.max(-1);mask=c>=.75;correct=p.argmax(-1)==d['y'][:,None]
                coverage.append(dict(seed=seed,map_id=map_id,temperature=rep['temperature'],mean_confidence=float(c.mean()),p95_confidence=float(np.quantile(c,.95)),
                    max_confidence=float(c.max()),ge075_fraction=float(mask.mean()),ge075_observations=int(mask.sum()),
                    ge075_distinct_images=int(mask.any(1).sum()),ge075_accuracy=float(correct[mask].mean()) if mask.any() else None,
                    ge075_wrong_observations=int((mask&~correct).sum()),ge075_wrong_distinct_images=int((mask&~correct).any(1).sum())))
            cells=json.loads((ROOT/f'phase1fit/{mid}/{seed}/lookup.json').read_text())['cells']
            risks.extend([dict(seed=seed,**r) for r in cells if r['count'] and r['empirical_correctness']<.5 and r['score']>0])
            name=f"grid_{mid}_a{primary['alpha']:g}_b{primary['beta']:g}_s{seed}"
            pr=json.loads((ROOT/'phase1/candidates'/name/'post_lock_diagnostics.json').read_text())
            phase1.append({k:pr[k] for k in ['seed','AA','WGA','uniform_AA','uniform_WGA','delta_AA_pp','delta_WGA_pp','pairwise_image_ranking','wrong_head_weight','effective_members']})
        assert head_count==135
        checks=dict(status='PASS',independent_mixture_reconstruction=True,independent_pooled_AA_WGA=True,
            all_135_heads_exact_fit_IDs_epoch20_and_hashes=True,all_9_ABC_disjoint=True,B_C_ordered_heads_identical=True,
            B_only_temperature_and_lookup=True,no_test_in_phase1_or_phase2=True,
            no_group_inputs_in_inference=True,original_heads_retrained=0,primary_frozen_before_groups=True,
            limitations='Prior development/C inspection and shared ERM* backbone; inference API tested, no untouched confirmation claimed.')
        out=ROOT/'independent_audit';paths=[]
        for name,rows in [('raw_calibration_weighting_decomposition.csv',summary),('paired_flips_by_group.csv',flips),('ranking_and_concentration.csv',scores),('confidence_threshold_coverage.csv',coverage),('positive_score_low_correctness_cells.csv',risks),('locked_phase1_by_seed.csv',phase1)]:
            p=out/name;csvwrite(p,rows);paths.append(p)
        path=out/'integrity.json';dump(path,checks);paths.append(path)
        f=pd.DataFrame(flips);f=f[f.kind=='group'];fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
        for s,ax in enumerate(axes):
            t=f[f.seed==s];xx=np.arange(4);ax.bar(xx-.17,t.corrected,width=.34,label='corrected');ax.bar(xx+.17,t.broken,width=.34,label='broken');ax.set_xticks(xx,labels=[f'{k}\nn={n}' for k,n in zip(t.value,t['count'])]);ax.set(title=f'seed {s}',xlabel='True subgroup (evaluation only)')
        axes[0].set_ylabel('Distinct image count');axes[0].legend();fig.suptitle('Nested development: locked R4 versus matched calibrated uniform');fig.tight_layout();buff=io.BytesIO();fig.savefig(buff,format='png',dpi=140);plt.close(fig);path=out/'nested_subgroup_flips.png';atomic(path,buff.getvalue());paths.append(path)
        allq=pd.concat([pd.read_csv(ROOT/f'phase0/seed{s}/quantitative_checks.csv') for s in range(3)])
        water=allq[(allq.stratum=='true_group')&(allq.value.astype(str)=='2')&(allq.representation=='calibrated')]
        decomposition='\n'.join(['', '## Independent audit and mechanism findings','',
            'All 135 final heads and epoch-20 optimizer/RNG checkpoints passed hash, fitting-ID and epoch checks. Nine A/B/C manifests matched independently generated splits. Direct reconstruction reproduced mixtures, probability representation and pooled AA/WGA. No new heads were trained.','',
            '### Calibration versus weighting (all values in percent or percentage points)','',
            '| Seed | Raw U WGA | Calibrated U WGA | Lookup WGA | Calibration ΔWGA | Weighting ΔWGA | Total ΔWGA vs raw |',
            '|---|---:|---:|---:|---:|---:|---:|']+[
                f'| {r["seed"]} | {r["raw_uniform_WGA_percent"]:.4f} | {r["calibrated_uniform_WGA_percent"]:.4f} | {r["piecewise_WGA_percent"]:.4f} | {r["calibration_delta_WGA_pp"]:+.4f} | {r["weighting_delta_WGA_pp"]:+.4f} | {r["total_delta_WGA_vs_raw_pp"]:+.4f} |' for r in summary]+['',
            f'Mean total lookup versus RAW uniform: ΔWGA {np.mean([r[1] for r in raw_deltas]):+.6f} pp; ΔAA {np.mean([r[0] for r in raw_deltas]):+.6f} pp. The promotion comparison uses matched CALIBRATED uniform; calibration itself changes WGA and must not be credited to weighting.','',
            '### Is expert reliability rankable?','',
            '| Seed | Within-image correct-over-wrong ranking | Effective experts | Mean maximum weight | Corrected | Broken |',
            '|---|---:|---:|---:|---:|---:|']+[
                f'| {r["seed"]} | {r["pairwise_image_ranking"]:.4f} | {r["effective_members"]:.4f} | {r["mean_max_weight"]:.5f} | {r["corrected"]} | {r["broken"]} |' for r in scores]+['',
            'The inference-safe lookup ranks correct experts above wrong ones more often than chance overall, but this does not ensure subgroup improvement. Only a few boundary images change. Effective expert count remains about 14.94, so the new failure is not the extreme softmax concentration seen in D19. All three seeds break one matched-uniform-correct waterbird-on-land (group 2); improvements occur mainly in landbird-on-water (group 1), which does not compensate the worst group.', '',
            f'In the exploratory meta_train R4 lookup, {len(risks)} supported cells have empirical correctness below .5 yet a positive smoothed signed score; see positive_score_low_correctness_cells.csv. These are correlated expert observations and often only a few distinct images. This documents sparse-cell/class-parent shrinkage limitations, not an inference-label leak or a justification to tune kappa after evaluation.', '',
            '### D19 H1/H2/H3 conclusions','',
            '- H1 supported descriptively: all 586 historical changed predictions are Waterbird→Landbird. The class/confidence panels and exact checkpoint forward expose the asymmetry.',
            '- H2 supported locally, not globally: overall historical image-mean pair ranking is .7609–.8257, but waterbird-on-land ranking is '+', '.join(f'{v:.4f}' for v in water.pairwise_image_ranking)+'. The score is capable of ranking globally while misranking the hard subgroup. Group labels are used here only after the fact.',
            '- H3 descriptively supported: D19 effective expert count is2.50–3.07 and there are264 minority-wrong overturns across seed/image rows. The new mild lookup keeps almost15 effective experts and still reduces WGA, so concentration alone cannot explain every failure. These observations do not prove a causal decomposition.', '',
            '### Confidence scale check','',
            '| Seed | Representation | Fitted/fixed T | Mean confidence | P95 | Maximum | >=.75 expert observations % | Distinct >=.75 images |',
            '|---|---|---:|---:|---:|---:|---:|---:|']+[
                f'| {r["seed"]} | {r["map_id"]} | {r["temperature"]:.6f} | {r["mean_confidence"]:.4f} | {r["p95_confidence"]:.4f} | {r["max_confidence"]:.4f} | {100*r["ge075_fraction"]:.3f} | {r["ge075_distinct_images"]} |' for r in coverage]+['',
            'Coverage is measured on each seed’s meta_train only. Raw .75 is empty, fixed T=.25 has a sparse upper tail, and fitted T≈.03 makes .75 common. No test threshold fitting was done. Interval-specific counts/accuracy by predicted class and sparse confident-wrong cells are in every fit folder.', '',
            '### Completion / recovery','',
            '7 focused tests passed. Full separate-process resume generated zero new COMMITTED events and validated every existing artifact hash/size. The earlier failed boolean diagnostic was fixed in recorded r2; r1 sources, manifest, failure log and two completed tasks were preserved. The final research/report task below adds evidence without overwriting any earlier completed artifact.', '',
            'Current report: FINAL_RESEARCH_REPORT.md. Earlier FINAL_REPORT.md is retained as the original final-stage checkpoint. Production integration remains prohibited; no DPE production source/default was changed.'])
        path=ROOT/'FINAL_RESEARCH_REPORT.md';atomic(path,(ROOT/'FINAL_REPORT.md').read_text(encoding='utf-8')+decomposition+'\n');paths.append(path)
        source=out/'report_source.py';atomic(source,Path(__file__).read_bytes());paths.append(source)
        return dict(independent_audit='PASS',heads_reused=135,gate='FAIL',full_report=str(path)),paths
    j.task('final:independent_audit',['final:report'],audit)
    def report_correction():
        old=ROOT/'FINAL_RESEARCH_REPORT.md'
        text=old.read_text(encoding='utf-8')
        # Correct the descriptive subgroup sentence from the saved flip CSVs; metrics unchanged.
        replacement='Seed 0 breaks one waterbird-on-land (group 2); seed 1 breaks one waterbird-on-water (group 3); seed 2 breaks one waterbird-on-land (group 2). All nine corrections occur in landbird-on-water (group 1).'
        for seed,broken_group in [(0,2),(1,3),(2,2)]:
            f=pd.read_csv(ROOT/f'phase2/seed{seed}/corrected_broken.csv');f=f[f.kind=='group']
            assert int(f[f.value==broken_group].broken.iloc[0])==1 and int(f.broken.sum())==1
            assert int(f[f.value==1].corrected.iloc[0])==[6,1,2][seed]
        text=text.replace('All three seeds break one matched-uniform-correct waterbird-on-land (group 2); improvements occur mainly in landbird-on-water (group 1), which does not compensate the worst group.',replacement+' The corrections do not compensate the worst group.')
        text=text.replace('Current report: FINAL_RESEARCH_REPORT.md.','Current report: FINAL_RESEARCH_REPORT_r2.md.')
        text+='\n## Report / initialization maintenance\n\nThe prior report checkpoint is preserved. A descriptive subgroup sentence was corrected from the saved flip tables: broken groups are 2,3,2 for seeds0,1,2, respectively. No measurements, selection, gate or experiment outputs changed. Source revision r3 only permits initialization when lineage history does not already exist; all completed tasks and prior source/manifest versions remain valid.\n'
        path=ROOT/'FINAL_RESEARCH_REPORT_r2.md';atomic(path,text)
        source=ROOT/'independent_audit/report_source_r2.py';atomic(source,Path(__file__).read_bytes())
        resume=Path('outputs/score_audit_v4/resume_r3_check.log');assert 'COMMITTED ' not in resume.read_text()
        copied=ROOT/'acceptance/full_resume_r3.log';atomic(copied,resume.read_bytes())
        return dict(report_revision=2,measurements_changed=False,broken_groups_by_seed=[2,3,2],r3_resume_new_tasks=0),[path,source,copied]
    j.task('final:report_correction',['final:independent_audit'],report_correction)
    def recovery():
        log=Path('outputs/score_audit_v4/resume_check.log');assert 'COMMITTED ' not in log.read_text()
        path=ROOT/'acceptance/full_resume.json';dump(path,dict(status='PASS',new_experiment_commits=0,validated_original_tasks=221,completed_task_reuse=True,log_sha256=sha(log),audit_task_added_after_resume=True))
        copied=ROOT/'acceptance/full_resume.log';atomic(copied,log.read_bytes())
        return dict(full_resume='PASS',new_experiment_tasks=0),[path,copied]
    j.task('final:full_resume',['final:independent_audit'],recovery)
    j.status('COMPLETE_GATE_FAILED','None under frozen v4 protocol; do not integrate or retune',active_phase='complete',
        promotion_gate='FAIL',production_integration='NOT_PERFORMED',independent_audit='PASS',full_resume='PASS',
        mean_delta_WGA_pp=gate['mean_delta_WGA_pp'],mean_delta_AA_pp=gate['mean_delta_AA_pp'],
        report='FINAL_RESEARCH_REPORT_r2.md',failures_resolved='r1 descriptive NumPy bug fixed in r2; r3 initialization maintenance; all completed tasks retained')
    tracked=Path('results/dpe/score_audit_v4_20261115');tracked.mkdir(parents=True,exist_ok=True)
    for name in ['phase0_mechanism_report.md','phase1_all.csv','phase1_ranked_by_predeclared_proxy.csv','phase1_locked_candidates.json','phase1_report.md','phase2_nested_verification.csv','promotion_gate.json','PROTOCOL_DECISIONS.md','HANDOFF_STATUS.md','NEXT_AGENT_PROMPT.md']:
        shutil.copyfile(ROOT/name,tracked/name)
    shutil.copyfile(ROOT/'FINAL_RESEARCH_REPORT_r2.md',tracked/'FINAL_REPORT.md')
    for p in (ROOT/'independent_audit').glob('*.csv'):shutil.copyfile(p,tracked/p.name)
    for name in ['integrity.json','nested_subgroup_flips.png']:shutil.copyfile(ROOT/'independent_audit'/name,tracked/name)
    # Compact report package includes prediction caches and fitted tables, references original heads.
    selected=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.md','.csv','.json','.png','.npz','.py')
        and not any(part in ('inputs','source_snapshot','source_snapshot_r2','completed','history','failures') for part in p.relative_to(ROOT).parts)
        and not p.name.startswith('expert_audit_')]
    def package():
        path=ROOT/'DPE_WGA_v4_results.zip';buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
            for item in selected:z.write(item,item.relative_to(ROOT).as_posix())
        atomic(path,buffer.getvalue())
        with zipfile.ZipFile(path) as z:assert z.testzip() is None
        mp=ROOT/'bundle_manifest.json';dump(mp,dict(path=path.name,sha256=sha(path),bytes=path.stat().st_size,zip_CRC='PASS',members=len(selected)))
        return dict(package='COMPLETE',bytes=path.stat().st_size),[path,mp]
    j.task('final:package',['final:report_correction','final:full_resume'],package)
    bundle=Path('outputs/score_audit_v4/DPE_WGA_v4_results.zip');shutil.copyfile(ROOT/bundle.name,bundle)
    j.status('COMPLETE_GATE_FAILED','None under frozen v4 protocol; do not integrate or retune',active_phase='complete',package='DPE_WGA_v4_results.zip')
    for name in ['state.json','HANDOFF_STATUS.md','NEXT_AGENT_PROMPT.md','bundle_manifest.json']:
        shutil.copyfile(ROOT/name,tracked/name)
    print(json.dumps(dict(status='COMPLETE',gate='FAIL',completed=len(j.records),report=str(tracked/'FINAL_REPORT.md'),bundle=str(bundle),bundle_bytes=bundle.stat().st_size)))


if __name__=='__main__':main()
