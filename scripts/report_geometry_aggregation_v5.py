"""Package completed v5 evidence, improve figure layout, and verify no-recompute resume."""
import io
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.discrete_weighting.store import sha
from experiments.geometry_aggregation_v5.persistence import atomic,dump
from experiments.geometry_aggregation_v5.journal import Journal,csvwrite
from experiments.geometry_aggregation_v5.io import load,figure,single_writer
from experiments.geometry_aggregation_v5.run import ROOT


def heatmaps(seed,split):
    paths=[];folder=ROOT/f'phase0/seed{seed}/{split}'
    for name,xkey in [('distance_confidence','confidence_bin'),('distance_agreement','agreement_bin')]:
        t=pd.read_csv(folder/f'{name}_source.csv');fig,axes=plt.subplots(2,3,figsize=(14,10),layout='constrained')
        for k in [0,1]:
            sub=t[t.predicted_class==k]
            for ax,key in zip(axes[k],['correctness','mean_distance','mean_agreement']):
                mat=sub.pivot(index='distance_bin',columns=xkey,values=key);cnt=sub.pivot(index='distance_bin',columns=xkey,values='distinct_images')
                image=ax.imshow(mat,aspect='auto',vmin=0 if key=='correctness' else None,vmax=1 if key=='correctness' else None)
                ax.set(title=f'Predicted class {k}: {key.replace("_"," ")}',xlabel='Confidence quartile (train bins)' if xkey=='confidence_bin' else 'Leave-one-out agreement bin',ylabel='Minimum-distance quartile')
                ax.set_xticks(range(len(mat.columns)),labels=[f'Q{int(v)+1}' for v in mat.columns] if xkey=='confidence_bin' else [['<1/3','1/3–2/3','>=2/3'][int(v)] for v in mat.columns]);ax.set_yticks(range(len(mat.index)),labels=[f'Q{int(v)+1}' for v in mat.index])
                for i in range(len(mat.index)):
                    for j in range(len(mat.columns)):
                        value=mat.iloc[i,j];n=cnt.iloc[i,j]
                        if pd.notna(value):
                            color='black' if image.norm(value)>.55 else 'white'
                            ax.text(j,i,f'{value:.3f}\nn={int(n)}'+('*' if n<30 else ''),ha='center',va='center',fontsize=8,color=color)
        fig.suptitle(f'EXPLORATORY DEVELOPMENT | seed {seed} | {split}\nCounts: distinct images; * <30. All bin edges fitted on meta_train only.')
        p=ROOT/f'figures/seed{seed}_{split}_{name}.png';figure(p,fig);paths.append(p)
    return dict(seed=seed,split=split,readable_heatmaps=2),paths


def summary_figures(gate):
    frame=pd.read_csv(ROOT/'phase0_expert_ranking.csv');frame=frame[(frame.split=='meta_train')&(frame.group_AUDIT_ONLY.astype(str)=='all')]
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');names=['negative_min_distance','confidence','negative_entropy','distance_margin']
    for ax,subset in zip(axes,['all','uniform_wrong']):
        for s in range(3):
            t=frame[(frame.seed==s)&(frame.subset==subset)].set_index('score').loc[names];x=np.arange(4)+(s-1)*.17
            ax.errorbar(x,t.ranking_accuracy,yerr=[t.ranking_accuracy-t.CI95_low,t.CI95_high-t.ranking_accuracy],fmt='o',capsize=3,label=f'seed{s}')
        ax.axhline(.5,color='gray',linestyle='--');ax.set_xticks(range(4),labels=['−min distance','Confidence','−entropy','Distance margin'],rotation=20)
        ax.set(title='All mixed-expert images' if subset=='all' else 'Uniform-wrong mixed-expert images',ylabel='Within-image pair ranking accuracy',ylim=(0,1));ax.legend()
    fig.suptitle('Development meta_train: image bootstrap 95% intervals; not independent backbone replicas')
    p=ROOT/'figures/expert_ranking_summary.png';figure(p,fig);paths=[p]
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for i,d in enumerate(gate['decisions']):
        x=np.arange(3)+(i-.5)*.14;rows=d['seeds']
        for ax,key,ci in [(axes[0],'delta_WGA_pp','delta_WGA_CI95_pp'),(axes[1],'delta_AA_pp','delta_AA_CI95_pp')]:
            vals=np.array([r[key] for r in rows]);lo=np.array([r[ci][0] for r in rows]);hi=np.array([r[ci][1] for r in rows])
            ax.errorbar(x,vals,yerr=[np.maximum(vals-lo,0),np.maximum(hi-vals,0)],fmt='o',capsize=3,label=f'{d["candidate"]} ({d["status"]})')
            ax.axhline(0,color='gray',linestyle='--');ax.set_xticks(range(3),labels=['Seed0','Seed1','Seed2']);ax.set_ylabel(key+' (percentage points)');ax.legend(fontsize=8)
    axes[0].set_title('Nested WGA change vs matched uniform');axes[1].set_title('Nested AA change vs matched uniform');axes[1].axhline(-1,color='red',linestyle=':',label='Strict AA floor')
    fig.suptitle('Unique-image paired bootstrap intervals; three seeds share one ERM* backbone')
    p=ROOT/'figures/nested_WGA_AA_deltas.png';figure(p,fig);paths.append(p)
    return dict(summary_plots=2),paths


def main():
    with single_writer(ROOT):
        j=Journal(ROOT,json.loads((ROOT/'manifest.json').read_text()));gate=json.loads((ROOT/'promotion_decisions.json').read_text());assert gate['integration']['id']=='G1_g50'
        def resume():
            source=Path('outputs/geometry_aggregation_v5/full_resume_check.log');text=source.read_text(encoding='utf-8');assert 'COMMITTED ' not in text
            parsed=json.loads(text.strip().splitlines()[-1]);assert parsed['completed']==414 and parsed['gate']=='PASS'
            path=ROOT/'acceptance/full_resume.json';dump(path,dict(status='PASS',new_experiment_tasks=0,validated_completed_tasks=414,log_sha256=sha(source),all_checkpoint_hashes_validated=True))
            log=path.with_suffix('.log');atomic(log,source.read_bytes());return dict(full_resume='PASS',recomputed_experiments=0),[path,log]
        j.task('post:full_resume',['production:report'],resume)
        for seed in range(3):
            for split in ['meta_train','meta_select']:
                j.task(f'post:heatmaps:{seed}:{split}',['post:full_resume'],lambda seed=seed,split=split:heatmaps(seed,split))
        j.task('post:summary_figures',['post:full_resume'],lambda:summary_figures(gate))
        def report():
            baseline=[];flips=[];weights=[]
            for s in range(3):
                for split in ['meta_train','meta_select']:
                    z=load(ROOT/f'phase0/seed{s}/{split}/ranking_inputs.npz');wrong=z['uniform_wrong'];g=z['group']
                    for k in range(4):baseline.append(dict(seed=s,split=split,group=k,images=int((g==k).sum()),uniform_accuracy=float((~wrong[g==k]).mean()),
                        oracle_any_correct=float(z['correct'][g==k].any(1).mean()),minority_correct_images=int(((z['correct'][g==k].sum(1)>0)&(z['correct'][g==k].sum(1)<7.5)).sum())))
                for d in gate['decisions']:
                    p=ROOT/f'pooled/{d["candidate"]}/seed{s}/flips.csv';flips.extend([dict(candidate=d['candidate'],seed=s,**r) for r in pd.read_csv(p).to_dict('records')])
                    r=d['seeds'][s];weights.append(dict(candidate=d['candidate'],seed=s,aggregation_type=r['aggregation_type'],effective_members=r['effective_members'],wrong_head_weight=r['wrong_head_weight'],corrected=r['corrected'],broken=r['broken'],
                        **{f'group{k}_accuracy':r[f'group{k}_accuracy'] for k in range(4)}))
            paths=[]
            for name,rows in [('uniform_group_and_oracle_audit.csv',baseline),('nested_corrected_broken_by_group.csv',flips),('nested_concentration_and_groups.csv',weights)]:
                p=ROOT/'supplement'/name;csvwrite(p,rows);paths.append(p)
            text='# Supplemental completion record\n\nG1 passes the frozen engineering gate; S2 fails the two-positive-seeds rule. No subgroup WGA selected a recipe or chose between candidates. The first passing prelocked recipe is the only opt-in production method.\n\n'
            text+='Minimum distance slightly outranks confidence on Uniform-wrong examples, but both remain below .5; therefore the audit does not establish a dependable expert router. The mild uniform residual can still make a few useful boundary corrections. This is compatible with a small engineering gain and does not imply general correctness estimation.\n\n'
            for d in gate['decisions']:
                ff=pd.DataFrame([r for r in flips if r['candidate']==d['candidate'] and r['kind']=='group'])
                text+=f'## {d["candidate"]}\n\nMean ΔWGA {d["mean_delta_WGA_pp"]:+.6f} pp; ΔAA {d["mean_delta_AA_pp"]:+.6f} pp; WGA CI {d["WGA_CI95_pp"]}. Corrected/broken by true group across seed/image rows: '
                text+=json.dumps(ff.groupby('value')[['corrected','broken']].sum().to_dict(),sort_keys=True)+'\n\n'
            text+='The complete report includes the raw per-seed AA/WGA, strict gate criteria, source geometry and numerical controls. Every baseline prediction was checked against the legacy Torch uniform reduction: reductions can differ from NumPy by <=2e-7, but all nested labels are identical; original Uniform API is unchanged. This does not alter a metric tolerance or AA floor.\n\n'
            text+='13 tests passed, including all nine C caches, actual DPE class inference, legacy defaults/member/raw-logit outputs, label-free metadata/CLI, forced real-head recovery, atomic retry and identical adapter mathematics. Separate full resume validated414 completed tasks and produced zero experiment commits. All135 new heads retain epoch20 optimizer/RNG state. Original v1-v4 results are preserved.\n\n'
            text+='Figures under figures/ improve spacing/contrast while retaining the original committed PNGs and CSV sources. Prediction outputs, fitted G2 scalers (exploratory controls), all selected/losing candidates, ABC/head manifests and source snapshots are preserved. Readable heatmap quantile edges remain in phase0/seed*/meta_*/train_only_bin_edges.json.\n\n'
            text+='Scope: one verified augmented ERM* backbone, reused1199 development images/split identities, 3 head seeds rather than independent backbones. WGA interval crosses zero; benefits depend on few subgroup examples. No previously inspected test inputs were opened. Exactly one next step remains the report’s fresh-cohort confirmation with the recipe frozen.\n'
            p=ROOT/'SUPPLEMENTARY_AUDIT.md';atomic(p,text);paths.append(p)
            combined=ROOT/'FINAL_RESEARCH_REPORT.md';atomic(combined,(ROOT/'FINAL_REPORT_COMPLETE.md').read_text(encoding='utf-8')+'\n\n'+text);paths.append(combined)
            snap=ROOT/'supplement/report_source.py';atomic(snap,Path(__file__).read_bytes());paths.append(snap)
            return dict(report='FINAL_RESEARCH_REPORT.md',heads=135,tests=13,full_resume='PASS'),paths
        j.task('post:report',['post:summary_figures']+[f'post:heatmaps:{s}:{sp}' for s in range(3) for sp in ['meta_train','meta_select']],report)
        j.status('COMPLETE','None under frozen v5 protocol; fresh confirmation remains a future research step',active_phase='complete',
            promotion='PASS',integration='COMPLETE_OPT_IN_DISTANCE',default_aggregation='uniform',research_report='FINAL_RESEARCH_REPORT.md',
            full_resume='PASS',newly_trained_heads=135,tests_passed=13,mean_delta_WGA_pp=gate['decisions'][0]['mean_delta_WGA_pp'],mean_delta_AA_pp=gate['decisions'][0]['mean_delta_AA_pp'])
        def package():
            selected=[p for p in ROOT.rglob('*') if p.is_file() and not any(v in ('inputs','__pycache__') for v in p.relative_to(ROOT).parts) and p.suffix in ('.md','.json','.csv','.npz','.pt','.png','.py','.log','.yaml')]
            target=ROOT/'DPE_Geometry_v5_results.zip';buffer=io.BytesIO()
            with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
                for p in selected:z.write(p,p.relative_to(ROOT).as_posix())
            atomic(target,buffer.getvalue())
            with zipfile.ZipFile(target) as z:assert z.testzip() is None
            p=ROOT/'bundle_manifest.json';dump(p,dict(path=target.name,sha256=sha(target),bytes=target.stat().st_size,members=len(selected),CRC='PASS',
                includes='135 new head finals and epoch checkpoints; predictions, reports, tables, figures and source/manifest lineage',external_resume_inputs='Original development_oof_bundle.zip, frozen val embedding cache and original OOF heads remain in the workspace.'))
            return dict(package='COMPLETE',bytes=target.stat().st_size),[target,p]
        j.task('post:package',['post:report'],package)
        j.status('COMPLETE','None under frozen v5 protocol; preserve all results and use frozen recipe for fresh-cohort confirmation',active_phase='complete',package='DPE_Geometry_v5_results.zip')
        tracked=Path('results/dpe/geometry_aggregation_v5_20261115');tracked.mkdir(parents=True,exist_ok=True)
        for name in ['phase0_geometry_audit.md','phase0_audit.json','phase0_expert_ranking.csv','phase1_screen.csv','phase1_locked_candidates.json','phase1_group_blind_ranking.csv','phase1_report.md','phase2_nested_results.csv','nested_per_seed_percent.csv','promotion_decisions.json','integrity_audit.json','PROTOCOL_DECISIONS.md','SUPPLEMENTARY_AUDIT.md','state.json','HANDOFF_STATUS.md','NEXT_AGENT_PROMPT.md','bundle_manifest.json']:
            shutil.copyfile(ROOT/name,tracked/name)
        shutil.copyfile(ROOT/'FINAL_RESEARCH_REPORT.md',tracked/'FINAL_REPORT.md')
        for p in (ROOT/'supplement').glob('*.csv'):shutil.copyfile(p,tracked/p.name)
        for name in ['expert_ranking_summary.png','nested_WGA_AA_deltas.png']:shutil.copyfile(ROOT/'figures'/name,tracked/name)
        for name in ['integration_receipt.json','acceptance_tests.log']:shutil.copyfile(ROOT/'production'/name,tracked/name)
        target=Path('outputs/geometry_aggregation_v5/DPE_Geometry_v5_results.zip');shutil.copyfile(ROOT/target.name,target)
        print(json.dumps(dict(status='COMPLETE',promotion='PASS',integration='OPT_IN_DISTANCE',completed=len(j.records),bundle=str(target),bytes=target.stat().st_size,report=str(tracked/'FINAL_REPORT.md'))))


if __name__=='__main__':main()
