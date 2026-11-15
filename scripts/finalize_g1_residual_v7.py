"""Archive fixed v7 results, add descriptive audits and verify zero-work resume."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd
from experiments.geometry_aggregation_v5.persistence import atomic
from experiments.geometry_aggregation_v5.journal import csvwrite
from experiments.geometry_aggregation_v5.io import single_writer
from experiments.discrete_weighting.store import sha
from experiments.g1_residual_v7.store import ROOT,ROLE,COMMAND,Journal,jsonwrite
from experiments.g1_residual_v7.run import manifest
from experiments.g1_residual_v7.model import METHODS


def supplement():
    b=pd.read_csv(ROOT/'metrics/B_OOF_MECHANISM.csv');pooled=pd.read_csv(ROOT/'metrics/B_SEED_UNIQUE_IMAGE_PREDICTIONS.csv');rows=[]
    for method in METHODS:
        own=b[b.method==method]
        for subset in ['all','Uniform_wrong','Uniform_correct']:
            selected=own if subset=='all' else own[own.uniform_wrong if subset=='Uniform_wrong' else ~own.uniform_wrong]
            for cls in [-1,0,1]:
                v=selected if cls==-1 else selected[selected.y==cls]
                def mean(key):return float(v[key].mean()) if v[key].notna().any() else None
                rows.append(dict(method=method,subset=subset,true_class=cls,rows=len(v),unique_image_IDs=v.sample_id.nunique(),
                    rank_valid_rows=int(v.rank_spearman.notna().sum()),rank_spearman=mean('rank_spearman'),geometry_rank_spearman=mean('geometry_rank_spearman'),
                    pair_ranking=mean('pair_ranking'),geometry_pair_ranking=mean('geometry_pair_ranking'),gain=mean('gain'),gain_difference_G1=mean('gain_difference_G1'),
                    true_probability_shift=mean('true_probability_shift'),weight_TV=mean('weight_TV'),
                    top_head0_fraction=float((v.top_head==0).mean()) if len(v) else None,corrected_U0=int(v.corrected_U0.sum()),broken_U0=int(v.broken_U0.sum()),role=ROLE))
    p=ROOT/'metrics/B_DIRECTION_DECOMPOSITION.csv';csvwrite(p,rows)
    flags=[];bs=pd.read_csv(ROOT/'metrics/B_SEED_METRICS.csv');cs=pd.read_csv(ROOT/'metrics/MEAN_AA_WGA_COMPARISON.csv')
    for m in METHODS:
        z=bs[bs.method==m];c=cs[cs.method==m].iloc[0]
        flags.append(dict(method=m,B_AA_floor_every_seed=bool((z.delta_AA_U0_pp> -1).all()),C_AA_floor_every_seed=bool(c.AA_floor_U0_every_seed),
            B_delta_AA_U0_pp=z.delta_AA_U0_pp.tolist(),C_delta_AA_U0_pp=json.loads(c.delta_AA_U0_pp_by_seed),role=ROLE))
    fp=ROOT/'metrics/ALL_METHOD_AA_ELIGIBILITY.json';jsonwrite(fp,dict(role=ROLE,strict_rule='each seed delta_AA_U0_pp > -1',methods=flags))
    flip=pd.read_csv(ROOT/'metrics/FLIP_AUDIT.csv');hard=[]
    for m in METHODS[2:]:
        changed=flip[(flip.group==2)&(flip[m+'_prediction']!=flip.G1_frozen_prediction)]
        for r in changed.to_dict('records'):
            hard.append(dict(seed=int(r['seed']),sample_id=r['sample_id'],method=m,y=int(r['y']),group=2,
                G1_prediction=int(r['G1_frozen_prediction']),new_prediction=int(r[m+'_prediction']),
                corrected_G1=bool(r[m+'_corrected_G1']),broken_G1=bool(r[m+'_broken_G1']),role=ROLE))
    hp=ROOT/'metrics/C_ALL_GROUP2_CHANGES_VS_G1.csv';csvwrite(hp,hard)
    selected=pooled[(pooled.method=='G1_RH')&pooled.uniform_wrong];counts=selected.groupby('seed').size().to_dict();union=int(selected.sample_id.nunique())
    ci=json.loads((ROOT/'metrics/BOOTSTRAP_CI.json').read_text())['comparisons'];rci=next(r for r in ci if r['method']=='G1_R' and r['reference']=='G1_frozen' and r['seed']==-1)
    text=['# v7 delivery: measured mechanism and remaining weakness','', '**'+ROLE+'**','', '**B_DIRECTION_NO_GO — EXPLORATORY_C_WGA_EQUAL (primary RH) — INDEPENDENT_CONFIRMATION_NOT_PERFORMED**','',
        'The following is a descriptive audit of the already locked results. No candidate, fit, scale or configuration was changed.',
        '',f'B-OOF contains {union} distinct image IDs that are Uniform-wrong in at least one seed after per-seed pooling; per-seed counts are {counts}. Overlap means these counts must not be added as independent samples.',
        '', 'Scale matching did have influence: all 36 B-inner pooled/head scales and all nine full-B pooled/head scales passed the fixed numerical scale guard. RH mean B expert-weight total variation is about 0.02573, and residual/geometry centered RMS about 0.36694. The measured failure therefore persists despite a substantially stronger score residual than CCW v6.',
        '', 'B_DIRECTION_DECOMPOSITION.csv contrasts all images, Uniform-wrong and Uniform-correct by true class, with valid per-image ranking counts. It is group-free. Poor ranking/direction on Uniform-wrong images cannot be rescued by average NLL improvement on the entire set. Head/class support was not sparse under the specified <30-image diagnostic, and same-ensemble sign agreement was substantial; these facts do not establish head-aware routing succeeds on the hard subset.',
        '', 'On C, pooled R WGA is +0.250627 pp relative to G1, with paired 95% CI '+str(rci['delta_WGA_CI95_pp'])+' pp. The gain comes from one group-2 correction in seed 0. RH also makes that correction but breaks a group-2 image in seed 2, giving no mean WGA gain. All group-2 changes, including breakages, are listed below. All five methods retain group 2 as worst group for every seed.',
        '', '| Seed | Method | Image ID | Corrected G1 | Broken G1 |', '|---:|---|---|---|---|']
    for r in hard:text.append(f"| {r['seed']} | {r['method']} | {r['sample_id']} | {r['corrected_G1']} | {r['broken_G1']} |")
    text+=['', 'RHM improves C mean AA by 0.027801 pp over RH but leaves the same mean WGA. On B-OOF it has the same classification corrections/breakages as RH while suppressing score magnitude; high-margin B images had no observed correction/breakage for either. This supports neither an AA guarantee nor a replacement of the primary.',
        '', 'No primary swap to pooled R follows its C result. Both pooled and head-aware B direction differences were negative on Uniform-wrong images in all seeds. The residual permutation null frequently matches/exceeds observed RH WGA. An oracle gap remains feasibility evidence only. Exact head identities, ridge fits, B target values and label-independent feature code are all retained for further review.',
        '', 'A later study would need a separately preregistered, B-grounded response to the signed-direction failure. The current result does not justify changing lambda, ridge, shrinkage, features or guard thresholds using these C labels. Existing G1/Uniform behavior and all v1-v6 artifacts remain unchanged.','']
    q=ROOT/'DELIVERY_SUPPLEMENT.md';atomic(q,'\n'.join(text))
    return dict(status='PASS',B_distinct_Uniform_wrong_union=union,all_method_AA_flags=flags,source_sha256=sha(Path(__file__))),[p,fp,hp,q]


def main():
    state=json.loads((ROOT/'state.json').read_text());assert state['status']=='DONE_EXPLORATORY'
    records=[json.loads(p.read_text()) for p in (ROOT/'completed').glob('*.json')];artifacts={p:v for r in records for p,v in r['artifacts'].items()};stamps={p:(ROOT/p).stat().st_mtime_ns for p in artifacts}
    env=os.environ|dict(PYTHONUTF8='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    outputs=[]
    for flags in [['--phase','all','--resume'],['--resume','--verify']]:
        result=subprocess.run([sys.executable,'-m','experiments.g1_residual_v7.run']+flags,check=True,env=env,capture_output=True,text=True,encoding='utf-8')
        assert 'COMMITTED ' not in result.stdout
        value=json.loads(result.stdout.strip().splitlines()[-1]);assert value['new_scientific_tasks']==0 and value['completed']==len(records)
        outputs.append(dict(command=flags,stdout=result.stdout,stderr=result.stderr))
    assert all((ROOT/p).stat().st_mtime_ns==t and sha(ROOT/p)==artifacts[p]['sha256'] for p,t in stamps.items())
    with single_writer(ROOT):
        j=Journal(ROOT,manifest())
        def resume_check():
            p=ROOT/'RESUME_VERIFICATION.json';jsonwrite(p,dict(role=ROLE,status='PASS',new_scientific_tasks=0,completed_before_delivery=len(records),
                artifacts_checked=len(stamps),all_completed_mtimes_and_hashes_unchanged=True,subprocesses=outputs,source_sha256=sha(Path(__file__))))
            q=ROOT/'RESUME_VERIFICATION.log';atomic(q,ROLE+'\n'+'\n'.join(r['stdout']+r['stderr'] for r in outputs));return dict(status='PASS',new_scientific_tasks=0),[p,q]
        j.task('delivery_resume_verification',['complete'],resume_check)
        j.task('delivery_mechanism_supplement',['delivery_resume_verification'],supplement)
        def package():
            content=ROOT/'CONTENTS.md';atomic(content,'\n'.join(['# G1 residual v7 results','',ROLE,'',
                'Read EXPLORATORY_C_REPORT.md and DELIVERY_SUPPLEMENT.md. Primary RH is B_DIRECTION_NO_GO and EXPLORATORY_C_WGA_EQUAL. No production promotion.',
                'All five methods are fixed. R improves observed C WGA slightly but is not selected from C. Original v6 NO_GO is preserved.',
                'metrics/: mandatory mean/per-seed/group/flip/mechanism tables, 2000 paired bootstraps and 3000 residual/random-weight permutations, all batches and RNG states.',
                'WEIGHT_AND_SCORE_DECOMPOSITION.npz: probs [5,3,1199,2], weights/r/Ahat/geom_scores/residual_scores [5,3,1199,15], member probabilities/distances, sample IDs, fold mapping, exact head/model hashes; no C labels/groups.',
                'C_EVALUATION_LABELS.npz: separate C y/group/sample-ID evaluator output. CACHE_SCHEMA.json explains axes and head order.',
                'checkpoints/: all 180 tiny ridge/head fits (45 final plus 135 secondary), nested scale OOF predictions, B validation inference and nine full-B models/C inference caches.',
                'acceptance/: atomic/lock/corruption tests and real forced secondary-fit/C-inference kill-restart controls with byte/hash equality.',
                'manifest.json, config.lock.json, EXPERIMENT_LOCK.json, completed/, CHECKSUMS.sha256: immutable source/input/config/split guards and authoritative receipts.',
                'source_snapshot/: exact registered source and unit tests. Runtime Python/numpy versions and source commit are in the manifest.',
                'RESUME_VERIFICATION.json: zero new scientific tasks and unchanged hashes/mtimes in separate-process full resume and --resume --verify.',
                'Original v5/v6 B/C inputs and prototype/backbone checkpoints are external dependencies and are not duplicated. Original images are excluded.',
                'Resume is supported in the original workspace containing those guarded inputs. A fresh extraction must restore the same input files/paths before executing.',
                'The archive is a snapshot taken before its own packaging receipt; all scientific tasks are complete. Current workspace state/handoffs remain authoritative.',
                '', '```powershell',COMMAND,'.venv/Scripts/python.exe -m experiments.g1_residual_v7.run --verify','```','']))
            files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts
                and p.suffix!='.tmp' and p.name not in ['writer.lock','DPE_G1_Residual_v7_Results.zip','PACKAGE_RECEIPT.json'] and 'failures' not in p.parts]
            buf=io.BytesIO()
            with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as archive:
                for p in sorted(files):archive.write(p,p.relative_to(ROOT).as_posix())
            dest=ROOT/'DPE_G1_Residual_v7_Results.zip';atomic(dest,buf.getvalue())
            with zipfile.ZipFile(dest) as archive:assert archive.testzip() is None
            receipt=ROOT/'PACKAGE_RECEIPT.json';jsonwrite(receipt,dict(role=ROLE,bytes=dest.stat().st_size,sha256=sha(dest),files=len(files),source_sha256=sha(Path(__file__))))
            return dict(status='PASS',files=len(files),bytes=dest.stat().st_size),[content,dest,receipt]
        j.task('delivery_package',['delivery_mechanism_supplement'],package)
        j.status('DONE_EXPLORATORY','All existing-data research and delivery complete; read report/supplement; no tuning or promotion.',active_task=None,active_phase='complete',
            report=str(ROOT/'EXPLORATORY_C_REPORT.md'),supplement=str(ROOT/'DELIVERY_SUPPLEMENT.md'),package=str(ROOT/'DPE_G1_Residual_v7_Results.zip'))
        print(json.dumps(dict(status='PASS',completed=len(j.records),B_verdict='B_DIRECTION_NO_GO',C_verdict='EXPLORATORY_C_WGA_EQUAL',independent_confirmation='INDEPENDENT_CONFIRMATION_NOT_PERFORMED')))


if __name__=='__main__':main()
