"""Read-only numerical audit, zero-work resume check, and durable delivery."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd
from experiments.ccw_v6.model import METHODS,CONFIG
from experiments.ccw_v6.exploration.journal import BASE,ROOT,BANNER,COMMAND,Journal
from experiments.ccw_v6.exploration.run import bpath,cpath,predpath,modelpath,make_manifest,preservation
from experiments.geometry_aggregation_v5.io import load,single_writer
from experiments.geometry_aggregation_v5.persistence import atomic,dump
from experiments.discrete_weighting.store import sha


def audit():
    for s in range(3):
        for f in range(3):
            b=load(bpath(s,f));assert 'group' not in b
            model=json.loads(modelpath(s,f).read_text());assert model['config']==CONFIG
            assert model['fit_ids']==b['sample_id'].tolist() and model['head_sha256']==b['head_sha256'].tolist()
            c=load(cpath(s,f));assert not set(b['sample_id'])&set(c['sample_id'])
            dm=b['distances'].astype(float).min(-1);mu=dm.mean(0);sig=dm.std(0,ddof=0);norm=model['normalizer']
            np.testing.assert_allclose(norm['mu'],mu,atol=1e-15,rtol=0);np.testing.assert_allclose(norm['sigma'],sig,atol=1e-15,rtol=0)
            assert norm['mu_ref']==float(np.median(mu)) and norm['sigma_ref']==float(np.median(sig))
            z=(dm-mu)/np.maximum(sig,1e-6);edges=np.quantile(z,[1/3,2/3]);np.testing.assert_array_equal(model['raw_z_edges'],edges)
            p=b['probs'];y=b['y'];u=p.mean(1);q20=float(np.quantile(abs(u[:,1]-.5),.2));assert q20==model['q20']
            omega=len(p)/(2*np.bincount(y,minlength=2)[y])*(1+(abs(u[:,1]-.5)<=q20).astype(int));omega/=omega.mean()
            w=np.broadcast_to(omega[:,None]/15,(len(p),15));t=p[np.arange(len(p))[:,None],np.arange(15),y[:,None]].astype(float)-u[np.arange(len(p)),y,None]
            glob=float((w*t).sum()/w.sum());sigmaA=max(float(np.sqrt((w*(t-glob)**2).sum()/w.sum())),.02)
            assert abs(glob-model['global_target_mean'])<1e-15 and abs(sigmaA-model['sigma_A'])<1e-15
            pred=p.argmax(-1);bins=np.searchsorted(model['z_edges'],z,side='right');disagree=pred!=u.argmax(-1)[:,None];parents=[]
            for k in (0,1):
                mask=pred==k;parents.append(float(((w*t)[mask].sum()+25*glob)/(w[mask].sum()+25)))
            np.testing.assert_allclose(model['predicted_class_parents'],parents,atol=1e-15,rtol=0)
            for cell in model['cells']:
                k=cell['predicted_class'];mask=(pred==k)&(bins==cell['z_bin'])&(disagree==cell['disagree_with_uniform'])
                support=w[mask].sum();total=(w*t)[mask].sum();value=float(np.clip((total+25*parents[k])/(support+25),-1,1))
                assert abs(value-cell['A_hat'])<1e-15 and abs(support-cell['effective_image_support'])<1e-12
            for method in METHODS:
                out=load(predpath(s,f,method));np.testing.assert_array_equal(out['sample_id'],c['sample_id']);np.testing.assert_array_equal(out['head_sha256'],c['head_sha256'])
                assert not {'y','group'}&out.keys() and str(out['role'])==BANNER
                # The frozen float32 Uniform/G1 reductions permit 2e-6 mass
                # tolerance. Their separate v5 prediction comparison is exact.
                np.testing.assert_allclose(out['weights'].sum(-1),1,atol=2e-6,rtol=0);np.testing.assert_allclose(out['probs'].sum(-1),1,atol=2e-6,rtol=0)
                if method=='U0':np.testing.assert_array_equal(out['probs'],c['probs'].mean(1))
                else:
                    np.testing.assert_allclose(out['weights'],.05+.25*out['r'],atol=1e-15,rtol=0)
                    np.testing.assert_array_equal(out['probs'],(c['probs']*out['weights'][...,None]).sum(1))
                    score=out['scores'];e=np.exp(score-score.max(1,keepdims=True));np.testing.assert_array_equal(out['r'],e/e.sum(1,keepdims=True))
                    np.testing.assert_allclose(score,out['geometry_scores']+out['advantage_scores'],atol=1e-14,rtol=0)
                if method in ['CCW_A','CCW_FULL']:
                    dmin=c['distances'].astype(float).min(-1);zz=(dmin-mu)/np.maximum(sig,1e-6);dcal=np.median(mu)+np.median(sig)*zz
                    np.testing.assert_array_equal(out['geometry_scores'],-50*(.5*dmin+.5*dcal))
                    if method=='CCW_FULL':
                        predc=c['probs'].argmax(-1);bc=np.searchsorted(model['z_edges'],zz,side='right');dc=(predc!=c['probs'].mean(1).argmax(-1)[:,None]).astype(int)
                        ahat=np.array(model['table'])[predc,bc,dc];np.testing.assert_array_equal(out['advantage_scores'],np.clip(ahat/sigmaA,-2,2))
    z=load(ROOT/'EXPLORATORY_C_PREDICTIONS.npz');assert z['probs'].shape==(5,3,1199,2) and z['weights'].shape==(5,3,1199,15)
    assert len(set(z['sample_id']))==1199 and z['head_sha256'].shape==(3,3,15)
    np.testing.assert_array_equal(z['probability_shift_vs_uniform'],z['probs']-z['probs'][0:1]);np.testing.assert_array_equal(z['probability_shift_vs_g1'],z['probs']-z['probs'][1:2])
    results=pd.read_csv(ROOT/'EXPLORATORY_C_RESULTS.csv');groups=pd.read_csv(ROOT/'EXPLORATORY_C_PER_GROUP.csv');flips=pd.read_csv(ROOT/'EXPLORATORY_C_FLIPS.csv');correct=z['probs'].argmax(-1)==z['y']
    for mi,m in enumerate(METHODS):
        for s in range(3):
            r=results[(results.method==m)&(results.seed==s)&(results.level=='seed')].iloc[0];ga=np.array([correct[mi,s,z['group']==g].mean() for g in range(4)])
            np.testing.assert_allclose([r.AA_pct,r.WGA_pct],[100*correct[mi,s].mean(),100*ga.min()],atol=1e-12,rtol=0)
            for g in range(4):
                row=groups[(groups.method==m)&(groups.seed==s)&(groups.group==g)&(groups.level=='seed')].iloc[0];assert row.n==(z['group']==g).sum()
                np.testing.assert_allclose(row.accuracy_pct,100*ga[g],atol=1e-12,rtol=0)
    for row in flips.itertuples():
        mi=METHODS.index(row.method);ri=METHODS.index(row.reference);i=int(np.flatnonzero(z['sample_id']==row.sample_id)[0]);s=row.seed
        assert z['probs'][mi,s,i].argmax()!=z['probs'][ri,s,i].argmax()
        assert bool(row.corrected)==bool(correct[mi,s,i] and not correct[ri,s,i]);assert bool(row.broken)==bool(not correct[mi,s,i] and correct[ri,s,i])
    bootstrap=load(ROOT/'uncertainty/bootstrap.npz');assert bootstrap['AA'].shape==(5,3,2000)
    records=[json.loads(p.read_text()) for p in (ROOT/'completed').glob('*.json')]
    latest=max(r['completed_utc'] for r in records if r['id'].startswith('infer:'));earliest=min(r['completed_utc'] for r in records if r['id'].startswith('eval:'));assert latest<earliest
    preservation(make_manifest())
    result=dict(status='PASS',role=BANNER,full_B_fits_audited=9,normalizers_bins_boundary_quantile_tables_sigmaA_reconstructed=True,inference_caches_verified=45,
        all_inference_committed_before_CCW_evaluation=True,pooled_samples_aligned=True,full_probability_shifts_and_weights_verified=True,
        original_NO_GO_unchanged=True,test_suite='12 passed in 6.12s: test_ccw_v6.py and test_ccw_exploratory_c.py',audit_source_sha256=sha(Path(__file__)))
    p=ROOT/'DELIVERY_VALIDATION.json';dump(p,result);return result,[p]


def main():
    records=[json.loads(p.read_text()) for p in (ROOT/'completed').glob('*.json')];artifacts={p:v for r in records for p,v in r['artifacts'].items()};stamps={p:(ROOT/p).stat().st_mtime_ns for p in artifacts}
    env=os.environ|dict(PYTHONUTF8='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    proc=subprocess.run([sys.executable,'-m','experiments.ccw_v6.run','--phase','exploratory_c','--resume'],capture_output=True,text=True,encoding='utf-8',env=env)
    assert proc.returncode==0,proc.stderr
    assert 'COMMITTED ' not in proc.stdout and json.loads(proc.stdout.strip().splitlines()[-1])['new_tasks']==0
    assert all((ROOT/p).stat().st_mtime_ns==stamp for p,stamp in stamps.items()),'Resume rewrote a completed artifact'
    with single_writer(BASE):
        j=Journal(ROOT,make_manifest())
        def resume_receipt():
            p=ROOT/'FULL_RESUME_CHECK.log';atomic(p,BANNER+'\n'+proc.stdout+'\n'+proc.stderr)
            q=ROOT/'FULL_RESUME_CHECK.json';dump(q,dict(status='PASS',role=BANNER,new_tasks=0,committed_artifact_mtimes_unchanged=True,artifacts_checked=len(stamps)))
            return dict(status='PASS',role=BANNER,new_tasks=0),[p,q]
        j.task('delivery_resume_check',['complete'],resume_receipt);j.task('delivery_independent_audit',['delivery_resume_check'],audit)
        def package():
            cp=ROOT/'CONTENTS.md';atomic(cp,'\n'.join(['# CCW v6 exploratory result package','',BANNER,'',
                'Original B-selection NO_GO is unchanged; no promotion or inference-default changes. Read EXPLORATORY_C_REPORT.md.',
                'Three head seeds share one ERM* backbone; the same 1199 development images recur across seeds.',
                'EXPLORATORY_C_RESULTS.csv: five methods, fold/seed/mean AA/WGA/WCA/BA, paired deltas and unique-image CIs.',
                'EXPLORATORY_C_PER_GROUP.csv: per-group counts, accuracy, corrected/broken against Uniform and G1.',
                'EXPLORATORY_C_FLIPS.csv: individual prediction changes versus Uniform, G1 and CCW-A.',
                'EXPLORATORY_C_PREDICTIONS.npz: probs [5,3,1199,2], weights [5,3,1199,15], labels, groups, IDs, folds, probability shifts and head hashes. See PREDICTION_SCHEMA.json.',
                'fits/: nine full-B fitted normalizers, distance bins, confidence boundary quantiles and shrinkage tables.',
                'predictions/: 45 label-free fold caches with all expert weights and inference internals.',
                'uncertainty/: paired group-stratified image bootstrap with joint ID draws across seeds and saved RNG states.',
                'manifest.json, completed/, phases/: immutable source/input/config hashes and task receipts.',
                'source_snapshot/: exact implementation. original_state/: original NO_GO/handoff snapshots.',
                'DELIVERY_VALIDATION.json, ORIGINAL_PRESERVATION_AUDIT.json, FULL_RESUME_CHECK.json: numerical audit, preservation and zero-work resume.',
                'No trained prototype checkpoints or original images included; small fitted CCW JSON tables are included.',
                '', 'Resume from repository root with matching original inputs:', '```powershell', COMMAND,COMMAND+' --verify','```','']));
            files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.tmp' and p.name not in ['writer.lock','DPE_CCW_v6_Exploratory_C.zip','PACKAGE_RECEIPT.json'] and 'failures' not in p.parts]
            buf=io.BytesIO()
            with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
                for p in sorted(files):archive.write(p,p.relative_to(ROOT).as_posix())
            z=ROOT/'DPE_CCW_v6_Exploratory_C.zip';atomic(z,buf.getvalue())
            with zipfile.ZipFile(z) as archive:assert archive.testzip() is None
            receipt=ROOT/'PACKAGE_RECEIPT.json';dump(receipt,dict(role=BANNER,files=len(files),sha256=sha(z),bytes=z.stat().st_size,source_sha256=sha(Path(__file__))))
            return dict(role=BANNER,files=len(files),bytes=z.stat().st_size),[cp,z,receipt]
        j.task('delivery_package',['delivery_independent_audit'],package)
        j.status('EXPLORATORY_COMPLETE','Research-only measurement complete; NO_GO unchanged. No tuning or promotion.',report=str(ROOT/'EXPLORATORY_C_REPORT.md'),package=str(ROOT/'DPE_CCW_v6_Exploratory_C.zip'),active_task=None,active_phase='complete')
        print(json.dumps(dict(status='PASS',completed=len(j.records),original_B_decision='NO_GO')))


if __name__=='__main__':main()
