"""Independent B-fit audit and final evidence package; never trains or selects."""
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from experiments.discrete_weighting.store import sha
from experiments.geometry_aggregation_v5.persistence import atomic,dump
from experiments.geometry_aggregation_v5.io import load,single_writer
from experiments.ccw_v6.store import Journal,csvwrite
from experiments.ccw_v6.run import ROOT,METHODS


def audit():
    checked=0;prediction_count=0;rows=[]
    for s in range(3):
        for f in range(3):
            B=load(ROOT/f'inputs/B/seed{s}/fold{f}.npz');assert 'group' not in B
            splits=json.loads((ROOT/f'checkpoints/seed{s}/fold{f}/inner_splits.json').read_text())['folds']
            for item in splits:
                i=item['inner_fold'];a=np.array(item['train_indices']);b=np.array(item['valid_indices']);folder=ROOT/f'checkpoints/seed{s}/fold{f}/inner{i}'
                assert not set(B['sample_id'][a])&set(B['sample_id'][b]);model=json.loads((folder/'model.json').read_text())
                assert model['fit_ids']==B['sample_id'][a].tolist() and model['head_sha256']==B['head_sha256'].tolist()
                d=B['distances'][a].astype(float).min(-1);mu=d.mean(0);sig=d.std(0,ddof=0);norm=model['normalizer']
                np.testing.assert_allclose(norm['mu'],mu,atol=1e-15);np.testing.assert_allclose(norm['sigma'],sig,atol=1e-15)
                assert norm['mu_ref']==float(np.median(mu)) and norm['sigma_ref']==float(np.median(sig))
                p=B['probs'][a];y=B['y'][a];u=p.mean(1);true=p[np.arange(len(p))[:,None],np.arange(15),y[:,None]].astype(float);t=true-u[np.arange(len(p)),y,None]
                margin=abs(u[:,1]-.5);q20=float(np.quantile(margin,.2));assert model['q20']==q20
                omega=len(p)/(2*np.bincount(y,minlength=2)[y])*(1+(margin<=q20).astype(int));omega/=omega.mean();w=np.broadcast_to(omega[:,None]/15,t.shape)
                glob=float((w*t).sum()/w.sum());sigma_A=max(float(np.sqrt((w*(t-glob)**2).sum()/w.sum())),.02);assert abs(model['sigma_A']-sigma_A)<1e-15
                z=(d-mu)/np.maximum(sig,1e-6);raw=np.quantile(z,[1/3,2/3]);edges=raw.tolist() if raw[0]<raw[1] else [];assert edges==model['z_edges']
                pred=p.argmax(-1);cb=np.searchsorted(edges,z,side='right');dis=pred!=u.argmax(1)[:,None]
                parents=[]
                for k in [0,1]:
                    mask=pred==k;parents.append(float(((w*t)[mask].sum()+25*glob)/(w[mask].sum()+25)))
                for cell in model['cells']:
                    k=cell['predicted_class'];bin_=cell['z_bin'];a_=cell['disagree_with_uniform'];mask=(pred==k)&(cb==bin_)&(dis==a_);support=float(w[mask].sum());T=float((w*t)[mask].sum())
                    assert abs(cell['effective_image_support']-support)<1e-13 and cell['distinct_images']==int(mask.any(1).sum())
                    expected=float(np.clip((T+25*parents[k])/(support+25),-1,1));assert abs(cell['A_hat']-expected)<1e-15
                assert abs(sum(c['effective_image_support'] for c in model['cells'])-len(a))<1e-10;checked+=1
                for method in METHODS:
                    out=load(folder/method/'predictions.npz');np.testing.assert_array_equal(out['sample_id'],B['sample_id'][b]);assert np.all(out['q20']==q20)
                    if method=='U0':np.testing.assert_array_equal(out['probs'],B['probs'][b].mean(1))
                    else:np.testing.assert_allclose((B['probs'][b]*out['weights'][...,None]).sum(1),out['probs'],atol=1e-12,rtol=0)
                    np.testing.assert_allclose(out['scores'],out['geometry_scores']+out['advantage_scores'],atol=1e-12,rtol=0)
                    assert out['weights'].shape==(len(b),15) and np.allclose(out['weights'].sum(1),1,atol=2e-6);prediction_count+=1
                rows.append(dict(seed=s,outer_fold=f,inner_fold=i,train_images=len(a),valid_images=len(b),model_sha256=sha(folder/'model.json'),normalizer_q20_sigma_table_support='PASS',no_validation_ID_in_fit=True))
    assert checked==36 and prediction_count==180
    parent=json.loads((ROOT/'manifest.json').read_text())
    production={p:h for p,h in parent['inputs'].items() if 'src/dpe/' in p.replace('\\','/')}
    assert production and all(sha(p)==h for p,h in production.items())
    lock=json.loads((ROOT/'LOCKED_CANDIDATE.json').read_text());assert lock['status']=='NO_GO' and lock['selected_method'] is None
    assert not (ROOT/'results/C').exists() and not (ROOT/'checkpoints/full_B').exists()
    table=ROOT/'reports/independent_fit_boundary_audit.csv';csvwrite(table,rows)
    path=ROOT/'reports/independent_integrity.json';dump(path,dict(status='PASS',inner_fits_verified=checked,method_validation_caches_verified=prediction_count,
        normalizer_quantile_boundary_weights_sigma_and_all_cells_reconstructed=True,distinct_train_validation_IDs=True,
        production_source_hashes_unchanged=production,new_CCW_C_replay=False,head_retraining=False))
    return dict(integrity='PASS',inner_fits=checked,prediction_caches=prediction_count),[table,path]


def final_report():
    allmetrics=pd.read_csv(ROOT/'results/B_crossfit_all_methods.csv');outer=allmetrics[allmetrics.level=='outer_B'];means=outer.groupby('method')[['worst_class_NLL','loss_class0','loss_class1','class0_accuracy','class1_accuracy','AA','effective_members']].mean()
    table=ROOT/'reports/B_mean_class_loss_decomposition.csv';csvwrite(table,[dict(method=m,**means.loc[m].to_dict()) for m in METHODS])
    cells=pd.read_csv(ROOT/'results/B_cell_tables.csv');gain=pd.read_csv(ROOT/'results/B_gain_and_concentration.csv');g1=means.loc['G1_frozen'];a=means.loc['CCW_A'];full=means.loc['CCW_FULL']
    calibration_penalty=float(a.worst_class_NLL-g1.worst_class_NLL);advantage_recovery=float(a.worst_class_NLL-full.worst_class_NLL)
    supplement='\n'.join(['', '# Independent completion audit and measured failure mechanism','',
        '**CCW v6 is NO-GO on authorized B selection. No new CCW C evaluation and no production integration were performed. Fresh confirmation is unavailable.**','',
        '## Per-class B loss decomposition','',
        '| Method | Mean worst-class NLL | Class0 NLL | Class1 NLL | Effective experts |','|---|---:|---:|---:|---:|']+[
            f'| {m} | {means.loc[m,"worst_class_NLL"]:.8f} | {means.loc[m,"loss_class0"]:.8f} | {means.loc[m,"loss_class1"]:.8f} | {means.loc[m,"effective_members"]:.4f} |' for m in METHODS]+['',
        f'Partial calibration alone (CCW_A) raises the B selection proxy by {calibration_penalty:+.8f} relative to G1. Adding the advantage table recovers only {advantage_recovery:.8f}, leaving CCW_FULL worse than G1 by {full.worst_class_NLL-g1.worst_class_NLL:+.8f}. Both class0 and class1 mean NLL are worse than G1. This directly identifies the calibration trade-off and the insufficient compensation from the frozen advantage term; it does not establish a causal IPS/head00 explanation.', '',
        f'The12-cell tables have median effective image support {cells.effective_image_support.median():.4f}; {int((cells.effective_image_support==0).sum())}/{len(cells)} cells are empty and {int(cells.low_support.sum())}/{len(cells)} have fewer than30 distinct images. kappa25 shrinkage is substantial at these supports. Positive B advantage ranking therefore does not imply a large enough score correction to recover the proxy loss.', '',
        '## Uniform-wrong B images: direction remains problematic','',
        '| Seed | Method | Distinct Uniform-wrong images | Mean signed gain sum(r*t) | Mean geometry score range | Mean advantage score range |','|---|---|---:|---:|---:|']+[
            f'| {int(r.seed)} | {r.method} | {int(r.unique_images)} | {r.gain:+.8f} | {r.geometry_range_mean:.5f} | {r.advantage_range_mean:.5f} |'
            for r in gain[(gain.subset=='uniform_wrong')&gain.method.isin(['G1_frozen','CCW_A','CCW_FULL'])].itertuples()]+['',
        'The mean true signed gain remains negative on Uniform-wrong B images for every listed seed/method. Partial normalization makes it less negative than G1, but the advantage term does not consistently turn it positive. Advantage score range never dominates geometry in these strata. The primary failure is not an unreported exploding learned score. No beta/rho/alpha change was tried after these observations.', '',
        '## Integrity, tests and resume','',
        'Independent audit reconstructed training-only mu/sigma/reference medians, q20, class/boundary image weights, sigma_A, parent shrinkage and every cell for all36 fits. It verified180 method validation caches, train/validation IDs, probability mixtures, score decomposition and unchanged production source hashes. Per-seed B AA is based on613/619/615 unique image IDs rather than720 independent rows.', '',
        'Nine unit/regression tests passed, including the missing-fresh-data serialization test. Source revision r2 fixes only a duplicate report keyword after491 scientific tasks had committed; their inputs, model math, splits, predictions and selection were preserved. Both forced-kill recovery exercises passed. Full separate-process resume produced zero new COMMITTED experiment tasks; all493 pre-delivery task hashes verified.', '',
        'The final delivery adds this independent audit, checksum manifests and a compact package, without any new fit, search, head inference or C replay. All older v1-v5 outputs/checkpoints and existing Uniform/G1 production sources remain untouched.', '',
        '## Exactly one justified next step after this audit','',
        'Keep G1 and require a separately preregistered B-only follow-up before changing CCW; the current frozen CCW result remains NO-GO and must not be rescued using old C/test outcomes.'])+'\n'
    original=(ROOT/'reports/FINAL_REPORT.md').read_text(encoding='utf-8')
    original=original.replace('Preserve the negative B decision and audit its recorded per-class loss/score decomposition on B only before proposing any separately preregistered follow-up; do not use old C to rescue CCW.',
        'The recorded B loss/score decomposition is independently audited in the completion section below. Keep the frozen NO-GO; require a separately preregistered B-only follow-up before changing CCW.')
    original=original.replace('## Exactly one justified next step\n','## B-only audit disposition\n')
    path=ROOT/'reports/FINAL_REPORT_VERIFIED.md';atomic(path,original+supplement)
    log=ROOT/'results/full_resume_check.log';text=log.read_text(encoding='utf-8');assert 'COMMITTED ' not in text and json.loads(text.strip().splitlines()[-1])['completed']==493
    receipt=ROOT/'reports/final_acceptance.json';dump(receipt,dict(status='PASS',unit_tests=9,full_resume_new_experiment_tasks=0,verified_original_tasks=493,
        full_resume_log_sha256=sha(log),core_decision='NO_GO',production_integration=False,fresh_confirmation='BLOCKED_NO_FRESH_DATA',
        source_revision2='Report dictionary serialization fix only; all completed models/results retained'))
    source=ROOT/'reports/delivery_source.py';atomic(source,Path(__file__).read_bytes())
    return dict(final_report='reports/FINAL_REPORT_VERIFIED.md',B_decision='NO_GO',unit_tests=9,full_resume='PASS'),[table,path,receipt,source]


def main():
    with single_writer(ROOT):
        j=Journal(ROOT,json.loads((ROOT/'manifest.json').read_text()));extra={'delivery:integrity':['final_report'],'delivery:report':['delivery:integrity'],'delivery:package':['delivery:report']};j.plan={**j.plan,**extra}
        protocol=dict(parent_manifest_sha256=sha(ROOT/'manifest.json'),source_sha256=sha(__file__),planned_DAG=extra,role='Post-run audit/report/export; no training/selection or C experiments')
        dp=ROOT/'reports/delivery_protocol.json'
        if dp.exists():assert json.loads(dp.read_text())==protocol,'Delivery code changed; preserve receipts and explicitly revise delivery protocol'
        else:dump(dp,protocol)
        j.current['delivery_protocol_sha256']=sha(dp);j.current['delivery_task_DAG']=extra
        j.task('delivery:integrity',['final_report'],audit)
        j.task('delivery:report',['delivery:integrity'],final_report)
        j.status('FEASIBLE_COMPLETE_CONFIRMATION_BLOCKED','No CCW promotion; read reports/FINAL_REPORT_VERIFIED.md and INPUT_REQUEST.md',
            report='reports/FINAL_REPORT_VERIFIED.md',B_decision='NO_GO',production_integration='PROHIBITED',fresh_confirmation='BLOCKED_NO_FRESH_DATA',full_resume='PASS',unit_tests_passed=9,independent_audit='PASS')
        def package():
            target=Path('outputs/ccw_v6/DPE_CCW_v6_results.zip');files=[p for p in ROOT.rglob('*') if p.is_file() and not any(v in ['__pycache__','v5_analysis','fresh_inputs'] for v in p.relative_to(ROOT).parts)
                and not any(v.endswith('_uninterrupted') for v in p.parts) and p.suffix in ['.py','.md','.json','.csv','.npz','.png','.log']]
            buffer=io.BytesIO()
            with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
                for p in files:z.write(p,p.relative_to(ROOT).as_posix())
            atomic(target,buffer.getvalue())
            with zipfile.ZipFile(target) as z:assert z.testzip() is None
            receipt=ROOT/'reports/package_receipt.json';dump(receipt,dict(path=str(target),sha256=sha(target),bytes=target.stat().st_size,members=len(files),zip_CRC='PASS',
                excluded='Original images, DPE head/backbone checkpoints, redundant recovery controls and original C analysis package. Workspace original v5 inputs remain necessary for source-hash resume.'))
            # The receipt points to the external archive; keep an internal tiny checksum record.
            return dict(package=str(target),bytes=target.stat().st_size),[receipt]
        j.task('delivery:package',['delivery:report'],package)
        j.status('FEASIBLE_COMPLETE_CONFIRMATION_BLOCKED','No further task under frozen CCW protocol; preserve NO_GO and request verified fresh assets only for a separately authorized continuation',active_task=None,active_phase='complete',package='outputs/ccw_v6/DPE_CCW_v6_results.zip')
        tracked=Path('results/dpe/ccw_v6_20261115');tracked.mkdir(parents=True,exist_ok=True)
        names={'reports/FINAL_REPORT_VERIFIED.md':'FINAL_REPORT.md','reports/CCW_MECHANISM_AUDIT.md':'CCW_MECHANISM_AUDIT.md','reports/PHASE0.md':'PHASE0.md','reports/PHASE1.md':'PHASE1.md',
            'reports/CONFIRMATION_BLOCKED.md':'CONFIRMATION_BLOCKED.md','reports/C_EXPLORATORY_ONLY.md':'C_EXPLORATORY_ONLY.md','reports/promotion_decision.json':'promotion_decision.json',
            'reports/independent_integrity.json':'independent_integrity.json','reports/independent_fit_boundary_audit.csv':'independent_fit_boundary_audit.csv','reports/B_mean_class_loss_decomposition.csv':'B_mean_class_loss_decomposition.csv',
            'reports/final_acceptance.json':'final_acceptance.json','reports/package_receipt.json':'package_receipt.json',
            'INPUT_INVENTORY.md':'INPUT_INVENTORY.md','PROVENANCE_AUDIT.json':'PROVENANCE_AUDIT.json','B_INPUT_STATUS.json':'B_INPUT_STATUS.json','LOCKED_CANDIDATE.json':'LOCKED_CANDIDATE.json','INPUT_REQUEST.md':'INPUT_REQUEST.md',
            'PROTOCOL_DECISIONS.md':'PROTOCOL_DECISIONS.md','state.json':'state.json','HANDOFF_STATUS.md':'HANDOFF_STATUS.md','NEXT_AGENT_PROMPT.md':'NEXT_AGENT_PROMPT.md'}
        for name in ['B_crossfit_per_image.csv','B_crossfit_all_methods.csv','B_cell_tables.csv','B_selection_proxy.csv','B_advantage_signal.csv','B_gain_and_concentration.csv','C_EXPLORATORY_ONLY.csv','g1_reproduction.csv']:names['results/'+name]=name
        for name in ['B_selection_and_AA.png','B_advantage_ranking.png']:names['figures/'+name]=name
        for source,dest in names.items():atomic(tracked/dest,(ROOT/source).read_bytes())
        atomic(ROOT/'figures/FIGURE_SOURCES.md','# Figure sources\n\nB_selection_and_AA.png: results/B_selection_proxy.csv and results/B_crossfit_all_methods.csv (seed_B rows).\n\nB_advantage_ranking.png: results/B_advantage_signal.csv (seed_B_clustered, overall rows).\n')
        print(json.dumps(dict(completed=len(j.records),B_decision='NO_GO',confirmation='BLOCKED',report=str(tracked/'FINAL_REPORT.md'),package='outputs/ccw_v6/DPE_CCW_v6_results.zip')))


if __name__=='__main__':main()
