"""Refresh checkpoint search and write a cost-limited Experiment 2 interim report.

Never trains, terminates jobs, or performs model inference. Original reports stay intact.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.counterfactual.common import read_csv,write_csv,write_json,sha256
from scripts.audit_failure_checkpoints import audit


def priority_plan(config,manifest):
    historical=manifest['training']
    one_hours=historical['two_additional_poter_seeds_serial_hours']/2
    return dict(status='awaiting_user_confirmation_and_cloud_runtime',
        current_finished_poter_seeds=[1],priority_poter_seeds=[1,config['next_training_seed']],
        additional_poter_seeds_to_train=[config['next_training_seed']],
        baseline_seeds_to_reuse={'ERM_enhanced':[0],'DPE_legacy':[0]},
        deferred=config['deferred'],training_started=False,training_authorized=False,
        fixed_poter_recipe=manifest['fixed_training_settings']['POTER'],precision='FP32',workers=0,
        old_plan_poter_training_hours=2*one_hours,new_plan_poter_training_hours=one_hours,
        savings_poter_gpu_hours=one_hours,relative_poter_training_reduction=.5,
        estimate_scope='Historical comparable T4 throughput; excludes setup, interruptions and final inference. Baseline training also deferred, cost not measured.',
        new_inference_needed=2100,reusable_prediction_rows=6300,priority_target_prediction_rows=8400,
        current_independent_seed_validation='incomplete; one finished POTER seed',
        after_second_seed='preliminary two-seed stability check only, not complete original three-seed validation',
        no_test_selection=True,training_policy=config['training_policy'])


def report(out,config,manifest,plan):
    existing=ROOT/config['existing_verification_root']
    profile=json.loads((out/'data_pipeline_profile.json').read_text())
    snapshot=json.loads((out/'process_snapshot.json').read_text(encoding='utf-8-sig'))
    snapshot=snapshot if isinstance(snapshot,list) else [snapshot]
    training=[r for r in snapshot if any(s in (r.get('CommandLine') or '') for s in
        ['kaggle_waterbirds.py','train_erm.py','profile_poter_training.py','run_dpe_ermstar.py','run_expt.py'])]
    status=dict(checked_utc=datetime.fromtimestamp((out/'process_snapshot.json').stat().st_mtime,timezone.utc).isoformat(),
        report_updated_utc=datetime.now(timezone.utc).isoformat(),
        local_training_processes=training,local_training_observed=bool(training),
        remote_training_status=config['remote_job_status'],processes_terminated=0,
        checkpoint_files_modified=0,new_training_started=False,new_inference_executed=False,
        existing_apps_preserved=True,completed_poter_seeds=[1],multi_seed_validation_complete=False)
    write_json(out/'training_status.json',status)
    metrics=read_csv(existing/'exp2_seed_metrics.csv')
    write_csv(out/'completed_seed_metrics.csv',metrics)
    headers=['Model','Seed','Sigma','AA (%)','WGA (%)','Landbird (%)','Waterbird (%)','Asymmetry (pp)']
    table=['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']
    for r in metrics:
        if r['control']!='natural':continue
        values=[r['model_name'],r['model_seed'],r['sigma']]+[f'{100*float(r[k]):.2f}' for k in
            ['semantic_accuracy','wga','landbird_accuracy','waterbird_accuracy','class_asymmetry']]
        table.append('| '+' | '.join(values)+' |')
    p0,p2=profile['loader'];cache=profile['compressed_byte_cache']
    t=manifest['training'];resume=json.loads((out/'resume_audit.json').read_text())
    feature=json.loads((out/'feature_cache_audit.json').read_text())
    text=f'''# Experiment 2 — interim report: two POTER seeds + existing baselines

Updated {status['checked_utc']}. This replaces execution priority, not the archived scientific results in `outputs/failure_verification/`.

## Status and cost decision

**Independent-seed validation remains INCOMPLETE. Only POTER seed1 has finished.**
Local process inspection found {len(training)} training jobs; Lab/Explorer are running and were preserved. No job was stopped, no training started, and no checkpoint was overwritten. Cloud status is **unverified**: no runtime/job ID was supplied. Local status is not evidence that a remote job is absent.

The retained seed1 checkpoint has completed epoch {resume['last_completed_epoch']} (zero-based; 300 total); the selected checkpoint is epoch {resume['selected_epoch']} by validation WGA. Its `last_training.pt` contains model, optimizer, RNG and loss state. It is a resume state for **seed1 only**, not initialization for independent seed2. Hashes/config/source fingerprints are saved in `resume_audit.json`. Older seed2/3 logs stop at epoch0 with no saved weights; their cause of stopping is unknown.

**New priority: obtain exactly one additional independently trained POTER seed (seed2), then compare seeds1/2 against existing ERM_enhanced seed0 and DPE_legacy seed0.** If a compatible in-progress seed2 is discovered, restore its full state and continue it; otherwise begin seed2 from the original ImageNet initialization, never from fine-tuned seed1. Defer the third POTER seed (seed0) and additional ERM/DPE training.

Historical 300-epoch run: mean {t['historical_poter_mean_epoch_seconds']:.2f}s/epoch, peak {t['historical_poter_peak_vram_gib']:.2f} GiB. New expected POTER training cost **{plan['new_plan_poter_training_hours']:.2f} GPU-hours** on comparable T4 throughput, versus {plan['old_plan_poter_training_hours']:.2f} for two additional seeds: about **50% less POTER training**. This is extrapolation from the saved run, not a new cloud benchmark or a price quote. Add dataset staging, checkpoint I/O, interruptions and 2,100-image evaluation. Baseline training cost is deferred and was not benchmarked. For an existing matching run at completed epoch e, remaining estimate is `(299-e)*108.65/3600` hours plus overhead, never reset to epoch0.

## Checkpoint search and preservation

Fresh read-only search hashed {manifest['additional_candidates']} candidates under the configured repository, sibling checkpoint/output roots, D:/POTER_logs, D:/DPE_Research and explicit Downloads candidates. No additional verified compatible independently trained POTER checkpoint was found. See `checkpoint_candidates.csv` for every candidate/reason. This is a bounded local search, not an authenticated search of all cloud storage. Metadata paths under `/kaggle/working` are not downloaded checkpoints.

- POTER seed1 best/last snapshots belong to one training run; duplicates do not increase n.
- The batch16/smoke seed1 checkpoints change recipe and/or repeat seed1; CMNIST checkpoints use another dataset.
- Standard-ERM seed3 and its partitioned DPE heads differ from the enhanced-ERM/full-validation legacy recipe; they remain excluded.
- Reuse current enhanced-ERM seed0 and legacy DPE seed0. Legacy DPE historical head/backbone hash binding remains absent; results retain that exploratory limitation.
- **Additional missing asset:** local OT cache `{feature['path']}` has SHA `{feature['sha256']}`, whereas seed1 records `{feature['expected_sha256']}`. Byte hashes differ; tensor equivalence to the historical cache is unverified. Recover the original cloud cache before primary seed2 training. Do not silently replace it or infer equivalence from matching filenames/shapes. Local dataset metadata/splits are complete; dataset availability does not resolve this cache provenance gap.
- Resume uses the original runner's atomic epoch checkpoint, exact config/environment/source comparison and optimizer/RNG/loss restoration. Copy a durable full output archive before a cloud session ends. Never interrupt mid-epoch to save compute; incomplete-epoch work is not committed. No active job should be stopped without verifying a valid durable checkpoint first. Keep output directories separate; a full-run checkpoint is not an independent seed.

## Measured bottleneck evidence and optimization decisions

The original runner computes fixed OT weights once, then trains with weighted replacement sampling, full-backbone SGD, FP32, batch128 and workers0. Each epoch also evaluates validation **and full test** and saves checkpoints. Test results do not affect selection; only validation WGA does. Training/validation/test phase breakdown was not recorded historically, so the dominant end-to-end bottleneck is **not yet established**.

CPU-only measurements used 256 predeclared **train** images and the actual POTER augmentation; no model/backward/optimizer was executed:

| Probe | Measured time | Meaning |
|---|---:|---|
| workers0, 2 batches | {p0['wall_seconds']:.3f}s | Local input-only cost, no GPU overlap |
| workers2, 2 batches | {p2['wall_seconds']:.3f}s | Includes Windows spawn/startup; not steady-state cloud throughput |
| 64 compressed file reads | {cache['file_read_seconds']:.4f}s | Warm/local file I/O sample |
| 64 file decode+RGB | {cache['file_decode_seconds']:.4f}s | JPEG decode cost |
| 64 in-memory-byte decode+RGB | {cache['memory_decode_seconds']:.4f}s | Exact decoded RGB equality verified |

The short workers2 probe is slower here and does not justify a cloud speedup claim. No production worker change is enabled. A prospective authorized run can use `scripts/profile_poter_training.py` to time original train/validation/test phases and host DataLoader wait without changing sample order, augmentation, optimizer or loss. It adds boundary synchronization overhead and is not a GPU-kernel profile; phase metrics remain distinct from the original epoch telemetry. It requires an explicit training confirmation. This GPU-side profiling has **not** run.

| Candidate optimization | Decision |
|---|---|
| Existing checkpoint and prediction cache | Adopt: no rerun of the 6,300 valid prediction rows. |
| Fixed OT feature cache | Preferred once recovered: reuse original SHA-bound cache; local similarly named cache does not match. No unvalidated fallback. |
| Stage unchanged compressed dataset/cache to cloud local disk | Adopt after byte-hash verification; preserve exact transforms. Staging is outside training RNG. |
| Cache decoded RGB before stochastic augmentation | Potentially method-preserving if exact pixels, ordering and RNG are verified; not enabled or speed-validated. |
| Cache augmented training tensors / freeze image features | Reject: changes stochastic augmentation or full-backbone training. |
| Extra workers/persistent workers/prefetch | Defer: same transform distribution can be retained, but RNG streams and exact resume behavior change; benchmark and prove parity before claiming recipe match. |
| Pin memory | Already enabled in original runner; not a new optimization. |
| AMP | Defer for primary seed2: autocast/gradient scaling changes numerical execution and may change optimization trajectory or validation-selected epoch. No AMP speed/memory or equivalence result measured; do not call it an FP32 replication. |
| Reduce epochs/batch, change architecture/POTER, early stop by test | Reject. Keep 300 epochs, batch128, original algorithm/selection. |
| Remove per-epoch full-test evaluation | Could reduce evaluation cost, but changes runner behavior/RNG consumption; not enabled in the exact-recipe job. |

PyTorch documents [AMP's autocast and gradient scaling](https://docs.pytorch.org/docs/stable/notes/amp_examples.html), [DataLoader workers/pinning](https://docs.pytorch.org/docs/stable/data.html), and [reproducibility/worker seeding](https://docs.pytorch.org/docs/stable/notes/randomness.html). Method-preservation judgments above are our inference from those behaviors and this runner's source; no speedup is assumed.

## Completed comparisons

All numbers below are **completed single-checkpoint comparisons**, reused from verified real predictions. They are not multi-seed means.

{chr(10).join(table)}

POTER seed1, blurred Landbird sigma2: gray86% versus natural65%, interference **−21 pp [−30, −12]**, amplification relative to clean **−15 pp [−25, −5]**. Land→Water changes P(Waterbird) by **−8.77 pp [−10.72, −6.76]**; this does not support a simple water-background→Waterbird narrative. These CIs resample foregrounds, not training seeds.

## Pending comparisons and stopping rule

6,300 cached rows are complete (2,100 per available method). One additional POTER checkpoint requires only **2,100 new predictions**, giving **8,400** total in this cost-limited comparison. Use the same 100 foregrounds, four backgrounds, gray controls and original controls, sigmas0/.5/1/2, masks/geometry/preprocessing and pixel/tensor hashes. Exclude sigma4 from primary analysis. New inference is pending, not executed.

After seed2 reaches the fixed full 300-epoch target, select its strictly best validation-WGA checkpoint, export/hash it, then compute each seed's WGA/class accuracy/asymmetry/interference, paired foreground CIs and POTER two-seed mean±sample SD (ddof1). Report both seeds even if effects disagree. With n=2 this is **preliminary stability evidence**, not full three-seed validation or a reliable estimate of the seed distribution. Current n=1 has undefined training-seed SD; never substitute foreground CI or inference randomness.

The third POTER seed and additional baseline training stay deferred until a separate explicit decision. Opposite signs, materially different magnitudes or an inconclusive two-seed result warrant discussing more evidence, not silently starting extra jobs. Baselines remain one checkpoint each and cannot establish baseline seed variability. No test-driven hyperparameter/epoch/seed selection. The repeatedly inspected test cohort is diagnostic; future method development needs a separate development set and untouched evaluation.

## Deliverables and execution

`notebooks/experiment2_poter_cost_optimized.ipynb` is an offline-generated Kaggle notebook for **one additional POTER seed only**, with historical Waterbirds training source/config/cache hashes, environment checks, optional phase timing, resume and backup. One unrelated CelebA dataset source hash differs; the missing historical bytes and included hash are explicitly recorded, rather than fabricated. Waterbirds/CUB algorithm sources match the historical fingerprints. Default `CONFIRM_TRAINING=False` skips training; no ERM/DPE or third-seed training cell is included. Missing exact OT cache blocks primary launch; the local mismatched cache is rejected. No extra 2-epoch probe training is performed. Restore a trusted backup before rerunning; do not overwrite an existing different run. No cloud job was submitted here.

The notebook trains/exports a checkpoint. Exact counterfactual evaluation still requires importing it into this repository with the locked evaluation assets; no cloud evaluation or seed2 predictions are claimed by the notebook. `execution_plan.json` records cost priorities. `training_status.json`, `resume_audit.json`, `data_pipeline_profile.json`, checkpoint manifests and hashes record the evidence. Old results/checkpoints remain unchanged.

**Interim decision: independent-seed verification incomplete.** Finish or obtain one compatible POTER seed, then reassess; do not claim method-level reproducibility from seed1 alone.
'''
    (out/'interim_report.md').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','report'],default='all')
    args=parser.parse_args()
    config=json.loads((ROOT/'configs/experiment2_interim.json').read_text())
    assert not config['allow_training']
    out=ROOT/config['output_root'];out.mkdir(exist_ok=True,parents=True)
    if args.stage=='all':
        original=json.loads((ROOT/'configs/failure_verification.json').read_text())
        manifest=audit(original,out)
        manifest['original_target_checkpoints']=manifest.pop('target_checkpoints')
        manifest['current_priority_checkpoint_count']=4
        manifest['current_priority']=config['priority']
        manifest['deferred']=config['deferred']
        write_json(out/'checkpoint_manifest.json',manifest)
        import torch
        accepted=next(r for r in manifest['available'] if r['method']=='POTER')
        last=Path(accepted['checkpoint']).with_name('last_training.pt')
        # Trusted self-produced state, checked against the fresh checkpoint inventory.
        candidates=read_csv(out/'checkpoint_candidates.csv')
        row=next(r for r in candidates if Path(r['path'])==last)
        if sha256(last)!=row['sha256']:raise ValueError('Training state changed during audit')
        state=torch.load(last,map_location='cpu',weights_only=False)
        required=['model','optimizer','rng','loss_computer','config','epoch','best_wga']
        if any(k not in state for k in required):raise ValueError('Incomplete resume state')
        write_json(out/'resume_audit.json',dict(checkpoint=str(last),sha256=row['sha256'],
            last_completed_epoch=state['epoch'],seed=state['config']['seed'],
            configured_epochs=state['config']['epochs'],selected_epoch=258,
            complete_state_keys=required,optimizer_state_present=bool(state['optimizer']['state']),
            config=state['config'],resume_scope='same seed1 only; not independent seed initialization'))
        del state
        cache=Path('D:/POTER_features/waterbirds_standard.pt')
        write_json(out/'feature_cache_audit.json',dict(path=str(cache),sha256=sha256(cache),
            expected_sha256='8a7636923bffaddce16f843b38afe0a97743f19a151339ddb96e200d2c4d2258',
            matches_original=sha256(cache)=='8a7636923bffaddce16f843b38afe0a97743f19a151339ddb96e200d2c4d2258'))
    else:manifest=json.loads((out/'checkpoint_manifest.json').read_text())
    if 'original_missing_seeds' not in manifest:
        manifest['original_missing_seeds']=manifest.pop('missing_seeds')
    manifest['current_priority_missing_seeds']={'POTER':[config['next_training_seed']],
                                               'ERM_enhanced':[],'DPE_legacy':[]}
    write_json(out/'checkpoint_manifest.json',manifest)
    plan=priority_plan(config,manifest)
    cache_audit=json.loads((out/'feature_cache_audit.json').read_text())
    plan['historical_feature_cache_available']=cache_audit['matches_original']
    if not cache_audit['matches_original']:
        plan['status']='blocked_exact_historical_OT_cache_and_user_confirmation'
    plan['deferred_source_asset']='Historical unused CelebA module bytes not recovered; CUB sources preserved and deviation recorded in notebook'
    write_json(out/'execution_plan.json',plan)
    # Replace only this new directory's audit-generated all-seeds pending plan.
    write_json(out/'pending_training_plan.json',plan)
    report(out,config,manifest,plan)
    artifact_paths=['configs/experiment2_interim.json','scripts/profile_poter_data.py',
        'scripts/profile_poter_training.py','scripts/build_experiment2_notebook.py',
        'scripts/report_experiment2_interim.py','tests/test_experiment2_interim.py',
        'notebooks/experiment2_poter_cost_optimized.ipynb']
    write_json(out/'implementation_provenance.json',dict(sha256={p:sha256(ROOT/p) for p in artifact_paths},
        generated_training_notebook_executed=False,new_model_training=False))
    write_csv(out/'files_manifest.csv',[dict(path=p.name,bytes=p.stat().st_size,sha256=sha256(p))
        for p in sorted(out.iterdir()) if p.is_file() and p.name not in ['integrity.json','files_manifest.csv']])
    write_json(out/'integrity.json',dict(sha256={p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='integrity.json'}))
    print(json.dumps(dict(report=str(out/'interim_report.md'),additional_seeds=1,
        estimated_training_hours=plan['new_plan_poter_training_hours'],training_started=False)))


if __name__=='__main__':main()
