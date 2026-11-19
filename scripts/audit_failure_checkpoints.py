"""Read-only checkpoint/recipe audit. Does not train or load inference models."""
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.counterfactual.common import sha256,read_csv,write_json,write_csv


def audit(config,out):
    source=ROOT/config['source_root']
    info=json.loads((source/'checkpoint_audit.json').read_text(encoding='utf-8'))
    bound={info[m]['checkpoint_sha256']:m for m in ('POTER','ERM_enhanced')}
    bound[info['DPE_legacy']['heads_sha256']]='DPE_legacy'
    accepted=[]
    for method,seed in [('POTER',1),('ERM_enhanced',0),('DPE_legacy',0)]:
        record=info[method]
        path=Path(record['heads_path'] if method=='DPE_legacy' else record['checkpoint'])
        expected=record['heads_sha256'] if method=='DPE_legacy' else record['checkpoint_sha256']
        if sha256(path)!=expected:raise ValueError('Available frozen checkpoint changed: '+str(path))
        accepted.append(dict(method=method,seed=seed,checkpoint=str(path),sha256=expected,
            backbone_sha256=record.get('backbone_sha256',record.get('checkpoint_sha256')),
            selection='validation WGA, epoch258' if method=='POTER' else 'fixed last stage0 checkpoint' if method=='ERM_enhanced' else 'fixed legacy 15-head ensemble',
            historical_binding=record.get('historical_pair_binding','checkpoint provenance audited')))
    candidates=[]
    for root in config['checkpoint_search_roots']:
        path=Path(root) if Path(root).is_absolute() else ROOT/root
        if path.exists():candidates.extend(p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in ('.pt','.pth','.ckpt','.safetensors'))
    candidates.extend(Path(p) for p in config['additional_checkpoint_candidates'] if Path(p).is_file())
    rows=[]
    known_other_backbone='f498212957c3cf780adee5ebac98dcf9344352eecf0a61ee20b0bb7c15cd9718'
    known_other_heads='e032881876c4a698383049856fca7254770428810381c27cf8febfc111057392'
    legacy=Path(json.loads((source/'blur_config.json').read_text())['legacy_root'])
    archived_sources=legacy/'results/dpe/strict_nested_20261115/sources.json'
    source_records=json.loads(archived_sources.read_text(encoding='utf-8'))['sources']
    other=next(r for r in source_records if r.get('backbone_sha256')==known_other_backbone)
    if other['ensemble_hashes']['partition0']!=known_other_heads:raise ValueError('Archived alternative checkpoint lineage differs')
    for path in sorted(set(candidates)):
        s=str(path).replace('\\','/');digest=sha256(path)
        eligible=False;seed=None;method='unknown'
        if digest in bound:
            method=bound[digest];seed=1 if method=='POTER' else 0
            reason='Available frozen checkpoint or byte-identical duplicate; not an additional training seed'
        elif digest==known_other_backbone:
            method='ERM_standard';seed=3;reason='Independent standard-ERM seed3; wrong recipe for enhanced-ERM400 target, excluded'
        elif digest==known_other_heads:
            method='DPE_partitioned_standard';seed=3;reason='15 heads on standard-ERM seed3, partition0; differs from legacy enhanced/full-validation recipe, excluded'
        elif 'recovery_tests/' in s:
            method='test_fixture';reason='Recovery/unit-test artifacts; not completed image-training checkpoints'
        elif 'CMNIST/' in s:
            method='POTER_other_dataset';seed=1;reason='Wrong dataset: CMNIST'
        elif 'last_training.pt' in s:
            method='POTER';seed=1;reason='Same seed1 training-state snapshot, not an independent training seed'
        elif 'waterbirds_bs16' in s:
            method='POTER';seed=1;reason='Same seed1 and batch16 recipe; smoke/changed-batch run cannot substitute batch128 independent seeds'
        elif '/outputs/' in s:
            method='DPE_or_weighting_heads';reason='Head-fitting/weighting artifacts; no new verified independently trained image backbone'
        else:
            reason='No verified compatible completed independent-seed provenance'
        rows.append(dict(path=str(path),bytes=path.stat().st_size,sha256=digest,method=method,seed=seed,
            additional_eligible_seed=eligible,reason=reason))
    write_csv(out/'checkpoint_candidates.csv',rows)
    dataset=Path(config['training_data_root'])
    metadata=dataset/'metadata.csv';metadata_rows=read_csv(metadata) if metadata.exists() else []
    missing=[r['img_filename'] for r in metadata_rows if not (dataset/r['img_filename']).is_file()]
    dataset_audit=dict(root=str(dataset),metadata_sha256=sha256(metadata) if metadata.exists() else None,
        canonical_metadata_matches=metadata.exists() and sha256(metadata)==sha256(ROOT/'data/waterbirds/metadata.csv'),
        split_counts=dict(Counter(r['split'] for r in metadata_rows)),missing_images=len(missing),
        missing_examples=missing[:5],training_executed=False)
    if metadata_rows and (len(metadata_rows)!=11788 or missing):raise ValueError('Training dataset audit found incomplete source assets')
    log=ROOT/'checkpoints/poter_kaggle_recovered/CUB/Clean/POTER_CUB_seed1_kaggle/log.txt'
    text=log.read_text(encoding='utf-8',errors='replace')
    times=[float(v) for v in re.findall(r'Epoch seconds:\s*([\d.]+)',text)]
    vram=[float(v) for v in re.findall(r'peak allocated VRAM MiB:\s*([\d.]+)',text)]
    try:
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):gpu='unavailable'
    training=dict(status='not_started_user_requires_prior_notification',new_checkpoints_trained=0,
        local_gpu=gpu,historical_poter_epochs_measured=len(times),historical_poter_mean_epoch_seconds=sum(times)/len(times),
        historical_poter_peak_vram_mib=max(vram),historical_poter_peak_vram_gib=max(vram)/1024,
        two_additional_poter_seeds_serial_hours=2*300*sum(times)/len(times)/3600,
        time_estimate_scope='Historical T4 throughput, not a new benchmark or a cloud runtime guarantee',
        unchanged_batch_size=128,local_memory_constraint='Historical measured peak exceeds 6GiB physical GPU; do not change batch/AMP/BN recipe to force local training',
        user_instruction=config['training_policy'],
        dataset=dataset_audit)
    incomplete=[]
    for path in (Path('D:/POTER_logs/waterbirds_standard')).rglob('log.txt'):
        t=path.read_text(encoding='utf-8',errors='replace');seed=re.search(r'^Seed:\s*(\d+)',t,re.M)
        incomplete.append(dict(log=str(path),sha256=sha256(path),seed=int(seed.group(1)) if seed else None,
            checkpoint_files=len(list(path.parent.glob('*.pth')))+len(list(path.parent.glob('*.pt'))),
            note='Log stops at training epoch0; no completed checkpoint. Failure cause is not recorded in this log.'))
    missing_seeds={m:[s for s in config['target_seeds'][m] if not any(r['method']==m and r['seed']==s for r in accepted)] for m in config['target_seeds']}
    recipe_source=legacy/'outputs/backbone_provenance_audit/remote_config.yaml'
    manifest=dict(status='incomplete_independent_seed_validation',target_checkpoints=9,available_method_checkpoints=3,
        unique_image_backbones=2,available=accepted,missing_seeds=missing_seeds,
        additional_candidates=len(rows),candidate_counts=dict(Counter(r['method'] for r in rows)),
        searched_roots=config['checkpoint_search_roots'],additional_paths=config['additional_checkpoint_candidates'],
        excluded_alternative_lineage=dict(source_id=other['id'],seed=other['seed'],recipe=other['recipe'],
            backbone_sha256=known_other_backbone,heads_sha256=known_other_heads,
            provenance=str(archived_sources),provenance_sha256=sha256(archived_sources)),
        incomplete_training_logs=incomplete,training=training,
        fixed_training_settings=dict(POTER=dict(batch_size=128,epochs=300,lr=1e-5,weight_decay=1.,
            f_power=.1,clip_cap=20.,reg_scale=.03,augmentation=True,pretrained='torchvision pretrained=True / ImageNet V1',
            selection='strictly best validation WGA; supplied baseline epoch258'),
            ERM_enhanced=dict(batch_size=128,epochs=400,lr=.001,momentum=.9,weight_decay=0.,sampling='class_balanced',
                augmentation=True,pretrained='IMAGENET1K_V2',freeze_bn=False,selection='last epoch, fixed stage0 policy',
                source=str(recipe_source),source_sha256=sha256(recipe_source)),
            DPE_legacy=dict(heads=15,epochs=20,batch_size=256,cov_reg=500000.,entropic_scale=30.,lr=.001,
                momentum=.9,wd_weight=10.,training_pool='canonical validation1199; never test',
                inference='equal mean probability, temperature1',historical_binding='absent')))
    write_json(out/'checkpoint_manifest.json',manifest)
    write_json(out/'pending_training_plan.json',dict(status='requires_user_notification_and_confirmation; not executable/started',
        missing_seeds=missing_seeds,requirements=manifest['fixed_training_settings'],training_data=dataset_audit,
        recommended_compute='Cloud GPU with at least16GiB VRAM; retain fixed recipes and checkpoint rules',
        dependencies='DPE seed1/2 require corresponding independently trained enhanced-ERM image backbones',
        no_test_selection=True,new_training_started=False))
    return manifest


if __name__=='__main__':
    config=json.loads((ROOT/'configs/failure_verification.json').read_text());out=ROOT/config['output_root'];out.mkdir(parents=True,exist_ok=True)
    result=audit(config,out);print(json.dumps(dict(available=result['available_method_checkpoints'],missing=result['missing_seeds'],training=result['training']['status'])))
