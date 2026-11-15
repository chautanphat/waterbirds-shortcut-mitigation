"""Input inventory/export only; CCW fitting loads B-only exported files."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
from experiments.discrete_weighting.store import sha,key
from experiments.geometry_aggregation_v5.io import load,save_npz
from experiments.geometry_aggregation_v5.persistence import atomic,dump
from .store import csvwrite

V5=Path('outputs/geometry_aggregation_v5/run1')


def locate(name):
    found=[]
    for base,dirs,files in os.walk('.'):
        dirs[:]=[d for d in dirs if d not in ['.git','.venv','waterbird_complete95_forest2water2','node_modules','__pycache__'] and not d.startswith('source_snapshot')]
        if Path(base).parts[:2]==('experiments','ccw_v6') and 'inputs' in Path(base).parts:continue
        if name in files:found.append(Path(base)/name)
    if not found:return None
    if len({sha(p) for p in found})!=1:raise ValueError('Ambiguous copies: '+name)
    return sorted(found,key=lambda p:(len(p.parts),str(p)))[0]


def discover():
    archive=locate('DPE_v5_Analysis_Inputs.zip')
    package=Path('outputs/geometry_aggregation_v5/DPE_v5_Analysis_Inputs')
    if not package.exists():package=Path('dpe_v6_analysis_inputs')
    paths=[Path('baselines/dpe4subpop/DPE_CCW_v6_Codex_Handoff.md')]
    if archive:paths.append(archive)
    elif (package/'manifest.json').exists():paths.extend(p for p in package.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for seed in range(3):
        for fold in range(3):
            folder=V5/f'nested/seed{seed}/fold{fold}'
            paths.extend(folder/name for name in ['member_predictions.npz','ABC_manifest.json','head_manifest.json','B_only_normalizer.json'])
    for p in ['baselines/dpe4subpop/src/dpe/core.py','baselines/dpe4subpop/src/dpe/geometry.py','baselines/dpe4subpop/src/dpe/verified_distance_v1.json']:paths.append(Path(p))
    fresh=Path('experiments/ccw_v6/fresh_inputs/CCW_FRESH_COHORT.json')
    return dict(archive=archive,extracted=package,base_inputs={str(p):sha(p) if p.is_file() else None for p in paths},fresh=fresh)


def inventory(root,discovery):
    rows=[]
    for path,h in discovery['base_inputs'].items():
        p=Path(path);schema=None
        if h and p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as f:schema={k:list(f[k].shape) for k in f.files}
        rows.append(dict(path=path,exists=h is not None,bytes=p.stat().st_size if h else None,sha256=h,schema=schema))
    path=root/'inputs/inventory.json';dump(path,rows)
    md=root/'INPUT_INVENTORY.md';atomic(md,'# CCW v6 input inventory\n\n'+ '\n'.join(f'- `{r["path"]}`: '+(f'{r["bytes"]} bytes, SHA256 `{r["sha256"]}`' if r['exists'] else '**MISSING**') for r in rows)+'\n\nAll original C data are reused exploratory development. No untouched cohort is assumed from a new filename.\n')
    return dict(found=sum(r['exists'] for r in rows),missing=sum(not r['exists'] for r in rows)),[path,md]


def extract_and_verify(root,discovery):
    dest=root/'inputs/v5_analysis';files=[]
    if discovery['archive']:
        with zipfile.ZipFile(discovery['archive']) as z:
            assert z.testzip() is None
            for info in z.infolist():
                name=Path(info.filename)
                if name.is_absolute() or '..' in name.parts or ':' in info.filename:raise ValueError('Unsafe analysis archive')
                if info.is_dir():continue
                path=dest/name;atomic(path,z.read(info));files.append(path)
    else:
        source=discovery['extracted']
        if not (source/'manifest.json').exists():raise FileNotFoundError('DPE_v5_Analysis_Inputs.zip or validated extracted analysis package')
        for p in source.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                path=dest/p.relative_to(source);atomic(path,p.read_bytes());files.append(path)
    proc=subprocess.run([sys.executable,str(dest/'verify_package.py'),'--root',str(dest),'--reproduce-g1'],capture_output=True,text=True,encoding='utf-8')
    log=root/'inputs/package_verification.log';atomic(log,proc.stdout+'\n'+proc.stderr)
    if proc.returncode:raise RuntimeError('Package/G1 verification failed: '+str(log))
    result=json.loads(proc.stdout);assert result['caches']==9 and result['frozen_g1_reproduction']=='PASS'
    report=root/'PROVENANCE_AUDIT.json';dump(report,dict(status='PASS',package_verifier_sha256=sha(dest/'verify_package.py'),package_manifest_sha256=sha(dest/'manifest.json'),verification=result,
        source_label_policy='Original official labels, known three-species caveat retained; no silent relabeling',training_or_selection_used_C=False))
    return dict(status='PASS',C_caches=9,G1_exact='PASS'),files+[log,report]


def canonical(root):
    path=root/'inputs/v5_analysis/provenance/development_index.csv'
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def frozen_geometry(root):
    spec=importlib.util.spec_from_file_location('ccw_frozen_v5_geometry',root/'inputs/v5_analysis/source/geometry.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def export_B(root,seed,fold):
    src=V5/f'nested/seed{seed}/fold{fold}';pkg=root/f'inputs/v5_analysis/caches/seed{seed}/fold{fold}'
    plan=json.loads((pkg/'ABC_manifest.json').read_text(encoding='utf-8'));head=json.loads((pkg/'head_manifest.json').read_text(encoding='utf-8'));assert plan==json.loads((src/'ABC_manifest.json').read_text(encoding='utf-8'))
    source=src/'member_predictions.npz'
    if not source.exists():raise FileNotFoundError('Matching B source missing: '+str(source))
    assert sha(source)==head['cache_sha256'];z=load(source,['B_probs','B_distances','B_logits','B_ids','head_hashes'])
    hashes=[r['sha256'] for r in head['ordered_heads']];np.testing.assert_array_equal(z['head_hashes'],hashes);np.testing.assert_array_equal(z['B_ids'],plan['B'])
    a,b,c=[set(plan[k]) for k in ['A','B','C']];assert not a&b and not b&c and not a&c
    label_rows=canonical(root);index={r['sample_id']:r for r in label_rows};y=np.array([int(index[str(s)]['y']) for s in z['B_ids']]);idx=np.array([int(index[str(s)]['development_index']) for s in z['B_ids']])
    np.testing.assert_array_equal(idx,plan['B_indices'])
    assert z['B_probs'].shape==z['B_distances'].shape==z['B_logits'].shape==(len(y),15,2)
    e=np.exp(z['B_logits']-z['B_logits'].max(-1,keepdims=True));np.testing.assert_allclose(e/e.sum(-1,keepdims=True),z['B_probs'],atol=2e-6,rtol=0)
    with np.load(pkg/'C_predictions.npz',allow_pickle=False) as f:scales=f['head_abs_scale'];np.testing.assert_array_equal(f['head_sha256'],hashes)
    np.testing.assert_allclose(z['B_logits'],-scales[None,:,None]*z['B_distances'],atol=2e-6,rtol=0)
    physical=0
    for j,h in enumerate(hashes):
        p=src/f'head{j:02d}/head.pt'
        if p.exists():assert sha(p)==h;physical+=1
    normalizer=src/'B_only_normalizer.json';normalizer_verified=False
    if normalizer.exists():
        old=json.loads(normalizer.read_text(encoding='utf-8'))['ensembles']['0'];np.testing.assert_array_equal(old['fit_ids'],z['B_ids']);assert old['head_hashes']==hashes
        np.testing.assert_allclose(old['mean'],z['B_distances'].min(-1).mean(0),atol=2e-6,rtol=0)
        np.testing.assert_allclose(old['std'],np.maximum(z['B_distances'].min(-1).std(0),1e-6),atol=2e-6,rtol=0);normalizer_verified=True
    path=root/f'inputs/B/seed{seed}/fold{fold}.npz';save_npz(path,probs=z['B_probs'],distances=z['B_distances'],logits=z['B_logits'],sample_id=z['B_ids'],y=y,head_sha256=z['head_hashes'])
    provenance=path.with_suffix('.json');dump(provenance,dict(seed=seed,outer_fold=fold,source=str(source),source_sha256=sha(source),head_sha256=hashes,physical_head_files_verified=physical,
        B_ids=plan['B'],A_ids_sha256=key(plan['A']),C_ids_sha256=key(plan['C']),label_source_sha256=sha(root/'inputs/v5_analysis/provenance/development_index.csv'),groups_exported=False,
        v5_B_normalizer_verified=normalizer_verified,v5_normalizer_not_used_for_inner_CV=True,training=False))
    return dict(seed=seed,outer_fold=fold,B_images=len(y),physical_heads_verified=physical,B_C_head_match=True),[path,provenance]
