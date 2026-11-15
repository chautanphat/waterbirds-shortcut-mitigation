"""Export saved C-only inputs; no training, experiment execution or parameter search."""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
import torch
from experiments.discrete_weighting.store import sha,key
from experiments.geometry_aggregation_v5.persistence import atomic,dump
from scripts.verify_dpe_v5_analysis_inputs import array_record,verify

SOURCE=Path('outputs/geometry_aggregation_v5/run1')
OUT=Path('outputs/geometry_aggregation_v5/DPE_v5_Analysis_Inputs')
ARCHIVE=OUT.with_suffix('.zip')


def put(path,data):
    path=Path(path);raw=data.encode('utf-8') if isinstance(data,str) else data
    if not path.exists() or path.read_bytes()!=raw:atomic(path,raw)


def jsonwrite(path,value):put(path,json.dumps(value,sort_keys=True,indent=2,allow_nan=False))


def main():
    original=json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
    label_source=Path('outputs/weighting/features/val.npz')
    assert sha(label_source)==original['inputs'][str(label_source)]
    with np.load(label_source,allow_pickle=False) as f:labels={k:f[k] for k in ['sample_id','y','group']}
    assert len(labels['y'])==len(set(labels['sample_id']))==1199
    metadata=Path('waterbird_complete95_forest2water2/waterbird_complete95_forest2water2/metadata.csv')
    with metadata.open(encoding='utf-8',newline='') as f:official={r['img_filename']:r for r in csv.DictReader(f)}
    for sid,y,g in zip(labels['sample_id'],labels['y'],labels['group']):
        row=official[str(sid)];assert int(row['split'])==1 and int(row['y'])==int(y) and 2*int(row['y'])+int(row['place'])==int(g)
    index={str(s):i for i,s in enumerate(labels['sample_id'])};rows=[];source_files={};head_count=0
    for seed in range(3):
        seen=[]
        for fold in range(3):
            src=SOURCE/f'nested/seed{seed}/fold{fold}';dest=OUT/f'caches/seed{seed}/fold{fold}'
            plan=json.loads((src/'ABC_manifest.json').read_text(encoding='utf-8'));head=json.loads((src/'head_manifest.json').read_text(encoding='utf-8'))
            assert [r['slot'] for r in head['ordered_heads']]==list(range(15))
            with np.load(src/'member_predictions.npz',allow_pickle=False) as f:z={k:f[k] for k in f.files}
            assert sha(src/'member_predictions.npz')==head['cache_sha256']
            np.testing.assert_array_equal(z['C_ids'],plan['C']);np.testing.assert_array_equal(z['B_ids'],plan['B'])
            np.testing.assert_array_equal(z['head_hashes'],[r['sha256'] for r in head['ordered_heads']])
            scales=[]
            for j,record in enumerate(head['ordered_heads']):
                path=src/f'head{j:02d}/head.pt';assert sha(path)==record['sha256']
                h=torch.load(path,map_location='cpu',weights_only=True)
                assert h['member_id']==f'head-{j:02d}' and h['fit_ids']==plan['A'] and key(h['fit_ids'])==record['fit_ids_sha256']
                scales.append(float(h['scale'].abs()));head_count+=1
            ordered=np.array([index[str(s)] for s in z['C_ids']]);np.testing.assert_array_equal(ordered,plan['C_indices'])
            n=len(ordered);assert n==[400,400,399][fold]
            arrays=dict(probs=z['C_probs'],distances=z['C_distances'],logits=z['C_logits'],sample_id=z['C_ids'],
                y=labels['y'][ordered],group=labels['group'][ordered],seed=np.full(n,seed,dtype=np.int64),fold=np.full(n,fold,dtype=np.int64),
                head_id=np.array([f'head-{j:02d}' for j in range(15)]),head_sha256=z['head_hashes'],head_abs_scale=np.array(scales,dtype=np.float32),classes=np.array([0,1],dtype=np.int64))
            buf=io.BytesIO();np.savez_compressed(buf,**arrays);cache=dest/'C_predictions.npz';put(cache,buf.getvalue())
            with np.load(cache,allow_pickle=False) as f:
                for k,v in arrays.items():np.testing.assert_array_equal(f[k],v)
            for name in ['ABC_manifest.json','head_manifest.json']:put(dest/name,(src/name).read_bytes())
            with np.load(src/'G1_g50/predictions.npz',allow_pickle=False) as f:
                np.testing.assert_array_equal(f['sample_id'],arrays['sample_id']);np.testing.assert_array_equal(f['head_hashes'],arrays['head_sha256'])
                reference={k:array_record(f[k]) for k in ['probs','weights','uniform']}
            provenance=dict(seed=seed,fold=fold,role='Held-out outer C; diagnostic-only, forbidden for SAGW selection/tuning',
                original_member_predictions=dict(path=str(src/'member_predictions.npz'),sha256=sha(src/'member_predictions.npz')),
                original_G1_predictions=dict(path=str(src/'G1_g50/predictions.npz'),sha256=sha(src/'G1_g50/predictions.npz')),
                original_split_sha256=sha(src/'ABC_manifest.json'),original_head_order_sha256=sha(src/'head_manifest.json'),
                label_source=dict(path=str(label_source),sha256=sha(label_source),alignment='Exact sample_id join independently matches C_indices and C ID list'),
                official_metadata=dict(path=str(metadata),sha256=sha(metadata),split=1,labels='Original official benchmark labels, uncorrected; exact y and group=2*y+place cross-check'),
                exported_arrays={k:array_record(v) for k,v in arrays.items()},saved_g1_reference=reference,
                note='Only cache validation and export performed. Original G1 arrays are represented by digests, not duplicate reference caches. No B arrays, embeddings or checkpoints included.')
            jsonwrite(dest/'export_provenance.json',provenance)
            rows.append(dict(seed=seed,fold=fold,samples=n,cache=cache.relative_to(OUT).as_posix()));seen.extend(arrays['sample_id'].tolist())
        assert len(seen)==len(set(seen))==1199
    assert head_count==135
    buf=io.StringIO(newline='');writer=csv.writer(buf);writer.writerow(['development_index','sample_id','y','group'])
    for i,(sid,y,g) in enumerate(zip(labels['sample_id'],labels['y'],labels['group'])):writer.writerow([i,sid,int(y),int(g)])
    put(OUT/'provenance/development_index.csv',buf.getvalue())
    mappings={
        Path('baselines/dpe4subpop/src/dpe/geometry.py'):'source/geometry.py',
        Path('experiments/geometry_aggregation_v5/model.py'):'source/v5_aggregation.py',
        Path('baselines/dpe4subpop/src/dpe/verified_distance_v1.json'):'source/verified_distance_v1.json',
        SOURCE/'manifest.json':'provenance/source_run_manifest.json',
        SOURCE/'history/manifest.r1.json':'provenance/manifest.r1.json',
        SOURCE/'history/manifest.r2.json':'provenance/manifest.r2.json',
        SOURCE/'inputs/verification.json':'provenance/input_verification.json',
        SOURCE/'integrity_audit.json':'provenance/v5_integrity_audit.json',
        SOURCE/'production/integration_receipt.json':'provenance/integration_receipt.json',
        Path('scripts/verify_dpe_v5_analysis_inputs.py'):'verify_package.py',
    }
    for path,relative in mappings.items():
        if relative.startswith('source/'):assert sha(path)==original['source'][str(path)]
        put(OUT/relative,path.read_bytes());source_files[relative]=dict(original_path=str(path),sha256=sha(path))
    current=json.loads((OUT/'provenance/source_run_manifest.json').read_text(encoding='utf-8'));r2=json.loads((OUT/'provenance/manifest.r2.json').read_text(encoding='utf-8'))
    assert current['parent_manifest_sha256']==sha(OUT/'provenance/manifest.r2.json') and r2['parent_manifest_sha256']==sha(OUT/'provenance/manifest.r1.json')
    put(OUT/'CONTENTS.md',CONTENTS)
    files={p.relative_to(OUT).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
    assert not any(Path(p).suffix in ['.pt','.pth','.ckpt','.safetensors','.jpg','.jpeg','.png'] for p in files)
    jsonwrite(OUT/'manifest.json',dict(schema_version=1,package='DPE_v5_Analysis_Inputs',purpose='Frozen v5 diagnostic inputs; not SAGW fitting/selection data',
        files=files,caches=rows,source_copies=source_files,validation=dict(original_heads_hash_and_fit_order_verified=135,labels_crosschecked_against_official_metadata=True,
            expected_caches=9,seed_image_rows=3597,unique_images=1199,no_training=True,no_experiments_rerun=True,no_parameter_optimization=True),
        numerical_environment=dict(numpy=np.__version__,python=sys.version.split()[0]),
        label_note='Original official v5 labels preserved; known species-label issue not corrected. Labels/groups are evaluation-only.'))
    result=verify(OUT,reproduce_g1=False)
    # Independent process validates only exported bytes/manifests; no model inference or diagnostics run.
    process=subprocess.run([sys.executable,str(OUT/'verify_package.py'),'--root',str(OUT)],capture_output=True,text=True,encoding='utf-8',check=True)
    assert json.loads(process.stdout)['status']=='PASS'
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for relative in sorted(set(files)|{'manifest.json'}):z.write(OUT/relative,relative)
    put(ARCHIVE,buf.getvalue())
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None and set(z.namelist())==set(files)|{'manifest.json'}
    receipt=dict(**result,archive=str(ARCHIVE),bytes=ARCHIVE.stat().st_size,sha256=sha(ARCHIVE),independent_process_verification='PASS',zip_CRC='PASS')
    jsonwrite(ARCHIVE.with_suffix('.verification.json'),receipt);print(json.dumps(receipt,indent=2))


CONTENTS='''# DPE v5 analysis inputs for v6 SAGW diagnostics

**HELD-OUT DIAGNOSTIC ONLY. Do not fit SAGW, select alpha/gamma, pick random
seeds/variants, choose experts or tune any threshold using these C labels or
subgroup results. No v6 model, random-weighting experiment or LOO study has been
run to create this package. All source v5 artifacts are preserved.**

## Layout

```text
CONTENTS.md
manifest.json                       SHA-256 + byte size of every other file
verify_package.py                   standalone checker; NumPy only
caches/seed{0,1,2}/fold{0,1,2}/
  C_predictions.npz                 one C-only cache; 9 caches total
  ABC_manifest.json                 exact original A/B/C IDs and global indices
  head_manifest.json                exact original ordered-head hashes/fit hashes
  export_provenance.json            source hashes, label join, array hashes,
                                   cached G1 output digests for reproduction
source/
  geometry.py                       exact production G1 distance_mix source
  verified_distance_v1.json          exact frozen gamma50/alpha.25 metadata
  v5_aggregation.py                 exact research G1 branch (and its context)
provenance/
  development_index.csv             1199 unique canonical IDs/y/group/index;
                                   verifies split indices and labels independently
  source_run_manifest.json          exact v5 r3 source/input/protocol manifest
  manifest.r1.json, manifest.r2.json exact source-manifest lineage
  input_verification.json           original development input audit
  v5_integrity_audit.json           original completed nested audit
  integration_receipt.json          original G1 source hashes/acceptance record
```

Fold counts per seed:400,400,399. Each seed covers1199 unique C IDs exactly once.
There are3597 seed/image rows but only1199 distinct images across the package.
The3 head seeds share one frozen augmented400-epoch ERM* backbone. Development
and fold identities were previously inspected; these are not a fresh test set.

## Cache keys and alignment

| Key | Shape | Meaning |
|---|---|---|
| probs | [n,15,2] | Original raw member probabilities, copied bit-for-bit from C_probs |
| distances | [n,15,2] | Original normalized Euclidean prototype distances |
| logits | [n,15,2] | Actual learned-scale member logits, not assumed negative distances |
| sample_id | [n] | Original image filename identifier, exact original C row order |
| y | [n] | Original official class:0 landbird,1 waterbird |
| group | [n] | Original official subgroup:2*y+background; evaluation-only |
| seed, fold | [n] each | Explicit per-row outer split identity |
| classes | [2] | [0,1], same last-axis order in every cache |
| head_id | [15] | head-00..head-14, slots only within this actual ensemble |
| head_sha256 | [15] | Actual ordered head checkpoint hashes, matches head_manifest |
| head_abs_scale | [15] | Exact saved positive learned head scales; logits=-scale*distance |

Group0=landbird/land,1=landbird/water,2=waterbird/land,3=waterbird/water.
Rows align by sample_id with the canonical index AND ABC_manifest C_indices.
Labels/groups were independently cross-checked against the original metadata.csv
validation split. The original benchmark labels are preserved, including its
known three-species labeling caveat; no silent label correction was applied.
Do not treat head-03 in another fold as the same classifier. Use the actual
checkpoint hash/order. All135 source head files and ordered fit IDs were checked
read-only; no checkpoint files are included.

## Verify and reproduce (after extracting this ZIP)

```bash
python verify_package.py
python verify_package.py --reproduce-g1
```

The first command validates file hashes, all9 cache shapes, normalization,
logit/distance scale, ID/label alignment, A/B/C disjointness, head order and
unique full-seed C coverage. It does not recompute G1 or calculate metrics.
The optional second command invokes the exact copied G1 source on probabilities
and distances only, then compares frozen output/weight digests to the saved v5
cache digests. No labels/groups enter G1. Bitwise reproduction expects the
NumPy/environment recorded in manifest.json; another math-library/platform may
require a separate documented floating-point comparison. No optimization occurs.
These optional diagnostic commands were not run with --reproduce-g1 at export.

G1: r=softmax_heads(-50*min_class_distance), w=.75/15+.25*r,
P_G1=sum_heads(w*probs). Source geometry.py works standalone via importlib;
it needs NumPy and the adjacent metadata only, not torch/netcal/the full DPE
package. v5_aggregation.py is the exact research implementation; its broader
context imports SciPy but is not required for the standalone G1 check.

## Intended diagnostic boundaries

1. Reproduce the already frozen G1, with the original heads/representation/order.
2. Geometry-versus-random weighting is a diagnostic, with prespecified randomness;
   never select a favorable random seed/weighting rule by C accuracy or WGA.
3. Leave-one-expert-out stability must respect actual ensemble identity. The
   production source deliberately enforces15 experts. A reduced14-head diagnostic
   must explicitly document its averaging/residual renormalization; it is not a
   new promoted G1 setting and cannot select experts using held-out outcomes.

Future SAGW fitting and selection need separate permissible A/B/development
data. This C-only package provides no authorized training split. Group and
correctness diagnostics must not be converted into tuning targets.

Excluded: original images, model/optimizer checkpoints, embeddings, B predictions,
historical test data, duplicate G1 prediction arrays, training code/rerun tools,
plots and unrelated experiment grids/results. Original manifests intentionally
retain external source paths as provenance references; those paths are not
required by the standalone checker. Full run source/receipts are attestation,
not a claim that omitted checkpoints can be reconstructed from this ZIP.
'''


if __name__=='__main__':main()
