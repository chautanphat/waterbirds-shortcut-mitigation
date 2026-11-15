"""Publish existing v7 evidence with G1 baseline / pooled R reference roles."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from experiments.discrete_weighting.store import sha
from experiments.geometry_aggregation_v5.persistence import atomic,dump

SOURCE=Path('experiments/g1_residual_v7')
DEST=Path('results/dpe/g1_residual_v7_20261115')


def main():
    state=json.loads((SOURCE/'state.json').read_text());assert state['status']=='DONE_EXPLORATORY'
    verdict=json.loads((SOURCE/'metrics/FINAL_VERDICT.json').read_text());assert verdict['primary']=='G1_RH' and verdict['promotion'] is False
    receipts=[json.loads(p.read_text()) for p in (SOURCE/'completed').glob('*.json')]
    artifacts={p:v for r in receipts for p,v in r['artifacts'].items()}
    names=['EXPLORATORY_C_REPORT.md','DELIVERY_SUPPLEMENT.md','LIMITATIONS_AND_NEXT_STEPS.md','CONFIRMATION_BLOCKED.md',
        'INPUT_INVENTORY.json','DATA_PROVENANCE.md','BASELINE_REPRODUCTION.csv','DELIVERY_VALIDATION.json','RESUME_VERIFICATION.json',
        'config.lock.json','EXPERIMENT_LOCK.json','manifest.json','state.json','HANDOFF_STATUS.md','NEXT_AGENT_PROMPT.md','CHECKSUMS.sha256']
    files=[SOURCE/n for n in names]
    files+=[p for p in (SOURCE/'metrics').glob('*') if p.is_file() and p.suffix in ['.csv','.json','.npz']]
    files+=list((SOURCE/'completed').glob('*.json'))
    files+=list((SOURCE/'checkpoints').rglob('model.json'))+list((SOURCE/'checkpoints').rglob('model_provenance.json'))
    files+=list((SOURCE/'checkpoints').rglob('head_effects.csv'))+list((SOURCE/'checkpoints').rglob('B_INNER_SPLITS.json'))+list((SOURCE/'checkpoints').rglob('split.json'))
    files+=[SOURCE/'acceptance'/kind/'receipt.json' for kind in ['store','secondary','C']]
    exported={}
    for source in sorted(set(files)):
        rel=source.relative_to(SOURCE).as_posix();h=sha(source)
        if rel in artifacts:assert h==artifacts[rel]['sha256'],'Committed artifact mismatch: '+rel
        dest=DEST/rel
        if not dest.exists() or sha(dest)!=h:atomic(dest,source.read_bytes())
        exported[rel]=dict(sha256=h,bytes=source.stat().st_size)
    role='EXPLORATORY_ONLY — REUSED DEVELOPMENT — NOT INDEPENDENT CONFIRMATION'
    dump(DEST/'EXPORT_MANIFEST.json',dict(role=role,files=exported,source=SOURCE.as_posix(),experiments_rerun=False,
        presentation_roles=dict(baseline='G1_frozen',reference_method='G1_R'),frozen_scientific_verdict=verdict,
        production_changed=False,original_receipts=len(receipts),
        receipt_paths='Original scientific receipts refer to the original workspace. Omitted per-fold caches, secondary OOF arrays, RNG batch outputs and inputs are preserved locally.',
        inference_cache_shape=dict(probs=[5,3,1199,2],weights=[5,3,1199,15]),
        original_v6_NO_GO_sha256=sha(Path('experiments/ccw_v6/LOCKED_CANDIDATE.json'))))
    atomic(DEST/'README.md','# G1 baseline / G1 + R reference evidence\n\n**'+role+'**\n\n'
        'Read [the branch report](../../../DPE_V5_G1_BASELINE_AND_R_REFERENCE.md), [original v7 report](EXPLORATORY_C_REPORT.md) '
        'and [mechanism supplement](DELIVERY_SUPPLEMENT.md). G1 is the branch comparison baseline; pooled G1_R is the reference method. '
        'The preregistered v7 primary remains RH and its original B_DIRECTION_NO_GO / EXPLORATORY_C_WGA_EQUAL verdict is unchanged.\n\n'
        'metrics/ contains all required result tables and the curated C decomposition NPZ with all five methods, 15 expert weights and probabilities. '
        'C_EVALUATION_LABELS.npz is a separate evaluator-only output. All 180 small ridge/head model JSONs, head-effect tables, split provenance and historical completion receipts are included. '
        'Checksums and receipts describe original artifacts, including dependencies intentionally omitted from this Git export. '
        'This is an archival export, not a complete standalone resume bundle. Inputs, prototype/backbone weights, individual inference/OOF duplicates, uncertainty batch arrays and ZIPs remain in the original workspace. '
        'Exported state/handoffs are snapshots; cloning does not restore the omitted local data.\n\n'
        'No experiments were rerun. scripts/export_g1_residual_v7_branch.py verifies source receipts; --verify-index checks the staged Git blobs byte-for-byte against EXPORT_MANIFEST.json.\n')
    print(json.dumps(dict(files=len(exported),bytes=sum(v['bytes'] for v in exported.values()),experiments_rerun=False)))


def verify_index():
    manifest=json.loads((DEST/'EXPORT_MANIFEST.json').read_text());listing=subprocess.run(['git','ls-files','-s','--',str(DEST)],capture_output=True,text=True,check=True).stdout
    objects={line.split('\t',1)[1]:line.split()[1] for line in listing.splitlines()};rows=[((DEST/p).as_posix(),v) for p,v in manifest['files'].items()]
    data=subprocess.run(['git','cat-file','--batch'],input=('\n'.join(objects[p] for p,v in rows)+'\n').encode(),capture_output=True,check=True).stdout;stream=io.BytesIO(data)
    for path,expected in rows:
        _,kind,n=stream.readline().split();assert kind==b'blob';content=stream.read(int(n));assert stream.read(1)==b'\n'
        assert hashlib.sha256(content).hexdigest()==expected['sha256'],'Staged bytes differ: '+path
    print(json.dumps(dict(staged_export_hashes='PASS',files=len(rows),byte_preservation=True)))


if __name__=='__main__':
    if '--verify-index' in sys.argv:verify_index()
    else:main()
