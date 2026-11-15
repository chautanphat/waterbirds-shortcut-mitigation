"""Export committed CCW research evidence for Git without rerunning experiments."""
import json
import hashlib
import io
from pathlib import Path
import subprocess
import sys
from experiments.discrete_weighting.store import sha
from experiments.geometry_aggregation_v5.persistence import atomic,dump

SOURCE=Path('experiments/ccw_v6/exploratory_c')
DEST=Path('results/dpe/ccw_v6_exploratory_c_20261115')


def main():
    state=json.loads((SOURCE/'state.json').read_text())
    assert state['status']=='EXPLORATORY_COMPLETE' and state['original_B_decision']=='NO_GO'
    receipts=[json.loads(p.read_text()) for p in (SOURCE/'completed').glob('*.json')]
    artifacts={k:v for r in receipts for k,v in r['artifacts'].items()}
    files=[SOURCE/n for n in ['EXPLORATORY_C_RESULTS.csv','EXPLORATORY_C_PER_GROUP.csv','EXPLORATORY_C_FLIPS.csv',
        'EXPLORATORY_C_PREDICTIONS.npz','EXPLORATORY_C_REPORT.md','PREDICTION_SCHEMA.json','ADVANTAGE_AND_WORST_GROUP.json',
        'BASELINE_REPRODUCTION.json','DELIVERY_VALIDATION.json','FULL_RESUME_CHECK.json','INPUT_AUDIT.json',
        'ORIGINAL_PRESERVATION_AUDIT.json','PROTOCOL_DECISIONS.md','manifest.json','state.json','HANDOFF_STATUS.md','NEXT_AGENT_PROMPT.md']]
    files+=list((SOURCE/'fits').rglob('*.json'))+list((SOURCE/'fits').rglob('*.csv'))
    files+=list((SOURCE/'completed').glob('*.json'))
    files+=[SOURCE/'uncertainty'/n for n in ['paired_CI.json','rng_state.json']]
    exported={}
    for source in sorted(files):
        rel=source.relative_to(SOURCE).as_posix();digest=sha(source)
        if rel in artifacts:assert digest==artifacts[rel]['sha256']
        dest=DEST/rel
        if not dest.exists() or sha(dest)!=digest:atomic(dest,source.read_bytes())
        exported[rel]=dict(sha256=digest,bytes=source.stat().st_size)
    banner='EXPLORATORY_ONLY — REUSED DEVELOPMENT — NOT INDEPENDENT CONFIRMATION'
    dump(DEST/'EXPORT_MANIFEST.json',dict(role=banner,files=exported,source=str(SOURCE.as_posix()),
        original_B_decision='NO_GO',production_changed=False,experiments_rerun=False,
        cache_probes=dict(probs=[5,3,1199,2],weights=[5,3,1199,15]),
        receipt_paths='Artifacts in original receipts are relative to the original exploratory_c run; heavy per-fold caches and acceptance controls remain local.',
        original_NO_GO_sha256=sha(Path('experiments/ccw_v6/LOCKED_CANDIDATE.json'))))
    atomic(DEST/'README.md','# CCW v6 archived exploratory results\n\n**'+banner+'**\n\n'
        'Read [the branch research disposition](../../../DPE_V4_RESEARCH_STATUS.md) and [the full report](EXPLORATORY_C_REPORT.md). '
        'The original B NO_GO remains unchanged. Broader variant exploration is on hold; the next research direction is direct G1 failure analysis.\n\n'
        'The NPZ contains all five methods, three seeds, 1199 aligned image IDs, probabilities, all 15 expert weights and evaluation labels/groups. '
        'Head order is head_sha256[seed, fold[seed,image], expert]. See PREDICTION_SCHEMA.json. '
        'Small fitted tables and all original completion receipts are retained; receipt artifact paths refer to the original local run, not a complete standalone resume bundle. '
        'Original images, per-fold duplicate caches, trained heads, acceptance controls and result ZIPs are excluded.\n\n'
        'The checkpoint state and handoffs are historical export snapshots. They do not mean a clone contains the omitted local inputs. '
        'Run scripts/export_ccw_v6_branch.py only from a workspace containing the completed original study.\n')
    print(json.dumps(dict(files=len(exported),bytes=sum(r['bytes'] for r in exported.values()),experiments_rerun=False)))


def verify_index():
    manifest=json.loads((DEST/'EXPORT_MANIFEST.json').read_text())
    listing=subprocess.run(['git','ls-files','-s','--',str(DEST)],check=True,capture_output=True,text=True).stdout
    objects={line.split('\t',1)[1]:line.split()[1] for line in listing.splitlines()}
    rows=[(str((DEST/p).as_posix()),v) for p,v in manifest['files'].items()]
    hashes=[objects[p] for p,v in rows]
    data=subprocess.run(['git','cat-file','--batch'],input=('\n'.join(hashes)+'\n').encode(),capture_output=True,check=True).stdout
    stream=io.BytesIO(data)
    for path,expected in rows:
        _,kind,size=stream.readline().split();assert kind==b'blob'
        content=stream.read(int(size));assert stream.read(1)==b'\n'
        assert hashlib.sha256(content).hexdigest()==expected['sha256'],'Staged bytes differ: '+path
    print(json.dumps(dict(staged_export_hashes='PASS',files=len(rows),byte_preservation=True)))


if __name__=='__main__':
    if '--verify-index' in sys.argv:verify_index()
    else:main()
