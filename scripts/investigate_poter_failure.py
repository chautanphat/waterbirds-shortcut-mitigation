"""Run frozen POTER, saved enhanced ERM and legacy DPE together; no training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import yaml
from PIL import Image
from src.counterfactual.common import read_csv,sha256,write_csv,write_json
from src.counterfactual.investigation import (FrozenComparison,audit_preliminary,select_cohort,
    original_records,intervention_records,evaluate_stream)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/poter_failure.yaml')
    parser.add_argument('--audit-smoke-only',action='store_true')
    args=parser.parse_args()
    config=yaml.safe_load((ROOT/args.config).read_text(encoding='utf-8'))
    output=ROOT/config['output_root']
    output.mkdir(exist_ok=True,parents=True)
    if (output/'experiment_config.yaml').exists():
        raise FileExistsError('Preserve completed/partial outputs; choose a fresh output_root in config')
    if shutil.disk_usage(output).free<120*1024**2:
        raise OSError('Need at least 120 MiB free for compact outputs; no full PNG cache is written')
    torch.set_num_threads(4)
    torch.manual_seed(config['sampling_seed'])
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    config['started_utc']=datetime.now(timezone.utc).isoformat()
    config['storage']='Compact streaming predictions; deterministic recipes/hashes; two preselected foregrounds saved as PNG examples'
    audit,excluded=audit_preliminary(config,output)
    fg=read_csv(ROOT/config['data_root']/'foregrounds_manifest.csv')
    cohort=select_cohort(fg,config,excluded)
    write_csv(output/'cohort.csv',cohort)
    config['cohort_sha256']=sha256(output/'cohort.csv')
    config['source_code_sha256']={str(p.relative_to(ROOT)):sha256(p) for p in [
        Path(__file__),ROOT/'src/counterfactual/investigation.py',ROOT/'src/counterfactual/compositor.py',
        ROOT/'src/counterfactual/model.py',ROOT/'src/methods/dpe_head_reweighting/ensemble.py',
        ROOT/'src/evaluation/dpe_export.py']}
    (output/'experiment_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    model=FrozenComparison(config,output)
    images=[]
    for y in (0,1):
        r=next(r for r in cohort if int(r['y'])==y)
        with Image.open(ROOT/config['data_root']/r['source_image_path']) as im: images.append(im.convert('RGB'))
    print(json.dumps(model.smoke(images,output)),flush=True)
    if args.audit_smoke_only:
        write_json(output/'status.json',dict(status='audit_smoke_passed'));return
    timings={}
    if config['full_original_test']:
        timings['full_original_test']=evaluate_stream(model,
            original_records(fg,ROOT/config['data_root'],'full_test'),output/'original_test_predictions.csv',
            total_sources=len(fg),phase='full_original_test',output=output)
    timings['replication100']=evaluate_stream(model,intervention_records(cohort,config,output),
        output/'predictions.csv',total_sources=len(cohort),phase='replication100',output=output)
    write_json(output/'timings.json',timings)
    write_json(output/'status.json',dict(status='inference_completed',timings=timings))
    print(json.dumps(timings),flush=True)


if __name__=='__main__':main()
