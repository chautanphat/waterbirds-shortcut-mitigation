"""Reproduce a saved factorial experiment without opening a browser."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.counterfactual.common import DEFAULT_DATA,DEFAULT_RUNS,DEFAULT_CHECKPOINT
from src.counterfactual.experiment import run_experiment
from src.counterfactual.model import ModelAdapter

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=DEFAULT_DATA)
    p.add_argument('--output-root',type=Path,default=DEFAULT_RUNS)
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    p.add_argument('--device',default='cpu')
    p.add_argument('--no-model',action='store_true')
    p.add_argument('--resume',action='store_true',help='Create a new immutable run linked to this saved config')
    args=p.parse_args()
    import torch
    torch.set_num_threads(4)
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    if args.resume:
        cfg['parent_run_id']=cfg.get('run_id')
    adapter=[] if args.no_model else [ModelAdapter(args.checkpoint,args.device,metadata=args.data_root/'metadata.csv')]
    if args.resume and cfg.get('models'):
        if len(cfg['models']) != len(adapter):
            p.error('Resume requires the same configured model set; use the UI for a multi-model saved run')
        for saved,current in zip(cfg['models'],adapter):
            now=current.info()
            if any(saved.get(k)!=now.get(k) for k in ('model_id','checkpoint_sha256','preprocess_fingerprint','heads_sha256')):
                p.error('Saved checkpoint, DPE heads or preprocessing differs from the current adapter')
    print(json.dumps(run_experiment(cfg,adapter,data_root=args.data_root,output_root=args.output_root),indent=2))
