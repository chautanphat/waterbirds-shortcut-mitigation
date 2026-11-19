"""Optional timing wrapper around the unchanged 300-epoch POTER runner.

This DOES train when explicitly confirmed. Never run it merely to inspect status.
Records phase wall time and DataLoader wait; no AMP or optimizer changes.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-training',action='store_true')
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--profile-output',type=Path,required=True)
    args,runner_args=parser.parse_known_args()
    if not args.confirm_training:
        parser.error('Training is disabled: explicit --confirm-training required')
    if runner_args and runner_args[0]=='--':runner_args=runner_args[1:]
    # Run unchanged full recipe; the runner checkpoints at epoch boundaries.
    for flag,expected in [('--epochs','300'),('--batch-size','128'),('--seed','2')]:
        if flag not in runner_args or runner_args[runner_args.index(flag)+1]!=expected:
            parser.error(f'{flag} must remain {expected}')
    import torch
    baseline=args.baseline_root.resolve()
    sys.path[:0]=[str(baseline/'scripts'),str(baseline)]
    spec=importlib.util.spec_from_file_location('poter_original_runner',baseline/'scripts/kaggle_waterbirds.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    original=runner.run_epoch;records=[]

    class TimedLoader:
        def __init__(self,loader):self.loader=loader;self.wait=0.;self.batches=0
        def __len__(self):return len(self.loader)
        def __iter__(self):
            iterator=iter(self.loader)
            while True:
                started=time.perf_counter()
                try:batch=next(iterator)
                except StopIteration:return
                self.wait+=time.perf_counter()-started;self.batches+=1
                yield batch

    def timed(*positional,**kwargs):
        values=list(positional);loader=TimedLoader(values[3]);values[3]=loader
        torch.cuda.synchronize();started=time.perf_counter()
        result=original(*values,**kwargs)
        torch.cuda.synchronize()
        records.append(dict(epoch=values[0],training=kwargs['is_training'],
            phase='train' if kwargs['is_training'] else ('validation' if values[5] is not None else 'test'),
            wall_seconds=time.perf_counter()-started,loader_wait_seconds=loader.wait,batches=loader.batches,
            timing_note='Loader wait is host time, not a GPU kernel profile; boundary synchronization adds overhead.'))
        args.profile_output.parent.mkdir(parents=True,exist_ok=True)
        args.profile_output.write_text(json.dumps(records,indent=2))
        return result

    runner.run_epoch=timed
    sys.argv=[str(baseline/'scripts/kaggle_waterbirds.py'),*runner_args]
    runner.main()


if __name__=='__main__':main()
