"""CPU-only input-pipeline measurement. No model, optimizer, or training."""
import argparse
import io
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]


def original_transform():
    # Import the existing POTER transform, never approximate its augmentation.
    baseline = ROOT / 'baselines/poter'
    sys.path.insert(0, str(baseline))
    from data.cub_dataset import get_transform_cub
    return get_transform_cub('resnet50', True, True)


class InputProbe(Dataset):
    def __init__(self, root, filenames, transform):
        self.root, self.filenames, self.transform = Path(root), filenames, transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        with Image.open(self.root / self.filenames[index]) as image:
            return self.transform(image.convert('RGB'))


def measure(root, samples, workers, seed):
    import csv
    with (root / 'metadata.csv').open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['split'] == '0']
    # Predeclared input ordering; no test labels used for optimization.
    files = [r['img_filename'] for r in rows[:samples]]
    torch.set_num_threads(2)
    transform = original_transform()
    results = []
    for count in workers:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        loader = DataLoader(InputProbe(root, files, transform), batch_size=128,
                            num_workers=count, shuffle=False, pin_memory=False)
        started = time.perf_counter(); waits = []; previous = started
        for batch in loader:
            now = time.perf_counter(); waits.append(now - previous)
            assert batch.shape[1:] == (3, 224, 224)
            previous = time.perf_counter()
        results.append(dict(workers=count,images=len(files),batches=len(waits),
                            wall_seconds=time.perf_counter()-started,
                            batch_wait_seconds=waits,includes_worker_startup=True))
    decode = []; byte_reads = []; memory_decode = []
    # Unmodified compressed bytes only; never cache augmented train tensors.
    for filename in files[:64]:
        started=time.perf_counter(); payload=(root/filename).read_bytes()
        byte_reads.append(time.perf_counter()-started)
        started=time.perf_counter()
        with Image.open(root/filename) as image: disk=image.convert('RGB'); disk.load()
        decode.append(time.perf_counter()-started)
        started=time.perf_counter()
        with Image.open(io.BytesIO(payload)) as image: memory=image.convert('RGB'); memory.load()
        memory_decode.append(time.perf_counter()-started)
        if disk.size != memory.size or disk.tobytes()!=memory.tobytes():
            raise ValueError('Byte caching changed decoded RGB')
    return dict(training_executed=False,model_loaded=False,device='CPU',
        split='train only',samples=len(files),seed=seed,torch_threads=2,
        torch=str(torch.__version__),numpy=np.__version__,loader=results,
        compressed_byte_cache=dict(images=len(decode),decoded_pixels_equal=True,
            file_read_seconds=sum(byte_reads),file_decode_seconds=sum(decode),
            memory_decode_seconds=sum(memory_decode)),
        limitations=['Local CPU/disk microbenchmark, not cloud end-to-end throughput.',
            'Worker startup included; short measurement cannot establish steady-state speedup.',
            'Worker count changes augmentation RNG streams; default production workers stays zero.',
            'No CUDA forward/backward, AMP, or training timing was executed.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=256)
    parser.add_argument('--workers',type=int,nargs='+',default=[0,2])
    parser.add_argument('--seed',type=int,default=20261118)
    args=parser.parse_args()
    result=measure(args.data_root,args.samples,args.workers,args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
