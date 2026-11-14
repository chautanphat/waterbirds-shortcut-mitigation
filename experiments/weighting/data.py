"""Feature extraction and no-group head fitting; held-out labels never enter fit."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from .cache import digest, identity, write_json, save_predictions, load_predictions


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


def assert_disjoint(*sets):
    pools = [set(map(str, s)) for s in sets]
    for i, a in enumerate(pools):
        if len(a) != len(sets[i]):
            raise ValueError('Duplicate IDs within a partition')
        for b in pools[i + 1:]:
            if a & b:
                raise ValueError('Leakage: overlapping sample IDs')


def partition(y, ids, seed=0, fractions=(.6, .2, .2)):
    if not np.isclose(sum(fractions), 1) or min(fractions) <= 0:
        raise ValueError('Invalid split fractions')
    train, other = train_test_split(np.arange(len(y)), train_size=fractions[0],
                                   stratify=y, random_state=seed)
    gate, select = train_test_split(other, train_size=fractions[1] / sum(fractions[1:]),
                                   stratify=y[other], random_state=seed + 1)
    assert_disjoint(ids[train], ids[gate], ids[select])
    return train, gate, select


def extract(config, split):
    """Cache frozen ResNet features once, preserving reference metadata order."""
    from src.datasets.waterbirds import Waterbirds
    from src.models.resnet import load_resnet_checkpoint
    from torch.utils.data import DataLoader
    root = Path(config['output_dir']) / 'features'
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{split}.npz'
    ds = Waterbirds(config['dataset_dir'], split)
    provenance = dict(backbone_sha256=digest(config['backbone']), metadata_sha256=digest(ds.metadata_path),
                      preprocessing='square256;center224;ImageNet;raw2048', split=split)
    marker = path.with_suffix('.json')
    if path.exists():
        if not marker.exists() or json.loads(marker.read_text()) != provenance:
            raise ValueError('Feature provenance mismatch')
        with np.load(path, allow_pickle=False) as f:
            data = dict(f)
        if data['sample_id'].tolist() != ds.filenames or not np.array_equal(data['y'], ds.y):
            raise ValueError('Feature metadata alignment differs')
        return data
    model = load_resnet_checkpoint(config['backbone'], device=config['device']).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    chunk_root = root / (split + '_chunks')
    chunk_root.mkdir(exist_ok=True)
    chunk_marker = chunk_root / 'identity.json'
    chunk_identity = provenance | dict(batch_size=config.get('feature_batch_size', 24))
    if chunk_marker.exists() and json.loads(chunk_marker.read_text()) != chunk_identity:
        raise ValueError('Feature chunk identity changed')
    write_json(chunk_marker, chunk_identity)
    blocks = []
    start = time.perf_counter()
    loader = DataLoader(ds, batch_size=config.get('feature_batch_size', 24), shuffle=False,
                        num_workers=0)
    with torch.inference_mode():
        for batch, (_, x, _, _) in enumerate(loader):
            chunk = chunk_root / f'{batch:05d}.npy'
            if chunk.exists():
                values = np.load(chunk, allow_pickle=False)
            else:
                values = model.extract_features(x.to(config['device'])).cpu().numpy()
                temporary = chunk.with_suffix('.tmp')
                with temporary.open('wb') as stream:
                    np.save(stream, values)
                temporary.replace(chunk)
            if values.shape != (len(x), 2048) or not np.isfinite(values).all():
                raise ValueError('Corrupt feature chunk')
            blocks.append(values)
            if len(blocks) % 10 == 0:
                print(f'Features {split}: {sum(len(b) for b in blocks)}/{len(ds)} ({time.perf_counter()-start:.0f}s)', flush=True)
    data = dict(embedding=np.concatenate(blocks), sample_id=np.asarray(ds.filenames), y=ds.y, group=ds.g)
    if not np.isfinite(data['embedding']).all():
        raise ValueError('Nonfinite embeddings')
    np.savez_compressed(path, **data)
    write_json(marker, provenance)
    return data


class PrototypeHead(nn.Module):
    def __init__(self, dim, classes):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(classes, dim) * .01)
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x):
        d = torch.cdist(F.normalize(x, dim=-1), F.normalize(self.prototypes, dim=-1),
                        compute_mode='donot_use_mm_for_euclid_dist')
        return -self.scale.abs() * d


def standardized(raw):
    raw = np.asarray(raw, dtype=np.float32)
    sd = raw.std(1, keepdims=True)
    if raw.ndim != 2 or not np.isfinite(raw).all() or (sd <= 0).any():
        raise ValueError('Invalid frozen embeddings')
    return torch.from_numpy((raw - raw.mean(1, keepdims=True)) / sd)


def fit_heads(raw, y, ids, config, seed, path):
    """Class-balanced subsets; fixed final epoch. No groups or external labels accepted.

    Same IsoMax+ distance/entropic/prototype/IPS math as upstream. Deliberate
    no-group adaptation: class balancing and no WGA checkpoint selection.
    """
    signature = dict(ids=list(map(str, ids)), seed=int(seed), training=config['head_training'],
                     feature_sha256=__import__('hashlib').sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
                     labels=list(map(int, y)), code_sha256=digest(__file__))
    path = Path(path)
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['signature'] != signature:
            raise ValueError('Refusing stale/in-sample head checkpoint')
        return saved
    seed_all(seed)
    settings = config['head_training']
    device = config['device']
    x = standardized(raw).to(device)
    labels = torch.tensor(y, dtype=torch.long, device=device)
    pools = [np.flatnonzero(y == c) for c in range(int(max(y)) + 1)]
    if any(len(p) == 0 for p in pools):
        raise ValueError('Missing head-training class')
    size = min(map(len, pools))
    heads, scales = [], []
    progress = path.with_suffix('.progress.pt')
    if progress.exists():
        state = torch.load(progress, map_location='cpu', weights_only=True)
        if state['signature'] != signature:
            raise ValueError('Head resume identity changed')
        heads, scales = list(state['prototypes']), list(state['scales'])
        torch.set_rng_state(state['torch_rng'])
        ns = state['numpy_rng']
        np.random.set_state((ns[0], np.asarray(ns[1], dtype='uint32'), *ns[2:]))
    for stage in range(len(heads), settings['members']):
        chosen = np.stack([np.random.permutation(p)[:size] for p in pools], 1).ravel()
        head = PrototypeHead(x.shape[1], len(pools)).to(device)
        opt = torch.optim.SGD(head.parameters(), lr=settings['lr'], momentum=.9)
        for _ in range(settings['epochs']):
            indices = np.random.permutation(chosen)
            # Retain the final partial batch: important for the smaller class-only pool.
            for offset in range(0, len(indices), settings['batch_size']):
                take = indices[offset:offset + settings['batch_size']]
                opt.zero_grad()
                loss = F.cross_entropy(30 * head(x[take]), labels[take])
                loss = loss + settings['wd_weight'] * head.prototypes.square().sum(1).mean()
                if heads:
                    ps = torch.stack([head.prototypes] + [p.to(device) for p in heads], 1)
                    cov = torch.einsum('cjd,ckd->cjk', ps, ps) / (x.shape[1] - 1)
                    reg = settings['cov_reg'] / (1 + .05 * stage)
                    loss = loss + reg * cov[:, 0, 1:].sum(1).div(len(heads) + 1).mean().abs()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite head loss')
                loss.backward()
                opt.step()
        heads.append(head.prototypes.detach().cpu().clone())
        scales.append(head.scale.detach().cpu().clone())
        ns = np.random.get_state()
        state = dict(signature=signature, prototypes=heads, scales=scales,
                     torch_rng=torch.get_rng_state(), numpy_rng=(ns[0], ns[1].tolist(), *ns[2:]))
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = progress.with_suffix('.tmp')
        torch.save(state, temp)
        temp.replace(progress)
    result = dict(signature=signature, prototypes=torch.stack(heads), scales=torch.stack(scales),
                  selected_epoch=settings['epochs'], selection='fixed_final_epoch_no_groups')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(result, temp)
    temp.replace(path)
    return result


def predict_heads(heads, raw):
    x = F.normalize(standardized(raw), dim=-1)
    logits, distances = [], []
    with torch.inference_mode():
        for p, s in zip(heads['prototypes'], heads['scales']):
            d = torch.cdist(x, F.normalize(p, dim=-1), compute_mode='donot_use_mm_for_euclid_dist')
            distances.append(d.numpy())
            logits.append((-s.abs() * d).numpy())
    logits = np.stack(logits, 1)
    p = torch.from_numpy(logits).softmax(-1).numpy()
    return dict(probs=p, logits=logits, distances=np.stack(distances, 1), embedding=raw)


def prepare_meta(data, config, seed):
    """Fit heads before constructing OOF gate-training/selection rows."""
    root = Path(config['output_dir']) / f'seed{seed}'
    x, y, ids = data['embedding'], data['y'], data['sample_id']
    n = len(y)
    records, merged = [], None
    if config['protocol'] == 'crossfit':
        splits = list(StratifiedKFold(config['folds'], shuffle=True, random_state=seed).split(x, y))
        gate, select = train_test_split(np.arange(n), test_size=.2, stratify=y, random_state=seed + 100)
    else:
        proto, gate, select = partition(y, ids, seed)
        splits = [(proto, np.concatenate([gate, select]))]
    for fold, (train, held) in enumerate(splits):
        assert_disjoint(ids[train], ids[held])
        path = root / f'heads_fold{fold}.pt'
        heads = fit_heads(x[train], y[train], ids[train], config, seed * 10 + fold, path)
        provenance = dict(heads_sha256=digest(path), held_ids=ids[held].tolist(),
                          fit_ids=ids[train].tolist(), protocol=config['protocol'])
        cache = root / f'predictions_fold{fold}.npz'
        if cache.exists():
            pred = load_predictions(cache, provenance)
        else:
            pred = predict_heads(heads, x[held])
            save_predictions(cache, **pred, sample_id=ids[held], y=y[held], split='oof', provenance=provenance)
        if merged is None:
            merged = {k: np.zeros((n,) + v.shape[1:], dtype=v.dtype)
                      for k, v in pred.items() if k in ('probs', 'logits', 'distances', 'embedding')}
        for key in merged:
            merged[key][held] = pred[key]
        records.append(provenance)
    assert_disjoint(ids[gate], ids[select])
    write_json(root / 'split_audit.json', dict(folds=records, gate_train_ids=ids[gate].tolist(),
                gate_select_ids=ids[select].tolist(), groups_used_for_training=False))
    def subset(indices):
        return {k: v[indices] for k, v in merged.items()} | dict(y=y[indices], sample_id=ids[indices])
    # Group annotations deliberately do not leave this function in meta data.
    return subset(gate), subset(select), splits[0][0] if config['protocol'] != 'crossfit' else np.arange(n)


if __name__ == '__main__':
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='experiments/weighting/config.yaml')
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    args = parser.parse_args()
    c = yaml.safe_load(Path(args.config).read_text())
    seed_all(0)
    print(extract(c, args.split)['embedding'].shape)
