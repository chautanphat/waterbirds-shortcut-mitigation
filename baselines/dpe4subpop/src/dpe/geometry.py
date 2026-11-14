"""Verified opt-in G1 inference; uniform DPE remains the default elsewhere."""
import argparse
import json
from pathlib import Path
import numpy as np


def read_metadata(value=None):
    if value is None:value=Path(__file__).with_name('verified_distance_v1.json')
    metadata=dict(value) if isinstance(value,dict) else json.loads(Path(value).read_text(encoding='utf-8'))
    if (metadata.get('schema_version')!=1 or metadata.get('mode')!='distance'
        or metadata.get('members')!=15 or metadata.get('class_order')!=[0,1]
        or metadata.get('gamma')!=50. or metadata.get('alpha')!=.25):
        raise ValueError('Unsupported geometry metadata; only the frozen verified G1 gamma50/alpha.25 recipe is enabled')
    return metadata


def distance_mix(probabilities,distances,metadata=None):
    """Use normalized Euclidean distances, not learned-scale logits or labels."""
    meta=read_metadata(metadata);p=np.asarray(probabilities);d=np.asarray(distances)
    if p.shape!=d.shape or p.ndim!=3 or p.shape[1:]!=(15,2):raise ValueError('Expected [samples,15,2] probabilities and normalized distances')
    if (not np.isfinite(p).all() or not np.isfinite(d).all() or (p<0).any() or (d<0).any()
        or (d>2+2e-6).any() or not np.allclose(p.sum(-1),1,atol=2e-6)):
        raise ValueError('Invalid probabilities or normalized Euclidean distances')
    score=-meta['gamma']*d.min(-1)
    r=np.exp(score-score.max(1,keepdims=True));r=r/r.sum(1,keepdims=True)
    weights=(1-meta['alpha'])/15+meta['alpha']*r
    mixture=(p*weights[...,None]).sum(1)
    if not np.isfinite(mixture).all() or not np.allclose(mixture.sum(1),1,atol=2e-6):raise ValueError('Mixture probability mass failed')
    return mixture,weights


def main():
    parser=argparse.ArgumentParser(description='Frozen DPE prediction-cache inference; never fits or reads labels/groups.')
    parser.add_argument('--input-cache',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--aggregation',choices=['uniform','distance'],default='uniform');parser.add_argument('--metadata',type=Path)
    args=parser.parse_args()
    with np.load(args.input_cache,allow_pickle=False) as f:
        prefix='C_' if 'C_probs' in f else ''
        probs=f[prefix+'probs'];ids=f['C_ids'] if prefix else f['sample_id']
        if args.aggregation=='distance':mixed,weights=distance_mix(probs,f[prefix+'distances'],args.metadata)
        else:mixed=probs.mean(1);weights=np.full(probs.shape[:2],1/15)
    if args.output.exists():raise FileExistsError('Output already exists; choose a new path')
    args.output.parent.mkdir(parents=True,exist_ok=True);tmp=args.output.with_name(args.output.name+'.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,probs=mixed,weights=weights,sample_id=ids,aggregation=np.array(args.aggregation),class_order=np.array([0,1]))
        f.flush();__import__('os').fsync(f.fileno())
    __import__('os').replace(tmp,args.output)


if __name__=='__main__':main()
