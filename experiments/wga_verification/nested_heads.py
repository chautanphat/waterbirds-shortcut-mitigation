"""Epoch-resumable class-balanced DPE heads; same math as prior OOF adapter."""
import argparse
import io
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from experiments.weighting.data import PrototypeHead,standardized,seed_all
from experiments.discrete_weighting.store import atomic,key

SETTINGS=dict(members=15,epochs=20,batch_size=256,lr=.001,cov_reg=500000.,wd_weight=10.)


def save(path,obj):
    b=io.BytesIO();torch.save(obj,b);atomic(path,b.getvalue())


def rng():
    n=np.random.get_state()
    return dict(torch=torch.get_rng_state(),numpy=(n[0],n[1].tolist(),*n[2:]))


def restore(state):
    torch.set_rng_state(state['torch']);n=state['numpy'];np.random.set_state((n[0],np.array(n[1],dtype='uint32'),*n[2:]))


def fit_member(raw,y,ids,previous,seed,stage,folder,pause=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    signature=key(dict(ids=list(map(str,ids)),y=list(map(int,y)),seed=seed,stage=stage,settings=SETTINGS,
                       raw=__import__('hashlib').sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
                       previous=[__import__('hashlib').sha256(p['prototypes'].numpy().tobytes()).hexdigest() for p in previous]))
    final=folder/'head.pt';latest=folder/'latest.pt'
    seed_all(seed)
    if final.exists():
        r=torch.load(final,map_location='cpu',weights_only=True)
        if r['signature']!=signature:raise ValueError('Head signature mismatch')
        return r
    if previous:restore(previous[-1]['rng'])
    x=standardized(raw);labels=torch.tensor(y,dtype=torch.long)
    pools=[np.flatnonzero(y==c) for c in (0,1)];size=min(map(len,pools))
    chosen=np.stack([np.random.permutation(p)[:size] for p in pools],1).ravel()
    head=PrototypeHead(x.shape[1],2);opt=torch.optim.SGD(head.parameters(),lr=.001,momentum=.9);first=0
    if latest.exists():
        s=torch.load(latest,map_location='cpu',weights_only=True)
        if s['signature']!=signature:raise ValueError('Epoch signature mismatch')
        head.load_state_dict(s['model']);opt.load_state_dict(s['optimizer']);chosen=np.array(s['chosen']);restore(s['rng']);first=s['epoch']
    resumed_from=first
    for epoch in range(first,20):
        indices=np.random.permutation(chosen)
        for offset in range(0,len(indices),256):
            take=indices[offset:offset+256];opt.zero_grad()
            loss=F.cross_entropy(30*head(x[take]),labels[take])+10*head.prototypes.square().sum(1).mean()
            if previous:
                ps=torch.stack([head.prototypes]+[p['prototypes'] for p in previous],1)
                cov=torch.einsum('cjd,ckd->cjk',ps,ps)/(x.shape[1]-1)
                loss=loss+(500000/(1+.05*stage))*cov[:,0,1:].sum(1).div(len(previous)+1).mean().abs()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite head objective')
            loss.backward();opt.step()
        save(latest,dict(signature=signature,model=head.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,chosen=chosen.tolist(),rng=rng()))
        if pause and epoch==0:
            atomic(folder/'pause.ready','epoch 1 committed');time.sleep(120)
    result=dict(signature=signature,prototypes=head.prototypes.detach().clone(),scale=head.scale.detach().clone(),rng=rng(),
                selected_epoch=20,resumed_from_epoch=resumed_from,fit_ids=list(map(str,ids)),member_id=f'head-{stage:02d}')
    save(final,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pool',required=True);p.add_argument('--folder',required=True);p.add_argument('--pause',action='store_true');a=p.parse_args()
    with np.load(a.pool,allow_pickle=False) as f:fit_member(f['embedding'],f['y'],f['sample_id'],[],1000,0,a.folder,a.pause)
