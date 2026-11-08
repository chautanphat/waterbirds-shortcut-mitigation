"""Execute actual reproduction-notebook functions with small CPU fixtures."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT=Path(__file__).resolve().parents[3]
NB=json.loads((ROOT/'notebooks/dpe_waterbirds_end_to_end.ipynb').read_text())
spec=importlib.util.spec_from_file_location('ref_head',ROOT/'baselines/dpe4subpop/utils/isomaxplus.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)


def environment():
    def metric(p,y,a,g):
        correct=p.argmax(1)==y
        accuracy=float(correct.mean())
        return {'overall':{'accuracy':accuracy}, 'min_group':{'accuracy':accuracy}, 'probabilities':p}
    env=dict(np=np,torch=torch,nn=nn,DataLoader=DataLoader,Path=Path,json=json,hashlib=hashlib,
             os=__import__('os'),eval_metrics=metric,get_acc=lambda *a,**kw:None,
             tqdm=lambda x,**kw:x,IsoMaxPlusLossFirstPart=ref.IsoMaxPlusLossFirstPart,
             IsoMaxPlusLossSecondPart=ref.IsoMaxPlusLossSecondPart)
    for cell in NB['cells']:
        if cell['cell_type']!='code':continue
        source=''.join(cell['source'])
        try:tree=ast.parse(source)
        except SyntaxError:continue  # Setup cells contain IPython magics.
        funcs=[node for node in tree.body if isinstance(node,ast.FunctionDef)]
        exec(compile(ast.Module(body=funcs,type_ignores=[]),'<notebook>','exec'),env)
    return env


class Features(Dataset):
    def __init__(self,x):
        self.x=x;self.y=[0]*len(x);self._a=[0]*len(x);self.g=['00']*len(x);self.num_labels=2
    def __len__(self):return len(self.x)
    def __getitem__(self,i):return i,torch.zeros(1),self.y[i],0,0,self.x[i]


def test_erm_uses_raw_features_with_singleton_last_batch():
    env=environment()
    raw=np.tile([10.,9.,9.,9.],(5,1)).astype('float32')
    normalized=(raw-raw.mean(1,keepdims=True))/raw.std(1,keepdims=True)
    ds=Features(torch.from_numpy(normalized));loader=DataLoader(ds,batch_size=2)
    model=nn.Sequential(nn.Linear(4,2))
    with torch.no_grad():
        model[0].weight.zero_();model[0].bias.copy_(torch.tensor([-5.,0.]));model[0].weight[0,0]=1
    assert env['evaluate'](model,loader,device='cpu')['overall']['accuracy']==0
    fixed=env['evaluate'](model,loader,device='cpu',raw_features=raw)
    assert fixed['overall']['accuracy']==1
    with torch.inference_mode():expected=model(torch.from_numpy(raw)).softmax(1).numpy()
    np.testing.assert_allclose(fixed['probabilities'],expected,atol=1e-7)
    with pytest.raises(ValueError,match='metadata-order'):
        env['evaluate'](model,DataLoader(ds,batch_size=2,shuffle=True),device='cpu',raw_features=raw)
    with pytest.raises(ValueError,match='matching'):
        env['evaluate'](model,loader,device='cpu',raw_features=raw[:-1])


def test_prototype_inference_preserves_singleton_batch():
    env=environment()
    x=torch.randn(5,4);ds=Features(x);loader=DataLoader(ds,batch_size=2)
    model=nn.Sequential(nn.Identity(),nn.Flatten(1),ref.IsoMaxPlusLossFirstPart(4,2))
    prototypes=[(torch.randn(2,1,4),torch.tensor([.7])),(torch.randn(2,1,4),torch.tensor([1.3]))]
    expected=[]
    for p,s in prototypes:
        h=ref.IsoMaxPlusLossFirstPart(4,2)
        with torch.no_grad():h.prototypes.copy_(p[:,0]);h.distance_scale.copy_(s);expected.append(h(x).softmax(1))
    actual=env['evaluate_ensemble'](prototypes,loader,model,device='cpu',verbose=False)
    np.testing.assert_allclose(actual['probabilities'],torch.stack(expected).mean(0).numpy(),atol=1e-7)


def test_cache_must_match_checkpoint_metadata_and_feature_bytes(tmp_path):
    env=environment();out=tmp_path/'checkpoint';out.mkdir()
    (out/'ckpt_last.pt').write_bytes(b'checkpoint1')
    data=tmp_path/'data/waterbirds';data.mkdir(parents=True)
    metadata=data/'metadata_waterbirds.csv';metadata.write_text('metadata1')
    class CacheDataset:
        x=['image1','image2']
        def __len__(self):return 2
    raw=np.random.RandomState(0).normal(size=(2,2048)).astype('float32')
    calls=[]
    def extract(model,dataset,path,**kw):calls.append(1);np.save(path,raw);return raw
    env.update(data_dir=str(tmp_path/'data'),datasets={'test':CacheDataset()},device='cpu',
               batch_size_eval=2,workers=0,init_erm_model=lambda *a,**kw:None,
               extract_and_save_features=extract)
    # Legacy cache without provenance is regenerated once, then reused.
    np.save(out/'feats_test.npy',raw)
    env['get_pre_extracted_features'](str(out),'test');assert len(calls)==1
    env['get_pre_extracted_features'](str(out),'test');assert len(calls)==1
    for path in [out/'ckpt_last.pt',metadata,out/'feats_test.npy']:
        path.write_bytes(path.read_bytes()+b'changed')
        env['get_pre_extracted_features'](str(out),'test')
    assert len(calls)==4
    (out/'feature_manifest_test.json').write_text('{broken')
    env['get_pre_extracted_features'](str(out),'test')
    assert len(calls)==5


def test_reproduction_uses_annotated_paper_column_and_valid_parser_flags():
    import yaml
    stage0=yaml.safe_load((ROOT/'configs/dpe4subpop/waterbirds_stage0_erm.yaml').read_text())
    stage1=yaml.safe_load((ROOT/'configs/dpe4subpop/waterbirds_stage1_dpe.yaml').read_text())
    assert stage0['dataset']['data_dir']==stage1['dataset']['data_dir']=='data'
    assert stage0['training']['optim_weight_decay']==0
    assert stage1['dataset']['train_attr']=='yes'
    assert '91.0% ± 0.4%' in ''.join(NB['cells'][-1]['source'])
    readme=(ROOT/'baselines/dpe4subpop/README.md').read_text()
    import re
    commands='\n'.join(re.findall(r'```bash\n(.*?)```',readme,re.S))
    assert '--subsample_type class' not in commands
    assert '--data_dir data/waterbirds' not in commands


def test_partial_checkpoint_load_rejects_unexpected_backbone_keys(tmp_path):
    env=environment()
    class TinyNetwork(nn.Sequential):
        def __init__(self):
            super().__init__(nn.Linear(4,4),nn.Linear(4,2))
            self.fc=self[1]
    env['torchvision']=SimpleNamespace(models=SimpleNamespace(resnet50=TinyNetwork))
    # Build the exact layout produced from this network's children.
    backbone=nn.Sequential(*list(TinyNetwork().children())[:-1])
    ce=nn.Sequential(backbone,nn.Flatten(),nn.Linear(4,2))
    path=tmp_path/'ce.pt';torch.save(ce.state_dict(),path)
    model=env['init_model'](str(path),device='cpu')
    assert isinstance(model[-1],ref.IsoMaxPlusLossFirstPart)
    state=ce.state_dict();del state['0.0.weight'];torch.save(state,path)
    with pytest.raises(ValueError,match='backbone checkpoint mismatch'):
        env['init_model'](str(path),device='cpu')


def test_prototype_loss_stays_finite_when_softmax_would_underflow():
    env=environment()
    class Progress:
        def __init__(self,loader,**kwargs):self.loader=loader
        def __iter__(self):return iter(self.loader)
        def set_postfix(self,*args,**kwargs):pass
    env['tqdm']=Progress
    x=torch.tensor([[-1.,0.,0.,0.]]*4)
    ds=Features(x);loader=DataLoader(ds,batch_size=2)
    head=ref.IsoMaxPlusLossFirstPart(4,2)
    with torch.no_grad():head.prototypes.copy_(torch.tensor([[1.,0.,0.,0.],[-1.,0.,0.,0.]]))
    model=nn.Sequential(nn.Identity(),nn.Flatten(1),head)
    # The old log(softmax) expression is infinite for these incorrect predictions.
    assert torch.isinf(ref.IsoMaxPlusLossSecondPart(entropic_scale=1e6)(head(x),torch.zeros(4,dtype=torch.long)))
    best=env['train_prototypes'](loader,loader,model,epochs=1,device='cpu',entropic=1e6,lr=0)
    assert best is not None and all(torch.isfinite(tensor).all() for tensor in best)


def test_stage0_reuse_requires_successful_matching_completion(tmp_path, monkeypatch):
    env=environment()
    # Execute the actual Stage-0 orchestration; substitute only the long trainer.
    source=''.join(next(c['source'] for c in NB['cells'] if 'stage0_command =' in ''.join(c.get('source',[]))))
    monkeypatch.chdir(tmp_path)
    metadata=tmp_path/'data/waterbirds/metadata_waterbirds.csv'
    metadata.parent.mkdir(parents=True);metadata.write_text('metadata')
    import subprocess
    calls=[]
    def complete(command,check):
        calls.append(command)
        (tmp_path/'checkpoints/waterbirds_stage0_standard_erm/ckpt_last.pt').write_bytes(b'completed fixture')
    monkeypatch.setattr(subprocess,'run',complete)
    exec(source,env)
    assert len(calls)==1
    marker=tmp_path/'checkpoints/waterbirds_stage0_standard_erm/stage0_complete.json'
    assert marker.is_file()
    exec(source,env)  # No feature caches are present: reuse the completed backbone.
    assert len(calls)==1
    metadata.write_text('modified metadata')
    with pytest.raises(ValueError,match='completion record/configuration changed'):
        exec(source,env)
    metadata.write_text('metadata')
    marker.unlink()
    with pytest.raises(ValueError,match='lacks a completion record'):
        exec(source,env)
    assert len(calls)==1
    # An interrupted trainer must never publish a successful completion record.
    checkpoint=marker.parent/'ckpt_last.pt';checkpoint.unlink()
    def interrupted(command,check):
        checkpoint.write_bytes(b'partial checkpoint')
        raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(subprocess,'run',interrupted)
    with pytest.raises(subprocess.CalledProcessError):exec(source,env)
    assert not marker.exists()
