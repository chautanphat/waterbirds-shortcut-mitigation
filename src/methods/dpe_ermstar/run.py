"""400-epoch ERM*; upstream DPE head routines; matched neural aggregation; GPU-per-seed workers."""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import yaml

from src.methods.dpe_head_reweighting import ensemble, neural_combiner as neural
from src.methods.dpe_head_reweighting.experiment import read_json, write_json
from src.methods.dpe_head_reweighting.aggregation_source import partition
from src.methods.dpe_head_reweighting.reweighting import metrics

ROOT=Path(__file__).resolve().parents[3]
DEFAULT=ROOT/'configs/dpe_ermstar/study.json'


def original_functions():
    from .upstream import (IsoMaxPlusLossFirstPart, IsoMaxPlusLossSecondPart,
                           subsample, train_prototypes)
    return dict(IsoMaxPlusLossFirstPart=IsoMaxPlusLossFirstPart,
                IsoMaxPlusLossSecondPart=IsoMaxPlusLossSecondPart,
                subsample=subsample, train_prototypes=train_prototypes)


def atomic_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent);os.close(fd)
    try:torch.save(value,tmp);os.replace(tmp,path)
    finally:Path(tmp).unlink(missing_ok=True)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def head_dataset(x,y,g):
    n=len(y);zero=torch.zeros(n,dtype=torch.long)
    dataset=TensorDataset(torch.arange(n),zero,torch.as_tensor(y,dtype=torch.long),zero,zero,x)
    dataset.groups=np.asarray(g)
    return dataset


def train_heads(raw,rows,path,seed,settings,device):
    """Preserved loss/trainer/subsampler; head-boundary optimizer-independent recovery."""
    functions=original_functions();seed_all(seed)
    x=ensemble.standardize(raw);y=np.array([r['y'] for r in rows]);g=np.array([r['group'] for r in rows])
    dataset=head_dataset(x,y,g);validation=DataLoader(dataset,batch_size=256,shuffle=False)
    inputs=dict(seed=seed,settings=settings,rows=rows,feature_shape=list(raw.shape))
    saved=[]
    if Path(path).exists():
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint['inputs']!=inputs or not torch.equal(checkpoint['raw_features'],torch.from_numpy(raw)):
            raise ValueError('Head checkpoint inputs/configuration changed')
        saved=checkpoint['heads'];npstate=checkpoint['numpy_rng']
        np.random.set_state((npstate[0],np.array(npstate[1],dtype='uint32'),*npstate[2:]))
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.startswith('cuda'):torch.cuda.set_rng_state(checkpoint['cuda_rng'])
    for stage in range(len(saved)+1,settings['heads']+1):
        frame=pd.DataFrame(dict(y=y,subgroup=g,split=1))
        sampled=functions['subsample'](frame,target='subgroup',target_split=1)
        indices=sampled.index.to_numpy()
        if len(indices)<settings['batch_size']:raise ValueError('Group-balanced pool produces zero optimizer batches')
        train=DataLoader(head_dataset(x[indices],y[indices],g[indices]),batch_size=settings['batch_size'],shuffle=True,drop_last=True)
        model=nn.Sequential(functions['IsoMaxPlusLossFirstPart'](x.shape[1],2)).to(device)
        for parameter in model.parameters():
            def finite_gradient(grad):
                if not torch.isfinite(grad).all():raise RuntimeError('Nonfinite upstream DPE gradient')
                return grad
            parameter.register_hook(finite_gradient)
        head=functions['train_prototypes'](train,validation,model,prototype_ensemble=[(p.to(device),s.to(device)) for p,s in saved],
            epochs=settings['epochs'],cov_reg=settings['cov_reg'],wd_weight=settings['wd_weight'],device=device,
            entropic=settings['entropic_scale'],lr=settings['lr'],stage=stage,verbose=False)
        if head is None or any(not torch.isfinite(value).all() for value in head):raise RuntimeError('Invalid upstream selected head')
        saved.append([value.detach().cpu() for value in head])
        state=np.random.get_state()
        atomic_torch(path,dict(inputs=inputs,raw_features=torch.from_numpy(raw.copy()),heads=saved,numpy_rng=(state[0],state[1].tolist(),*state[2:]),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if device.startswith('cuda') else None))
        print(f'Completed head {stage}/{settings["heads"]}',flush=True)
    return dict(version=1,prototypes=torch.stack([p[:,0] for p,s in saved]),distance_scales=torch.stack([s.reshape(()) for p,s in saved]),
        temperatures=torch.ones(len(saved)),head_ids=[f'head-{i:02d}' for i in range(len(saved))],selected_epochs=[-1]*len(saved),
        classes=[0,1],groups=['00','01','10','11'],preprocessing=ensemble.PREPROCESSING,provenance=dict(seed=seed,training_pool_ids=[r['id'] for r in rows],settings=settings))


def backbone_config(data,out,seed,config,device):
    from src.methods.erm.config import load_config
    c=load_config(ROOT/'configs/erm/waterbirds_enhanced.yaml')
    c.training.epochs=config['backbone_epochs'];c.training.seed=seed;c.training.device=device
    c.dataset.data_dir=str(data);c.dataset.workers=config['workers'];c.dataset.batch_size_eval=512
    c.output_dir=str(out)
    return c.validate()


def features(data,backbone,out,c,device):
    from src.datasets.waterbirds import Waterbirds
    from src.datasets.loaders import evaluation_loader
    from src.evaluation.features import extract_features
    from src.models.resnet import load_resnet_checkpoint
    identity=dict(backbone=str(backbone),recipe=c.as_dict())
    if out.exists():
        saved=read_json(out/'complete.json')
        if saved['identity']!=identity:raise ValueError('Raw feature identity differs')
        for name in ('val.npz','test.npz'):
            if not (out/name).is_file():raise ValueError('Incomplete raw feature cache')
    else:
        out.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out.parent) as temp:
            stage=Path(temp)/'features';stage.mkdir()
            model=load_resnet_checkpoint(backbone,device=device)
            for split in ('val','test'):
                ds=Waterbirds(data,split);loader=evaluation_loader(ds,512,c.dataset.workers)
                values=extract_features(model,loader,device)['features']
                np.savez_compressed(stage/f'{split}.npz',features=values,ids=np.asarray(ds.filenames),y=ds.y,g=np.array([f'{y}{a}' for y,a in zip(ds.y,ds.a)]))
            write_json(stage/'complete.json',dict(identity=identity,files=['val.npz','test.npz']))
            os.replace(stage,out)
    values={}
    for split in ('val','test'):
        with np.load(out/f'{split}.npz',allow_pickle=False) as f:values[split]=dict(f)
    return values


def archive_seed(seed_dir):
    target=seed_dir.parent/(seed_dir.name+'-results.zip');temporary=target.with_suffix('.part')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        for path in seed_dir.rglob('*'):
            if path.is_file() and 'backbone' not in path.relative_to(seed_dir).parts and 'features' not in path.relative_to(seed_dir).parts and path.name!='head_progress.pt':
                z.write(path,path.relative_to(seed_dir.parent))
    os.replace(temporary,target)


def worker(work,data,seed,config,device='cuda:0'):
    from src.methods.erm.train import run_training
    work=Path(work);out=work/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if config['backbone_epochs']!=400:raise ValueError('This experiment requires the upstream 400-epoch ERM* recipe')
    torch.set_num_threads(config['cpu_threads'])
    c=backbone_config(data,out/'backbone',seed,config,device)
    settings=dict(config=config,backbone=c.as_dict(),implementation='src/methods/dpe_ermstar/run.py')
    if (out/'plan.json').exists() and read_json(out/'plan.json')!=settings:raise ValueError('Seed plan changed; preserve old run')
    write_json(out/'plan.json',settings)
    summary=out/'backbone/summary.json'
    if not summary.exists():run_training(c,resume=(out/'backbone/training_state.pt').exists())
    saved=read_json(summary)
    if saved['updates']!=37*config['backbone_epochs'] or saved['selected_checkpoint']!='ckpt_last.pt':raise ValueError('Backbone completion does not match 400-epoch recipe')
    values=features(data,out/'backbone/ckpt_last.pt',out/'features',c,device)
    val=values['val'];rows=[dict(id=str(i),y=int(y),group=str(g),split=1) for i,y,g in zip(val['ids'],val['y'],val['g'])]
    parts=partition(rows);lookup={r['id']:i for i,r in enumerate(rows)}
    record={}
    for arm in ('paper_reference','neural_split'):
        arm_dir=out/arm;arm_dir.mkdir(exist_ok=True)
        pool=rows if arm=='paper_reference' else parts['head_training']
        index=[lookup[r['id']] for r in pool]
        heads=train_heads(val['features'][index],pool,arm_dir/'head_progress.pt',seed,config['head_training'],device)
        ensemble.save(arm_dir/'ensemble.pt',heads)
        if arm=='neural_split':
            fitrows=parts['combiner_training'];index=[lookup[r['id']] for r in fitrows]
            probs=ensemble.predict(heads,val['features'][index],device=device)
            np.savez_compressed(arm_dir/'combiner_training.npz',probabilities=probs,ids=np.array([r['id'] for r in fitrows]),labels=np.array([r['y'] for r in fitrows]),groups=np.array([r['group'] for r in fitrows]))
            fitted=neural.fit_all(probs,np.array([r['y'] for r in fitrows]),np.array([r['group'] for r in fitrows]),config['neural'],seed)
            write_json(arm_dir/'models.json',fitted)
            write_json(arm_dir/'partition.json',parts)
        record[arm]=heads
    # Both arms and all neural models freeze before test scoring.
    write_json(out/'fit_complete.json',dict(models='neural_split/models.json',heads={k:k+'/ensemble.pt' for k in record}))
    test=values['test'];result=dict(seed=seed,status='exploratory',results={})
    for arm,heads in record.items():
        probs=ensemble.predict(heads,test['features'],device=device)
        np.savez_compressed(out/arm/'test_heads.npz',probabilities=probs,ids=test['ids'],labels=test['y'],groups=test['g'])
        candidates=([dict(id='uniform',family='uniform')] if arm=='paper_reference' else fitted['candidates'])
        uniform=metrics(probs.astype(float).mean(0),test['y'],test['g'])
        for candidate in candidates:
            q,_=neural.candidate_prediction(candidate,probs)
            score=metrics(q,test['y'],test['g'])
            result['results'][arm+'/'+candidate['id']]=dict(metrics=score,delta_matched_uniform_pp=100*(score['wga']-uniform['wga']))
    write_json(out/'results.json',result);archive_seed(out)
    return result


def launch(work,data,seeds,config,gpus=None,timeout_hours=10):
    """One visible GPU per process; shared data preparation occurs before this call."""
    if not seeds or len(set(seeds))!=len(seeds) or any(s not in (0,1,2) for s in seeds):raise ValueError('Choose distinct seeds 0,1,2')
    gpus=list(range(torch.cuda.device_count())) if gpus is None else list(gpus)
    if not gpus or len(set(gpus))!=len(gpus):raise RuntimeError('Select a GPU runtime; distinct GPU indices required')
    if timeout_hours<=0:raise ValueError('Positive timeout required')
    work=Path(work).resolve();work.mkdir(parents=True,exist_ok=True)
    conf=work/'study.json';write_json(conf,config)
    pending=list(seeds);running={};errors=[];started=time.monotonic()
    def stop():
        for process,handle,seed in running.values():
            if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
        for process,handle,seed in running.values():
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            handle.close()
    try:
        while pending or running:
            for gpu in gpus:
                if gpu in running or not pending:continue
                seed=pending.pop(0);handle=(work/f'seed{seed}.log').open('a')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS=str(config['cpu_threads']),MKL_NUM_THREADS=str(config['cpu_threads']))
                command=[sys.executable,'-m','src.methods.dpe_ermstar.run','--worker','--work',str(work),'--data',str(data),'--seed',str(seed),'--config',str(conf)]
                try:process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                except Exception:handle.close();raise
                running[gpu]=(process,handle,seed);print(f'Seed {seed} -> GPU {gpu}; {work}/seed{seed}.log',flush=True)
            for gpu,(process,handle,seed) in list(running.items()):
                code=process.poll()
                if code is None:continue
                handle.close();del running[gpu]
                if code or not (work/f'seed{seed}/results.json').exists():errors.append(dict(seed=seed,returncode=code))
            if time.monotonic()-started>timeout_hours*3600:raise TimeoutError('Session budget reached. Resume saved epoch/head checkpoints next session; incomplete seeds are not results.')
            if running:
                print('Active:',{gpu:seed for gpu,(_,_,seed) in running.items()},flush=True)
                time.sleep(30)
    except (KeyboardInterrupt,TimeoutError) as exc:
        write_json(work/'completion.json',dict(planned=seeds,status='interrupted',reason=type(exc).__name__,active_seeds=[seed for _,_,seed in running.values()],pending_seeds=pending))
        raise
    finally:stop()
    failed={e['seed'] for e in errors}
    complete=[read_json(work/f'seed{s}/results.json') for s in seeds if s not in failed and (work/f'seed{s}/results.json').exists()]
    write_json(work/'completion.json',dict(planned=seeds,completed=[r['seed'] for r in complete],errors=errors))
    if errors:raise RuntimeError(f'Seed workers failed: {errors}; successful outputs retained. Inspect per-seed logs.')
    return complete

def prepare_and_launch(work, *, seeds=(0, 1), data_dir=None, resume_from=None,
                       config_path=DEFAULT, timeout_hours=10):
    """Prepare shared inputs once, restore missing checkpoints, then start seed workers."""
    import shutil
    import torchvision
    from src.methods.dpe_head_reweighting.study import ensure_dataset

    if torch.cuda.device_count() < 2:
        raise RuntimeError('Select two GPUs before preparing this experiment')
    if not seeds or len(set(seeds)) != len(seeds) or any(s not in (0, 1, 2) for s in seeds):
        raise ValueError('Choose distinct source seeds 0, 1, 2')
    work = Path(work)
    jobs = work / 'jobs'
    jobs.mkdir(parents=True, exist_ok=True)
    if resume_from is not None:
        prior = Path(resume_from).resolve()
        if prior == jobs.resolve() or not prior.is_dir():
            raise ValueError('resume_from must identify a prior full jobs directory')
        needed = sum(p.stat().st_size for p in prior.rglob('*')
                     if p.is_file() and not (jobs / p.relative_to(prior)).exists())
        if shutil.disk_usage(jobs).free < needed + 512 * 1024**2:
            raise RuntimeError('Insufficient disk to restore checkpoints')
        for path in prior.rglob('*'):
            if not path.is_file():
                continue
            target = jobs / path.relative_to(prior)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.restore-part')
            shutil.copyfile(path, temporary)
            temporary.replace(target)
    data = ensure_dataset(work, [Path('/kaggle/input')], data_dir, True)
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True)
    del weights
    config = read_json(config_path)
    print('Configuration:', config)
    return launch(jobs, data, list(seeds), config, gpus=[0, 1], timeout_hours=timeout_hours)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--config', type=Path, default=DEFAULT)
    args = parser.parse_args()
    worker(args.work, args.data, args.seed, read_json(args.config))
