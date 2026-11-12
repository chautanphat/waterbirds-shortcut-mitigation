"""No CNN training: original synthetic head trainer, process mocks and full worker orchestration."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from src.methods.dpe_ermstar import run as r
from src.methods.dpe_head_reweighting.experiment import read_json,write_json

torch.set_num_threads(1)

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.config=read_json(r.DEFAULT)
        rng=np.random.RandomState(3)
        self.raw=rng.randn(32,8).astype('float32')
        self.rows=[dict(id=f'val-{g}-{i}',y=int(g[0]),group=g,split=1) for g in ('00','01','10','11') for i in range(8)]
        self.small=dict(self.config['head_training'],heads=2,epochs=1,batch_size=8)
    def tearDown(self):self.temp.cleanup()
    def test_recipe_matches_launcher_and_neural_config(self):
        self.assertEqual(self.config['backbone_epochs'],400)
        self.assertIn('--epochs 400',(r.ROOT/'baselines/dpe4subpop/scripts/train_all.sh').read_text())
        self.assertIn('--cov_reg 5.e5',(r.ROOT/'baselines/dpe4subpop/scripts/train_all_pe.sh').read_text())
        c=r.backbone_config('data',self.root,0,self.config,'cpu')
        self.assertEqual((c.training.epochs,c.training.lr,c.training.weight_decay,c.dataset.augmentation),(400,.001,0.,True))
        self.assertNotIn('weighting_rules',self.config['neural'])
    def test_upstream_head_resume_identical_after_interruption(self):
        original=r.original_functions
        functions=original();train=functions['train_prototypes'];calls=[]
        def fail(*args,**kwargs):
            calls.append(kwargs['stage'])
            if len(calls)==2:raise RuntimeError('interruption')
            return train(*args,**kwargs)
        functions['train_prototypes']=fail
        path=self.root/'head_progress.pt'
        with patch.object(r,'original_functions',return_value=functions):
            with self.assertRaisesRegex(RuntimeError,'interruption'):r.train_heads(self.raw,self.rows,path,0,self.small,'cpu')
        resumed=r.train_heads(self.raw,self.rows,path,0,self.small,'cpu')
        fresh=r.train_heads(self.raw,self.rows,self.root/'fresh.pt',0,self.small,'cpu')
        self.assertTrue(torch.equal(resumed['prototypes'],fresh['prototypes']))
        self.assertTrue(torch.equal(resumed['distance_scales'],fresh['distance_scales']))
        with self.assertRaisesRegex(ValueError,'changed'):
            r.train_heads(self.raw+1,self.rows,path,0,self.small,'cpu')
    def test_small_pool_fails_before_empty_training(self):
        with self.assertRaisesRegex(ValueError,'zero optimizer'):
            r.train_heads(self.raw,self.rows,self.root/'head.pt',0,dict(self.small,batch_size=256),'cpu')
    def test_complete_worker_two_arms_no_cnn_training(self):
        from src.methods.erm import train
        config=copy.deepcopy(self.config);config['head_training'].update(epochs=1,batch_size=8)
        config['neural']['steps']=2
        def backbone(c,resume=False):
            path=Path(c.output_dir);path.mkdir(parents=True,exist_ok=True)
            (path/'ckpt_last.pt').write_bytes(b'no CNN training')
            write_json(path/'summary.json',dict(updates=37*400,selected_checkpoint='ckpt_last.pt'))
        val=dict(features=self.raw,ids=np.array([v['id'] for v in self.rows]),y=np.array([v['y'] for v in self.rows]),g=np.array([v['group'] for v in self.rows]))
        test=copy.deepcopy(val);test['ids']=np.array([f'test-{i}' for i in range(32)])
        with patch.object(train,'run_training',side_effect=backbone),patch.object(r,'features',return_value=dict(val=val,test=test)):
            result=r.worker(self.root,'unused',0,config,'cpu')
            repeated=r.worker(self.root,'unused',0,config,'cpu')
        self.assertEqual(result,repeated)
        self.assertEqual(len(result['results']),12)
        self.assertTrue((self.root/'seed0-results.zip').is_file())
        self.assertTrue((self.root/'seed0/fit_complete.json').exists())
        self.assertTrue((self.root/'seed0/neural_split/combiner_training.npz').exists())
    def scheduler(self,fail=False):
        launched=[]
        class Process:
            pid=123456
            def __init__(self,seed):self.seed=seed;self.polls=0
            def poll(self):
                self.polls+=1
                return None if self.polls==1 else (1 if fail and self.seed==0 else 0)
            def wait(self,timeout=None):return 0
        def popen(command,**kwargs):
            seed=int(command[command.index('--seed')+1]);launched.append((command,kwargs))
            if not (fail and seed==0):write_json(self.root/f'seed{seed}/results.json',dict(seed=seed))
            return Process(seed)
        with patch.object(r.subprocess,'Popen',side_effect=popen),patch.object(r.time,'sleep'),patch.object(r.os,'killpg'):
            if fail:
                with self.assertRaisesRegex(RuntimeError,'workers failed'):r.launch(self.root,'data',[0,1],self.config,gpus=[0,1])
            else:r.launch(self.root,'data',[0,1],self.config,gpus=[0,1])
        self.assertEqual([kwargs['env']['CUDA_VISIBLE_DEVICES'] for _,kwargs in launched],['0','1'])
        self.assertTrue(all(kwargs['start_new_session'] for _,kwargs in launched))
    def test_dual_gpu_isolation(self):self.scheduler()
    def test_failed_worker_retains_other_seed(self):
        self.scheduler(fail=True)
        self.assertEqual(read_json(self.root/'completion.json')['completed'],[1])
    def test_interrupt_terminates_both_worker_groups(self):
        class Process:
            def __init__(self,pid):self.pid=pid
            def poll(self):return None
            def wait(self,timeout=None):return 0
        with patch.object(r.subprocess,'Popen',side_effect=[Process(910001),Process(910002)]),patch.object(r.time,'sleep',side_effect=KeyboardInterrupt),patch.object(r.os,'killpg') as kill:
            with self.assertRaises(KeyboardInterrupt):r.launch(self.root,'data',[0,1],self.config,gpus=[0,1])
        self.assertEqual(kill.call_count,2)
        self.assertEqual(read_json(self.root/'completion.json')['status'],'interrupted')

    def test_timeout_preserves_recovery_status(self):
        class Process:
            pid=910003
            def poll(self):return None
            def wait(self,timeout=None):return 0
        with patch.object(r.subprocess,'Popen',return_value=Process()),patch.object(r.time,'monotonic',side_effect=[0,10]),patch.object(r.os,'killpg'):
            with self.assertRaises(TimeoutError):r.launch(self.root,'data',[0],self.config,gpus=[0],timeout_hours=.00001)
        self.assertEqual(read_json(self.root/'completion.json')['reason'],'TimeoutError')

    def test_no_gpu_and_duplicate_seed_preflight(self):
        with self.assertRaises(RuntimeError):r.launch(self.root,'data',[0],self.config,gpus=[])
        with self.assertRaises(ValueError):r.launch(self.root,'data',[0,0],self.config,gpus=[0,1])
    def test_actual_notebook_cells_execute_with_synthetic_workers(self):
        import subprocess,sys
        notebook=read_json(r.ROOT/'notebooks/dpe_ermstar_neural_two_gpu_kaggle.ipynb')
        import shutil
        checkout=self.root/'notebook/repository'
        for name in ('src','configs'):
            shutil.copytree(r.ROOT/name,checkout/name)
        for name in ('utils/isomaxplus.py','utils/datasets.py','notebooks/src/utils.py'):
            target=checkout/'baselines/dpe4subpop'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(r.ROOT/'baselines/dpe4subpop'/name,target)
        (checkout/'.git').mkdir()
        (checkout/'scripts').mkdir()
        shutil.copyfile(r.ROOT/'scripts/run_dpe_ermstar.py',checkout/'scripts/run_dpe_ermstar.py')
        code=["import torch,torchvision\ntorch.set_num_threads(1)\ntorch.cuda.device_count=lambda:2\ntorch.cuda.get_device_name=lambda i:'synthetic GPU'\ntorchvision.models.ResNet50_Weights.get_state_dict=lambda *args,**kwargs:{}"]
        for cell in notebook['cells']:
            if cell['cell_type']!='code':continue
            text=''.join(cell['source']).replace('/kaggle/working/dpe_ermstar',str(self.root/'notebook')).replace('/kaggle/working/waterbirds-shortcut-mitigation',str(checkout));code.append(text)
            if 'WORK = Path' in text:code.append(f"WORK=Path({str(self.root/'notebook')!r});WORK.mkdir(parents=True,exist_ok=True)")
            if 'print([torch.cuda.get_device_name' in text:
                code.append("""
from src.methods.dpe_head_reweighting import study as infrastructure
from src.methods.dpe_ermstar import run as runtime
from src.methods.erm import train as backbone_module
infrastructure.ensure_dataset=lambda *args,**kwargs:Path('synthetic-data')
def fake_backbone(c,resume=False):
    path=Path(c.output_dir);path.mkdir(parents=True,exist_ok=True)
    (path/'ckpt_last.pt').write_bytes(b'synthetic checkpoint')
    runtime.write_json(path/'summary.json',dict(updates=37*400,selected_checkpoint='ckpt_last.pt'))
backbone_module.run_training=fake_backbone
rng=runtime.np.random.RandomState(1)
g=runtime.np.repeat(['00','01','10','11'],8)
val=dict(features=rng.randn(32,8).astype('float32'),ids=runtime.np.array([f'v{i}' for i in range(32)]),y=runtime.np.array([int(x[0]) for x in g]),g=g)
test=dict(val,ids=runtime.np.array([f't{i}' for i in range(32)]))
runtime.features=lambda *args,**kwargs:dict(val=val,test=test)
def fake_launch(work,data,seeds,config,gpus=None,**kwargs):
    assert gpus==[0,1] and seeds==[0,1]
    config['head_training'].update(epochs=1,batch_size=8)
    config['neural']['steps']=2
    return [runtime.worker(work,data,seed,config,'cpu') for seed in seeds]
runtime.launch=fake_launch
""")
        code.append("assert len(results)==2\nassert all(len(x['results'])==12 for x in results)")
        script=self.root/'execute.py';script.write_text('\n\n'.join(code))
        completed=subprocess.run([sys.executable,str(script)],cwd=self.root,capture_output=True,text=True,timeout=90)
        self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)

    def test_backbone_resume_without_file_checksums(self):
        from PIL import Image
        import pandas as pd
        from torch import nn
        from src.methods.erm import train
        from src.methods.erm.config import ERMConfig
        class Tiny(nn.Sequential):
            def __init__(self,*args):super().__init__(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(3,2))
        data=self.root/'data';data.mkdir();rows=[]
        for split in (0,1):
            for group in range(4):
                name=f'{split}_{group}.png';Image.new('RGB',(32,32),(group*40,20,70)).save(data/name)
                rows.append(dict(filename=name,y=group//2,a=group%2,split=split))
        pd.DataFrame(rows).to_csv(data/'metadata.csv',index=False)
        c=ERMConfig();c.dataset.data_dir=str(data);c.dataset.workers=0;c.dataset.batch_size=4;c.dataset.batch_size_eval=4
        c.training.device='cpu';c.training.epochs=2;c.output_dir=str(self.root/'backbone')
        with patch.object(train,'ResNetClassifier',Tiny):
            first=train.run_training(c)
            self.assertEqual(first,train.run_training(c,resume=True))
        state=torch.load(self.root/'backbone/training_state.pt',weights_only=True)
        self.assertIn('dataset_rows',state)
        self.assertNotIn('metadata_sha256',state)
        self.assertIsInstance(state['implementation'],list)

    def test_notebook_is_only_a_launcher(self):
        n=read_json(r.ROOT/'notebooks/dpe_ermstar_neural_two_gpu_kaggle.ipynb')
        code='\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
        for forbidden in ('b64decode', 'hexdigest', 'check_hash', 'file_sha256', 'hashlib',
                          'record_file_checksums', 'write_text(', 'source =', 'exec(', 'ast.'):
            self.assertNotIn(forbidden,code)
        self.assertIn('from src.methods.dpe_ermstar.run import prepare_and_launch',code)
        for i,c in enumerate(n['cells']):
            if c['cell_type']=='code':
                self.assertLessEqual(len(c['source']),30)
                compile(''.join(c['source']),f'cell{i}','exec')

    def test_original_routines_copied_without_math_changes(self):
        import ast
        from src.methods.dpe_ermstar import upstream
        target=ast.parse(Path(upstream.__file__).read_text())
        copied={node.name:ast.dump(node,include_attributes=False) for node in target.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
        for path,names in [('utils/isomaxplus.py',['IsoMaxPlusLossFirstPart','IsoMaxPlusLossSecondPart']),
                           ('utils/datasets.py',['subsample']),('notebooks/src/utils.py',['train_prototypes'])]:
            original=ast.parse((r.ROOT/'baselines/dpe4subpop'/path).read_text())
            for node in original.body:
                if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
                    self.assertEqual(copied[node.name],ast.dump(node,include_attributes=False))

if __name__=='__main__':unittest.main()
