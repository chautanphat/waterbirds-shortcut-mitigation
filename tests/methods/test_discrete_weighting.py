import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from experiments.discrete_weighting.store import Store,atomic,dump,key
from experiments.discrete_weighting.reliability import Rule,recipes,ablations


def test_stable_hash():assert key({'b':2,'a':1})==key({'a':1,'b':2})


def test_atomic_and_corruption_recovery(tmp_path):
    s=Store(tmp_path,{'test':True},'resume')
    def work():
        p=tmp_path/'data.json';dump(p,{'value':1});return {'accuracy':.9,'seed':0},[p]
    s.task('a','phase1',[],work);stamp=(tmp_path/'data.json').stat().st_mtime_ns
    (tmp_path/'data.json.tmp').write_bytes(b'truncated')
    resumed=Store(tmp_path,{'test':True},'resume');resumed.task('a','phase1',[],lambda:pytest.fail('repeated'))
    assert (tmp_path/'data.json').stat().st_mtime_ns==stamp
    (tmp_path/'data.json').write_bytes(b'corrupt')
    recovered=Store(tmp_path,{'test':True},'resume');assert 'a' not in recovered.valid
    recovered.task('a','phase1',[],work)
    rows=(tmp_path/'results/phase1_incremental.csv').read_text().splitlines();assert len(rows)==2


def test_distinct_process_budget_and_resume(tmp_path):
    code='''from pathlib import Path
from experiments.discrete_weighting.store import Store,atomic
import sys
r=Path(sys.argv[1]);s=Store(r,{'test':True},'resume',max_tasks=2)
for i in range(5):
 def f(i=i):
  p=r/f'item{i}.txt';atomic(p,str(i));return {'seed':0,'accuracy':.5},[p]
 try:s.task(str(i),'phase1',[],f)
 except InterruptedError:break
'''
    subprocess.run([sys.executable,'-c',code,str(tmp_path)],check=True,capture_output=True)
    stamps={p.name:p.stat().st_mtime_ns for p in tmp_path.glob('item*.txt')};assert len(stamps)==2
    subprocess.run([sys.executable,'-c',code,str(tmp_path)],check=True,capture_output=True)
    assert len(list(tmp_path.glob('item*.txt')))==4
    assert all((tmp_path/k).stat().st_mtime_ns==v for k,v in stamps.items())
    subprocess.run([sys.executable,'-c',code,str(tmp_path)],check=True,capture_output=True)
    assert len(list(tmp_path.glob('item*.txt')))==5
    assert len((tmp_path/'results/phase1_incremental.csv').read_text().splitlines())==6


def test_inference_safety_and_all_recipes():
    rng=np.random.default_rng(9);p=rng.dirichlet([3,3],size=(90,15));y=rng.integers(0,2,90)
    for c in recipes()+[v['config'] for v in ablations({'family':'D5','head':True,'agreement':True})]:
        model=Rule(c).fit(p[:60],y[:60]);mix,w=model.predict(p[60:])
        assert np.isfinite(mix).all() and np.isfinite(w).all() and (w>=0).all()
        np.testing.assert_allclose(w.sum(1),1);np.testing.assert_allclose(mix.sum(1),1)
        np.testing.assert_array_equal(mix,model.predict(p[60:])[0])
    assert {v['ablation'] for v in ablations({'family':'D2'})}=={f'A{i}' for i in range(21)}


def test_baseline_and_empty_cells():
    p=np.full((20,15,2),.5);y=np.tile([0,1],10)
    for c in ({'family':'D0a'},{'family':'D3','head':True,'kappa':0},{'family':'D7','ordinal':True}):
        model=Rule(c).fit(p,y);mix,w=model.predict(p)
        np.testing.assert_allclose(mix,.5);np.testing.assert_allclose(w,1/15)
