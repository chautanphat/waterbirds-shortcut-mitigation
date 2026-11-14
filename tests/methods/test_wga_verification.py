import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from experiments.wga_verification.journal import Journal
from experiments.wga_verification.solver import clusters,solve
from experiments.discrete_weighting.store import atomic,key
from threadpoolctl import threadpool_limits


def test_frozen_solver_and_normalization():
    rng=np.random.default_rng(12);p=rng.dirichlet([3,3],size=(100,15));y=rng.integers(0,2,100);labels=np.tile([0,1],50)
    w,a=solve(p,y,labels);d,b=solve(p,y,labels,False)
    assert a['success'] and b['success']
    assert abs(a['objective']-b['objective'])<1e-6
    assert w.min()>=-1e-8 and w.max()<=.25+1e-8
    np.testing.assert_allclose(w.sum(),1,atol=1e-8)
    np.testing.assert_allclose(np.einsum('nmc,m->nc',p,w).sum(1),1,atol=1e-8)
    np.testing.assert_allclose(np.einsum('nmc,m->nc',p,np.full(15,1/15)),p.mean(1))


def test_pca_kmeans_reproducibility_and_train_only():
    rng=np.random.default_rng(0);z=rng.normal(size=(80,32))
    with threadpool_limits(1):a,k,c=clusters(z);b,l,d=clusters(z)
    np.testing.assert_array_equal(c,d);np.testing.assert_allclose(a.mean_,z.mean(0))
    assert a.n_components==24 and k.n_init==6 and k.random_state==1


def test_corrupt_output_dependency_and_temp(tmp_path):
    def a():
        p=tmp_path/'a.txt';atomic(p,'valid');return {'seed':0},[p]
    def b():
        p=tmp_path/'b.txt';atomic(p,'valid child');return {},[p]
    j=Journal(tmp_path,{'test':1},'resume');j.task('a',[],a);j.task('b',['a'],b)
    stamp=(tmp_path/'a.txt').stat().st_mtime_ns
    atomic(tmp_path/'abandoned.tmp','incomplete')
    r=Journal(tmp_path,{'test':1},'resume');r.task('a',[],lambda:1/0)
    assert stamp==(tmp_path/'a.txt').stat().st_mtime_ns
    (tmp_path/'a.txt').unlink();r=Journal(tmp_path,{'test':1},'resume');assert not r.records
    r.task('a',[],a);r.task('b',['a'],b)
    assert len(r.records)==2


def test_killed_separate_process_resumes(tmp_path):
    code='''import sys,time
from pathlib import Path
from experiments.wga_verification.journal import Journal
from experiments.discrete_weighting.store import atomic
r=Path(sys.argv[1]);j=Journal(r,{'test':1},'resume')
def first():
 p=r/'first.txt';atomic(p,'done');return {},[p]
j.task('first',[],first)
def second():
 atomic(r/'started','yes');atomic(r/'partial.tmp','truncated')
 if len(sys.argv)>2:time.sleep(60)
 p=r/'second.txt';atomic(p,'done');return {},[p]
j.task('second',['first'],second)
'''
    proc=subprocess.Popen([sys.executable,'-c',code,str(tmp_path),'wait'],stdout=subprocess.DEVNULL)
    try:
        for _ in range(200):
            if (tmp_path/'started').exists():break
            time.sleep(.025)
        assert (tmp_path/'started').exists();stamp=(tmp_path/'first.txt').stat().st_mtime_ns
    finally:proc.kill();proc.wait(timeout=10)
    subprocess.run([sys.executable,'-c',code,str(tmp_path)],check=True,capture_output=True)
    assert (tmp_path/'first.txt').stat().st_mtime_ns==stamp
    assert len(list((tmp_path/'completed').glob('*.json')))==2
    assert (tmp_path/'second.txt').read_text()=='done'


def test_stable_id():assert key({'b':2,'a':1})==key({'a':1,'b':2})
