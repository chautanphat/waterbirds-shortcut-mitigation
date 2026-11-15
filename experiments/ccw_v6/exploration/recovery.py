"""Kill/restart real full-B fitting before the nine-ensemble diagnostic."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.ccw_v6.model import fit
from experiments.geometry_aggregation_v5.io import load
from experiments.geometry_aggregation_v5.persistence import atomic,dump
from experiments.discrete_weighting.store import sha
from .journal import BASE,ROOT,BANNER,Journal


def worker(folder,pause=False):
    folder=Path(folder)
    paths=[BASE/f'inputs/B/seed0/fold{f}.npz' for f in range(2)]
    manifest=dict(role=BANNER,test='real full-B forced termination',inputs={str(p):sha(p) for p in paths},
                  model_sha256=sha(BASE/'model.py'),worker_sha256=sha(Path(__file__)))
    j=Journal(folder,manifest,bridge=False)
    for f,p in enumerate(paths):
        def work(f=f,p=p):
            b=load(p);model=fit(b['probs'],b['distances'],b['y'],b['sample_id'],b['head_sha256'])
            dest=folder/f'fold{f}/model.json';dump(dest,model)
            if pause and f==1:
                atomic(folder/'pause.ready','First fit committed; second fit not committed.');time.sleep(120)
            return dict(fold=f,fit_images=len(b['y']),role=BANNER),[dest]
        j.task(f'fit:{f}',[] if f==0 else ['fit:0'],work)


def acceptance():
    folder=ROOT/'acceptance/interrupted';control=ROOT/'acceptance/uninterrupted'
    cmd=[sys.executable,'-m','experiments.ccw_v6.exploration.recovery',str(folder)]
    env=os.environ|dict(PYTHONUTF8='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    log=ROOT/'acceptance/recovery.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as stream:
        if len(list((folder/'completed').glob('*.json')))<2:
            # A stale pre-commit marker is safe to remove only in this acceptance namespace.
            (folder/'pause.ready').unlink(missing_ok=True)
            proc=subprocess.Popen(cmd+['--pause'],stdout=stream,stderr=subprocess.STDOUT,env=env)
            try:
                for _ in range(1000):
                    if (folder/'pause.ready').exists():break
                    if proc.poll() is not None:raise RuntimeError('Recovery worker failed; inspect '+str(log))
                    time.sleep(.05)
                assert (folder/'pause.ready').exists(),'Recovery timeout'
            finally:
                if proc.poll() is None:proc.kill()
                proc.wait(timeout=15)
            assert len(list((folder/'completed').glob('*.json')))==1
            first=folder/'fold0/model.json';stamp=first.stat().st_mtime_ns;digest=sha(first)
            subprocess.run(cmd,check=True,env=env,stdout=stream,stderr=subprocess.STDOUT)
            assert first.stat().st_mtime_ns==stamp and sha(first)==digest
        subprocess.run([sys.executable,'-m','experiments.ccw_v6.exploration.recovery',str(control)],check=True,env=env,stdout=stream,stderr=subprocess.STDOUT)
    paths=[log]
    for f in range(2):
        a=folder/f'fold{f}/model.json';b=control/f'fold{f}/model.json'
        assert json.loads(a.read_text())==json.loads(b.read_text())
        paths.extend([a,b])
    receipt=ROOT/'acceptance/recovery_receipt.json'
    result=dict(status='PASS',role=BANNER,forced_kill_mid_second_full_B_fit=True,
        completed_first_fit_unchanged=True,resumed_equal_uninterrupted=True,
        completed_fit_recomputed=False,unfinished_fit_replayed=True,reuse_fits_in_main_run=True)
    dump(receipt,result)
    return result,paths+[receipt]


if __name__=='__main__':worker(sys.argv[1],pause='--pause' in sys.argv)
