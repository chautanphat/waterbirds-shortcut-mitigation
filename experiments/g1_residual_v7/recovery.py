"""Real nested-secondary and C-fold forced termination tests; reuse validated work."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch
from experiments.geometry_aggregation_v5.io import single_writer
from experiments.geometry_aggregation_v5.persistence import atomic
from experiments.discrete_weighting.store import sha
from .store import ROOT,ROLE,Journal,jsonwrite


def store_acceptance():
    import experiments.geometry_aggregation_v5.persistence as persistence
    root=ROOT/'acceptance/store';root.mkdir(parents=True,exist_ok=True);signals={s:signal.getsignal(s) for s in [signal.SIGINT,signal.SIGTERM]}
    m=dict(test='atomic corruption recovery acceptance',source_sha256=sha(Path(__file__)))
    try:
        old=persistence.os.replace;attempts=[]
        def transient(a,b):
            attempts.append(1)
            if len(attempts)<3:raise PermissionError('simulated OneDrive rename lock')
            return old(a,b)
        with patch.object(persistence.os,'replace',side_effect=transient):atomic(root/'atomic_payload','atomic-valid')
        assert len(attempts)==3 and (root/'atomic_payload').read_text()=='atomic-valid'
        with single_writer(root):
            try:
                with single_writer(root):raise AssertionError('Concurrent writer accepted')
            except RuntimeError:pass
        j=Journal(root/'journal',m)
        def work():
            p=root/'journal/data.json';jsonwrite(p,dict(value='valid',role=ROLE));return dict(status='PASS'),[p]
        j.task('test',[],work);stamp=(root/'journal/data.json').stat().st_mtime_ns
        atomic(root/'journal/state.json','truncated state')
        j=Journal(root/'journal',m);j.task('test',[],lambda:(_ for _ in ()).throw(AssertionError('recomputed')))
        assert (root/'journal/data.json').stat().st_mtime_ns==stamp
        from shutil import copytree
        corrupt=root/'corrupt'
        if not corrupt.exists():copytree(root/'journal',corrupt)
        atomic(corrupt/'data.json','corrupt')
        try:Journal(corrupt,m)
        except ValueError:pass
        else:raise AssertionError('Corrupt checkpoint accepted')
        truncated=root/'truncated'
        if not truncated.exists():copytree(root/'journal',truncated)
        receipt=next((truncated/'completed').glob('*.json'));atomic(receipt,'{"truncated":')
        try:Journal(truncated,m)
        except ValueError:pass
        else:raise AssertionError('Truncated receipt accepted')
        p=root/'receipt.json';jsonwrite(p,dict(role=ROLE,status='PASS',atomic_replace_retry=True,OS_single_writer=True,
            stale_state_rebuilt=True,valid_checkpoint_reused=True,corrupt_artifact_rejected=True,truncated_receipt_rejected=True))
        return dict(status='PASS'),[p,root/'atomic_payload',root/'journal/data.json']
    finally:
        for s,h in signals.items():signal.signal(s,h)


def worker(kind,root,pause=False):
    from .run import secondary,prediction,folder,bpath,cpath
    root=Path(root)
    inputs=[bpath(0,0)] if kind=='secondary' else [cpath(0,f) for f in range(2)]+[folder(0,f,4)/'model.json' for f in range(2)]
    m=dict(test='real '+kind+' forced kill',role=ROLE,inputs={str(p):sha(p) for p in inputs},
        source={str(p):sha(p) for p in [Path(__file__),ROOT/'model.py',ROOT/'run.py',ROOT/'store.py']})
    j=Journal(root,m)
    for i in range(2):
        def work(i=i):
            result,paths=secondary(0,0,0,i,dest=root/f'job{i}') if kind=='secondary' else prediction(0,i,4,'G1_RH',dest=root/f'job{i}/predictions.npz')
            if pause and i==1:
                atomic(root/'pause.ready','One committed real task; next task written but not committed');time.sleep(120)
            return result,paths
        j.task(f'job:{i}',[] if i==0 else ['job:0'],work)


def acceptance(kind):
    root=ROOT/f'acceptance/{kind}';resumed=root/'interrupted';control=root/'uninterrupted';log=root/'recovery.log';log.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ|dict(PYTHONUTF8='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=[sys.executable,'-m','experiments.g1_residual_v7.recovery',kind,str(resumed)]
    with log.open('ab') as stream:
        if len(list((resumed/'completed').glob('*.json')))<2:
            (resumed/'pause.ready').unlink(missing_ok=True)
            proc=subprocess.Popen(command+['--pause'],env=env,stdout=stream,stderr=subprocess.STDOUT)
            try:
                for _ in range(1100):
                    if (resumed/'pause.ready').exists():break
                    if proc.poll() is not None:raise RuntimeError('Recovery worker failed: '+str(log))
                    time.sleep(.05)
                assert (resumed/'pause.ready').exists(),'Recovery timed out'
            finally:
                if proc.poll() is None:proc.kill()
                proc.wait(timeout=15)
            assert len(list((resumed/'completed').glob('*.json')))==1
            first=list((resumed/'job0').glob('*'));stamps={p.name:p.stat().st_mtime_ns for p in first};hashes={p.name:sha(p) for p in first}
            subprocess.run(command,check=True,env=env,stdout=stream,stderr=subprocess.STDOUT)
            assert all((resumed/'job0'/n).stat().st_mtime_ns==t and sha(resumed/'job0'/n)==hashes[n] for n,t in stamps.items())
        subprocess.run([sys.executable,'-m','experiments.g1_residual_v7.recovery',kind,str(control)],check=True,env=env,stdout=stream,stderr=subprocess.STDOUT)
    paths=[log]
    for i in range(2):
        for a in (resumed/f'job{i}').glob('*'):
            b=control/f'job{i}'/a.name;assert sha(a)==sha(b),'Recovered bytes differ: '+str(a);paths.extend([a,b])
    p=root/'receipt.json';jsonwrite(p,dict(role=ROLE,status='PASS',kind=kind,forced_process_kill=True,first_committed_task_not_rewritten=True,
        only_uncommitted_task_replayed=True,all_bytes_equal_uninterrupted=True,duplicate_receipts=False,reuse_scientific_outputs=True))
    return dict(status='PASS',kind=kind),paths+[p]


def reuse_secondary(s,f,part,k):
    from .run import secondary,folder
    if (s,f,part)==(0,0,0) and k<2:
        source=ROOT/f'acceptance/secondary/interrupted/job{k}';dest=folder(s,f,part)/f'scale_subcv/sub{k}';paths=[]
        for p in source.glob('*'):
            out=dest/p.name;atomic(out,p.read_bytes());assert sha(out)==sha(p);paths.append(out)
        return dict(seed=s,fold=f,part=part,secondary=k,reused_acceptance=True),paths
    return secondary(s,f,part,k)


def reuse_C(s,f,method):
    from .run import prediction,pred_path
    if s==0 and f<2 and method=='G1_RH':
        source=ROOT/f'acceptance/C/interrupted/job{f}/predictions.npz';dest=pred_path(s,f,4,method)
        atomic(dest,source.read_bytes());assert sha(dest)==sha(source)
        return dict(seed=s,fold=f,method=method,reused_acceptance=True),[dest]
    return prediction(s,f,4,method)


if __name__=='__main__':worker(sys.argv[1],sys.argv[2],pause='--pause' in sys.argv)
