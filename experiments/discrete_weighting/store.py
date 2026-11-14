import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time
import traceback


def key(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        f.write(data.encode('utf-8') if isinstance(data,str) else data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def dump(path,value):atomic(path,json.dumps(value,sort_keys=True,indent=2,allow_nan=False))


class Store:
    def __init__(self,root,manifest,command,max_tasks=None,max_minutes=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.command=command;self.limit=max_tasks;self.minutes=max_minutes;self.start=time.monotonic();self.count=0
        self.manifest=manifest;self.valid={};self.stop=False;self.last=None
        marker=self.root/'manifest.json'
        if marker.exists() and json.loads(marker.read_text())!=manifest:raise ValueError('Incompatible manifest; new run ID required')
        if not marker.exists():dump(marker,manifest)
        for p in (self.root/'completed').glob('*.json'):
            try:
                r=json.loads(p.read_text())
                if all((self.root/a).is_file() and sha(self.root/a)==h for a,h in r['artifacts'].items()):self.valid[r['task_id']]=r
            except (ValueError,OSError):pass
        changed=True
        while changed:
            changed=False
            for t,r in list(self.valid.items()):
                if any(d not in self.valid for d in r['dependencies']):del self.valid[t];changed=True
        prior_state=self.root/'state.json'
        previous=json.loads(prior_state.read_text()) if prior_state.exists() else {}
        self.refresh(previous.get('status','RUNNING'),previous.get('next_pending','discover next pending task'))

    def refresh(self,status,next_task):
        rows=[r['result']|dict(task_id=t,phase=r['phase']) for t,r in sorted(self.valid.items()) if r.get('result')]
        for phase in ('phase1','phase2','test'):
            selected=[r for r in rows if r['phase']==phase]
            if selected:
                cols=sorted(set(k for r in selected for k in r));s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=cols);w.writeheader()
                for r in selected:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})
                atomic(self.root/f'results/{phase}_incremental.csv',s.getvalue())
        best=sorted([r for r in rows if r['phase']=='phase1' and r.get('eligible',True)],key=lambda r:(-r.get('accuracy',0),-r.get('balanced_accuracy',0),r.get('nll',999)))
        counts={p:sum(r['phase']==p for r in self.valid.values()) for p in sorted({r['phase'] for r in self.valid.values()})}
        by_seed={str(s):sum(r.get('result',{}).get('seed')==s for r in self.valid.values()) for s in (0,1,2)}
        state=dict(run_id=self.root.name,status=status,updated_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                   completed=len(self.valid),counts_by_phase=counts,counts_by_seed=by_seed,last_completed=self.last,next_pending=next_task,
                   best_selection_only_task=best[0]['task_id'] if best else None,planned=self.manifest.get('planned','adaptive Phase 2 after locked winner'),resume_command=self.command)
        dump(self.root/'state.json',state)
        text='# Durable handoff status\n\n'+json.dumps(state,indent=2)+'\n\nRead manifest.json, config.lock.yaml, DATA_AUDIT.md and completed/ markers. '
        text+='Markers are authoritative; verify hashes, reuse valid completed tasks, rebuild only corrupt/missing artifacts. '
        text+='Test scores must never select recipes. No new CNN/head training is needed.\n\n'+self.command+'\n'
        atomic(self.root/'HANDOFF_STATUS.md',text)
        atomic(self.root/'NEXT_AGENT_PROMPT.md','Continue the discrete reliability v2 study from files alone.\n\n'+text)
        atomic(self.root/'REPORT.md','# Progressive research record\n\nStatus: '+status+'\n\nCompleted tasks: '+str(len(self.valid))+'\n\n'+
               'Selection-only leader: '+str(state['best_selection_only_task'])+'\n\nFull incremental metrics are under results/. Test is previously explored; all test results are exploratory.\n')
        if 'report' in self.valid and (self.root/'final_report.md').exists():atomic(self.root/'REPORT.md',(self.root/'final_report.md').read_text())

    def task(self,task_id,phase,dependencies,fn):
        if any(d not in self.valid for d in dependencies):raise RuntimeError('Required dependency unavailable: '+task_id)
        if task_id in self.valid:return self.valid[task_id]
        if self.stop or (self.limit is not None and self.count>=self.limit) or (self.minutes and time.monotonic()-self.start>=60*self.minutes):
            self.refresh('INTERRUPTED',task_id);raise InterruptedError('Task/time budget or signal; resume from durable markers')
        self.refresh('RUNNING',task_id)
        try:
            result,paths=fn()
            artifacts={str(Path(p).relative_to(self.root)).replace('\\','/'):sha(p) for p in paths}
            record=dict(task_id=task_id,phase=phase,dependencies=dependencies,result=result,artifacts=artifacts)
            dump(self.root/'completed'/f'{key(task_id)[:20]}.json',record)
            self.valid[task_id]=record;self.count+=1;self.last=task_id
            with (self.root/'progress.ndjson').open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(event='committed',task_id=task_id,artifacts=artifacts))+'\n');f.flush();os.fsync(f.fileno())
            self.refresh('RUNNING','next dependency-ready task')
            print('COMMITTED '+task_id,flush=True)
            return record
        except Exception as e:
            path=self.root/'failed'/f'{key(task_id)[:20]}.json'
            old=json.loads(path.read_text()) if path.exists() else {}
            dump(path,dict(task_id=task_id,error=str(e),traceback=traceback.format_exc(),retry_count=old.get('retry_count',0)+1))
            raise

