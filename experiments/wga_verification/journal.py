import csv
import io
import json
import os
from pathlib import Path
from datetime import datetime,timezone
from experiments.discrete_weighting.store import atomic,dump,key,sha


class Journal:
    def __init__(self,root,manifest,command,limit=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.command=command;self.limit=limit;self.executed=0;self.stop=False;self.records={}
        p=self.root/'manifest.json'
        if p.exists() and json.loads(p.read_text(encoding='utf-8'))!=manifest:raise ValueError('Manifest mismatch')
        if not p.exists():dump(p,manifest)
        for p in (self.root/'completed').glob('*.json'):
            try:
                r=json.loads(p.read_text(encoding='utf-8'))
                if all((self.root/a).is_file() and (self.root/a).stat().st_size==v['bytes'] and sha(self.root/a)==v['sha256'] for a,v in r['artifacts'].items()):self.records[r['id']]=r
            except (ValueError,OSError):pass
        while True:
            bad=[t for t,r in self.records.items() if any(d not in self.records for d in r['dependencies'])]
            if not bad:break
            for t in bad:del self.records[t]
        state=self.root/'state.json';previous=json.loads(state.read_text()) if state.exists() else {}
        extras={k:v for k,v in previous.items() if k in ('gate_A','gate_B','production_integration','measured_delta_WGA_pp','measured_delta_AA_pp')}
        self.status(previous.get('status','READY'),previous.get('next_pending','resume next valid dependency'),**extras)

    def status(self,status,next_task,**details):
        metrics=[dict(task_id=t,**r['metrics']) for t,r in sorted(self.records.items()) if r['metrics']]
        s=io.StringIO(newline='');cols=sorted({k for r in metrics for k in r}) or ['task_id'];w=csv.DictWriter(s,fieldnames=cols);w.writeheader()
        for r in metrics:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})
        atomic(self.root/'results/partial_metrics.csv',s.getvalue())
        state=dict(status=status,completed=len(self.records),last_completed=getattr(self,'last',None),next_pending=next_task,
                   run_id=self.root.name,updated_utc=datetime.now(timezone.utc).isoformat(),resume=self.command,**details)
        dump(self.root/'state.json',state)
        content='# WGA verify-first handoff\n\n'+json.dumps(state,indent=2)+'\n\nRead manifest.json, frozen.yaml and gate decisions. Completion markers and hashes are authoritative. '
        content+='Do not repeat valid completed tasks. Do not change the frozen recipe or select on historical test/groups. Gate A must pass before nested heads; both gates must pass before production integration.\n\n'+self.command+'\n'
        atomic(self.root/'HANDOFF_STATUS.md',content);atomic(self.root/'NEXT_AGENT_PROMPT.md','Continue this v3 verification from files.\n\n'+content)

    def task(self,id_,deps,fn):
        if any(d not in self.records for d in deps):raise RuntimeError('Missing task dependency')
        if id_ in self.records:return self.records[id_]
        if self.stop or (self.limit is not None and self.executed>=self.limit):
            self.status('INTERRUPTED',id_);raise InterruptedError('Stopped at durable boundary')
        self.status('RUNNING',id_)
        metrics,paths=fn()
        artifacts={Path(p).relative_to(self.root).as_posix():dict(sha256=sha(p),bytes=Path(p).stat().st_size) for p in paths}
        record=dict(id=id_,dependencies=deps,metrics=metrics,artifacts=artifacts)
        dump(self.root/'completed'/f'{key(id_)[:20]}.json',record)
        self.records[id_]=record;self.executed+=1;self.last=id_
        with (self.root/'results/events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(event='COMMITTED',task_id=id_))+'\n');f.flush();os.fsync(f.fileno())
        self.status('RUNNING','next task');print('COMMITTED '+id_,flush=True)
        return record
