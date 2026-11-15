"""Separate v7 journal, hashed semantic tasks, stale-state rebuild and safe IO."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
from experiments.geometry_aggregation_v5.persistence import atomic,dump
from experiments.geometry_aggregation_v5.journal import Journal as BaseJournal,csvwrite
from experiments.discrete_weighting.store import sha

ROOT=Path('experiments/g1_residual_v7')
ROLE='EXPLORATORY_ONLY — REUSED DEVELOPMENT — NOT INDEPENDENT CONFIRMATION'
COMMAND='.venv/Scripts/python.exe -m experiments.g1_residual_v7.run --phase all --resume'


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def jsonwrite(path,value):
    dump(path,value);assert json.loads(Path(path).read_text(encoding='utf-8'))==value


class Journal(BaseJournal):
    def __init__(self,root,manifest,max_tasks=None):
        self.plan=manifest.get('planned_DAG',{});self.specs=manifest.get('task_specs',{})
        self.fingerprint=digest({k:v for k,v in manifest.items() if k not in ['planned_DAG','task_specs']})
        state=Path(root)/'state.json'
        if state.exists():
            try:json.loads(state.read_text())
            except (ValueError,UnicodeError):
                atomic(Path(root)/'failures/stale_state_preserved.txt',state.read_bytes());jsonwrite(state,dict(status='REBUILDING_FROM_RECEIPTS'))
        super().__init__(root,manifest,max_tasks)
        mh=sha(self.root/'manifest.json')
        for r in self.records.values():
            if r['manifest_sha256']!=mh:raise ValueError('Receipt manifest mismatch')

    def ident(self,name):return name+'#'+digest(dict(manifest=self.fingerprint,name=name,spec=self.specs.get(name,{})))[:20]
    def has(self,name):return self.ident(name) in self.records
    def result(self,name):return self.records[self.ident(name)]['result']

    def task(self,name,deps,fn):
        def work():
            result,paths=fn();result=dict(result,role=ROLE)
            return result,paths
        return super().task(self.ident(name),[self.ident(d) for d in deps],work)

    def status(self,status,next_task,**extra):
        self.current.update(extra)
        self.current.update(status=status,next_pending=next_task,completed=len(self.records),
            updated_utc=datetime.now(timezone.utc).isoformat(),role=ROLE,independent_confirmation='INDEPENDENT_CONFIRMATION_NOT_PERFORMED',
            production_promotion=False,resume_command=COMMAND,manifest_sha256=sha(self.root/'manifest.json'),
            task_DAG={k:dict(dependencies=v,status='COMPLETE' if self.ident(k) in self.records else 'PENDING') for k,v in self.plan.items()})
        jsonwrite(self.root/'state.json',self.current)
        compact={k:v for k,v in self.current.items() if k!='task_DAG'}
        atomic(self.root/'HANDOFF_STATUS.md','# G1 residual v7 checkpoint\n\n**'+ROLE+'**\n\n'+json.dumps(compact,indent=2)+'\n\n'
            'Task receipts are authoritative. All feature statistics and scale CV fit only the owning B training IDs. Preserve v1-v6; never tune from C.\n\n'+COMMAND+'\n')
        atomic(self.root/'NEXT_AGENT_PROMPT.md','\n'.join([
            'Continue the frozen G1 residual v7 study, not a G1-only confirmation.',
            'Read the v7 handoff, config.lock.json, EXPERIMENT_LOCK.json, state.json and HANDOFF_STATUS.md.',
            'Run '+COMMAND,'Validate with --verify; honor every existing receipt; never rerun completed fits.',
            'Exactly U0, G1_frozen, G1_R, primary G1_RH, G1_RHM. No lambda1 sensitivity or hyperparameter grid.',
            'Four B inner folds; three secondary scale folds entirely inside B-training. C labels/groups evaluation-only.',
            'Run all preregistered C exploratory diagnostics even if B_DIRECTION_NO_GO; never promote.',
            'Resume all bootstrap/permutation batches with their saved RNG states. Preserve v1-v6 original artifacts.',
            'If DONE_EXPLORATORY, read EXPLORATORY_C_REPORT.md; no fresh data means INDEPENDENT_CONFIRMATION_NOT_PERFORMED.','']))
