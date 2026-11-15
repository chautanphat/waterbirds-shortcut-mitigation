"""Reuse durable atomic infrastructure with isolated CCW state and a planned DAG."""
import json
from datetime import datetime,timezone
from pathlib import Path
from experiments.geometry_aggregation_v5.journal import Journal as BaseJournal,csvwrite
from experiments.geometry_aggregation_v5.persistence import atomic,dump
from experiments.discrete_weighting.store import sha


class Journal(BaseJournal):
    def __init__(self,root,manifest,max_tasks=None):
        self.plan=manifest.get('planned_DAG',{})
        super().__init__(root,manifest,max_tasks)

    def status(self,status,next_task,**extra):
        self.current.update(extra);skipped=set(self.current.get('skipped_tasks',[]))
        self.current.update(status=status,next_pending=next_task,completed=len(self.records),updated_utc=datetime.now(timezone.utc).isoformat(),
            manifest_sha256=sha(self.root/'manifest.json'),resume_command='.venv/Scripts/python.exe -m experiments.ccw_v6.run --phase all --resume',
            task_DAG={k:dict(dependencies=v,status='COMPLETE' if k in self.records else 'SKIPPED' if k in skipped else 'ACTIVE' if k==next_task and status=='RUNNING' else 'PENDING') for k,v in self.plan.items()},
            completed_fits=[k for k in self.records if k.startswith(('fit:','full_fit:'))])
        dump(self.root/'state.json',self.current)
        compact={k:v for k,v in self.current.items() if k not in ['task_DAG','completed_fits','skipped_tasks']}
        text='# CCW v6 durable status\n\n'+json.dumps(compact,indent=2)+'\n\n'
        text+='All valid completion receipts are authoritative. Preserve v1-v5. Fit/select exclusively on matching B predictions; old C is exploratory, never selection evidence. CCW promotion requires a genuinely fresh cohort.\n'
        atomic(self.root/'HANDOFF_STATUS.md',text)
        atomic(self.root/'NEXT_AGENT_PROMPT.md','\n'.join([
            'Continue the frozen CCW v6 workflow in this repository.',
            'Read experiments/ccw_v6/state.json, HANDOFF_STATUS.md, PROTOCOL_DECISIONS.md and the v6 handoff.',
            'Run .venv/Scripts/python.exe -m experiments.ccw_v6.run --phase all --resume',
            'Use --verify to validate all completion hashes; --status gives the current boundary.',
            'Reuse valid completed fits/predictions; do not change guarded code/config or repeat completed tasks.',
            'Only B may fit/select. Inner validation statistics must never enter its own table, bins or normalizer.',
            'Old C/historical test cannot select CCW or certify improvement. Group annotations are evaluation-only.',
            'Read LOCKED_CANDIDATE.json before any retrospective CCW C replay; respect NO_GO.',
            'No fresh cohort means CONFIRMATION BLOCKED and no production integration; read INPUT_REQUEST.md.',
            'If feasible work is complete, read reports/FINAL_REPORT.md and preserve the frozen negative/positive diagnostics.',
            'The existing Uniform default and v5 opt-in G1 must remain unchanged.','']))
