"""The existing CCW durable journal with a separate diagnostic namespace."""
import json
from datetime import datetime,timezone
from pathlib import Path
from experiments.geometry_aggregation_v5.journal import Journal as BaseJournal,csvwrite
from experiments.geometry_aggregation_v5.persistence import atomic,dump
from experiments.discrete_weighting.store import sha

BASE=Path('experiments/ccw_v6')
ROOT=BASE/'exploratory_c'
BANNER='EXPLORATORY_ONLY — REUSED DEVELOPMENT — NOT INDEPENDENT CONFIRMATION'
COMMAND='.venv/Scripts/python.exe -m experiments.ccw_v6.run --phase exploratory_c --resume'


class Journal(BaseJournal):
    def __init__(self,root,manifest,max_tasks=None,bridge=True):
        self.plan=manifest.get('planned_DAG',{});self.bridge=bridge
        super().__init__(root,manifest,max_tasks)

    def status(self,status,next_task,**extra):
        self.current.update(extra)
        self.current.update(status=status,next_pending=next_task,completed=len(self.records),role=BANNER,
            original_B_decision='NO_GO',production_promotion=False,resume_command=COMMAND,
            manifest_sha256=sha(self.root/'manifest.json'),updated_utc=datetime.now(timezone.utc).isoformat(),
            task_DAG={k:dict(dependencies=v,status='COMPLETE' if k in self.records else 'ACTIVE' if status=='RUNNING' and k==next_task else 'PENDING') for k,v in self.plan.items()})
        dump(self.root/'state.json',self.current)
        compact={k:v for k,v in self.current.items() if k!='task_DAG'}
        text='# CCW v6 separate exploratory C status\n\n**'+BANNER+'**\n\n'+json.dumps(compact,indent=2)+'\n\n'+COMMAND+'\n\nOriginal B NO_GO remains unchanged. This requested diagnostic cannot select a recipe, trigger promotion, or change Uniform/G1 production inference.\n'
        atomic(self.root/'HANDOFF_STATUS.md',text)
        prompt='\n'.join(['Continue only the explicitly requested CCW exploratory C diagnostic.',
            'Read experiments/ccw_v6/exploratory_c/state.json, PROTOCOL_DECISIONS.md and HANDOFF_STATUS.md.',
            'Run '+COMMAND,
            'Reuse all valid completion receipts and source/input/config fingerprints; never repeat completed fits.',
            'Original experiments/ccw_v6/LOCKED_CANDIDATE.json must remain byte-identical NO_GO.',
            'Fit only B; C inference accepts probabilities/distances and B metadata, never labels/groups.',
            'Use C labels/groups only after predictions commit. Preserve the full exploratory banner.',
            'Exactly five frozen methods; no parameter optimization or production promotion.',
            'If complete, read EXPLORATORY_C_REPORT.md and do not rerun scientific tasks.',''])
        atomic(self.root/'NEXT_AGENT_PROMPT.md',prompt)
        if self.bridge:
            original=BASE/'LOCKED_CANDIDATE.json'
            assert sha(original)==self.manifest['original_NO_GO_sha256'] and json.loads(original.read_text())['status']=='NO_GO'
            statepath=BASE/'state.json';old=json.loads(statepath.read_text());old['exploratory_c']={k:v for k,v in compact.items() if k in ['status','next_pending','completed','resume_command','role','original_B_decision','production_promotion','report']};dump(statepath,old)
            original_dir=ROOT/'original_state'
            for name in ['HANDOFF_STATUS.md','NEXT_AGENT_PROMPT.md']:
                prior=(original_dir/name).read_text(encoding='utf-8');current=text if name=='HANDOFF_STATUS.md' else prompt
                atomic(BASE/name,'# New user-authorized diagnostic addendum\n\n'+current+'\n---\n\n# Preserved original NO_GO handoff (historical workflow)\n\n'+prior)
