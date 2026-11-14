"""Immutable task commits with source/config/input guards and atomic handoffs."""
import csv
import io
import json
import os
import signal
import traceback
from datetime import datetime, timezone
from pathlib import Path
from experiments.discrete_weighting.store import atomic, dump, key, sha


def csvwrite(path, rows):
    rows = list(rows); cols = list(dict.fromkeys(k for r in rows for k in r)) or ['empty']
    s = io.StringIO(newline=''); w = csv.DictWriter(s, fieldnames=cols); w.writeheader()
    for r in rows: w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    atomic(path, s.getvalue())


class Journal:
    def __init__(self, root, manifest, max_tasks=None):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest; self.records = {}; self.count = 0; self.limit = max_tasks; self.stop = False
        path = self.root/'manifest.json'
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError('Input/source/config hash mismatch; preserve this run and explicitly version any revision')
        if not path.exists(): dump(path, manifest)
        for path in (self.root/'completed').glob('*.json'):
            r = json.loads(path.read_text())
            for a, v in r['artifacts'].items():
                p = self.root/a
                if not p.is_file() or p.stat().st_size != v['bytes'] or sha(p) != v['sha256']:
                    raise ValueError('Committed artifact corrupt/missing: '+str(p))
            self.records[r['id']] = r
        for r in self.records.values():
            if any(d not in self.records for d in r['dependencies']): raise ValueError('Committed dependency missing')
        previous = self.root/'state.json'
        self.current = json.loads(previous.read_text()) if previous.exists() else {}
        for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: setattr(self, 'stop', True))
        self.status(self.current.get('status', 'READY'), self.current.get('next_pending', 'acceptance/input verification'))

    def status(self, status, next_task, **extra):
        self.current.update(extra)
        self.current.update(status=status, next_pending=next_task, completed=len(self.records),
                            updated_utc=datetime.now(timezone.utc).isoformat(), manifest_sha256=sha(self.root/'manifest.json'),
                            task_DAG={k: v['dependencies'] for k,v in self.records.items()},
                            done_configs=[k for k in self.records if k.startswith(('screen:', 'grid:', 'outer:'))],
                            resume_command='.venv/Scripts/python.exe -m experiments.score_audit_v4.run --phase all --resume')
        dump(self.root/'state.json', self.current)
        text = '# Score audit v4 checkpoint\n\n'+json.dumps({k:v for k,v in self.current.items() if k not in ('task_DAG','done_configs')}, indent=2)+'\n\n'
        text += 'Completion markers and checksums are authoritative. Preserve v1/v2/v3 outputs. Never tune on test/groups. Phase 2 folds were previously inspected in v3; disclose this.\n'
        atomic(self.root/'HANDOFF_STATUS.md', text)
        atomic(self.root/'NEXT_AGENT_PROMPT.md', '\n'.join([
            'Continue the score-mechanism/piecewise v4 workflow from this directory.',
            'Read manifest.json, PROTOCOL_DECISIONS.md, state.json and HANDOFF_STATUS.md.',
            'Use .venv/Scripts/python.exe from the repository root.',
            'Run .venv/Scripts/python.exe -m experiments.score_audit_v4.run --phase all --resume',
            'Valid completed markers must be reused; do not edit guarded sources or silently change parameters.',
            'Groups are evaluation-only; historical test may be opened only by Phase 0 diagnostics.',
            'Reuse verified nested heads, exact hashes and ordered B/C predictions; no OOF-slot transfer.',
            'Production integration requires the preregistered primary candidate to pass every promotion criterion.',
            'If finished, read FINAL_REPORT.md and do not repeat experiments.', '']))

    def task(self, id_, deps, fn):
        if id_ in self.records: return self.records[id_]['result']
        if any(d not in self.records for d in deps): raise ValueError('Missing dependencies '+id_)
        if self.stop or self.limit is not None and self.count >= self.limit:
            self.status('INTERRUPTED', id_); raise InterruptedError(id_)
        self.status('RUNNING', id_, active_task=id_, active_phase=id_.split(':')[0])
        try:
            result, paths = fn()
            artifacts = {Path(p).relative_to(self.root).as_posix(): dict(sha256=sha(p), bytes=Path(p).stat().st_size) for p in paths}
            r = dict(id=id_, dependencies=deps, result=result, artifacts=artifacts,
                     manifest_sha256=sha(self.root/'manifest.json'), completed_utc=datetime.now(timezone.utc).isoformat())
            dump(self.root/'completed'/f'{key(id_)[:20]}.json', r)
            self.records[id_] = r; self.count += 1
            with (self.root/'events.jsonl').open('a') as f:
                f.write(json.dumps(dict(event='COMMITTED', id=id_, utc=r['completed_utc']))+'\n'); f.flush(); os.fsync(f.fileno())
            csvwrite(self.root/'results/task_results.csv', [dict(task=k, **v['result']) for k,v in self.records.items()])
            self.status('RUNNING', 'next dependency-ready task', last_completed=id_, active_task=None, error=None)
            print('COMMITTED '+id_, flush=True); return result
        except Exception as exc:
            dump(self.root/'failures'/f'{key(id_)[:20]}.json', dict(task=id_, error=str(exc), traceback=traceback.format_exc()))
            self.status('FAILED', id_, error=str(exc)); raise
