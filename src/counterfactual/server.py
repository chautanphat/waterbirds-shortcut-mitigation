"""Local apps over the repository's existing standard-library HTTP stack."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import socket
import traceback
from urllib.parse import parse_qs, urlsplit
import uuid
import zipfile

import torch

from .common import DEFAULT_CHECKPOINT, DEFAULT_DATA, DEFAULT_RUNS, ROOT, contained, read_csv, write_json,sha256
from .data import BackgroundRepository, ForegroundRepository
from .experiment import export_case, load_run, run_experiment
from .importers import import_foreground
from .model import ModelAdapter

TAGS = ['potential_shortcut','possible_preprocessing_artifact','mask_problem','label_uncertain','no_bird_control','needs_review']


class LocalHTTPServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can send traffic unpredictably to two listeners.
    # Keep each research app bound to exactly one process.
    allow_reuse_address=os.name!='nt'

    def server_bind(self):
        if hasattr(socket,'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()


class AppState:
    def __init__(self, data_root, runs_root, adapters, app='lab'):
        self.data_root, self.runs_root = Path(data_root).resolve(), Path(runs_root).resolve()
        self.adapters, self.app = adapters, app
        self.jobs, self.lock = {}, threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='counterfactual')
        self.runs_root.mkdir(parents=True,exist_ok=True)

    def submit(self, config, model_ids):
        adapters = [a for a in self.adapters if a.model_id in model_ids]
        if set(model_ids) != {a.model_id for a in adapters}:
            raise ValueError('Selected model is not configured')
        job_id=uuid.uuid4().hex
        with self.lock:
            self.jobs[job_id]=dict(status='queued')
        def progress(value):
            with self.lock:
                self.jobs[job_id].update(value)
        def work():
            progress(dict(status='running'))
            try:
                result=run_experiment(config,adapters,data_root=self.data_root,output_root=self.runs_root,progress=progress)
                progress(dict(status='completed',result=result))
            except Exception as error:
                traceback.print_exc()
                progress(dict(status='failed',error=str(error)))
        self.executor.submit(work)
        return job_id


def create_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass

        def send(self,status,value,kind='application/json'):
            body=json.dumps(value,allow_nan=False).encode() if kind=='application/json' else value
            self.send_response(status)
            self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Referrer-Policy','no-referrer')
            self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'")
            self.end_headers()
            self.wfile.write(body)

        def checked_run(self,run_id):
            if not run_id or Path(run_id).name != run_id or run_id in ('.','..'):
                raise ValueError('Choose a valid saved run ID')
            p=contained(state.runs_root,run_id)
            if not (p/'run_config.json').is_file():
                raise ValueError('Run not found')
            return p

        def do_GET(self):
            try:
                self.get()
            except (ValueError,KeyError,OSError) as error:
                self.send(400,dict(error=str(error)))
            except Exception as error:
                traceback.print_exc()
                self.send(500,dict(error=str(error)))

        def get(self):
            u=urlsplit(self.path)
            q={k:v[0] for k,v in parse_qs(u.query).items()}
            static={'/':'lab.html' if state.app=='lab' else 'explorer.html', '/lab':'lab.html',
                '/explorer':'explorer.html','/app.css':'app.css','/lab.js':'lab.js','/explorer.js':'explorer.js',
                '/common.js':'common.js','/mixer_core.js':'mixer_core.js'}
            if u.path in static:
                name=static[u.path]
                file=ROOT/'scripts/mixer_core.js' if name=='mixer_core.js' else ROOT/'src/counterfactual/web'/name
                self.send(200,file.read_bytes(),mimetypes.guess_type(name)[0] or 'text/plain')
            elif u.path=='/api/info':
                audit=state.data_root/'dataset_audit.json'
                self.send(200,dict(models=[a.info() for a in state.adapters],tags=TAGS,
                    dpe_status='configured' if any(a.method=='dpe' for a in state.adapters) else 'DPE adapter not configured: provide saved original-DPE heads and their hash-matched ERM backbone',
                    dataset=json.loads(audit.read_text(encoding='utf-8')) if audit.exists() else None,
                    setup_command='python scripts/setup_waterbirds_data.py --download-cub',app=state.app))
            elif u.path=='/api/catalog':
                fg,bg=ForegroundRepository(state.data_root),BackgroundRepository(state.data_root)
                self.send(200,dict(foregrounds=fg.rows,backgrounds=bg.rows))
            elif u.path=='/api/runs':
                records=[]
                for p in sorted(state.runs_root.iterdir(),reverse=True):
                    if p.is_dir() and (p/'run_config.json').exists():
                        config=json.loads((p/'run_config.json').read_text(encoding='utf-8'))
                        status=json.loads((p/'status.json').read_text(encoding='utf-8'))
                        records.append(dict(run_id=p.name,status=status,source_split=config['source_split'],
                            foregrounds=len(config['foreground_ids']),models=[a['model_id'] for a in config['models']],
                            parent_run_id=config.get('parent_run_id')))
                self.send(200,records)
            elif u.path=='/api/run':
                self.send(200,load_run(self.checked_run(q['id'])))
            elif u.path=='/api/job':
                with state.lock:
                    job=dict(state.jobs[q['id']])
                self.send(200,job)
            elif u.path=='/api/reviews':
                run=self.checked_run(q['id'])
                p=run/'reviews.json'
                rows=json.loads(p.read_text(encoding='utf-8')) if p.exists() else []
                rows += [json.loads(f.read_text(encoding='utf-8')) for f in sorted((run/'reviews').glob('*.json'))]
                self.send(200,rows)
            elif u.path=='/api/asset':
                scope=q.get('scope')
                value=q['path']
                if scope=='data':
                    if not value.startswith(('hf/','foregrounds/','external_foregrounds/','local_backgrounds/')):
                        raise ValueError('Not a public dataset image')
                    p=contained(state.data_root,value)
                    if p.suffix.lower() not in ('.png','.jpg','.jpeg','.webp'):
                        raise ValueError('Only dataset image assets can be served')
                elif scope=='run':
                    run=self.checked_run(q['id'])
                    p=contained(run,value)
                    if p.suffix.lower() not in ('.png','.jpg','.pdf','.csv','.json','.md'):
                        raise ValueError('Unsupported exported asset')
                    integrity=run/'integrity.json'
                    if integrity.exists():
                        digest=json.loads(integrity.read_text(encoding='utf-8'))['assets_sha256'].get(p.relative_to(run).as_posix())
                        if digest and sha256(p)!=digest:
                            raise ValueError('Saved image asset changed since experiment completion')
                else:
                    raise ValueError('Invalid asset scope')
                self.send(200,p.read_bytes(),mimetypes.guess_type(p.name)[0] or 'application/octet-stream')
            elif u.path=='/api/download':
                p=self.checked_run(q['id'])
                b=io.BytesIO()
                with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
                    for file in sorted(p.rglob('*')):
                        if file.is_file() and file.resolve().is_relative_to(p.resolve()):
                            z.write(file,file.relative_to(p).as_posix())
                self.send(200,b.getvalue(),'application/zip')
            else:
                self.send(404,dict(error='Not found'))

        def do_POST(self):
            try:
                host=self.headers.get('Host','')
                if host.split(':')[0] not in ('127.0.0.1','localhost'):
                    self.send(403,dict(error='Open the app at its localhost URL'));return
                if self.headers.get('Origin') not in (None,'http://'+host):
                    self.send(403,dict(error='Origin must match the localhost app'));return
                length=int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 45_000_000:
                    self.send(413,dict(error='Request exceeds 45 MB'));return
                request=json.loads(self.rfile.read(length))
                path=urlsplit(self.path).path
                if path=='/api/run':
                    self.send(202,dict(job_id=state.submit(request['config'],request.get('model_ids',[]))))
                elif path=='/api/export':
                    run=self.checked_run(request['run_id'])
                    dest=export_case(run,request['foreground_id'],request.get('row_ids'))
                    self.send(200,dict(path=dest.relative_to(run).as_posix()))
                elif path=='/api/import-foreground':
                    with state.lock:
                        result=import_foreground(state.data_root,request['image'],request['mask'],
                            name=request.get('name','external'),parent=request.get('parent'))
                    self.send(201,result)
                elif path=='/api/review':
                    run=self.checked_run(request['run_id'])
                    if request.get('reviewer') not in ('reviewer_1','reviewer_2','adjudicator'):
                        raise ValueError('Choose reviewer_1, reviewer_2, or adjudicator')
                    if not set(request.get('tags',[])).issubset(TAGS):
                        raise ValueError('Unknown review tag')
                    if request['row_id'] not in {r['row_id'] for r in read_csv(run/'predictions.csv')}:
                        raise ValueError('Unknown prediction row ID')
                    from datetime import datetime,timezone
                    # One immutable event per review avoids lost updates across
                    # the separate Lab/Explorer processes or reviewer sessions.
                    folder=run/'reviews';folder.mkdir(exist_ok=True)
                    write_json(folder/(uuid.uuid4().hex+'.json'),dict(row_id=request['row_id'],
                        reviewer=request['reviewer'],tags=request.get('tags',[]),
                        note=str(request.get('note',''))[:10000],created_at=datetime.now(timezone.utc).isoformat()))
                    self.send(201,dict(saved=True))
                else:
                    self.send(404,dict(error='Not found'))
            except (ValueError,KeyError,OSError) as error:
                self.send(400,dict(error=str(error)))
            except Exception as error:
                traceback.print_exc()
                self.send(500,dict(error=str(error)))
    return Handler


def main(app='lab'):
    p=argparse.ArgumentParser(description='Counterfactual Lab' if app=='lab' else 'Failure Case Explorer')
    p.add_argument('--data-root',type=Path,default=DEFAULT_DATA)
    p.add_argument('--runs-root',type=Path,default=DEFAULT_RUNS)
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    p.add_argument('--device',default='cpu')
    p.add_argument('--port',type=int,default=8786 if app=='lab' else 8787)
    p.add_argument('--no-model',action='store_true',help='Composition/browsing only; predictions remain not_run')
    p.add_argument('--erm-checkpoint',type=Path)
    p.add_argument('--dpe-heads',type=Path)
    p.add_argument('--dpe-backbone',type=Path)
    p.add_argument('--comparison-root',type=Path,
        help='Read saved three-model investigation CSVs and reconstruct verified images without model inference')
    args=p.parse_args()
    torch.set_num_threads(4)
    if args.comparison_root:
        if app!='explorer':
            p.error('--comparison-root is only supported by Failure Case Explorer')
        from .comparison_explorer import serve
        return serve(args.comparison_root,args.data_root,args.port)
    adapters=[]
    if not args.no_model:
        if args.checkpoint.is_file():
            adapters.append(ModelAdapter(args.checkpoint,args.device,metadata=args.data_root/'metadata.csv'))
        else:
            print(f'POTER checkpoint missing: {args.checkpoint}. Composition available; predictions will be not_run.',flush=True)
    if args.erm_checkpoint:
        adapters.append(ModelAdapter(args.erm_checkpoint,args.device,method='erm'))
    if bool(args.dpe_heads)!=bool(args.dpe_backbone):
        p.error('--dpe-heads and --dpe-backbone must be supplied together')
    if args.dpe_heads:
        adapters.append(ModelAdapter(args.dpe_backbone,args.device,method='dpe',dpe_heads=args.dpe_heads,metadata=args.data_root/'metadata.csv'))
    state=AppState(args.data_root,args.runs_root,adapters,app)
    server=LocalHTTPServer(('127.0.0.1',args.port),create_handler(state))
    print(f'{p.description} ready: http://127.0.0.1:{args.port}',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.executor.shutdown(wait=True)
