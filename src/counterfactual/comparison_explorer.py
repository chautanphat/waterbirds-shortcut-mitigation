"""Read-only gallery for the compact, saved real three-model investigation.

No checkpoints are loaded. Reconstruct images with the existing compositor and
verify raw/model-view pixel hashes against the inference CSV before serving.
"""
from collections import defaultdict
import csv
from functools import lru_cache
from http.server import BaseHTTPRequestHandler
import io
import json
from pathlib import Path
from urllib.parse import parse_qs,urlsplit

import numpy as np
from PIL import Image
from torchvision import transforms
import yaml

from .common import ROOT,contained,pixel_hash,read_csv,sha256
from .compositor import aligned,composite,foreground_layer,mask_overlay
from .data import ForegroundRepository,BackgroundRepository
from .server import LocalHTTPServer
from src.datasets.waterbirds import waterbirds_transform

MODELS=('POTER','ERM_enhanced','DPE_legacy')
FILES={'replication':'predictions.csv','original':'original_test_predictions.csv'}
WEB=Path(__file__).parent/'web'


class ComparisonStore:
    def __init__(self,results,data_root):
        self.results,self.data_root=Path(results).resolve(),Path(data_root).resolve()
        integrity=json.loads((self.results/'integrity.json').read_text(encoding='utf-8'))['sha256']
        self.mild_v2=(self.results/'blur_config.json').is_file()
        config_name='blur_config.json' if self.mild_v2 else 'experiment_config.yaml'
        files=['predictions.csv'] if self.mild_v2 else list(FILES.values())
        for name in [*files,config_name,'checkpoint_audit.json','cohort.csv']:
            if sha256(self.results/name)!=integrity[name]:raise ValueError('Saved result integrity mismatch: '+name)
        self.config=yaml.safe_load((self.results/config_name).read_text(encoding='utf-8'))
        self.provenance=json.loads((self.results/'checkpoint_audit.json').read_text(encoding='utf-8'))
        self.foregrounds,self.backgrounds=ForegroundRepository(self.data_root),BackgroundRepository(self.data_root)
        self.preview=transforms.Compose(waterbirds_transform().transforms[:2])
        self.rows,self.images,self.catalog={},{},{}
        for cohort,filename in FILES.items():
            records=([r for r in self.rows['replication'] if r['control']=='original']
                if self.mild_v2 and cohort=='original' else read_csv(self.results/filename))
            images={}
            seen=set()
            for r in records:
                key=(r['image_id'],r['model_name'])
                if key in seen or r['model_name'] not in MODELS:raise ValueError('Duplicate/unsupported model record')
                seen.add(key)
                image=images.setdefault(r['image_id'],dict(image_id=r['image_id'],
                    source_image_id=r['source_image_id'],foreground_id=r['foreground_id'],species=r['species_id'],
                    true_y=int(r['true_y']) if r['true_y'] else None,source_y=int(r['source_y']),
                    severity=r['severity'],sigma=float(r['sigma']),background_id=r['background_id'],
                    bg_type=r['bg_type'],control=r['control'],pair_index=int(r['pair_index']),
                    raw_input_sha256=r['raw_input_sha256'],model_input_sha256=r['model_input_sha256'],predictions={}))
                if image['raw_input_sha256']!=r['raw_input_sha256'] or image['model_input_sha256']!=r['model_input_sha256']:
                    raise ValueError('Models did not see the same pixels')
                image['predictions'][r['model_name']]=dict(pred_y=int(r['pred_y']),
                    p_waterbird=float(r['p_waterbird']),p_landbird=float(r['p_landbird']),
                    score_landbird=float(r['logit_landbird']),score_waterbird=float(r['logit_waterbird']),
                    score_type=r['score_type'],seed=int(r['model_seed']))
            if any(set(im['predictions'])!=set(MODELS) for im in images.values()):raise ValueError('Incomplete image/model coverage')
            self.rows[cohort],self.images[cohort]=records,images
            self.catalog[cohort]=list(images.values())

    def info(self):
        return dict(models=list(MODELS),inference=False,integrity='verified_saved_csvs',
            experiment=self.results.name,severities=self.config['severities'],
            supplementary=self.config.get('supplementary_severities',[]),
            image_policy='Reconstruct with existing compositor; pixel hashes must match saved model inference',
            limitations=self.provenance['comparison_limitations'],
            cohorts={k:dict(prediction_rows=len(self.rows[k]),image_records=len(self.images[k]),
                source_foregrounds=len({i['source_image_id'] for i in self.images[k].values()}),
                species=len({i['species'] for i in self.images[k].values()}),
                failures_by_model={m:sum(im['true_y'] is not None and im['predictions'][m]['pred_y']!=im['true_y']
                    for im in self.images[k].values()) for m in MODELS}) for k in FILES})

    @lru_cache(maxsize=128)
    def asset(self,cohort,image_id,view):
        record=self.images[cohort][image_id]
        fg=self.foregrounds.by_id[record['foreground_id']]
        if view=='source' or record['control']=='original' and view in ('raw','model'):
            path=contained(self.data_root,fg['source_image_path'])
            if sha256(path)!=fg['source_sha256']:raise ValueError('Original source JPEG hash changed')
            with Image.open(path) as im:image=im.convert('RGB')
        elif view in ('mask','overlay','rgba'):
            rgba,_=self.foregrounds.get(record['foreground_id'])
            image=rgba.getchannel('A') if view=='mask' else mask_overlay(rgba) if view=='overlay' else rgba
        elif view in ('raw','model'):
            if record['control']=='background_only':
                background,_=self.backgrounds.get(record['background_id'])
                image=aligned(background,(256,256))
            else:
                rgba,_=self.foregrounds.get(record['foreground_id'])
                layer=foreground_layer(rgba,record['sigma'])
                background=Image.new('RGB',(256,256),(127,127,127)) if record['control']=='foreground_only' else self.backgrounds.get(record['background_id'])[0]
                image,_=composite(layer,background)
        else:raise ValueError('Unknown image view')
        if view in ('raw','model'):
            if pixel_hash(image)!=record['raw_input_sha256']:raise ValueError('Reconstructed raw pixels disagree with real inference')
            if view=='model':
                image=self.preview(image)
                if pixel_hash(image)!=record['model_input_sha256']:raise ValueError('Reconstructed 224 pixels disagree with real inference')
        buffer=io.BytesIO();image.save(buffer,'PNG')
        return buffer.getvalue()

    def download(self,cohort,bird):
        rows=[r for r in self.rows[cohort] if not bird or r['source_image_id']==bird]
        if not rows:raise ValueError('No saved rows for this source image')
        buffer=io.StringIO(newline='')
        writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        return buffer.getvalue().encode('utf-8-sig')


def create_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,status,body,kind='application/json',download=None):
            if kind=='application/json':body=json.dumps(body,ensure_ascii=False,allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
            self.send_header('X-Content-Type-Options','nosniff');self.send_header('Referrer-Policy','no-referrer')
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'")
            if download:self.send_header('Content-Disposition','attachment; filename="'+download+'"')
            self.end_headers();self.wfile.write(body)
        def do_GET(self):
            try:
                host=self.headers.get('Host','')
                allowed={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
                if host not in allowed: return self.send(403,dict(error='Localhost host header required'))
                origin=self.headers.get('Origin')
                if origin and origin not in {'http://'+h for h in allowed}:return self.send(403,dict(error='Foreign origin refused'))
                u=urlsplit(self.path);q={k:v[0] for k,v in parse_qs(u.query).items()}
                if u.path in ('/','/explorer'):
                    self.send(200,(WEB/'comparison.html').read_bytes(),'text/html; charset=utf-8')
                elif u.path=='/comparison.js':self.send(200,(WEB/'comparison.js').read_bytes(),'text/javascript; charset=utf-8')
                elif u.path=='/api/info':self.send(200,store.info())
                elif u.path=='/api/catalog':self.send(200,store.catalog[q.get('cohort','replication')])
                elif u.path=='/api/asset':self.send(200,store.asset(q['cohort'],q['id'],q.get('view','model')),'image/png')
                elif u.path=='/api/download':self.send(200,store.download(q['cohort'],q.get('bird')),'text/csv; charset=utf-8','failure_cases_'+q['cohort']+'.csv')
                else:self.send(404,dict(error='Not found'))
            except (KeyError,ValueError,OSError) as error:self.send(400,dict(error=str(error)))
        def do_POST(self):self.send(405,dict(error='This saved-results explorer is read-only; no inference or writes'))
    return Handler


def serve(results,data_root,port):
    store=ComparisonStore(results,data_root)
    server=LocalHTTPServer(('127.0.0.1',port),create_handler(store))
    print(f'Failure Case Explorer (three saved models) ready: http://127.0.0.1:{port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
