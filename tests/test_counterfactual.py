"""Deterministic scientific-contract tests; real checkpoint checks are opt-in."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
import urllib.request
import urllib.error

import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import models, transforms

from src.counterfactual.common import (DEFAULT_CHECKPOINT,DEFAULT_DATA,ROOT,pixel_hash,
    read_csv,write_csv,write_json,sha256)
from src.counterfactual.compositor import composite,foreground_layer,difference_heatmap
from src.counterfactual.data import prepare_dataset,ForegroundRepository,BackgroundRepository,metadata_index
from src.counterfactual.experiment import run_experiment,load_run,export_case,selected_backgrounds
from src.counterfactual.metrics import paired_metrics,summarize,annotate_failures
from src.counterfactual.model import ModelAdapter
from src.counterfactual.server import AppState,create_handler,LocalHTTPServer
from src.counterfactual.importers import import_foreground,import_prediction_csv


def cutout():
    # Synthetic geometric test fixture, never presented as an experimental bird.
    a=np.zeros((256,256,4),dtype=np.uint8)
    a[60:190,75:180,:3]=[210,80,35]
    a[60:190,75:180,3]=255
    a[59,75:180,:3]=[210,80,35]
    a[59,75:180,3]=100
    return Image.fromarray(a,'RGBA')


def fixture(root):
    root=Path(root)
    hf=root/'hf'
    hf.mkdir(parents=True)
    metadata,files=[],[]
    for y in (0,1):
        for a in (0,1):
            bird_id=str(1+2*y+a)
            filename=f'{y+1:03d}.FixtureBird/Fixture_{bird_id}.jpg'
            metadata.append(dict(img_id=bird_id,img_filename=filename,y=y,place=a,split=2,
                place_filename=f"/{'b/bamboo_forest' if a==0 else 'o/ocean'}/{bird_id}.jpg"))
            group=f"{'waterbird' if y else 'landbird'}_on_{'water' if a else 'land'}"
            for suffix in ('','_fg_only','_bg_only'):
                p=f'{group}{suffix}/{y}/{filename.replace("/","_")}.jpg'
                file=hf/p
                file.parent.mkdir(parents=True,exist_ok=True)
                im=Image.new('RGB',(256,256),(20+int(bird_id)*15,120+a*60,80+y*80))
                if suffix=='_fg_only':
                    im=Image.new('RGB',(256,256),(150,150,150))
                    ImageDraw.Draw(im).ellipse((75,60,180,190),fill=(220,50,35))
                im.save(file)
                files.append(p)
    write_csv(hf/'metadata.csv',metadata)
    files.append('metadata.csv')
    record=dict(repo_id='deterministic_test_fixture',revision='test-v1',files=files)
    report=prepare_dataset(root,record)
    return root,report


def metric_fixture():
    # Explicit numeric evaluation fixture, not an application prediction source.
    rows=[]
    for y in (0,1):
        for severity in ('clean','mild'):
            for a,bg in enumerate(('land','water')):
                # clean predicts background class; mild predicts waterbird everywhere.
                pred=a if severity=='clean' else 1
                logits=[2.,0.] if pred==0 else [0.,2.]
                rows.append(dict(row_id=f'{y}-{severity}-{bg}',model_id='numeric_fixture',source_bird_id=str(y),
                    foreground_id='fg'+str(y),severity=severity,pair_index=0,bg_type=bg,
                    foreground_layer_sha256=f'{y}-{severity}',source_split='test',true_y=y,
                    background_id=bg,control='natural',known_label=True,valid_target=True,status='completed',
                    pred_y=pred,is_correct=pred==y,logit_landbird=logits[0],logit_waterbird=logits[1],
                    p_landbird=.880797 if pred==0 else .119203,p_waterbird=.119203 if pred==0 else .880797,
                    raw_input_sha256=f'{y}-{severity}-{bg}',preprocess_fingerprint='fixture',
                    seed=0,score_type='logit',composite_quality_status='verified_original_cub'))
    return rows


class CounterfactualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_layer_invariant_all_severities_before_and_after_crop(self):
        rgba=cutout()
        for sigma in (0,1.5,4,9):
            f=foreground_layer(rgba,sigma)
            f2=foreground_layer(rgba,sigma)
            self.assertEqual(f.tobytes(),f2.tobytes())
            land,a=composite(f,Image.new('RGB',(300,500),(17,110,35)))
            water,b=composite(f,Image.new('RGB',(500,300),(20,90,230)))
            self.assertEqual(a['foreground_layer_sha256'],b['foreground_layer_sha256'])
            self.assertTrue(a['opaque_foreground_exact'] and a['outside_support_exact'])
            alpha=np.asarray(f)[...,3]
            for sl in (np.s_[:,:],np.s_[16:240,16:240]):
                opaque=alpha[sl]==255
                np.testing.assert_array_equal(np.asarray(land)[sl][opaque],np.asarray(water)[sl][opaque])
            self.assertEqual(np.asarray(f)[100,100,0],210)

    def test_alpha_normalized_blur_has_no_black_halo(self):
        f=foreground_layer(cutout(),9)
        a=np.asarray(f)
        support=a[...,3]>0
        np.testing.assert_allclose(a[...,:3][support],np.tile([210,80,35],(support.sum(),1)),atol=1)

    def test_invalid_masks_and_severity_fail(self):
        for mask in (Image.new('RGBA',(50,50),(0,0,0,0)),Image.new('RGBA',(50,50),(0,0,0,255)),Image.new('RGBA',(50,50),(0,0,0,128))):
            with self.assertRaises(ValueError):foreground_layer(mask,1.5)
        for sigma in (-1,float('nan'),31):
            with self.assertRaises(ValueError):foreground_layer(cutout(),sigma)
        with self.assertRaises(ValueError):foreground_layer(cutout(),0,1.2)

    def test_secondary_operators_are_separate_deterministic_and_invariant(self):
        rgba=np.array(cutout())
        rgba[80:160,85:170,0]=np.arange(85)[None,:]+40
        rgba=Image.fromarray(rgba,'RGBA')
        clean=foreground_layer(rgba,0)
        for operator,strength,identity in [('contrast',.4,1.),('occlusion',.3,0.)]:
            self.assertEqual(foreground_layer(rgba,0,operator=operator,strength=identity,seed=4).tobytes(),clean.tobytes())
            layer=foreground_layer(rgba,0,operator=operator,strength=strength,seed=4)
            self.assertNotEqual(layer.tobytes(),clean.tobytes())
            self.assertEqual(layer.tobytes(),foreground_layer(rgba,0,operator=operator,strength=strength,seed=4).tobytes())
            self.assertEqual(layer.getchannel('A').tobytes(),clean.getchannel('A').tobytes())
            _,land=composite(layer,Image.new('RGB',(300,500),'green'))
            _,water=composite(layer,Image.new('RGB',(500,300),'blue'))
            self.assertEqual(land['foreground_layer_sha256'],water['foreground_layer_sha256'])
            with self.assertRaises(ValueError):foreground_layer(rgba,1.5,operator=operator,strength=strength)

    def test_pair_metrics_wga_denominators_controls_and_ci(self):
        rows=metric_fixture()
        control=dict(rows[0],row_id='control',control='background_only',valid_target=False,known_label=False,
            true_y=None,is_correct=None,pred_y=1)
        rows.append(control)
        pairs=paired_metrics(rows)
        self.assertEqual(len(pairs),4)
        self.assertEqual([p['delta_margin'] for p in pairs if p['severity']=='clean'],[4.,4.])
        self.assertEqual([p['amplification'] for p in pairs if p['severity']=='mild'],[-4.,-4.])
        stats=summarize(rows,pairs,bootstrap_samples=50)
        clean=next(s for s in stats if s['severity']=='clean' and s['control']=='natural')
        self.assertEqual(clean['accuracy'],.5)
        self.assertEqual(clean['wga'],0.)
        self.assertEqual(clean['valid_target_n'],4)
        self.assertEqual(clean['D_s_ci95'],[4.,4.])
        bg=next(s for s in stats if s['control']=='background_only')
        self.assertIsNone(bg['accuracy'])
        self.assertIsNone(bg['wga'])
        self.assertEqual(bg['valid_target_n'],0)
        self.assertFalse(annotate_failures(rows,pairs)[-1]['is_failure'])
        incomplete=summarize(rows[:2],bootstrap_samples=10)[0]
        self.assertIsNone(incomplete['wga'])
        self.assertEqual(incomplete['wga_status'],'undefined/incomplete')

    def test_unknown_target_never_enters_accuracy(self):
        rows=[dict(r,known_label=False,valid_target=False,true_y=None,is_correct=None) for r in metric_fixture()]
        for s in summarize(rows,bootstrap_samples=10):
            self.assertIsNone(s['accuracy']);self.assertIsNone(s['wga']);self.assertIsNone(s['D_s'])

    def test_metadata_join_uses_canonical_labels_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,report=fixture(tmp)
            self.assertEqual(report['metadata_join_coverage'],1.)
            self.assertEqual(report['backgrounds'],{'land':2,'water':2})
            rows=ForegroundRepository(root).rows
            self.assertEqual({r['y'] for r in rows},{'0','1'})
            self.assertEqual({r['original_split'] for r in rows},{'test'})
            self.assertEqual(report['invalid_files'],0)
            with self.assertRaises(ValueError):metadata_index([dict(img_id='1',img_filename='a.jpg',y='2',place='0',split='2')])

    def test_no_model_workflow_export_reconciliation_and_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,report=fixture(Path(tmp)/'data')
            cfg=dict(foreground_ids=['hf-fg-1'],seed=8,allow_approximate=True,
                include_background_only=True,include_neutral=True)
            runs=Path(tmp)/'runs'
            first=run_experiment(cfg,[],data_root=root,output_root=runs)
            loaded=load_run(first['path'])
            self.assertEqual(len(loaded['rows']),15)
            self.assertEqual(len([r for r in loaded['rows'] if r['control']=='natural']),8)
            self.assertEqual({r['status'] for r in loaded['rows']},{'not_run'})
            self.assertTrue(all(r['p_waterbird']=='' for r in loaded['rows']))
            self.assertFalse(any(r['is_failure'] for r in loaded['rows']))
            self.assertTrue(all(r['true_y']=='' and r['known_label']=='False' for r in loaded['rows'] if r['control']=='background_only'))
            packet=export_case(first['path'],'hf-fg-1')
            csv_rows=read_csv(packet/'cases.csv')
            json_rows=json.loads((packet/'case_packet.json').read_text())['rows']
            self.assertEqual([r['row_id'] for r in csv_rows],[r['row_id'] for r in json_rows])
            self.assertEqual(len(csv_rows),15)
            self.assertTrue((packet/'montage.png').is_file())
            with Image.open(packet/'montage.png') as im:
                self.assertEqual(json.loads(im.info['row_ids']),[r['row_id'] for r in csv_rows])
            second=run_experiment(cfg,[],data_root=root,output_root=runs)
            a=read_csv(Path(first['path'])/'manifest.csv');b=read_csv(Path(second['path'])/'manifest.csv')
            self.assertNotEqual(first['run_id'],second['run_id'])
            self.assertEqual([r['raw_input_sha256'] for r in a],[r['raw_input_sha256'] for r in b])
            self.assertEqual([r['source_split'] for r in a],['test']*15)
            with (Path(first['path'])/'predictions.csv').open('a') as stream:stream.write('\n')
            with self.assertRaises(ValueError):load_run(first['path'])

    def test_cross_split_backgrounds_and_changed_assets_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,_=fixture(tmp)
            repo=BackgroundRepository(root)
            with self.assertRaises(ValueError):selected_backgrounds(repo,{'background_ids':['hf-bg-1','hf-bg-2']},'val')
            r=repo.rows[0]
            (root/r['source_path']).write_bytes(b'changed')
            with self.assertRaises(ValueError):repo.get(r['background_id'])

    def test_external_unknown_labels_and_mask_correction_identity(self):
        def url(image):
            b=io.BytesIO();image.save(b,'PNG')
            return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
        with tempfile.TemporaryDirectory() as tmp:
            root,_=fixture(tmp)
            source=ForegroundRepository(root).rows[0]
            with Image.open(root/source['source_image_path']) as im:rgb=im.convert('RGB')
            with Image.open(root/source['mask_path']) as im:mask=im.convert('L')
            external=import_foreground(root,url(rgb),url(mask),name='external fixture')
            self.assertEqual(external['y'],'')
            self.assertEqual(external['original_split'],'unknown')
            result=run_experiment(dict(foreground_ids=[external['foreground_id']],allow_approximate=True,
                seed=0,external_background_split='test'),[],data_root=root,output_root=root/'external_runs')
            unknown_run=load_run(result['path'])
            self.assertTrue(all(r['provenance_status']=='external_unknown_ground_truth' for r in unknown_run['rows']))
            self.assertTrue(all(s['accuracy'] is None for s in unknown_run['summary']))
            corrected=import_foreground(root,url(rgb),url(mask),parent=source['foreground_id'])
            self.assertEqual(str(corrected['y']),source['y'])
            self.assertEqual(corrected['bird_source_id'],source['bird_source_id'])
            self.assertEqual(corrected['original_split'],source['original_split'])
            self.assertEqual(corrected['quality_status'],'approximate_manual_mask')
            with self.assertRaises(ValueError):import_foreground(root,url(Image.new('RGB',rgb.size,'white')),url(mask),parent=source['foreground_id'])
            self.assertTrue((root/source['rgba_path']).exists())

    def test_local_http_complete_read_only_and_origin_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,_=fixture(Path(tmp)/'data')
            state=AppState(root,Path(tmp)/'runs',[])
            server=LocalHTTPServer(('127.0.0.1',0),create_handler(state))
            with self.assertRaises(OSError):LocalHTTPServer(('127.0.0.1',server.server_port),create_handler(state))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base=f'http://127.0.0.1:{server.server_port}'
            try:
                for path in ('/','/explorer','/lab.js','/explorer.js','/common.js','/mixer_core.js','/api/catalog','/api/info'):
                    with urllib.request.urlopen(base+path) as response:self.assertEqual(response.status,200)
                req=urllib.request.Request(base+'/api/run',data=b'{}',headers={'Origin':'http://evil.example','Content-Type':'application/json'})
                with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(req)
                self.assertEqual(error.exception.code,403)
                with self.assertRaises(urllib.error.HTTPError):urllib.request.urlopen(base+'/api/asset?scope=data&path=.hf_cache/token')
            finally:
                server.shutdown();server.server_close();state.executor.shutdown(wait=True)


@unittest.skipUnless(os.environ.get('RUN_REAL_INFERENCE')=='1','Set RUN_REAL_INFERENCE=1 for real-checkpoint integration')
class RealCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.device='cuda' if torch.cuda.is_available() else 'cpu'
        cls.adapter=ModelAdapter(DEFAULT_CHECKPOINT,cls.device,metadata=DEFAULT_DATA/'metadata.csv')
        cls.repo=ForegroundRepository(DEFAULT_DATA)
        with Image.open(DEFAULT_DATA/cls.repo.rows[0]['source_image_path']) as im:cls.image=im.convert('RGB')

    def test_original_runner_loading_preprocessing_single_batch_parity(self):
        # Independent frozen reference of scripts/poter_preview.py before refactor.
        state=torch.load(DEFAULT_CHECKPOINT,map_location='cpu',weights_only=True)
        model=models.resnet50(weights=None);model.fc=torch.nn.Linear(2048,2)
        model.load_state_dict(state['model'],strict=True);model.to(self.device).eval()
        crop=transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop((224,224))])
        tensor=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        with torch.inference_mode():expected=model(torch.stack([tensor(crop(self.image))]*2).to(self.device))
        batch=self.adapter.predict_batch([self.image,self.image],batch_size=2)
        single=self.adapter.predict_batch([self.image],batch_size=1)[0]
        repeat=self.adapter.predict_batch([self.image],batch_size=1)[0]
        np.testing.assert_allclose(batch[0]['logits'],expected[0].cpu().numpy(),atol=2e-5,rtol=2e-5)
        np.testing.assert_allclose(batch[0]['probabilities'],expected.softmax(1)[0].cpu().numpy(),atol=2e-6,rtol=2e-5)
        # cuDNN uses different convolution reductions for batch 1 and batch 2;
        # observed maximum was 4.70e-5 logits / 8.17e-6 probabilities on CUDA.
        np.testing.assert_allclose(single['logits'],batch[0]['logits'],atol=1e-4,rtol=2e-5)
        np.testing.assert_allclose(single['probabilities'],batch[0]['probabilities'],atol=2e-5,rtol=2e-5)
        np.testing.assert_allclose(single['logits'],repeat['logits'],atol=1e-6,rtol=1e-6)
        self.assertEqual(single['preview'].tobytes(),crop(self.image).tobytes())
        self.assertEqual(self.adapter.class_index_to_name,{0:'landbird',1:'waterbird'})
        self.assertEqual(self.adapter.epoch,258)

    def test_existing_runner_http_contract_matches_shared_adapter(self):
        from scripts.poter_preview import create_app
        handler=create_app(DEFAULT_CHECKPOINT,DEFAULT_DATA/'metadata.csv',DEFAULT_DATA,self.device)
        server=ThreadingHTTPServer(('127.0.0.1',0),handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        b=io.BytesIO();self.image.save(b,'PNG');url='data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
        try:
            req=urllib.request.Request(f'http://127.0.0.1:{server.server_port}/api/predict',data=json.dumps(dict(original=url,edited=url)).encode(),headers={'Content-Type':'application/json'})
            result=json.load(urllib.request.urlopen(req))
            expected=self.adapter.predict_batch([self.image])[0]
            np.testing.assert_allclose(list(result['edited']['probabilities'].values()),expected['probabilities'],atol=2e-5,rtol=2e-5)
        finally:server.shutdown();server.server_close()

    def test_real_four_by_two_end_to_end_and_store_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=json.loads((ROOT/'configs/counterfactual_lab/waterbird_demo.json').read_text())
            result=run_experiment(cfg,[self.adapter],data_root=DEFAULT_DATA,output_root=tmp)
            run=load_run(result['path'])
            natural=[r for r in run['rows'] if r['control']=='natural']
            self.assertEqual(len(natural),8)
            self.assertTrue(all(r['status']=='completed' for r in natural))
            self.assertEqual(len(run['pairs']),4)
            self.assertTrue(all(s['accuracy'] is None for s in run['summary'] if s['control'] not in ('natural','original')))
            self.assertTrue(all(s['wga'] is None for s in run['summary']))
            for r in natural:
                with Image.open(Path(result['path'])/r['model_input_path']) as im:
                    self.assertEqual(im.size,(224,224));self.assertEqual(pixel_hash(im),r['model_input_sha256'])

    def test_secondary_operators_real_inference_in_separate_runs(self):
        for operator in ('contrast','occlusion'):
            with tempfile.TemporaryDirectory() as tmp:
                cfg=dict(foreground_ids=['hf-fg-1'],operator=operator,severities={'clean':0,'mild':0},
                    include_original=True,include_background_only=True,include_neutral=True,seed=2026)
                result=run_experiment(cfg,[self.adapter],data_root=DEFAULT_DATA,output_root=tmp)
                run=load_run(result['path'])
                natural=[r for r in run['rows'] if r['control']=='natural']
                self.assertEqual(len(natural),4)
                self.assertEqual({r['operator'] for r in natural},{operator})
                self.assertEqual({float(r['sigma']) for r in natural},{0.})
                self.assertTrue(all(r['status']=='completed' for r in natural))
                self.assertEqual(len(run['pairs']),2)


if __name__=='__main__':unittest.main()
