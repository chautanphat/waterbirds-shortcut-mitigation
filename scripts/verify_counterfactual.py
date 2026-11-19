"""Persist an independently checked real-POTER parity report and controlled demo."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from torchvision import models,transforms
from src.counterfactual.common import DEFAULT_DATA,DEFAULT_CHECKPOINT,DEFAULT_RUNS,write_json,pixel_hash
from src.counterfactual.data import ForegroundRepository
from src.counterfactual.model import ModelAdapter
from src.counterfactual.experiment import run_experiment,load_run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=DEFAULT_DATA)
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',type=Path,default=Path('experiments/counterfactual_verification'))
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    adapter=ModelAdapter(args.checkpoint,args.device,metadata=args.data_root/'metadata.csv')
    repo=ForegroundRepository(args.data_root)
    with Image.open(args.data_root/repo.by_id['hf-fg-1']['source_image_path']) as im:
        image=im.convert('RGB')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    legacy=models.resnet50(weights=None);legacy.fc=torch.nn.Linear(2048,2)
    legacy.load_state_dict(state['model'],strict=True);legacy.to(args.device).eval()
    crop=transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop((224,224))])
    tensor=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    with torch.inference_mode():
        logits=legacy(torch.stack([tensor(crop(image))]*2).to(args.device))
    expected=logits.cpu().numpy()
    batch=adapter.predict_batch([image,image],batch_size=2)
    single=adapter.predict_batch([image],batch_size=1)[0]
    repeat=adapter.predict_batch([image],batch_size=1)[0]
    np.testing.assert_allclose(batch[0]['logits'],expected[0],atol=2e-5,rtol=2e-5)
    np.testing.assert_allclose(single['logits'],batch[0]['logits'],atol=1e-4,rtol=2e-5)
    np.testing.assert_allclose(single['probabilities'],batch[0]['probabilities'],atol=2e-5,rtol=2e-5)
    np.testing.assert_allclose(single['probabilities'],repeat['probabilities'],atol=1e-6,rtol=1e-6)
    assert single['preview'].tobytes()==crop(image).tobytes()
    report=dict(model=adapter.info(),source_bird_id='1',preview_pixels_identical=True,
        legacy_same_batch_max_logit_error=float(np.max(np.abs(np.array(batch[0]['logits'])-expected[0]))),
        batch_size_1_vs_2_max_logit_error=float(np.max(np.abs(np.array(single['logits'])-batch[0]['logits']))),
        batch_size_1_vs_2_max_probability_error=float(np.max(np.abs(np.array(single['probabilities'])-batch[0]['probabilities']))),
        repeat_max_probability_error=float(np.max(np.abs(np.array(single['probabilities'])-repeat['probabilities']))),
        tolerances=dict(same_batch_logits_atol=2e-5,cross_batch_logits_atol=1e-4,cross_batch_probabilities_atol=2e-5,rtol=2e-5))
    del legacy,state
    cfg=dict(foreground_ids=['hf-fg-1'],backgrounds_per_category=1,seed=2026,
        severities=dict(clean=0.,mild=1.5,moderate=4.,strong=9.),opacity=1.,
        include_background_only=True,include_neutral=True,include_original=True,
        allow_approximate=False,batch_size=16,background_fit='cover')
    start=time.monotonic()
    run=run_experiment(cfg,[adapter],data_root=args.data_root,output_root=DEFAULT_RUNS)
    loaded=load_run(run['path'])
    natural=[r for r in loaded['rows'] if r['control']=='natural']
    assert len(natural)==8 and len(loaded['pairs'])==4
    assert all(r['status']=='completed' for r in loaded['rows'])
    assert all(s['accuracy'] is None for s in loaded['summary'] if s['control'] in ('background_only','foreground_only_on_neutral'))
    for r in natural:
        with Image.open(Path(run['path'])/r['model_input_path']) as im:
            assert im.size==(224,224) and pixel_hash(im)==r['model_input_sha256']
    report.update(demo_run=run['run_id'],demo_rows=run['rows'],natural_rows=len(natural),paired_rows=len(loaded['pairs']),
                  elapsed_seconds=time.monotonic()-start,persisted_reload_verified=True)
    write_json(args.output/'real_verification.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
