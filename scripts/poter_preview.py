"""Local image-editing and inference preview for a recovered POTER checkpoint."""
import argparse
import base64
import csv
import hashlib
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['landbird', 'waterbird']


def create_app(checkpoint, metadata, data_dir, device='cpu'):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(2048, 2)
    model.load_state_dict(state['model'], strict=True)
    model.to(device).eval()
    crop = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop((224, 224))])
    tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    with Path(metadata).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    paths = {r.get('img_filename', r.get('filename')): r for r in rows}
    by_name = {}
    for path, row in paths.items():
        by_name.setdefault(Path(path).name, []).append((path, row))
    lock = threading.Lock()

    def encode(im):
        out = io.BytesIO()
        im.save(out, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(out.getvalue()).decode()

    def decode(value):
        if not isinstance(value, str) or not value.startswith('data:image/'):
            raise ValueError('Expected image data URL')
        raw = base64.b64decode(value.split(',', 1)[1], validate=True)
        with Image.open(io.BytesIO(raw)) as im:
            if im.width * im.height > 25_000_000:
                raise ValueError('Image is too large (maximum 25 megapixels)')
            return ImageOps.exif_transpose(im).convert('RGB')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, payload, kind='application/json'):
            body = json.dumps(payload).encode() if kind == 'application/json' else payload
            self.send_response(status)
            self.send_header('Content-Type', kind + '; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/':
                self.send(200, (ROOT / 'scripts/poter_preview.html').read_bytes(), 'text/html')
            elif self.path == '/api/info':
                self.send(200, dict(epoch=state.get('epoch'), device=device, model='POTER · ResNet-50',
                    checkpoint=Path(checkpoint).name, sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                    samples=[dict(path=p, species=p.split('/')[0]) for p, r in paths.items() if r['split'] == '2']))
            else:
                self.send(404, {'error': 'Not found'})

        def do_POST(self):
            try:
                origin = self.headers.get('Origin')
                if origin and origin != 'http://' + self.headers.get('Host', ''):
                    self.send(403, {'error': 'Open this tool on its localhost URL'}); return
                size = int(self.headers.get('Content-Length', '0'))
                if size < 1 or size > 35_000_000:
                    self.send(413, {'error': 'Request is too large'}); return
                request = json.loads(self.rfile.read(size))
                if self.path == '/api/sample':
                    path = request.get('path')
                    if path not in paths or paths[path]['split'] != '2':
                        raise ValueError('Choose a known test image')
                    file = (Path(data_dir) / path).resolve()
                    if not file.is_relative_to(Path(data_dir).resolve()):
                        raise ValueError('Invalid image path')
                    with Image.open(file) as im:
                        self.send(200, dict(image=encode(ImageOps.exif_transpose(im).convert('RGB')), name=path))
                    return
                if self.path != '/api/predict':
                    self.send(404, {'error': 'Not found'}); return
                original, edited = decode(request['original']), decode(request['edited'])
                if original.size != edited.size:
                    raise ValueError('Original and edited dimensions must match')
                name = request.get('name', '')
                matches = [(name, paths[name])] if name in paths else by_name.get(Path(name).name, [])
                truth = None
                if len(matches) == 1:
                    path, row = matches[0]
                    truth = dict(label=LABELS[int(row['y'])], background=['land', 'water'][int(row.get('place', row.get('a')))],
                        split={'0': 'train', '1': 'val', '2': 'test'}[row['split']], path=path,
                        source='Metadata matched by filename; verify identity for uploaded files')
                manual = request.get('truth')
                if truth is None and manual in LABELS:
                    truth = dict(label=manual, source='User supplied label')
                previews = [crop(original), crop(edited)]
                with lock, torch.inference_mode():
                    probs = model(torch.stack([tensor(im) for im in previews]).to(device)).softmax(1).cpu().tolist()
                predictions = [dict(label=LABELS[max(range(2), key=lambda i: p[i])], probabilities=dict(zip(LABELS, p))) for p in probs]
                self.send(200, dict(original=predictions[0], edited=predictions[1], ground_truth=truth,
                    model_view=encode(previews[1]), epoch=state.get('epoch')))
            except (ValueError, KeyError, OSError, UnidentifiedImageError) as error:
                self.send(400, {'error': str(error)})
            except Exception:
                self.send(500, {'error': 'Inference failed. Check the server console.'})
                import traceback
                traceback.print_exc()
    return Handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/poter_kaggle_recovered/CUB/Clean/POTER_CUB_seed1_kaggle/best_validation.pt')
    parser.add_argument('--metadata', type=Path, default=ROOT / 'data/waterbirds/metadata.csv')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'waterbird_complete95_forest2water2/waterbird_complete95_forest2water2')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    torch.set_num_threads(4)
    handler = create_app(args.checkpoint, args.metadata, args.data_dir, args.device)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
    print(f'POTER preview ready: http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()
