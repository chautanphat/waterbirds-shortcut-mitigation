"""One image-inference adapter shared with the existing POTER Runner."""
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
import torch
import torchvision
from PIL import ImageOps, __version__ as pillow_version
from torchvision import transforms

from src.datasets.waterbirds import waterbirds_transform
from src.evaluation.prediction_export import load_image_model
from .common import LABELS, pixel_hash, sha256

PREPROCESS = dict(resize=[256, 256], interpolation='PIL bilinear', center_crop=[224, 224],
    color='EXIF transpose then RGB', tensor='float32 RGB / 255 CHW',
    mean=[.485, .456, .406], std=[.229, .224, .225])


class ModelAdapter:
    def __init__(self, checkpoint, device='cpu', *, method='poter', model_id=None,
                 dpe_heads=None, metadata=None):
        self.checkpoint = Path(checkpoint)
        self.checkpoint_sha256 = sha256(checkpoint)
        self.device, self.method = device, method
        self.model_id = model_id or method.upper()
        if method not in ('poter', 'erm', 'dpe'):
            raise ValueError('Only explicit POTER, ERM or compatible original-DPE adapters are supported')
        self.model, self.config = load_image_model('erm' if method == 'dpe' else method, checkpoint)
        self.model.to(device).eval()
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.epoch = state.get('epoch') if isinstance(state, dict) else None
        if metadata and self.config.get('metadata_sha256') and self.config['metadata_sha256'] != sha256(metadata):
            raise ValueError('POTER checkpoint canonical metadata hash disagrees with dataset')
        self.heads = None
        self.heads_sha256 = None
        if method == 'dpe':
            if not dpe_heads:
                raise ValueError('DPE needs original saved heads plus their exact ERM backbone, not a POTER model')
            self.heads = torch.load(dpe_heads, map_location='cpu', weights_only=True)
            if self.heads.get('backbone_sha256') != self.checkpoint_sha256:
                raise ValueError('DPE heads do not match this backbone SHA-256')
            if metadata and self.heads.get('metadata_sha256') != sha256(metadata):
                raise ValueError('DPE heads canonical metadata hash disagrees with dataset')
            from src.evaluation.dpe_export import predict_heads
            # Validate saved format and dimensions before accepting the adapter.
            predict_heads(self.heads, np.arange(2048, dtype=np.float32)[None], device=device)
            self.heads_sha256 = sha256(dpe_heads)
        t = waterbirds_transform().transforms
        self.crop = transforms.Compose(t[:2])
        self.tensor = transforms.Compose(t[2:])
        self.preprocess_fingerprint = hashlib.sha256(json.dumps(dict(settings=PREPROCESS,
            torchvision=str(torchvision.__version__), pillow=pillow_version), sort_keys=True).encode()).hexdigest()
        self.class_index_to_name = dict(enumerate(LABELS))
        self.lock = threading.Lock()

    def info(self):
        return dict(model_id=self.model_id, method=self.method, checkpoint=str(self.checkpoint),
            checkpoint_sha256=self.checkpoint_sha256, epoch=self.epoch, device=self.device,
            class_index_to_name=self.class_index_to_name, preprocess=PREPROCESS,
            preprocess_fingerprint=self.preprocess_fingerprint, heads_sha256=self.heads_sha256,
            numeric_runtime=dict(torch=str(torch.__version__),cudnn=torch.backends.cudnn.version(),
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cudnn_benchmark=torch.backends.cudnn.benchmark),
            score_type='log_probability_of_mean_heads' if self.method == 'dpe' else 'logit',
            selection='Fixed supplied checkpoint; no fitting or test-label model selection')

    def preview(self, image):
        return self.crop(ImageOps.exif_transpose(image).convert('RGB'))

    def predict_batch(self, images, batch_size=16):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Batch size must be positive')
        if not images:
            return []
        result = []
        with self.lock, torch.inference_mode():
            for start in range(0, len(images), batch_size):
                previews = [self.preview(im) for im in images[start:start+batch_size]]
                inputs = torch.stack([self.tensor(im) for im in previews])
                x = inputs.to(self.device)
                if self.method == 'dpe':
                    from src.evaluation.dpe_export import predict_heads
                    features = self.model.extract_features(x).cpu().numpy()
                    p = predict_heads(self.heads, features, device=self.device)
                    scores = np.log(np.maximum(p, np.finfo(np.float32).tiny))
                else:
                    logits = self.model(x)
                    if logits.shape != (len(x), 2) or not torch.isfinite(logits).all():
                        raise ValueError('Expected finite two-class logits')
                    scores = logits.cpu().numpy()
                    p = logits.softmax(1).cpu().numpy()
                for im, tensor, score, prob in zip(previews, inputs, scores, p):
                    result.append(dict(logits=score.astype(float).tolist(), probabilities=prob.astype(float).tolist(),
                        pred_y=int(np.argmax(prob)), preview=im, model_input_sha256=pixel_hash(im),
                        normalized_tensor_sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest()))
        return result
