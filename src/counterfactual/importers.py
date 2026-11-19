"""Explicit external foreground and existing prediction import contracts."""
import base64
import io
import json
from pathlib import Path
import uuid

import numpy as np
from PIL import Image, ImageOps

from .common import contained, read_csv, sha256, write_csv, write_json
from .data import LICENSE


def decode_image(value, mode):
    if not isinstance(value,str) or not value.startswith('data:image/'):
        raise ValueError('Expected image data URL')
    blob = base64.b64decode(value.split(',',1)[1], validate=True)
    with Image.open(io.BytesIO(blob)) as im:
        if im.width*im.height > 16_000_000:
            raise ValueError('Maximum imported image size is 16 megapixels')
        return ImageOps.exif_transpose(im).convert(mode)


def import_foreground(root, image_url, mask_url, *, name='external', parent=None):
    root = Path(root)
    rgb, alpha = decode_image(image_url,'RGB'), decode_image(mask_url,'L')
    alpha_min,alpha_max=alpha.getextrema()
    if rgb.size != alpha.size or alpha_max==0 or alpha_min>0:
        raise ValueError('Mask must match image dimensions and isolate a nonempty foreground')
    if parent:
        existing = {r['foreground_id']:r for r in read_csv(root/'foregrounds_manifest.csv')}
        if (root/'external_foregrounds_manifest.csv').exists():
            existing.update({r['foreground_id']:r for r in read_csv(root/'external_foregrounds_manifest.csv')})
        original=existing[parent]
        with Image.open(contained(root,original['source_image_path'])) as im:
            expected=ImageOps.exif_transpose(im).convert('RGB')
        if expected.size!=rgb.size or expected.tobytes()!=rgb.tobytes():
            raise ValueError('Mask correction must retain the exact canonical source pixels')
    identifier = 'external-' + uuid.uuid4().hex[:16]
    target = root / 'external_foregrounds' / identifier
    target.mkdir(parents=True,exist_ok=False)
    rgb.save(target/'source.png')
    alpha.save(target/'mask.png')
    rgba=rgb.convert('RGBA')
    rgba.putalpha(alpha)
    rgba.save(target/'foreground.png')
    row=dict(foreground_id=identifier,bird_source_id=identifier,species=name[:150],y='',original_group='',
        original_split='unknown',source_image_path=(target/'source.png').relative_to(root).as_posix(),
        canonical_filename='',foreground_source_path=(target/'source.png').relative_to(root).as_posix(),
        rgba_path=(target/'foreground.png').relative_to(root).as_posix(),
        mask_path=(target/'mask.png').relative_to(root).as_posix(),source_sha256=sha256(target/'source.png'),
        rgba_sha256=sha256(target/'foreground.png'),mask_sha256=sha256(target/'mask.png'),
        mask_version=identifier,quality_status='external_manual_unknown_ground_truth',source_revision='',
        parent_foreground_id=parent or '',unknown_ground_truth=True)
    if parent:
        existing = {r['foreground_id']:r for r in read_csv(root/'foregrounds_manifest.csv')}
        if (root/'external_foregrounds_manifest.csv').exists():
            existing.update({r['foreground_id']:r for r in read_csv(root/'external_foregrounds_manifest.csv')})
        p=existing[parent]
        # Mask corrections retain canonical identity/split/label; a corrected mask
        # never becomes an official original CUB mask through user editing.
        for key in ['bird_source_id','species','y','original_group','original_split','canonical_filename','source_image_path','source_sha256']:
            row[key]=p[key]
        row['quality_status']='approximate_manual_mask'
        row['unknown_ground_truth']=str(row['y']) not in ('0','1')
    write_json(target/'version.json',row)
    manifest=root/'external_foregrounds_manifest.csv'
    rows=read_csv(manifest) if manifest.exists() else []
    write_csv(manifest,rows+[row])
    return row


def import_prediction_csv(csv_path, mapping_path, metadata, destination):
    """Validate a supplied column mapping; never derive confidence from metrics.

    This importer stores external results separately from immutable lab runs.
    It cannot invent paired rows or model-input assets missing from the source.
    """
    mapping=json.loads(Path(mapping_path).read_text(encoding='utf-8'))
    required={'image_path','pred_y','p_landbird','p_waterbird','checkpoint_sha256','preprocess_fingerprint'}
    if not required.issubset(mapping):
        raise ValueError(f'Explicit mapping missing {sorted(required-set(mapping))}')
    by_path={r['img_filename']:r for r in read_csv(metadata)}
    result=[]
    seen=set()
    for raw in read_csv(csv_path):
        row={k:raw[v] for k,v in mapping.items()}
        path=row['image_path'].replace('\\','/')
        if path not in by_path or path in seen:
            raise ValueError('Unknown or duplicate canonical prediction image')
        seen.add(path)
        truth=by_path[path]
        y=int(truth['y'])
        pred=int(row['pred_y'])
        probs=np.array([float(row['p_landbird']),float(row['p_waterbird'])])
        if pred not in (0,1) or not np.isfinite(probs).all() or (probs<0).any() or (probs>1).any() or not np.isclose(probs.sum(),1,atol=1e-5) or pred!=int(probs.argmax()):
            raise ValueError('Invalid probabilities or predicted class')
        for key in ('checkpoint_sha256','preprocess_fingerprint'):
            if len(row[key])!=64 or any(c not in '0123456789abcdef' for c in row[key]):
                raise ValueError('Provide actual checkpoint and preprocessing hashes as mapped columns')
        if 'true_y' in row and int(row['true_y'])!=y:
            raise ValueError('Imported label disagrees with canonical metadata')
        row.update(source_bird_id=truth['img_id'],true_y=y,source_split={'0':'train','1':'val','2':'test'}[truth['split']],
            original_group=2*y+int(truth['place']),is_correct=y==pred,
            provenance_status='imported_predictions_explicit_mapping',paired_metrics_status='unavailable',
            model_input_status='not_supplied; exact preview cannot be reconstructed from probabilities')
        result.append(row)
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    write_csv(destination/'predictions.csv',result)
    write_json(destination/'provenance.json',dict(source_csv_sha256=sha256(csv_path),
        mapping_sha256=sha256(mapping_path),metadata_sha256=sha256(metadata),
        note='Imported real predictions; no inference, checkpoint selection or synthetic pairing'))
    return dict(rows=len(result),destination=str(destination))
