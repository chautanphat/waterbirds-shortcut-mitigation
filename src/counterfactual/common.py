"""Small storage primitives; JSON never contains NaN or invented results."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / 'data/waterbirds'
DEFAULT_RUNS = ROOT / 'experiments/counterfactual_lab'
DEFAULT_CHECKPOINT = ROOT / 'checkpoints/poter_kaggle_recovered/CUB/Clean/POTER_CUB_seed1_kaggle/best_validation.pt'
LABELS = ['landbird', 'waterbird']
KNOWN_WATERBIRDS_LABEL_ERROR_SPECIES = frozenset({
    '021.Eastern_Towhee', '088.Western_Meadowlark', '102.Western_Wood_Pewee',
})
SPLITS = {0: 'train', 1: 'val', 2: 'test'}
SEVERITIES = {'clean': 0., 'mild': 1.5, 'moderate': 4., 'strong': 9.}
OPERATOR_STRENGTHS = {
    'contrast': {'clean': 1., 'mild': .75, 'moderate': .4, 'strong': .1},
    'occlusion': {'clean': 0., 'mild': .1, 'moderate': .3, 'strong': .5},
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pixel_hash(image):
    return hashlib.sha256(f'{image.mode}:{image.size}:'.encode() + image.tobytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def contained(root, value):
    root = Path(root).resolve()
    result = (root / value).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Path escapes its asset directory')
    return result
