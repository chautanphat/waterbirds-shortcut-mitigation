"""Export the verified POTER seed 1 failures on the filtered Waterbirds test set."""
import argparse
import base64
from collections import Counter
import io
import json
import math
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.counterfactual.common import (  # noqa: E402
    KNOWN_WATERBIRDS_LABEL_ERROR_SPECIES, contained, pixel_hash, read_csv, sha256, write_csv,
)

LABELS = ('Landbird', 'Waterbird')
BACKGROUND_NAMES = ('Đất', 'Nước')
GALLERY_TEMPLATE = ROOT / 'src/counterfactual/web/poter_clean_test_gallery.html'


def matched_rows(source_root):
    results = source_root / 'results/poter_failure'
    data = source_root / 'data/waterbirds'
    hashes = json.loads((results / 'integrity.json').read_text(encoding='utf-8'))['sha256']
    for name in ('original_test_predictions.csv', 'checkpoint_audit.json'):
        if sha256(results / name) != hashes[name]:
            raise ValueError(f'Saved benchmark file changed: {name}')
    model = json.loads((results / 'checkpoint_audit.json').read_text(encoding='utf-8'))['POTER']
    metadata = [row for row in read_csv(data / 'metadata.csv') if row['split'] == '2']
    test = {row['img_id']: row for row in metadata}
    predictions = [row for row in read_csv(results / 'original_test_predictions.csv') if row['model_name'] == 'POTER']
    if len(test) != len(metadata) or len(predictions) != len(test) or {r['source_image_id'] for r in predictions} != set(test):
        raise ValueError('POTER predictions must match canonical test IDs one-to-one')

    total, wrong, excluded = Counter(), Counter(), Counter()
    errors = []
    for row in predictions:
        source_id = row['source_image_id']
        official = test[source_id]
        filename = official['img_filename']
        species_id = filename.split('/')[0]
        y, place, pred = int(official['y']), int(official['place']), int(row['pred_y'])
        group = 2 * y + place
        if (row['true_y'], row['original_group'], row['species_id'], row['bg_type'],
                row['source_split'], row['control'], row['model_seed']) != (
                str(y), str(group), species_id, ('land', 'water')[place], 'test', 'original', '1'):
            raise ValueError(f'Prediction provenance differs for image {source_id}')
        if species_id in KNOWN_WATERBIRDS_LABEL_ERROR_SPECIES:
            excluded[species_id] += 1
            continue
        total[group] += 1
        if pred == y:
            continue
        probabilities = [float(row['p_landbird']), float(row['p_waterbird'])]
        if (pred not in (0, 1) or any(not 0 <= p <= 1 for p in probabilities)
                or not math.isclose(sum(probabilities), 1., abs_tol=1e-6)
                or pred != int(probabilities[1] > probabilities[0])):
            raise ValueError(f'Saved scores disagree with prediction for image {source_id}')
        wrong[group] += 1
        errors.append(dict(image_id=int(source_id), species=species_id.split('.', 1)[1].replace('_', ' '),
                           group=group, background=BACKGROUND_NAMES[place], label=LABELS[y], prediction=LABELS[pred],
                           confidence=probabilities[pred], p_landbird=probabilities[0], p_waterbird=probabilities[1],
                           canonical_filename=filename, source_path=row['source_path'],
                           source_sha256=row['source_sha256'], raw_input_sha256=row['raw_input_sha256']))

    if excluded != {species: 30 for species in KNOWN_WATERBIRDS_LABEL_ERROR_SPECIES}:
        raise ValueError(f'Unexpected number of excluded label errors: {excluded}')
    groups = [dict(group=g, label=f'{LABELS[g // 2]} · nền {BACKGROUND_NAMES[g % 2].lower()}',
                   total=total[g], errors=wrong[g], accuracy=(total[g] - wrong[g]) / total[g]) for g in range(4)]
    n = sum(total.values())
    summary = dict(total=n, errors=len(errors), excluded=sum(excluded.values()),
                   accuracy=(n-len(errors))/n, wga=min(r['accuracy'] for r in groups), groups=groups,
                   epoch=model['epoch'], checkpoint_sha256=model['checkpoint_sha256'],
                   predictions_sha256=hashes['original_test_predictions.csv'])
    saved = next(r for r in read_csv(results / 'original_test_label_sensitivity.csv')
                 if r['model_name'] == 'POTER' and r['label_policy'] == 'exclude_known_errors')
    if (n != int(saved['n']) or not math.isclose(summary['accuracy'], float(saved['semantic_accuracy']))
            or not math.isclose(summary['wga'], float(saved['wga']))):
        raise ValueError('Gallery counts disagree with the saved evaluation')
    return data, summary, sorted(errors, key=lambda r: (-r['confidence'], r['image_id']))


def export(source_root, output):
    data, summary, rows = matched_rows(source_root)
    gallery = []
    for row in rows:
        image_path = contained(data, row['source_path'])
        if sha256(image_path) != row['source_sha256']:
            raise ValueError(f'Original JPEG changed: {row["image_id"]}')
        image_bytes = image_path.read_bytes()
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format != 'JPEG' or pixel_hash(image.convert('RGB')) != row['raw_input_sha256']:
                raise ValueError(f'Original JPEG differs from the inference image: {row["image_id"]}')
        public_row = {k: v for k, v in row.items() if k not in {'source_path', 'source_sha256', 'raw_input_sha256'}}
        gallery.append(dict(**public_row, jpeg=base64.b64encode(image_bytes).decode('ascii')))

    payload = json.dumps(dict(summary=summary, images=gallery), ensure_ascii=False, allow_nan=False)
    page = GALLERY_TEMPLATE.read_text(encoding='utf-8').replace('__GALLERY_DATA__', payload.replace('<', r'\u003c'))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'poter-clean-test-gallery.html').write_text(page, encoding='utf-8')
    write_csv(output / 'poter-clean-test-errors.csv', [{k: v for k, v in r.items() if k != 'jpeg'} for r in gallery])
    print(f'Exported {len(gallery)} verified POTER failures from {summary["total"]} test images to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT, help='Checkout with original saved predictions and Waterbirds data')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/poter_clean_test_gallery')
    args = parser.parse_args()
    export(args.source_root, args.output)
