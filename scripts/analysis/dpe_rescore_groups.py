"""Per-group test accuracy of key arms of the strictly nested run, on official labels and on the
5,704 test images left after removing the 90 images of the three mislabelled species.
Rebuilds predictions exactly as dpe_rescore_corrected_test.py (same checks apply).
Usage: python scripts/analysis/dpe_rescore_groups.py <checkpoints dir> <out csv>"""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rescore', HERE / 'dpe_rescore_corrected_test.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ARMS = ('fullval_uniform', 'crossfit_uniform', 'crossfit_affine_sel_nll', 'fullval_affine_oof_sel_nll',
        'pool_uniform', 'pool_affine_sel_nll', 'pool_linear_sel_nll', 'crossfit_postmean_thrwga')

old = m.old
cfg = old.read_json(m.STRICT / 'protocol.json')['config']
specs = {s['id']: s for s in old.read_json(m.STRICT / 'sources.json')['sources']}
src = old.adapters()['sources'].Source(dict(metadata=str(m.CODE / 'data' / 'metadata.csv')))
src.val_y = np.array([r['y'] for r in src.rows])
src.val_g = np.array([r['group'] for r in src.rows])
ty = np.array([r['y'] for r in src.test_rows])
ta = np.array([r['a'] for r in src.test_rows])
aff = np.array([str(r['image_path']).startswith(m.SPECIES) for r in src.test_rows])
rows = []
for ck in sorted(Path(sys.argv[1]).glob('*/split*')):
    sid, split = ck.parent.name, int(ck.name[5:])
    pr = m.rebuild(ck, specs[sid], split, src, cfg)
    for arm in ARMS:
        for view, keep in (('official', np.ones(len(ty), bool)), ('removed', ~aff)):
            p, y, a = pr[arm][keep].astype(int), ty[keep], ta[keep]
            rows.append(dict(source=sid, recipe=specs[sid]['recipe'], split=split, arm=arm, view=view,
                             **{f'g{i}{k}': 100 * float((p[(y == i) & (a == k)] == i).mean()) for i in (0, 1) for k in (0, 1)}))
    print(sid, split, flush=True)
with open(sys.argv[2], 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
