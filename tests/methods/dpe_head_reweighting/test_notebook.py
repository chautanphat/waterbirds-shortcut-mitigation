"""Optional full notebook audit: pip install nbclient ipykernel to enable."""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

nbformat = pytest.importorskip('nbformat')
nbclient = pytest.importorskip('nbclient')
pytest.importorskip('ipykernel')

from src.methods.dpe_head_reweighting.experiment import digest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize('track', ['standard', 'enhanced'])
def test_notebook_full_run_and_resume(tmp_path, monkeypatch, track):
    repo = tmp_path / 'repo'
    subprocess.run(['git', 'clone', '--quiet', '--no-hardlinks', str(ROOT), str(repo)], check=True)
    # Audit working-tree edits too, before they are committed.
    shutil.copytree(ROOT / 'src/methods/dpe_head_reweighting',
                    repo / 'src/methods/dpe_head_reweighting', dirs_exist_ok=True)
    kernels = tmp_path / 'jupyter/kernels/h2-audit'
    kernels.mkdir(parents=True)
    (kernels / 'kernel.json').write_text(json.dumps(dict(
        argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        display_name='H2 audit', language='python',
        env=dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=''))))
    monkeypatch.setenv('JUPYTER_PATH', str(tmp_path / 'jupyter'))
    data = repo / 'data/dpe_head_reweighting' / track / 'seed0'
    data.mkdir(parents=True)
    rows = []
    # Official group sizes exercise the smallest balanced A batch as well.
    for split, counts in enumerate(([3498, 184, 56, 1057], [467, 466, 133, 133], [2255, 2255, 642, 642])):
        for group, count in zip(('00', '01', '10', '11'), counts):
            for i in range(count):
                rows.append(dict(img_filename=f'{split}/{group}/{i}.jpg',
                                 y=int(group[0]), place=int(group[1]), split=split))
    with (data / 'metadata.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (data / 'ckpt_last.pt').write_bytes(b'SYNTHETIC FIXTURE ONLY')
    rng = np.random.RandomState(9)
    for split, name in ((1, 'val'), (2, 'test')):
        ids = [r['img_filename'] for r in rows if r['split'] == split]
        np.savez(data / f'{name}_raw.npz', ids=np.array(ids),
                 features=rng.normal(size=(len(ids), 2048)).astype('float32'))
    provenance = dict(track=track, seed=0, checkpoint_sha256=digest(data / 'ckpt_last.pt'),
                      val_features_sha256=digest(data / 'val_raw.npz'), model='resnet50',
                      pretrained_weights='synthetic', transforms='synthetic',
                      extraction_order='metadata_split_order', feature_format='raw',
                      checkpoint_selection='final_epoch_no_validation_selection', source_commit='synthetic')
    (data / 'provenance.json').write_text(json.dumps(provenance))
    run = repo / 'outputs/dpe_head_reweighting_runs' / f'h2-{track}-seed0'
    frozen = None
    for phase in ('full', 'resume'):
        notebook = nbformat.read(ROOT / 'notebooks/dpe_head_reweighting.ipynb', as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                cell.source = cell.source.replace('TRACK = "standard"', f'TRACK = "{track}"')
        nbformat.validate(notebook)
        executed = nbclient.NotebookClient(
            notebook, timeout=180, kernel_name='h2-audit',
            resources={'metadata': {'path': str(repo / 'notebooks')}}).execute()
        assert not any(output.output_type == 'error' for cell in executed.cells
                       if cell.cell_type == 'code' for output in cell.outputs)
        assert run.with_suffix('.zip').is_file()
        assert (run / 'notebook_environment.json').is_file()
        assert (run / 'experiment/predictions/test.npz').is_file()
        assert not (repo / 'outputs/dpe_head_reweighting').exists()
        selection = (run / 'experiment/selection.json').read_bytes()
        result = (run / 'results/metrics.json').read_bytes()
        if phase == 'full':
            frozen = (selection, result)
        else:
            assert frozen == (selection, result)
