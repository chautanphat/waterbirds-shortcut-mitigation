"""Optional full notebook audit: pip install nbclient ipykernel to enable."""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

nbformat = pytest.importorskip('nbformat')
nbclient = pytest.importorskip('nbclient')
pytest.importorskip('ipykernel')

ROOT = Path(__file__).resolve().parents[3]

# Run the actual shared optimization loop, data ordering, checkpoint persistence,
# feature extraction and all H2 stages. Only image decoding/ResNet are replaced
# with a tiny synthetic model and inputs, so no downloads or GPU are needed.
SYNTHETIC_BACKBONE = '''
import torch
from torch import nn
import src.datasets.waterbirds as waterbirds
import src.methods.erm.train as erm_training
import src.methods.dpe_head_reweighting.stage0 as stage0_module

class TinyClassifier(nn.Sequential):
    def __init__(self, *args):
        super().__init__(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(3, 2))
    def extract_features(self, inputs):
        base = self[1](self[0](inputs))
        return base.repeat(1, 683)[:, :2048]

def synthetic_image(self, index):
    y, a = int(self.y[index]), int(self.a[index])
    image = torch.tensor([0.2 + y, 0.3 + a, 0.01 + (index % 13) / 20.]).reshape(3, 1, 1)
    return index, image, y, a

def load_tiny(path, *, device='cpu'):
    model = TinyClassifier()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    return model.to(device)

waterbirds.Waterbirds.__getitem__ = synthetic_image
erm_training.ResNetClassifier = TinyClassifier
stage0_module.load_resnet_checkpoint = load_tiny
'''


@pytest.mark.parametrize('track', ['standard', 'enhanced'])
def test_notebook_full_run_and_resume(tmp_path, monkeypatch, track):
    repo = tmp_path / 'repo'
    subprocess.run(['git', 'clone', '--quiet', '--no-hardlinks', str(ROOT), str(repo)], check=True)
    # Audit working-tree edits too, before they are committed.
    for folder in ('src', 'scripts', 'configs'):
        shutil.copytree(ROOT / folder, repo / folder, dirs_exist_ok=True)
    kernels = tmp_path / 'jupyter/kernels/h2-audit'
    kernels.mkdir(parents=True)
    (kernels / 'kernel.json').write_text(json.dumps(dict(
        argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        display_name='H2 audit', language='python',
        env=dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONPATH=''))))
    monkeypatch.setenv('JUPYTER_PATH', str(tmp_path / 'jupyter'))
    data = repo / 'data/waterbirds'
    data.mkdir(parents=True)
    # Existence checks remain real; decoding is replaced by synthetic_image.
    (data / 'image.jpg').write_bytes(b'SYNTHETIC FIXTURE ONLY')
    rows = []
    for split, counts in enumerate(([3498, 184, 56, 1057], [467, 466, 133, 133], [2255, 2255, 642, 642])):
        for group, count in zip(('00', '01', '10', '11'), counts):
            for i in range(count):
                filename = f'{split}_{group}_{i}.jpg'
                (data / filename).symlink_to('image.jpg')
                rows.append(dict(img_filename=filename, y=int(group[0]), place=int(group[1]), split=split))
    with (data / 'metadata.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    run = repo / 'outputs/dpe_head_reweighting_runs' / f'h2-{track}-seed0'
    backbone = repo / f'checkpoints/waterbirds_stage0_{track}_erm/seed0'
    bundle = repo / f'outputs/dpe_features/{track}/seed0'
    frozen = None
    for phase in ('full', 'resume'):
        notebook = nbformat.read(ROOT / 'notebooks/archive/dpe_head_reweighting.ipynb', as_version=4)
        notebook.cells.insert(2, nbformat.v4.new_code_cell(SYNTHETIC_BACKBONE))
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                cell.source = cell.source.replace('TRACK = "standard"', f'TRACK = "{track}"')
                cell.source = cell.source.replace('WORKERS = 2', 'WORKERS = 0')
        nbformat.validate(notebook)
        executed = nbclient.NotebookClient(
            notebook, timeout=240, kernel_name='h2-audit',
            resources={'metadata': {'path': str(repo / 'notebooks')}}).execute()
        assert not any(output.output_type == 'error' for cell in executed.cells
                       if cell.cell_type == 'code' for output in cell.outputs)
        assert run.with_suffix('.zip').is_file()
        assert (run / 'notebook_environment.json').is_file()
        assert (run / 'stage0_records/handoff.json').is_file()
        assert (run / 'experiment/predictions/test.npz').is_file()
        assert not (repo / 'outputs/dpe_head_reweighting').exists()
        summary = json.loads((backbone / 'summary.json').read_text())
        assert summary['updates'] == 3700  # 100 epochs, 37 complete batches each.
        assert summary['selected_checkpoint'] == 'ckpt_last.pt'
        provenance = json.loads((bundle / 'provenance.json').read_text())
        assert provenance['track'] == track
        assert 'val/test: Resize' in provenance['transforms']
        assert ('RandomResizedCrop' in provenance['transforms']) == (track == 'enhanced')
        selection = (run / 'experiment/selection.json').read_bytes()
        result = (run / 'results/metrics.json').read_bytes()
        identities = [p.read_bytes() for p in (backbone / 'ckpt_last.pt', bundle / 'handoff.json')]
        audit = subprocess.run([sys.executable, str(repo / 'scripts/audit_dpe_head_reweighting.py'),
                                '--run-dir', str(run)], capture_output=True, text=True, check=True)
        assert json.loads(audit.stdout)['status'] == 'passed'
        if phase == 'full':
            frozen = (selection, result, identities)
        else:
            assert frozen == (selection, result, identities)
    # A stale final report cannot be reused even if all model/input locks match.
    metrics_path = run / 'results/metrics.json'
    stale = json.loads(metrics_path.read_text())
    stale['selection_hash'] = 'changed'
    metrics_path.write_text(json.dumps(stale))
    with pytest.raises(nbclient.exceptions.CellExecutionError, match='Existing test result inputs changed'):
        nbclient.NotebookClient(notebook, timeout=240, kernel_name='h2-audit',
            resources={'metadata': {'path': str(repo / 'notebooks')}}).execute()
