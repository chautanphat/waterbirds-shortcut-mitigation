"""Epoch-boundary continuation must match uninterrupted CPU optimization."""
import json
from pathlib import Path

import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn

from src.methods.erm.config import ERMConfig
from src.methods.erm import train as training


class Tiny(nn.Sequential):
    def __init__(self, *args):
        super().__init__(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 2))


def setup(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(training, 'ResNetClassifier', Tiny)
    data = tmp_path / 'data'
    data.mkdir()
    rows = []
    for split in (0, 1):
        for g in range(4):
            name = f'{split}_{g}.jpg'
            Image.new('RGB', (32, 32), (30+50*g, 40+20*split, 50)).save(data / name)
            rows.append(dict(filename=name, y=g//2, a=g%2, split=split))
    pd.DataFrame(rows).to_csv(data / 'metadata.csv', index=False)
    c = ERMConfig()
    c.dataset.data_dir = str(data)
    c.dataset.batch_size = c.dataset.batch_size_eval = 4
    c.dataset.workers = 0
    c.training.device = 'cpu'
    c.training.epochs = 4
    c.output_dir = str(tmp_path / 'full')
    return c


@pytest.mark.parametrize('completed_epochs',[0,2])
def test_resume_matches_full_run_and_repairs_log(tmp_path, monkeypatch, completed_epochs):
    c = setup(tmp_path, monkeypatch)
    full_summary = training.run_training(c)
    full = Path(c.output_dir)
    c.output_dir = str(tmp_path / 'interrupted')
    original = training.train_epoch
    calls = 0
    def interrupt(*args):
        nonlocal calls
        calls += 1
        if calls == completed_epochs + 1:
            raise KeyboardInterrupt('simulated interruption')
        return original(*args)
    monkeypatch.setattr(training, 'train_epoch', interrupt)
    with pytest.raises(KeyboardInterrupt):
        training.run_training(c)
    partial = Path(c.output_dir)
    assert not (partial / 'summary.json').exists()
    with (partial / 'metrics.jsonl').open('a') as f:
        f.write('partial log write')
    monkeypatch.setattr(training, 'train_epoch', original)
    assert training.run_training(c, resume=True) == full_summary
    for name in ('ckpt_last.pt', 'ckpt_best_wga.pt'):
        a = torch.load(full / name, weights_only=True)
        b = torch.load(partial / name, weights_only=True)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert (full / 'metrics.jsonl').read_text() == (partial / 'metrics.jsonl').read_text()
    state = torch.load(partial / 'training_state.pt', weights_only=True)
    assert state['position']['epoch'] == 4 and len(state['history']) == 4
    assert json.loads((partial / 'run.json').read_text())['resumes'][0]['start_epoch'] == completed_epochs


def test_resume_rejects_recipe_or_metadata_change(tmp_path, monkeypatch):
    c = setup(tmp_path, monkeypatch)
    training.run_training(c)
    c.training.lr = .01
    with pytest.raises(ValueError, match='recipe or training implementation'):
        training.run_training(c, resume=True)
    c.training.lr = .001
    metadata = Path(c.dataset.data_dir) / 'metadata.csv'
    metadata.write_text(metadata.read_text()+'\n')
    with pytest.raises(ValueError, match='Resume data'):
        training.run_training(c, resume=True)
