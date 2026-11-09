"""Study automation and notebook integration; no real data/weight downloads."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from src.methods.dpe_head_reweighting import study
from src.methods.dpe_head_reweighting import experiment as ex

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('weighting_fixtures', Path(__file__).with_name('test_weighting_rules.py'))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


def source_seed(tmp_path, seed, with_test=True):
    source, _, cache = fixtures.make_bundle(tmp_path / f'fixture-{seed}')
    manifest = ex.read_json(source / 'experiment/manifest.json')
    manifest['provenance']['seed'] = seed
    ex.write_json(source / 'experiment/manifest.json', manifest)
    selection = ex.read_json(source / 'experiment/selection.json')
    selection['lock']['manifest'] = ex.digest(source / 'experiment/manifest.json')
    ex.write_json(source / 'experiment/selection.json', selection)
    if with_test:
        cache('test')
    return source, cache


def test_cached_multiseed_study_and_summary(tmp_path, monkeypatch):
    sources = [source_seed(tmp_path, seed)[0] for seed in (0,1)]
    monkeypatch.setattr(study, 'ensure_dataset', lambda *a, **k: pytest.fail('Cached mode must not train/download'))
    work = tmp_path / 'work'
    result = study.run_study(work, source=sources, mode='cached', device='cpu')
    assert result['summary']['status'] == 'complete'
    assert result['summary']['completed_seeds'] == [0,1]
    assert study.run_study(work, source=sources, mode='cached', device='cpu')['summary'] == result['summary']
    path = Path(result['output_dir'])
    assert len(list((path / 'source_archives').glob('*.zip'))) == 2
    with (path / 'per_seed.csv').open() as f:
        assert len(list(csv.DictReader(f))) == 12
    assert Path(result['archive']).is_file()
    # The fixed pipeline must reject test-set changes across seeds.
    changed = sources[1] / 'experiment/predictions/test.npz'
    with np.load(changed) as f:
        data = dict(f)
    data['ids'] = np.array([str(x)+'-different' for x in data['ids']])
    np.savez(changed, **data)
    with pytest.raises(ValueError, match='identical test IDs'):
        study.run_study(tmp_path / 'other-work', source=sources, mode='cached', device='cpu')


def test_new_rule_selection_precedes_test_inference(tmp_path, monkeypatch):
    pending = {seed: source_seed(tmp_path, seed, with_test=False) for seed in (1,2)}
    existing, _ = source_seed(tmp_path, 0)
    dataset_calls = []
    def dataset(*a, **kw):
        dataset_calls.append(True)
        return tmp_path / 'dataset'
    monkeypatch.setattr(study, 'ensure_dataset', dataset)
    monkeypatch.setattr(study, 'build_source', lambda work,data,track,seed,*a: (pending[seed][0], dict(seed=seed)))
    work = tmp_path / 'work'
    def test_cache(c):
        root, cache = pending[c['seed']]
        assert not (root / 'experiment/predictions/test.npz').exists()
        assert len(list((work / 'results').glob(f'h2-standard-seed{c["seed"]}-*/selection.json'))) == 1
        cache('test')
    monkeypatch.setattr(study, 'ensure_test_cache', test_cache)
    result = study.run_study(work, source=[existing], search_roots=[], device='cpu')
    assert len(dataset_calls) == 1
    assert result['summary']['completed_seeds'] == [0,1,2]


def test_automatic_dataset_download_and_reuse(tmp_path, monkeypatch):
    from src.methods.dpe_head_reweighting import stage0
    target = tmp_path / 'data/waterbirds'
    ready, calls = [], []
    def validate(path):
        if ready and Path(path) == target:
            return target / 'metadata.csv', []
        raise FileNotFoundError('fixture dataset not prepared')
    def prepare(command, check):
        assert command[1] == 'scripts/prepare_waterbirds.py' and check
        assert command[-2:] == ['--data_dir', str(tmp_path / 'data')]
        calls.append(command)
        ready.append(True)
    monkeypatch.setattr(stage0,'validate_dataset',validate)
    monkeypatch.setattr(study.subprocess,'run',prepare)
    assert study.ensure_dataset(tmp_path,[]) == target
    assert study.ensure_dataset(tmp_path,[]) == target
    assert len(calls) == 1


def test_summary_uses_paired_deltas_and_sample_sd(tmp_path):
    families = ('uniform','power','softmax','top_k','joint','overall_selected')
    completed = {seed:dict(results={family:dict(candidate='uniform', metrics=dict(overall=a,wga=w),
                 delta_pp=dict(overall=d,wga=d)) for family in families})
                 for seed,a,w,d in [(0,.90,.80,-1.),(1,.92,.84,1.)]}
    result = study.summarize(tmp_path, completed, [0,1,2], 'standard')
    assert result['status'] == 'incomplete'
    assert result['results']['uniform']['wga']['mean'] == pytest.approx(.82)
    assert result['results']['uniform']['wga']['sample_std'] == pytest.approx(np.sqrt(.0008))
    assert result['results']['uniform']['wga_delta_pp']['mean'] == 0
    assert result['results']['uniform']['wga_delta_pp']['sample_std'] == pytest.approx(np.sqrt(2))
    assert study.summarize(tmp_path,{0:completed[0]},[0,1,2],'standard')['results']['uniform']['wga']['sample_std'] is None


def test_interrupted_zip_preserves_previous_download(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'report.md').write_text('first completed seed')
    destination = tmp_path / 'study.zip'
    study.publish_zip(source, destination)
    before = destination.read_bytes()
    def interrupted(base_name, *args, **kwargs):
        Path(base_name + '.zip').write_bytes(b'partial ZIP')
        raise OSError('simulated interrupted archive')
    with monkeypatch.context() as patch:
        patch.setattr(study.shutil, 'make_archive', interrupted)
        with pytest.raises(OSError, match='interrupted archive'):
            study.publish_zip(source, destination)
    assert destination.read_bytes() == before
    (source / 'report.md').write_text('all seeds complete')
    study.publish_zip(source, destination)
    with zipfile.ZipFile(destination) as archive:
        assert archive.read('report.md') == b'all seeds complete'


def test_invalid_rules_fail_before_backbone_training(tmp_path, monkeypatch):
    import yaml
    settings = yaml.safe_load((ROOT / 'configs/dpe_head_reweighting/study.yaml').read_text())
    rules = yaml.safe_load((ROOT / settings['rules']).read_text())
    rules['top_k'] = [16]
    rules_path = tmp_path / 'rules.yaml'
    rules_path.write_text(yaml.safe_dump(rules))
    settings['rules'] = str(rules_path)
    config_path = tmp_path / 'study.yaml'
    config_path.write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(study, 'ensure_dataset', lambda *a, **kw: pytest.fail('Must fail before training'))
    with pytest.raises(ValueError, match='Invalid top_k'):
        study.run_study(tmp_path / 'work', config_path=config_path, device='cpu')


def test_full_study_notebook_without_any_h2_zip(tmp_path, monkeypatch):
    nbformat = pytest.importorskip('nbformat')
    nbclient = pytest.importorskip('nbclient')
    pytest.importorskip('ipykernel')
    repo = tmp_path / 'repo'
    subprocess.run(['git','clone','--quiet','--no-hardlinks',str(ROOT),str(repo)],check=True)
    for folder in ('src','scripts','configs'):
        shutil.copytree(ROOT/folder,repo/folder,dirs_exist_ok=True)
    data = repo / 'data/waterbirds'
    data.mkdir(parents=True)
    (data / 'image.jpg').write_bytes(b'Synthetic fixture only')
    rows = []
    for split,counts in enumerate(([3498,184,56,1057],[467,466,133,133],[2255,2255,642,642])):
        for group,count in zip(('00','01','10','11'),counts):
            for i in range(count):
                filename=f'{split}_{group}_{i}.jpg'
                (data/filename).symlink_to('image.jpg')
                rows.append(dict(img_filename=filename,y=int(group[0]),place=int(group[1]),split=split))
    with (data/'metadata.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    previous_spec=importlib.util.spec_from_file_location('original_notebook_fixture',Path(__file__).with_name('test_notebook.py'))
    previous=importlib.util.module_from_spec(previous_spec);previous_spec.loader.exec_module(previous)
    notebook=nbformat.read(ROOT/'notebooks/dpe_weighting_rules.ipynb',as_version=4)
    work=tmp_path/'work'
    for cell in notebook.cells:
        if cell.cell_type=='code' and 'MODE = "auto"' in cell.source:
            for old,new in [('REPO_PATH = None',f'REPO_PATH = {str(repo)!r}'),
                            ('WORK_DIR = None',f'WORK_DIR = {str(work)!r}'),
                            ('SEEDS = None','SEEDS = [1, 2]'),('WORKERS = None','WORKERS = 0'),
                            ('DEVICE = "auto"','DEVICE = "cpu"'),('SEARCH_PATHS = None','SEARCH_PATHS = []')]:
                cell.source=cell.source.replace(old,new)
    notebook.cells.insert(3,nbformat.v4.new_code_cell(previous.SYNTHETIC_BACKBONE+'\ntorch.set_num_threads(1)\n'))
    notebook.cells[3].pop('id',None)
    path=repo/'notebooks/dpe_weighting_rules.ipynb'
    nbformat.write(notebook,path)
    kernels=tmp_path/'jupyter/kernels/full-study';kernels.mkdir(parents=True)
    (kernels/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        display_name='Full study fixture',language='python',env=dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=''))))
    monkeypatch.setenv('JUPYTER_PATH',str(tmp_path/'jupyter'))
    frozen=None
    for phase in ('full','reuse'):
        current=nbformat.read(path,as_version=4)
        nbclient.NotebookClient(current,timeout=240,kernel_name='full-study',
            on_cell_executed=lambda **kw:nbformat.write(current,path),
            resources={'metadata':{'path':str(repo/'notebooks')}}).execute()
        studies=[p for p in (work/'studies').iterdir() if p.is_dir()]
        assert len(studies)==1
        summary=ex.read_json(studies[0]/'summary.json')
        assert summary['completed_seeds']==[1,2] and summary['status']=='complete'
        assert len(list((studies[0]/'source_archives').glob('*.zip')))==2
        assert studies[0].with_suffix('.zip').exists()
        checkpoints=[work/'backbones/standard'/f'seed{s}/ckpt_last.pt' for s in (1,2)]
        identities=[p.read_bytes() for p in checkpoints]
        for seed in (1,2):
            assert ex.read_json(work/'backbones/standard'/f'seed{seed}/summary.json')['updates']==3700
        if frozen is not None:
            assert frozen==(summary,identities)
        frozen=(summary,identities)
