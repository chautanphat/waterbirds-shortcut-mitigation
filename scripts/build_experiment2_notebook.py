"""Build a self-contained one-additional-seed notebook; does not execute it."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def historical_bytes(path,digest):
    raw=path.read_bytes();lf=raw.replace(b'\r\n',b'\n')
    for candidate in [raw,lf,lf.replace(b'\n',b'\r\n')]:
        if hashlib.sha256(candidate).hexdigest()==digest:return candidate
    raise ValueError('Historical source differs beyond newline encoding: '+str(path))


def build():
    old=json.loads((ROOT/'notebooks/poter_waterbirds_kaggle.ipynb').read_text(encoding='utf-8'))
    tree=ast.parse(''.join(old['cells'][1]['source']))
    payload=next(ast.literal_eval(x.value) for x in tree.body if isinstance(x,ast.Assign) and
        any(isinstance(a,ast.Name) and a.id=='payload' for a in x.targets))
    old_zip=zipfile.ZipFile(io.BytesIO(base64.b64decode(payload)))
    reference=json.loads((ROOT/'checkpoints/poter_kaggle_recovered/CUB/Clean/POTER_CUB_seed1_kaggle/config.json').read_text())
    contents={};source_deviations={}
    for name in old_zip.namelist():
        if name.endswith('/'):continue
        path=ROOT/'baselines/poter'/name
        if name in reference['source_sha256']:
            try:contents[name]=historical_bytes(path,reference['source_sha256'][name])
            except ValueError:
                if name!='data/celebA_dataset.py':raise
                # Only this unrelated dataset implementation is unrecovered.
                # CUB does not instantiate CelebADataset; do not invent old bytes.
                contents[name]=path.read_bytes()
                source_deviations[name]=dict(historical_sha256=reference['source_sha256'][name],
                    included_sha256=hashlib.sha256(contents[name]).hexdigest(),
                    reason='Unrelated CelebA dataset module; not the Waterbirds/CUB implementation. Historical bytes not recovered.')
        else:contents[name]=path.read_bytes() if path.is_file() else old_zip.read(name)
    wrapper=(ROOT/'scripts/profile_poter_training.py').read_bytes()
    contents['profile_poter_training.py']=wrapper
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(contents.items()):
            info=zipfile.ZipInfo(name,(2026,11,19,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
    encoded=base64.b64encode(buffer.getvalue()).decode('ascii')
    cells=[]
    def md(text):cells.append(dict(cell_type='markdown',metadata={},source=textwrap.dedent(text).strip()+'\n'))
    def code(text):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=textwrap.dedent(text).strip()+'\n'))
    md('''
    # Experiment 2 — one additional POTER seed, existing baselines reused

    **Cost-limited plan: POTER seeds1/2 total, ERM_enhanced seed0 and DPE_legacy seed0.**
    Seed1 is already complete. This notebook trains only seed2, from ImageNet V1,
    using the original 300-epoch, batch128, FP32, OT-weighted sampling recipe.
    No extra probe training, third POTER seed, ERM or DPE fitting is scheduled.

    **Writing/opening this notebook does not authorize training.** Training is skipped by
    default. Notify/agree with the user on cloud compute before changing the confirmation flag.
    Local 6 GB GPU is insufficient for the historical peak10.6 GiB. Use a cloud GPU with
    at least16 GiB. Historical comparable-T4 estimate: about9.1 GPU-hours plus setup/evaluation;
    no new training benchmark or measured speedup is claimed.

    Historical Waterbirds training source bytes are embedded here; newline-only encoding
    differences are restored to audited hashes. One unused CelebA module differs and is
    recorded explicitly; its historical bytes were not recovered. No CUB algorithm is edited.
    No Git clone or current branch download is required. Preserve the original framework versions
    below for primary replication. If unavailable, discuss the deviation before training.
    ''')
    code('''
    from pathlib import Path
    import os, sys, json, hashlib, zipfile, io, base64, subprocess, shutil
    from datetime import datetime, timezone

    CONFIRM_TRAINING = False  # Set True only after user confirmation of this cloud job.
    SEED = 2                 # Exactly one additional seed; never initialize from seed1 weights.
    PROFILE_PHASES = False   # Optional host/phase timings during the authorized full run.
    MAX_HOURS = 9.5          # Session budget; original runner pauses only at epoch boundaries.
    ROOT = Path('/kaggle/working/experiment2-poter-source')
    DATA = Path('/kaggle/working/poter-data')
    FEATURE = Path('/kaggle/working/poter-features/waterbirds_standard.pt')
    RESULTS = Path('/kaggle/working/poter-results')
    FEATURE_INPUT = None     # Optional attached exact waterbirds_standard.pt path.
    DATA_INPUT = None        # Optional extracted waterbird_complete95_forest2water2 directory.
    RESTORE_ARCHIVE = None   # Optional trusted full prior output ZIP, before resuming seed2.

    def digest(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(4*1024*1024), b''): h.update(chunk)
        return h.hexdigest()

    def preserve_write(path, payload):
        path = Path(path)
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(f'Existing file differs; preserved, not overwritten: {path}')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    ''')
    source_hashes={name:hashlib.sha256(data).hexdigest() for name,data in contents.items()}
    code(f"PAYLOAD = {encoded!r}\nREFERENCE = json.loads({json.dumps(reference)!r})\nSOURCE_HASHES = json.loads({json.dumps(source_hashes)!r})\nSOURCE_DEVIATIONS = json.loads({json.dumps(source_deviations)!r})\n"+textwrap.dedent('''
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
            for name in archive.namelist():
                target = (ROOT / name).resolve()
                assert target.is_relative_to(ROOT.resolve()), name
                preserve_write(target, archive.read(name))
        for name, expected in SOURCE_HASHES.items():
            assert digest(ROOT / name) == expected, f'Source fingerprint differs: {name}'
        for name, expected in REFERENCE['source_sha256'].items():
            assert name in SOURCE_DEVIATIONS or digest(ROOT/name) == expected
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch, torchvision, numpy as np, pandas as pd
        environment = dict(torch=str(torch.__version__), torchvision=str(torchvision.__version__), numpy=np.__version__)
        env_matches = all(environment[key] == REFERENCE[key] for key in environment)
        print('Current:', environment)
        print('Required:', {key:REFERENCE[key] for key in environment})
        print('Environment matches:', env_matches, '; training enabled:', CONFIRM_TRAINING)
        print('Historical source deviations:', SOURCE_DEVIATIONS)
        print('Waterbirds/CUB training sources match historical fingerprints; unused CelebA deviation recorded.')
        ''').strip())
    md('''
    ## Restore progress first

    Only restore your own trusted output ZIP. It must include the whole run directory,
    `last_training.pt`, `best_validation.pt`, config and metric CSVs. A selected checkpoint
    alone cannot resume optimizer/RNG state. Existing different files are preserved and cause
    an error. Restore into a fresh working directory rather than overwriting live progress.
    A seed1 snapshot cannot resume seed2. Runtime/source/config must match the saved run.
    ''')
    code('''
    import stat
    if RESTORE_ARCHIVE is not None:
        with zipfile.ZipFile(RESTORE_ARCHIVE) as archive:
            for item in archive.infolist():
                if item.is_dir(): continue
                target = (RESULTS / item.filename).resolve()
                assert target.is_relative_to(RESULTS.resolve()), item.filename
                assert not stat.S_ISLNK(item.external_attr >> 16), 'ZIP symlinks not accepted'
                preserve_write(target, archive.read(item))
        print('Trusted archive restored without overwriting different files.')
    run_dir = RESULTS / 'CUB' / 'Clean' / f'POTER_CUB_seed{SEED}_kaggle'
    state_path = run_dir / 'last_training.pt'
    if state_path.exists():
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        assert all(k in state for k in ['model','optimizer','rng','loss_computer','epoch','config'])
        assert state['config']['seed'] == SEED and state['config']['epochs'] == 300
        assert state['config']['batch_size'] == 128
        print('Resume from completed epoch', state['epoch'], '; same seed only.')
        print('Approx remaining hours:', max(0, 299-state['epoch'])*108.65/3600)
        del state
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError('Existing run without resume state: preserve/recover it before training.')
    else:
        print('No saved seed2 state. Fresh seed2 will use original ImageNet initialization.')
    ''')
    md('''
    ## Reuse exact dataset and fixed OT cache

    Attach the existing `waterbirds_standard.pt` cache (about61 MB) as a private Kaggle input.
    The currently audited local `D:/POTER_features/waterbirds_standard.pt` has a different hash
    and must **not** be assumed compatible. Recover the exact cache from the original cloud run.
    The notebook discovers that name under `/kaggle/input` and accepts it only if its SHA matches.
    It never
    recomputes OT features as an untracked fallback. Train/validation/test counts must be
    4795/1199/5794. The official dataset archive is downloaded only if no extracted dataset
    is available. Nothing here uses test accuracy to choose settings or checkpoints.
    ''')
    code('''
    import urllib.request, tarfile
    target = DATA / 'cub/data/waterbird_complete95_forest2water2'
    if DATA_INPUT is not None and not target.exists():
        source = Path(DATA_INPUT)
        assert digest(source/'metadata.csv') == REFERENCE['metadata_sha256']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)  # Stage unchanged files on local cloud disk.
    if not (target/'metadata.csv').exists():
        archive = DATA.parent/'waterbirds.tar.gz'
        if not archive.exists():
            temporary = archive.with_suffix('.download')
            urllib.request.urlretrieve('https://downloads.cs.stanford.edu/nlp/data/dro/waterbird_complete95_forest2water2.tar.gz', temporary)
            assert digest(temporary) == '56c51b77f7d17283ac97003d973e9107e8af9ceebffcb0c8be14d024f846f951'
            temporary.replace(archive)
        assert digest(archive) == '56c51b77f7d17283ac97003d973e9107e8af9ceebffcb0c8be14d024f846f951'
        destination = target.parent
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, 'r:gz') as tar:
            for item in tar:
                path = (destination/item.name).resolve()
                assert path.is_relative_to(destination.resolve()) and (item.isfile() or item.isdir())
                if item.isdir(): path.mkdir(parents=True, exist_ok=True)
                else: preserve_write(path, tar.extractfile(item).read())
    assert digest(target/'metadata.csv') == REFERENCE['metadata_sha256']
    metadata = pd.read_csv(target/'metadata.csv')
    assert metadata['split'].value_counts().to_dict() == {2:5794, 0:4795, 1:1199}
    assert set(metadata['y']) == {0,1} and set(metadata['place']) == {0,1}
    assert all((target/name).is_file() for name in metadata['img_filename'])
    if not FEATURE.exists():
        candidates = [Path(FEATURE_INPUT)] if FEATURE_INPUT is not None else list(Path('/kaggle/input').rglob('waterbirds_standard.pt'))
        matching = [p for p in candidates if p.is_file() and digest(p) == REFERENCE['cache_sha256']]
        if not matching:
            raise FileNotFoundError('Attach the exact original waterbirds_standard.pt cache, then rerun this cell.')
        FEATURE.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(matching[0], FEATURE)
    assert digest(FEATURE) == REFERENCE['cache_sha256']
    print('Dataset and fixed OT cache verified. Existing predictions require no rerun.')
    ''')
    md('''
    ## Optional timing and the single training job

    `PROFILE_PHASES=True` wraps unchanged `run_epoch` calls and times DataLoader host wait
    plus train/validation/test wall time. It does not change batches, precision, loss or
    optimizer; synchronization adds timing overhead. This is not a CUDA kernel profile.
    AMP and extra workers are **not enabled**: they have no validated parity/speedup here.
    Selection stays strictly best validation WGA. The original runner logs test metrics
    each epoch; those metrics are not selection inputs. No epoch reduction or early stopping.

    For a runtime version mismatch, recreate the original environment before confirming:
    torch2.11.0+cu130, torchvision0.26.0+cu130, NumPy2.1.3. No automatic package upgrade is done.
    ''')
    code('''
    if not CONFIRM_TRAINING:
        print('SKIPPED: training not confirmed. No optimizer/model training is launched.')
    else:
        assert SEED == 2, 'Only seed2 is prioritized; third seed is deferred.'
        assert env_matches, 'Primary replication requires the original environment versions.'
        assert torch.cuda.is_available(), 'Cloud CUDA GPU required.'
        assert torch.cuda.get_device_properties(0).total_memory >= 12*1024**3, 'Need at least12 GiB visible VRAM; use a16 GB class GPU or larger.'
        assert digest(FEATURE) == REFERENCE['cache_sha256']
        marker = RESULTS.parent/'experiment2-seed2-active-process.json'
        if marker.exists():
            previous = json.loads(marker.read_text())
            assert Path('/proc').exists(), 'Cannot verify prior process; inspect runtime before launching another job.'
            commandline = Path(f"/proc/{previous['pid']}/cmdline")
            if commandline.exists():
                running = commandline.read_bytes().replace(b'\\0',b' ').decode(errors='replace')
                assert str(run_dir) not in running, 'Prior seed2 process is still running; do not launch a duplicate.'
        arguments = ['--data-dir',str(DATA),'--cache',str(FEATURE),'--run-dir',str(run_dir),
                     '--seed','2','--epochs','300','--batch-size','128','--max-hours',str(MAX_HOURS)]
        command = [sys.executable,str(ROOT/'scripts/kaggle_waterbirds.py'),*arguments]
        if PROFILE_PHASES:
            command = [sys.executable,str(ROOT/'profile_poter_training.py'),'--confirm-training',
                       '--baseline-root',str(ROOT),'--profile-output',str(run_dir/'phase_profile.json'),'--',*arguments]
        print('Launching one seed; full original recipe:', command)
        process = subprocess.Popen(command, cwd=ROOT)
        marker.write_text(json.dumps(dict(pid=process.pid,run_dir=str(run_dir),command=command),indent=2))
        try:
            returncode = process.wait()
            if returncode: raise subprocess.CalledProcessError(returncode,command)
        except KeyboardInterrupt:
            print('Wait interrupted. Child PID', process.pid, 'was NOT terminated. Inspect it before rerunning/closing the session.')
            raise
        finally:
            if process.poll() is None:
                print('Job still running: no termination and no live-state ZIP snapshot attempted. Wait for an epoch checkpoint and job exit before backup.')
            elif RESULTS.exists():
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                backup = shutil.make_archive(str(RESULTS.parent/f'experiment2-seed2-backup-{stamp}'), 'zip', RESULTS)
                print('SAVE/DOWNLOAD this full archive before session ends:', backup)
                print('Archive SHA256:', digest(backup))
        print('Run returned. It may have paused at an epoch boundary; verify completion below.')
    ''')
    md('''
    ## Verify completion and export progress

    Rerunning resumes a matching partial seed2. A session-budget pause is **not** completion.
    Keep `last_training.pt`, best checkpoint, config and logs together. `/kaggle/working`
    alone is not a durable backup: save the notebook output version or download the ZIP.
    A hard runtime kill can lose uncommitted work; never stop a known live job before
    verifying an accessible durable full-state checkpoint. The notebook does not kill jobs.
    ''')
    code('''
    if state_path.exists():
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        assert state['config']['seed'] == 2 and state['config']['epochs'] == 300
        completed = state['epoch'] == 299
        record = dict(seed=2,last_completed_epoch=int(state['epoch']),training_complete=completed,
                      multi_seed_validation_complete=False,selection='strictly best validation WGA',
                      precision='FP32',epochs=300,batch_size=128,
                      last_training_sha256=digest(state_path),
                      best_validation_sha256=digest(run_dir/'best_validation.pt'),
                      next_action='Import selected checkpoint and run locked 2100-image evaluation' if completed else 'Restore full state and resume same seed')
        del state
        (run_dir/'export_status.json').write_text(json.dumps(record,indent=2))
        print(json.dumps(record,indent=2))
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup = shutil.make_archive(str(RESULTS.parent/f'experiment2-seed2-final-{stamp}'), 'zip', RESULTS)
        print('Download complete state bundle:', backup, '\\nSHA256:', digest(backup))
    else:
        print('No seed2 progress: training was skipped or did not commit an epoch. Seed1 alone remains available.')
    ''')
    md('''
    ## Evaluation handoff — still pending

    Return the full seed2 bundle to the repository. Evaluate its validation-selected checkpoint
    on the exact locked 2,100 inputs (1,600 natural +400 gray +100 original; sigma0/.5/1/2).
    Reuse all6,300 existing POTER seed1/ERM seed0/DPE seed0 predictions. No new baseline inference
    is needed. This notebook does **not** produce those counterfactual predictions or claim evaluation.

    Report both POTER seeds and two-seed mean±sample SD separately from foreground-bootstrap CIs.
    Two seeds give preliminary stability evidence only; the original three-seed target remains deferred.
    ERM/DPE seed variability is not established by one baseline checkpoint. Do not silently train
    a third seed based on test outcomes or tune the method on the repeatedly inspected test cohort.
    ''')
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python',version='3.11'),
        experiment2=dict(training_default=False,additional_poter_seeds=[2],baseline_training_deferred=True,
            source_capsule_sha256=hashlib.sha256(buffer.getvalue()).hexdigest(),reference=reference,
            source_deviations=source_deviations,source_hashes=source_hashes)),nbformat=4,nbformat_minor=5)
    for i,c in enumerate(cells):c['id']=f'exp2-{i:02d}'
    destination=ROOT/'notebooks/experiment2_poter_cost_optimized.ipynb'
    destination.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(destination)
    return destination


if __name__=='__main__':build()
