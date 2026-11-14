"""Build dpe_label_corrected_backbones_kaggle.ipynb from runner.py, summarize.py and the cells below.

    python experiments/dpe_label_corrected_backbones/build_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = (HERE/"runner.py").read_text(encoding="utf-8")
SUMMARIZE = (HERE/"summarize.py").read_text(encoding="utf-8")

MD_INTRO = r"""# DPE on Waterbirds with corrected labels: fresh backbones, both label sets

Select **GPU T4 x2**, turn **Internet on**, then **Save Version -> Save & Run All**. Optional but
recommended: **Add Input -> Notebook Output -> `noerula/dpe-reproduction`** so the two official
ERM backbones of seeds 5 and 6 are reused (their SHA-256 is checked) instead of retrained.

Three CUB species (021 Eastern Towhee, 088 Western Meadowlark, 102 Western Wood-Pewee) are labelled
waterbird in Waterbirds only because the generator matches `tern` inside `eastern`/`western`.
This run crosses **backbone training labels** (official / corrected, 72 training images differ)
with **DPE validation labels** (official / corrected, 18 images) on fresh SubpopBench ERM backbones,
and scores every model on **official and corrected test labels** (90 images).

Pipeline: the audited runner of `noerula/dpe-reproduction` (SubpopBench ERM, default
hyper-parameters, 5,001 steps; the authors' unmodified DPE `main.py`, 15 heads, `wga_val`), with
a label-set switch. Upstream code is pinned and unmodified.

**Declared before the run** (also in `summarize.py`): primary contrast = (official backbone,
corrected validation labels) minus (official backbone, official validation labels) on corrected
test labels, averaged over head seeds within a backbone and then over backbone seeds 5-8.
Secondary: corrected backbone + corrected validation labels, and corrected backbone only; every
contrast is also reported on official test labels. No test result chooses any setting.

The notebook first runs a ~20 minute smoke test and stops if it fails. Work is scheduled on both
GPUs in priority order and stops launching new jobs after `TIME_LIMIT_HOURS`, so the version
saves before Kaggle's 12-hour limit. `report/REPORT.md` is refreshed after every finished job."""

CELL_CONFIG = r"""from pathlib import Path
import os, sys, json, subprocess, time, shutil, glob, hashlib
T0 = time.time()
WORK = Path('/kaggle/working/dpe-labels')
WORK.mkdir(parents=True, exist_ok=True)
RUNNER = WORK / 'runner.py'
SUMMARIZE = WORK / 'summarize.py'
RESULTS = WORK / 'results'
REPORT_DIR = WORK / 'report'
SMOKE_DIR = WORK / 'smoke'
RESULTS.mkdir(parents=True, exist_ok=True)

REUSE_SEEDS = [5, 6]          # official backbones reused from noerula/dpe-reproduction when attached
NEW_SEEDS = [7, 8]            # fresh backbones for both label sets (declared before the run)
HEAD_OFFSETS = [0, 1000, 2000]  # DPE head seeds = backbone seed + offset
TRAIN_ATTR = 'yes'            # group-annotated validation (DPE Table 2 setting)
WORKERS_PER_JOB = 2
TIME_LIMIT_HOURS = 11.0       # no new job starts if it would not finish by then
RUN_SMOKE_TEST = True
EXPECTED_REUSE_SHA = {
    5: 'ead5ae8118eba69a31a96cf54c6e34bd78fa78a382a97f1824fb11f5822bdaef',
    6: 'e088b07b34911d3a569e252ed807490e4380cc2474cb8d36a6184537c40fb2f1',
}
REGIMES = [('official', 'official'), ('official', 'corrected'), ('corrected', 'official'), ('corrected', 'corrected')]

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def hours():
    return (time.time() - T0) / 3600

print(subprocess.check_output(['nvidia-smi'], text=True))"""

MD_ENV = "## Pinned training environment (same as `noerula/dpe-reproduction`)"

CELL_ENV = r"""BOOT = WORK / 'bootstrap'
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--upgrade', '--target', str(BOOT), 'uv==0.9.8'], check=True)
UV_ENV = dict(os.environ, PYTHONPATH=str(BOOT), UV_CACHE_DIR=str(WORK / 'uv-cache'),
              UV_PYTHON_INSTALL_DIR=str(WORK / 'python'))
VENV = WORK / 'venv'
PY = VENV / 'bin/python'
if not PY.exists():
    subprocess.run([sys.executable, '-m', 'uv', 'venv', '--python', '3.11', '--seed', str(VENV)], env=UV_ENV, check=True)
ISOLATED_ENV = dict(os.environ, PYTHONNOUSERSITE='1')
for key in ('PYTHONHOME', 'PYTHONPATH'):
    ISOLATED_ENV.pop(key, None)
subprocess.run([str(PY), '-m', 'pip', 'install', '--quiet', 'torch==2.6.0+cu118',
                'torchvision==0.21.0+cu118', '--index-url', 'https://download.pytorch.org/whl/cu118'], env=ISOLATED_ENV, check=True)
PINS = ['Pillow==10.3.0', 'jsonargparse==4.37.0', 'matplotlib==3.8.3', 'netcal==1.3.5',
        'numpy==1.26.4', 'pandas==2.2.3', 'scikit-learn==1.3.2', 'scipy==1.12.0',
        'seaborn==0.13.2', 'tqdm==4.66.2', 'transformers==4.46.3', 'wandb==0.19.1',
        'timm==1.0.12', 'tensorboard==2.18.0', 'tensorboard-logger==0.1.0', 'protobuf==3.20.3']
subprocess.run([str(PY), '-m', 'pip', 'install', '--quiet', *PINS], env=ISOLATED_ENV, check=True)
subprocess.run([str(PY), '-m', 'pip', 'check'], env=ISOLATED_ENV, check=True)
BASE_ENV = dict(ISOLATED_ENV, TORCH_HOME=str(WORK / 'cache/torch'), HF_HOME=str(WORK / 'cache/hf'),
                MPLCONFIGDIR=str(WORK / 'cache/matplotlib'), MPLBACKEND='Agg',
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
subprocess.run([str(PY), '-c', "import torch, torchvision; print(torch.__version__, torchvision.__version__); "
                "assert torch.cuda.device_count() == 2, 'Choose Kaggle T4 x2'; "
                "print([torch.cuda.get_device_name(i) for i in range(2)])"], env=BASE_ENV, check=True)"""

MD_DATA = """## Pinned upstream code, official data, and a corrected-label copy

The corrected copy shares the same image files (symlink) and changes only `y` for the 180 images
of the three species (72 train, 18 validation, 90 test). Each worker re-verifies group counts."""

CELL_DATA = r"""import urllib.request, tarfile
def checkout(url, destination, commit):
    if not destination.exists():
        subprocess.run(['git', 'clone', url, str(destination)], check=True)
    current = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    if current != commit:
        subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', commit], check=True)
    assert subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip() == commit
    assert not subprocess.check_output(['git', '-C', str(destination), 'status', '--porcelain', '--untracked-files=no'], text=True)

DPE = WORK / 'dpe-upstream'
SUBPOP = WORK / 'subpopbench-upstream'
checkout('https://github.com/minhto2802/dpe4subpop.git', DPE, '68b811461e238d44cced6ae6c8ce75b66cad5449')
checkout('https://github.com/YyzHarry/SubpopBench.git', SUBPOP, '4d3dbbe21029666ef19d040e110ec22908640c5b')
DATA_OFF = WORK / 'data_official'
WB = DATA_OFF / 'waterbirds'
WB.mkdir(parents=True, exist_ok=True)
IMAGE_ROOT = WB / 'waterbird_complete95_forest2water2'
if not (IMAGE_ROOT / 'metadata.csv').exists():
    archive = WORK / 'waterbirds.tar.gz'
    partial = WORK / 'waterbirds.tar.gz.part'
    if not archive.exists():
        errors = []
        for base in ('https://downloads.cs.stanford.edu/nlp/data/dro/', 'https://nlp.stanford.edu/data/dro/'):
            try:
                urllib.request.urlretrieve(base + 'waterbird_complete95_forest2water2.tar.gz', partial)
                partial.replace(archive)
                break
            except Exception as e:
                errors.append(str(e))
        else:
            raise RuntimeError('Waterbirds download failed: ' + str(errors))
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            candidate = (WB / member.name).resolve()
            if not candidate.is_relative_to(WB.resolve()) or member.issym() or member.islnk() or member.isdev():
                raise RuntimeError('Unexpected archive entry: ' + member.name)
        tar.extractall(WB)
format_code = "\nimport pandas as pd, sys\nfrom pathlib import Path\nwb = Path(sys.argv[1])\ndf = pd.read_csv(wb / 'waterbird_complete95_forest2water2/metadata.csv')\ndf['filename'] = df['img_filename']\ndf['a'] = df['place']\ndf.to_csv(wb / 'metadata_waterbirds.csv', index=False)\nprint(df.groupby(['split','y','a']).size())\n"
subprocess.run([str(PY), '-c', format_code, str(WB)], env=BASE_ENV, check=True)

DATA_COR = WORK / 'data_corrected'
WBC = DATA_COR / 'waterbirds'
WBC.mkdir(parents=True, exist_ok=True)
link = WBC / 'waterbird_complete95_forest2water2'
if not link.exists():
    link.symlink_to(IMAGE_ROOT.resolve(), target_is_directory=True)
correct_code = (
    "import pandas as pd, sys\n"
    "src, dst = sys.argv[1], sys.argv[2]\n"
    "df = pd.read_csv(src)\n"
    "species = ('021.Eastern_Towhee/', '088.Western_Meadowlark/', '102.Western_Wood_Pewee/')\n"
    "mask = df['filename'].str.startswith(species)\n"
    "assert int(mask.sum()) == 180 and (df.loc[mask, 'y'] == 1).all(), int(mask.sum())\n"
    "df.loc[mask, 'y'] = 0\n"
    "df.to_csv(dst, index=False)\n"
    "print('relabelled', int(mask.sum()), 'images'); print(df.groupby(['split','y','a']).size())\n")
subprocess.run([str(PY), '-c', correct_code, str(WB / 'metadata_waterbirds.csv'), str(WBC / 'metadata_waterbirds.csv')],
               env=BASE_ENV, check=True)
DATA = {'official': DATA_OFF, 'corrected': DATA_COR}
subprocess.run([str(PY), '-m', 'subpopbench.train', '--help'], cwd=SUBPOP, env=BASE_ENV, check=True, stdout=subprocess.DEVNULL)
subprocess.run([str(PY), 'main.py', '--help'], cwd=DPE, env=BASE_ENV, check=True, stdout=subprocess.DEVNULL)
subprocess.run([str(PY), '-c', "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V1')"], env=BASE_ENV, check=True)"""

MD_RUNNER = """## Runner (audited runner of `noerula/dpe-reproduction` plus a label-set switch) and scorer"""

CELL_RUNNER = '%%writefile "{RUNNER}"\n' + RUNNER
CELL_SUMMARIZE = '%%writefile "{SUMMARIZE}"\n' + SUMMARIZE

MD_REUSE = "## Reuse the official seed-5 and seed-6 backbones when their outputs are attached"

CELL_REUSE = r"""BACKBONES = {}   # (label_set, seed) -> dict(ckpt=Path, provenance=str)
REUSED = set()
for s in REUSE_SEEDS:
    found = None
    for p in sorted(glob.glob(f'/kaggle/input/**/seed{s}/ckpt_last.pt', recursive=True)):
        if sha256(p) == EXPECTED_REUSE_SHA[s]:
            found = Path(p); break
    if found:
        BACKBONES[('official', s)] = dict(ckpt=found, provenance=f'official SubpopBench ERM backbone seed {s} reused from noerula/dpe-reproduction ({found}); sha256 verified')
        REUSED.add(s)
    print(f'seed {s}:', 'reusing ' + str(found) if found else 'not attached; it will be retrained')"""

MD_SCHED = """## Job scheduler and smoke test

`train` jobs train a backbone on one label set and fit DPE with the same labels (head seed = backbone
seed). `heads` jobs fit DPE on an existing backbone with either label set and any head seed."""

CELL_SCHED = r"""import signal
sys.path.insert(0, str(WORK))
import importlib; summarize = importlib.import_module('summarize')

def train_job(labels, seed, extra=()):
    return dict(kind='train', name=f'train-{labels}-s{seed}', backbone_labels=labels, val_labels=labels,
                backbone_seed=seed, head_seed=seed, extra=list(extra))

def heads_job(bb, seed, val, head, extra=()):
    return dict(kind='heads', name=f'heads-bb{bb}-s{seed}-val{val}-h{head}', backbone_labels=bb, val_labels=val,
                backbone_seed=seed, head_seed=head, extra=list(extra))

def launch(job, gpu, results):
    out = results / job['name']
    cmd = [str(PY), str(RUNNER), '--dpe-root', str(DPE), '--subpop-root', str(SUBPOP),
           '--data-dir', str(DATA[job['val_labels']]), '--output-dir', str(out), '--seed', str(job['head_seed']),
           '--train-attr', TRAIN_ATTR, '--head-implementation', 'official_cli', '--workers', str(WORKERS_PER_JOB),
           '--label-set', job['val_labels']]
    if job['kind'] == 'train':
        cmd += ['--protocol', 'subpopbench']
    else:
        bb = BACKBONES[(job['backbone_labels'], job['backbone_seed'])]
        cmd += ['--protocol', 'checkpoint', '--checkpoint', str(bb['ckpt']), '--checkpoint-provenance', bb['provenance']]
    cmd += job['extra']
    env = dict(BASE_ENV, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONHASHSEED=str(job['head_seed']))
    log = open(results / f"{job['name']}.log", 'w')
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return dict(job=job, proc=proc, log=log, gpu=gpu, start=time.time(), out=out)

def finish(run, results):
    run['log'].close()
    job, out = run['job'], run['out']
    seconds = time.time() - run['start']
    record = {k: v for k, v in job.items() if k != 'extra'}
    record.update(returncode=run['proc'].returncode, seconds=seconds, gpu=run['gpu'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'job.json').write_text(json.dumps(record, indent=1))
    ok = run['proc'].returncode == 0 and (out / 'test_predictions.npz').exists()
    if ok and job['kind'] == 'train':
        BACKBONES[(job['backbone_labels'], job['backbone_seed'])] = dict(
            ckpt=out / 'ckpt_last.pt', provenance=f"trained in this notebook ({job['name']})")
    if not ok:
        print(f"FAILED {job['name']} (rc={run['proc'].returncode}); last log lines:")
        print((results / f"{job['name']}.log").read_text(errors='replace')[-3000:])
    return ok, seconds

def stop(run):
    if run['proc'].poll() is None:
        os.killpg(run['proc'].pid, signal.SIGTERM)
        try:
            run['proc'].wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(run['proc'].pid, signal.SIGKILL); run['proc'].wait()

def wait_all(runs, results):
    results_ok = {}
    while any(r['proc'].poll() is None for r in runs):
        time.sleep(30)
    for r in runs:
        results_ok[r['job']['name']] = finish(r, results)[0]
    return results_ok

if RUN_SMOKE_TEST:
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)
    SMOKE_DIR.mkdir(parents=True)
    smoke_train = train_job('corrected', 99, extra=['--steps', '30', '--num-stages', '3'])
    smoke_train['name'] = 'smoke-train'
    ok = wait_all([launch(smoke_train, 0, SMOKE_DIR)], SMOKE_DIR)
    assert ok['smoke-train'], 'Smoke training job failed: see the log above. Nothing else was started.'
    saved = dict(BACKBONES)
    smoke_heads = heads_job('corrected', 99, 'official', 1099, extra=['--num-stages', '3'])
    smoke_heads['name'] = 'smoke-heads'
    ok = wait_all([launch(smoke_heads, 1, SMOKE_DIR)], SMOKE_DIR)
    assert ok['smoke-heads'], 'Smoke heads job failed: see the log above.'
    BACKBONES.clear(); BACKBONES.update({k: v for k, v in saved.items() if k[1] != 99})
    print(summarize.summarize(SMOKE_DIR, SMOKE_DIR / 'report'))
    print(f'Smoke test passed after {hours():.2f} h')"""

MD_RUN = """## Main run (both GPUs, priority order, time-limited)"""

CELL_RUN = r"""PLAN = []
for s in REUSE_SEEDS:
    if s not in REUSED:
        PLAN.append(train_job('official', s))
    PLAN.append(train_job('corrected', s))
for s in REUSE_SEEDS:
    PLAN += [heads_job('official', s, 'corrected', s), heads_job('corrected', s, 'official', s)]
    if s in REUSED:
        PLAN.append(heads_job('official', s, 'official', s))
for s in NEW_SEEDS:
    PLAN += [train_job('official', s), train_job('corrected', s),
             heads_job('official', s, 'corrected', s), heads_job('corrected', s, 'official', s)]
for off in HEAD_OFFSETS[1:]:
    for s in REUSE_SEEDS + NEW_SEEDS:
        PLAN += [heads_job(bb, s, val, s + off) for bb, val in REGIMES]
print(len(PLAN), 'jobs planned:', [j['name'] for j in PLAN])

ESTIMATE = {'train': 2.5 * 3600, 'heads': 0.5 * 3600}
DURATIONS = {'train': [], 'heads': []}
pending, running, failed, skipped = list(PLAN), {}, set(), set()

def expected(kind):
    return max(DURATIONS[kind]) if DURATIONS[kind] else ESTIMATE[kind]

def pick():
    for job in list(pending):
        key = (job['backbone_labels'], job['backbone_seed'])
        if job['kind'] == 'heads' and key not in BACKBONES:
            source = f"train-{key[0]}-s{key[1]}"
            if source in failed or source in skipped:
                pending.remove(job); skipped.add(job['name'])
            continue
        if hours() * 3600 + 1.15 * expected(job['kind']) > TIME_LIMIT_HOURS * 3600:
            continue
        return job
    return None

last_print = 0
try:
    while True:
        for gpu in (0, 1):
            if gpu in running:
                run = running[gpu]
                if run['proc'].poll() is not None:
                    ok, seconds = finish(run, RESULTS)
                    DURATIONS[run['job']['kind']].append(seconds)
                    (failed if not ok else set()).add(run['job']['name'])
                    del running[gpu]
                    summarize.summarize(RESULTS, REPORT_DIR)
                    print(f"[{hours():.2f} h] finished {run['job']['name']} ok={ok} in {seconds/3600:.2f} h", flush=True)
            if gpu not in running:
                job = pick()
                if job is not None:
                    pending.remove(job)
                    running[gpu] = launch(job, gpu, RESULTS)
                    print(f"[{hours():.2f} h] started {job['name']} on GPU {gpu}", flush=True)
        if not running and (not pending or pick() is None):
            break
        if hours() > TIME_LIMIT_HOURS + 0.5:
            print('Hard time limit reached; stopping running jobs.')
            for run in running.values():
                stop(run); finish(run, RESULTS); failed.add(run['job']['name'])
            running.clear()
            break
        if time.time() - last_print > 600:
            last_print = time.time()
            print(f"[{hours():.2f} h] running: {[r['job']['name'] for r in running.values()]}; "
                  f"pending {len(pending)}, failed {len(failed)}, skipped {len(skipped)}", flush=True)
        time.sleep(30)
finally:
    for run in running.values():
        stop(run)
print('Not started (time limit or missing backbone):', [j['name'] for j in pending] + sorted(skipped))
print('Failed:', sorted(failed))"""

MD_REPORT = """## Report and compact evidence bundle

`report/REPORT.md` and `report/summary.csv` hold the scored results. The bundle keeps manifests,
metrics, logs, per-image test predictions and one feature cache per backbone (for CPU follow-ups
such as DFR-style retraining); backbone weights stay in the working directory."""

CELL_REPORT = r"""import zipfile
print(summarize.summarize(RESULTS, REPORT_DIR))
(REPORT_DIR / 'plan.json').write_text(json.dumps(dict(
    plan=[j['name'] for j in PLAN], failed=sorted(failed), not_started=[j['name'] for j in pending] + sorted(skipped),
    reused=sorted(REUSED), hours=hours()), indent=1))
bundle = WORK / 'label-corrected-evidence.zip'
with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in list(REPORT_DIR.rglob('*')) + list(RESULTS.rglob('*')) + list(SMOKE_DIR.rglob('*.log')):
        if not p.is_file() or p.suffix in ('.pt', '.pkl') or 'tfevents' in p.name:
            continue
        if p.name.startswith('feats_'):
            job = p.parent.name
            if not (job.startswith('train-') or (job.startswith('heads-bbofficial') and '-valofficial-' in job
                                                  and job.endswith(f"-h{job.split('-s')[1].split('-')[0]}"))):
                continue
        z.write(p, p.relative_to(WORK))
print('Bundle:', bundle, round(bundle.stat().st_size / 1e6, 1), 'MB')
# Free the large environment so the saved output stays small.
for name in ('venv', 'uv-cache', 'python', 'bootstrap', 'cache', 'data_official', 'data_corrected', 'waterbirds.tar.gz'):
    p = WORK / name
    if p.is_symlink() or p.is_file():
        p.unlink()
    elif p.exists():
        shutil.rmtree(p, ignore_errors=True)
print(f'Done after {hours():.2f} h')"""


def md(text):
    return dict(cell_type="markdown", metadata={}, source=text.splitlines(keepends=True))


def code(text):
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=text.splitlines(keepends=True))


cells = [md(MD_INTRO), code(CELL_CONFIG), md(MD_ENV), code(CELL_ENV), md(MD_DATA), code(CELL_DATA),
         md(MD_RUNNER), code(CELL_RUNNER), code(CELL_SUMMARIZE), md(MD_REUSE), code(CELL_REUSE),
         md(MD_SCHED), code(CELL_SCHED), md(MD_RUN), code(CELL_RUN), md(MD_REPORT), code(CELL_REPORT)]
nb = dict(cells=cells, nbformat=4, nbformat_minor=5,
          metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                        language_info=dict(name="python")))
target = HERE/"dpe_label_corrected_backbones_kaggle.ipynb"
target.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", target, len(cells), "cells")
