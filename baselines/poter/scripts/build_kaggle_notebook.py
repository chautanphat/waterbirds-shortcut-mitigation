"""Package the local POTER source into an importable Kaggle notebook."""
import base64
import io
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
repo_root = Path(__file__).resolve().parents[3]
files = [*root.glob('*.py'), *root.joinpath('data').glob('*.py'),
         root / 'scripts/kaggle_waterbirds.py', root / 'scripts/training_checkpoint.py',
         root / 'scripts/summarize_waterbirds.py']
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
    for file in sorted(files):
        archive.write(file, file.relative_to(root).as_posix())
payload = base64.b64encode(buffer.getvalue()).decode('ascii')

def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(True)}

def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text.splitlines(True)}

cells = [
md('''# POTER Waterbirds — Kaggle Reproduction Notebook

Select **GPU T4 x2** and enable Internet in Kaggle settings. Run cells sequentially. Training uses batch size 128, then trains for 300 epochs per seed on a single GPU.

Checkpoints are committed after each epoch to `/kaggle/working/poter-results`. Before exiting, run the backup cell and download the resulting zip archive. In a new session, restore that backup into `/kaggle/working/poter-results` before continuing.
'''),
code('''import base64, io, zipfile, pathlib, os, sys, subprocess
ROOT = pathlib.Path('/kaggle/working/POTER')
ROOT.mkdir(parents=True, exist_ok=True)
payload = "''' + payload + '''"
with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as z:
    z.extractall(ROOT)
os.chdir(ROOT)
import torch, torchvision, numpy, pandas, scipy, sklearn, tqdm, PIL
print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'Enable GPU T4 x2 in Settings > Accelerator.'
print('GPU:', torch.cuda.get_device_name(0), 'VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
'''),
md('''## 1. Download Waterbirds and extract features
Source dataset from Stanford NLP. Requires Internet connection (~490 MB archive).
'''),
code('''import urllib.request, tarfile, hashlib
DATA = pathlib.Path('/kaggle/working/poter-data')
FEATURE = pathlib.Path('/kaggle/working/poter-features/waterbirds_standard.pt')
archive = pathlib.Path('/kaggle/working/waterbirds.tar.gz')
dest = DATA / 'cub/data'
meta = dest / 'waterbird_complete95_forest2water2/metadata.csv'
if not meta.exists():
    if not archive.exists():
        urllib.request.urlretrieve('https://downloads.cs.stanford.edu/nlp/data/dro/waterbird_complete95_forest2water2.tar.gz', archive)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == '56c51b77f7d17283ac97003d973e9107e8af9ceebffcb0c8be14d024f846f951', 'Archive checksum mismatch'
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as tar:
        for m in tar.getmembers():
            assert (dest / m.name).resolve().is_relative_to(dest.resolve()) and (m.isfile() or m.isdir())
        tar.extractall(dest)
assert meta.exists()
FEATURE.parent.mkdir(parents=True, exist_ok=True)
if not FEATURE.exists():
    subprocess.run([sys.executable, 'extract_features.py', '--dataset','CUB','--data_dir',str(DATA),'--ref_per_group','0:0,1:466,2:133,3:0','--ref_seed','0','--output',str(FEATURE),'--batch_size','32','--gpu','0'], check=True)
print('Dataset:', meta, 'Cache:', FEATURE)
'''),
md('''## 2. Probe run with batch size 128 (2 epochs)
Measures execution time and VRAM usage to verify batch 128 fits comfortably on a Tesla T4.
'''),
code('''RESULTS = pathlib.Path('/kaggle/working/poter-results')
probe = RESULTS / 'probe_batch128'
subprocess.run([sys.executable, 'scripts/kaggle_waterbirds.py', '--data-dir',str(DATA),'--cache',str(FEATURE),'--run-dir',str(probe),'--seed','1','--epochs','2','--batch-size','128','--max-hours','1'], check=True)
'''),
md('''## 3. Train 300 epochs per seed
Change `SEED` to 1, 2, or 3. Each run automatically resumes from the saved checkpoint if available. Near session timeout, it commits the current epoch checkpoint and exits gracefully.
'''),
code('''SEED = 1  # change to 2, 3 after previous seed completes
run_dir = RESULTS / 'CUB' / 'Clean' / f'POTER_CUB_seed{SEED}_kaggle'
subprocess.run([sys.executable, 'scripts/kaggle_waterbirds.py', '--data-dir',str(DATA),'--cache',str(FEATURE),'--run-dir',str(run_dir),'--seed',str(SEED),'--epochs','300','--batch-size','128','--max-hours','9.5'], check=True)
'''),
md('''## Save checkpoint archive before session ends
Download the zip archive from the Kaggle Output panel. For a new session, upload the zip to Input and extract into `/kaggle/working/poter-results` before continuing.
'''),
code('''import shutil
backup = shutil.make_archive('/kaggle/working/poter-results-backup', 'zip', RESULTS)
print('Download this archive from Output:', backup)
'''),
md('''## 4. Summarize benchmark results
Selects epoch by validation Worst-Group Accuracy (WGA), then evaluates held-out test split at that exact epoch.
'''),
code('''# Summarize single seed or all 3 seeds:
subprocess.run([sys.executable, 'scripts/summarize_waterbirds.py', str(RESULTS), '--seeds', str(SEED)], check=True)
'''),
]
notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.10'}}, 'nbformat': 4, 'nbformat_minor': 5}
output = repo_root / 'notebooks/poter_waterbirds_kaggle.ipynb'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print('Built notebook at:', output)
