"""Build the self-contained Kaggle notebook for the G1 test-set evaluation.

Bundle = frozen code of experiments/dpe_nested_corrected/code + g1_eval.py + g1_selftest.py +
g1_expected.json (recorded full-validation DPE WGA and the strict run's validation-selected offset,
per source and split, from results/dpe/strict_nested_20261115 and corrected_test_rescore_20261115).
"""
import ast
import base64
import csv
import hashlib
import io
import json
import textwrap
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN = ROOT / 'experiments/dpe_nested_corrected/code'
STRICT = ROOT / 'results/dpe/strict_nested_20261115'
RESCORE = ROOT / 'results/dpe/corrected_test_rescore_20261115/per_arm.csv'
OUT = HERE / 'artifacts'
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- expected reference values
expected = {'_config': json.loads((STRICT / 'protocol.json').read_text())['config']}
for path in sorted((STRICT / 'results').glob('*/split*.json')):
    cell = json.loads(path.read_text())
    expected.setdefault(cell['source'], {})[str(cell['split'])] = dict(
        fullval_uniform=dict(official=cell['arms']['fullval_uniform']['wga']), fullval_offset_nll={})
for row in csv.DictReader(RESCORE.open()):
    if row['arm'] == 'fullval_postmean_bias_sel_nll':
        expected[row['source']][row['split']]['fullval_offset_nll'] = dict(
            official=float(row['wga_official']), corrected=float(row['wga_corrected']), clean=float(row['wga_removed']))
    if row['arm'] == 'fullval_uniform':
        e = expected[row['source']][row['split']]['fullval_uniform']
        assert abs(e['official'] - float(row['wga_official'])) < 1e-9
        e.update(corrected=float(row['wga_corrected']), clean=float(row['wga_removed']))
cells = sum(len(v) for k, v in expected.items() if k != '_config')
assert cells == 70, cells
(HERE / 'g1_expected.json').write_text(json.dumps(expected, indent=1))

# ---------------------------------------------------------------- code bundle
archive = io.BytesIO()
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for path in sorted(FROZEN.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pyo'):
            z.write(path, str(path.relative_to(FROZEN)).replace('\\', '/'))
    for name in ('g1_eval.py', 'g1_selftest.py', 'g1_expected.json'):
        z.write(HERE / name, name)
payload = archive.getvalue()
bundle_hash = hashlib.sha256(payload).hexdigest()
(OUT / 'g1_test_eval_code.zip').write_bytes(payload)

nb_cells = []


def md(text):
    nb_cells.append(dict(cell_type='markdown', id=f'cell-{len(nb_cells):02d}', metadata={},
                         source=textwrap.dedent(text).strip().splitlines(True)))


def code(text):
    text = textwrap.dedent(text).strip() + '\n'
    ast.parse(text)
    nb_cells.append(dict(cell_type='code', id=f'cell-{len(nb_cells):02d}', metadata={}, execution_count=None,
                         outputs=[], source=text.splitlines(True)))


md("""
    # G1 geometry weighting on full-validation DPE: Waterbirds test set

    **Self-contained: import this notebook into Kaggle, turn Internet On, choose a CPU runtime,
    then Save & Run All.** About 20-40 minutes. No backbone is retrained.

    For each of the seven saved backbones (five standard ERM, two ERM*) and ten split seeds, the
    notebook retrains full-validation DPE with exactly the seeds of the strictly nested run, checks
    that uniform averaging reproduces the recorded test WGA, and scores on the test set:

    - **uniform** (DPE), **g1** (locked: gamma 50, alpha 0.25), g1 with alpha 0.1 / 0.5 (sensitivity only),
    - **perm**: G1 weights permuted across heads (does geometry matter?),
    - **conf**: confidence routing instead of distance.

    Primary, declared before running: g1 - uniform on **official test labels**, standard ERM.
    PASS if mean >= +0.30 pp, 95% interval above 0, >= 4/5 backbones positive, and g1 - perm
    interval above 0. Corrected labels and the clean subset are logged as secondary columns.
    The test set and backbones were inspected in earlier work: this is exploratory evidence.
    """)

code("""
    WORKERS = 4
    import os, sys, json, subprocess, zipfile, hashlib, platform
    from pathlib import Path
    WORK, INPUT = Path('/kaggle/working'), Path('/kaggle/input')
    if not WORK.is_dir():
        raise SystemExit('Run this notebook in Kaggle.')
    CODE, STATE = WORK / 'g1_code', WORK / 'g1_state'
    OUT, RECOVERED, SOURCE_INDEX = STATE / 'g1_v1', STATE / 'recovered_sources', STATE / 'source_index'
    STATE.mkdir(parents=True, exist_ok=True)
    WORKERS = max(1, min(WORKERS, os.cpu_count() or 1))

    def sh(args, check=True):
        args = [str(a) for a in args]
        print('$', ' '.join(args), flush=True)
        p = subprocess.Popen(args, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end='', flush=True)
        rc = p.wait()
        if check and rc:
            raise RuntimeError(f'Command failed ({rc}): {args}')
        return rc
    """)

encoded = base64.b64encode(payload).decode()
code('import base64, io\n' +
     f'BUNDLE_SHA256 = {bundle_hash!r}\n' +
     'BUNDLE_B64 = (\n' + '\n'.join(repr(encoded[i:i + 100]) for i in range(0, len(encoded), 100)) + '\n)\n' +
     'bundle_bytes = base64.b64decode(BUNDLE_B64)\n' +
     "assert hashlib.sha256(bundle_bytes).hexdigest() == BUNDLE_SHA256, 'Embedded bundle changed'\n" +
     'CODE.mkdir(parents=True, exist_ok=True)\n' +
     'with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as bundle:\n' +
     '    for member in bundle.infolist():\n' +
     '        if not (CODE / member.filename).resolve().is_relative_to(CODE.resolve()):\n' +
     "            raise ValueError('Unsafe archive path')\n" +
     '    bundle.extractall(CODE)\n' +
     "print('Verified embedded source bundle:', BUNDLE_SHA256)\n")

code("""
    # Preflight: versions, G1 unit tests (matches the teammate's verified recipe), synthetic smoke.
    import importlib
    for name in ('numpy', 'scipy', 'sklearn', 'torch', 'yaml'):
        print(f'{name:8}', getattr(importlib.import_module(name), '__version__', '?'))
    print('Python', platform.python_version(), '| workers', WORKERS)
    sh([sys.executable, 'g1_selftest.py'])
    sh([sys.executable, 'g1_selftest.py', '--smoke'])
    """)

code("""
    # Recover the same seven saved feature sets used by the strict nested run (Internet On).
    roots = [INPUT, RECOVERED]
    sh([sys.executable, 'nested_selection.py', 'discover', '--roots', *roots, '--out', SOURCE_INDEX], check=False)
    source_file = SOURCE_INDEX / 'sources.json'
    specs = json.loads(source_file.read_text())['sources'] if source_file.exists() else []
    expected_ids = {*(f'normal-erm-seed{i}' for i in range(5)), *(f'ermstar-seed{i}' for i in range(2))}
    if {s['id'] for s in specs} != expected_ids:
        print('Recovering saved features from public Kaggle outputs.')
        sh([sys.executable, '-c', 'from src.methods.dpe_independent.recovery import recover; recover(' + repr(str(RECOVERED)) + ')'])
        sh([sys.executable, 'nested_selection.py', 'discover', '--roots', *roots, '--out', SOURCE_INDEX])
        specs = json.loads(source_file.read_text())['sources']
    if {s['id'] for s in specs} != expected_ids:
        raise RuntimeError('Expected the original seven sources')
    print('Sources:', sorted(s['id'] for s in specs))
    """)

code("""
    # Run all 70 cells (resumable: finished cells are skipped), then summarize and package.
    rc = sh([sys.executable, 'g1_eval.py', 'run', '--sources', source_file, '--out', OUT, '--workers', WORKERS], check=False)
    result_zip = WORK / 'g1_test_eval_results.zip'
    with zipfile.ZipFile(result_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file():
                z.write(path, str(path.relative_to(OUT)))
    print('Results:', result_zip, '| report:', OUT / 'REPORT.md')
    if rc:
        for p in sorted((OUT / 'errors').glob('*.txt'))[:3]:
            print(p, '\\n', p.read_text())
        raise RuntimeError('Some cells failed; results so far were packaged. Re-run to resume.')
    """)

notebook = dict(cells=nb_cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                             language_info=dict(name='python')), nbformat=4, nbformat_minor=5)
nb_path = OUT / 'g1_test_eval_kaggle.ipynb'
nb_path.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
manifest = dict(notebook=nb_path.name, embedded_code_sha256=bundle_hash,
                files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (nb_path, OUT / 'g1_test_eval_code.zip')})
(OUT / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(dict(cells=len(nb_cells), notebook_bytes=nb_path.stat().st_size, code_zip_bytes=len(payload),
                      sha256=bundle_hash, reference_cells=cells), indent=2))
