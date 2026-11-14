"""Build a self-contained Kaggle notebook and its reviewable source bundle."""
import ast
import base64
import hashlib
import io
import json
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
OUT = ROOT / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)

archive = io.BytesIO()
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
    for path in sorted(CODE.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"):
            z.write(path, str(path.relative_to(CODE)).replace("\\", "/"))
payload = archive.getvalue()
bundle_hash = hashlib.sha256(payload).hexdigest()
(OUT / "dpe_nested_corrected_code.zip").write_bytes(payload)

cells = []
def md(text):
    cells.append(dict(cell_type="markdown", id=f"cell-{len(cells):02d}", metadata={}, source=textwrap.dedent(text).strip().splitlines(True)))
def code(text):
    text = textwrap.dedent(text).strip() + "\n"
    ast.parse(text)
    cells.append(dict(cell_type="code", id=f"cell-{len(cells):02d}", metadata={}, execution_count=None, outputs=[], source=text.splitlines(True)))

md("""
    # Corrected DPE: strict nested selection and exact threshold controls

    **Self-contained code: upload this notebook to Kaggle; no code dataset is required.**
    Turn **Internet On**, select a **CPU runtime**, then **Save & Run All**.
    The experiment uses CPU head fitting; GPU acceleration is not required.

    Uses the same seven saved feature sets (five standard ERM, two ERM*) as
    `noerula/dpe-nested-code`, recovered from attached inputs or the existing public
    output API. No backbones are retrained. Missing feature sets cause an explicit stop.

    ## Declared corrected protocol

    - Five inner evaluation folds. For each fold, rebuild five OOF ensembles and
      a full-refit ensemble **using that inner training pool only**. The entire
      inner evaluation fold is excluded from every head and calibration fit.
    - Inner evaluation uses the complete prediction procedure used on test:
      calibrated fold averaging, post-average bias/thresholding, or OOF calibration
      transferred to the full-refit ensemble. Each family selects independently.
    - Preserve the original four (loss, penalty) settings: average/0.1,
      maximum/0.1, average/0.01, maximum/0.01. Fixed setting: average/0.1.
      **Primary selection: balanced NLL; secondary: WGA.** Original tie breakers remain.
    - Direct WGA thresholds search every attainable decision partition, with
      equal scores moved together. They are applied **after averaging probabilities**.
      Foldwise bias calibration is reported under its own explicit name.
    - Keep full-validation DPE, a 75-head bag, cross-fit uniform, pooled 60/40
      affine/linear controls, and DFR-lite. DFR standardization is fitted inside CV.
    - All ten split seeds and seven sources are declared. Folds are not independent
      backbones; intervals use per-source split means. ERM* has only two sources.
    - The test set and source backbones were previously inspected. This remains
      **exploratory**, even with the corrected selection pipeline.

    Expect substantially more CPU time than the original run: **41 head ensembles
    per source/split instead of 11** (each ensemble contains 15 heads). Stage caches
    permit resuming after a time limit; no completed old-protocol results are reused.
    """)

code("""
    # Keep these declared seeds/rules fixed while running this experiment.
    SPLITS = list(range(10))
    EXPERIMENT_TIME_LIMIT_HOURS = 8.5
    WORKERS = 4  # capped at the number of available CPU cores below
    RUN_SAVED_REGRESSION_CHECK = True

    import os, sys, json, shutil, subprocess, zipfile, hashlib, platform
    from pathlib import Path
    WORK = Path('/kaggle/working')
    INPUT = Path('/kaggle/input')
    if not WORK.is_dir():
        raise SystemExit('Run this notebook in Kaggle. The code ZIP also supports local CLI use.')
    STATE = WORK / 'corrected_dpe_state'
    OUT = STATE / 'nested_v2'
    CODE = WORK / 'corrected_code_v2'
    RECOVERED = STATE / 'recovered_sources'
    STATE.mkdir(parents=True, exist_ok=True)
    WORKERS = max(1, min(WORKERS, os.cpu_count() or 1))

    def sh(args, check=True):
        args = [str(a) for a in args]
        print('$', ' '.join(args), flush=True)
        process = subprocess.Popen(args, cwd=CODE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
        rc = process.wait()
        if check and rc:
            raise RuntimeError(f'Command failed ({rc}): {args}')
        return rc
    """)

encoded = base64.b64encode(payload).decode()
code("import base64, io\n" +
     f"BUNDLE_SHA256 = {bundle_hash!r}\n" +
     "# Embedded source ZIP; a separate copy is provided for review.\n" +
     "BUNDLE_B64 = (\n" + "\n".join(repr(encoded[i:i+100]) for i in range(0, len(encoded), 100)) + "\n)\n" +
     "bundle_bytes = base64.b64decode(BUNDLE_B64)\n" +
     "assert hashlib.sha256(bundle_bytes).hexdigest() == BUNDLE_SHA256, 'Embedded bundle changed'\n" +
     "CODE.mkdir(parents=True, exist_ok=True)\n" +
     "with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as bundle:\n" +
     "    for member in bundle.infolist():\n" +
     "        target = (CODE / member.filename).resolve()\n" +
     "        if not target.is_relative_to(CODE.resolve()):\n" +
     "            raise ValueError('Unsafe archive path')\n" +
     "    bundle.extractall(CODE)\n" +
     "print('Verified embedded source bundle:', BUNDLE_SHA256)\n")

code("""
    # Preflight: software versions and the regression tests (no Waterbirds fits).
    import importlib
    required = ('numpy', 'scipy', 'sklearn', 'torch', 'torchvision', 'yaml', 'pandas', 'tqdm', 'PIL')
    for name in required:
        mod = importlib.import_module(name)
        print(f'{name:12} {getattr(mod, "__version__", "?")}')
    print('Python:', platform.python_version(), '| CPU workers:', WORKERS)
    sh([sys.executable, 'corrected_selftest.py'])
    sys.path.insert(0, str(CODE))
    import corrected_nested as corrected
    print('Protocol:', corrected.PROTOCOL)
    """)

md("""
    ## Resume an interrupted corrected run

    Attach its Kaggle output, or upload `corrected_nested_resume.zip` as a Dataset
    (Kaggle can extract it). This cell chooses one compatible state with the most
    progress. Old `e2e_state/nested` results are never imported. If software/source
    fingerprints differ, the run stops instead of silently mixing results.
    """)
code("""
    # Support an uploaded resume ZIP when Kaggle has not already extracted it.
    resume_zips = sorted(INPUT.rglob('corrected_nested_resume.zip'))
    resume_roots = []
    for number, path in enumerate(resume_zips):
        target_root = WORK / f'resume_archive_{number}'
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if not (target_root / member.filename).resolve().is_relative_to(target_root.resolve()):
                    raise ValueError('Unsafe resume archive path')
            archive.extractall(target_root)
        resume_roots.append(target_root)
    candidates = []
    for root in [INPUT, *resume_roots]:
        for prior in root.rglob('corrected_dpe_state'):
            nested = prior / 'nested_v2'
            marker = nested / 'protocol.json'
            if marker.is_file():
                protocol = json.loads(marker.read_text())
                if protocol.get('protocol') == corrected.PROTOCOL and protocol.get('code_sha256') == corrected.code_identity():
                    candidates.append(prior)
    if candidates and not (OUT / 'protocol.json').exists():
        prior = max(candidates, key=lambda p: (len(list((p/'nested_v2/results').glob('*/split*.json'))),
                                               len(list((p/'nested_v2/checkpoints').rglob('complete.json')))))
        shutil.copytree(prior / 'nested_v2', OUT, dirs_exist_ok=True)
        print('Restored corrected checkpoints from:', prior)
    else:
        print('Using current state, or starting a fresh corrected run.')
    corrected.prepare_protocol(OUT)  # validates code, settings and runtime identity
    """)

code("""
    # Recover the SAME saved feature sets; no backbone retraining fallback.
    SOURCE_INDEX = STATE / 'source_index'
    roots = [INPUT, RECOVERED]
    sh([sys.executable, 'nested_selection.py', 'discover', '--roots', *roots,
        '--out', SOURCE_INDEX], check=False)
    source_file = SOURCE_INDEX / 'sources.json'
    specs = json.loads(source_file.read_text())['sources'] if source_file.exists() else []
    expected = {*(f'normal-erm-seed{i}' for i in range(5)), *(f'ermstar-seed{i}' for i in range(2))}
    if {s['id'] for s in specs} != expected:
        print('Recovering missing saved features from public Kaggle outputs (Internet must be On).')
        recovery_code = 'from src.methods.dpe_independent.recovery import recover; recover(' + repr(str(RECOVERED)) + ')'
        sh([sys.executable, '-c', recovery_code])
        sh([sys.executable, 'nested_selection.py', 'discover', '--roots', *roots, '--out', SOURCE_INDEX])
        specs = json.loads(source_file.read_text())['sources']
    actual = {s['id'] for s in specs}
    if actual != expected or len(specs) != 7:
        raise RuntimeError(f'Expected the original seven sources. Missing={expected-actual}; unexpected={actual-expected}')
    (OUT / 'sources.json').write_text(json.dumps(dict(sources=specs, blockers=[]), indent=2))
    print('Sources:', sorted(actual))
    if RUN_SAVED_REGRESSION_CHECK:
        sh([sys.executable, 'nested_selection.py', 'check', '--out', OUT])
    """)

md("""
    ## Run all declared cells

    The budget is checked between head-ensemble fits. Completed stages are saved
    immediately; a timeout returns a partial run that can be resumed. Do not change
    settings after reading interim test results. Run all ten seeds before interpreting
    the final comparison.
    """)
code("""
    run_rc = sh([sys.executable, 'corrected_nested.py', 'run', '--out', OUT,
                 '--splits', *SPLITS, '--workers', WORKERS,
                 '--time-limit-hours', EXPERIMENT_TIME_LIMIT_HOURS], check=False)
    print('Run return code:', run_rc, '(0=complete, 3=budget reached, 1=error)')
    """)

code("""
    # Always package progress, including partial runs, before surfacing a run error.
    sh([sys.executable, 'corrected_nested.py', 'summarize', '--out', OUT])
    result_zip = WORK / 'corrected_nested_results.zip'
    with zipfile.ZipFile(result_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and 'checkpoints' not in path.relative_to(OUT).parts:
                archive.write(path, str(path.relative_to(OUT)))
    resume_zip = WORK / 'corrected_nested_resume.zip'
    with zipfile.ZipFile(resume_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and not any(part.startswith('.pending-') for part in path.parts):
                archive.write(path, str(path.relative_to(WORK)))
    print('Analysis bundle:', result_zip)
    print('Resume bundle (includes prediction checkpoints):', resume_zip)
    print('Reports:', OUT/'REPORT.md', OUT/'CORRECTED_DIAGNOSTICS.md')
    completion = json.loads((OUT/'completion.json').read_text()) if (OUT/'completion.json').exists() else {}
    print('Completion:', completion)
    if run_rc == 3:
        print('PARTIAL: attach this output or the resume ZIP to the same notebook and Save & Run All again.')
    elif run_rc != 0:
        errors = sorted((OUT/'errors').glob('*.txt'))
        for path in errors[:3]:
            print(path, '\\n', path.read_text())
        raise RuntimeError('Experiment failed; progress was packaged. Inspect the error files before resuming.')
    """)

notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                          language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
notebook_path = OUT / "dpe_nested_corrected_kaggle.ipynb"
notebook_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

instructions = """Corrected DPE Kaggle rerun

1. In Kaggle, create a notebook and import dpe_nested_corrected_kaggle.ipynb.
2. Turn Internet On. CPU is sufficient; the head trainers run on CPU.
3. Optionally attach the earlier notebook's output to recover features from inputs.
   Otherwise the notebook retrieves the same seven feature sets from public outputs.
4. Choose Save & Run All. Keep the declared ten split seeds and selection rules.
5. Download corrected_nested_results.zip for analysis.
6. If the budget is reached, attach the corrected output or upload
   corrected_nested_resume.zip as a Dataset, then Save & Run All again.

The notebook embeds the complete code ZIP. A separate dpe_nested_corrected_code.zip
is provided for inspection/local CLI use; uploading it as a code dataset is optional.

Changes: fully nested head/calibrator rebuilding; full inference-graph inner
evaluation; independent family selection; exact empirical WGA threshold search;
explicit bias/threshold after averaging; foldwise bias kept as a separate control;
DFR preprocessing inside folds; retained original pooled controls; stage checksums
and source/config/runtime fingerprints; failures propagated; partial-run labeling.

The primary selection rule remains balanced NLL, secondary WGA. The old four
calibrated candidates and the fixed average/0.1 arm are retained. Old-protocol result
caches are never imported. Reusing raw frozen backbone features is supported.

Compute: 41 fifteen-head ensembles per source/split (previously 11), with 70 cells.
The default budget is 8.5 hours; it may require more than one Kaggle session.
Completed head-ensemble predictions are retained between sessions.

Validation: unit tests and synthetic smoke tests are software checks. No corrected
Waterbirds performance is claimed until you execute the full notebook. The test
set was inspected previously, so this rerun remains exploratory evidence.
"""
(OUT / "RUN_INSTRUCTIONS.txt").write_text(instructions, encoding="utf-8")
manifest = dict(notebook=notebook_path.name, embedded_code_sha256=bundle_hash,
                files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       (notebook_path, OUT / "dpe_nested_corrected_code.zip", OUT / "RUN_INSTRUCTIONS.txt")})
(OUT / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(dict(cells=len(cells), notebook_bytes=notebook_path.stat().st_size,
                      code_zip_bytes=len(payload), embedded_code_sha256=bundle_hash), indent=2))
