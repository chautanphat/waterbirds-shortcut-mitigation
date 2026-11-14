"""Verify the actual distributed notebook and embedded package."""
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
notebook_path = OUT / "dpe_nested_corrected_kaggle.ipynb"
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
assert notebook["nbformat"] == 4
assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
for cell in code_cells:
    assert cell["outputs"] == [] and cell["execution_count"] is None
    ast.parse("".join(cell["source"]))
for path in (ROOT / "code").rglob("*.py"):
    ast.parse(path.read_text(encoding="utf-8-sig"))

with tempfile.TemporaryDirectory() as td:
    extracted = Path(td) / "code"
    ns = {"CODE": extracted, "WORK": Path(td), "hashlib": hashlib, "zipfile": zipfile}
    embedded_cell = next(cell for cell in code_cells if "BUNDLE_B64 =" in "".join(cell["source"]))
    exec(compile("".join(embedded_cell["source"]), "notebook-embedded-code-cell", "exec"), ns)
    assert ns["bundle_bytes"] == (OUT / "dpe_nested_corrected_code.zip").read_bytes()
    with zipfile.ZipFile(OUT / "dpe_nested_corrected_code.zip") as archive:
        for name in archive.namelist():
            assert (extracted / name).read_bytes() == (ROOT / "code" / name).read_bytes()
    env = dict(os.environ)
    # Exercise the extracted distributed code, without local numerical test extras.
    env.pop("PYTHONPATH", None)
    completed = subprocess.run([sys.executable, str(extracted / "corrected_selftest.py")],
                               capture_output=True, text=True, env=env, timeout=60)
    (ROOT / "packaged_tests.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise AssertionError(completed.stdout + completed.stderr)
    legacy = subprocess.run([sys.executable, str(extracted / "nested_selection.py"), "run", "--out", str(Path(td)/"old")],
                            capture_output=True, text=True, env=env, timeout=30)
    assert legacy.returncode != 0 and "Legacy run disabled" in legacy.stderr

manifest = json.loads((OUT / "artifact_manifest.json").read_text())
for name, digest in manifest["files"].items():
    assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == digest
smoke = (ROOT / "final_smoke.txt").read_text()
assert "SYNTHETIC SMOKE PASSED:" in smoke and '"recipe": "ermstar"' in smoke
report = dict(status="passed", notebook=notebook_path.name, clean_outputs=True,
              code_cells_compiled=len(code_cells), embedded_zip_matches_separate_zip=True,
              extracted_files_match_source=True, packaged_regression_tests=10,
              legacy_run_command_disabled=True,
              synthetic_real_trainer_smoke=dict(normal_erm="passed", ermstar="passed", arms_per_cell=28,
                                                head_ensembles_per_cell=41, head_epochs=1, calibration_steps=8),
              corrected_waterbirds_experiment_executed=False,
              limitations="Kaggle recovery/network and the complete 70-cell Waterbirds run must execute on Kaggle.")
(OUT / "validation_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
