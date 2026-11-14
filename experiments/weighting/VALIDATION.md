# Implementation validation

Executed in the repository-local Python 3.13 environment:

```powershell
.venv/Scripts/python.exe -m pytest tests/methods/test_dpe_weighting.py -q --disable-warnings
```

**51 passed.** Includes all 41 deployable method families, oracle prohibition,
normalization, uniform parity, signed-score polarity, label/group isolation,
ID overlap rejection, gradient flow, seed determinism, head/cache reuse,
epoch-resume equality and an automatic two-phase synthetic orchestration run.
Synthetic outputs are confined to pytest temporary directories.

The native saved-head implementation and the new prediction adapter were also
compared on all 1,199 actual validation examples: maximum probability
difference **0**, with predeclared tolerance 1e-5. The checkpoint provenance's
training pool overlaps all 1,199 validation examples. Evidence and original
checkpoint hashes are in `outputs/weighting/preflight_audit.json`.

Broader historical tests were attempted. The existing ERM* notebook test writes
an unescaped Windows `C:\Users...` path into a generated Python string (and
initially used a non-UTF-8 default encoding). Existing two-GPU process tests
also patch POSIX-only `os.killpg`, unavailable on Windows. Those failures are
outside the new weighting workflow and are retained as platform limitations;
the preserved historical code/tests have not been modified to hide them.

After completion, run the independent artifact audit:

```powershell
.venv/Scripts/python.exe -m scripts.audit_dpe_weighting
```

It recomputes final accuracy, balanced accuracy, WGA and NLL from saved sample
outputs, checks unique full-test coverage, validates fold separation and
normalized weights, confirms all 42 ranks and verifies that original
checkpoint hashes have not changed. It does not prove the unavailable
historical backbone training provenance.
