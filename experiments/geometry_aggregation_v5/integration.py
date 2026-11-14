"""Conditional production acceptance after a passing, frozen nested gate."""
import json
import os
from pathlib import Path
import subprocess
import sys
from experiments.discrete_weighting.store import sha
from .persistence import atomic,dump


def integrate(journal,gate):
    if gate['status']!='PASS' or gate['integration']['id']!='G1_g50':raise ValueError('No supported passed recipe for production integration')
    root=journal.root
    def acceptance():
        command=[sys.executable,'-m','pytest','tests/methods/test_geometry_aggregation_v5.py','tests/methods/test_dpe_geometry_inference.py','-q']
        env=os.environ|dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1')
        run=subprocess.run(command,env=env,capture_output=True,text=True,encoding='utf-8')
        log=root/'production/acceptance_tests.log';atomic(log,run.stdout+'\n'+run.stderr)
        if run.returncode:raise RuntimeError('Production acceptance tests failed: '+str(log))
        files=[Path(p) for p in ['baselines/dpe4subpop/src/dpe/core.py','baselines/dpe4subpop/src/dpe/geometry.py','baselines/dpe4subpop/src/dpe/verified_distance_v1.json','tests/methods/test_dpe_geometry_inference.py']]
        path=root/'production/integration_receipt.json';dump(path,dict(status='COMPLETE',mode='distance',default='uniform',candidate='G1_g50',
            gate_sha256=sha(root/'promotion_decisions.json'),sources={str(p):sha(p) for p in files},
            tests=run.stdout.strip(),real_DPE_API_cache_parity=True,all9_C_prediction_caches_exact=True,
            command='model.predict_proba(target="val", aggregation="distance")',
            CLI='PYTHONPATH=baselines/dpe4subpop/src python -m dpe.geometry --input-cache CACHE.npz --output NEW.npz --aggregation distance',
            serialization='versioned verified_distance_v1.json; no fitted calibration or normalizer needed by G1',
            dependency_note='netcal1.3.5 installed without dependencies, as required by vendor DPE evaluation import; numerical experiment dependencies unchanged.'))
        snapshots=[]
        for source in files:
            p=root/'production/source_snapshot'/source;atomic(p,source.read_bytes());snapshots.append(p)
        return dict(integration='COMPLETE',mode='distance',default='uniform',tests_returncode=0),[path,log]+snapshots
    journal.task('production:acceptance',['final:report'],acceptance)
    def addendum():
        original=(root/'FINAL_REPORT.md').read_text(encoding='utf-8')
        original=original.replace('Production integration: REQUIRED for the first passing prelocked recipe; uniform must remain default.',
            'Production integration: COMPLETE for G1_g50 as opt-in distance mode; default remains uniform. S2 is not promoted.')
        original+='\n## Verified opt-in use\n\n`model.predict_proba(target="val", aggregation="distance")` enables only the frozen G1 recipe gamma50/alpha.25. Omitting aggregation preserves the prior uniform probability averaging, member probabilities and raw-logit API. `model.evaluate(target="val", aggregation="distance")` also opts in explicitly.\n\n'
        original+='Versioned metadata is at `baselines/dpe4subpop/src/dpe/verified_distance_v1.json`. It states exact normalized-distance/logit representation, class order,15 members, fixed parameters and verification limits. G1 needs no trainable normalizer/calibrator. The implementation computes normalized geometric distances once, then uses learned head scale for probabilities while retaining unscaled distance for routing.\n\n'
        original+='Prediction-cache CLI: `PYTHONPATH=baselines/dpe4subpop/src python -m dpe.geometry --input-cache CACHE.npz --output NEW.npz --aggregation distance`. Its default is uniform; it loads only named probabilities/distances/IDs, accepts no labels/groups, and refuses to overwrite an existing output.\n\n'
        original+='Production acceptance verifies exact probabilities/weights on all nine saved C caches, the actual DPE class with real newly trained heads/features, unchanged uniform/member/raw-logit outputs, metadata roundtrip and CLI behavior. `production/integration_receipt.json` records the test log and source hashes.\n\n'
        original+='Revision r2 adds bounded retries for Windows/OneDrive atomic rename locks. The local training adapter has AST-identical functions to the historical adapter; no model, grid, split or selection math changed. The incomplete head resumed from its last valid epoch, completed tasks were retained and the original sources/manifests/logs remain preserved. Revision r3 wires conditional production acceptance into the single resume command after the nested gate; it does not recompute experiments.\n'
        path=root/'FINAL_REPORT_COMPLETE.md';atomic(path,original)
        return dict(final_report='FINAL_REPORT_COMPLETE.md',integration='COMPLETE'),[path]
    journal.task('production:report',['production:acceptance'],addendum)
    journal.status('COMPLETE','None under frozen v5 protocol; preserve results, require fresh confirmatory data for scientific claims',
        promotion='PASS',integration='COMPLETE_OPT_IN_DISTANCE',default_aggregation='uniform',report='FINAL_REPORT_COMPLETE.md')
