"""Launch, preservation, provenance and cost-boundary checks; never train."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def notebook():
    return json.loads((ROOT/'notebooks/experiment2_poter_cost_optimized.ipynb').read_text(encoding='utf-8'))


def test_notebook_code_compiles_and_has_no_executed_predictions(notebook):
    assert notebook['nbformat']==4
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            compile(cell['source'],'notebook-cell','exec')
            assert cell['outputs']==[] and cell['execution_count'] is None
    try:import nbformat
    except ImportError:return
    nbformat.validate(nbformat.from_dict(notebook))


def test_capsule_matches_historical_cub_algorithm_and_declares_other_dataset_gap(notebook):
    cell=next(c['source'] for c in notebook['cells'] if c['source'].startswith('PAYLOAD ='))
    variables={}
    for node in ast.parse(cell).body:
        if isinstance(node,ast.Assign) and node.targets[0].id=='PAYLOAD':
            payload=ast.literal_eval(node.value);break
    capsule=base64.b64decode(payload)
    provenance=notebook['metadata']['experiment2']
    assert hashlib.sha256(capsule).hexdigest()==provenance['source_capsule_sha256']
    assert set(provenance['source_deviations'])=={'data/celebA_dataset.py'}
    with zipfile.ZipFile(io.BytesIO(capsule)) as archive:
        for name,expected in provenance['source_hashes'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==expected
        for name in ['train.py','loss.py','ot_reweight.py','data/cub_dataset.py',
                     'data/dro_dataset.py','scripts/kaggle_waterbirds.py','scripts/training_checkpoint.py']:
            assert hashlib.sha256(archive.read(name)).hexdigest()==provenance['reference']['source_sha256'][name]


def test_notebook_default_is_nontraining_and_preserves_existing_files(notebook,tmp_path):
    settings=notebook['cells'][1]['source'];namespace={}
    exec(compile(settings,'settings','exec'),namespace)
    assert namespace['CONFIRM_TRAINING'] is False and namespace['SEED']==2
    path=tmp_path/'progress.pt';path.write_bytes(b'committed optimizer/RNG state')
    preserve=namespace['preserve_write']
    preserve(path,path.read_bytes())
    with pytest.raises(ValueError,match='preserved, not overwritten'):
        preserve(path,b'new seed or altered state')
    assert path.read_bytes()==b'committed optimizer/RNG state'
    launch=next(c['source'] for c in notebook['cells'] if c['source'].startswith('if not CONFIRM_TRAINING:'))
    # Execute the default skip branch: no ROOT/torch/subprocess/run_dir exists.
    exec(compile(launch,'launch','exec'),{'CONFIRM_TRAINING':False})


@pytest.mark.parametrize('arguments',[
    [],['--confirm-training','--','--epochs','2','--batch-size','128'],
    ['--confirm-training','--','--epochs','300','--batch-size','16']])
def test_profile_wrapper_rejects_unconfirmed_or_changed_recipe(arguments,tmp_path):
    command=[sys.executable,str(ROOT/'scripts/profile_poter_training.py'),
             '--baseline-root',str(tmp_path/'missing-baseline'),
             '--profile-output',str(tmp_path/'no-profile.json'),*arguments]
    result=subprocess.run(command,text=True,capture_output=True)
    assert result.returncode==2
    assert not (tmp_path/'no-profile.json').exists()
    assert 'Training is disabled' in result.stderr or 'must remain' in result.stderr


def test_current_plan_cost_and_incomplete_seed_claim():
    from scripts.report_experiment2_interim import priority_plan
    config=json.loads((ROOT/'configs/experiment2_interim.json').read_text())
    manifest=json.loads((ROOT/'outputs/experiment2_interim/checkpoint_manifest.json').read_text())
    plan=priority_plan(config,manifest)
    assert plan['additional_poter_seeds_to_train']==[2]
    assert plan['baseline_seeds_to_reuse']=={'ERM_enhanced':[0],'DPE_legacy':[0]}
    assert plan['new_plan_poter_training_hours']==plan['old_plan_poter_training_hours']/2
    assert plan['new_inference_needed']==2100 and plan['reusable_prediction_rows']==6300
    assert plan['priority_target_prediction_rows']==8400
    assert not plan['training_started'] and not plan['training_authorized']
    assert config['deferred']=={'POTER':[0],'ERM_enhanced':[1,2],'DPE_legacy':[1,2]}
    assert 'incomplete' in plan['current_independent_seed_validation']


def test_interim_records_real_profile_and_missing_cache_without_faking_speedup():
    folder=ROOT/'outputs/experiment2_interim'
    profile=json.loads((folder/'data_pipeline_profile.json').read_text())
    assert not profile['training_executed'] and not profile['model_loaded']
    assert profile['split']=='train only' and profile['samples']==256
    assert profile['compressed_byte_cache']['decoded_pixels_equal']
    assert {r['workers'] for r in profile['loader']}=={0,2}
    assert all(r['wall_seconds']>0 for r in profile['loader'])
    cache=json.loads((folder/'feature_cache_audit.json').read_text())
    assert cache['matches_original']==(cache['sha256']==cache['expected_sha256'])
    status=json.loads((folder/'training_status.json').read_text())
    assert not status['multi_seed_validation_complete']
    assert status['processes_terminated']==status['checkpoint_files_modified']==0
