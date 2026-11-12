import csv,json,hashlib,shutil,argparse
from pathlib import Path
import numpy as np
from scipy.stats import t
parser=argparse.ArgumentParser(description='Lightweight independent audit of an extracted DPE verification ZIP; no torch or training.')
parser.add_argument('--input',type=Path,required=True,help='Extracted bundle containing results/ and logs/')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--archive',type=Path,help='Original ZIP for SHA256 provenance')
args=parser.parse_args();RAW=args.input
contexts=[p for p in (RAW/'results').iterdir() if (p/'status.json').exists()]
if len(contexts)!=1:raise ValueError('Expected exactly one result context')
SRC=contexts[0];DEST=args.output;DEST.mkdir(parents=True,exist_ok=True)
read=lambda n:list(csv.DictReader((SRC/n).open()))
rows=read('per_seed.csv');part=read('partition_sensitivity.csv');rand=read('combiner_randomness.csv');controls=read('controls.csv')
assert [len(x) for x in [rows,part,rand,controls]]==[150,300,200,45]
assert len({(r['source_seed'],r['configuration']) for r in rows})==150
def numpy_predict(row, p):
    """Independent forward calculation: does not call the model's PyTorch forward."""
    p = np.asarray(p, dtype=np.float64)
    if row['family'] == 'uniform': return np.einsum('h,hnc->nc', np.ones(len(p)) / len(p), p)
    if 'weights' in row: return np.einsum('h,hnc->nc', np.asarray(row['weights']), p)
    model = row['model']; state = {k: np.asarray(v) for k, v in model['state'].items()}
    q = p[:, :, 1].T; x = 2 * q - 1
    if model['kind'] == 'logistic': z = x @ state['network.weight'].T + state['network.bias']
    else: z = np.tanh(x @ state['network.0.weight'].T + state['network.0.bias']) @ state['network.2.weight'].T + state['network.2.bias']
    if model['kind'] == 'gate':
        weight = np.exp(z - z.max(1, keepdims=True)); weight /= weight.sum(1, keepdims=True)
        value = (weight * q).sum(1)
    else: value = 1 / (1 + np.exp(-z[:, 0]))
    return np.column_stack([1 - value, value])
ns={"numpy_predict":numpy_predict}
lookup={(int(r['source_seed']),r['configuration']):r for r in rows};checked=0;ids0=None;y0=None;g0=None;certs=[]
for s in range(5):
 src=SRC/f'source_bundles/seed{s}';m=json.loads((src/'experiment/manifest.json').read_text());h=json.loads((src/'experiment/heads.json').read_text())
 digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 assert digest(src/'checkpoints/ensemble.pt')==h['lock']['ensemble'] and digest(src/'experiment/manifest.json')==h['lock']['manifest']
 assert len(m['parts']['head_training'])==720 and len(m['parts']['combiner_training'])==479
 assert not {r['id'] for r in m['parts']['head_training']}&{r['id'] for r in m['parts']['combiner_training']}
 f=np.load(src/'experiment/predictions/test.npz',allow_pickle=False);p=f['probabilities'];y=f['labels'];g=f['groups'].astype(str);ids=f['ids'].astype(str)
 if ids0 is None:ids0=ids.copy();y0=y.copy();g0=g.copy()
 else:assert np.array_equal(ids0,ids) and np.array_equal(y0,y) and np.array_equal(g0,g)
 assert not set(ids)&{r['id'] for r in m['validation']}
 frozen=json.loads((SRC/f'frozen/primary-seed{s}.json').read_text());states=frozen['value']
 assert frozen['identity']['heads']==h['lock']['ensemble'] and frozen['identity']['manifest']==h['lock']['manifest']
 assert frozen['identity']['training_predictions']==digest(src/'experiment/predictions/combiner_training.npz')
 assert str(f['ensemble_hash'])==h['lock']['ensemble'] and str(f['role'])=='test'
 for state in states:
  r=lookup[s,state['id']];correct=ns['numpy_predict'](state,p).argmax(1)==y
  counts=[int(np.sum(g==v)) for v in ['00','01','10','11']];assert counts==[2255,2255,642,642]
  accuracy=[]
  for group in ['00','01','10','11']:
   count=int(correct[g==group].sum());assert count==int(r['correct_'+group]);accuracy.append(count/int(r['count_'+group]))
  assert abs(min(accuracy)-float(r['wga']))<1e-14;checked+=1
 image=json.loads((SRC/f'images/seed{s}.json').read_text());assert image['feature_max_abs_diff']==0 and image['head_probability_max_abs_diff']==0 and image['changed_head_decisions']==0
 assert all(x['changed_decisions']==0 for x in image['configurations'].values())
 for split in [1,2]:
  manifest=json.loads((SRC/f'partitions/seed{s}-split{split}-manifest.json').read_text());a,b=manifest['parts'].values();assert len(a)==720 and len(b)==479 and not {r['id'] for r in a}&{r['id'] for r in b}
  f2=np.load(SRC/f'partitions/seed{s}-split{split}-test.npz',allow_pickle=False);pp=f2['probabilities'];yy=f2['labels'];gg=f2['groups'].astype(str)
  saved=json.loads((SRC/f'frozen/partition{split}-seed{s}.json').read_text())['value']
  for state in saved:
   r=next(r for r in part if int(r['source_seed'])==s and int(r['partition_seed'])==split and r['configuration']==state['id'])
   pred=ns['numpy_predict'](state,pp).argmax(1);wga=min(float((pred[gg==group]==yy[gg==group]).mean()) for group in ['00','01','10','11']);assert abs(wga-float(r['wga']))<1e-14
   if state.get('fit',{}):
    cert=state['fit'].get('certificate')
    if cert:assert cert['objective_gap_bound']<=1e-7;certs.append(cert['objective_gap_bound'])
# Check all nested combiner predictions, independently of the worker's torch forward.
for seed in range(5):
    f=np.load(SRC/f'source_bundles/seed{seed}/experiment/predictions/test.npz',allow_pickle=False)
    p,y,g=f['probabilities'],f['labels'],f['groups'].astype(str)
    fitted=json.loads((SRC/f'frozen/randomness-seed{seed}.json').read_text())['value']
    for state in fitted:
        row=next(r for r in rand if int(r['source_seed'])==seed and int(r['combiner_seed'])==state['combiner_seed'] and float(r['weight_decay'])==state['weight_decay'] and r['family']==state['family'])
        correct=numpy_predict(state,p).argmax(1)==y
        assert abs(min(float(correct[g==v].mean()) for v in ['00','01','10','11'])-float(row['wga']))<1e-14
for r in controls:
    assert abs(min(int(r['correct_'+v])/int(r['count_'+v]) for v in ['00','01','10','11'])-float(r['wga']))<1e-14
for cohort,seeds in [('original_0_1_2',[0,1,2]),('new_3_4',[3,4]),('combined_five',list(range(5)))]:
    for r in read('summary_by_cohort.csv'):
        if r['cohort']!=cohort:continue
        values=[float(lookup[s,r['configuration']]['wga']) for s in seeds]
        assert len(values)==int(r['n_source_seeds'])
        assert abs(np.mean(values)-float(r['mean_wga']))<1e-14
        assert abs(np.std(values,ddof=1)-float(r['sample_sd_wga']))<1e-14
assert all(int(r['changed_reference_decisions'])==0 for r in rows if int(r['source_seed'])<3)
intervals=[]
for method,ref in [(f'{fam}-wd{wd}',ref) for fam in ['mlp','mlp_dropout'] for wd in [.01,.001] for ref in ['uniform']]+[(f'mlp_dropout-wd{wd}',f'mlp-wd{wd}') for wd in [.01,.001]]:
 d=np.array([100*(float(lookup[s,method]['wga'])-float(lookup[s,ref]['wga'])) for s in range(5)]);ci=t.ppf(.975,4)*d.std(ddof=1)/np.sqrt(5)
 intervals.append(dict(method=method,reference=ref,mean_delta_pp=float(d.mean()),source_t95_lower_pp=float(d.mean()-ci),source_t95_upper_pp=float(d.mean()+ci),wins=int((d>0).sum()),per_source_delta_pp=d.tolist()))
print('AUDIT',checked,'primary and300 partition model metrics independently reproduced')
print('INTERVALS',json.dumps(intervals,indent=1))
for split in [1,2]:
 for name in ['uniform','mlp-wd0.01','mlp_dropout-wd0.01','mlp-wd0.001','mlp_dropout-wd0.001']:
  v=[r for r in part if int(r['partition_seed'])==split and r['configuration']==name];print('PARTITION',split,name,100*np.mean([float(r['wga']) for r in v]),'gain',np.mean([float(r['delta_uniform_within_partition_pp']) for r in v]),'wins',sum(float(r['delta_uniform_within_partition_pp'])>0 for r in v))
for p in SRC.iterdir():
 if p.is_file() and p.suffix in ['.csv','.json','.txt']:shutil.copy2(p,DEST/p.name)
shutil.copytree(SRC/'images',DEST/'images',dirs_exist_ok=True)
for name in ['solver_recovery.jsonl','solver_partition_preflight.json','solver_recovery_provenance.json']:shutil.copy2(RAW/'logs'/name,DEST/name)
audit=dict(status='passed',independent_primary_forward_and_counts=150,independent_partition_forward_and_metrics=300,source_bundle_hashes_verified=5,raw_image_replays_exact=5,groups=[2255,2255,642,642],independent_nested_combiner_forward_and_metrics=200,control_rows=45,intervals=intervals,uploaded_zip_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest() if args.archive else None,limits='No backbone retraining, raw-image replay or control-head inference performed locally; image and control evidence supplied by Kaggle. Numerical forward checks on saved predictions only.')
(DEST/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
