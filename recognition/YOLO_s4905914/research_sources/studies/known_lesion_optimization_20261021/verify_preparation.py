"""Verify study boundaries and exact ablations before GPU submission."""
import hashlib, json, sys
from pathlib import Path
S=Path(__file__).resolve().parent
R=S.parents[1]
sys.path.insert(0,str(S/'code'))
from runtime import SPLITS, load_config
from dataset import read_manifests
from train import evaluate, check_yaml, plan
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((S/'plan.json').read_text())
assert SPLITS==('train','val')
assert not p['new_reassessment_inference']
for name,digest in p['snapshot_source_sha256'].items(): assert sha(S/'code'/name)==digest,name
base=load_config(R/'recognition/YOLO_s4905914/experiment_provisional.json')
checks={}
for n,delta in p['variant_training_changes'].items():
    file=S/f'{n}.json'
    assert sha(file)==p['config_sha256'][n]
    c=load_config(file)
    assert {k:v for k,v in c['train'].items() if v!=base['train'][k]}==delta
    assert c['evaluation']==base['evaluation']
    assert c['models']==base['models']
    rows,_=read_manifests(R/'data/known_lesion_v1',c)
    assert {k:len(v) for k,v in rows.items()}=={'train':768,'val':165}
    checks[n]=plan(file,R/'data/known_lesion_v1')
# These calls must reject before trying to read any argument or file.
try:
    evaluate(None,None,None,None,None,None,split='reassessment')
    raise AssertionError('Reassessment was not rejected')
except ValueError as e: assert 'prohibited' in str(e)
try:
    check_yaml(None,'reassessment')
    raise AssertionError('Reassessment YAML was not rejected')
except ValueError as e: assert 'only' in str(e)
# Evaluation/scoring, preprocessing, model building and schedule stay byte-identical.
for name in ('evaluation.py','preprocessing.py','modules.py','schedule.py','checkpoints.py'):
    assert p['original_source_sha256'][name]==p['snapshot_source_sha256'][name]
result={'status':'passed','plan_sha256':sha(S/'plan.json'),'reassessment_rejected_before_io':True,
        'known_ids_and_no_known_train_val_overlap':True,'images_decoded':0,
        'candidate_plans':checks}
(S/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print({'status':'passed','variants':list(checks),'train':768,'val':165,'reassessment_reads':0})
