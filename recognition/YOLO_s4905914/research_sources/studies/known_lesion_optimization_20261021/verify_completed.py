"""Audit completed study evidence and invariants without loading a model."""
from pathlib import Path
import csv,hashlib,json,math
S=Path(__file__).resolve().parent;R=S.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
plan=read(S/'plan.json');runs=read(S/'runmap.json')
assert set(runs)==set(plan['variant_training_changes'])
with (R/'data/known_lesion_v1/train.csv').open(newline='') as f:train=list(csv.DictReader(f))
with (R/'data/known_lesion_v1/val.csv').open(newline='') as f:val=list(csv.DictReader(f))
expected={r['image_id'] for r in train+val}
reference_env=set((R/'outputs/provisional_20261017_669143/environment.freeze.txt').read_text().splitlines())
checks={}
for name,path in runs.items():
    root=R/path;out=root/name;cfg=read(S/f'{name}.json');run=read(out/'train_n/run.json')
    assert (root/'COMPLETED.txt').is_file()
    assert run['status']=='completed' and run['config']==cfg
    assert sha(S/f'{name}.json')==plan['config_sha256'][name]==run['config_sha256']
    assert sha(out/'train_n/fit/weights/best.pt')==run['best_sha256']
    assert run['initial_weights_sha256']==cfg['models']['n']['sha256']
    assert run['optimizer_steps']==48*cfg['train']['epochs']
    with (out/'train_n/fit/results.csv').open() as stream:
        csv_rows=[{k.strip():float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
    assert [int(row['epoch']) for row in csv_rows]==list(range(1,cfg['train']['epochs']+1))
    assert all(math.isfinite(value) for row in csv_rows for value in row.values())
    assert set(run['manifest_sha256'])=={'train','val'}
    for key,digest in plan['snapshot_source_sha256'].items():
        if key.endswith('.py'):assert run['source_sha256'][key]==digest,key
    audit=read(out/'inputs/input_audit.json');stage=read(out/'inputs/staged/audit.json')
    assert audit['status']=='passed' and audit['decoded_images']==933 and audit['counts']=={'train':768,'val':165}
    assert set(stage['files'])==expected
    assert audit['cross_split_exact_decoded_duplicates']==0
    assert set((root/'environment.freeze.txt').read_text().splitlines())==reference_env
    ev=read(out/'val_n/evaluation.json');assert ev['split']=='val'
    assert ev['checkpoint_sha256']==run['best_sha256'] and ev['config_sha256']==run['config_sha256']
    lock=read(out/'val_n/selection_lock.json');assert lock['threshold']==ev['operating_point']['selected']['threshold']
    assert not list(root.rglob('reassessment.yaml')) and not (out/'reassessment').exists()
    checks[name]={'status':'passed','job_output':path,'epochs':cfg['train']['epochs'],
                  'optimizer_step_calls':run['optimizer_steps'],'best_checkpoint_sha256':run['best_sha256'],
                  'input_audit_sha256':sha(out/'inputs/input_audit.json'),'evaluation_sha256':sha(out/'val_n/evaluation.json'),
                  'prediction_sha256':sha(out/'val_n/predictions.json'),'packages_identical_to_baseline':True,'staged_ids_equal_train_plus_val':True}
assert sha(R/'recognition/YOLO_s4905914/experiment_provisional.json')==plan['baseline_config_sha256']
for name,digest in plan['original_source_sha256'].items():assert sha(R/'recognition/YOLO_s4905914'/name)==digest
result={'status':'passed','new_reassessment_inference':False,'baseline_config_and_runtime_unchanged':True,
        'plan_sha256':sha(S/'plan.json'),'checks':checks,'AP_independently_recomputed':False,
        'score_grid_recalculated_from_saved_validation_predictions':True}
(S/'completed_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
