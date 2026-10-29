"""Compare the two required seed3711 fits against their matched seed3710 parents."""
from pathlib import Path
import csv,hashlib,json,sys,math
import numpy as np
S=Path(__file__).resolve().parent;R=S.parents[1]
sys.path.insert(0,str(S/'code'))
from evaluation import select_threshold,score_records,gt_record
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
plan=read(S/'confirmation_plan.json');initial=read(S/'initial_results.json');runmap=read(S/'confirmation_runmap.json')
assert sha(S/'initial_results.json')==plan['initial_results_sha256']
assert sha(S/'confirmation_policy.json')==plan['policy_sha256']
assert initial['status']=='completed' and initial['decision']['confirmation_required']
assert set(runmap)==set(plan['variants'])
with (R/'data/known_lesion_v1/val.csv').open(newline='') as f:val=list(csv.DictReader(f))
with (R/'data/known_lesion_v1/train.csv').open(newline='') as f:train=list(csv.DictReader(f))
expected={r['image_id']:gt_record(r) for r in val}
expected_stage={r['image_id'] for r in train+val}
expected_manifests={split:sha(R/f'data/known_lesion_v1/{split}.csv') for split in ('train','val')}
original_files=read(R/'outputs/provisional_20261017_669143/inputs/staged/audit.json')['files']
reports={};provenance={}
for name,path in runmap.items():
    root=R/path;out=root/name;cfg=read(S/f'{name}.json')
    assert (root/'COMPLETED.txt').is_file()
    run=read(out/'train_n/run.json');ev=read(out/'val_n/evaluation.json');records=read(out/'val_n/predictions.json')
    assert run['status']=='completed' and run['config']==cfg and cfg['train']['seed']==3711
    assert sha(S/f'{name}.json')==run['config_sha256']==plan['variants'][name]['config_sha256']
    parent_info=plan['variants'][name];parent=read(R/parent_info['parent_config'])
    assert sha(R/parent_info['parent_config'])==parent_info['parent_config_sha256']
    parent['train']['seed']=3711
    metadata_keys={'version','purpose','model_policy','decisions_pending','design_notes',
                   'release_decision','teacher_guidance'}
    # Run labels/notes change for confirmation. The isolated loader also removes
    # reassessment entries; train/val identities and every execution setting stay fixed.
    for key in ('manifest_sha256','split_class_counts'):
        parent[key]={split:parent[key][split] for split in ('train','val')}
    assert {k:v for k,v in cfg.items() if k not in metadata_keys}=={
        k:v for k,v in parent.items() if k not in metadata_keys
    }, 'Only the training seed may differ among execution settings'
    assert run['initial_weights_sha256']==cfg['models']['n']['sha256']
    assert run['manifest_sha256']==expected_manifests
    assert run['optimizer_steps']==48*cfg['train']['epochs']
    with (out/'train_n/fit/results.csv').open() as stream:
        csv_rows=[{k.strip():float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
    assert [int(row['epoch']) for row in csv_rows]==list(range(1,cfg['train']['epochs']+1))
    assert all(math.isfinite(value) for row in csv_rows for value in row.values())
    assert sha(out/'train_n/fit/weights/best.pt')==run['best_sha256']==ev['checkpoint_sha256']
    assert ev['split']=='val' and ev['manifest_sha256']==expected_manifests
    assert ev['config_sha256']==run['config_sha256']
    assert ev['training_run_sha256']==sha(out/'train_n/run.json')
    assert {r['image_id'] for r in records}==set(expected) and len(records)==165
    for r in records:
        assert r['class_id']==expected[r['image_id']]['class_id']
        assert np.allclose(r['box'],expected[r['image_id']]['box'],atol=1e-12,rtol=0)
    selected=select_threshold(records,cfg['evaluation']);assert selected==ev['operating_point']
    lock=read(out/'val_n/selection_lock.json')
    assert lock['threshold']==selected['selected']['threshold']
    stage=read(out/'inputs/staged/audit.json');assert set(stage['files'])==expected_stage
    audit=read(out/'inputs/input_audit.json');assert audit['status']=='passed' and audit['decoded_images']==933
    assert stage['manifest_sha256']==audit['manifest_sha256']==expected_manifests
    assert stage['config_sha256']==audit['config_sha256']==run['config_sha256']
    assert audit['staged_audit_sha256']==sha(out/'inputs/staged/audit.json')
    assert audit['counts']=={'train':768,'val':165} and audit['cross_split_exact_decoded_duplicates']==0
    for image_id, item in stage['files'].items():
        for field in ('size','file_sha256','pixel_sha256'):
            assert item[field]==original_files[image_id][field], (image_id,field)
    assert not list(root.rglob('reassessment.yaml')) and not (out/'reassessment').exists()
    assert set((root/'environment.freeze.txt').read_text().splitlines())==set((R/'outputs/provisional_20261017_669143/environment.freeze.txt').read_text().splitlines())
    for key,digest in read(S/'plan.json')['snapshot_source_sha256'].items():
        if key.endswith('.py'):assert run['source_sha256'][key]==digest
    reports[name]={'selected':selected['selected'],'grid':selected['grid'],'mAP50':ev['mAP50'],'mAP50_95':ev['mAP50_95'],'ap':ev['ap'],
        'resources':{k:run[k] for k in ('elapsed_seconds','optimizer_steps','parameters','peak_cuda_allocated_bytes','checkpoint_selection')},
        'fixed_thresholds':{str(t):score_records(records,t) for t in (0.05,0.5)}}
    provenance[name]={'output_root':path,'config_sha256':run['config_sha256'],'checkpoint_sha256':run['best_sha256'],
        'evaluation_sha256':sha(out/'val_n/evaluation.json'),'predictions_sha256':sha(out/'val_n/predictions.json'),
        'input_audit_sha256':sha(out/'inputs/input_audit.json'),'environment_equal_to_original':True,
        'only_training_parameter_changed_from_parent':'seed: 3710 to 3711',
        'configuration_metadata_may_differ':sorted(metadata_keys),
        'scope_normalization':'Reassessment entries removed; train/val hashes and counts unchanged',
        'staged_file_and_pixel_hashes_match_original':True}
best=plan['candidate'];a=reports['n_baseline_seed3711']['selected'];b=reports[best+'_seed3711']['selected']
am=a['classes']['Malignant'];bm=b['classes']['Malignant']
passed=(bm['tp']-am['tp']>=3 and bm['iou']['all_gt_class_correct']['mean']>am['iou']['all_gt_class_correct']['mean'])
rows=[]
for name,d in [('Original n / 3710',initial['reports']['baseline_n']),('Candidate / 3710',initial['reports'][best]),('Original n / 3711',reports['n_baseline_seed3711']),('Candidate / 3711',reports[best+'_seed3711'])]:
    p=d['selected'];m=p['classes']['Malignant']
    rows.append([name,f"{p['threshold']:.2f}",m['tp'],p['image_alerts']['benign_fp'],f"{m['iou']['all_gt_class_correct']['mean']:.4f}"])
interpretation=f"At seed 3711, the candidate detects {bm['tp']}/41 malignant lesions versus {am['tp']}/41 for the matched baseline, with {b['image_alerts']['benign_fp']} versus {a['image_alerts']['benign_fp']} benign alerts. Malignant all-GT IoU is {bm['iou']['all_gt_class_correct']['mean']:.4f} versus {am['iou']['all_gt_class_correct']['mean']:.4f}. "
interpretation+=('The gain passes the prespecified confirmation rule on this validation set. This does not establish performance on new patients.' if passed else 'This fails the prespecified confirmation rule. The initial improvement is therefore insufficient for replacing the original main reference; all runs remain reported.')
result={'status':'completed','scope':'validation_only','candidate':best,'new_reassessment_inference':False,
    'confirmation_plan_sha256':sha(S/'confirmation_plan.json'),'reports':reports,'provenance':provenance,
    'passes_predeclared_confirmation_rule':passed,'report_rows':rows,'report_interpretation':interpretation}
(S/'confirmation_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'rows':rows,'passes':passed,'interpretation':interpretation},indent=2))
