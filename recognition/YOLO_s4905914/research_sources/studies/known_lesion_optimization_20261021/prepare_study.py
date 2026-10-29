"""Prepare isolated, validation-only ablations. Run once before submitting training."""
import copy, hashlib, json
from datetime import datetime
from pathlib import Path

STUDY=Path(__file__).resolve().parent
ROOT=STUDY.parents[1]
BASE=ROOT/'recognition/YOLO_s4905914'
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def replace(s,old,new):
    assert s.count(old)==1, old
    return s.replace(old,new)
code=STUDY/'code'
code.mkdir()
files=['runtime.py','dataset.py','train.py','evaluation.py','modules.py','preprocessing.py',
       'schedule.py','checkpoints.py','audit_real_inputs.py','preflight.py',
       'weights_registry.json','requirements.txt','requirements-linux-a100.lock.txt']
for name in files:
    s=(BASE/name).read_text()
    if name=='runtime.py':
        s=replace(s,"SPLITS = ('train', 'val', 'reassessment')","SPLITS = ('train', 'val')")
    elif name=='dataset.py':
        s=replace(s,'import math','import math\nfrom itertools import combinations')
        s=replace(s,'((0, 1), (0, 2), (1, 2))','combinations(range(len(SPLITS)), 2)')
        s=replace(s,"[('trainval', 'val'), ('reassessment', 'reassessment')]","[('trainval', 'val')]")
    elif name=='audit_real_inputs.py':
        s=replace(s,"splits=('train', 'val', 'reassessment')","splits=('train', 'val')")
    elif name=='train.py':
        s=replace(s,"def check_yaml(stage, name):\n", "def check_yaml(stage, name):\n    if name != 'trainval':\n        raise ValueError('This study permits train/val only')\n")
        s=replace(s,"def evaluate(config, prepared, stage, checkpoint, output, device, split='val', lock=None):\n", "def evaluate(config, prepared, stage, checkpoint, output, device, split='val', lock=None):\n    if split != 'val':\n        raise ValueError('Reassessment inference is prohibited in this optimization study')\n")
        s=replace(s,"choices=('val', 'reassessment')", "choices=('val',)")
    (code/name).write_text(s)
base=json.loads((BASE/'experiment_provisional.json').read_text())
variants={
 'n_long120': {'epochs':120},
 'n_cls1': {'cls':1.0},
 'n_geometry': {'degrees':15.0,'translate':0.05,'scale':0.15},
}
for name,delta in variants.items():
    cfg=copy.deepcopy(base)
    cfg.update(version=name+'-20261021', purpose='known_lesion_validation_optimization',
               model_policy='YOLOv8n ablations; original n/s results retained as validation references',
               manifest_sha256={k:base['manifest_sha256'][k] for k in ('train','val')},
               split_class_counts={k:base['split_class_counts'][k] for k in ('train','val')},
               teacher_guidance={'source':'User report in this conversation on 2026-10-21',
                   'known_lesion_cohort_approved_as_reported':True,
                   'full_course_compliance_or_checkoff_confirmed':False},
               release_decision={'locked_at':datetime.now().astimezone().isoformat(),
                   'basis':'User authorizes known-lesion-only validation optimization',
                   'run_scope':'Train 768 and validate 165; do not open reassessment images or results'},
               design_notes=['All variants start from original COCO n weights; same train/val manifests',
                  'No Mosaic, mixed-image augmentation, oversampling or color augmentation',
                  'A longer epoch budget also changes the linear learning-rate schedule',
                  'Increasing cls scales classification loss for BOTH classes',
                  'Geometry is one preset bundle; effects of rotation/translation/scale cannot be separated',
                  'Validation has already been used for model development; it is not an unbiased test'],
               decisions_pending=[s for s in base['decisions_pending'] if not s.startswith('Teacher acceptance')])
    cfg['train'].update(delta)
    cfg['checkpoint_selection']=f"best.pt by maximum validation macro mAP50:95 over the fixed {cfg['train']['epochs']} epochs; upstream later epoch wins exact ties"
    write(STUDY/f'{name}.json',cfg)
plan={
 'status':'locked_before_training','created_at':datetime.now().astimezone().isoformat(),
 'cohort':{'known_lesion_total':1098,'train':768,'validation':165,'excluded_from_this_study':165},
 'scope':'Known-lesion train/validation only; no new reassessment inference or selection',
 'baseline_run':'outputs/provisional_20261017_669143',
 'baseline_config_sha256':sha(BASE/'experiment_provisional.json'),
 'original_source_sha256':{p:sha(BASE/p) for p in files},
 'snapshot_source_sha256':{p:sha(code/p) for p in files},
 'variant_training_changes':variants,
 'config_sha256':{n:sha(STUDY/f'{n}.json') for n in variants},
 'same_for_all':['train/val images and labels','COCO n initialization','seed 3710','FP32','batch 16','imgsz 640',
                 'AdamW lr0 0.001','NMS IoU 0.7 single-label class-aware','match IoU 0.5',
                 '14-point threshold grid and 10% benign-image false-alert budget','EMA best macro mAP50:95 checkpoint'],
 'primary':'Malignant detection recall at the validation-selected threshold within <=10% benign image false alerts (max 12/124)',
 'candidate_selection':['highest malignant recall within budget','highest malignant all-GT class-correct IoU at selected threshold',
                        'highest malignant AP50:95','fewer benign image false alerts','prefer original n then lexicographic variant name'],
 'fixed_threshold_comparisons':[0.05,0.5],
 'secondary':['Malignant AP50 and AP50:95','Benign AP50:95','macro mAP50:95',
              'per-class all-GT class-correct IoU (missing correct-class box = 0)',
              'class-agnostic IoU and TP-only IoU, explicitly different denominators',
              'detection recall at matching IoU 0.75, with already-selected threshold'],
 'material_improvement_rule':'Recommend a development candidate only if malignant TP at selected point rises by >=3/41 and malignant all-GT class-correct IoU also increases versus baseline n; otherwise keep baseline and report no convincing gain',
 'confirmation_rule':'If a candidate meets the improvement rule, repeat baseline n and the best candidate once at seed 3711; report both seeds without choosing the better seed',
 'compute_limit':'Three initial n runs (240 total epochs); at most two confirmation runs',
 'no_additional_trials_after_results':True,
 'uncertainty':'Validation selection is optimistic; 41 malignant images and group dependence limit precision. Any bootstrap interval is conditional on fixed selected thresholds, not corrected for model/threshold selection.',
 'patient_independence_proven':False,
 'teacher_guidance':'User reports teacher directed using the known-lesion subset; no evidence of other course approvals',
 'new_reassessment_inference':False,
 'limitations':['one small repeatedly-used validation set','initial single seed','longer training changes LR decay and checkpoint-selection opportunities',
                'cls is not malignant-specific weighting','geometry bundle does not isolate individual transforms'],
}
write(STUDY/'plan.json',plan)
print(STUDY)
