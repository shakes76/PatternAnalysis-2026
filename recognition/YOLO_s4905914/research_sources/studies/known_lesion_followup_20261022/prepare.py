"""Lock a new user-authorized round, preserving the completed earlier study."""
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path

S = Path(__file__).resolve().parent
R = S.parents[1]
OLD = R / 'studies/known_lesion_optimization_20261021'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, x): p.write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

if (S/'plan.json').exists():
    raise RuntimeError('The plan is immutable; do not rerun preparation')
base = json.loads((OLD/'n_geometry.json').read_text())
variants = {
    'n_resolution896': {'model': 'n', 'train': {'imgsz': 896}, 'positive_weights': [1., 1.]},
    'n_positive2': {'model': 'n', 'train': {}, 'positive_weights': [1., 2.]},
    's_geometry': {'model': 's', 'train': {}, 'positive_weights': [1., 1.]},
}
now = datetime.now().astimezone().isoformat()
for name, change in variants.items():
    cfg = copy.deepcopy(base)
    cfg.update(version=name+'-20261022', purpose='known_lesion_validation_followup',
        project_loss={'positive_weights': change['positive_weights'], 'validation_loss': 'unweighted'},
        model_policy='One-factor follow-up relative to n_geometry; seed 3710 is the reporting seed',
        release_decision={'locked_at': now, 'basis': 'New user request to improve malignant recall and IoU using known-lesion train/validation only'},
        design_notes=['Same 768 train / 165 validation; reassessment prohibited',
            'Fresh original COCO weights; 60 epochs; same optimizer and light geometry',
            'One comparison changes resolution, one changes positive BCE weighting, one changes model size',
            'Weighting changes the malignant positive BCE term on soft targets; no resampling',
            'Training-only weighting; validation loss remains ordinary BCE',
            'Already-used validation remains exploratory development evidence'])
    cfg['train'].update(change['train'])
    write(S/f'{name}.json', cfg)

protected = [OLD/n for n in ('plan.json','initial_results.json','confirmation_policy.json',
    'confirmation_plan.json','results.json','decision.json','confirmation_results.json')]
protected += list((OLD/'code').glob('*.py'))
protected += list((R/'recognition/YOLO_s4905914').glob('*.py'))
protected += [R/'recognition/YOLO_s4905914/README.md', R/'output/pdf/COMP3710_Project3_Report.pdf']
write(S/'plan.json', {
    'status':'locked_before_training', 'created_at':now,
    'authorization':'User requested a new round after the previous bounded study was completed; earlier plan is preserved',
    'cohort':{'known_lesion_total':1098, 'train':768, 'validation':165, 'excluded_reassessment':165},
    'manifest_sha256':base['manifest_sha256'], 'scope':'validation only',
    'new_reassessment_inference':False, 'patient_independence_proven':False,
    'reference_study':'studies/known_lesion_optimization_20261021',
    'reference_variant':'n_geometry', 'reference_threshold':0.4,
    'variants':variants, 'config_sha256':{n:sha(S/f'{n}.json') for n in variants},
    'source_sha256':{p.name:sha(p) for p in (S/'code').iterdir() if p.is_file()},
    'protected_sha256':{str(p.relative_to(R)):sha(p) for p in protected},
    'primary':'Malignant detection recall at the original 14-point validation threshold grid under <=12/124 benign image false alerts',
    'ranking':['highest malignant TP','highest malignant all-GT class-correct IoU','highest malignant AP50:95','fewer benign false alerts','lexicographic name'],
    'improvement_rule':{'extra_malignant_TP_at_least':3,'malignant_all_GT_IoU':'strictly greater',
        'benign_all_GT_IoU':'no more than 0.03 lower than same-seed geometry reference'},
    'confirmation':'If any initial candidate meets the rule, confirm only the highest-ranked eligible candidate at seed 3711 against the existing geometry seed3711 result; same rule must pass; report original seed3710 model rather than selecting a seed',
    'compute_limit':'Three initial 60-epoch runs, at most one 60-epoch confirmation; no tuning within this round',
    'secondary':['AP per class','IoU per class: all-GT class-correct, class-agnostic, TP-only',
        'recall at IoU0.75 and IoU0.8 at selected threshold','fixed thresholds 0.05 and 0.4',
        'descriptive operating points at 20% and 30% benign false-alert budgets, not selection criteria'],
    'diagnostic_only':'Raw two-class scores and before/after single-label NMS on validation; any GT-based best candidate is an oracle diagnostic, never a prediction or performance claim',
    'limitations':['Repeated validation reuse and only 41 malignant cases','No guarantee of achieving course IoU target',
        'Course IoU denominator and passing hurdle remain unconfirmed','More comparisons increase selection optimism'],
    'sources':['https://github.com/ultralytics/ultralytics/blob/v8.3.221/ultralytics/utils/loss.py',
        'https://docs.pytorch.org/docs/2.8/generated/torch.nn.BCEWithLogitsLoss.html']})
print('Locked', S/'plan.json')
