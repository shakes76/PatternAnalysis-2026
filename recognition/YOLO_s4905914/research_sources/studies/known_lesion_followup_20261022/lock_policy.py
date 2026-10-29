"""Lock one diagnosis-driven postprocessing comparison before scoring it."""
import hashlib,json
from datetime import datetime
from pathlib import Path
S=Path(__file__).resolve().parent;R=S.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
path=S/'single_lesion_plan.json'
assert not path.exists()
parent=json.loads((S/'plan.json').read_text())
old=R/parent['reference_study']
cfg=json.loads((old/'n_geometry.json').read_text())
x={'created_at':datetime.now().astimezone().isoformat(),'status':'locked_before_policy_scoring',
    'basis':'Raw-score autopsy found class argmax losses but no localized malignant loss to NMS; user authorized further validation improvements',
    'extension':'One explicit additional inference-policy trial; original three training comparisons remain unchanged',
    'input_models':['n_geometry seed3710','n_geometry seed3711 only if initial rule passes'],
    'rule':'At raw max-class-score floor 0.001 retain the single candidate with highest max(B,M); M if M/(B+M)>threshold, otherwise B; no GT is used in selection or decision',
    'ratio_is_malignancy_probability':False,'multi_lesion_applicability_proven':False,
    'threshold_grid':cfg['evaluation']['threshold_grid'],'benign_false_alert_budget':.1,
    'selection_rule':'Highest malignant TP; ties: detection precision, fewer benign alerts, higher ratio threshold',
    'improvement_rule':parent['improvement_rule'],
    'confirmation':'Apply unchanged policy and same grid to existing geometry seed3711 only if seed3710 meets rule; same-seed original operating point is reference',
    'comparison_to_training_round':'If both a training variant and this policy confirm, choose higher malignant TP then higher all-GT M IoU, fewer B alerts, then prefer unchanged standard inference; do not combine policy with new training variants',
    'new_reassessment_inference':False,'manifest_sha256':parent['manifest_sha256'],
    'code_sha256':sha(S/'single_lesion_policy.py'),
    'raw_diagnosis_sha256':sha(R/'outputs/known_followup_diagnostics_20261022/raw_diagnosis.json'),
    'scope':'exploratory validation-only extension after diagnostic evidence, before observing policy results'}
path.write_text(json.dumps(x,indent=2)+'\n')
print('Locked one additional GT-free policy trial')
