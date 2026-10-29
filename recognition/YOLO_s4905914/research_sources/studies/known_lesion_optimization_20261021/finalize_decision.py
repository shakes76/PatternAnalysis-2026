"""Apply the locked stopping/confirmation rules to completed validation results."""
from pathlib import Path
import hashlib,json
S=Path(__file__).resolve().parent
R=S.parents[1]
r=json.loads((S/'results.json').read_text())
assert r['status']=='completed'
policy=json.loads((S/'confirmation_policy.json').read_text())
best=r['decision']['highest_ranked'];trigger=r['decision']['meets_predeclared_material_improvement_rule']
b=r['reports']['baseline_n']['selected'];p=r['reports'][best]['selected']
bm=b['classes']['Malignant'];pm=p['classes']['Malignant']
if trigger:
    c=json.loads((S/'confirmation_results.json').read_text())
    assert c['status']=='completed' and c['candidate']==best and not c['new_reassessment_inference']
    r['confirmation']=c
    success=c['passes_predeclared_confirmation_rule']
else:success=False
chosen=best if success else 'baseline_n'
point=r['reports'][chosen]['selected']; provenance=r['provenance'][chosen]
config_path=(R/'recognition/YOLO_s4905914/experiment_provisional.json'
             if chosen=='baseline_n' else S/f'{chosen}.json')
evaluation_path=R/provenance['evaluation_path']
checkpoint_path=evaluation_path.parent/'train_n/fit/weights/best.pt'
lock_path=evaluation_path/'selection_lock.json'
lock=json.loads(lock_path.read_text())
assert lock['selection_split']=='val' and lock['threshold']==point['threshold']
assert lock['config_sha256']==hashlib.sha256(config_path.read_bytes()).hexdigest()
assert lock['checkpoint_sha256']==r['reports'][chosen]['checkpoint_sha256']==hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
reason=(f"The initial comparison ranked {best} highest: {pm['tp']}/41 malignant TP and {p['image_alerts']['benign_fp']}/124 benign alerts at confidence {p['threshold']:.2f}, with malignant all-GT correct-class IoU {pm['iou']['all_gt_class_correct']['mean']:.4f}. Original n had {bm['tp']}/41 TP, {b['image_alerts']['benign_fp']}/124 benign alerts and IoU {bm['iou']['all_gt_class_correct']['mean']:.4f} at confidence {b['threshold']:.2f}. ")
if not trigger:
    reason+='The prespecified material-improvement rule was not met, so no confirmation runs were triggered. I retain original n at 0.50 as the main reference. Changes in AP or a small difference in TP are reported, but do not establish a reliable solution to the malignant-detection weakness.'
elif not success:
    reason+='The gain did not meet the confirmation rule at seed 3711. I therefore retain original n at 0.50 as the main reference and report the candidate improvement as seed-sensitive.'
else:
    reason+=f"The gain met the same rule at seed 3711. I recommend the seed 3710 {best} checkpoint at confidence {point['threshold']:.2f} for further coursework development, while preserving the original n/s evaluation as the historical baseline. This is repeated improvement on the same validation images; it is not new independent test evidence."
r['final_decision']={'status':'completed','candidate':chosen,'confidence':point['threshold'],
    'checkpoint_sha256':r['reports'][chosen]['checkpoint_sha256'],
    'config_path':str(config_path.relative_to(R)),
    'checkpoint_path':str(checkpoint_path.relative_to(R)),
    'selection_lock_path':str(lock_path.relative_to(R)),
    'evaluation_path':provenance['evaluation_path'],'model_selection_evidence':'validation only',
    'confirmation_policy_sha256':hashlib.sha256((S/'confirmation_policy.json').read_bytes()).hexdigest(),
    'confirmation_triggered':trigger,'confirmation_passed':success,
    'report_summary':reason,'further_training_planned':False,'new_reassessment_inference':False,
    'expanded_training_models_eligible':False,'clinical_safety_or_course_pass_claim':False}
# Keep the initial selection record, then make the current status agree with
# the completed confirmation decision. Never alter the pretraining policy.
r.setdefault('initial_decision', dict(r['decision']))
r['decision'].update(recommendation=chosen, confirmation_required=False,
    confirmation_triggered=trigger, confirmation_completed=trigger,
    confirmation_passed=success, status='completed')
(S/'results.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
(S/'decision.json').write_text(json.dumps(r['final_decision'],indent=2,allow_nan=False)+'\n')
print(json.dumps(r['final_decision'],indent=2))
