"""Finalize a bounded round; do not substitute reassessment evidence."""
import json,hashlib,csv,sys
from pathlib import Path
S=Path(__file__).resolve().parent;R=S.parents[1]
sys.path.insert(0,str(S/'code'))
from evaluation import gt_record
from compare import report,qualifies
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    x=read(S/'results.json');plan=read(S/'plan.json');policy=read(S/'policy_decision.json')
    assert x['status']=='completed_initial' and policy['recommendation']=='reject_policy'
    winner=x['confirmation_candidate'];passed=False;confirmation=None
    if winner:
        cp=read(S/'confirmation_plan.json');name=cp['candidate'];runmap=read(S/'runmap.json')
        assert cp['parent_plan_sha256']==sha(S/'plan.json')
        assert cp['initial_results_sha256']==sha(S/'initial_results.json')
        assert cp['parent_variant']==winner
        cfg=S/f'{name}.json';assert sha(cfg)==cp['config_sha256']
        c=read(cfg);orig=read(S/f'{winner}.json')
        assert {k:v for k,v in c['train'].items() if v!=orig['train'][k]}=={'seed':3711}
        for field in ('evaluation','project_loss','models','manifest_sha256','split_class_counts'):assert c[field]==orig[field]
        expected={r['image_id']:gt_record(r) for r in csv.DictReader((R/'data/known_lesion_v1/val.csv').open())}
        cand=report(name,R/runmap[name]/name,cp['model'],cfg,expected,plan)
        old=R/plan['reference_study'];old_confirm=read(old/'confirmation_results.json')
        prev=old_confirm['reports']['n_geometry_seed3711']
        prov=old_confirm['provenance']['n_geometry_seed3711']
        prev_path=R/prov['output_root']/'n_geometry_seed3711'
        assert sha(prev_path/'val_n/evaluation.json')==prov['evaluation_sha256']
        assert sha(prev_path/'val_n/predictions.json')==prov['predictions_sha256']
        assert sha(prev_path/'train_n/fit/weights/best.pt')==prov['checkpoint_sha256']
        passed=qualifies(cand,prev)
        confirmation={'reference':prev,'candidate':cand,'passed':passed,'seed':3711,
            'new_reassessment_inference':False,'same_small_validation_set':True}
        (S/'confirmation_results.json').write_text(json.dumps(confirmation,indent=2)+'\n')
    selected=winner if winner and passed else 'reference_geometry'
    selected_report=x['reports'][selected]
    decision={'status':'completed','selected':selected,
        'confidence':selected_report['selected']['threshold'],
        'checkpoint_path':selected_report['checkpoint_path'],'checkpoint_sha256':selected_report['checkpoint_sha256'],
        'config_path':selected_report['config_path'],'evaluation_path':selected_report['evaluation_path'],
        'confirmation_triggered':bool(winner),'confirmation_passed':passed if winner else None,
        'single_lesion_policy':'rejected','further_trials_planned':False,
        'new_reassessment_inference':False,'course_test_IoU_target_proven':False,
        'reason':'Candidate passed both predeclared same-seed comparisons' if passed else
            'No candidate passed the complete improvement rule; retain the prior geometry model',
        'evidence_scope':'Repeatedly-used validation; exploratory selection, not new independent test',
        'initial_results_sha256':sha(S/'results.json'),'parent_plan_sha256':sha(S/'plan.json')}
    assert all(sha(R/n)==d for n,d in plan['protected_sha256'].items())
    (S/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    status=read(S/'RUN_STATUS.json');status.update(status='completed',selected=selected,
        confidence=decision['confidence'],further_trials_planned=False,new_reassessment_inference=False)
    (S/'RUN_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    (S/'ACTIVE_WORK.txt').write_text('Completed: this bounded validation-only round.\n'
        f'Selected: {selected}; confidence {decision["confidence"]}.\n'
        'No further training planned; no reassessment inference; see decision.json.\n')
    print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
