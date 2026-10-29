"""Release exactly the two seed-confirmation runs if the locked trigger is met."""
import copy,hashlib,json
from datetime import datetime
from pathlib import Path
S=Path(__file__).resolve().parent;R=S.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
r=read(S/'results.json');policy=read(S/'confirmation_policy.json')
assert r['status']=='completed' and r['decision']['confirmation_required']
assert sha(S/'plan.json')==policy['parent_plan_sha256']
assert not (S/'confirmation_plan.json').exists()
(S/'initial_results.json').write_bytes((S/'results.json').read_bytes())
best=r['decision']['highest_ranked']
variants={}
for name,parent in [('n_baseline_seed3711',R/'recognition/YOLO_s4905914/experiment_provisional.json'),(best+'_seed3711',S/f'{best}.json')]:
    cfg=copy.deepcopy(read(S/f'{best}.json'))
    cfg['train']=read(parent)['train'].copy();cfg['train']['seed']=3711
    cfg.update(version=name+'-20261021',purpose='known_lesion_validation_seed_confirmation',
               release_decision={'locked_at':datetime.now().astimezone().isoformat(),'basis':'Prespecified confirmation trigger was met','parent_config_sha256':sha(parent),'only_training_difference':'seed3711','run_scope':'Train/val only'},
               checkpoint_selection=f"best.pt by maximum validation macro mAP50:95 over the fixed {cfg['train']['epochs']} epochs; upstream later epoch wins exact ties")
    cfg['design_notes']=['Second-seed confirmation; all training settings except seed match its parent run','No reassessment inference','Report both seeds; do not choose the better seed']
    write(S/f'{name}.json',cfg)
    variants[name]={'parent_config':str(parent.relative_to(R)),'parent_config_sha256':sha(parent),'config_sha256':sha(S/f'{name}.json')}
write(S/'confirmation_plan.json',{'locked_at':datetime.now().astimezone().isoformat(),'candidate':best,'policy_sha256':sha(S/'confirmation_policy.json'),'initial_results_sha256':sha(S/'results.json'),'variants':variants,'new_reassessment_inference':False})
script=(S/'optimization_array_a100.sh').read_text()
script=script.replace('--time=04:00:00','--time=00:45:00')
script=script.replace('--array=0-2%3','--array=0-1%1').replace('--job-name=lesion-known-opt','--job-name=lesion-known-seed')
script=script.replace('outputs/known_opt_%j','outputs/known_seed_%j').replace('outputs/known_lesion_optimization_20261021_', 'outputs/known_lesion_confirmation_20261021_')
script=script.replace('"$study/verify_preparation.py"','"$study/verify_confirmation.py"')
script=script.replace('cp "$study/plan.json" "$run/plan.json"','cp "$study/confirmation_plan.json" "$run/confirmation_plan.json"\ncp "$study/confirmation_policy.json" "$run/confirmation_policy.json"')
script=script.replace('cp "$study/verification.json" "$run/verification.json"', 'cp "$study/confirmation_plan.json" "$run/verification_plan.json"')
script=script.replace('variants=(n_cls1 n_geometry n_long120)','variants=('+' '.join(variants)+')')
(S/'confirmation_array_a100.sh').write_text(script)
print(json.dumps({'candidate':best,'variants':list(variants)},indent=2))
