"""Create exactly one confirmation config only after all initial trials finish."""
import copy,hashlib,json
from datetime import datetime
from pathlib import Path
S=Path(__file__).resolve().parent;R=S.parents[1]
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
results=read(S/'results.json');plan=read(S/'plan.json')
assert results['status']=='completed_initial'
name=results['confirmation_candidate']
assert name and name in results['eligible']
assert not (S/'confirmation_plan.json').exists()
cfg=read(S/f'{name}.json');cfg['train']['seed']=3711
cfg['version']=name+'-seed3711-20261022'
cfg['model_policy']='Confirmation only; original seed3710 remains reporting seed'
candidate=name+'_seed3711'
path=S/f'{candidate}.json';path.write_text(json.dumps(cfg,indent=2)+'\n')
confirmation={'status':'locked_before_confirmation','created_at':datetime.now().astimezone().isoformat(),
    'candidate':candidate,'parent_variant':name,'model':plan['variants'][name]['model'],
    'config_sha256':sha(path),'initial_results_sha256':sha(S/'results.json'),'parent_plan_sha256':sha(S/'plan.json'),
    'reference':'existing n_geometry seed3711, chosen under the same 10% benign-alert budget',
    'new_reassessment_inference':False,'no_further_trials':True}
(S/'initial_results.json').write_bytes((S/'results.json').read_bytes())
(S/'confirmation_plan.json').write_text(json.dumps(confirmation,indent=2)+'\n')
script=(S/'array_a100.sh').read_text()
script=script.replace('#SBATCH --array=0-2%3\n','')
script=script.replace('variants=(n_resolution896 n_positive2 s_geometry)\nmodels=(n n s)\nmodel=${models[$SLURM_ARRAY_TASK_ID]}\nvariant=${variants[$SLURM_ARRAY_TASK_ID]}',
    f'model={confirmation["model"]}\nvariant={candidate}')
script=script.replace('cp "$study/plan.json" "$run/plan.json"',
    'cp "$study/plan.json" "$run/plan.json"\ncp "$study/confirmation_plan.json" "$run/confirmation_plan.json"')
script=script.replace('cp "$study/verification.json" "$run/verification.json"',
    'cp "$study/verification.json" "$run/verification.json"\n"$py" "$study/verify_confirmation.py" > "$run/confirmation_check.log" 2>&1')
(S/'confirmation_a100.sh').write_text(script)
print(json.dumps(confirmation,indent=2))
