"""Create a small, isolated final-evaluation bundle from the fixed closeout plan."""
import hashlib,json,shutil,tarfile
from pathlib import Path
study=Path(__file__).resolve().parent
root=study.parents[1]
code=root/'recognition/YOLO_s4905914'
plan=json.loads((code/'course_closeout_plan.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected_plan='d83574913228042f529e8a8b479d223ada9d7b5c13b87ba3a3169610245652ad'
assert digest(code/'course_closeout_plan.json')==expected_plan
for name,h in plan['reporting_code_sha256'].items():assert digest(code/name)==h,name
chosen=plan['selected_model']
files=set(plan['selection_records_sha256'])|{chosen['config_path'],chosen['checkpoint_path'],'outputs/provisional_20261017_669143/inputs/staged/audit.json'}
files.update('data/known_lesion_v1/'+s+'.csv' for s in ('train','val','reassessment'))
for rel,h in plan['selection_records_sha256'].items():assert digest(root/rel)==h,rel
assert digest(root/chosen['checkpoint_path'])==chosen['checkpoint_sha256']
for s,h in plan['manifest_sha256'].items():assert digest(root/'data/known_lesion_v1'/f'{s}.csv')==h
lock=json.loads((root/chosen['evaluation_path']/'selection_lock.json').read_text())
assert digest(root/chosen['config_path'])==lock['config_sha256']
for name in ('code','locked_inputs'):
    if (study/name).exists():raise FileExistsError('Do not overwrite a frozen bundle: '+name)
(study/'code').mkdir();(study/'locked_inputs').mkdir()
codefiles=set(plan['reporting_code_sha256'])|{'evaluate_locked_candidate.py','preprocessing.py','modules.py','weights_registry.json','course_closeout_plan.json','requirements.txt','requirements-linux-a100.lock.txt'}
for name in sorted(codefiles):shutil.copy2(code/name,study/'code'/name)
for rel in sorted(files):
    dst=study/'locked_inputs'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/rel,dst)
paths=list((study/'code').iterdir())+list((study/'locked_inputs').rglob('*'))+[study/'evaluate_a100.sh',study/'verify_bundle.py']
manifest={str(p.relative_to(study)):digest(p) for p in paths if p.is_file()}
(study/'bundle_manifest.json').write_text(json.dumps({'plan_sha256':expected_plan,'files':manifest,'new_training':False,'threshold_search':False,'selection_changed':False},indent=2)+'\n')
archive=root/'outputs/closeout_20261022/final_reassessment_bundle.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for rel in sorted(manifest):tar.add(study/rel,arcname='final_reassessment_20261022/'+rel)
    tar.add(study/'bundle_manifest.json',arcname='final_reassessment_20261022/bundle_manifest.json')
print(json.dumps({'archive':str(archive),'size':archive.stat().st_size,'sha256':digest(archive),'files':len(manifest)},indent=2))
