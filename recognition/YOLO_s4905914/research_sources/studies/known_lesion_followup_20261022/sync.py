"""Download completed runs with no staged images or labels."""
import json,subprocess
from pathlib import Path
S=Path(__file__).resolve().parent;R=S.parents[1]
host='s4905914@rangpur.compute.eait.uq.edu.au';sock='/tmp/comp3710-knownid-20261021.sock'
remote='/home/Student/s4905914/comp3710-restart-20261017'
status=json.loads(subprocess.check_output(['ssh','-S',sock,host,
    f'cd {remote} && .python313/bin/python studies/known_lesion_followup_20261022/remote_status.py'],text=True))
runmap={}
for name,item in status.items():
    if 'model' not in item: continue
    path=item['run'];assert path.startswith('outputs/known_lesion_followup_20261022_') and '..' not in path
    runmap[name]=path
    if item['phase']!='completed':continue
    local=R/path;model=item['model']
    if (local/name/f'val_{model}/evaluation.json').exists() and (local/'COMPLETED.txt').exists():continue
    local.mkdir(exist_ok=True,parents=True)
    subprocess.run(['rsync','-a','-e',f'ssh -S {sock}', '--exclude=images/','--exclude=labels/',
        '--exclude=*.cache','--exclude=last.pt',f'{host}:{remote}/{path}/',str(local)+'/'],check=True)
for file,x in [('runmap.json',runmap),('remote_progress.json',status)]:
    (S/file).write_text(json.dumps(x,indent=2)+'\n')
state=json.loads((S/'RUN_STATUS.json').read_text())
if state['status']!='completed':
    state.update(status='running',runs=status,new_reassessment_inference=False)
    (S/'RUN_STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
    (S/'ACTIVE_WORK.txt').write_text('Active: bounded validation-only follow-up. Job 684063.\n'
        'Read RUN_STATUS.json and remote_progress.json before resubmitting anything.\n'
        'Do not evaluate the reassessment split. Original selected model remains active until confirmation.\n')
print(json.dumps(status,indent=2))
