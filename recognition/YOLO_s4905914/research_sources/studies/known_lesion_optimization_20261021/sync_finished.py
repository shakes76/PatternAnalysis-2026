"""Fetch completed validation-only runs; never follows staged source-image links."""
import json,subprocess
from pathlib import Path
S=Path(__file__).resolve().parent;R=S.parents[1]
host='s4905914@rangpur.compute.eait.uq.edu.au'
sock='/tmp/comp3710-knownid-20261021.sock'
remote='/home/Student/s4905914/comp3710-restart-20261017'
status=json.loads(subprocess.check_output(['ssh','-S',sock,host,
    f'cd {remote} && .python313/bin/python studies/known_lesion_optimization_20261021/remote_status.py'],text=True))
runmap={}
for name,item in status.items():
    if name not in ('n_cls1','n_geometry','n_long120'):continue
    path=item['run'];assert path.startswith('outputs/known_lesion_optimization_20261021_') and '..' not in path
    runmap[name]=path
    if item['phase']!='completed':continue
    done=R/path/'COMPLETED.txt'
    best=R/path/name/'train_n/fit/weights/best.pt'
    if done.exists() and best.exists():continue
    (R/path).mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-a','-e',f'ssh -S {sock}','--exclude=images/','--exclude=labels/',
        '--exclude=*.cache','--exclude=last.pt',f'{host}:{remote}/{path}/',str(R/path)+'/'],check=True)
(S/'runmap.json').write_text(json.dumps(runmap,indent=2)+'\n')
(S/'remote_progress.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status,indent=2))
