"""Reject unplanned confirmation changes before any image loading."""
import hashlib,json
from pathlib import Path
S=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=read(S/'confirmation_plan.json');initial=read(S/'initial_results.json')
assert sha(S/'initial_results.json')==p['initial_results_sha256']
assert sha(S/'plan.json')==p['parent_plan_sha256']
assert initial['status']=='completed_initial' and initial['confirmation_candidate']==p['parent_variant']
assert not p['new_reassessment_inference']
path=S/(p['candidate']+'.json');assert sha(path)==p['config_sha256']
a=read(path);b=read(S/(p['parent_variant']+'.json'))
assert {k:v for k,v in a['train'].items() if v!=b['train'][k]}=={'seed':3711}
for field in set(a)-{'train','version','model_policy'}:assert a[field]==b[field],field
print('Confirmation plan/config verified; seed is the only training change')
