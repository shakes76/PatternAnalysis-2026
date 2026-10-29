from pathlib import Path
import hashlib,json,sys
S=Path(__file__).resolve().parent;R=S.parents[1]
sys.path.insert(0,str(S/'code'))
from runtime import load_config
from dataset import read_manifests
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
p=read(S/'confirmation_plan.json');policy=read(S/'confirmation_policy.json')
assert not p['new_reassessment_inference']
assert sha(S/'confirmation_policy.json')==p['policy_sha256']
for name,source in p['variants'].items():
    assert sha(R/source['parent_config'])==source['parent_config_sha256']
    assert sha(S/f'{name}.json')==source['config_sha256']
    c=load_config(S/f'{name}.json');base=load_config(R/source['parent_config'])
    assert {k:v for k,v in c['train'].items() if v!=base['train'][k]}=={'seed':3711}
    assert c['evaluation']==base['evaluation']
    rows,_=read_manifests(R/'data/known_lesion_v1',c)
    assert {k:len(v) for k,v in rows.items()}=={'train':768,'val':165}
for name,digest in read(S/'plan.json')['snapshot_source_sha256'].items():assert sha(S/'code'/name)==digest
print('Confirmation configuration, data and frozen runtime verified')
