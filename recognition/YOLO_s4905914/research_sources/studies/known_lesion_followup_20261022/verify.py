"""Boundary checks plus numerical tests of the actual changed loss."""
import json
import sys
import hashlib
from pathlib import Path
S=Path(__file__).resolve().parent; R=S.parents[1]
sys.path.insert(0,str(S/'code'))
from runtime import SPLITS, load_config, write_json
from dataset import read_manifests
from train import evaluate, check_yaml, plan
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((S/'plan.json').read_text())
assert SPLITS==('train','val')
assert not p['new_reassessment_inference']
for name,d in p['source_sha256'].items(): assert sha(S/'code'/name)==d,name
base=json.loads((R/p['reference_study']/'n_geometry.json').read_text())
checks={}
for name,change in p['variants'].items():
    path=S/f'{name}.json'; assert sha(path)==p['config_sha256'][name]
    cfg=load_config(path)
    assert {k:v for k,v in cfg['train'].items() if v!=base['train'][k]}==change['train']
    assert cfg['project_loss']['positive_weights']==change['positive_weights']
    assert cfg['evaluation']==base['evaluation'] and cfg['models']==base['models']
    rows,_=read_manifests(R/'data/known_lesion_v1',cfg)
    assert {k:len(v) for k,v in rows.items()}=={'train':768,'val':165}
    checks[name]=plan(path,R/'data/known_lesion_v1')
for func,args in ((evaluate,(None,None,None,None,None,None)),):
    try: func(*args,split='reassessment')
    except ValueError as e: assert 'prohibited' in str(e)
    else: raise AssertionError('Reassessment accepted')
try: check_yaml(None,'reassessment')
except ValueError: pass
else: raise AssertionError('Reassessment YAML accepted')

import torch
from types import SimpleNamespace
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from loss_weighting import install_training_loss
torch.set_num_threads(2);torch.manual_seed(20261022)
model=DetectionModel('yolov8n.yaml',nc=2,verbose=False)
model.args=SimpleNamespace(**base['train'])
batch={'img':torch.rand(2,3,96,96),'batch_idx':torch.tensor([0.,1.]),
       'cls':torch.tensor([[0.],[1.]]),'bboxes':torch.tensor([[.5,.5,.5,.5],[.5,.5,.4,.4]])}
feats=model(batch['img'])
plain=v8DetectionLoss(model)
ordinary=plain(feats,batch)[0]
unit=install_training_loss(model,[1.,1.])(feats,batch)[0]
torch.testing.assert_close(ordinary,unit,rtol=0,atol=0)
weighted=install_training_loss(model,[1.,2.])(feats,batch)[0]
torch.testing.assert_close(ordinary[[0,2]],weighted[[0,2]],rtol=0,atol=0)
assert weighted[1]>ordinary[1]
weighted.sum().backward()
assert all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
# Positive/negative/soft targets: malignant positive term only must change.
z=torch.tensor([[.3,-.2],[-.7,.6],[.2,.8]],requires_grad=True)
y=torch.tensor([[0.,0.],[1.,1.],[.2,.7]])
a=plain.bce(z,y); b=model.criterion.bce(z,y)
torch.testing.assert_close(a[:,0],b[:,0],rtol=0,atol=0)
torch.testing.assert_close(a[0,1],b[0,1],rtol=0,atol=0)
expected=a[:,1]-y[:,1]*torch.nn.functional.logsigmoid(z[:,1])
torch.testing.assert_close(b[:,1],expected)
write_json(S/'verification.json', {'status':'passed','plan_sha256':sha(S/'plan.json'),
    'reassessment_rejected_before_io':True,'real_images_read':0,'candidate_plans':checks,
    'unit_weights_exactly_match_upstream':True,'box_and_dfl_unchanged_for_fixed_predictions':True,
    'actual_detection_loss_backward_finite':True,'positive_term_formula_verified':True})
print('PASS: cohort guards, exact one-factor configs, upstream equivalence and actual loss backward')
