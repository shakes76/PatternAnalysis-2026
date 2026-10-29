"""R3b only: A42/B42 three epochs, read-only E3/E4 prefix comparison."""
import argparse
import ast
import json
import math
from pathlib import Path
import traceback
import platform
import torch
import PIL
from check_dataloader import require,sha256
from train import write_json
from run_e4 import validate_arm
from run_r3_cnn import run_cnn

HISTORICAL={'modules.py':'5d7d27560eb7fe3c8bf8350b93f310670cfcabf1f24663574859b4c60a105bff','dataset.py':'6adf19bec8385501311f309ddbeb2d2f7dd554c8e198bd9494009ce28743cfd4'}

def source_bridge():
 root=Path(__file__).parent;result={}
 for name,digest in HISTORICAL.items():
  old=root/'reference_sources/cnn_v1'/(name+'.txt');require(sha256(old)==digest,'Historical source checksum mismatch')
  old_nodes=ast.parse(old.read_text()).body;new_nodes=ast.parse((root/name).read_text()).body
  dumps=[ast.dump(x,include_attributes=False) for x in new_nodes]
  for node in old_nodes:
   if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):continue
   require(ast.dump(node,include_attributes=False) in dumps,'Historical CNN statement changed: '+name)
  result[name]=dict(historical_sha256=digest,current_sha256=sha256(root/name),old_statements_retained=True)
 return dict(passed=True,files=result,reason='ConvNeXt added later; archived CNN statements retained')

def compare_prefix(actual,old):
 require(actual['initial_state_sha256']==old['initial_state_sha256'],'Initialization mismatch')
 require([h['epoch'] for h in actual['history']]==[1,2,3],'Three epochs required')
 require([h['epoch'] for h in old['history'][:3]]==[1,2,3],'Reference prefix invalid')
 def compare(a,b,path):
  if isinstance(b,dict):
   for k,v in b.items():require(k in a,'Missing '+path+k);compare(a[k],v,path+'.'+k)
  elif isinstance(b,float):require(math.isclose(a,b,rel_tol=1e-5,abs_tol=1e-6),'Trajectory mismatch: '+path)
  else:require(a==b,'Trajectory mismatch: '+path)
 for a,b in zip(actual['history'],old['history'][:3]):
  for split in ('train','validation'):compare(a[split],b[split],f"epoch{a['epoch']}.{split}")
 return dict(passed=True,epochs=3,rtol=1e-5,atol=1e-6,scope='all retained slice metrics/order/counts; old per-epoch patient probabilities unavailable')

def read_reference(root,mode,manifest,summary,device,seed=42):
 root=Path(root);t,p=validate_arm(root,mode,30,seed=seed)
 require(t['input_sha256']==dict(split_samples=sha256(manifest),split_summary=sha256(summary)),'Reference input mismatch')
 fixed=dict(model='SimpleCNN',epochs=30,batch_size=16,learning_rate=.001,optimizer='Adam',seed=seed,device=device,
  num_workers=0,drop_last=False,augmentation=False,early_stopping=False,scheduler=None,validation_loss_weighting='none')
 require(all(t['config'].get(k)==v for k,v in fixed.items()),'Reference training config mismatch')
 from dataset import PREPROCESSING
 require(t['config']['preprocessing']==PREPROCESSING,'Reference preprocessing mismatch')
 for name,digest in t['code_sha256'].items():
  require(Path(name).name==name,'Invalid source name')
  require((HISTORICAL[name] if name in HISTORICAL else sha256(Path(__file__).with_name(name)))==digest,'Reference source mismatch: '+name)
 versions=dict(python=platform.python_version(),torch=str(torch.__version__),pillow=PIL.__version__,cuda_build=torch.version.cuda)
 require(t['versions']==versions,'Reference environment mismatch')
 require(t['gpu_name']==(torch.cuda.get_device_name() if device=='cuda' else None),'Reference GPU mismatch')
 names=['training/training_summary.json','training/config.json','training/epochs.jsonl','prediction_full/prediction_summary.json','prediction_full/slice_predictions.csv','prediction_full/patient_predictions.csv']
 frozen={str(root/n):sha256(root/n) for n in names}
 weight=root/'training/best.pt';require(weight.is_file(),'Reference checkpoint missing')
 require(sha256(weight)==t['checkpoint_sha256'],'Reference checkpoint hash mismatch');frozen[str(weight)]=t['checkpoint_sha256']
 return t,frozen

def run(manifest,summary,a_ref,b_ref,out,device='cuda'):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);result=dict(status='running',arms={},comparisons={});frozen={}
 try:
  result['source_bridge']=source_bridge();references={}
  for arm,root,mode in [('A',a_ref,'none'),('B',b_ref,'balanced')]:
   references[arm],hashes=read_reference(root,mode,manifest,summary,device);frozen.update(hashes)
  result['reference_sha256']=frozen
  for arm in ('A','B'):
   s=run_cnn(manifest,summary,out/arm,arm=arm,epochs=3,seed=42,device=device);result['arms'][arm]=s
   for k in ('class_weight_details','batch_size','learning_rate','optimizer','seed','preprocessing','class_weighting'):
    require(s['config'][k]==references[arm]['config'][k],'Config mismatch: '+k)
   require(s['versions']==references[arm]['versions'] and s['gpu_name']==references[arm]['gpu_name'],'Runtime mismatch')
   result['comparisons'][arm]=compare_prefix(s,references[arm])
   require(s['training_read_counts']==dict(train=24480,validation=8160,test=0),'Training counts mismatch')
   require(s['reload_read_counts']==dict(train=0,validation=10880,test=0),'Reload counts mismatch')
  a,b=result['arms']['A'],result['arms']['B'];require(a['initial_state_sha256']==b['initial_state_sha256'],'A/B initial mismatch')
  for x,y in zip(a['history'],b['history']):require(x['train']['order_sha256']==y['train']['order_sha256'],'A/B order mismatch')
  require(all(sha256(Path(n))==h for n,h in frozen.items()),'Historical evidence changed')
  result.update(status='passed',historical_evidence_unchanged=True,total_read_counts=dict(train=48960,validation=38080,test=0))
 except Exception as exc:
  result.update(status='failed',error=f'{type(exc).__name__}: {exc}');(out/'error.log').write_text(traceback.format_exc());raise
 finally:write_json(out/'compatibility_summary.json',result)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('manifest','summary','a-reference','b-reference','output-dir'):p.add_argument('--'+k,type=Path,required=True)
 args=p.parse_args();run(args.manifest,args.summary,args.a_reference,args.b_reference,args.output_dir)
