"""R3c seeds42/43/44: A/B 30 epochs and full V1 trajectory audit.

Reuses the accepted CNN entry and R3b reference preflight. One paired seed per invocation.
"""
import argparse
import math
from pathlib import Path
import traceback
from check_dataloader import require,sha256
from train import write_json
from run_r3_cnn import run_cnn
from run_r3_compat import source_bridge,read_reference

def compare_full(actual,old):
 require(actual['initial_state_sha256']==old['initial_state_sha256'],'Initialization mismatch')
 require([h['epoch'] for h in actual['history']]==list(range(1,31)),'Thirty epochs required')
 require([h['epoch'] for h in old['history'][:30]]==list(range(1,31)),'Reference prefix invalid')
 def compare(a,b,path):
  if isinstance(b,dict):
   for k,v in b.items():require(k in a,'Missing '+path+k);compare(a[k],v,path+'.'+k)
  elif isinstance(b,float):require(math.isclose(a,b,rel_tol=1e-5,abs_tol=1e-6),'Trajectory mismatch: '+path)
  else:require(a==b,'Trajectory mismatch: '+path)
 for a,b in zip(actual['history'],old['history'][:30]):
  for split in ('train','validation'):compare(a[split],b[split],f"epoch{a['epoch']}.{split}")
 return dict(passed=True,epochs=30,rtol=1e-5,atol=1e-6,scope='all retained slice metrics/order/counts; old per-epoch patient probabilities unavailable')

def run(manifest,summary,a_ref,b_ref,out,device='cuda',seed=42):
 require(seed in (42,43,44),'Unsupported training seed')
 out=Path(out);out.mkdir(parents=True,exist_ok=False);result=dict(status='running',seed=seed,arms={},comparisons={});frozen={}
 try:
  result['source_bridge']=source_bridge();references={}
  for arm,root,mode in [('A',a_ref,'none'),('B',b_ref,'balanced')]:
   references[arm],hashes=read_reference(root,mode,manifest,summary,device,seed=seed);frozen.update(hashes)
  result['reference_sha256']=frozen
  for arm in ('A','B'):
   s=run_cnn(manifest,summary,out/arm,arm=arm,epochs=30,seed=seed,device=device);result['arms'][arm]=s
   for k in ('class_weight_details','batch_size','learning_rate','optimizer','seed','preprocessing','class_weighting'):
    require(s['config'][k]==references[arm]['config'][k],'Config mismatch: '+k)
   require(s['versions']==references[arm]['versions'] and s['gpu_name']==references[arm]['gpu_name'],'Runtime mismatch')
   result['comparisons'][arm]=compare_full(s,references[arm])
   require(s['training_read_counts']==dict(train=244800,validation=81600,test=0),'Training counts mismatch')
   require(s['reload_read_counts']==dict(train=0,validation=10880,test=0),'Reload counts mismatch')
  a,b=result['arms']['A'],result['arms']['B'];require(a['initial_state_sha256']==b['initial_state_sha256'],'A/B initial mismatch')
  for x,y in zip(a['history'],b['history']):require(x['train']['order_sha256']==y['train']['order_sha256'],'A/B order mismatch')
  require(all(sha256(Path(n))==h for n,h in frozen.items()),'Historical evidence changed')
  result.update(status='passed',historical_evidence_unchanged=True,total_read_counts=dict(train=489600,validation=184960,test=0))
 except Exception as exc:
  result.update(status='failed',error=f'{type(exc).__name__}: {exc}');(out/'error.log').write_text(traceback.format_exc());raise
 finally:write_json(out/'formal_summary.json',result)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('manifest','summary','a-reference','b-reference','output-dir'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--seed',type=int,choices=(42,43,44),default=42)
 args=p.parse_args();run(args.manifest,args.summary,args.a_reference,args.b_reference,args.output_dir,seed=args.seed)
