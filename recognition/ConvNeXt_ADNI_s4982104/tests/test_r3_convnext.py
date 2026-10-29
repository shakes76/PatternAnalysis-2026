"""CPU synthetic acceptance; not real MRI/GPU evidence."""
import importlib.util,json,unittest
from copy import deepcopy
from unittest.mock import patch
from contextlib import ExitStack
import torch
import test_n3, test_n5, run_n5, run_r2

class R3ConvNeXtTests(unittest.TestCase):
 setUp=test_n3.N3Tests.setUp
 tearDown=test_n3.N3Tests.tearDown
 def api(self):
  self.assertIsNotNone(importlib.util.find_spec('run_r3_convnext'))
  import run_r3_convnext
  return run_r3_convnext
 def patches(self,api):
  stack=ExitStack();path=self.p/'weights.pt'
  if not path.exists():
   torch.manual_seed(900);torch.save(test_n5.HeadTiny().state_dict(),path)
  for mod in (run_n5,api):
   stack.enter_context(patch.object(mod,'build_model',side_effect=lambda name,num_classes=2:test_n5.HeadTiny(num_classes)))
  stack.enter_context(patch.object(run_n5,'verify_weights',return_value='synthetic'))
  return stack
 def execute(self,api,name,seed=42,**kw):
  return api.run_formal(self.manifest,self.summary,self.p/'weights.pt',None,self.p/name,device='cpu',seed=seed,expected_lengths=None,expected_slices=3,**kw)
 def test_thirty_epochs_tail_dual_best_reload_and_rng(self):
  api=self.api()
  with self.patches(api):s=self.execute(api,'run')
  self.assertEqual(s['status'],'passed');self.assertEqual(len(s['history']),30)
  self.assertEqual(s['training_read_counts'],dict(train=180,validation=180,test=0))
  self.assertEqual(s['reload_read_counts'],dict(train=0,validation=24,test=0))
  self.assertEqual(s['optimizer_actual']['lr'],3e-5);self.assertEqual(s['optimizer_actual']['weight_decay'],.01)
  self.assertTrue(all(h['train']['samples']==6 and h['validation']['samples']==6 and h['diagnostic_rng_unchanged'] and h['diagnostic_additional_reads']==0 for h in s['history']))
  for rule in ('patient_nll','slice_ce'):
   best=min(s['selection']['history'],key=lambda h:(h[rule],h['epoch']))
   self.assertEqual(s['reload'][rule]['epoch'],best['epoch']);self.assertTrue(s['reload'][rule]['repeat_equal'])
  self.assertEqual(len(list((self.p/'run').rglob('*.pt'))),2)
  with self.assertRaises(FileExistsError):self.execute(api,'run')
 def test_seed_real_head_order_repeat_and_r2_prefix(self):
  api=self.api()
  with self.patches(api):
   a=self.execute(api,'a',43);b=self.execute(api,'b',43);c=self.execute(api,'c',42)
   with patch.object(run_r2,'initialize_model',side_effect=lambda *a:run_n5.initialize_model(*a,seed=42)),patch.object(run_r2,'build_model',side_effect=lambda name,num_classes=2:test_n5.HeadTiny(num_classes)):
    old=run_r2.run_arm('F',self.manifest,self.summary,self.p/'weights.pt',None,self.p/'r2','cpu',None,3)
  self.assertEqual(a['initial_state_sha256'],b['initial_state_sha256']);self.assertNotEqual(a['initial_state_sha256'],c['initial_state_sha256'])
  self.assertEqual([h['train'] for h in a['history']],[h['train'] for h in b['history']])
  self.assertNotEqual(a['history'][0]['train']['order_sha256'],c['history'][0]['train']['order_sha256'])
  for i in range(5):self.assertTrue(api.check_epoch(c['history'][i],old['history'][i],patient=True)['passed'])
 def test_trajectory_rejects_counts_order_nonfinite_and_patient_drift(self):
  api=self.api();row=dict(epoch=1,train=dict(loss=.5,samples=6,order_sha256='same'),validation=dict(loss=.4,samples=6,order_sha256='same'),patient_metrics=dict(nll=.4))
  for section,key,value in [('train','samples',5),('train','order_sha256','bad'),('validation','loss',float('nan')),('patient_metrics','nll',.9)]:
   bad=deepcopy(row);bad[section][key]=value
   with self.assertRaises(ValueError):api.check_epoch(bad,row,patient=True)
 def test_failure_retains_epoch_and_no_test_reads(self):
  api=self.api();real=api.check_epoch
  reference={'history':[{}]*30}
  with self.patches(api),patch.object(api,'read_reference',return_value=(reference,{})),patch.object(api,'check_start'),patch.object(api,'check_epoch',side_effect=ValueError('trajectory mismatch')):
   with self.assertRaisesRegex(ValueError,'trajectory'):self.execute(api,'failed',n5_reference=self.p/'unused')
  s=json.loads((self.p/'failed/training_summary.json').read_text());self.assertEqual(s['status'],'failed');self.assertEqual(len(s['history']),1);self.assertEqual(s['actual_training_read_counts']['test'],0)
 def test_protocol_seed_and_reference_required_for_real_cohort(self):
  api=self.api()
  for seed in (42,43,44):
   c=api.protocol_config(seed);self.assertEqual((c['seed'],c['split_seed'],c['epochs']),(seed,42,30))
  with self.assertRaises(ValueError):api.protocol_config(45)
  with self.assertRaises(ValueError):api.run_formal(None,None,None,None,self.p/'missing',device='cpu')
 def test_n5_thirty_epoch_reference_and_reference_tamper(self):
  api=self.api();fixture=test_n5.N5Tests();fixture.__dict__.update(self.__dict__)
  ref=fixture.reference(run_n5)
  with fixture.patches(run_n5):
   fixture.run_formal(run_n5,'old_n5',ref,44)
  folder=self.p/'old_n5/training'
  with self.patches(api),patch.object(run_n5,'verify_weights',return_value='synthetic-verified-weight'):s=self.execute(api,'new_n5',44,n5_reference=folder)
  self.assertEqual(len(s['reference_checks']['N5']),30)
  original=(folder/'config.json').read_text();changed=json.loads(original);changed['seed']=43;(folder/'config.json').write_text(json.dumps(changed))
  with self.assertRaises(ValueError):api.read_reference(folder,'N5',44,self.manifest,self.summary)
  (folder/'config.json').write_text(original)
  t=json.loads((folder/'training_summary.json').read_text());t['code_sha256']['run_n5.py']='wrong';(folder/'training_summary.json').write_text(json.dumps(t))
  with self.assertRaises(ValueError):api.read_reference(folder,'N5',44,self.manifest,self.summary)
 def test_dual_reload_rejects_changed_config_and_reference_start(self):
  api=self.api()
  with self.patches(api):s=self.execute(api,'reload')
  from dual_checkpoint import load_selected
  for key,value in [('seed',43),('learning_rate',1e-4),('preprocessing',{})]:
   config=dict(s['config']);config[key]=value
   with self.assertRaises(ValueError):load_selected(self.p/'reload/selection','patient_nll',lambda:test_n5.HeadTiny(2),config,s['provenance'])
  for key in ('initial_state_sha256','versions','gpu_name','optimizer_actual'):
   bad=deepcopy(s);bad[key]='wrong'
   with self.assertRaises(ValueError):api.check_start(bad,s)
 def test_reference_rejects_frozen_backbone_or_changed_split_seed(self):
  api=self.api();folder=self.p/'reference';folder.mkdir()
  cfg=run_n5.protocol_config(42)
  from check_dataloader import sha256
  from pathlib import Path
  original=dict(status='passed',config=cfg,history=[dict(epoch=i) for i in range(1,31)],input_sha256=dict(split_samples=sha256(self.manifest),split_summary=sha256(self.summary)),code_sha256={'run_n5.py':sha256(Path(run_n5.__file__))})
  (folder/'epochs.jsonl').write_text(''.join(json.dumps(h)+'\n' for h in original['history']))
  for key,value in [('full_finetuning',False),('split_seed',43)]:
   t=deepcopy(original);t['config'][key]=value
   (folder/'config.json').write_text(json.dumps(t['config']));(folder/'training_summary.json').write_text(json.dumps(t))
   with self.subTest(key=key),self.assertRaises(ValueError):api.read_reference(folder,'N5',42,self.manifest,self.summary)
 def test_r2_start_requires_same_full_trainable_set(self):
  api=self.api();s=dict(initial_state_sha256='same',versions={},gpu_name=None,optimizer_actual={'lr':3e-5},config={},trainable_parameters=['head','backbone'],trainable_count=10)
  for key,value in [('trainable_parameters',['head']),('trainable_count',2)]:
   old=deepcopy(s);old[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):api.check_start(s,old)
