import importlib.util
import tempfile
import unittest
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
class FormalCNNTests(unittest.TestCase):
 def api(self):
  self.assertIsNotNone(importlib.util.find_spec('run_r3_formal'))
  import run_r3_formal
  return run_r3_formal
 def sample(self):
  return dict(initial_state_sha256='same',history=[dict(epoch=i,train=dict(loss=.5,order_sha256=str(i)),validation=dict(loss=.6,order_sha256='v')) for i in range(1,31)])
 def test_full_thirty_epoch_comparison_includes_final_epoch(self):
  api=self.api();a=self.sample();self.assertTrue(api.compare_full(a,a)['passed'])
  bad=deepcopy(a);bad['history'][-1]['validation']['loss']=.7
  with self.assertRaises(ValueError):api.compare_full(bad,a)
  with self.assertRaises(ValueError):api.compare_full(dict(a,history=a['history'][:3]),a)
 def test_wrapper_fixed_30_seed42_and_no_overwrite(self):
  api=self.api();calls=[];config={k:0 for k in ('class_weight_details','batch_size','learning_rate','optimizer','seed','preprocessing','class_weighting')}
  sample=dict(self.sample(),config=config,versions={},gpu_name='synthetic',training_read_counts=dict(train=244800,validation=81600,test=0),reload_read_counts=dict(train=0,validation=10880,test=0))
  def fake(*args,**kw):calls.append(kw);return deepcopy(sample)
  with tempfile.TemporaryDirectory() as d,patch.object(api,'source_bridge',return_value={'passed':True}),patch.object(api,'read_reference',return_value=(sample,{})),patch.object(api,'run_cnn',side_effect=fake):
   out=Path(d)/'run';r=api.run(None,None,None,None,out,device='cpu')
   self.assertEqual([(c['arm'],c['epochs'],c['seed']) for c in calls],[('A',30,42),('B',30,42)])
   self.assertEqual(r['total_read_counts'],dict(train=489600,validation=184960,test=0))
   with self.assertRaises(FileExistsError):api.run(None,None,None,None,out)

 def test_remaining_seeds_reach_training_and_reference(self):
  api=self.api()
  for seed in (43,44):
   config={k:0 for k in ('class_weight_details','batch_size','learning_rate','optimizer','seed','preprocessing','class_weighting')};config['seed']=seed
   sample=dict(self.sample(),config=config,versions={},gpu_name='synthetic',training_read_counts=dict(train=244800,validation=81600,test=0),reload_read_counts=dict(train=0,validation=10880,test=0))
   with tempfile.TemporaryDirectory() as d,patch.object(api,'source_bridge',return_value={}),patch.object(api,'read_reference',return_value=(sample,{})) as ref,patch.object(api,'run_cnn',return_value=sample) as train:
    result=api.run(None,None,None,None,Path(d)/'run',device='cpu',seed=seed)
    self.assertEqual(result['seed'],seed)
    self.assertEqual([c.kwargs['seed'] for c in train.call_args_list],[seed,seed])
    self.assertEqual([c.kwargs['seed'] for c in ref.call_args_list],[seed,seed])
 def test_invalid_seed_rejected_before_output(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'run'
   with self.assertRaises(ValueError):self.api().run(None,None,None,None,out,seed=45)
   self.assertFalse(out.exists())
