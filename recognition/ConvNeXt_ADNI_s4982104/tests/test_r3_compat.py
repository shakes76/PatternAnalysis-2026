import importlib.util
import unittest
from copy import deepcopy
class R3CompatibilityTests(unittest.TestCase):
 def api(self):
  self.assertIsNotNone(importlib.util.find_spec('run_r3_compat'))
  import run_r3_compat
  return run_r3_compat
 def test_prefix_tolerance_and_order_failures(self):
  api=self.api();h=[dict(epoch=i,train=dict(loss=.5,order_sha256='same',samples=8160),validation=dict(loss=.6,order_sha256='v',samples=2720)) for i in (1,2,3)]
  a=dict(initial_state_sha256='s',history=h)
  self.assertTrue(api.compare_prefix(a,a)['passed'])
  for key,value in [('loss',.7),('order_sha256','changed')]:
   bad=deepcopy(a);bad['history'][1]['train'][key]=value
   with self.assertRaises(ValueError):api.compare_prefix(bad,a)
 def test_missing_epoch_and_initialization_fail(self):
  api=self.api()
  with self.assertRaises(ValueError):api.compare_prefix(dict(initial_state_sha256='a',history=[]),dict(initial_state_sha256='b',history=[]))
 def test_source_bridge_checks_full_historical_files(self):
  api=self.api();self.assertTrue(api.source_bridge()['passed'])
