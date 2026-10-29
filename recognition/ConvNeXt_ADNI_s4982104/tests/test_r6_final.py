import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

class R6Tests(unittest.TestCase):
    def api(self):return importlib.import_module('run_r6_final')
    def test_committed_contract_and_missing_inputs_fail_before_predict(self):
        a=self.api()
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out'
            with patch.object(a,'predict_one') as predict:
                with self.assertRaises(FileNotFoundError):a.run(Path(d),out)
                predict.assert_not_called()
            self.assertEqual(json.loads((out/'summary.json').read_text())['status'],'failed')
            self.assertTrue((out/'failure.log').exists())
    def test_refuse_overwrite(self):
        a=self.api()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):a.run(Path(d),Path(d))
    def test_paths_cannot_escape(self):
        a=self.api()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):a.safe_path(Path(d),'../outside')
    def test_partial_failure_saves_previous_prediction(self):
        a=self.api();from tests.test_final_evaluation import fixture
        p,m,r=fixture();p['code_sha256']={}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'out'
            def pred(*args):
                if args[2]['seed']==43:raise RuntimeError('synthetic interruption')
                return r[0],{'image_reads':120}
            with patch.object(a,'preflight',return_value=(p,m,{})),patch.object(a,'predict_one',side_effect=pred),patch.object(a,'environment',return_value={}):
                with self.assertRaisesRegex(RuntimeError,'interruption'):a.run(root,out)
            self.assertTrue((out/'CNN-A_seed42/slice_predictions.csv').exists())
            self.assertFalse((out/'evaluation.json').exists())
            self.assertEqual(json.loads((out/'summary.json').read_text())['status'],'failed')
    def test_synthetic_complete_entrypoint(self):
        a=self.api();from tests.test_final_evaluation import fixture
        p,m,r=fixture();p['code_sha256']={}
        from final_evaluation import digest
        for item in r:item['protocol_sha256']=digest(p)
        by_key={(v['entry']['model'],v['entry']['seed']):v for v in r}
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out'
            def pred(root,protocol,entry):return by_key[entry['model'],entry['seed']],{'image_reads':120}
            with patch.object(a,'preflight',return_value=(p,m,{})),patch.object(a,'predict_one',side_effect=pred),patch.object(a,'environment',return_value={}):
                result=a.run(Path(d),out)
            self.assertEqual(result['status'],'passed');self.assertEqual(len(result['runs']),9)
            self.assertEqual(len(json.loads((out/'evaluation.json').read_text())['metrics']),18)
    def test_predict_cpu_synthetic_pixels(self):
        import torch
        a=self.api()
        class Model(torch.nn.Module):
            def forward(self,x):return torch.stack((x[:,0,0,0],-x[:,0,0,0]),1)
        batch=dict(image=torch.ones(4,1,2,2),label=torch.tensor([0,1,0,1]),file_path=['a','b','c','d'],patient_id=['p']*4,image_id=['i']*4)
        rows=a.infer(Model(),[batch],'cpu')
        self.assertEqual(len(rows),4);self.assertTrue(all(r['split']=='test' for r in rows))
        self.assertAlmostEqual(rows[0]['p_ad'],torch.softmax(torch.tensor([1.,-1.]),0)[1].item())
    def test_nonfinite_logits_fail(self):
        import torch
        a=self.api()
        class Model(torch.nn.Module):
            def forward(self,x):return torch.full((1,2),float('nan'))
        with self.assertRaises(ValueError):a.infer(Model(),[dict(image=torch.ones(1,1,2,2),label=[0])],'cpu')
    def test_no_cli_backend_override(self):
        a=self.api()
        with self.assertRaises(SystemExit):a.parser().parse_args(['--batch-size','8'])
