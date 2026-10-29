import sys
from pathlib import Path
import unittest
import tempfile
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules import SimpleCNN
from check_cnn import train_step, check_checkpoint

class CNNTests(unittest.TestCase):
    def test_logits_batch_one_and_four(self):
        model=SimpleCNN()
        for n in [1,4]:
            out=model(torch.rand(n,1,240,256))
            self.assertEqual(tuple(out.shape),(n,2))
            self.assertTrue(torch.isfinite(out).all())

    def test_loss_gradients_and_parameters_update(self):
        torch.manual_seed(42)
        model=SimpleCNN(); opt=torch.optim.Adam(model.parameters(),lr=0.001)
        result=train_step(model,opt,torch.rand(4,1,240,256),torch.tensor([0,1,0,1]))
        self.assertTrue(result['parameters_changed'])
        self.assertGreater(result['gradient_norm'],0)
        self.assertGreater(result['loss'],0)

    def test_nonfinite_input_rejected(self):
        model=SimpleCNN(); opt=torch.optim.Adam(model.parameters())
        with self.assertRaises(ValueError):
            train_step(model,opt,torch.full((1,1,240,256),float('nan')),torch.tensor([0]))

    def test_invalid_label_rejected(self):
        model=SimpleCNN(); opt=torch.optim.Adam(model.parameters())
        with self.assertRaises(ValueError):
            train_step(model,opt,torch.rand(1,1,240,256),torch.tensor([2]))

    def test_checkpoint_predictions_match(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(check_checkpoint(SimpleCNN(),torch.rand(2,1,240,256),Path(directory)/'smoke.pt'))

class CNNIntegrationTests(unittest.TestCase):
    def test_real_loader_to_checkpoint_without_test_images(self):
        from test_check_dataloader import LoaderTests
        from check_cnn import run_smoke
        fixture=LoaderTests()
        fixture.setUp()
        try:
            result=run_smoke(fixture.manifest,fixture.summary,fixture.root,steps=2)
            self.assertEqual(result['status'],'passed')
            self.assertEqual(result['read_counts'],dict(train=5,validation=4,test=0,unique_images=9))
            self.assertEqual([s['batch_size'] for s in result['steps']],[4,1])
            self.assertTrue(result['checkpoint_reload_equal'])
            self.assertIsNone(result['peak_cuda_memory'])
        finally: fixture.tearDown()

    def test_cli_missing_input_writes_failed_summary(self):
        import json
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'check_cnn.py'),
                '--manifest',str(root/'missing.csv'),'--summary',str(root/'missing.json'),
                '--output-dir',str(root/'out'),'--device','cpu'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            evidence=json.loads((root/'out/cnn_check_summary.json').read_text())
            self.assertEqual(evidence['status'],'failed')
