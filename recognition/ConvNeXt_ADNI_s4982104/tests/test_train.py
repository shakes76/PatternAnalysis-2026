import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import run_training, run_epoch, classification_metrics
from modules import SimpleCNN
from check_dataloader import make_loader
import test_check_dataloader

class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_check_dataloader.LoaderTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def run_pilot(self, name='pilot', **kwargs):
        return run_training(self.fixture.manifest,self.fixture.summary,self.fixture.root/name,
                            device='cpu',epochs=2,batch_size=4,**kwargs)

    def test_full_epochs_reload_and_no_test_images(self):
        result=self.run_pilot()
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['read_counts'],dict(train=10,validation=10,test=0))
        self.assertEqual([e['train']['samples'] for e in result['history']],[5,5])
        self.assertTrue(all(e['training_parameters_changed'] for e in result['history']))
        self.assertTrue(all(e['validation_parameters_unchanged'] for e in result['history']))
        self.assertTrue(result['checkpoint_reload_equal'])
        losses=[e['validation']['loss'] for e in result['history']]
        self.assertEqual(result['best_epoch'],losses.index(min(losses))+1)
        self.assertTrue((self.fixture.root/'pilot/epochs.jsonl').exists())
        self.assertTrue((self.fixture.root/'pilot/config.json').exists())

    def test_seed_reproduces_losses(self):
        a=self.run_pilot('a'); b=self.run_pilot('b')
        self.assertEqual([e['train']['loss'] for e in a['history']],
                         [e['train']['loss'] for e in b['history']])

    def test_validation_no_grad_and_weighted_loss_with_tail(self):
        from train import CountedDataset
        ds=CountedDataset(self.fixture.manifest,'validation',self.fixture.summary)
        model=SimpleCNN(); before={k:v.clone() for k,v in model.state_dict().items()}
        flags=[]
        handle=model.register_forward_hook(lambda m,i,o: flags.append(torch.is_grad_enabled()))
        result=run_epoch(model,make_loader(ds,4,42),'cpu')
        handle.remove()
        self.assertEqual(flags,[False,False])
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        with torch.no_grad():
            batch=next(iter(make_loader(ds,5,42)))
            expected=torch.nn.functional.cross_entropy(model(batch['image']),batch['label']).item()
        self.assertAlmostEqual(result['loss'],expected,places=6)

    def test_known_metrics_tied_auc_and_absent_class(self):
        m=classification_metrics([0,0,1,1],[0.1,0.5,0.5,0.9])
        self.assertAlmostEqual(m['auroc'],0.875)
        self.assertEqual(m['confusion_matrix'],[[1,1],[0,2]])
        self.assertAlmostEqual(m['macro_f1'],(2/3+0.8)/2)
        self.assertIsNone(classification_metrics([0,0],[0.1,0.2])['auroc'])

    def test_failure_summary_and_no_overwrite(self):
        out=self.fixture.root/'failed'
        with self.assertRaises(ValueError):
            run_training(self.fixture.manifest,self.fixture.summary,out,device='cpu',epochs=0)
        self.assertEqual(json.loads((out/'training_summary.json').read_text())['status'],'failed')
        with self.assertRaises(FileExistsError):
            run_training(self.fixture.manifest,self.fixture.summary,out,device='cpu')

        bad_lr=self.fixture.root/'bad_lr'
        with self.assertRaises(ValueError):
            run_training(self.fixture.manifest,self.fixture.summary,bad_lr,device='cpu',learning_rate=float('nan'))
        self.assertEqual(json.loads((bad_lr/'training_summary.json').read_text())['status'],'failed')

    def test_corrupt_image_saves_failure(self):
        (self.fixture.root/'train_0.png').write_bytes(b'bad')
        with self.assertRaises(RuntimeError): self.run_pilot()
        result=json.loads((self.fixture.root/'pilot/training_summary.json').read_text())
        self.assertEqual(result['status'],'failed')

    def test_best_checkpoint_not_last_or_tie(self):
        from train import run_epoch as real_epoch
        losses=iter([0.2,0.2])
        def controlled(model,loader,device,optimizer=None):
            r=real_epoch(model,loader,device,optimizer)
            if optimizer is None: r['loss']=next(losses)
            return r
        with patch('train.run_epoch',side_effect=controlled): r=self.run_pilot()
        self.assertEqual(r['best_epoch'],1)
        ck=torch.load(self.fixture.root/'pilot/best.pt',weights_only=True)
        self.assertEqual(ck['epoch'],1)

    def test_cli_rejects_non_frozen_cohort(self):
        out=self.fixture.root/'cli'
        r=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'train.py'),
            '--manifest',str(self.fixture.manifest),'--summary',str(self.fixture.summary),
            '--output-dir',str(out),'--device','cpu'],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0)
        self.assertIn('cohort',json.loads((out/'training_summary.json').read_text())['error'])
