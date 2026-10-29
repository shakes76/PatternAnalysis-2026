import csv
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import test_check_dataloader
from train import class_weight_config, run_epoch, run_training
from predict import run_prediction

class WeightingTests(unittest.TestCase):
    def records(self):
        return [dict(patient_id=str(i),label='NC' if i<3 else 'AD',image_id=str(i),file_path=f'{i}_{j}')
                for i in range(5) for j in range(2)]

    def test_weights_training_patient_counts_and_order(self):
        c=class_weight_config(self.records(),'balanced')
        self.assertEqual(c['patient_counts'],{'NC':3,'AD':2})
        self.assertEqual(c['weights'],[5/6,5/4])
        self.assertIsNone(class_weight_config(self.records(),'none')['weights'])
        with self.assertRaises(ValueError):class_weight_config(self.records(),'unknown')
        with self.assertRaises(ValueError):class_weight_config(self.records()[:6],'balanced')
        with self.assertRaises(ValueError):class_weight_config(self.records()[:-1],'balanced')

    def test_weighted_gradient_and_epoch_denominator(self):
        x=torch.tensor([[2.,0.],[0.,1.],[1.,0.]])
        y=torch.tensor([0,1,1]);w=torch.tensor([0.5,2.])
        rows=[dict(image=a,label=b,file_path=str(i)) for i,(a,b) in enumerate(zip(x,y))]
        model=torch.nn.Linear(2,2,bias=False)
        with torch.no_grad():model.weight.copy_(torch.eye(2))
        expected=torch.nn.functional.cross_entropy(model(x),y,weight=w)
        plain=torch.nn.functional.cross_entropy(model(x),y).item()
        result=run_epoch(model,DataLoader(rows,batch_size=2),'cpu',torch.optim.SGD(model.parameters(),lr=0),w)
        self.assertAlmostEqual(result['loss'],expected.item(),places=6)
        self.assertAlmostEqual(result['unweighted_loss'],plain,places=6)
        self.assertAlmostEqual(result['loss_denominator'],4.5)
        # Full-batch gradient must equal PyTorch's weighted objective.
        model.zero_grad();expected=torch.nn.functional.cross_entropy(model(x),y,weight=w);expected.backward()
        gradient=model.weight.grad.clone()
        run_epoch(model,DataLoader(rows,batch_size=3),'cpu',torch.optim.SGD(model.parameters(),lr=0),w)
        self.assertTrue(torch.allclose(model.weight.grad,gradient))
        with self.assertRaisesRegex(ValueError,'validation'):
            run_epoch(model,DataLoader(rows,batch_size=2),'cpu',class_weights=w)

    def setUp(self):
        self.f=test_check_dataloader.LoaderTests();self.f.setUp();self.addCleanup(self.f.tearDown)
        with self.f.manifest.open() as handle: rows=list(csv.DictReader(handle))
        for i,r in enumerate(rows[:5]):
            r.update(patient_id=f'002_S_{i:04d}',image_id=str(200+i),label='NC' if i<3 else 'AD')
        with self.f.manifest.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        s=json.loads(self.f.summary.read_text());s['output_sha256']['split_samples']=hashlib.sha256(self.f.manifest.read_bytes()).hexdigest();self.f.summary.write_text(json.dumps(s))

    def test_paired_initialization_order_and_unweighted_validation(self):
        results=[]
        for mode in ('none','balanced'):
            r=run_training(self.f.manifest,self.f.summary,self.f.root/mode,device='cpu',epochs=1,
                           batch_size=4,class_weighting=mode)
            results.append(r)
            self.assertEqual(r['history'][0]['validation']['loss'],r['history'][0]['validation']['unweighted_loss'])
            self.assertEqual(r['read_counts']['test'],0)
            self.assertFalse((self.f.root/mode/'first3').exists())
        self.assertEqual(results[0]['initial_state_sha256'],results[1]['initial_state_sha256'])
        self.assertEqual(results[0]['history'][0]['train']['order_sha256'],results[1]['history'][0]['train']['order_sha256'])
        self.assertNotEqual(results[0]['checkpoint_sha256'],results[1]['checkpoint_sha256'])

    def test_first3_best_snapshot_survives_later_improvement_and_predicts(self):
        from train import run_epoch as real
        losses=iter([0.7,0.3,0.5,0.1])
        def controlled(model,loader,device,optimizer=None,**kw):
            r=real(model,loader,device,optimizer,**kw)
            if optimizer is None:r['loss']=next(losses)
            return r
        out=self.f.root/'four'
        with patch('train.run_epoch',side_effect=controlled):
            r=run_training(self.f.manifest,self.f.summary,out,device='cpu',epochs=4,batch_size=4)
        snap=json.loads((out/'first3/training_summary.json').read_text())
        self.assertEqual(snap['best_epoch'],2);self.assertEqual(len(snap['history']),3)
        self.assertEqual(r['best_epoch'],4)
        self.assertEqual(snap['read_counts'],dict(train=15,validation=15,test=0))
        self.assertNotEqual(snap['checkpoint_sha256'],r['checkpoint_sha256'])
        pred=run_prediction(self.f.manifest,self.f.summary,out/'first3/best.pt',out/'first3/training_summary.json',
            self.f.root/'pred',device='cpu',expected_slices=5,expected_patients=1,expected_lengths=None)
        self.assertEqual(pred['status'],'passed')

    def test_none_matches_original_loss_and_parameter_updates(self):
        import copy
        from train import CountedDataset
        from check_dataloader import make_loader
        from modules import SimpleCNN
        torch.manual_seed(42);new=SimpleCNN();old=copy.deepcopy(new)
        ds=CountedDataset(self.f.manifest,'train',self.f.summary)
        opt_old=torch.optim.Adam(old.parameters(),lr=0.001)
        total=0.0
        for batch in make_loader(ds,4,42):
            old.train();opt_old.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(old(batch['image']),batch['label'])
            loss.backward();opt_old.step();total+=loss.item()*len(batch['label'])
        actual=run_epoch(new,make_loader(ds,4,42),'cpu',torch.optim.Adam(new.parameters(),lr=0.001))
        self.assertEqual(actual['loss'],total/len(ds))
        self.assertTrue(all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items()))

    def test_pair_runner_and_tampered_comparison(self):
        from run_e2 import run_pair, compare_pair
        out=self.f.root/'pair'
        r=run_pair(self.f.manifest,self.f.summary,out,device='cpu',expected_lengths=None,
                   expected_slices=5,expected_patients=1)
        self.assertEqual(r['status'],'passed')
        self.assertTrue(r['comparison']['initialization_equal'])
        path=out/'B/training/training_summary.json'; b=json.loads(path.read_text());b['initial_state_sha256']='bad';path.write_text(json.dumps(b))
        with self.assertRaises(ValueError):compare_pair(out)
