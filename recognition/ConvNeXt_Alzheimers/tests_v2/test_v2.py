"""Artificial-data tests only. No network downloads, patient data or GPU required."""
from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataset import DataError, SliceRecord, SplitBundle
from modules import ConvNeXtTiny
from modules_v2 import RGBTransferConvNeXt, import_torchvision_backbone
from train_convnext_v2 import (Config, WeightedDataset, atomic_json, artifacts,
    make_model, make_optimizer, patient_weights, restore_rng, rng_state, run_lock,
    save_state, selection_key, set_epoch, train_epoch, train_run, load_pretrained)

torch.set_num_threads(1)


def small_model(config):
    return ConvNeXtTiny(depths=(1,1,1,1),dims=(4,8,16,32))


def records(split='train'):
    answer=[]
    # Patient A: two scans of 2 slices. Patient B: one scan of 2 slices.
    for patient,scans,label in [('artificial-A',['1','2'],0),('artificial-B',['3'],1)]:
        for scan in scans:
            for i in range(2):
                cls='AD' if label else 'NC'
                answer.append(SliceRecord(f'AD_NC/train/{cls}/{scan}_{i}.jpeg',patient,scan,i,
                                          'train',cls,2*label,label,split))
    return tuple(answer)


def fixture(base):
    root=base/'data';root.mkdir()
    rec={}
    for split in ('train','validation','test'):
        rec[split]=tuple(replace(r,subject_id=split+r.subject_id,
                                 scan_id=split+r.scan_id,
                                 image_path=f'{split}/{i}.jpeg') for i,r in enumerate(records(split)))
        if split!='test':
            for i,r in enumerate(rec[split]):
                path=root/r.image_path;path.parent.mkdir(exist_ok=True)
                a=np.full((240,256),45+120*r.label,dtype=np.uint8);a[40:200,60:190]+=i
                Image.fromarray(a).save(path)
    return SplitBundle(root,rec,{}, {'train.csv':'synthetic-train','validation.csv':'synthetic-validation','test.csv':'synthetic-test'})


class CoreTests(unittest.TestCase):
    def test_default_recipes(self):
        for r in ('scratch_augmented','scratch_patient','transfer_patient'):Config(r).validate()

    def test_bad_config(self):
        for kw in ({'recipe':'bad'},{'workers':1},{'batch_size':0},{'rotation_degrees':6}):
            with self.assertRaises(DataError):replace(Config('scratch_patient'),**kw).validate()

    def test_equal_patient_weight(self):
        rec=records();w=patient_weights(rec)
        self.assertAlmostEqual(w[:4].sum().item(),w[4:].sum().item())
        self.assertAlmostEqual(w.sum().item(),len(rec))

    def test_equal_scan_within_patient(self):
        w=patient_weights(records());self.assertEqual(w[:2].sum(),w[2:4].sum())

    def test_weights_reject_validation(self):
        with self.assertRaises(DataError):patient_weights(records('validation'))

    def test_selection_primary_accuracy(self):
        a={'patient':{'accuracy':.81,'macro_f1':.7,'log_loss':.8}}
        b={'patient':{'accuracy':.8,'macro_f1':.9,'log_loss':.3}}
        self.assertGreater(selection_key(a),selection_key(b))

    def test_selection_ties(self):
        a={'patient':{'accuracy':.8,'macro_f1':.75,'log_loss':.4}}
        b={'patient':{'accuracy':.8,'macro_f1':.75,'log_loss':.5}}
        self.assertGreater(selection_key(a),selection_key(b))

    def test_rng_roundtrip(self):
        state=rng_state();a=(random.random(),np.random.random(),torch.rand(3))
        restore_rng(state);b=(random.random(),np.random.random(),torch.rand(3))
        self.assertEqual(a[:2],b[:2]);self.assertTrue(torch.equal(a[2],b[2]))

    def test_full_state_safe_load(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'PRIVATE_resume.pt';save_state(p,{'rng':rng_state(),'epoch':0})
            s=torch.load(p,weights_only=True);restore_rng(s['rng']);self.assertEqual(s['epoch'],0)

    def test_previous_state_retained(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'PRIVATE_resume.pt';save_state(p,{'epoch':1});save_state(p,{'epoch':2})
            self.assertEqual(torch.load(p,weights_only=True)['epoch'],2)
            self.assertEqual(torch.load(Path(d)/'PRIVATE_previous.pt',weights_only=True)['epoch'],1)

    def test_file_lock_released(self):
        with tempfile.TemporaryDirectory() as d:
            with run_lock(d): pass
            with run_lock(d): pass

    def test_head_only_then_unfreeze(self):
        c=Config('transfer_patient');m=small_model(c);o=make_optimizer(m,c)
        self.assertTrue(set_epoch(m,o,c,1))
        self.assertTrue(all(p.requires_grad==n.startswith('head.') for n,p in m.named_parameters()))
        self.assertFalse(set_epoch(m,o,c,3));self.assertTrue(all(p.requires_grad for p in m.parameters()))

    def test_decay_excludes_vectors(self):
        c=Config('scratch_patient');m=small_model(c);o=make_optimizer(m,c)
        for g in o.param_groups:
            for p in g['params']:self.assertEqual(g['weight_decay'],c.weight_decay if p.ndim>1 else 0.)

    def test_transfer_lr_differential(self):
        c=Config('transfer_patient');m=small_model(c);o=make_optimizer(m,c);set_epoch(m,o,c,5)
        for g in o.param_groups:
            self.assertAlmostEqual(g['lr'],c.learning_rate if g['is_head'] else c.transfer_backbone_lr)

    def test_scratch_lr_warmup_and_end(self):
        c=Config('scratch_patient');m=small_model(c);o=make_optimizer(m,c)
        set_epoch(m,o,c,1);self.assertAlmostEqual(o.param_groups[0]['lr'],1e-4)
        set_epoch(m,o,c,30);self.assertAlmostEqual(o.param_groups[0]['lr'],1e-6)

    def test_approval_missing_before_download(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('torch.hub.load_state_dict_from_url',side_effect=AssertionError('No network')):
                with self.assertRaises(DataError):load_pretrained(small_model(Config('scratch_patient')),Path(d),None)

    def test_adapter_input(self):
        m=RGBTransferConvNeXt(depths=(1,1,1,1),dims=(4,8,16,32))
        x=torch.zeros(2,1,224,224)
        expected=(torch.full((2,3,224,224),.5)-m.rgb_mean)/m.rgb_std
        self.assertTrue(torch.equal(m.adapt_input(x),expected))

    def test_mapping_incomplete_rejected(self):
        m=RGBTransferConvNeXt(depths=(1,1,1,1),dims=(4,8,16,32))
        with self.assertRaises(ValueError):import_torchvision_backbone(m,{})

    def test_full_tiny_count_unchanged(self):
        m=make_model(Config('scratch_augmented'))
        self.assertEqual(sum(p.numel() for p in m.parameters()),27818594)

    def test_torchvision_backbone_equivalence_no_download(self):
        from torchvision.models import convnext_tiny
        ref=convnext_tiny(weights=None).eval()
        model=RGBTransferConvNeXt().eval();import_torchvision_backbone(model,ref.state_dict())
        ref.classifier[2]=nn.Linear(768,2)
        ref.classifier[2].load_state_dict(model.head.state_dict())
        x=torch.rand(1,1,224,224)*2-1
        with torch.no_grad():a=ref(model.adapt_input(x));b=model(x)
        torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)

    def test_full_transfer_backward(self):
        m=RGBTransferConvNeXt();x=torch.zeros(1,1,224,224)
        m(x).sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.b=fixture(self.base)
        self.c=Config('scratch_patient',epochs=3,min_epochs=3,patience=4,
                      batch_size=2,accumulation_steps=2,device='cpu')

    def run_model(self,path,resume=False,max_epochs=None,c=None):
        return train_run(path,self.b,c or self.c,None,self.base,resume=resume,
                         model_factory=small_model,max_new_epochs=max_epochs)

    def test_augmented_and_patient_weights_differ(self):
        a=WeightedDataset(self.b,replace(self.c,recipe='scratch_augmented'))
        b=WeightedDataset(self.b,self.c)
        self.assertTrue(torch.equal(a.weights,torch.ones(6)))
        self.assertFalse(torch.equal(a.weights,b.weights))

    def test_completed_without_test_images(self):
        folder=self.base/'run';folder.mkdir();s=self.run_model(folder)
        self.assertTrue(s['completed']);self.assertEqual(s['epoch'],3)
        summary=json.loads((folder/'v2_summary.json').read_text())
        self.assertEqual(summary['new_test_images_decoded'],0)
        self.assertTrue((folder/'PRIVATE_best_model.pt').exists())

    def test_resume_matches_uninterrupted_weights(self):
        a=self.base/'whole';a.mkdir();b=self.base/'resume';b.mkdir()
        whole=self.run_model(a);part=self.run_model(b,max_epochs=1)
        self.assertFalse(part['completed']);continued=self.run_model(b,resume=True)
        for key in whole['model_state']:
            self.assertTrue(torch.equal(whole['model_state'][key],continued['model_state'][key]),key)
        self.assertEqual(whole['best']['epoch'],continued['best']['epoch'])
        self.assertEqual(whole['best']['metrics'],continued['best']['metrics'])
        for k in whole['optimizer_state']['state']:
            for name,value in whole['optimizer_state']['state'][k].items():
                other=continued['optimizer_state']['state'][k][name]
                if isinstance(value,torch.Tensor):self.assertTrue(torch.equal(value,other))
                else:self.assertEqual(value,other)

    def test_transfer_resume_across_unfreeze(self):
        from modules_v2 import RGBTransferConvNeXt
        def builder(c):
            return RGBTransferConvNeXt(depths=(1,1,1,1),dims=(4,8,16,32))
        c=replace(self.c,recipe='transfer_patient')
        a=self.base/'transfer_whole';a.mkdir();b=self.base/'transfer_resumed';b.mkdir()
        with patch('train_convnext_v2.load_pretrained',return_value={'used':False,'synthetic_test':True}):
            whole=train_run(a,self.b,c,None,self.base,model_factory=builder)
            train_run(b,self.b,c,None,self.base,model_factory=builder,max_new_epochs=2)
            resumed=train_run(b,self.b,c,None,self.base,model_factory=builder,resume=True)
        for key in whole['model_state']:
            self.assertTrue(torch.equal(whole['model_state'][key],resumed['model_state'][key]),key)

    def test_resume_config_change_refused(self):
        p=self.base/'run';p.mkdir();self.run_model(p,max_epochs=1)
        with self.assertRaises(DataError):self.run_model(p,resume=True,c=replace(self.c,seed=43))

    def test_resume_source_change_refused(self):
        p=self.base/'run';p.mkdir();self.run_model(p,max_epochs=1)
        with patch('train_convnext_v2.source_hashes',return_value={'changed':'hash'}):
            with self.assertRaises(DataError):self.run_model(p,resume=True)

    def test_resume_environment_change_refused(self):
        p=self.base/'run';p.mkdir();self.run_model(p,max_epochs=1)
        with patch('train_convnext_v2.environment',return_value={'changed':'software'}):
            with self.assertRaises(DataError):self.run_model(p,resume=True)

    def test_completed_resume_rebuilds_outputs(self):
        p=self.base/'run';p.mkdir();self.run_model(p)
        (p/'PRIVATE_best_model.pt').unlink();(p/'PRIVATE_validation_predictions.json').unlink()
        with patch('train_convnext_v2.train_epoch',side_effect=AssertionError('Must not retrain')):
            self.run_model(p,resume=True)
        self.assertTrue((p/'PRIVATE_best_model.pt').exists())
        self.assertTrue((p/'PRIVATE_validation_predictions.json').exists())

    def test_corrupt_latest_uses_previous(self):
        p=self.base/'run';p.mkdir();self.run_model(p,max_epochs=2)
        (p/'PRIVATE_resume.pt').write_bytes(b'corrupt')
        s=self.run_model(p,resume=True);self.assertTrue(s['completed'])

    def test_tail_accumulation_updates(self):
        data=WeightedDataset(self.b,self.c)
        loader=DataLoader(data,batch_size=2,shuffle=False)
        m=small_model(self.c);o=make_optimizer(m,self.c)
        r=train_epoch(m,loader,o,torch.device('cpu'),self.c)
        self.assertEqual(r['images'],6);self.assertEqual(r['updates'],2)

    def test_weighted_epoch_gradient_matches_large_batch(self):
        # Use a deterministic tiny linear model so batch-dependent layers are absent.
        c=replace(self.c,rotation_degrees=0.,batch_size=2,accumulation_steps=3,gradient_clip=1000.)
        data=WeightedDataset(self.b,c)
        class Toy(nn.Module):
            def __init__(self):
                super().__init__();self.head=nn.Linear(1,2)
            def forward(self,x):return self.head(x.mean(dim=(2,3)))
        a=Toy();b=Toy();b.load_state_dict(a.state_dict())
        oa=torch.optim.SGD(a.parameters(),lr=.001);ob=torch.optim.SGD(b.parameters(),lr=.001)
        train_epoch(a,DataLoader(data,batch_size=2,shuffle=False),oa,torch.device('cpu'),c)
        batch=next(iter(DataLoader(data,batch_size=6,shuffle=False)))
        y=batch['label'];logits=b(batch['image'])
        loss=(-torch.log_softmax(logits,1).gather(1,y[:,None]).squeeze(1)*batch['loss_weight']).mean()
        loss.backward();ob.step()
        for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y)

if __name__=='__main__':unittest.main()
