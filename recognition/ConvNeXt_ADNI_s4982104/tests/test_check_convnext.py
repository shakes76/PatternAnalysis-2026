"""Offline N1 tests use random official states, never claim pretrained validation."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modules


class ConvNeXtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_model_implements_official_state_and_eval_function(self):
        self.assertTrue(hasattr(modules, 'ConvNeXtTiny'), 'Missing self-implemented ConvNeXtTiny')
        torch.manual_seed(17)
        reference = convnext_tiny(weights=None).eval()
        model = modules.ConvNeXtTiny(num_classes=1000).eval()
        model.load_state_dict(reference.state_dict(), strict=True)
        with torch.no_grad():
            for n in (1, 2):
                x = torch.randn(n, 3, 224, 224)
                torch.testing.assert_close(model(x), reference(x), rtol=1e-5, atol=1e-6)

    def test_stochastic_depth_train_eval_and_reproducibility(self):
        self.assertTrue(hasattr(modules, 'DropPath'), 'Missing DropPath')
        from torchvision.ops import stochastic_depth
        layer = modules.DropPath(0.4)
        x = torch.ones(16, 3, 2, 2)
        torch.manual_seed(3)
        got = layer(x)
        torch.manual_seed(3)
        expected = stochastic_depth(x, 0.4, 'row', training=True)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        layer.eval()
        self.assertTrue(torch.equal(layer(x), x))

    def test_model_factory_keeps_cnn_and_rejects_unknown(self):
        self.assertTrue(hasattr(modules, 'build_model'), 'Missing model selector')
        self.assertIsInstance(modules.build_model('simple_cnn'), modules.SimpleCNN)
        with self.assertRaises(ValueError): modules.build_model('wrong')

    def test_preprocessing_matches_official_and_preserves_metadata(self):
        import dataset
        self.assertTrue(hasattr(dataset, 'convnext_transform'), 'Missing ConvNeXt preprocessing')
        pixels = torch.arange(240*256).remainder(256).to(torch.uint8).numpy()
        gray = Image.fromarray(pixels, 'L')
        result = dataset.convnext_transform(gray)
        expected = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()(gray.convert('RGB'))
        self.assertEqual(tuple(result.shape), (3,224,224))
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(result, dataset.convnext_transform(gray)))


class ConvNeXtSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.weights_tmp = tempfile.TemporaryDirectory()
        cls.weights = Path(cls.weights_tmp.name)/'weights.pth'
        torch.manual_seed(7)
        torch.save(convnext_tiny(weights=None).state_dict(), cls.weights)
        cls.digest = hashlib.sha256(cls.weights.read_bytes()).hexdigest()
        cls.prepared = Path(cls.weights_tmp.name)/'summary.json'
        cls.prepared.write_text(json.dumps(dict(status='passed', sha256=cls.digest,
            size_bytes=cls.weights.stat().st_size, strict_cpu_load=True,
            original_classifier_outputs=1000,
            weights='ConvNeXt_Tiny_Weights.IMAGENET1K_V1',
            url='https://download.pytorch.org/models/convnext_tiny-983f1562.pth')))

    @classmethod
    def tearDownClass(cls): cls.weights_tmp.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rows=[]
        for s, split in enumerate(('train','validation','test')):
            for label in (0,1):
                for patient in (0,1):
                    number=100*s+10*label+patient
                    for slice_id in (0,1):
                        path=self.root/f'{split}_{number}_{slice_id}.png'
                        if split != 'test':
                            Image.new('L',(256,240),30+label*130+patient*5+slice_id).save(path)
                        rows.append(dict(file_path=str(path),original_split='train',image_id=str(number+1),
                            patient_id=f'001_S_{number:04d}',scan_date='',label=('NC','AD')[label],split=split))
        self.manifest=self.root/'split_samples.csv'
        with self.manifest.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        self.summary=self.root/'split_summary.json'
        self.summary.write_text(json.dumps(dict(status='passed',output_sha256={
            'split_samples':hashlib.sha256(self.manifest.read_bytes()).hexdigest()})))

    def tearDown(self): self.tmp.cleanup()

    def test_end_to_end_updates_reload_previews_and_no_test_reads(self):
        import check_convnext as check
        out=self.root/'out';out.mkdir()
        r=check.run_check(self.manifest,self.summary,self.weights,self.prepared,out,
                          device='cpu',batch_size=2,expected_weight_sha256=self.digest)
        self.assertEqual(r['status'],'passed')
        self.assertFalse(r['official_pretrained_digest'])
        self.assertEqual(len(r['steps']),2)
        for step in r['steps']:
            self.assertTrue(step['head_changed']);self.assertTrue(step['backbone_changed'])
            self.assertGreater(step['gradient_norm'],0)
        self.assertTrue(r['validation_state_unchanged'])
        self.assertTrue(r['checkpoint_reload_equal'])
        self.assertEqual(r['read_counts']['test'],0)
        self.assertEqual(r['read_counts']['train'],4)
        self.assertEqual(r['read_counts']['validation'],2)
        self.assertEqual(r['class_weight_details']['weights'],[1.0,1.0])
        self.assertTrue(r['preprocessing_matches_official'])
        self.assertTrue(r['official_model_equivalence']['passed'])
        previews=json.loads((out/'crop_previews.json').read_text())
        self.assertEqual({x['label'] for x in previews}, {'NC','AD'})
        self.assertTrue(all(x['split']=='train' for x in previews))
        self.assertTrue(all((out/x['preview']).exists() for x in previews))
        config=json.loads((out/'config.json').read_text())
        config['preprocessing']['crop_size']=128
        with self.assertRaisesRegex(ValueError,'configuration'):
            check.load_checkpoint(out/'smoke_checkpoint.pt',config,'cpu')

    def test_weight_missing_hash_and_metadata_rejected(self):
        import check_convnext as check
        with self.assertRaises(FileNotFoundError):
            check.verify_weights(self.root/'missing',self.prepared)
        with self.assertRaisesRegex(ValueError,'SHA256'):
            check.verify_weights(self.weights,self.prepared)
        invalid=self.root/'bad.json'
        data=json.loads(self.prepared.read_text());data['strict_cpu_load']=False
        invalid.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            check.verify_weights(self.weights,invalid,self.digest)

    def test_nonfinite_and_bad_labels_fail_before_update(self):
        import check_convnext as check
        model=modules.ConvNeXtTiny(); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        for x,y in [(torch.full((1,3,224,224),float('nan')),torch.tensor([0])),
                    (torch.zeros(1,3,224,224),torch.tensor([2]))]:
            with self.assertRaises(ValueError):
                check.train_step(model,optimizer,x,y,torch.ones(2))

    def test_cli_missing_input_records_failure(self):
        out=self.root/'cli'
        r=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'check_convnext.py'),
            '--manifest',str(self.root/'missing'),'--summary',str(self.summary),
            '--weights',str(self.weights),'--weights-summary',str(self.prepared),
            '--output-dir',str(out),'--device','cpu'],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0)
        evidence=json.loads((out/'convnext_check_summary.json').read_text())
        self.assertEqual(evidence['status'],'failed')

if __name__=='__main__': unittest.main()
