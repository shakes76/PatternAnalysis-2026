"""Constructed CPU/sample-display checks; no real MRI, weights or test split."""

import argparse
import contextlib
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))

import predict
from dataset import preprocess_pair, collate_padded
from losses import segmentation_cross_entropy
from metrics import SegmentationMetrics
from modules import build_model
from prediction_panels import overlap_statistics, save_prediction_panels


class ConstructedPredictionDisplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.parent = REPO / "outputs/constructed_prediction_display_20261026"
        cls.parent.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="check_", dir=self.parent)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def base_args(self):
        return ["--checkpoint", str(self.root / "constructed.pt"), "--data-root", str(self.root / "data"),
                "--output-dir", str(self.root / "new_output"), "--device", "cpu"]

    def test_gate_precedes_framework_import_or_data_and_cuda_access_in_clean_process(self):
        code = '''
import builtins,sys
sys.path.insert(0,sys.argv[1])
blocked={'torch','numpy','nibabel','dataset','modules','losses','metrics','prediction_panels'}
old=builtins.__import__
def guard(name,*args,**kwargs):
 if name.split('.')[0] in blocked: raise AssertionError('blocked import '+name)
 return old(name,*args,**kwargs)
builtins.__import__=guard
import predict
try: predict.main(['--checkpoint','nonexistent.pt','--split','test'])
except SystemExit as error:
 assert error.code==2
else: raise AssertionError('test unexpectedly accepted')
assert not blocked.intersection(sys.modules)
print('gate_before_framework_and_data_passed')
'''
        result = subprocess.run([sys.executable, "-B", "-c", code, str(PROJECT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("gate_before_framework_and_data_passed", result.stdout)

    def test_execute_gate_cannot_be_bypassed_with_manual_namespace(self):
        args = argparse.Namespace(split="test", allow_test_example=False)
        with self.assertRaises(PermissionError):
            predict.execute_prediction(args)

    def test_explicit_test_option_is_required_and_folder_mapping_is_exact(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            predict.parse_args(self.base_args() + ["--split", "test"])
        args = predict.parse_args(self.base_args() + ["--split", "test", "--allow-test-example"])
        self.assertEqual(args.split, "test")
        self.assertEqual(predict.SPLIT_FOLDERS[args.split], ("keras_slices_test", "keras_slices_seg_test"))
        self.assertFalse(args.output_dir.exists())

    def test_filename_and_existing_output_rejected_before_runtime(self):
        for filename in ("../case_004_week_0_slice_12.nii.gz", "folder\\case_004_week_0_slice_12.nii.gz", "mask.nii.gz"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                predict.parse_args(self.base_args() + ["--filename", filename])
        (self.root / "new_output").mkdir()
        with self.assertRaises(FileExistsError):
            predict.parse_args(self.base_args())

    def test_same_preprocessing_loss_metrics_and_crop_with_no_updates(self):
        image = torch.arange(17 * 19, dtype=torch.float32).reshape(1,17,19)
        target = torch.zeros((17,19), dtype=torch.long)
        target[3:7,4:8] = 5
        sample = preprocess_pair(image, target)
        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.scale = torch.nn.Parameter(torch.tensor(1.0)); self.observed = []
            def forward(self, values):
                self.observed.append((self.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
                logits = torch.zeros((1,6,*values.shape[-2:]), dtype=values.dtype)
                logits[:,5,3:7,4:8] = 3 * self.scale
                return logits
        model = Probe()
        before = {name:value.clone() for name,value in model.state_dict().items()}
        result = predict.infer_preprocessed_sample(model,sample,torch.device('cpu'),6)
        self.assertEqual(model.observed, [(False,False,True)])
        self.assertTrue(all(torch.equal(before[name],value) for name,value in model.state_dict().items()))
        self.assertIsNone(model.scale.grad)
        self.assertEqual(result['prediction'].shape,(17,19))
        self.assertEqual(result['metrics']['valid_pixels'],17*19)
        self.assertEqual(result['metrics']['dice_per_class'][5],1.0)
        batch = collate_padded([sample])
        with torch.inference_mode():
            logits=model(batch['image']); loss=segmentation_cross_entropy(logits,batch['target'],batch['valid_mask'])
        metrics=SegmentationMetrics(6);metrics.update(logits.argmax(1),batch['target'],batch['valid_mask'])
        self.assertEqual(result['metrics'],metrics.compute())
        self.assertEqual(result['cross_entropy'],float(loss.item()))
        self.assertEqual(result['logit_shape'],[1,6,32,32])

    def make_constructed_files(self, split, model_name):
        image_folder,mask_folder = predict.SPLIT_FOLDERS[split]
        for folder in (image_folder,mask_folder):
            (self.root/'data'/folder).mkdir(parents=True)
        filename='case_004_week_0_slice_12.nii.gz'
        image=np.arange(17*19,dtype=np.float32).reshape(17,19)
        target=np.zeros((17,19),dtype=np.int16);target[3:8,5:10]=5
        nib.save(nib.Nifti1Image(image,np.eye(4)),self.root/'data'/image_folder/filename)
        nib.save(nib.Nifti1Image(target,np.eye(4)),self.root/'data'/mask_folder/filename.replace('case_','seg_',1))
        config={'in_channels':1,'num_classes':6,'base_channels':2}
        model=build_model(model_name,**config)
        checkpoint={'model':model_name,'model_config':config,'model_state':model.state_dict(),'epoch':17 if model_name=='improved_unet' else 5,'limited_training_smoke':False}
        path=self.root/'constructed.pt';torch.save(checkpoint,path)
        return path

    def run_constructed(self, split, model_name):
        checkpoint=self.make_constructed_files(split,model_name)
        digest=sha256(checkpoint.read_bytes()).hexdigest()
        argv=self.base_args()+['--split',split]
        if split=='test':argv+=['--allow-test-example']
        args=predict.parse_args(argv)
        with patch('torch.cuda.is_available',side_effect=AssertionError('CPU check queried CUDA')),contextlib.redirect_stdout(io.StringIO()):
            result=predict.execute_prediction(args)
        self.assertEqual(result['test_split_read'],split=='test')
        self.assertEqual(result['model'],model_name)
        self.assertEqual(result['restored_shape'],[17,19])
        self.assertEqual(result['metrics']['valid_pixels'],323)
        self.assertEqual(result['checkpoint_sha256'],digest)
        self.assertEqual(sha256(checkpoint.read_bytes()).hexdigest(),digest)
        self.assertEqual(result['optimizer_steps'],0)
        self.assertEqual(result['prediction_panels']['scores'][f"{model_name}: epoch {result['checkpoint_epoch']}"]['valid_pixels'],323)
        for filename in ['prediction.png','prediction_panels.png','prediction_summary.json']:
            self.assertTrue((args.output_dir/filename).is_file())
        saved=json.loads((args.output_dir/'prediction_summary.json').read_text())
        self.assertEqual(saved,result)

    def test_constructed_test_baseline_load_dispatch_crop_and_panels(self):
        self.run_constructed('test','baseline_unet')

    def test_constructed_test_improved_load_dispatch_crop_and_panels(self):
        self.run_constructed('test','improved_unet')

    def test_constructed_validation_legacy_outputs_and_rule_regression(self):
        self.run_constructed('validation','baseline_unet')

    def test_constructed_training_legacy_outputs_and_rule_regression(self):
        self.run_constructed('training','baseline_unet')

    def test_independent_panel_statistics_roi_and_empty_reference(self):
        image=np.linspace(0,1,40*50,dtype=np.float32).reshape(40,50)
        reference=np.zeros((40,50),dtype=np.int64);reference[20,20]=5;reference[21,20]=5
        first=np.zeros_like(reference);first[20,20]=5;first[20,21]=5
        second=np.zeros_like(reference)
        output=self.root/'panels.png'
        result=save_prediction_panels(output,image,reference,{'Baseline5':first,'Improved17':second},{'split':'validation','filename':'constructed.nii.gz'})
        score=result['scores']['Baseline5']['per_class'][5]
        self.assertEqual((score['tp'],score['fp'],score['fn']),(1,1,1))
        self.assertEqual(score['dice'],0.5)
        self.assertEqual(result['roi_bounds'],[12,30,12,30])
        self.assertEqual((result['rows'],result['columns']),(2,6))
        self.assertEqual(result['scores']['Improved17']['per_class'][5]['dice'],0.0)
        empty=overlap_statistics(np.zeros((3,4),dtype=np.int64),np.zeros((3,4),dtype=np.int64))
        self.assertIsNone(empty['per_class'][5]['dice'])
        fp=np.zeros((3,4),dtype=np.int64);fp[0,0]=5
        empty_fp=overlap_statistics(np.zeros((3,4),dtype=np.int64),fp)
        self.assertEqual(empty_fp['per_class'][5]['dice'],0)
        with self.assertRaises(FileExistsError):
            save_prediction_panels(output,image,reference,{'Baseline5':first},{'split':'validation','filename':'constructed.nii.gz'})

    def test_invalid_original_arrays_and_pixel_labels_rejected(self):
        original=np.zeros((4,5),dtype=np.int64)
        for invalid in (original.astype(np.float32),np.full((4,5),-100,dtype=np.int64),np.full((4,5),6,dtype=np.int64)):
            with self.assertRaises(ValueError):overlap_statistics(original,invalid)
        with self.assertRaises(ValueError):
            save_prediction_panels(self.root/'bad.png',np.full((4,5),np.nan),original,{'x':original},{'split':'validation','filename':'x.nii.gz'})


if __name__=='__main__':
    unittest.main()
