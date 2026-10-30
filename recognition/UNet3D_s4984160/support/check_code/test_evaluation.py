"""Small synthetic correctness checks for final evaluation; no real images."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import nibabel as nib
import numpy as np
import torch

import evaluate as evaluator
from support.baseline_model import make_model
from support.evaluation_utils import boundary_regions, class_metrics, geometry, hd95, load_model, summarize
from predict import planes_for, predict_case
from support.plot_training_curves import read_history


class MetricTests(unittest.TestCase):
    def test_direct_checkpoint_inference_uses_saved_config(self):
        torch.manual_seed(42)
        model=make_model(base_channels=2,normalization='instance')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);checkpoint=root/'weights.pt'
            torch.save(dict(model_state_dict=model.state_dict(),epoch=11,
                            model_config=dict(base_channels=2,normalization='instance'),
                            training_config=dict(target_shape=[16,16,16])),checkpoint)
            rng=np.random.default_rng(42)
            image=rng.normal(size=(32,32,32)).astype(np.float32)
            target=np.zeros((32,32,32),np.uint8);target[8:24,8:24,8:24]=5
            nib.save(nib.Nifti1Image(image,np.eye(4)),root/'mri.nii.gz')
            nib.save(nib.Nifti1Image(target,np.eye(4)),root/'mask.nii.gz')
            from support.evaluation_utils import checkpoint_sha256
            before=checkpoint_sha256(checkpoint)
            with patch('predict.render') as plotting:
                meta=predict_case(checkpoint,root/'mri.nii.gz',root/'mask.nii.gz',root/'output',
                                  'cpu','fp32',before)
                self.assertEqual(plotting.call_count,1)
            self.assertEqual(meta['epoch'],11)
            self.assertEqual(meta['target_shape'],[16,16,16])
            self.assertEqual(np.load(root/'output/prediction_dhw.npy').shape,(16,16,16))
            self.assertEqual(checkpoint_sha256(checkpoint),before)
            self.assertTrue((root/'output/prediction_complete.txt').exists())
            with self.assertRaisesRegex(ValueError,'SHA256'):
                predict_case(checkpoint,root/'mri.nii.gz',root/'mask.nii.gz',root/'bad',
                             'cpu','fp32','0'*64)
            self.assertFalse((root/'bad').exists())

    def test_history_plot_offset_and_missing_epochs(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'history.csv'
            path.write_text('epoch,train_loss,'+','.join(f'val_dice_{c}' for c in range(6))+
                            '\n1,0.5,0,0,0,0,0,0.4\n2,0.4,0,0,0,0,0,0.7\n')
            rows=read_history(path,'refinement',15)
            self.assertEqual([r['cumulative_epoch'] for r in rows],[16,17])
            self.assertEqual(rows[-1]['val_dice_5'],.7)
            path.write_text(path.read_text().replace('2,0.4','3,0.4'))
            with self.assertRaises(ValueError):read_history(path,'refinement',15)

    def test_overlap_counts_and_empty_class(self):
        target = np.array([5,5,0,0]).reshape(1,2,2)
        pred = np.array([5,0,5,0]).reshape(1,2,2)
        rows = class_metrics(pred,target)
        self.assertEqual((rows[5]['tp'],rows[5]['fp'],rows[5]['fn']),(1,1,1))
        self.assertEqual(rows[5]['dice'],.5)
        self.assertAlmostEqual(rows[5]['iou'],1/3)
        self.assertIsNone(rows[4]['dice'])
        with self.assertRaises(ValueError):
            class_metrics(pred,target+.5)

    def test_case_patient_and_pooled_scores_are_distinct(self):
        rows = []
        for case,patient,pred in [('one','A',5),('two','A',0),('three','B',5)]:
            for row in class_metrics(np.array([[[pred]]]),np.array([[[5]]])):
                rows.append(dict(row,case=case,patient_id=patient))
        result = summarize(rows)[5]
        self.assertAlmostEqual(result['mean_case_dice'],2/3)
        self.assertAlmostEqual(result['mean_patient_dice'],.75)
        self.assertAlmostEqual(result['pooled_dice'],.8)

    def test_original_axis_mapping_and_spacing(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'image.nii.gz'
            nib.save(nib.Nifti1Image(np.zeros((16,256,8),np.uint8),
                                    np.diag([-2.,-3.,4.,1.])),path)
            info = geometry(path,(4,8,128))
        self.assertEqual(info['spacing_dhw_mm'],[8.,4.,6.])
        self.assertEqual(info['shared_target_w_indices'],list(range(49,79)))
        self.assertEqual(info['original_axcodes'],['L','P','S'])

    def test_extent_proxy_planes(self):
        target = np.zeros((6,8,8),np.uint8)
        target[1:5,2:5,2:5]=5
        rows = boundary_regions(target,target,'S')
        self.assertEqual(rows[0]['region'],'inferior_extent_apex_proxy')
        self.assertEqual(rows[0]['depth_indices'],'1,2')
        self.assertEqual(rows[1]['depth_indices'],'3,4')
        info = dict(original_axcodes=['L','P','S'],shared_target_w_indices=[3,4])
        self.assertEqual([p[1] for p in planes_for(target,info)],[1,3,4,4])

    def test_instance_checkpoint_loader(self):
        torch.set_num_threads(1)
        model = make_model(base_channels=2,normalization='instance').eval()
        image = torch.randn(1,1,16,16,16)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'model.pt'
            torch.save(dict(model_state_dict=model.state_dict(),
                            model_config=dict(base_channels=2,normalization='instance')),path)
            loaded,_ = load_model(path,'cpu')
            with torch.inference_mode():
                self.assertTrue(torch.equal(model(image),loaded(image)))

    def test_frozen_split_is_checked(self):
        pairs = [(p,Path(f'{p}_scan.nii.gz'),Path('mask')) for p in ['A','B','C']]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'split.json'
            document = dict(train=[dict(patient='A',case='A_scan.nii.gz')],
                            validation=[dict(patient='B',case='B_scan.nii.gz')])
            path.write_text(json.dumps(document))
            with patch.object(evaluator,'get_data_pairs',return_value=pairs), patch.object(
                    evaluator,'split_by_patient',return_value=([pairs[0]],[pairs[1]],[pairs[2]])):
                self.assertEqual(evaluator.select_split(temp,'test',path),[pairs[2]])
                document['validation'][0]['patient']='C'
                path.write_text(json.dumps(document))
                with self.assertRaises(ValueError):
                    evaluator.select_split(temp,'test',path)


@unittest.skipUnless(importlib.util.find_spec('scipy'),'SciPy required for distance checks')
class DistanceTests(unittest.TestCase):
    def test_identical_and_translated_objects(self):
        target = np.zeros((6,6,6),bool); target[2,2,2]=True
        pred = np.zeros_like(target); pred[3,2,2]=True
        self.assertEqual(hd95(target,target,(2.5,1.,1.)),(0.,'ok'))
        self.assertEqual(hd95(pred,target,(2.5,1.,1.)),(2.5,'ok'))

    def test_missing_masks_do_not_receive_zero_distance(self):
        empty = np.zeros((4,4,4),bool)
        full = empty.copy(); full[1,1,1]=True
        self.assertEqual(hd95(empty,full,(1.,1.,1.)),(None,'prediction_empty'))
        self.assertEqual(hd95(full,empty,(1.,1.,1.)),(None,'ground_truth_empty'))
        self.assertEqual(hd95(empty,empty,(1.,1.,1.)),(None,'both_empty'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
