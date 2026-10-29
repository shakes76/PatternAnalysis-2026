"""Check diagnostic selection, metric consistency, preserved grids and five PNG exports."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from compare_results import metric_rows
from failure_analysis import crop_bounds, export_examples, select_examples

RUNTIME_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('torch', 'nibabel', 'matplotlib'))


def record(index, patient, label, tp, fp, count=100):
    matrix = [[0]*6 for _ in range(6)]
    matrix[0][0], matrix[0][label] = 1024-count-fp, fp
    matrix[label][label], matrix[label][1] = tp, count-tp
    classes=[]
    for k in range(6):
        actual=sum(matrix[k]); predicted=sum(row[k] for row in matrix); hit=matrix[k][k]
        denominator=actual+predicted; union=actual+predicted-hit
        classes.append({'label_id':k,'tp':hit,'fp':predicted-hit,'fn':actual-hit,
                        'target_pixels':actual,'predicted_pixels':predicted,
                        'dice':2*hit/denominator if denominator else None,
                        'iou':hit/union if union else None})
    result={'index':index,'patient_id':patient,'week':0,'slice_id':index,
            'image':f'case_{patient:03d}_week_0_slice_{index}.nii.gz',
            'mask':f'seg_{patient:03d}_week_0_slice_{index}.nii.gz',
            'shape':[32,32],'pixels':1024,'confusion':matrix,'classes':classes}
    metric_rows(result)
    return result


def reports():
    left=[record(0,38,3,90,5),record(1,37,4,80,20),record(2,38,5,50,10),
          record(3,39,5,70,20),record(4,36,5,40,10)]
    right=[record(0,38,3,30,5),record(1,37,4,90,80),record(2,38,5,20,10),
           record(3,39,5,40,40),record(4,36,5,90,5)]
    return [dict(split='validate',smoke_only=False,processed_samples=5,split_samples=5,slices=rows)
            for rows in (left,right)]


class FailureAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)

    def test_selection_records_rules_and_expected_cases(self):
        selection=select_examples(*reports())
        self.assertEqual([row['index'] for row in selection],[0,1,2,3,4])
        self.assertEqual([row['label_id'] for row in selection],[3,4,5,5,5])
        self.assertEqual(selection[0]['reference_metrics']['ImprovedUNet2D']['fn'],70)
        self.assertEqual(selection[1]['reference_metrics']['ImprovedUNet2D']['fp'],80)
        self.assertEqual(selection[-1]['selection_rule'],'dice_gain')
        self.assertTrue(all(row['minimum_target_pixels']>0 for row in selection))

    def test_selection_is_order_independent_and_ties_use_lowest_index(self):
        a,b=reports()
        a['slices'].append(record(8,38,3,90,5)); b['slices'].append(record(8,38,3,30,5))
        for report in (a,b):
            report['processed_samples']=report['split_samples']=6
            report['slices'].reverse()
        selection=select_examples(a,b)
        self.assertEqual(selection[0]['index'],0)
        self.assertEqual(selection[0]['eligible_candidates'],2)

    def test_partial_test_and_missing_slice_records_are_rejected(self):
        for field,value in [('split','test'),('smoke_only',True),('processed_samples',4)]:
            a,b=reports(); b[field]=value
            with self.assertRaises(ValueError):
                select_examples(a,b)

    def test_mismatched_identity_target_counts_and_corrupt_scores_rejected(self):
        a,b=reports(); b['slices'][0]['image']='case_038_week_0_slice_7.nii.gz'
        with self.assertRaisesRegex(ValueError,'identity'):
            select_examples(a,b)
        a,b=reports(); b['slices'][0]=record(0,38,3,30,5,count=99)
        with self.assertRaisesRegex(ValueError,'Ground-truth counts'):
            select_examples(a,b)
        a,b=reports(); b['slices'][0]['classes'][3]['dice']=0.99
        with self.assertRaisesRegex(ValueError,'overlap metric'):
            select_examples(a,b)

    def test_minimum_target_size_is_enforced(self):
        a,b=reports()
        a['slices'][0]=record(0,38,3,30,5,count=63)
        b['slices'][0]=record(0,38,3,10,5,count=63)
        with self.assertRaisesRegex(ValueError,'No eligible'):
            select_examples(a,b)

    def test_crop_preserves_native_coordinates_and_clips_edges(self):
        self.assertEqual(crop_bounds([32,48],[[0,1],[10,20]],margin=3),[0,14,0,24])
        self.assertEqual(crop_bounds([32,48],[],margin=3),[0,32,0,48])
        with self.assertRaises(ValueError):
            crop_bounds([32,48],[[32,0]])
        with self.assertRaises(ValueError):
            crop_bounds([32,48],[[0,0]],margin=-1)

    def test_existing_output_is_preserved_before_runtime_imports(self):
        destination=self.root/'existing';destination.mkdir()
        marker=destination/'keep.txt';marker.write_text('preserve')
        with self.assertRaises(FileExistsError):
            export_examples({},None,[],destination,'cpu',{})
        self.assertEqual(marker.read_text(),'preserve')
        self.assertEqual(len(list(destination.iterdir())),1)

    @unittest.skipUnless(RUNTIME_AVAILABLE,'PyTorch, NiBabel and Matplotlib required')
    def test_five_png_exports_use_actual_inference_and_hand_checked_counts(self):
        import torch
        from torch import nn
        torch.set_num_threads(1)
        selection=select_examples(*reports())
        class ConstantModel(nn.Module):
            def __init__(self,label):
                super().__init__();self.label=label
            def forward(self,x):
                logits=x.new_zeros((x.shape[0],6,*x.shape[-2:]))
                logits[:,self.label]=1
                return logits
        class TinyDataset:
            def __getitem__(self,index):
                image=torch.linspace(-1,1,1024).reshape(1,32,32)
                target=torch.zeros(32,32,dtype=torch.long)
                target.reshape(-1)[:100]=selection[index]['label_id']
                return image,target
        models={'UNet2D':ConstantModel(1),'ImprovedUNet2D':ConstantModel(3)}
        destination=self.root/'figures'
        with contextlib.redirect_stdout(io.StringIO()):
            report=export_examples(models,TinyDataset(),selection,destination,'cpu',
                                   {'scope':'Synthetic validation renderer test'})
        self.assertEqual(len(report['examples']),5)
        self.assertEqual(report['status'],'rendered_pending_integrity_check')
        first=report['examples'][0]['rendered_metrics']
        self.assertEqual((first['UNet2D']['tp'],first['UNet2D']['fp'],first['UNet2D']['fn']),(0,0,100))
        self.assertEqual((first['ImprovedUNet2D']['tp'],first['ImprovedUNet2D']['fp'],first['ImprovedUNet2D']['fn']),(100,924,0))
        self.assertEqual(len(list(destination.glob('*.png'))),5)
        for path in destination.glob('*.png'):
            self.assertTrue(path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
        saved=json.loads((destination/'analysis_manifest.json').read_text())
        self.assertEqual(saved['examples'][0]['shape'],[32,32])
        self.assertTrue(all(not model.training for model in models.values()))


if __name__=='__main__':
    unittest.main()
