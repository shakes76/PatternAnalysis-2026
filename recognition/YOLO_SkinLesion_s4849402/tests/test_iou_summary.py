import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from summarize_iou import summarize
from modules import CLASS_NAMES
from test_pipeline import fixture


class IouSummaryTests(unittest.TestCase):
    def test_missed_target_is_zero_in_all_gt_mean(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            fixture(root)
            (root/'images/test').mkdir(parents=True)
            (root/'labels/test').mkdir(parents=True)
            for i in range(2):
                shutil.copy2(root/f'sample{i}.jpg',root/f'images/test/ISIC_{i}.jpg')
                (root/f'labels/test/ISIC_{i}.txt').write_text(f'0 {5/24} {5/16} {10/24} {10/16}\n')
            config=root/'data.yaml'
            config.write_text(json.dumps({'path':str(root),'test':'images/test','names':list(CLASS_NAMES)}))
            preds=root/'predictions.json'
            preds.write_text(json.dumps([
                {'image_id':'ISIC_0','detections':[{'class_id':0,'confidence':0.8,'xyxy':[0,0,10,10]}]},
                {'image_id':'ISIC_1','detections':[{'class_id':1,'confidence':0.9,'xyxy':[0,0,10,10]}]},
            ]))
            result=summarize(config,preds,root/'summary.json')
            m=result['per_class']['melanoma']
            self.assertEqual(m['ground_truth'],2)
            self.assertEqual(m['matched'],1)
            self.assertAlmostEqual(m['mean_iou_matched'],1.0)
            self.assertAlmostEqual(m['mean_iou_all_gt_zero_for_misses'],0.5)
            self.assertIsNone(result['per_class']['nevus']['mean_iou_matched'])


if __name__=='__main__':
    unittest.main()
