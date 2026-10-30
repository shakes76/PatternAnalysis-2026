import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_failures import analyze, classify_case
from dataset import export
from modules import CLASS_NAMES
from test_pipeline import fixture


class FailureAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.gt = {'class_id': 0, 'xyxy': [0, 0, 10, 10]}

    def test_no_detection_and_wrong_class(self):
        self.assertEqual(classify_case([], self.gt, 0.15)['status'], 'no_detection')
        wrong = [{'class_id': 1, 'confidence': 0.9, 'xyxy': [0, 0, 10, 10]}]
        self.assertEqual(classify_case(wrong, self.gt, 0.15)['status'], 'classification_error')

    def test_localization_and_duplicate_predictions(self):
        poor = [{'class_id': 0, 'confidence': 0.9, 'xyxy': [5, 0, 15, 10]}]
        self.assertEqual(classify_case(poor, self.gt, 0.15)['status'], 'localization_error')
        box = {'class_id': 0, 'confidence': 0.9, 'xyxy': [0, 0, 10, 10]}
        case = classify_case([box, dict(box)], self.gt, 0.15)
        self.assertFalse(case['false_negative_at_iou05'])
        self.assertEqual(case['false_positives_at_iou05'], 1)

    def test_report_and_overlay_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = fixture(root)
            export(manifest, root/'dataset', list(CLASS_NAMES), 'official-isic', 'Synthetic test only', root)
            predictions = root/'predictions.json'
            predictions.write_text(json.dumps([{'image_id': 'sample1', 'detections': [
                {'class_id': 0, 'confidence': 0.8, 'xyxy': [3, 2, 18, 12]}]}]))
            report = analyze(root/'dataset/data.yaml', predictions, root/'errors')
            self.assertEqual(report['case_counts']['classification_error'], 1)
            self.assertTrue((root/'errors/sample1.jpg').is_file())
            self.assertTrue((root/'errors/case_review.md').is_file())
            self.assertEqual(report['operating_iou05']['overall']['fn'], 1)
            with self.assertRaises(FileExistsError):
                analyze(root/'dataset/data.yaml', predictions, root/'errors')


if __name__ == '__main__':
    unittest.main()
