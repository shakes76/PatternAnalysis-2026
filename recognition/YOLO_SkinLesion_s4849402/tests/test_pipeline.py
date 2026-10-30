import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import audit_manifest, export, mask_to_box
from evaluate import choose_threshold, operating_metrics
from modules import CLASS_NAMES
from train import TrainConfig, validate_training_data
from prepare_isic import prepare, resolve_train_overlap


def fixture(root):
    rows = []
    for i, split in enumerate(('train', 'val')):
        image = root / f'sample{i}.jpg'
        mask = root / f'sample{i}_segmentation.png'
        Image.fromarray(np.full((16, 24, 3), i*70, dtype=np.uint8)).save(image)
        pixels = np.zeros((16, 24), dtype=np.uint8)
        pixels[2:12, 3:18] = 255
        Image.fromarray(pixels).save(mask)
        rows.append(dict(image_id=f'sample{i}', patient_id='', split=split,
                         class_id=i, image_path=str(image), mask_path=str(mask)))
        title = 'Training' if split == 'train' else 'Validation'
        with (root / f'ISIC-2017_{title}_Part3_GroundTruth.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image_id', 'melanoma', 'seborrheic_keratosis'])
            writer.writerow([f'sample{i}', int(i == 0), int(i == 1)])
    manifest = root / 'manifest.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


class PipelineTests(unittest.TestCase):
    def test_train_overlap_removed_without_changing_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            (root / 'sample1.jpg').write_bytes((root / 'sample0.jpg').read_bytes())
            rows = [{'image_id': 'sample0', 'split': 'train', 'class_id': 2, 'image_path': root/'sample0.jpg'},
                    {'image_id': 'sample1', 'split': 'val', 'class_id': 2, 'image_path': root/'sample1.jpg'},
                    {'image_id': 'unique', 'split': 'train', 'class_id': 0, 'image_path': root/'unused.jpg'}]
            report = root/'duplicates.json'
            report.write_text(json.dumps([[{'image_id': 'sample0', 'split': 'train'},
                                           {'image_id': 'sample1', 'split': 'val'}]]))
            kept, exclusions = resolve_train_overlap(rows, report)
            self.assertEqual([r['image_id'] for r in kept], ['sample1', 'unique'])
            self.assertEqual(exclusions[0]['excluded_image_id'], 'sample0')
            self.assertTrue((root/'sample0.jpg').exists())
            rows[1]['class_id'] = 1
            with self.assertRaisesRegex(ValueError, 'conflicting'):
                resolve_train_overlap(rows, report)

    def test_false_duplicate_report_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            rows = [{'image_id': f'sample{i}', 'split': split, 'class_id': 0,
                     'image_path': root/f'sample{i}.jpg'} for i, split in enumerate(('train', 'val'))]
            report = root/'duplicates.json'
            report.write_text(json.dumps([[{'image_id': 'sample0', 'split': 'train'},
                                           {'image_id': 'sample1', 'split': 'val'}]]))
            with self.assertRaisesRegex(ValueError, 'not pixel-identical'):
                resolve_train_overlap(rows, report)

    def test_archive_import(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            prepare(root, root / 'export', 'Synthetic import test only')
            self.assertTrue((root / 'export/manifest_official.csv').is_file())

    def test_patient_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = fixture(root)
            text = manifest.read_text().replace('sample0,,', 'sample0,p1,').replace('sample1,,', 'sample1,p1,')
            manifest.write_text(text)
            with self.assertRaisesRegex(ValueError, 'Patient leakage'):
                audit_manifest(manifest, 3)

    def test_duplicate_decoded_image_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = fixture(root)
            (root / 'sample1.jpg').write_bytes((root / 'sample0.jpg').read_bytes())
            with self.assertRaisesRegex(ValueError, 'Duplicate decoded'):
                audit_manifest(manifest, 3, 'official-isic', root)

    def test_mask_last_pixel(self):
        a = np.zeros((10, 20), dtype=np.uint8)
        a[-1, -1] = 255
        self.assertEqual(mask_to_box(a), (0.975, 0.95, 0.05, 0.1))

    def test_empty_mask_rejected(self):
        with self.assertRaises(ValueError):
            mask_to_box(np.zeros((10, 20)))

    def test_official_export_and_training_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = fixture(root)
            with self.assertRaisesRegex(ValueError, 'patient_id'):
                audit_manifest(manifest, 3)
            out = root / 'export'
            export(manifest, out, list(CLASS_NAMES), 'official-isic', 'Synthetic unit test only', root)
            result = validate_training_data(TrainConfig(out / 'data.yaml', root / 'run'))
            self.assertFalse(result['audit']['patient_independence_programmatically_verified'])
            self.assertTrue((out / 'labels/val/sample1.txt').exists())
            self.assertFalse((out / 'images/test').exists())

    def test_official_class_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = fixture(root)
            text = manifest.read_text().replace('val,1,', 'val,2,')
            manifest.write_text(text)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                audit_manifest(manifest, 3, 'official-isic', root)

    def test_duplicate_detection_and_miss(self):
        gt = [{'image_id': 'a', 'detections': [{'class_id': 0, 'xyxy': [0, 0, 10, 10]}]},
              {'image_id': 'b', 'detections': [{'class_id': 0, 'xyxy': [0, 0, 10, 10]}]}]
        pred = [{'image_id': 'a', 'detections': [
            {'class_id': 0, 'xyxy': [0, 0, 10, 10], 'confidence': 0.9},
            {'class_id': 0, 'xyxy': [0, 0, 10, 10], 'confidence': 0.8}]},
                {'image_id': 'b', 'detections': []}]
        result = operating_metrics(pred, gt, 0.5)
        self.assertEqual(result['overall']['tp'], 1)
        self.assertEqual(result['overall']['fp'], 1)
        self.assertEqual(result['overall']['fn'], 1)
        self.assertEqual(result['overall']['recall'], 0.5)
        sweep = choose_threshold(pred, gt)
        self.assertEqual(sweep['status'], 'descriptive')
        self.assertIsNone(sweep['selected'])
        self.assertNotIn('min_recall', sweep['policy'])
        self.assertEqual(sweep['curve'][0]['overall']['recall'], 0.5)
        self.assertEqual(choose_threshold(pred, gt, {'min_recall': 0.75})['status'], 'unmet')

    def test_wrong_class_is_fp_and_fn(self):
        gt = [{'image_id': 'a', 'detections': [{'class_id': 0, 'xyxy': [0, 0, 10, 10]}]}]
        pred = [{'image_id': 'a', 'detections': [
            {'class_id': 1, 'xyxy': [0, 0, 10, 10], 'confidence': 0.9}]}]
        result = operating_metrics(pred, gt, 0.5)
        self.assertEqual(result['per_class']['melanoma']['fn'], 1)
        self.assertEqual(result['per_class']['seborrheic_keratosis']['fp'], 1)


if __name__ == '__main__':
    unittest.main()
