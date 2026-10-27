"""Explicit synthetic fixtures; never reads actual model outputs or pixels."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import plot_hard_followup as plot


def aggregate(cases):
    grouped = {}
    for case in cases:
        grouped.setdefault(case['patient'], []).append(case)
    patients = []
    for name, rows in sorted(grouped.items()):
        patients.append({'patient': name, 'num_cases': len(rows),
                         'per_class': [{'class_index': c, 'dice': sum(x['per_class'][c]['dice'] for x in rows)/len(rows)}
                                       for c in range(6)],
                         'foreground_mean_dice': sum(x['foreground_mean_dice'] for x in rows)/len(rows)})
    return {'num_cases': len(cases), 'num_patients': len(patients), 'patients': patients,
            'per_class': [{'class_index': c, 'dice': sum(x['per_class'][c]['dice'] for x in patients)/len(patients)}
                          for c in range(6)],
            'foreground_mean_dice': sum(x['foreground_mean_dice'] for x in patients)/len(patients)}


def fixture(split='test'):
    visits = [8, 6, 5, 4, 4, 3] if split == 'test' else [8, 8, 6, 6, 5, 5]
    result = {'status': 'completed', 'split': split, 'scientific_status': 'Synthetic exploratory fixture, not actual model results',
              'synthetic_fixture': True, 'scope_plan_sha256': 'a'*64, 'models': {}}
    for mi, key in enumerate(plot.ORDER):
        cases = []
        for pi, count in enumerate(visits):
            for visit in range(count):
                rows = []
                for ci in range(6):
                    tp = 95-ci*2 + mi
                    if ci == 5 and pi == 0:
                        tp = [55, 15, 10][mi]  # A new-model failure must remain visible.
                    tp = min(100, tp)
                    rows.append({'class_index': ci, 'dice': tp/100, 'iou': tp/(200-tp),
                                 'target_voxels': 100, 'predicted_voxels': 100,
                                 'intersection': tp, 'union': 200-tp})
                cases.append({'patient': f'SYN{pi}', 'visit': f'Week{visit}', 'per_class': rows,
                              'foreground_mean_dice': sum(r['dice'] for r in rows[1:])/5})
        resource = {'training_loop_seconds': [60., 180., 240.][mi], 'successful_updates': 300,
                    'amp_skipped_updates': 0, 'parameters': 1000+mi*1000,
                    'peak_training_allocated_cuda_bytes': [2**28, 2**31, 2**31][mi],
                    'mean_whole_case_inference_ms': [10., 50., 300.][mi],
                    'peak_inference_allocated_cuda_bytes': 2**28}
        result['models'][key] = {'cases': cases, 'aggregate': aggregate(cases), 'resources': resource}
    return result


class HardPlotTests(unittest.TestCase):
    def test_full_synthetic_pipeline_renders_two_nonempty_pngs_with_bound_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root/'comparison.json'
            data = fixture()
            report.write_text(json.dumps(data))
            output = root/'plots'
            receipt = plot.plot_report(report, output)
            self.assertTrue(receipt['synthetic_fixture'])
            self.assertEqual(receipt['model_order'], list(plot.ORDER))
            self.assertEqual(sum(receipt['patient_visit_counts'].values()), 30)
            self.assertEqual(len(receipt['patient_visit_counts']), 6)
            self.assertEqual(receipt['input_report_sha256'], hashlib.sha256(report.read_bytes()).hexdigest())
            self.assertFalse(receipt['new_independent_test_claimed'])
            self.assertGreater(receipt['cpu_seconds'], 0)
            self.assertEqual(set(receipt['figures']), {'common_grid_dice.png', 'resource_costs.png'})
            for name, record in receipt['figures'].items():
                path = output/name
                self.assertEqual(record['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
                pixels = np.asarray(Image.open(path).convert('RGB'))
                self.assertGreater(pixels.shape[0], 800)
                self.assertGreater(pixels.shape[1], 1800)
                self.assertGreater(float(pixels.std()), 15)
            # The lowest new-model patient remains present rather than being dropped.
            values = {p['patient']: p['per_class'][5]['dice'] for p in data['models']['hard_improved3d']['aggregate']['patients']}
            self.assertAlmostEqual(values['SYN0'], .1)
            with self.assertRaises(FileExistsError):
                plot.plot_report(report, output)

    def test_validation_population_and_title_are_not_called_blind_test(self):
        data = fixture('validation')
        self.assertEqual(sum(plot.validate(data).values()), 38)
        self.assertIn('Development VAL', plot.cohort_title('validation'))
        self.assertIn('Previously inspected TEST', plot.cohort_title('test'))

    def test_incomplete_wrong_identity_or_population_is_rejected_before_output(self):
        examples = []
        data = fixture(); data['status'] = 'running'; examples.append(data)
        data = fixture(); del data['models']['v1_unet2d']; examples.append(data)
        data = fixture(); data['models']['v1_unet2d']['cases'].pop(); examples.append(data)
        data = fixture(); data['models']['v1_unet2d']['cases'][1] = copy.deepcopy(data['models']['v1_unet2d']['cases'][0]); examples.append(data)
        data = fixture(); data['scope_plan_sha256'] = 'unbound'; examples.append(data)
        data = fixture(); data['models']['hard_improved3d']['aggregate']['per_class'][5]['dice'] = .99; examples.append(data)
        data = fixture(); data['models']['hard_improved3d']['resources']['mean_whole_case_inference_ms'] = 0; examples.append(data)
        data = fixture(); data['models']['hard_improved3d']['cases'][0]['per_class'][5]['dice'] = float('nan'); examples.append(data)
        for i, data in enumerate(examples):
            with self.subTest(index=i), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); source = root/'bad.json'; source.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    plot.plot_report(source, root/'plots')
                self.assertFalse((root/'plots').exists())


if __name__ == '__main__':
    unittest.main()
