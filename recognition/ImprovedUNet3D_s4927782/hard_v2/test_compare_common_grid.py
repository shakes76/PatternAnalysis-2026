"""Meaningful synthetic fixtures for affine mapping, empties and comparison."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import compare_common_grid as c


def scored_case(patient, visit, prediction, target, spacing=(1, 1, 1)):
    return {'patient': patient, 'visit': visit, **c.class_metrics(prediction, target),
            'prostate_surface': c.prostate_surface_metrics(prediction, target, spacing)}


class GeometryTests(unittest.TestCase):
    def test_actual_resolution_ratio_uses_voxel_centres_and_axis_order(self):
        source = np.indices((6, 6, 6))
        source = (100 * source[0] + 10 * source[1] + source[2]).astype(np.int16)
        source_affine = np.diag([2., 3., 4., 1.])
        transform = np.diag([1.5, 1.5, 1.5, 1.])
        transform[:3, 3] = .25
        target_affine = source_affine @ transform
        mapped, _ = c.categorical_to_grid(source, source_affine, target_affine, (4, 4, 4))
        expected = source[np.ix_([0, 2, 3, 5], [0, 2, 3, 5], [0, 2, 3, 5])]
        np.testing.assert_array_equal(mapped, expected)

    def test_world_x_translation_changes_width_not_depth_and_uses_background_outside(self):
        source = np.arange(3 * 4 * 5).reshape(3, 4, 5).astype(np.int16) + 1
        source_affine = np.diag([2., 3., 4., 1.])
        target_affine = source_affine.copy()
        target_affine[0, 3] = 2.
        mapped, _ = c.categorical_to_grid(source, source_affine, target_affine, source.shape)
        np.testing.assert_array_equal(mapped[:, :, :-1], source[:, :, 1:])
        np.testing.assert_array_equal(mapped[:, :, -1], 0)


class MetricTests(unittest.TestCase):
    def test_absent_class_and_missed_target_policies(self):
        empty = np.zeros((3, 3, 3), dtype=np.uint8)
        actual = empty.copy(); actual[1, 1, 1] = 5
        both_empty = c.class_metrics(empty, empty)['per_class'][5]
        self.assertIsNone(both_empty['dice'])
        self.assertIsNone(both_empty['iou'])
        missed = c.class_metrics(empty, actual)['per_class'][5]
        fp_only = c.class_metrics(actual, empty)['per_class'][5]
        self.assertEqual((missed['dice'], fp_only['dice']), (0, 0))
        self.assertEqual(c.prostate_surface_metrics(empty, actual, (1, 1, 1)), {'empty_status': 'missed', 'hd95_mm': None})
        self.assertEqual(c.prostate_surface_metrics(actual, empty, (1, 1, 1)), {'empty_status': 'fp_only', 'hd95_mm': None})

    def test_surface_distance_uses_dhw_physical_spacing(self):
        actual = np.zeros((5, 5, 5), dtype=np.uint8); actual[1, 2, 2] = 5
        predicted = np.zeros_like(actual); predicted[2, 2, 2] = 5
        result = c.prostate_surface_metrics(predicted, actual, (7., 2., 3.))
        self.assertEqual(result['hd95_mm'], 7.)

    def test_patient_weighting_and_hd95_common_support(self):
        target = np.zeros((3, 3, 3), dtype=np.uint8); target[1, 1, 1] = 5
        empty = np.zeros_like(target)
        before = [scored_case('A001', 'Week0', target, target),
                  scored_case('A001', 'Week1', empty, target),
                  scored_case('B001', 'Week0', target, target)]
        after = [scored_case(r['patient'], r['visit'], target, target) for r in before]
        result = c.aggregate(before)
        self.assertEqual(result['per_class'][5]['dice'], .75)
        self.assertEqual(result['prostate_empty_status_counts'], {'both_present': 2, 'missed': 1})
        paired = c.paired_summary(before, after)['paired_hd95']
        self.assertEqual((paired['eligible_case_pairs'], paired['excluded_case_pairs'], paired['num_patients']), (2, 1, 2))
        self.assertEqual(paired['delta_hd95_mm'], 0.)

    def test_gate_boundary_is_fixed_and_test_never_adopts(self):
        baseline = {'per_class': [{}] * 5 + [{'dice': .68}], 'foreground_mean_dice': .50}
        candidate = {'per_class': [{}] * 5 + [{'dice': .69}], 'foreground_mean_dice': .49}
        self.assertTrue(c.comparison_gate(baseline, candidate, 'validation')['replacement_supported_by_validation_gate'])
        self.assertIsNone(c.comparison_gate(baseline, candidate, 'test')['replacement_supported_by_validation_gate'])
        candidate['per_class'][5]['dice'] = .689999
        self.assertFalse(c.comparison_gate(baseline, candidate, 'validation')['replacement_supported_by_validation_gate'])


class FileComparisonTests(unittest.TestCase):
    def build_inputs(self, directory):
        root = Path(directory)
        keys = [('A001', 'Week0'), ('A001', 'Week1'), ('B001', 'Week0')]
        shape = (4, 4, 4)
        actual = np.zeros(shape, dtype=np.uint8); actual[1, 1, 1] = 5
        high = actual.repeat(2, 0).repeat(2, 1).repeat(2, 2)
        old_affine = np.diag([2., 2., 2., 1.]); old_affine[:3, 3] = .5
        new_affine = np.eye(4)
        manifests = {}
        for version in ('v1', 'v2'):
            records = []
            for patient, visit in keys:
                record = {'patient': patient, 'visit': visit, 'split': 'validation',
                          'cache_path': f'volumes/{patient}_{visit}.npz',
                          'resampled_affine_xyz': (old_affine if version == 'v1' else new_affine).tolist(),
                          'spacing_dhw_mm': [2., 2., 2.] if version == 'v1' else [1., 1., 1.]}
                if version == 'v1':
                    record['class_voxel_counts'] = np.bincount(actual.ravel(), minlength=6).tolist()
                    destination = root / version / record['cache_path']
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(destination, mask=actual, spacing=np.array([2., 2., 2.], dtype=np.float32))
                records.append(record)
            manifest = {'label_ids': list(range(6)), 'num_classes': 6,
                        'target_shape_dhw': list(shape if version == 'v1' else high.shape),
                        'patient_splits': {'train': ['T001'], 'validation': ['A001', 'B001'], 'test': ['T002']},
                        'records': records}
            path = root / version / 'manifest.json'; path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(manifest)); manifests[version] = path
        kwargs = {'v1_manifest': manifests['v1'], 'v2_manifest': manifests['v2'],
                  'split': 'validation', 'cache_root': None, 'selection_freeze': None,
                  'retrospective_test_ack': False}
        spacing_array = np.asarray([2., 2., 2.], dtype=np.float32)
        identity = {'manifest_sha256': c.file_sha256(manifests['v1']), 'records': {
            f'volumes/{patient}_{visit}.npz': {'arrays': {'mask': {'sha256': c.decoded_array_sha256(actual)},
                                                        'spacing': {'sha256': c.decoded_array_sha256(spacing_array)}}}
            for patient, visit in keys}}
        identity_path = manifests['v1'].parent / 'cache_identity.json'
        identity_path.write_text(json.dumps(identity))
        self.identity_digest = c.file_sha256(identity_path)
        for version in ('v1', 'v2'):
            for model in c.MODEL_NAMES:
                cases = []
                for patient, visit in keys:
                    prediction = np.zeros_like(actual) if version == 'v1' and model == 'unet2d' else actual
                    output_prediction = high if version == 'v2' else prediction
                    base = root / version / model
                    destination = base / 'predictions' / f'{patient}_{visit}.npz'
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(destination, prediction=output_prediction,
                                        spacing_dhw=[2., 2., 2.] if version == 'v1' else [1., 1., 1.],
                                        label_ids=np.arange(6), affine_xyz=old_affine if version == 'v1' else new_affine)
                    row = scored_case(patient, visit, prediction, actual)
                    row['prediction_file'] = destination.relative_to(base).as_posix()
                    cases.append(row)
                evaluation = {'split': 'validation', 'cases': cases,
                              'checkpoint_sha256': c.FROZEN_V1_CHECKPOINTS[model] if version == 'v1' else 'a' * 64}
                # Exercise both v1 top-level and v2 identity-nested digest schemas.
                if version == 'v1':
                    evaluation['manifest_sha256'] = c.file_sha256(manifests[version])
                    evaluation['model_config'] = {'model': model}
                else:
                    evaluation['identity'] = {'manifest_sha256': c.file_sha256(manifests[version]), 'model': model,
                                              'protocol_sha256': 'c' * 64}
                path = base / 'evaluation.json'; path.write_text(json.dumps(evaluation))
                kwargs[f'{version}_evaluation_{model}'] = path
        return argparse.Namespace(**kwargs)

    def test_four_complete_model_outputs_map_to_same_gt_and_keep_negative_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_inputs(directory)
            # Only the fixture replaces the production-pinned identity-book hash.
            with mock.patch.object(c, 'FROZEN_V1_CACHE_IDENTITY_SHA256', self.identity_digest):
                result = c.compare_files(args)
            two_d, three_d = (result['models'][x] for x in c.MODEL_NAMES)
            self.assertEqual(two_d['v1']['aggregate']['per_class'][5]['dice'], 0.)
            self.assertEqual(two_d['v2']['aggregate']['per_class'][5]['dice'], 1.)
            self.assertTrue(two_d['gate']['replacement_supported_by_validation_gate'])
            self.assertFalse(three_d['gate']['replacement_supported_by_validation_gate'])
            self.assertEqual(three_d['v2']['aggregate']['num_cases'], 3)

    def test_incomplete_case_set_and_wrong_freeze_fail_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_inputs(directory)
            freeze = Path(directory) / 'freeze.json'
            freeze.write_text(json.dumps({'models': {x: {'checkpoint_sha256': 'b' * 64} for x in c.MODEL_NAMES}}))
            args.selection_freeze = freeze
            with self.assertRaisesRegex(ValueError, 'frozen selection'):
                c.compare_files(args)
            args.selection_freeze = None
            path = args.v2_evaluation_improved3d
            evaluation = json.loads(path.read_text()); evaluation['cases'].pop()
            path.write_text(json.dumps(evaluation))
            with self.assertRaisesRegex(ValueError, 'complete declared split'):
                c.compare_files(args)

    def test_model_path_swap_is_rejected_and_gt_cannot_keep_counts_but_move_voxels(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_inputs(directory)
            correct = args.v2_evaluation_unet2d
            args.v2_evaluation_unet2d = args.v2_evaluation_improved3d
            with self.assertRaisesRegex(ValueError, 'model identity'):
                c.compare_files(args)
            args.v2_evaluation_unet2d = correct
            gt = args.v1_manifest.parent / 'volumes' / 'A001_Week0.npz'
            altered = np.zeros((4, 4, 4), dtype=np.uint8); altered[2, 2, 2] = 5
            np.savez_compressed(gt, mask=altered, spacing=np.asarray([2., 2., 2.], dtype=np.float32))
            with mock.patch.object(c, 'FROZEN_V1_CACHE_IDENTITY_SHA256', self.identity_digest):
                with self.assertRaisesRegex(ValueError, 'decoded hashes'):
                    c.compare_files(args)

    def test_identity_book_and_optional_protocol_binding_are_required_when_given(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_inputs(directory)
            with self.assertRaisesRegex(ValueError, 'identity book differs'):
                c.compare_files(args)
            freeze = Path(directory) / 'freeze.json'
            freeze.write_text(json.dumps({'models': {x: {'checkpoint_sha256': 'a' * 64} for x in c.MODEL_NAMES},
                                          'manifest_sha256': c.file_sha256(args.v2_manifest),
                                          'protocol_sha256': 'd' * 64}))
            args.selection_freeze = freeze
            with self.assertRaisesRegex(ValueError, 'different v2 protocol'):
                c.compare_files(args)

    def test_another_same_architecture_baseline_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_inputs(directory)
            path = args.v1_evaluation_unet2d
            evaluation = json.loads(path.read_text()); evaluation['checkpoint_sha256'] = 'b' * 64
            path.write_text(json.dumps(evaluation))
            with self.assertRaisesRegex(ValueError, 'frozen seed42 checkpoint'):
                c.compare_files(args)

    def test_test_cli_requires_ack_and_freeze_before_any_missing_input_read(self):
        arguments = ['--v1-manifest', '/missing/v1', '--v2-manifest', '/missing/v2',
                     '--output-dir', '/missing/output', '--split', 'test']
        for version in ('v1', 'v2'):
            for model in c.MODEL_NAMES:
                arguments += [f'--{version}-evaluation-{model}', '/missing/eval']
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as context:
                c.main(arguments)
        self.assertEqual(context.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
