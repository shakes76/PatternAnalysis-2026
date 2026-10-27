"""Engineering recovery metadata contracts, never actual MRI validation."""
import copy
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import parallel_prepare_v2 as recovery
import v2_pipeline as v2


class RecoveryFixtures(unittest.TestCase):
    def test_amendment_binds_original_protocol_and_one_bounded_engineering_retry(self):
        protocol = {'experiment_id': 'fixture', '_sha256': 'a' * 64}
        amendment = {'experiment_id': 'fixture', 'original_protocol_sha256': 'a' * 64,
                     'failed_job_id': 704519, 'engineering_only': True,
                     'scientific_configuration_changed': False, 'reason': 'Recorded preparation timeout',
                     'recovery': {'workers': 4, 'maximum_walltime_minutes': 20}, 'helper_sha256': 'b' * 64}
        recovery.validate_amendment(amendment, protocol, 'b' * 64, 4)
        for key, value in [('scientific_configuration_changed', True), ('failed_job_id', 1),
                           ('helper_sha256', 'c' * 64), ('reason', ''),
                           ('recovery', {'workers': 4, 'maximum_walltime_minutes': 30})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                recovery.validate_amendment(dict(amendment, **{key: value}), protocol, 'b' * 64, 4)
        with self.assertRaises(ValueError):
            recovery.validate_amendment(amendment, protocol, 'b' * 64, 8)

    def test_test_placeholder_removes_old_pixel_metadata_and_never_enters_worker(self):
        geometry = {'shape_xyz': [9, 7, 5], 'affine_xyz': np.diag([1., 2., 3., 1.]).tolist()}
        source = {'patient': 'patient', 'visit': '0', 'split': 'test', 'source_image': 'missing.nii',
                  'source_mask': 'missing_mask.nii', 'class_voxel_counts': [0, 1, 2, 3, 4, 5],
                  **{k: copy.deepcopy(geometry) for k in ('original_image', 'original_mask', 'canonical_image', 'canonical_mask')}}
        dependencies = SimpleNamespace(utils=SimpleNamespace(safe_case_name=lambda p, v: p + '_' + v),
                                       prepare=SimpleNamespace(preprocess_pair=Mock(side_effect=AssertionError('TEST pixels opened'))))
        record = recovery.initial_record(source, v2.DEFAULT_CONFIGURATION, dependencies)
        self.assertTrue(record['pending'])
        self.assertNotIn('class_voxel_counts', record)
        self.assertNotIn('normalization', record)
        self.assertNotIn('cache_sha256', record)
        with self.assertRaises(ValueError):
            recovery.complete_record(record, Path('unused'), Path('unused'), v2.DEFAULT_CONFIGURATION, dependencies)
        dependencies.prepare.preprocess_pair.assert_not_called()

    def test_recovery_manifest_equals_frozen_sequential_manifest_for_all211_records(self):
        """A synthetic metadata transaction, not a pixel or multiprocessing claim."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = {'num_classes': 6, 'label_ids': list(range(6)),
                      'raw_label_to_index': {str(i): i for i in range(6)},
                      'patient_splits': {'train': ['TRAIN'], 'validation': ['VAL'], 'test': ['TEST']},
                      'audit_sha256': '1' * 64, 'label_semantics': {'fixture': True}, 'records': []}
            geometry = {'shape_xyz': [8, 8, 8], 'affine_xyz': np.diag([1., 2., 3., 1.]).tolist()}
            raw = root / 'raw'; raw.mkdir()
            for split, count, patient in [('train', 143, 'TRAIN'), ('validation', 38, 'VAL'), ('test', 30, 'TEST')]:
                for index in range(count):
                    r = {'patient': patient, 'visit': str(index), 'split': split,
                         'source_image': f'{split}_{index}_image.nii', 'source_mask': f'{split}_{index}_mask.nii',
                         'class_voxel_counts': [512, 0, 0, 0, 0, 0],
                         **{k: copy.deepcopy(geometry) for k in ('original_image', 'original_mask', 'canonical_image', 'canonical_mask')}}
                    source['records'].append(r)
                    # Only TRAIN/VAL fake sources exist: opening TEST would fail.
                    if split != 'test':
                        for role in ('source_image', 'source_mask'):
                            (raw / r[role]).write_bytes(b'metadata fixture source')
            source_path = root / 'source.json'; v2.write_json(source_path, source)
            plan_path = root / 'protocol.json'
            plan = {'experiment_id': 'fixture', 'configuration': copy.deepcopy(v2.DEFAULT_CONFIGURATION),
                    'required_original_sha256': {'artifacts/cache64/manifest.json': v2.sha256(source_path)}}
            v2.write_json(plan_path, plan); plan = v2.load_protocol(plan_path)
            opened = []

            def preprocess(image, mask, shape, labels):
                opened.append(Path(image).name)
                r = next(r for r in source['records'] if r['source_image'] == Path(image).name)
                metadata = v2.expected_geometry(r, shape)
                metadata.update(class_voxel_counts=[int(np.prod(shape)), 0, 0, 0, 0, 0],
                                normalization={'method': 'synthetic metadata only'},
                                observed_raw_label_ids=[0], retained_class_indices=[0],
                                resampling={'image_order': 1, 'mask_order': 0})
                return {}, metadata

            def fake_write(destination, arrays):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b'fixture cache, not MRI')
                return v2.sha256(destination)

            dependencies = SimpleNamespace(dataset=SimpleNamespace(load_manifest=v2.read_json),
                prepare=SimpleNamespace(preprocess_pair=preprocess),
                utils=SimpleNamespace(safe_case_name=lambda p, v: p + '_' + v),
                source_hashes={'tools/prepare_cache.py': '2' * 64})
            with patch.object(v2, '_write_cache', fake_write), redirect_stdout(StringIO()):
                reference_path = v2.prepare(source_path, root / 'reference', raw, plan, plan_path, dependencies)
                reference = v2.read_json(reference_path)
                rebuilt = [recovery.initial_record(r, plan['configuration'], dependencies) for r in source['records']]
                for index, record in enumerate(rebuilt):
                    if not record['pending']:
                        rebuilt[index] = recovery.complete_record(record, root / 'recovery', raw, plan['configuration'], dependencies)
                actual = recovery.build_manifest(source, rebuilt, plan['configuration'], v2.sha256(source_path),
                                                  plan['_sha256'], dependencies)
            self.assertEqual(actual, reference)
            self.assertEqual([r['patient'] + '/' + r['visit'] for r in actual['records']],
                             [r['patient'] + '/' + r['visit'] for r in source['records']])
            self.assertEqual(len(opened), 362)  # 181 reference plus181 rebuilt metadata records.
            self.assertFalse(any(x.startswith('test_') for x in opened))
            self.assertEqual(actual['prepared_record_count'], 181)
            self.assertEqual(actual['protocol_sha256'], plan['_sha256'])


if __name__ == '__main__':
    unittest.main()
