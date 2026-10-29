"""Synthetic data-safety tests; no patient data or GPU is required.

Run: python -m unittest -v test_dataset
"""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (HipMRIVolumeDataset, build_inventory, check_training_loader,
                     load_preprocessed_pair, make_patient_split,
                     read_patient_manifest, volume_to_slices)


class PreparedDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "semantic_MRs").mkdir()
        (self.root / "semantic_labels_only").mkdir()
        self.affine = np.diag([-2., -3., 4., 1.])  # LPS, deliberately not isotropic
        self.affine[:3, 3] = [22., 27., 0.]
        self.pixels = np.indices((12, 10, 8)).astype(np.float32)
        self.pixels = self.pixels[0] + 10*self.pixels[1] + 100*self.pixels[2]
        self.labels = np.broadcast_to(np.arange(12)[:, None, None] // 2, (12, 10, 8)).copy().astype(np.int16)
        for case in ['A001_Week0', 'A001_Week1', 'B002_Week0', 'C003_Week0']:
            self.write_case(case, self.pixels, self.labels)
        self.inventory = build_inventory(self.root)
        self.manifest = make_patient_split(self.inventory, val_patients=1, test_patients=1)
        self.path = self.root / "split.json"
        self.path.write_text(json.dumps(self.manifest))
        self.pair = self.inventory['pairs'][0]

    def write_case(self, case, pixels, labels, affine=None):
        nib.save(nib.Nifti1Image(pixels, self.affine), self.root/'semantic_MRs'/f'{case}_LFOV.nii.gz')
        nib.save(nib.Nifti1Image(labels, self.affine if affine is None else affine),
                 self.root/'semantic_labels_only'/f'{case}_SEMANTIC.nii.gz')

    def test_lps_axes_interpolation_and_world_grid(self):
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.root.rglob('*.nii.gz')}
        sample = load_preprocessed_pair(self.pair, (4, 5, 6))
        # Independent analytic ramp at half-voxel centres after LPS->RAS.
        z, y, x = np.indices((4, 5, 6))
        expected = (10.5-2*x) + 10*(8.5-2*y) + 100*(.5+2*z)
        expected = (expected-self.pixels.mean()) / self.pixels.std()
        np.testing.assert_allclose(sample['image'][0].numpy(), expected, atol=2e-6)
        expected_mask = torch.arange(5, -1, -1).expand(4, 5, 6)
        self.assertTrue(torch.equal(sample['mask'], expected_mask))
        self.assertTrue(torch.equal(sample['target'].argmax(0), expected_mask))
        self.assertTrue(torch.all(sample['target'].sum(0) == 1))
        # First output voxel corresponds to original XYZ=(10.5,8.5,.5).
        expected_world = self.affine @ [10.5, 8.5, .5, 1.]
        np.testing.assert_allclose(sample['affine_xyz'].numpy() @ [0, 0, 0, 1.], expected_world)
        for p, digest in before.items():
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), digest)

    def test_variable_depth_is_resized_without_truncation(self):
        x, y, z = np.indices((12, 10, 14)).astype(np.float32)
        pixels = x+10*y+100*z
        labels = np.broadcast_to(np.arange(12)[:, None, None]//2, pixels.shape).astype(np.int16)
        self.write_case(self.pair['case_id'], pixels, labels)
        sample = load_preprocessed_pair(self.pair, (4, 5, 6))
        last_z = (3+.5)*14/4-.5
        expected = (10.5+85+100*last_z-pixels.mean())/pixels.std()
        self.assertAlmostEqual(sample['image'][0, -1, 0, 0].item(), expected, places=5)
        self.assertEqual(sample['original_shape_xyz'].tolist(), [12, 10, 14])

    def test_constant_image_and_absent_classes(self):
        self.write_case(self.pair['case_id'], np.ones_like(self.pixels), np.full_like(self.labels, 5))
        sample = load_preprocessed_pair(self.pair, (4, 5, 6))
        self.assertTrue(torch.all(sample['image'] == 0))
        self.assertEqual(sample['target'].shape[0], 6)
        self.assertTrue(torch.all(sample['target'][5] == 1))

    def test_bad_masks_and_nonfinite_mri_are_rejected(self):
        for value in [0.5, 6, float('nan')]:
            with self.subTest(value=value):
                labels = self.labels.astype(np.float32)
                labels[1, 1, 1] = value
                self.write_case(self.pair['case_id'], self.pixels, labels)
                with self.assertRaises(ValueError): load_preprocessed_pair(self.pair)
        pixels = self.pixels.copy()
        pixels[0, 0, 0] = np.inf
        self.write_case(self.pair['case_id'], pixels, self.labels)
        with self.assertRaises(ValueError): load_preprocessed_pair(self.pair)

    def test_affine_mismatch_rejected(self):
        shifted = self.affine.copy()
        shifted[0, 3] += 1
        self.write_case(self.pair['case_id'], self.pixels, self.labels, shifted)
        with self.assertRaises(ValueError): load_preprocessed_pair(self.pair)

    def test_manifest_rejects_leakage_and_tampering(self):
        for corruption in ['leakage', 'hash', 'filename', 'week']:
            with self.subTest(corruption=corruption):
                m = copy.deepcopy(self.manifest)
                if corruption == 'leakage':
                    m['splits']['test'] = copy.deepcopy(m['splits']['train'])
                elif corruption == 'hash': m['inventory_identity_sha256'] = 'invalid'
                elif corruption == 'filename': m['splits']['train']['scans'][0]['mask_path'] = 'wrong.nii.gz'
                else: m['splits']['train']['scans'][0]['week'] = 99
                self.path.write_text(json.dumps(m))
                with self.assertRaises(ValueError): read_patient_manifest(self.path)

    def test_loader_opens_only_selected_split_and_slices_roundtrip(self):
        train_patients = self.manifest['splits']['train']['patient_ids']
        for name in ['test', 'validation']:
            for pair in self.manifest['splits'][name]['scans']:
                Path(pair['image_path']).unlink()
                Path(pair['mask_path']).unlink()
        ds = HipMRIVolumeDataset(self.path, data_root=self.root, target_shape=(4, 5, 6))
        sample = ds[0]
        self.assertIn(sample['patient_id'], train_patients)
        image, target = volume_to_slices(sample['image'], sample['target'])
        self.assertTrue(torch.equal(image.permute(1, 0, 2, 3), sample['image']))
        self.assertTrue(torch.equal(target.permute(1, 0, 2, 3), sample['target']))
        batch = next(iter(DataLoader(ds, batch_size=1)))
        self.assertEqual(tuple(batch['image'].shape), (1, 1, 4, 5, 6))
        self.assertEqual(tuple(batch['target'].shape), (1, 6, 4, 5, 6))

    def test_paired_flip_is_training_only(self):
        base = HipMRIVolumeDataset(self.path, data_root=self.root, target_shape=(4, 5, 6))[0]
        ds = HipMRIVolumeDataset(self.path, data_root=self.root, target_shape=(4, 5, 6), flip_probability=1)
        flipped = ds[0]
        self.assertTrue(torch.equal(flipped['image'], base['image'].flip(-1)))
        self.assertTrue(torch.equal(flipped['target'].argmax(0), flipped['mask']))
        np.testing.assert_allclose(flipped['affine_xyz'].numpy() @ [0, 0, 0, 1.],
                                   base['affine_xyz'].numpy() @ [5, 0, 0, 1.])
        with self.assertRaises(ValueError):
            HipMRIVolumeDataset(self.path, 'validation', self.root, flip_probability=.5)

    def test_training_only_diagnostic(self):
        output = self.root/'loader.json'
        report = check_training_loader(self.path, self.root, (4, 5, 6), 3, output)
        self.assertTrue(report['complete'] and report['passed'])
        self.assertFalse(report['is_training_result'])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
