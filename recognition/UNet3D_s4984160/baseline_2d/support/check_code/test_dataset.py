"""Synthetic NIfTI checks; no patient data required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('hipmri_2d_dataset', PROJECT / 'dataset.py')
dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'keras_slices_data'
        for split in ('train', 'validate', 'test'):
            (self.root / f'keras_slices_{split}').mkdir(parents=True)
            (self.root / f'keras_slices_seg_{split}').mkdir()
        self.paths = {}
        for split, patient in [('train', '004'), ('validate', '036'), ('test', '040')]:
            image = np.arange(256*128, dtype=np.float32).reshape(256,128)
            mask = (np.indices(image.shape)[0] % 6).astype(np.int16)
            self.paths[patient] = self.write(split, patient, image, mask)

    def write(self, split, patient, image, mask, index=0):
        image_path = self.root / f'keras_slices_{split}' / f'case_{patient}_week_0_slice_{index}.nii.gz'
        mask_path = self.root / f'keras_slices_seg_{split}' / f'seg_{patient}_week_0_slice_{index}.nii.gz'
        nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
        nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
        return image_path, mask_path

    def test_padding_normalization_dtypes_and_metadata(self):
        item = dataset.HipMRI2DDataset(self.root.parent, ['M004'])
        image, mask, meta = item[0]
        self.assertEqual(tuple(image.shape), (1,256,144))
        self.assertEqual(tuple(mask.shape), (256,144))
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(mask.dtype, torch.int64)
        self.assertAlmostEqual(float(image[:,:,:128].mean()), 0., places=5)
        self.assertAlmostEqual(float(image[:,:,:128].std(unbiased=False)), 1., places=5)
        self.assertTrue((image[:,:,128:] == 0).all())
        self.assertTrue((mask[:,128:] == 255).all())
        self.assertEqual(int((mask != 255).sum()), 256*128)
        self.assertEqual(meta['original_w'], 128)
        self.assertEqual(meta['source_index'], 98)

    def test_manifest_overrides_original_folder_split(self):
        manifest = self.root / 'custom_split.json'
        manifest.write_text(json.dumps(dict(train=['040'], validation=['004'], test=['036'])))
        loaders = dataset.get_dataloaders(self.root, batch_size=1, split_manifest=manifest)
        ids = [next(iter(loader))[2]['patient_id'][0] for loader in loaders]
        self.assertEqual(ids, ['040','004','036'])

    def test_wide_slice_and_singleton_are_preserved(self):
        image = np.ones((256,144,1), dtype=np.float32)
        mask = np.full((256,144,1), 5, dtype=np.int16)
        self.write('train', '004', image, mask)
        x, y, meta = dataset.HipMRI2DDataset(self.root, ['004'])[0]
        self.assertTrue(torch.isfinite(x).all())
        self.assertTrue((y == 5).all())
        self.assertEqual(meta['pad_right'], 0)

    def test_missing_mask_is_an_error(self):
        self.paths['004'][1].unlink()
        with self.assertRaises(FileNotFoundError):
            dataset.get_data_pairs(self.root)

    def test_fractional_labels_are_not_silently_truncated(self):
        image = np.zeros((256,128), dtype=np.float32)
        mask = np.zeros_like(image)
        mask[5,5] = 1.5
        self.write('train', '004', image, mask)
        with self.assertRaisesRegex(ValueError, 'integer labels'):
            dataset.HipMRI2DDataset(self.root, ['004'])[0]

    def test_nonfinite_input_and_multislice_volume_are_rejected(self):
        mask = np.zeros((256,128), dtype=np.int16)
        image = np.zeros(mask.shape, dtype=np.float32)
        image[0,0] = np.nan
        self.write('train', '004', image, mask)
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            dataset.HipMRI2DDataset(self.root, ['004'])[0]
        self.write('train', '004', np.zeros((256,128,2), dtype=np.float32), mask)
        with self.assertRaisesRegex(ValueError, '2D slice'):
            dataset.HipMRI2DDataset(self.root, ['004'])[0]

    def test_overlapping_patients_are_rejected(self):
        manifest = self.root / 'invalid_split.json'
        manifest.write_text(json.dumps(dict(train=['004'], validation=['004'], test=['040'])))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            dataset.load_patient_split(manifest)

    def test_duplicates_and_numeric_slice_sort(self):
        image, mask = self.paths['004']
        a, b = np.asarray(nib.load(image).dataobj), np.asarray(nib.load(mask).dataobj)
        self.write('train', '004', a, b, index=10)
        self.write('train', '004', a, b, index=2)
        items = dataset.HipMRI2DDataset(self.root, ['004'])
        self.assertEqual([r['slice_id'] for r in items.samples], [0,2,10])
        self.write('validate', '004', a, b)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            dataset.get_data_pairs(self.root)

    def test_joint_transform_is_applied(self):
        def flip(image, mask):
            return image.flip(1), mask.flip(0)
        _, mask, _ = dataset.HipMRI2DDataset(self.root, ['004'], transform=flip)[0]
        raw = np.asarray(nib.load(self.paths['004'][1]).dataobj)
        self.assertTrue(np.array_equal(mask[:,:128].numpy(), raw[::-1]))

    def test_default_manifest_matches_existing_3d_split(self):
        # Also support a frozen run snapshot nested inside the 2D directory.
        path = next(parent / 'UNet3D_s4984160' / 'dataset.py'
                    for parent in PROJECT.parents
                    if (parent / 'UNet3D_s4984160' / 'dataset.py').is_file())
        spec3d = importlib.util.spec_from_file_location('existing_hipmri3d', path)
        module3d = importlib.util.module_from_spec(spec3d)
        spec3d.loader.exec_module(module3d)
        document = json.loads(dataset.DEFAULT_SPLIT.read_text())
        patients = [p for group in document['three_d_patient_ids'].values() for p in group]
        pairs = [(p, None, None) for p in patients]
        groups3d = module3d.split_by_patient(pairs, seed=42)
        groups2d = dataset.load_patient_split()
        for name, group in zip(('train','validation','test'), groups3d):
            self.assertEqual({p[0][1:] for p in group}, groups2d[name])


if __name__ == '__main__':
    unittest.main()
