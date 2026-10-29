"""Check label alignment, fixed channels, batching and validation."""
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import HipMRI2DDataset


class OneHotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.images = Path(self.temp.name) / 'images'
        self.masks = Path(self.temp.name) / 'masks'
        self.images.mkdir()
        self.masks.mkdir()
        self.labels = np.array([[5, 0, 3], [1, 4, 2]], dtype=np.int16)
        self.write_pair(self.labels)

    def write_pair(self, labels):
        image = np.arange(6, dtype=np.float32).reshape(2, 3)
        nib.save(nib.Nifti1Image(image, np.eye(4)), str(self.images / 'case_check.nii.gz'))
        nib.save(nib.Nifti1Image(labels, np.eye(4)), str(self.masks / 'seg_check.nii.gz'))

    def test_alignment_and_optional_normalization(self):
        for normalize in (False, True):
            with self.subTest(normalize=normalize):
                image, labels = HipMRI2DDataset(self.images, self.masks, normalize=normalize)[0]
                encoded_image, encoded = HipMRI2DDataset(
                    self.images, self.masks, normalize=normalize, one_hot=True
                )[0]
                self.assertTrue(torch.equal(image, encoded_image))
                self.assertEqual(labels.dtype, torch.long)
                self.assertTrue(torch.equal(labels, torch.tensor(self.labels, dtype=torch.long)))
                self.assertEqual(tuple(encoded.shape), (6, 2, 3))
                self.assertEqual(encoded.dtype, torch.float32)
                self.assertTrue(torch.all((encoded == 0) | (encoded == 1)).item())
                self.assertTrue(torch.equal(encoded.sum(dim=0), torch.ones(2, 3)))
                self.assertTrue(torch.equal(encoded.argmax(dim=0), labels))
                for row, col in np.ndindex(self.labels.shape):
                    self.assertEqual(encoded[int(self.labels[row, col]), row, col].item(), 1)

    def test_missing_classes_and_batching(self):
        self.write_pair(np.zeros((2, 3), dtype=np.int16))
        dataset = HipMRI2DDataset(self.images, self.masks, one_hot=True)
        images, encoded = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
        self.assertEqual(tuple(images.shape), (1, 1, 2, 3))
        self.assertEqual(tuple(encoded.shape), (1, 6, 2, 3))
        self.assertTrue(torch.all(encoded[:, 0] == 1).item())
        self.assertEqual(torch.count_nonzero(encoded[:, 1:]).item(), 0)

    def test_invalid_masks_still_rejected(self):
        for value in (-1, 6, 1.5, float('nan')):
            self.write_pair(np.full((2, 3), value, dtype=np.float64))
            for one_hot in (False, True):
                with self.subTest(value=value, one_hot=one_hot):
                    with self.assertRaises(ValueError):
                        HipMRI2DDataset(self.images, self.masks, one_hot=one_hot)[0]


if __name__ == '__main__':
    unittest.main()
