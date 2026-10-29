"""Verify paired flips, training guard and preprocessing compatibility."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import HipMRI2DDataset


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.images = Path(self.temp.name) / 'images'
        self.masks = Path(self.temp.name) / 'masks'
        self.images.mkdir()
        self.masks.mkdir()
        self.labels = torch.tensor([[0, 5, 1], [4, 2, 3]], dtype=torch.long)
        for index in range(2):
            labels = self.labels.numpy().astype(np.int16)
            image = labels.astype(np.float32)
            nib.save(nib.Nifti1Image(image, np.eye(4)), str(self.images / f'case_{index}.nii.gz'))
            nib.save(nib.Nifti1Image(labels, np.eye(4)), str(self.masks / f'seg_{index}.nii.gz'))

    def test_training_guard(self):
        with self.assertRaisesRegex(ValueError, 'training=True'):
            HipMRI2DDataset(self.images, self.masks, augment=True)
        # A training dataset may also explicitly disable augmentation.
        HipMRI2DDataset(self.images, self.masks, training=True, augment=False)

    def test_both_random_branches_preserve_alignment(self):
        dataset = HipMRI2DDataset(self.images, self.masks, training=True, augment=True)
        for draw, expected in ((0.0, torch.flip(self.labels, (-1,))), (0.9, self.labels)):
            with self.subTest(draw=draw):
                with patch('dataset.torch.rand', return_value=torch.tensor(draw)):
                    image, mask = dataset[0]
                self.assertTrue(torch.equal(mask, expected))
                self.assertTrue(torch.equal(image[0], expected.float()))
                self.assertEqual(tuple(image.shape), (1, 2, 3))
                self.assertEqual(mask.dtype, torch.long)

    def test_normalized_one_hot_batch(self):
        raw = HipMRI2DDataset(self.images, self.masks, normalize=True)
        reference, _ = raw[0]
        dataset = HipMRI2DDataset(
            self.images, self.masks, normalize=True, one_hot=True,
            training=True, augment=True,
        )
        with patch('dataset.torch.rand', return_value=torch.tensor(0.0)):
            images, targets = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
        self.assertEqual(tuple(images.shape), (2, 1, 2, 3))
        self.assertEqual(tuple(targets.shape), (2, 6, 2, 3))
        self.assertEqual(targets.dtype, torch.float32)
        expected = torch.flip(self.labels, (-1,))
        for index in range(2):
            torch.testing.assert_close(images[index], torch.flip(reference, (-1,)))
            self.assertTrue(torch.equal(targets[index].argmax(dim=0), expected))
        self.assertTrue(torch.all(targets.sum(dim=1) == 1).item())

    def test_disabled_augmentation_uses_no_random_draw(self):
        for training in (False, True):
            dataset = HipMRI2DDataset(self.images, self.masks, training=training)
            with patch('dataset.torch.rand', side_effect=AssertionError('Unexpected random draw')):
                image, mask = dataset[0]
            self.assertTrue(torch.equal(image[0], self.labels.float()))
            self.assertTrue(torch.equal(mask, self.labels))

    def test_seed_reproducibility(self):
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        dataset = HipMRI2DDataset(self.images, self.masks, training=True, augment=True)
        torch.manual_seed(3710)
        first = [dataset[0] for _ in range(8)]
        torch.manual_seed(3710)
        second = [dataset[0] for _ in range(8)]
        for (image_a, mask_a), (image_b, mask_b) in zip(first, second):
            self.assertTrue(torch.equal(image_a, image_b))
            self.assertTrue(torch.equal(mask_a, mask_b))


if __name__ == '__main__':
    unittest.main()
