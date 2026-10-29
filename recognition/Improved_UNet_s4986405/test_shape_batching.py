"""Regression checks for heterogeneous native slice sizes and pooled metrics."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import HipMRI2DDataset
from shape_batching import ShapeBatchSampler
from train import run_epoch


class ShapeBatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        images, masks = self.root / 'images', self.root / 'masks'
        images.mkdir()
        masks.mkdir()
        self.shapes = [(32, 32)] * 3 + [(32, 48)] * 2 + [(48, 32)]
        for index, shape in enumerate(self.shapes):
            suffix = f'{index:03d}_week_0_slice_0.nii.gz'
            image = np.full(shape, index + 0.25, dtype=np.float32)
            mask = np.full(shape, index, dtype=np.int16)
            nib.save(nib.Nifti1Image(image, np.eye(4)), images / ('case_' + suffix))
            nib.save(nib.Nifti1Image(mask, np.eye(4)), masks / ('seg_' + suffix))
        self.dataset = HipMRI2DDataset(images, masks, one_hot=True)

    def test_all_indices_covered_once_with_small_final_batches(self):
        sampler = ShapeBatchSampler(self.dataset, batch_size=2)
        batches = list(sampler)
        self.assertEqual(len(sampler), 4)
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(6)))
        self.assertEqual(sorted(map(len, batches)), [1, 1, 2, 2])
        self.assertEqual(list(sampler), batches)
        self.assertEqual(sampler.shape_counts, [
            {'shape': [32, 32], 'samples': 3},
            {'shape': [32, 48], 'samples': 2},
            {'shape': [48, 32], 'samples': 1},
        ])
        for batch in batches:
            self.assertEqual(len({self.shapes[index] for index in batch}), 1)

    def test_loader_preserves_native_pixels_and_one_hot_masks(self):
        sampler = ShapeBatchSampler(self.dataset, batch_size=2)
        loader = DataLoader(self.dataset, batch_sampler=sampler, num_workers=0)
        seen = []
        for images, targets in loader:
            self.assertEqual(images.shape[-2:], targets.shape[-2:])
            self.assertEqual(targets.shape[1], 6)
            self.assertTrue(torch.all(targets.sum(dim=1) == 1).item())
            for image, target in zip(images, targets):
                index = int(image[0, 0, 0].item() - 0.25)
                seen.append(index)
                self.assertEqual(tuple(image.shape), (1, *self.shapes[index]))
                self.assertTrue(torch.all(image == index + 0.25).item())
                self.assertTrue(torch.all(target.argmax(dim=0) == index).item())
        self.assertEqual(sorted(seen), list(range(6)))

    def test_seeded_shuffle_is_reproducible_and_keeps_every_sample(self):
        first = ShapeBatchSampler(self.dataset, 2, shuffle=True,
                                  generator=torch.Generator().manual_seed(3710))
        second = ShapeBatchSampler(self.dataset, 2, shuffle=True,
                                   generator=torch.Generator().manual_seed(3710))
        for _ in range(3):
            left, right = list(first), list(second)
            self.assertEqual(left, right)
            self.assertEqual(sorted(index for batch in left for index in batch), list(range(6)))
            for batch in left:
                self.assertEqual(len({self.shapes[index] for index in batch}), 1)

    def test_invalid_batch_sizes_and_non_2d_images_rejected(self):
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                ShapeBatchSampler(self.dataset, size)
        path = self.root / 'volume.nii.gz'
        nib.save(nib.Nifti1Image(np.zeros((16, 16, 2), dtype=np.float32), np.eye(4)), path)
        with self.assertRaisesRegex(ValueError, '2D image'):
            ShapeBatchSampler(SimpleNamespace(pairs=[(path, path)]), 2)

    def test_dataset_still_rejects_mismatched_masks(self):
        _, mask_path = self.dataset.pairs[0]
        nib.save(nib.Nifti1Image(np.zeros((32, 48), dtype=np.int16), np.eye(4)), mask_path)
        loader = DataLoader(self.dataset, batch_sampler=ShapeBatchSampler(self.dataset, 2))
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            next(iter(loader))

    def test_epoch_pools_only_original_pixels_across_different_shapes(self):
        loader = DataLoader(self.dataset, batch_sampler=ShapeBatchSampler(self.dataset, 2))
        model = nn.Conv2d(1, 6, kernel_size=1)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        result = run_epoch(model, loader, 'cpu')
        expected = torch.zeros(6, 6, dtype=torch.long)
        for index, (height, width) in enumerate(self.shapes):
            expected[index, 0] = height * width
        self.assertEqual(result['samples'], 6)
        self.assertEqual(result['batches'], 4)
        self.assertEqual(result['pixels'], 7680)
        self.assertTrue(torch.equal(result['confusion'], expected))
        self.assertTrue(torch.isfinite(torch.tensor(result['loss'])).item())


if __name__ == '__main__':
    unittest.main()
