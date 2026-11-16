"""Synthetic NIfTI fixtures check pairing, class stability, alignment and errors."""
from pathlib import Path
import sys
import tempfile
import unittest
import nibabel as nib
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import HipMRISlices, check_patient_splits, load_pair, collate_slices
from metrics import segmentation_loss, confusion_counts


class DatasetTests(unittest.TestCase):
    def test_padding_excluded_from_loss_and_metrics(self) -> None:
        target = torch.nn.functional.one_hot(torch.zeros(8, 8, dtype=torch.long), 6)
        target = target.permute(2, 0, 1).float()
        batch = [(torch.zeros(1, 8, 8), target, 'small'),
                 (torch.zeros(1, 8, 10), torch.nn.functional.pad(target, (0, 2)), 'wide')]
        images, targets, names = collate_slices(batch)
        self.assertEqual(tuple(images.shape), (2, 1, 8, 10))
        logits = torch.randn_like(targets)
        original_loss = segmentation_loss(logits, targets)
        original_counts = confusion_counts(logits, targets)
        logits[:, 5, :, 8:] = 1000
        torch.testing.assert_close(original_loss, segmentation_loss(logits, targets))
        self.assertTrue(torch.equal(original_counts, confusion_counts(logits, targets)))
        self.assertEqual(int(original_counts.sum()), 128)

    def test_fixed_channels_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, patient in (('train', '004'), ('validate', '036'), ('test', '040')):
                for prefix in ('keras_slices_', 'keras_slices_seg_'):
                    (root/f'{prefix}{split}').mkdir()
                image_path = root/f'keras_slices_{split}'/f'case_{patient}_week_0_slice_0.nii.gz'
                label_path = root/f'keras_slices_seg_{split}'/f'seg_{patient}_week_0_slice_0.nii.gz'
                image = np.ones((32, 16), dtype=np.float32)
                labels = np.zeros_like(image)
                labels[8:10, 4:6] = 5
                nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
                nib.save(nib.Nifti1Image(labels, np.eye(4)), label_path)
            self.assertEqual(check_patient_splits(root)['train'], ['004'])
            dataset = HipMRISlices(root, 'train')
            image, target, _ = dataset[0]
            cached_image, cached_target, _ = HipMRISlices(root, 'train', cache=True)[0]
            torch.testing.assert_close(image, cached_image, rtol=0, atol=0)
            torch.testing.assert_close(target, cached_target, rtol=0, atol=0)
            self.assertEqual(tuple(target.shape), (6, 32, 16))
            self.assertTrue(torch.isfinite(image).all())
            torch.testing.assert_close(target.sum(0), torch.ones(32, 16))
            with self.assertRaises(IndexError):
                dataset[1]
            with self.assertRaises(ValueError):
                HipMRISlices(root, 'invalid')
            with self.assertRaises(ValueError):
                HipMRISlices(root, 'train', limit=0)
            shifted = np.eye(4)
            shifted[0, 3] = 2
            nib.save(nib.Nifti1Image(labels, shifted), label_path)
            with self.assertRaises(ValueError):
                load_pair(image_path, label_path)
            labels[0, 0] = 6
            nib.save(nib.Nifti1Image(labels, np.eye(4)), label_path)
            with self.assertRaises(ValueError):
                load_pair(image_path, label_path)
            with self.assertRaises(FileNotFoundError):
                HipMRISlices(root/'missing', 'train')


if __name__ == '__main__':
    unittest.main()
