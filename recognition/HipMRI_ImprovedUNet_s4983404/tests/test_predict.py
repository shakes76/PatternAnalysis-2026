"""Exercise saved-weight inference without requiring ground-truth masks."""
from pathlib import Path
import argparse
import sys
import tempfile
import unittest
from unittest.mock import patch
import nibabel as nib
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import load_pair, normalize_image
from modules import build_model
from predict import main, prediction_paths


class PredictionTests(unittest.TestCase):
    def test_unlabeled_and_labeled_predictions_match(self) -> None:
        torch.set_num_threads(2)
        for model_name in ('basic', 'improved'):
            with self.subTest(model=model_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pixels = np.arange(64*32, dtype=np.float32).reshape(64, 32, 1)
                affine = np.diag([1.5, 2.0, 3.0, 1.0])
                image_path = root/'scan.nii.gz'
                nib.save(nib.Nifti1Image(pixels, affine), image_path)
                model = build_model(model_name, 4).eval()
                checkpoint = root/'best.pt'
                torch.save({'config': {'model': model_name, 'width': 4},
                            'state_dict': model.state_dict()}, checkpoint)
                command = ['predict.py', '--checkpoint', str(checkpoint), '--device', 'cpu',
                           '--input', str(image_path), '--count', '1']
                with patch.object(sys, 'argv', command + ['--output', str(root/'unlabeled')]):
                    main()
                labels = root/'labels'
                labels.mkdir()
                nib.save(nib.Nifti1Image(np.zeros_like(pixels), affine), labels/image_path.name)
                with patch.object(sys, 'argv', command + ['--labels', str(labels),
                                                         '--output', str(root/'labeled')]):
                    main()
                normalized, _ = load_pair(image_path, labels/image_path.name)
                original = np.asarray(nib.load(image_path).dataobj, dtype=np.float32).squeeze()
                original = (original-original.mean()) / max(float(original.std()), 1e-6)
                np.testing.assert_array_equal(normalized[0].numpy(), original)
                with torch.inference_mode():
                    expected = model(normalized.unsqueeze(0)).argmax(1)[0].numpy()
                for mode in ('unlabeled', 'labeled'):
                    mask = nib.load(root/mode/'pred_scan.nii.gz')
                    self.assertEqual(mask.shape, pixels.shape)
                    self.assertEqual(mask.get_data_dtype(), np.dtype('uint8'))
                    np.testing.assert_array_equal(mask.affine, affine)
                    np.testing.assert_array_equal(np.asarray(mask.dataobj).squeeze(), expected)
                    with Image.open(root/mode/'scan.png') as image:
                        image.verify()

    def test_directory_selection_and_missing_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('one.nii', 'two.nii.gz', 'notes.txt'):
                (root/name).touch()
            args = argparse.Namespace(input=root, labels=None, count=5)
            self.assertEqual([path.name for path, _ in prediction_paths(args)],
                             ['one.nii', 'two.nii.gz'])
            args.labels = root/'missing'
            with self.assertRaises(FileNotFoundError):
                prediction_paths(args)
            args.labels, args.count = None, 0
            with self.assertRaises(ValueError):
                prediction_paths(args)
            # Single non-NIfTI file, absent input and orphan labels must all fail early.
            args = argparse.Namespace(input=root/'notes.txt', labels=None, count=1)
            with self.assertRaises(ValueError):
                prediction_paths(args)
            args = argparse.Namespace(input=root/'absent', labels=None, count=1)
            with self.assertRaises(FileNotFoundError):
                prediction_paths(args)
            args = argparse.Namespace(input=None, labels=root, count=1)
            with self.assertRaises(ValueError):
                prediction_paths(args)

    def test_normalization_rejects_invalid_inputs(self) -> None:
        for image in (np.zeros((32, 32, 4)), np.full((32, 32), np.nan)):
            with self.assertRaises(ValueError):
                normalize_image(image)
        self.assertTrue(torch.equal(normalize_image(np.ones((32, 32))),
                                    torch.zeros(1, 32, 32)))


if __name__ == '__main__':
    unittest.main()
