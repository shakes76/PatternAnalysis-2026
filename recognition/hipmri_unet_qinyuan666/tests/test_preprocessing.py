"""Geometry, class preservation and old-checkpoint preprocessing compatibility."""

from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import resize_pair
from predict import _validate_checkpoint
from train import parse_args


class PreprocessingTests(unittest.TestCase):
    def test_course_shapes_are_padded_without_resampling_at_256(self):
        for width in (128, 144):
            with self.subTest(width=width):
                image = torch.arange(256*width, dtype=torch.float32).reshape(1, 256, width)
                mask = torch.arange(256*width).reshape(256, width) % 6
                x, y = resize_pair(image, mask, 256, "pad")
                left = 64 if width == 128 else 56
                self.assertEqual(tuple(x.shape), (1, 256, 256))
                self.assertTrue(torch.equal(x[:, :, left:left+width], image))
                self.assertTrue(torch.equal(y[:, left:left+width], mask))
                self.assertEqual(torch.count_nonzero(y[:, :left]).item(), 0)
                self.assertEqual(torch.count_nonzero(y[:, left+width:]).item(), 0)
                for label in range(1, 6):
                    self.assertEqual(int((y == label).sum()), int((mask == label).sum()))

    def test_proportional_resize_retains_integer_label_ids(self):
        x = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
        y = torch.zeros((4, 8), dtype=torch.long)
        y[:, 4:] = 5
        out_x, out_y = resize_pair(x, y, 16, "pad")
        self.assertEqual(tuple(out_y.shape), (16, 16))
        self.assertEqual(set(out_y.unique().tolist()), {0, 5})
        self.assertTrue(torch.all(out_y[4:12, 8:] == 5))
        self.assertEqual(int(out_y[:4].sum()) + int(out_y[12:].sum()), 0)
        self.assertTrue(torch.isfinite(out_x).all())

    def test_legacy_stretch_matches_previous_transform(self):
        x = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
        y = torch.arange(32).reshape(4, 8) % 6
        a, b = resize_pair(x, y, 16, "stretch")
        self.assertTrue(torch.equal(a, F.interpolate(x[None], (16,16), mode="bilinear", align_corners=False)[0]))
        self.assertTrue(torch.equal(b, F.interpolate(y[None,None].float(), (16,16), mode="nearest")[0,0].long()))

    def test_checkpoint_policy_is_explicit_and_old_default_is_stretch(self):
        config = {"model":"unet", "base_channels":2, "image_size":32, "num_classes":2,
                  "label_values":[0,1], "synthetic":True, "prostate_label":None, "seed":1, "samples":2}
        payload = {"format_version":1, "config":config, "model_state":{"dummy":torch.zeros(1)}, "epoch":1}
        self.assertEqual(_validate_checkpoint(payload, True)[0]["resize_mode"], "stretch")
        config["resize_mode"] = "pad"
        self.assertEqual(_validate_checkpoint(payload, True)[0]["resize_mode"], "pad")
        config["resize_mode"] = "unknown"
        with self.assertRaisesRegex(ValueError, "resize_mode"):
            _validate_checkpoint(payload, True)

    def test_new_training_cli_defaults_to_pad(self):
        args = parse_args(["--model", "unet", "--synthetic", "--output", "unused"])
        self.assertEqual(args.resize_mode, "pad")


if __name__ == "__main__":
    unittest.main()
