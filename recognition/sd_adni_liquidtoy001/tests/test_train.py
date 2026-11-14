"""CPU tests of train.fit: records, early stopping and reloadable weights."""

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

if importlib.util.find_spec("diffusers") is None:
    raise unittest.SkipTest("needs diffusers and peft (training environment)")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules import configure_trainable, load_weights  # noqa: E402
from test_modules import sd_schedule, tiny_embeddings, tiny_unet  # noqa: E402
from train import (  # noqa: E402
    EncodedSplit,
    TrainConfig,
    exclude_defective_scans,
    fit,
)


class FitTest(unittest.TestCase):
    def test_fit_writes_records_and_weights_that_reload(self):
        unet = tiny_unet()
        parameters = configure_trainable(unet, "unet")
        generator = torch.Generator().manual_seed(4)
        split = EncodedSplit(
            torch.randn((8, 4, 8, 8), generator=generator),
            torch.full((8, 4, 8, 8), 0.1),
            torch.tensor([0, 1] * 4),
        )
        config = TrainConfig(
            epochs=2,
            batch_size=4,
            lr=1e-3,
            warmup_steps=2,
            log_every=1,
            save_epochs=(1,),
        )
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            summary = fit(
                config,
                unet,
                parameters,
                sd_schedule(),
                tiny_embeddings(),
                split,
                split,
                0.18215,
                output,
                output / "weights",
            )
            self.assertEqual(summary["steps"], 4)
            self.assertEqual(summary["stop_reason"], "completed")
            with (output / "epochs.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["epoch"] for r in rows], ["1", "2"])
            self.assertTrue((output / "curves.png").stat().st_size > 0)
            self.assertEqual(
                len(summary["weights"]), 2
            )  # epoch 1 by request, 2 as last
            # A differently initialised UNet must end up holding the saved weights.
            fresh = tiny_unet(seed=7)
            self.assertEqual(load_weights(fresh, Path(summary["weights"][-1])), "unet")
            untouched = tiny_unet(seed=7)
            self.assertTrue(
                any(
                    not torch.equal(a, b)
                    for a, b in zip(fresh.parameters(), untouched.parameters())
                )
            )

    def test_max_steps_stops_inside_an_epoch_and_still_saves(self):
        unet = tiny_unet()
        parameters = configure_trainable(unet, "lora", lora_rank=2)
        split = EncodedSplit(
            torch.randn(8, 4, 8, 8), torch.zeros(8, 4, 8, 8), torch.tensor([0, 1] * 4)
        )
        config = TrainConfig(
            epochs=3, batch_size=2, max_steps=3, log_every=50, save_epochs=()
        )
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            summary = fit(
                config,
                unet,
                parameters,
                sd_schedule(),
                tiny_embeddings(),
                split,
                split,
                0.18215,
                output,
                output / "weights",
                weight_prefix="lora",
            )
            self.assertEqual(
                (summary["steps"], summary["stop_reason"]), (3, "max_steps")
            )
            self.assertEqual(len(summary["weights"]), 1)
            fresh = tiny_unet()
            self.assertEqual(load_weights(fresh, Path(summary["weights"][0])), "lora")


class ExclusionTest(unittest.TestCase):
    def test_every_slice_of_a_flagged_scan_is_removed_in_order(self):
        rows = [{"scan_id": s} for s in ("1", "1", "2", "2", "3")]
        split = EncodedSplit(
            torch.arange(5.0).view(5, 1, 1, 1),
            torch.ones(5, 1, 1, 1),
            torch.tensor([0, 0, 1, 1, 1]),
            torch.tensor([0.4, 0.3, 0.01, 0.5, 0.2]),  # scan 2 has an empty slice
        )
        kept, flagged = exclude_defective_scans(split, rows)
        self.assertEqual(flagged, {"2"})
        self.assertEqual(kept.mean.flatten().tolist(), [0.0, 1.0, 4.0])
        self.assertEqual(kept.label_index.tolist(), [0, 0, 1])
        self.assertTrue(torch.allclose(kept.foreground, torch.tensor([0.4, 0.3, 0.2])))


class ValidationSeedTest(unittest.TestCase):
    def validation_loss_after_one_epoch(self, seed, val_seed):
        # lr 0 leaves the weights untouched, so only the validation draws can
        # change the reported validation loss.
        unet = tiny_unet()
        parameters = configure_trainable(unet, "unet")
        generator = torch.Generator().manual_seed(4)
        split = EncodedSplit(
            torch.randn((8, 4, 8, 8), generator=generator),
            torch.full((8, 4, 8, 8), 0.1),
            torch.tensor([0, 1] * 4),
        )
        config = TrainConfig(
            epochs=1, batch_size=4, lr=0.0, save_epochs=(), seed=seed, val_seed=val_seed
        )
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            fit(
                config,
                unet,
                parameters,
                sd_schedule(),
                tiny_embeddings(),
                split,
                split,
                0.18215,
                output,
                output / "weights",
            )
            with (output / "epochs.csv").open() as stream:
                return float(next(csv.DictReader(stream))["val_loss_live"])

    def test_training_seed_does_not_move_validation_draws(self):
        first = self.validation_loss_after_one_epoch(seed=3710, val_seed=3710)
        replicate = self.validation_loss_after_one_epoch(seed=4710, val_seed=3710)
        other_draws = self.validation_loss_after_one_epoch(seed=3710, val_seed=11)
        self.assertEqual(first, replicate)
        self.assertNotEqual(first, other_draws)


if __name__ == "__main__":
    unittest.main()
