"""Hand-ordered validation candidates and constructed CPU checkpoint flow."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from checkpoint_selection import checkpoint_candidate, checkpoint_key, is_better_checkpoint
from dataset import preprocess_pair
from train import execute_training, manifest_hash, validate_formal_plan


def candidate(epoch, dice, ce):
    return checkpoint_candidate(epoch, {"cross_entropy": ce, "dice_per_class": [0.0] * 5 + [dice]})


class TinyLogits(torch.nn.Module):
    """State changes are scripted below, not a claimed segmentation experiment."""

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, image):
        # Real convolution outputs own storage. Avoid a synthetic CPU view
        # into the parameter that would mutate the saved reference logits.
        return self.bias.reshape(1, 1, 1, 1).expand(image.shape[0], 6, *image.shape[-2:]).clone()


class CheckpointSelectionChecks(unittest.TestCase):
    def test_higher_id5_wins_even_with_worse_ce(self):
        earlier = candidate(1, 0.7, 0.1)
        later = candidate(2, 0.8, 0.3)
        self.assertTrue(is_better_checkpoint(later, earlier, "id5_dice"))
        self.assertFalse(is_better_checkpoint(later, earlier, "ce"))

    def test_exact_dice_tie_uses_lower_ce(self):
        self.assertTrue(is_better_checkpoint(candidate(3, 0.8, 0.2), candidate(2, 0.8, 0.3), "id5_dice"))
        self.assertFalse(is_better_checkpoint(candidate(4, 0.8, 0.4), candidate(2, 0.8, 0.3), "id5_dice"))

    def test_exact_dice_and_ce_ties_use_earliest_even_out_of_order(self):
        earlier, later = candidate(3, 0.8, 0.2), candidate(4, 0.8, 0.2)
        for metric in ("id5_dice", "ce"):
            with self.subTest(metric=metric):
                self.assertTrue(is_better_checkpoint(earlier, later, metric))
                self.assertFalse(is_better_checkpoint(later, earlier, metric))
                self.assertFalse(is_better_checkpoint(earlier, earlier, metric))

    def test_no_tolerance_silently_turns_a_small_improvement_into_a_tie(self):
        self.assertTrue(is_better_checkpoint(candidate(2, math.nextafter(0.8, 1.0), 0.9), candidate(1, 0.8, 0.1), "id5_dice"))
        self.assertTrue(is_better_checkpoint(candidate(2, 0.8, math.nextafter(0.2, 0.0)), candidate(1, 0.8, 0.2), "id5_dice"))

    def test_undefined_id5_is_allowed_only_for_ce(self):
        value = candidate(1, None, 0.2)
        self.assertEqual(checkpoint_key(value, "ce"), (0.2, 1))
        with self.assertRaisesRegex(ValueError, "defined validation ID5"):
            checkpoint_key(value, "id5_dice")
        two_class = checkpoint_candidate(1, {"cross_entropy": 0.2, "dice_per_class": [0.3, 0.4]})
        self.assertIsNone(two_class["validation_id5_dice"])

    def test_invalid_ce_or_dice_stops_selection(self):
        for ce in (None, math.nan, math.inf, -math.inf, -0.1):
            with self.subTest(ce=ce), self.assertRaisesRegex(ValueError, "validation CE"):
                candidate(1, 0.8, ce)
        for dice in (math.nan, math.inf, -math.inf, -0.1, 1.1):
            with self.subTest(dice=dice), self.assertRaisesRegex(ValueError, "ID5 Dice"):
                candidate(1, dice, 0.2)

    def test_unknown_rule_or_invalid_epoch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            checkpoint_key(candidate(1, 0.8, 0.2), "test_score")
        for epoch in (0, -1, 1.5, True):
            with self.subTest(epoch=epoch), self.assertRaisesRegex(ValueError, "positive integer"):
                candidate(epoch, 0.8, 0.2)

    def test_formal_plan_rejects_selection_drift(self):
        temporary_root = PROJECT_DIR.parents[1] / "outputs" / "test_tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            path = Path(directory) / "plan.json"
            train_pairs, val_pairs = [(Path("case_004"), Path("seg_004"))], [(Path("case_036"), Path("seg_036"))]
            plan = {
                "status": "approved_for_execution", "user_confirmation_received": True,
                "epochs_per_model": 20,
                "shared_protocol": {"models": ["baseline_unet", "improved_unet"], "batch_size": 4,
                    "model_config": {"base_channels": 16, "num_classes": 6}, "optimizer": {"learning_rate": 0.001}, "seed": 3710},
                "metrics_and_model_selection": {"checkpoint_selection": {"selection_metric": "id5_dice"}},
                "data": {"training": {"manifest_sha256": manifest_hash(train_pairs)}, "validation": {"manifest_sha256": manifest_hash(val_pairs)}},
                "source_sha256": {},
            }
            path.write_text(json.dumps(plan), encoding="utf-8")
            args = argparse.Namespace(formal_plan=path, epochs=20, batch_size=4, base_channels=16,
                num_classes=6, lr=0.001, seed=3710, device="cuda", train_limit=None, model="baseline_unet", selection_metric="id5_dice")
            self.assertEqual(validate_formal_plan(args, train_pairs, val_pairs), plan)
            args.selection_metric = "ce"
            with self.assertRaisesRegex(ValueError, "checkpoint selection differs"):
                validate_formal_plan(args, train_pairs, val_pairs)

    def test_constructed_cpu_flow_saves_three_choices_and_primary_alias(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        temporary_root = PROJECT_DIR.parents[1] / "outputs" / "test_tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        sample = preprocess_pair(torch.arange(256, dtype=torch.float32).reshape(1, 16, 16), torch.full((16, 16), 5, dtype=torch.long))
        train_pair = (Path("case_004_week_0_slice_12.nii.gz"), Path("seg_004_week_0_slice_12.nii.gz"))
        val_pair = (Path("case_036_week_0_slice_12.nii.gz"), Path("seg_036_week_0_slice_12.nii.gz"))
        values = [(0.7, 0.1), (0.8, 0.3), (0.8, 0.2), (0.8, 0.2)]
        for primary in ("id5_dice", "ce"):
            with self.subTest(primary=primary), tempfile.TemporaryDirectory(dir=temporary_root) as directory:
                args = argparse.Namespace(output_dir=Path(directory) / "new_run", data_root=Path("constructed"),
                    model="baseline_unet", epochs=4, batch_size=1, base_channels=2, num_classes=6,
                    lr=0.001, seed=3710, train_limit=None, device="cpu", progress_every=0,
                    reference_name=train_pair[0].name, formal_plan=None, selection_metric=primary)
                state = {"epoch": 0}

                def fake_epoch(model, loader, device, num_classes, optimizer=None, **kwargs):
                    if optimizer is not None:
                        state["epoch"] += 1
                        with torch.no_grad():
                            model.bias.fill_(state["epoch"])
                    dice, ce = values[state["epoch"] - 1]
                    return {"cross_entropy": ce, "dice_per_class": [0.0] * 5 + [dice],
                            "valid_pixels": 256, "optimizer_steps": int(optimizer is not None)}

                with patch("train.collect_pairs", side_effect=[[train_pair], [val_pair]]), \
                        patch("train.load_pair", return_value=sample), \
                        patch("train.HipMRIPairDataset", return_value=[sample]), \
                        patch("train.build_model", side_effect=lambda *args, **kwargs: TinyLogits()), \
                        patch("train.run_epoch", side_effect=fake_epoch), patch("train.plot_history"):
                    execute_training(args)
                report = json.loads((args.output_dir / "run_summary.json").read_text(encoding="utf-8"))
                expected_primary = 3 if primary == "id5_dice" else 1
                expected_epochs = {"id5_best.pt": 3, "ce_best.pt": 1, "last.pt": 4, "best.pt": expected_primary}
                for filename, expected_epoch in expected_epochs.items():
                    saved = torch.load(args.output_dir / filename, map_location="cpu", weights_only=True)
                    self.assertEqual(saved["epoch"], expected_epoch)
                    self.assertEqual(saved["model_state"]["bias"].item(), expected_epoch)
                    self.assertEqual(saved["selection_metric"], primary)
                    self.assertEqual(saved["validation_id5_dice"], values[expected_epoch - 1][0])
                alias = "id5_best.pt" if primary == "id5_dice" else "ce_best.pt"
                digest = lambda name: hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
                self.assertEqual(digest("best.pt"), digest(alias))
                self.assertEqual(report["status"], "passed")
                self.assertEqual(report["checkpoint_selection"]["primary"]["epoch"], expected_primary)
                self.assertEqual(report["checkpoint_selection"]["id5_best"]["epoch"], 3)
                self.assertEqual(report["checkpoint_selection"]["ce_best"]["epoch"], 1)
                self.assertEqual(report["checkpoint_selection"]["last"]["epoch"], 4)
                self.assertTrue(report["checkpoint_reload"]["passed"])
                self.assertEqual(report["checkpoint_reload"]["best_epoch"], expected_primary)
                self.assertFalse(report["test_split_read"])


if __name__ == "__main__":
    unittest.main()
