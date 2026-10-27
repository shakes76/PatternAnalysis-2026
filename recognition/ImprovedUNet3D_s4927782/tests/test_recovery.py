"""Recovery policy contracts without MRI, weights, training, or remote access."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from experiments import appearance as old
from experiments import recovery as rec
from experiments import train_recovery as trainer


class RecoveryPolicyTests(unittest.TestCase):
    def test_gamma_and_polarity_use_the_original_independent_case_epoch_streams(self):
        observations = []
        for case in ("B006/Week0", "K019/Week1", "T014/Week3"):
            for epoch in range(1, 9):
                expected = old.deterministic_params(42, epoch, case, "gamma_polarity")
                clean = rec.recovery_params(42, epoch, case, "clean_adapt")
                rehearsal = rec.recovery_params(42, epoch, case, "polarity_rehearsal")
                for key in ("gamma", "gamma_applied", "gamma_draw"):
                    self.assertEqual(clean[key], expected[key])
                    self.assertEqual(rehearsal[key], expected[key])
                draw = float(old._keyed_rng(42, epoch, case, "polarity").random())
                self.assertFalse(clean["polarity"])
                self.assertEqual(rehearsal["polarity"], draw < .10)
                self.assertFalse(rehearsal["polarity"] and not expected["polarity"])
                observations.append(rehearsal["polarity"])
                self.assertEqual(rehearsal, rec.recovery_params(42, epoch, case, "polarity_rehearsal"))
        self.assertIn(True, observations)
        self.assertIn(False, observations)

    def test_keyed_draws_leave_global_numpy_rng_unchanged(self):
        np.random.seed(81)
        before = np.random.get_state()
        rec.recovery_params(42, 4, "B006/Week0", "polarity_rehearsal")
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_wrapper_keeps_original_training_flips_labels_geometry_and_input(self):
        image = np.array([0, -4, -1, -.1, .1, 1, 2, 4], dtype=np.float32).reshape(1, 2, 2, 2)
        before = image.copy()
        mask = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
        spacing = np.array([2, 1, 1], dtype=np.float32)
        sample = {"image": image, "mask": mask, "spacing": spacing, "patient": "B006", "visit": "Week0"}
        class Base:
            records, split, augment = [{"patient": "B006"}], "train", True
            def __len__(self): return 1
            def __getitem__(self, index): return sample
        fake = types.SimpleNamespace(HipMRIVolumeDataset=mock.Mock(return_value=Base()))
        with mock.patch.dict(sys.modules, {"dataset": fake}):
            wrapper = rec.RecoveryTrainingDataset("unused", "polarity_rehearsal", seed=42)
        fake.HipMRIVolumeDataset.assert_called_once_with("unused", "train", augment=True)
        wrapper.set_epoch(4)
        result = wrapper[0]
        params = rec.recovery_params(42, 4, "B006/Week0", "polarity_rehearsal")
        np.testing.assert_array_equal(result["image"], old.augment_image(image, params["gamma"], params["polarity"]))
        np.testing.assert_array_equal(image, before)
        np.testing.assert_array_equal(result["image"] != 0, image != 0)
        self.assertIs(result["mask"], mask)
        self.assertIs(result["spacing"], spacing)
        self.assertEqual((result["patient"], result["visit"]), ("B006", "Week0"))
        self.assertEqual(wrapper.records, Base.records)
        self.assertEqual(len(wrapper), 1)
        self.assertEqual(wrapper.policy_counts, {"noninverted": int(not params["polarity"]),
                                               "inverted": int(params["polarity"]),
                                               "gamma_applied": int(params["gamma_applied"])})
        wrapper.reset_counts()
        self.assertEqual(wrapper.policy_counts, {"noninverted": 0, "inverted": 0, "gamma_applied": 0})

    def test_nontraining_split_and_invalid_policy_inputs_are_rejected(self):
        for split in ("validation", "test"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                rec.RecoveryTrainingDataset("unused", "clean_adapt", split=split)
        for seed, epoch, case, variant in ((42, 0, "B006/Week0", "clean_adapt"),
                                          (2**32, 1, "B006/Week0", "clean_adapt"),
                                          (42, 1, "", "clean_adapt"), (42, 1, "case", "unplanned")):
            with self.subTest(seed=seed, epoch=epoch, case=case, variant=variant), self.assertRaises(ValueError):
                rec.recovery_params(seed, epoch, case, variant)


class RecoveryMetadataTests(unittest.TestCase):
    def appearance(self):
        return {"experiment_id": "appearance_augmentation_v2", "training_conditions": {
            "model": "improved3d", "seed": 42, "workers": 0, "amp": True,
            "lr": .001, "batch_size": 1, "base_channels": 16, "label_ids": list(range(6))}}

    def test_training_changes_only_the_declared_learning_rate(self):
        appearance = self.appearance()
        expected = dict(appearance["training_conditions"], lr=.0001)
        plan = {"training_conditions": expected}
        self.assertEqual(trainer.config_for_plan(plan, appearance), expected)
        self.assertEqual(appearance["training_conditions"]["lr"], .001)
        for field, value in (("lr", .01), ("batch_size", 2), ("base_channels", 8), ("seed", 43)):
            bad = copy.deepcopy(plan); bad["training_conditions"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                trainer.config_for_plan(bad, appearance)

    def test_warm_start_rejects_another_arm_epoch_or_configuration(self):
        appearance = self.appearance()
        checkpoint = {"config": appearance["training_conditions"], "epoch": 29, "best_epoch": 29,
                      "experiment": {"experiment_id": appearance["experiment_id"], "arm_tag": "gamma_polarity",
                                     "protocol_sha256": trainer.APPEARANCE_SHA}}
        trainer.warm_start_guard(checkpoint, {}, appearance)
        for field in ("arm", "epoch", "seed", "protocol"):
            bad = copy.deepcopy(checkpoint)
            if field == "arm": bad["experiment"]["arm_tag"] = "gamma_only"
            elif field == "epoch": bad["epoch"] = 30
            elif field == "seed": bad["config"]["seed"] = 43
            else: bad["experiment"]["protocol_sha256"] = "f" * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                trainer.warm_start_guard(bad, {}, appearance)

    def test_budget_or_test_scope_is_refused_before_other_inputs_are_read(self):
        for field, value in (("training_epochs_per_model", 9), ("allowed_splits", ["train", "validation", "test"])):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                plan = {"experiment_id": trainer.EXPERIMENT, "training_seed": 42,
                        "training_epochs_per_model": 8, "allowed_splits": ["train", "validation"]}
                plan[field] = value
                path = Path(directory) / "bad_protocol.json"; path.write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError, "scope/budget"):
                    trainer.verified_plan(path)


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch unavailable; policy tests still run")
class RecoveryTorchRNGTests(unittest.TestCase):
    def test_appearance_does_not_consume_the_original_flip_rng(self):
        import torch
        sample = {"image": torch.linspace(-2, 2, 8).reshape(1, 2, 2, 2),
                  "mask": torch.zeros((2, 2, 2), dtype=torch.long), "spacing": torch.ones(3),
                  "patient": "B006", "visit": "Week0"}
        class Base:
            def __len__(self): return 1
            def __getitem__(self, index):
                torch.rand(())  # The original H/W flip decisions consume exactly these draws.
                torch.rand(())
                return sample
        fake = types.SimpleNamespace(HipMRIVolumeDataset=mock.Mock(return_value=Base()))
        with mock.patch.dict(sys.modules, {"dataset": fake}):
            wrapper = rec.RecoveryTrainingDataset("unused", "polarity_rehearsal")
        torch.manual_seed(93)
        torch.rand(()); torch.rand(())
        expected = torch.get_rng_state().clone()
        torch.manual_seed(93)
        wrapper.set_epoch(3); wrapper[0]
        torch.testing.assert_close(torch.get_rng_state(), expected)


if __name__ == "__main__":
    unittest.main()
