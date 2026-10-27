"""Image-only v2 identity/selection guards; no patient data or GPU required."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import infer_v2 as adapter
import v2_pipeline as v2


def fixture():
    protocol = {"experiment_id": "fixture", "_sha256": "a" * 64,
                "configuration": copy.deepcopy(v2.DEFAULT_CONFIGURATION), "required_original_sha256": {}}
    checkpoint = {"identity": {"experiment_id": "fixture", "protocol_sha256": "a" * 64,
        "configuration": copy.deepcopy(protocol["configuration"]), "model": "unet2d",
        "v2_pipeline_sha256": "b" * 64, "source_hashes": copy.deepcopy(v2.FROZEN_SOURCE_SHA256),
        "manifest_sha256": "c" * 64},
        "config": {**protocol["configuration"], "model": "unet2d", "label_ids": list(range(6))},
        "epoch": 5, "best_epoch": 5, "best_score": .8,
        "history": [{"epoch": 1, "validation_prostate_dice": .4},
                    {"epoch": 5, "validation_prostate_dice": .8}]}
    return protocol, checkpoint


class AdapterIdentityTests(unittest.TestCase):
    def validate(self, protocol, checkpoint):
        return adapter.validate_checkpoint_identity(checkpoint, protocol, "b" * 64,
                                                     v2.FROZEN_SOURCE_SHA256["modules.py"])

    def test_valid_identity_needs_no_mask_or_data_manifest(self):
        protocol, checkpoint = fixture()
        with patch.object(v2, "read_json", side_effect=AssertionError("No data file may be read")):
            self.assertEqual(self.validate(protocol, checkpoint), "unet2d")

    def test_wrong_protocol_pipeline_or_frozen_model_source_is_refused(self):
        protocol, checkpoint = fixture()
        for key, value in [("protocol_sha256", "d" * 64), ("v2_pipeline_sha256", "d" * 64),
                           ("manifest_sha256", "not a digest")]:
            wrong = copy.deepcopy(checkpoint)
            wrong["identity"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(protocol, wrong)
        with self.assertRaises(ValueError):
            adapter.validate_checkpoint_identity(checkpoint, protocol, "b" * 64, "d" * 64)

    def test_final_weights_are_not_mistaken_for_selected_best(self):
        protocol, checkpoint = fixture()
        checkpoint["epoch"] = 30
        with self.assertRaises(ValueError):
            self.validate(protocol, checkpoint)

    def test_earliest_tie_and_scheduled_validation_selection_are_checked(self):
        protocol, checkpoint = fixture()
        checkpoint["history"][0]["validation_prostate_dice"] = .8
        with self.assertRaises(ValueError):
            self.validate(protocol, checkpoint)
        protocol, checkpoint = fixture()
        checkpoint["history"].insert(1, {"epoch": 3, "validation_prostate_dice": .6})
        with self.assertRaises(ValueError):
            self.validate(protocol, checkpoint)

    def test_frozen_pair_must_bind_requested_weights(self):
        protocol, checkpoint = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            receipt = {"status": "frozen", "protocol_sha256": protocol["_sha256"],
                "manifest_sha256": checkpoint["identity"]["manifest_sha256"],
                "models": {name: {"completed_epochs": 30, "checkpoint_sha256": "e" * 64} for name in v2.MODELS}}
            v2.write_json(path, receipt)
            self.assertEqual(adapter.validate_selection_freeze(path, checkpoint, "e" * 64, protocol, "unet2d"),
                             v2.sha256(path))
            with self.assertRaises(ValueError):
                adapter.validate_selection_freeze(path, checkpoint, "f" * 64, protocol, "unet2d")

    def test_cli_refuses_existing_output_before_reading_any_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit) as caught:
                adapter.main(["--image", "missing.nii.gz", "--checkpoint", "missing.pt",
                    "--protocol", "missing.json", "--project-root", "missing", "--output-dir", directory])
            self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
