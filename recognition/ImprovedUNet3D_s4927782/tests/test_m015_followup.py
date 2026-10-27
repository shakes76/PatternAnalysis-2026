"""Synthetic scope, identity, overlap and finite-forward guards; no real MRI."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools import evaluate_m015_followup as tool


def plans():
    old = {"models": {n: {"checkpoint": f"artifacts/{n}/best.pt"} for n in tool.MODEL_IDS}}
    new = {"experiment_id": tool.EXPERIMENT, "test_informed_exploratory_acknowledged": True,
           "case_keys": list(tool.CASE_KEYS), "manifest_sha256": tool.original.MANIFEST_SHA,
           "cache_identity_sha256": tool.original.IDENTITY_SHA, "training_sources_sha256": tool.original.SOURCES_SHA,
           "inference": {"split": "test", "device": "cpu", "amp": False, "saved_arrays": False},
           "models": {n: {**old["models"][n], "checkpoint_sha256": "a" * 64} for n in tool.MODEL_IDS}}
    new["models"]["seed42_reference"]["checkpoint_sha256"] = tool.original.REFERENCES["seed42_reference"][2]
    return new, old


def manifest():
    return {"partial": False, "development_only": False,
            "records": [{"patient": "M015", "visit": f"Week{i}", "split": "test"} for i in range(8)]
                       + [{"patient": "B037", "visit": "Week0", "split": "test"}]}


class ScopeTests(unittest.TestCase):
    def test_exact_protocol_and_case_order(self):
        new, old = plans()
        tool.validate_protocol(new, old)
        result = tool.select_records(manifest())
        self.assertEqual([r["visit"] for r in result], [f"Week{i}" for i in range(8)])
        self.assertEqual({r["patient"] for r in result}, {"M015"})

    def test_ack_extra_model_or_altered_checkpoint_are_rejected(self):
        for problem in ("ack", "seed43", "checkpoint", "case", "precision", "device", "arrays"):
            with self.subTest(problem=problem):
                new, old = plans()
                if problem == "ack": new["test_informed_exploratory_acknowledged"] = False
                if problem == "seed43": new["models"]["seed43_reference"] = new["models"]["gamma_only"]
                if problem == "checkpoint": new["models"]["gamma_only"]["checkpoint"] = "unapproved/best.pt"
                if problem == "case": new["case_keys"][0] = "B037/Week0"
                if problem == "precision": new["inference"]["amp"] = True
                if problem == "device": new["inference"]["device"] = "cuda"
                if problem == "arrays": new["inference"]["saved_arrays"] = True
                with self.assertRaises(ValueError): tool.validate_protocol(new, old)

    def test_missing_duplicate_wrong_split_and_development_fail(self):
        for problem in ("missing", "duplicate", "split", "development"):
            with self.subTest(problem=problem):
                value = manifest()
                if problem == "missing": value["records"].pop(0)
                if problem == "duplicate": value["records"].append(copy.deepcopy(value["records"][0]))
                if problem == "split": value["records"][0]["split"] = "train"
                if problem == "development": value["development_only"] = True
                with self.assertRaises(ValueError): tool.select_records(value)

    def test_existing_output_refused_before_input_or_torch_read(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "must be new"):
                tool.main(["--protocol", "/missing.json", "--manifest", "/missing.json", "--output-dir", directory])

    def test_explicit_fp_fn_does_not_change_zero_or_undefined_dice(self):
        value = {"cases": [{"per_class": [
            {"intersection": 0, "predicted_voxels": 2, "target_voxels": 3, "dice": 0.0},
            {"intersection": 0, "predicted_voxels": 0, "target_voxels": 0, "dice": None}]}]}
        tool.add_error_counts(value)
        rows = value["cases"][0]["per_class"]
        self.assertEqual((rows[0]["false_positive_voxels"], rows[0]["false_negative_voxels"]), (2, 3))
        self.assertEqual(rows[0]["dice"], 0.0)
        self.assertIsNone(rows[1]["dice"])


@unittest.skipIf(importlib.util.find_spec("numpy") is None, "NumPy required for synthetic cache fixture")
class CacheIdentityTests(unittest.TestCase):
    def test_decoded_cache_hash_accepts_exact_and_rejects_modified_pixels(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
            mask = (image % 2).astype(np.uint8)
            arrays = {"image": image, "mask": mask, "spacing": np.ones(3, dtype=np.float32)}
            record = {"patient": "M015", "visit": "Week0", "split": "test", "cache_path": "M015_Week0.npz",
                      "target_shape_dhw": [2, 2, 2], "spacing_dhw_mm": [1, 1, 1],
                      "resampled_affine_xyz": np.eye(4).tolist(), "class_voxel_counts": [4, 4]}
            data = {"target_shape_dhw": [2, 2, 2], "num_classes": 2}
            expected = {}
            for name, value in arrays.items():
                descriptor = {"dtype": value.dtype.str, "shape": list(value.shape)}
                digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode() + b"\0"
                                        + value.tobytes(order="C")).hexdigest()
                expected[name] = {**descriptor, "sha256": digest}
            identity = {"records": {record["cache_path"]: {k: record[k] for k in ("patient", "visit", "split")}}}
            identity["records"][record["cache_path"]]["arrays"] = expected
            np.savez_compressed(root / record["cache_path"], **arrays)
            loaded, _ = tool.decode_case(root / "manifest.json", data, record, identity)
            np.testing.assert_array_equal(loaded["image"], image)
            arrays["image"] = image + np.float32(0.01)
            np.savez_compressed(root / record["cache_path"], **arrays)
            with self.assertRaisesRegex(ValueError, "Decoded M015"):
                tool.decode_case(root / "manifest.json", data, record, identity)


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch required for finite main-logit guard")
class ForwardGuardTests(unittest.TestCase):
    def test_nonfinite_main_logits_rejected_before_argmax(self):
        import torch
        with self.assertRaisesRegex(RuntimeError, "before argmax"):
            tool.original.finite_main_logits(None, (), {"logits": torch.tensor([0., float("nan")])})


if __name__ == "__main__":
    unittest.main()
