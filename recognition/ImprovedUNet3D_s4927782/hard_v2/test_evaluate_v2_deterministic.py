"""Execution/identity fixtures; synthetic metadata never claims model accuracy."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import evaluate_v2_deterministic as adapter

HERE = Path(__file__).resolve().parent


class FakePipeline:
    def __init__(self):
        self.events = []
        self.torch = SimpleNamespace(device=lambda kind: SimpleNamespace(type=kind),
                                     cuda=SimpleNamespace(is_available=lambda: True))
        self.bad_identity = False

    def load_protocol(self, path):
        self.events.append("load_protocol")
        result = adapter.read_json(path)
        result["_sha256"] = adapter.sha256(path)
        return result

    def verify_review_sources(self, protocol, directory):
        self.events.append("verify_sources")

    def seed_everything(self, seed):
        if seed != 3711:
            raise AssertionError("Unexpected seed")
        self.events.append("seed3711")

    def main(self, argv):
        assert self.events[-1] == "seed3711"
        assert "--execution-amendment" not in argv
        self.events.append("main_evaluate")
        def value(flag, default=None):
            return argv[argv.index(flag) + 1] if flag in argv else default
        out = Path(value("--output-dir"))
        out.mkdir()
        data = {
            "identity": {"model": value("--model"),
                         "protocol_sha256": adapter.sha256(value("--protocol")),
                         "manifest_sha256": adapter.sha256(value("--manifest"))},
            "checkpoint_sha256": adapter.sha256(value("--checkpoint")),
            "split": value("--split", "validation"),
            "fixture": "Synthetic metadata only; no inference or metrics measured.",
        }
        if self.bad_identity:
            data["identity"]["manifest_sha256"] = "0" * 64
        raw = json.dumps(data, indent=2) + "\n"
        (out / "evaluation.json").write_text(raw)
        (out / "metrics.json").write_text(raw)
        self.raw_evaluation = raw.encode()
        self.forwarded = argv
        return 0


def runtime_fixture(torch_api, device, configuration):
    return {
        "device": device.type, "gpu_name": "Synthetic GPU; not measured",
        "python_version": "fixture", "torch_version": "fixture", "cuda_version": "fixture",
        "numpy_version": "fixture", "scipy_version": "fixture", "nibabel_version": "fixture",
        "amp": bool(configuration["amp"] and device.type == "cuda"), "seed": configuration["seed"],
        "cudnn_deterministic": True, "cudnn_benchmark": False,
        "deterministic_algorithms_enabled": True, "deterministic_algorithms_warn_only": True,
        "slurm_job_id": "synthetic", "slurm_node_list": "synthetic",
    }


class EvaluationAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.protocol = self.root / "protocol_v2.json"
        self.protocol.write_bytes((HERE / "protocol_v2.json").read_bytes())
        self.checkpoint = self.root / "checkpoint.pt"
        self.checkpoint.write_bytes(b"Synthetic checkpoint bytes, never loaded as a model")
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text('{"fixture":"metadata only"}\n')
        self.amendment = self.root / adapter.AMENDMENT_NAME
        self.plan = adapter.read_json(self.protocol)
        self.amend = {
            "experiment_id": self.plan["experiment_id"],
            "original_protocol_sha256": adapter.sha256(self.protocol),
            "engineering_only": True, "scientific_configuration_changed": False,
            "wrapper_sha256": adapter.sha256(adapter.__file__),
            "new_driver_sha256": adapter.sha256(HERE / adapter.DRIVER_NAME),
            "reason": "Synthetic execution-contract fixture, not measured numerical discrepancy",
        }
        self.save_amendment()
        self.pipeline = FakePipeline()

    def save_amendment(self):
        self.amendment.write_text(json.dumps(self.amend))

    def argv(self, output=None):
        return ["--mode", "evaluate", "--protocol", str(self.protocol),
                "--project-root", "/synthetic/frozen/project", "--manifest", str(self.manifest),
                "--checkpoint", str(self.checkpoint), "--model", "improved3d",
                "--output-dir", str(output or self.root / "output"), "--device", "cuda",
                "--split", "validation", "--execution-amendment", str(self.amendment)]

    def execute(self, argv=None, collector=runtime_fixture):
        with contextlib.redirect_stdout(io.StringIO()):
            return adapter.execute(argv or self.argv(), self.pipeline, collector)

    def test_seed_precedes_frozen_evaluation_and_receipt_binds_original_files(self):
        self.assertEqual(self.execute(), 0)
        self.assertEqual(self.pipeline.events, ["load_protocol", "verify_sources", "seed3711", "main_evaluate"])
        out = self.root / "output"
        receipt = adapter.read_json(out / "runtime_receipt.json")
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["evaluation_sha256"], adapter.sha256(out / "evaluation.json"))
        self.assertEqual(receipt["metrics_sha256"], adapter.sha256(out / "metrics.json"))
        self.assertEqual(receipt["checkpoint_sha256"], adapter.sha256(self.checkpoint))
        self.assertEqual(receipt["manifest_sha256"], adapter.sha256(self.manifest))
        self.assertEqual(receipt["adapter_sha256"], adapter.sha256(adapter.__file__))
        self.assertEqual(receipt["execution_amendment_sha256"], adapter.sha256(self.amendment))
        self.assertEqual(receipt["execution_amendment_file"], adapter.AMENDMENT_NAME)
        self.assertTrue(receipt["amp"])
        self.assertEqual((out / "evaluation.json").read_bytes(), self.pipeline.raw_evaluation)
        self.assertEqual((out / "metrics.json").read_bytes(), self.pipeline.raw_evaluation)
        for key in runtime_fixture(None, SimpleNamespace(type="cuda"), self.plan["configuration"]):
            self.assertIn(key, receipt)

    def test_refuses_train_prepare_and_existing_evidence_before_pipeline(self):
        for mode in ("train", "prepare", "prepare-test"):
            argv = self.argv(); argv[1] = mode
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.execute(argv)
        existing = self.root / "old_result"; existing.mkdir()
        with self.assertRaises(FileExistsError):
            self.execute(self.argv(existing))
        self.assertEqual(self.pipeline.events, [])

    def test_wrong_amendment_source_protocol_and_scientific_change_are_rejected(self):
        for field, value in [("wrapper_sha256", "a" * 64), ("new_driver_sha256", "b" * 64),
                             ("original_protocol_sha256", "c" * 64),
                             ("scientific_configuration_changed", True), ("engineering_only", False)]:
            with self.subTest(field=field):
                current = copy.deepcopy(self.amend)
                self.amend[field] = value; self.save_amendment()
                with self.assertRaises(ValueError): self.execute()
                self.assertNotIn("seed3711", self.pipeline.events)
                self.amend = current; self.save_amendment(); self.pipeline.events.clear()

    def test_changed_protocol_cannot_be_laundered_by_a_new_amendment_digest(self):
        changed = adapter.read_json(self.protocol)
        changed["extra_fixture_field"] = "different bytes"
        self.protocol.write_text(json.dumps(changed))
        self.amend["original_protocol_sha256"] = adapter.sha256(self.protocol)
        self.save_amendment()
        with self.assertRaises(ValueError): self.execute()
        self.assertNotIn("seed3711", self.pipeline.events)

    def test_wrong_output_identity_preserves_raw_output_but_does_not_certify_it(self):
        self.pipeline.bad_identity = True
        with self.assertRaises(ValueError): self.execute()
        self.assertTrue((self.root / "output/evaluation.json").exists())
        self.assertFalse((self.root / "output/runtime_receipt.json").exists())

    def test_wrong_flags_or_environment_change_are_not_certified(self):
        def wrong_flags(*args):
            data = runtime_fixture(*args); data["cudnn_deterministic"] = False
            return data
        with self.assertRaises(RuntimeError): self.execute(collector=wrong_flags)
        self.assertNotIn("main_evaluate", self.pipeline.events)
        calls = []
        def changed_runtime(*args):
            data = runtime_fixture(*args); calls.append(1)
            if len(calls) == 2: data["torch_version"] = "changed"
            return data
        with self.assertRaises(RuntimeError): self.execute(collector=changed_runtime)
        self.assertFalse((self.root / "output/runtime_receipt.json").exists())

    def test_amendment_equal_syntax_is_removed_without_changing_other_arguments(self):
        argv = self.argv()
        argv[-2:] = ["--execution-amendment=" + str(self.amendment)]
        self.execute(argv)
        self.assertEqual(self.pipeline.forwarded, argv[:-1])

    def test_new_driver_changes_only_two_evaluation_entry_points(self):
        old = (HERE / "formal_v2_gpu.slurm").read_text()
        new = (HERE / adapter.DRIVER_NAME).read_text()
        new_call = ('"$DEMO3_PYTHON" -u evaluate_v2_deterministic.py --mode evaluate --protocol protocol_v2.json \\\n'
                    '    --execution-amendment engineering_execution_amendment.json \\\n')
        old_call = '"$DEMO3_PYTHON" -u v2_pipeline.py --mode evaluate --protocol protocol_v2.json \\\n'
        self.assertEqual(new.count(new_call), 2)
        self.assertEqual(new.replace(new_call, old_call), old)
        source = adapter.read_json(HERE / "SOURCE_FREEZE.json")
        for relative, expected in source["code_sha256"].items():
            self.assertEqual(adapter.sha256(HERE / relative), expected)


HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ("torch", "scipy", "nibabel"))


@unittest.skipUnless(HAS_RUNTIME, "Actual Torch/SciPy/NiBabel runtime required; metadata mocks are not model execution")
class ActualRuntimeFlagTests(unittest.TestCase):
    def test_actual_training_flag_setup_is_observable_before_any_inference(self):
        torch = adapter.v2.torch
        original = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
                    torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
        try:
            adapter.v2.seed_everything(3711)
            measured = adapter.collect_runtime(torch, torch.device("cpu"), {"seed": 3711, "amp": True})
            adapter.require_matching_flags(measured)
            self.assertFalse(measured["amp"])
            self.assertIsNone(measured["gpu_name"])
        finally:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = original[:2]
            torch.use_deterministic_algorithms(original[2], warn_only=original[3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
