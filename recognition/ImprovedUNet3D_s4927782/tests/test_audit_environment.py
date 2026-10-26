"""Test file-name discovery and reporting using temporary synthetic paths."""

import contextlib
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "audit_environment.py"
SPEC = importlib.util.spec_from_file_location("audit_environment", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class AuditEnvironmentTests(unittest.TestCase):
    def make_dataset(self, root: Path, images: int = 1, labels: int = 1) -> None:
        for name, count in (("semantic_MRs", images), ("semantic_labels_only", labels)):
            directory = root / name
            directory.mkdir(parents=True)
            for index in range(count):
                # Empty files are deliberate: the audit must not read scan pixels.
                suffix = ".nii" if index % 2 else ".nii.gz"
                (directory / f"scan_{index:02d}{suffix}").touch()

    def test_recursive_counts_and_eight_sample_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_dataset(root, images=12, labels=3)
            nested = root / "semantic_MRs" / "nested"
            nested.mkdir()
            (nested / "extra.NII.GZ").touch()
            (nested / "ignore.nii.gz.backup").touch()
            (nested / "ignore.txt").touch()
            images = audit.discover_folder(root, "semantic_MRs")
            labels = audit.discover_folder(root, "semantic_labels_only")
            self.assertEqual(images["nifti_count"], 13)
            self.assertEqual(labels["nifti_count"], 3)
            self.assertEqual(len(images["sample_relative_paths"]), 8)
            self.assertEqual(len(labels["sample_relative_paths"]), 3)
            for sample in images["sample_relative_paths"]:
                self.assertFalse(Path(sample).is_absolute())
                self.assertTrue(sample.startswith("semantic_MRs/"))

    @mock.patch.object(audit, "gpu_inventory", return_value={"status": "unavailable"})
    def test_missing_root_is_blocked_and_still_writes_json(self, _gpu):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "missing"
            output = Path(temporary) / "reports" / "audit.json"
            with contextlib.redirect_stdout(io.StringIO()):
                code = audit.main(["--data-root", str(root), "--output", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertNotEqual(code, 0)
            self.assertEqual(report["status"], "blocked")
            self.assertFalse(report["data_root_exists"])
            self.assertTrue(report["blockers"])
            self.assertFalse(root.exists())
            for folder in report["folders"].values():
                self.assertFalse(folder["exists"])
                self.assertEqual(folder["nifti_count"], 0)

    @mock.patch.object(audit, "gpu_inventory", return_value={"status": "unavailable"})
    def test_successful_json_explicitly_keeps_validation_unverified(self, _gpu):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            self.make_dataset(root)
            output = Path(temporary) / "audit.json"
            with contextlib.redirect_stdout(io.StringIO()):
                code = audit.main(["--data-root", str(root), "--output", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "inventory_complete")
            self.assertEqual(report["blockers"], [])
            self.assertEqual(report["data_root"], str(root.resolve()))
            self.assertFalse(any(report["validation"].values()))
            self.assertIn("python_version", report["environment"])
            self.assertIn("platform", report["environment"])

    @mock.patch.object(audit, "gpu_inventory", return_value={"status": "unavailable"})
    def test_missing_expected_folder_is_blocked(self, _gpu):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "semantic_MRs").mkdir()
            (root / "semantic_MRs" / "scan.nii").touch()
            report = audit.audit_environment(root)
            self.assertEqual(report["status"], "blocked")
            self.assertFalse(report["folders"]["semantic_labels_only"]["exists"])

    def test_distribution_versions_use_metadata_and_allow_missing_packages(self):
        def version(name):
            if name == "nibabel":
                raise audit.importlib.metadata.PackageNotFoundError(name)
            return "test-version"

        with mock.patch.object(audit.importlib.metadata, "version", side_effect=version) as lookup:
            versions = audit.distribution_versions()
        self.assertEqual(set(versions), set(audit.DISTRIBUTIONS))
        self.assertIsNone(versions["nibabel"])
        self.assertEqual(versions["torch"], "test-version")
        self.assertEqual(lookup.call_count, 4)

    @mock.patch.object(audit.shutil, "which", return_value=None)
    @mock.patch.object(audit.subprocess, "run")
    def test_gpu_command_absent_does_not_start_subprocess(self, run, _which):
        self.assertEqual(audit.gpu_inventory()["status"], "unavailable")
        run.assert_not_called()

    @mock.patch.object(audit.shutil, "which", return_value="/test/nvidia-smi")
    def test_gpu_query_has_five_second_timeout(self, _which):
        with mock.patch.object(
            audit.subprocess, "run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)
        ) as run:
            result = audit.gpu_inventory()
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(run.call_args.kwargs["timeout"], 5)
        self.assertFalse(run.call_args.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
