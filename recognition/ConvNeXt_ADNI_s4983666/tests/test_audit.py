"""Exercise the M0 source audit with small, wholly synthetic JPEG datasets.

These checks cover reporting, leakage detection, and read-only source handling.
They do not claim that the real course dataset has been audited locally.
"""

import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional
import unittest

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "adni_splits.py"
IMPLEMENTATION = SCRIPT.parent / "adni_data" / "splits.py"


class SourceAuditTests(unittest.TestCase):
    """Check audit outcomes without creating any training split or model."""

    def setUp(self) -> None:
        """Create four distinct patients, one per supplied split and class."""
        temporary = tempfile.TemporaryDirectory(prefix="adni_m0_test_")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "source"
        self.output = self.base / "audit"
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.scans: Dict[str, str] = {}
        self.patient_ids: Dict[str, str] = {}
        for index, (split, diagnosis) in enumerate(
            (("train", "AD"), ("train", "NC"), ("test", "AD"), ("test", "NC")),
            start=1,
        ):
            image_id = str(100000 + index)
            patient_id = f"001_S_{4000 + index}"
            self.scans[f"{split}/{diagnosis}"] = image_id
            self.patient_ids[f"{split}/{diagnosis}"] = patient_id
            self.metadata[image_id] = {
                "raw": f"ADNI_T1_3T/ADNI_{patient_id}_MR_MPRAGE_I{image_id}.nii",
                "label": 2 if diagnosis == "AD" else 0,
            }
            folder = self.root / "AD_NC" / split / diagnosis
            folder.mkdir(parents=True)
            for slice_index in (78, 79):
                generator = random.Random(index * 100 + slice_index)
                pixels = bytes(generator.randrange(256) for _ in range(120))
                Image.frombytes("L", (12, 10), pixels).save(
                    folder / f"{image_id}_{slice_index}.jpeg", quality=95
                )
        self.save_metadata()

    def save_metadata(self) -> None:
        """Persist deliberately synthetic scan-to-patient metadata."""
        (self.root / "meta_data_with_label.json").write_text(
            json.dumps(self.metadata), encoding="utf-8"
        )

    def run_audit(self, output: Optional[Path] = None) -> subprocess.CompletedProcess:
        """Invoke the public audit command with two slices per synthetic scan."""
        return subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "audit", "--data-root", str(self.root),
             "--output", str(output or self.output), "--expected-slices", "2"],
            cwd=SCRIPT.parent, capture_output=True, text=True, timeout=30,
        )

    def read_metrics(self, output: Optional[Path] = None) -> Dict[str, Any]:
        """Load the public JSON report from a completed audit attempt."""
        return json.loads(((output or self.output) / "metrics.json").read_text())

    def snapshot(self, directory: Path) -> Dict[str, str]:
        """Fingerprint every file to detect additions, removals, or changes."""
        return {
            path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()
        }

    def share_ad_patient(self) -> None:
        """Put distinct train and test AD scans under the same patient ID."""
        image_id = self.scans["test/AD"]
        patient_id = self.patient_ids["train/AD"]
        self.metadata[image_id]["raw"] = (
            f"ADNI_T1_3T/ADNI_{patient_id}_MR_MPRAGE_I{image_id}.nii"
        )
        self.save_metadata()

    def assert_failed_report(self, result: subprocess.CompletedProcess) -> None:
        """Require a recorded source-validation failure and visible tree."""
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        metrics = self.read_metrics()
        self.assertEqual(metrics["report_type"], "data_audit")
        self.assertEqual(metrics["status"], "failed")
        self.assertTrue(metrics.get("error"), metrics)
        self.assertIn("AD_NC", result.stdout)
        self.assertFalse(list(self.output.rglob("*.csv")))

    def test_valid_disjoint_sources_report_statistics_without_modification(self) -> None:
        """Report actual supplied groups, dimensions, and channels unchanged."""
        before = self.snapshot(self.root)
        result = self.run_audit()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        metrics = self.read_metrics()
        self.assertEqual(metrics["report_type"], "data_audit")
        self.assertEqual(metrics["status"], "passed")
        self.assertEqual(metrics["source_audit"]["patients"], 4)
        self.assertEqual(metrics["source_audit"]["image_records"], 4)
        self.assertEqual(metrics["source_audit"]["images"], 8)
        for split in ("train", "test"):
            supplied = metrics["supplied_splits"][split]
            self.assertEqual(supplied["patients"], 2)
            self.assertEqual(supplied["images"], 4)
            self.assertEqual(supplied["images_by_label"], {"AD": 2, "NC": 2})
            self.assertEqual(supplied["dimensions"], [
                {"width": 12, "height": 10, "channels": 1, "mode": "L", "images": 4}
            ])
            self.assertTrue(supplied["filename_examples"])
        self.assertEqual(set(metrics["original_split_checks"]), {
            "patient_id", "image_id", "file_sha256", "pixel_sha256"
        })
        for check in metrics["original_split_checks"].values():
            self.assertEqual(check["overlap_count"], 0)
        self.assertEqual(before, self.snapshot(self.root))
        self.assertEqual({path.name for path in self.output.iterdir()},
                         {"config.json", "metrics.json"})
        self.assertIn("AD_NC", result.stdout)
        self.assertIn("train", result.stdout)
        self.assertIn("test", result.stdout)

    def test_patient_overlap_blocks_without_resplitting(self) -> None:
        """Distinguish patient leakage from zero overlapping scan IDs."""
        self.share_ad_patient()
        before = self.snapshot(self.root)
        result = self.run_audit()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        metrics = self.read_metrics()
        self.assertEqual(metrics["status"], "blocked")
        self.assertEqual(metrics["original_split_checks"]["patient_id"]["overlap_count"], 1)
        self.assertEqual(metrics["original_split_checks"]["image_id"]["overlap_count"], 0)
        self.assertEqual(metrics["source_audit"]["original_patient_overlap"], 1)
        self.assertIn("existing verified patient manifests", metrics["next_step"])
        self.assertIn("does not assess manifest boundaries", metrics["next_step"])
        self.assertEqual(before, self.snapshot(self.root))
        self.assertFalse(list(self.output.rglob("*.csv")))

    def test_exact_content_leakage_is_reported_across_distinct_scans(self) -> None:
        """Retain hash checks even when renamed slices have different IDs."""
        self.share_ad_patient()
        for slice_index in (78, 79):
            source = self.root / "AD_NC/train/AD" / f"{self.scans['train/AD']}_{slice_index}.jpeg"
            destination = self.root / "AD_NC/test/AD" / f"{self.scans['test/AD']}_{slice_index}.jpeg"
            shutil.copyfile(source, destination)
        result = self.run_audit()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        checks = self.read_metrics()["original_split_checks"]
        self.assertEqual(checks["image_id"]["overlap_count"], 0)
        self.assertEqual(checks["file_sha256"]["overlap_count"], 2)
        self.assertEqual(checks["pixel_sha256"]["overlap_count"], 2)

    def test_malformed_filename_records_failure_after_directory_listing(self) -> None:
        """Reject filenames that cannot identify a scan and slice."""
        image = next((self.root / "AD_NC/train/AD").glob("*.jpeg"))
        image.rename(image.with_name("unknown_patient.jpeg"))
        self.assert_failed_report(self.run_audit())

    def test_malformed_patient_metadata_records_failure(self) -> None:
        """Do not invent a patient ID from an image-ID filename prefix."""
        image_id = self.scans["train/AD"]
        self.metadata[image_id]["raw"] = f"ADNI_unknown_MR_I{image_id}.nii"
        self.save_metadata()
        self.assert_failed_report(self.run_audit())

    def test_corrupt_jpeg_records_failure(self) -> None:
        """Decode source pixels and record unreadable JPEGs as failures."""
        image = next((self.root / "AD_NC/train/AD").glob("*.jpeg"))
        image.write_bytes(b"synthetic invalid JPEG")
        self.assert_failed_report(self.run_audit())

    def test_mixed_history_is_reported_without_relabeling_scans(self) -> None:
        """Report longitudinal diagnosis conflicts without assigning one label."""
        image_id = self.scans["train/NC"]
        patient_id = self.patient_ids["train/AD"]
        self.metadata[image_id]["raw"] = f"ADNI_{patient_id}_MR_MPRAGE_I{image_id}.nii"
        self.save_metadata()
        result = self.run_audit()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        metrics = self.read_metrics()
        self.assertEqual(metrics["source_audit"]["mixed_label_patients"], [patient_id])
        self.assertEqual(metrics["supplied_splits"]["train"]["images_by_label"], {"AD": 2, "NC": 2})

    def test_existing_output_is_never_overwritten(self) -> None:
        """Preserve a completed report when its output path is reused."""
        first = self.run_audit()
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        before = self.snapshot(self.output)
        second = self.run_audit()
        self.assertEqual(second.returncode, 1, second.stdout + second.stderr)
        self.assertEqual(before, self.snapshot(self.output))

    def test_source_and_ancestor_paths_are_protected(self) -> None:
        """Reject outputs within, equal to, or above the source directory."""
        before = self.snapshot(self.root)
        for output in (self.root / "audit", self.root, self.base):
            with self.subTest(output=str(output)):
                result = self.run_audit(output)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertEqual(before, self.snapshot(self.root))
        self.assertFalse((self.root / "audit").exists())
        self.assertFalse((self.base / "config.json").exists())

    def test_source_fingerprints_are_reproducible(self) -> None:
        """Bind successful reports to unchanged metadata and image contents."""
        first = self.run_audit()
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        other = self.base / "another_audit"
        second = self.run_audit(other)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        first_audit = self.read_metrics()["source_audit"]
        second_audit = self.read_metrics(other)["source_audit"]
        self.assertEqual(first_audit["source_fingerprint"], second_audit["source_fingerprint"])
        expected = hashlib.sha256((self.root / "meta_data_with_label.json").read_bytes()).hexdigest()
        self.assertEqual(first_audit["metadata_sha256"], expected)
        # The CLI is only a wrapper; provenance must identify the executed code.
        config = json.loads((self.output / "config.json").read_text())
        self.assertEqual(config["script_sha256"], hashlib.sha256(IMPLEMENTATION.read_bytes()).hexdigest())
        self.assertNotEqual(config["script_sha256"], hashlib.sha256(SCRIPT.read_bytes()).hexdigest())

    def test_audit_runs_without_training_libraries(self) -> None:
        """Keep the lightweight audit independent of PyTorch and NumPy."""
        runner = """
import builtins
import runpy
import sys
original_import = builtins.__import__
def guarded_import(name: str, *args: object, **kwargs: object) -> object:
    '''Reject model-library imports in the lightweight audit subprocess.'''
    if name.split('.')[0] in {'torch', 'torchvision', 'numpy', 'sklearn'}:
        raise ImportError('Training dependency imported by M0: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", runner, str(SCRIPT), "audit", "--data-root",
             str(self.root), "--output", str(self.output), "--expected-slices", "2"],
            cwd=SCRIPT.parent, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.read_metrics()["status"], "passed")

    def test_parser_defaults_target_course_data_and_local_run(self) -> None:
        """Inspect default arguments without accessing course data or writing."""
        runner = """
import json
import runpy
import sys
namespace = runpy.run_path(sys.argv[1], run_name='audit_default_test')
def capture(args: object) -> int:
    '''Read parsed defaults without executing the source-data audit.'''
    print(json.dumps({'data_root': str(args.data_root), 'output': str(args.output)}))
    return 0
namespace['main'].__globals__['audit_sources'] = capture
raise SystemExit(namespace['main'](['audit']))
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", runner, str(SCRIPT)],
            cwd=SCRIPT.parent, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout), {
            "data_root": "/home/groups/comp3710/ADNI", "output": "runs/m0_audit"
        })


if __name__ == "__main__":
    unittest.main()
