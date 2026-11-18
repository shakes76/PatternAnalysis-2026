"""End-to-end checks using synthetic patients and locally generated images.

Run with: python3 -m unittest discover -s tests -v
Pillow is the only non-standard-library dependency.
No real patient data is used. Passing these tests checks the implemented
behavior; it does not replace auditing the actual dataset before training.
"""

import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "adni_splits.py"


def read_csv(path):
    """Read a generated manifest for assertions or deliberate corruption."""
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    """Rewrite a manifest to simulate an accidental or deliberate edit."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def patients(rows):
    """Extract unique patients rather than treating slices as independent."""
    return {row["patient_id"] for row in rows}


def paths(rows):
    """Extract slice paths to test complete, nonduplicated coverage."""
    return {row["relative_path"] for row in rows}


class ADNISplitTests(unittest.TestCase):
    """Exercise the CLI with isolated fixtures and realistic failure cases."""
    @classmethod
    def setUpClass(cls):
        """Ensure the tests target an existing implementation."""
        if not SCRIPT.is_file():
            raise RuntimeError(f"Missing implementation: {SCRIPT}")

    def setUp(self):
        """Create 84 synthetic patients, repeated scans, and mixed diagnoses.

        Two slices per scan keep tests small; production uses the configured
        expected count of 20. Every test gets its own temporary source tree.
        """
        self.temp = tempfile.TemporaryDirectory(prefix="adni_split_test_")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "source"
        self.out = self.base / "splits"
        self.metadata = {}
        self.expected = {}
        self.mixed_patients = set()
        image_number = 100000
        for patient_number in range(84):
            patient_id = f"{patient_number % 10:03d}_S_{patient_number + 4000:04d}"
            if patient_number < 40:
                diagnoses = ["AD"]
            elif patient_number < 80:
                diagnoses = ["NC"]
            else:
                diagnoses = ["NC", "AD"]
                self.mixed_patients.add(patient_id)
            # Simulate the supplied split's patient overlap with distinct scans
            # of the same patient in original train and test directories.
            if patient_number in (0, 1, 2, 40, 41, 42):
                diagnoses.append(diagnoses[0])
            for scan_number, diagnosis in enumerate(diagnoses):
                image_number += 1
                image_id = str(image_number)
                original_split = "train" if scan_number == 0 else "test"
                folder = self.root / "AD_NC" / original_split / diagnosis
                folder.mkdir(parents=True, exist_ok=True)
                self.metadata[image_id] = {
                    "raw": f"ADNI_T1_3T/ADNI_{patient_id}_MR_MPRAGE_I{image_id}.nii",
                    "label": 2 if diagnosis == "AD" else 0,
                }
                for slice_number in (78, 79):
                    image_path = folder / f"{image_id}_{slice_number}.jpeg"
                    rng = random.Random(image_number * 100 + slice_number)
                    pixels = bytes(rng.randrange(256) for _ in range(120))
                    Image.frombytes("L", (12, 10), pixels).save(
                        image_path, format="JPEG", quality=95
                    )
                    self.expected[image_path.relative_to(self.root).as_posix()] = {
                        "patient_id": patient_id,
                        "image_id": image_id,
                        "label": "1" if diagnosis == "AD" else "0",
                    }
        # Model metadata that includes scans outside the available AD/NC JPEG
        # subset. An unused diagnosis must not invalidate the selected subset.
        self.metadata["999999"] = {
            "raw": "ADNI_T1_3T/ADNI_999_S_1234_MR_MPRAGE_I999999.nii",
            "label": 1,
        }
        self.save_metadata()

    def save_metadata(self):
        """Persist the current synthetic metadata, including test mutations."""
        (self.root / "meta_data_with_label.json").write_text(
            json.dumps(self.metadata), encoding="utf-8"
        )

    def run_cli(self, action, output=None):
        """Run the public CLI using fixed fixture-specific settings."""
        output = output or self.out
        command = [
            sys.executable,
            str(SCRIPT),
            action,
            "--data-root", str(self.root),
            "--output", str(output),
        ]
        if action == "prepare":
            command.extend([
                "--folds", "5",
                "--seed", "3710",
                "--test-fraction", "0.2",
                "--calibration-fraction", "0.1",
                "--early-stop-fraction", "0.1",
                "--expected-slices", "2",
            ])
        return subprocess.run(command, capture_output=True, text=True, timeout=60)

    def prepare(self, output=None):
        """Create a valid baseline split before checking or corrupting it."""
        result = self.run_cli("prepare", output)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return output or self.out

    def assert_prepare_rejected(self):
        """Require invalid sources to fail without publishing completed output."""
        result = self.run_cli("prepare")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((self.out / "COMPLETED.json").exists())

    def assert_verify_rejected(self):
        """Require changed or inconsistent artifacts/sources to fail verification."""
        result = self.run_cli("verify")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_patient_isolation_cv_coverage_and_original_labels(self):
        """Check per-fold isolation, one validation turn, and per-scan labels."""
        self.prepare()
        self.assertTrue((self.out / "COMPLETED.json").is_file())
        self.assertTrue((self.out / "report.json").is_file())
        self.assertTrue((self.out / "patients.csv").is_file())
        all_rows = read_csv(self.out / "all.csv")
        self.assertEqual(len(all_rows), len(self.expected))
        self.assertEqual(paths(all_rows), set(self.expected))
        self.assertEqual(len(paths(all_rows)), len(all_rows))

        assigned = {}
        patient_folds = {}
        for row in all_rows:
            expected = self.expected[row["relative_path"]]
            for field, value in expected.items():
                self.assertEqual(row[field], value)
            self.assertEqual(len(row["file_sha256"]), 64)
            self.assertEqual(len(row["pixel_sha256"]), 64)
            self.assertEqual(
                row["file_sha256"],
                hashlib.sha256((self.root / row["relative_path"]).read_bytes()).hexdigest(),
            )
            patient = row["patient_id"]
            assigned.setdefault(patient, set()).add(row["partition"])
            patient_folds.setdefault(patient, set()).add(row["fold"])
            if row["partition"] == "development":
                self.assertIn(row["fold"], {"1", "2", "3", "4", "5"})
            else:
                self.assertEqual(row["fold"], "")
        self.assertEqual(len(assigned), 84)
        self.assertTrue(all(len(values) == 1 for values in assigned.values()))
        self.assertTrue(all(len(values) == 1 for values in patient_folds.values()))
        for patient in self.mixed_patients:
            self.assertEqual(
                {row["label"] for row in all_rows if row["patient_id"] == patient},
                {"0", "1"},
            )

        partitions = {
            name: read_csv(self.out / f"{name}.csv")
            for name in ("development", "calibration", "test")
        }
        union = set()
        for name, rows in partitions.items():
            self.assertTrue(rows)
            self.assertEqual({row["partition"] for row in rows}, {name})
            self.assertEqual({row["label"] for row in rows}, {"0", "1"})
            self.assertTrue(union.isdisjoint(patients(rows)))
            union.update(patients(rows))
        self.assertEqual(len(union), 84)

        development = partitions["development"]
        seen_validation = set()
        seen_validation_patients = set()
        for fold in range(1, 6):
            # Roles may change between folds. Within one fold, train,
            # early_stop, and val must contain entirely different patients.
            fold_dir = self.out / f"fold_{fold:02d}"
            train = read_csv(fold_dir / "train.csv")
            early_stop = read_csv(fold_dir / "early_stop.csv")
            validation = read_csv(fold_dir / "val.csv")
            for rows in (train, early_stop, validation):
                self.assertTrue(rows)
                self.assertEqual({row["label"] for row in rows}, {"0", "1"})
                self.assertTrue(patients(rows).isdisjoint(patients(partitions["test"])))
                self.assertTrue(patients(rows).isdisjoint(patients(partitions["calibration"])))
            self.assertTrue(patients(train).isdisjoint(patients(early_stop)))
            self.assertTrue(patients(train).isdisjoint(patients(validation)))
            self.assertTrue(patients(early_stop).isdisjoint(patients(validation)))
            self.assertEqual(paths(train) | paths(early_stop) | paths(validation), paths(development))
            self.assertEqual(len(train) + len(early_stop) + len(validation), len(development))
            self.assertTrue(seen_validation.isdisjoint(paths(validation)))
            self.assertTrue(seen_validation_patients.isdisjoint(patients(validation)))
            self.assertEqual({row["fold"] for row in validation}, {str(fold)})
            seen_validation.update(paths(validation))
            seen_validation_patients.update(patients(validation))
        self.assertEqual(seen_validation, paths(development))
        self.assertEqual(seen_validation_patients, patients(development))
        result = self.run_cli("verify")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_same_seed_reproduces_all_csv_manifests(self):
        """A fixed seed and unchanged sources must reproduce every CSV byte."""
        self.prepare()
        second = self.prepare(self.base / "splits_again")
        first_names = {path.relative_to(self.out) for path in self.out.rglob("*.csv")}
        second_names = {path.relative_to(second) for path in second.rglob("*.csv")}
        self.assertEqual(first_names, second_names)
        for relative in first_names:
            self.assertEqual((self.out / relative).read_bytes(), (second / relative).read_bytes())

    def test_identical_files_across_patients_are_rejected(self):
        """Reject an exact image copy assigned to another patient."""
        relatives = list(self.expected)
        first = relatives[0]
        second = next(p for p in relatives if self.expected[p]["patient_id"] != self.expected[first]["patient_id"])
        shutil.copyfile(self.root / first, self.root / second)
        self.assert_prepare_rejected()

    def test_identical_decoded_pixels_with_different_jpeg_encoding_are_rejected(self):
        """Catch equal pixels even when the JPEG file bytes differ."""
        relatives = list(self.expected)
        first = relatives[0]
        second = next(p for p in relatives if self.expected[p]["patient_id"] != self.expected[first]["patient_id"])
        image = Image.new("L", (12, 10), color=80)
        image.save(self.root / first, format="JPEG", quality=95, optimize=False)
        image.save(self.root / second, format="JPEG", quality=100, optimize=True)
        self.assertNotEqual((self.root / first).read_bytes(), (self.root / second).read_bytes())
        with Image.open(self.root / first) as a, Image.open(self.root / second) as b:
            self.assertEqual(a.tobytes(), b.tobytes())
        self.assert_prepare_rejected()

    def test_missing_metadata_is_rejected(self):
        """Do not assign an image whose scan has no metadata record."""
        del self.metadata[next(iter(self.metadata))]
        self.save_metadata()
        self.assert_prepare_rejected()

    def test_metadata_folder_label_conflict_is_rejected(self):
        """Reject disagreement between a diagnosis folder and metadata."""
        first = next(iter(self.metadata))
        self.metadata[first]["label"] = 0
        self.save_metadata()
        self.assert_prepare_rejected()

    def test_missing_slice_is_rejected(self):
        """Reject a scan with fewer slices than the configured count."""
        (self.root / next(iter(self.expected))).unlink()
        self.assert_prepare_rejected()

    def test_unreadable_image_is_rejected(self):
        """Reject JPEG filenames whose contents cannot be decoded."""
        (self.root / next(iter(self.expected))).write_bytes(b"not an image")
        self.assert_prepare_rejected()

    def test_forged_patient_in_manifest_is_rejected_by_verify(self):
        """Detect a patient identity edited after manifest creation."""
        self.prepare()
        path = self.out / "all.csv"
        rows = read_csv(path)
        rows[0]["patient_id"] = "999_S_9999"
        write_csv(path, rows)
        self.assert_verify_rejected()

    def test_resealed_forged_patient_still_fails_independent_verification(self):
        """Recheck source identities even if an edited CSV has a new checksum."""
        self.prepare()
        path = self.out / "all.csv"
        rows = read_csv(path)
        rows[0]["patient_id"] = "999_S_9999"
        write_csv(path, rows)
        # A checksum validates file integrity, not the truth of the mapping.
        # Recomputing it must not bypass comparison against source metadata.
        seal_path = self.out / "COMPLETED.json"
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        seal["sha256"]["all.csv"] = hashlib.sha256(path.read_bytes()).hexdigest()
        seal_path.write_text(json.dumps(seal), encoding="utf-8")
        self.assert_verify_rejected()

    def test_changed_source_image_is_rejected_by_verify(self):
        """Detect source image changes after a split has been frozen."""
        self.prepare()
        source = self.root / next(iter(self.expected))
        Image.new("L", (12, 10), color=91).save(source, format="JPEG")
        self.assert_verify_rejected()

    def test_changed_source_patient_mapping_is_rejected_by_verify(self):
        """Detect metadata changes to a scan's patient identity."""
        self.prepare()
        first = next(iter(self.metadata))
        self.metadata[first]["raw"] = (
            f"ADNI_T1_3T/ADNI_999_S_9999_MR_MPRAGE_I{first}.nii"
        )
        self.save_metadata()
        self.assert_verify_rejected()

    def test_deleted_source_image_is_rejected_by_verify(self):
        """Detect a source slice removed after preparation."""
        self.prepare()
        (self.root / next(iter(self.expected))).unlink()
        self.assert_verify_rejected()

    def test_training_row_added_to_validation_is_rejected_by_verify(self):
        """Reject a training sample injected into an outer-validation list."""
        self.prepare()
        folder = self.out / "fold_01"
        train = read_csv(folder / "train.csv")
        validation = read_csv(folder / "val.csv")
        validation.append(train[0])
        write_csv(folder / "val.csv", validation)
        self.assert_verify_rejected()

    def test_validation_row_added_to_training_is_rejected_by_verify(self):
        """Reject an outer-validation sample injected into a training list."""
        self.prepare()
        folder = self.out / "fold_01"
        train = read_csv(folder / "train.csv")
        validation = read_csv(folder / "val.csv")
        train.append(validation[0])
        write_csv(folder / "train.csv", train)
        self.assert_verify_rejected()

    def test_existing_output_is_not_overwritten(self):
        """Preserve an existing frozen split when prepare is run again."""
        self.prepare()
        before = {
            p.relative_to(self.out): p.read_bytes()
            for p in self.out.rglob("*") if p.is_file()
        }
        result = self.run_cli("prepare")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        after = {
            p.relative_to(self.out): p.read_bytes()
            for p in self.out.rglob("*") if p.is_file()
        }
        self.assertEqual(before, after)

    def test_output_inside_source_is_rejected(self):
        """Keep generated manifests outside the source dataset tree."""
        output = self.root / "generated_splits"
        result = self.run_cli("prepare", output)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((output / "COMPLETED.json").exists())


if __name__ == "__main__":
    unittest.main()
