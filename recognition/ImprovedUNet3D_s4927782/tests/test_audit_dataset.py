"""Test strict pairing, patient isolation and geometry using synthetic data only."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "audit_dataset.py"
SPEC = importlib.util.spec_from_file_location("audit_dataset", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class PairingAndSplitTests(unittest.TestCase):
    def make_names(self, root, patients=("B006", "B037", "B100")):
        for folder in audit.FOLDERS.values():
            (root / folder).mkdir(parents=True, exist_ok=True)
        for patient in patients:
            (root / "semantic_MRs" / f"{patient}_Week0_LFOV.nii.gz").touch()
            (root / "semantic_labels_only" / f"{patient}_Week0_SEMANTIC.nii.gz").touch()

    def test_pairing_uses_patient_and_visit_not_sorted_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_names(root)
            for folder, kind in (("semantic_MRs", "LFOV"), ("semantic_labels_only", "SEMANTIC")):
                (root / folder / f"B006_Week12_{kind}.nii").touch()
            report = audit.discover_pairs(root)
            self.assertEqual(report["blockers"], [])
            self.assertEqual(len(report["records"]), 4)
            self.assertEqual({record["visit"] for record in report["records"]}, {"Week0", "Week12"})
            self.assertTrue(all(record["pairing_valid"] for record in report["records"]))

    def test_multiple_observed_patient_code_prefixes_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_names(root, patients=("B006", "C032", "D031", "G021"))
            report = audit.discover_pairs(root)
            self.assertEqual(report["blockers"], [])
            self.assertEqual({item["patient"] for item in report["records"]},
                             {"B006", "C032", "D031", "G021"})

    def test_unknown_nifti_and_wrong_directory_kind_block_instead_of_disappearing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_names(root)
            (root / "semantic_MRs" / "unrecognised.nii.gz").touch()
            (root / "semantic_MRs" / "B999_Week0_SEMANTIC.nii.gz").touch()
            (root / "semantic_MRs" / "notes.txt").touch()
            report = audit.discover_pairs(root)
            self.assertEqual(len(report["unknown_files"]), 2)
            self.assertEqual(sum(item["code"] == "unsupported_filename" for item in report["blockers"]), 2)
            self.assertEqual(report["folders"]["image"]["nifti_count"], 5)

    def test_missing_and_duplicate_pairs_are_blockers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_names(root)
            (root / "semantic_MRs" / "B006_Week0_LFOV.nii").touch()
            (root / "semantic_labels_only" / "B037_Week0_SEMANTIC.nii.gz").unlink()
            report = audit.discover_pairs(root)
            self.assertEqual({item["code"] for item in report["blockers"]}, {"duplicate_pair", "missing_pair"})
            self.assertEqual(sum(record["pairing_valid"] for record in report["records"]), 1)

    def test_split_is_reproducible_covers_patients_and_keeps_repeated_visits_together(self):
        patients = [f"B{index:03d}" for index in range(38)]
        splits = audit.generate_patient_splits(patients + patients[:5])
        self.assertEqual(splits, audit.generate_patient_splits(reversed(patients)))
        self.assertEqual([len(splits[name]) for name in audit.SPLIT_NAMES], [26, 6, 6])
        audit.validate_patient_splits(splits, patients)
        self.assertEqual(len(set().union(*(set(values) for values in splits.values()))), 38)
        self.assertTrue(all(len(audit.generate_patient_splits(patients[:3])[name]) == 1 for name in audit.SPLIT_NAMES))

    def test_split_validator_rejects_overlap_duplicates_missing_and_empty(self):
        cases = [
            {"train": ["B001"], "validation": ["B001"], "test": ["B003"]},
            {"train": ["B001", "B001"], "validation": ["B002"], "test": ["B003"]},
            {"train": ["B001"], "validation": ["B002"], "test": ["B004"]},
            {"train": [], "validation": ["B002"], "test": ["B003"]},
        ]
        for splits in cases:
            with self.subTest(splits=splits), self.assertRaises(ValueError):
                audit.validate_patient_splits(splits, ["B001", "B002", "B003"])
        with self.assertRaises(ValueError):
            audit.generate_patient_splits(["B001", "B002"])
        with self.assertRaises(ValueError):
            audit.generate_patient_splits(["B001", "B002", "B003"], fractions=(0.8, 0.2, 0.2))


@unittest.skipIf(audit.nib is None or audit.np is None, "nibabel and NumPy required for synthetic NIfTI tests")
class NiftiAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "data"
        for folder in audit.FOLDERS.values():
            (self.root / folder).mkdir(parents=True)
        self.make_pairs()

    def save(self, patient, role, *, visit="Week0", shape=(3, 4, 5), affine=None, data=None):
        if affine is None:
            affine = audit.np.diag([1.0, 1.5, 2.0, 1.0])
        if data is None:
            data = audit.np.zeros(shape, dtype=audit.np.int16 if role == "mask" else audit.np.float32)
        path = self.root / audit.FOLDERS[role] / f"{patient}_{visit}_{audit.EXPECTED_KINDS[role]}.nii.gz"
        audit.nib.save(audit.nib.Nifti1Image(data, affine), str(path))
        return path

    def make_pairs(self):
        for patient in ("B006", "B037", "B100", "B101"):
            for role in audit.FOLDERS:
                self.save(patient, role)

    def test_complete_audit_records_headers_bounded_labels_and_patient_splits(self):
        original = audit._inspect_labels
        with mock.patch.object(audit, "_inspect_labels", wraps=original) as inspect:
            report = audit.audit_dataset(self.root)
        self.assertEqual(report["status"], "audit_complete")
        self.assertEqual(report["blockers"], [])
        self.assertEqual(report["distinct_patients"], 4)
        self.assertEqual(inspect.call_count, 3)
        self.assertEqual(report["mask_voxel_inspections_attempted"], 3)
        self.assertEqual(report["label_semantics"]["status"], "unverified")
        self.assertFalse(report["validation"]["model_evaluated_on_test"])
        first = report["records"][0]
        self.assertEqual(first["image"]["shape"], [3, 4, 5])
        self.assertEqual(first["image"]["zooms"], [1.0, 1.5, 2.0])
        self.assertEqual(first["image"]["orientation"], ["R", "A", "S"])
        self.assertFalse(Path(first["image"]["relative_path"]).is_absolute())
        self.assertEqual(first["mask"]["label_inspection"]["label_ids_and_counts"], [{"label_id": 0, "count": 60}])
        self.assertEqual(report["records"][-1]["mask"]["label_inspection"]["status"], "header_only")
        audit.validate_patient_splits(report["patient_splits"], report["patient_grouping_keys"])
        json.dumps(report, allow_nan=False)

    def test_zero_mask_limit_reads_no_mask_voxels(self):
        with mock.patch.object(audit, "_inspect_labels") as inspect:
            report = audit.audit_dataset(self.root, inspect_labels_limit=0)
        inspect.assert_not_called()
        self.assertEqual(report["status"], "audit_complete")
        with self.assertRaises(ValueError):
            audit.audit_dataset(self.root, inspect_labels_limit=4)

    def test_geometry_shape_affine_spacing_and_orientation_mismatch_block_splits(self):
        self.save("B006", "mask", shape=(4, 4, 5))
        affine = audit.np.diag([-2.0, 1.5, 2.0, 1.0])
        self.save("B037", "mask", affine=affine)
        report = audit.audit_dataset(self.root)
        mismatches = [item for item in report["blockers"] if item["code"] == "geometry_mismatch"]
        self.assertEqual(len(mismatches), 2)
        self.assertIn("shape", mismatches[0]["message"])
        self.assertIn("affine", mismatches[1]["message"])
        self.assertIn("zooms", mismatches[1]["message"])
        self.assertIn("orientation", mismatches[1]["message"])
        self.assertEqual(report["status"], "blocked")
        self.assertIsNone(report["patient_splits"])

    def test_non_3d_header_is_blocked(self):
        self.save("B006", "image", shape=(3, 4, 5, 1))
        report = audit.audit_dataset(self.root)
        self.assertIn("invalid_dimensions", {item["code"] for item in report["blockers"]})
        self.assertIsNone(report["patient_splits"])

    def test_header_rejects_nonfinite_zero_or_negative_spacing(self):
        # NIfTI writers may repair invalid zooms; inject the loaded header so
        # this checks validation rather than depending on writer behaviour.
        volume = mock.Mock()
        volume.shape = (3, 4, 5)
        volume.affine = audit.np.eye(4)
        for bad_zoom in (float("nan"), 0.0, -1.0):
            with self.subTest(zoom=bad_zoom):
                volume.header.get_zooms.return_value = (1.0, bad_zoom, 2.0)
                with mock.patch.object(audit.nib, "load", return_value=volume):
                    _, header, errors = audit._inspect_header(Path("synthetic.nii"), "synthetic.nii")
                self.assertFalse(header["header_valid"])
                self.assertIn("invalid_spacing", {item["code"] for item in errors})
                json.dumps(header, allow_nan=False)

    def test_fractional_and_nonfinite_labels_are_blocked(self):
        data = audit.np.zeros((3, 4, 5), dtype=audit.np.float32)
        data[0, 0, 0] = 0.5
        data[0, 0, 1] = audit.np.nan
        self.save("B006", "mask", data=data)
        report = audit.audit_dataset(self.root)
        codes = {item["code"] for item in report["blockers"]}
        self.assertTrue({"fractional_labels", "nonfinite_labels"}.issubset(codes))
        self.assertIsNone(report["patient_splits"])
        inspection = report["records"][0]["mask"]["label_inspection"]
        self.assertEqual(inspection["noninteger_count"], 1)
        self.assertEqual(inspection["nonfinite_count"], 1)
        self.assertIsNone(inspection["label_ids_and_counts"])
        json.dumps(report, allow_nan=False)

    def test_unknown_filename_blocks_split_even_when_all_known_pairs_are_valid(self):
        (self.root / "semantic_MRs" / "unknown.nii.gz").touch()
        report = audit.audit_dataset(self.root)
        self.assertEqual(report["status"], "blocked")
        self.assertIsNone(report["patient_splits"])

    def test_cli_writes_successful_report_and_blocked_report(self):
        output = Path(self.temporary.name) / "reports" / "dataset.json"
        with contextlib.redirect_stdout(io.StringIO()):
            code = audit.main(["--data-root", str(self.root), "--output", str(output)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.read_text())["status"], "audit_complete")
        self.save("B006", "mask", shape=(1, 2, 3))
        with contextlib.redirect_stdout(io.StringIO()):
            code = audit.main(["--data-root", str(self.root), "--output", str(output)])
        self.assertEqual(code, 2)
        self.assertIsNone(json.loads(output.read_text())["patient_splits"])


if __name__ == "__main__":
    unittest.main()
