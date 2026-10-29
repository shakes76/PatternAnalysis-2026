"""Synthetic fixtures only: no real patient data is stored in these tests."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "generate_samples.py"


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.data = self.base / "source"
        for split in ("train", "test"):
            for label in ("AD", "NC"):
                (self.data / "AD_NC" / split / label).mkdir(parents=True)
        self.metadata = {}
        self.output = self.base / "output"

    def add(self, name, label="AD", split="train", metadata=None):
        path = self.data / "AD_NC" / split / label / name
        # Manifest generation reads names and metadata, not image pixels.
        path.write_bytes(b"synthetic-image-placeholder")
        if metadata is not None:
            self.metadata[name.split("_")[0]] = metadata
        return path

    def run_script(self, output=None):
        source = self.data / "meta_data_with_label.json"
        source.write_text(json.dumps(self.metadata))
        before = {p: p.read_bytes() for p in self.data.rglob("*") if p.is_file()}
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--data-root", str(self.data),
             "--output-dir", str(output or self.output)],
            capture_output=True, text=True,
        )
        for p, content in before.items():
            self.assertEqual(p.read_bytes(), content, "Source data was changed")
        return result

    def read(self, name):
        with (self.output / name).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def test_maps_each_slice_without_guessing_filename_date(self):
        raw = "ADNI_001_S_0001_MR_MPRAGE_br_raw_20200102123456789_1_S9_I101.nii"
        first = self.add("101_78.jpeg", metadata={"raw": raw, "label": 2})
        self.add("101_79.jpeg")
        self.add("202_90.jpeg", label="NC", split="test",
                 metadata={"raw": "ADNI_002_S_0002_MR_I202.nii", "label": 0})
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = self.read("samples.csv")
        self.assertEqual(len(rows), 3)
        self.assertEqual(list(rows[0]), [
            "file_path", "original_split", "image_id", "patient_id",
            "scan_date", "label",
        ])
        self.assertEqual(rows[0], {
            "file_path": str(first), "original_split": "train",
            "image_id": "101", "patient_id": "001_S_0001",
            "scan_date": "", "label": "AD",
        })
        self.assertEqual(rows[1]["patient_id"], "001_S_0001")
        self.assertEqual(rows[2]["label"], "NC")
        self.assertEqual(rows[2]["original_split"], "test")
        self.assertEqual(len(self.read("samples_issues.csv")), 3)
        # A repeat run must not silently replace an existing manifest.
        self.assertNotEqual(self.run_script().returncode, 0)

    def test_preserves_rows_with_missing_metadata_and_label_conflicts(self):
        self.add("101_78.jpeg")  # no metadata
        self.add("202_78.jpeg", metadata={
            "raw": "ADNI_002_S_0002_MR_I202.nii", "label": 0,
        })  # AD folder disagrees with NC metadata
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = self.read("samples.csv")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["patient_id"], "")
        self.assertEqual(rows[0]["label"], "")
        self.assertEqual(rows[1]["label"], "")
        codes = {row["issue"] for row in self.read("samples_issues.csv")}
        self.assertTrue({"metadata_missing", "label_mismatch"} <= codes)

    def test_unparseable_filename_and_wrong_raw_image_id_are_not_guessed(self):
        self.add("unknown.jpeg")
        self.add("101_78.jpeg", metadata={
            "raw": "ADNI_001_S_0001_MR_I999.nii", "label": 2,
        })
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        by_name = {Path(r["file_path"]).name: r for r in self.read("samples.csv")}
        self.assertEqual(by_name["unknown.jpeg"]["image_id"], "")
        self.assertEqual(by_name["101_78.jpeg"]["patient_id"], "")
        codes = {row["issue"] for row in self.read("samples_issues.csv")}
        self.assertTrue({"filename_unparsed", "raw_image_id_mismatch"} <= codes)

    def test_refuses_output_inside_source_tree(self):
        self.add("101_78.jpeg")
        result = self.run_script(output=self.data / "generated")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside the source data tree", result.stderr)
        self.assertFalse((self.data / "generated").exists())


if __name__ == "__main__":
    unittest.main()
