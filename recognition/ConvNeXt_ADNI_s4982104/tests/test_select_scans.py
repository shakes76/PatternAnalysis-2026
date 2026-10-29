import csv
import random
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from select_scans import check_scans, choose_scans, run, FIELDS


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "images").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def scan(self, image_id="10", patient="001_S_0001", label="AD", count=20):
        rows = []
        for i in range(count):
            path = self.root / "images" / f"{image_id}_{i}.jpeg"
            Image.new("L", (16, 12)).save(path)
            rows.append(dict(zip(FIELDS, (str(path), "train", image_id,
                                         patient, "", label))))
        return rows

    def test_complete_decode_and_dimensions_are_not_exclusions(self):
        rows = self.scan()
        Image.new("RGB", (18, 14)).save(rows[0]["file_path"])
        checks = check_scans(rows)
        self.assertTrue(checks[0]["qualified"])
        self.assertEqual(checks[0]["readable_count"], 20)
        self.assertEqual(len(checks[0]["dimensions"]), 2)
        self.assertEqual(checks[0]["modes"], ["L", "RGB"])

    def test_missing_corrupt_and_duplicate_slices_exclude_whole_scan(self):
        for kind in ("missing", "corrupt", "duplicate", "short", "truncated"):
            with self.subTest(kind=kind):
                rows = self.scan()
                if kind == "missing":
                    Path(rows[0]["file_path"]).unlink()
                elif kind == "corrupt":
                    Path(rows[0]["file_path"]).write_bytes(b"not a jpeg")
                elif kind == "truncated":
                    p = Path(rows[0]["file_path"])
                    p.write_bytes(p.read_bytes()[:-10])
                elif kind == "duplicate":
                    rows[-1] = dict(rows[0])
                else:
                    rows.pop()
                self.assertFalse(check_scans(rows)[0]["qualified"])

    def test_patient_and_label_conflicts_fail(self):
        rows = self.scan()
        rows[0]["patient_id"] = "002_S_0002"
        self.assertFalse(check_scans(rows)[0]["qualified"])
        rows = self.scan() + self.scan("11", label="NC")
        self.assertTrue(all(not c["qualified"] for c in check_scans(rows)))

    def test_one_rng_numeric_order_and_shuffle_invariance(self):
        rows = (self.scan("100") + self.scan("2") +
                self.scan("11", "002_S_0002", "NC") +
                self.scan("3", "002_S_0002", "NC"))
        checks = check_scans(rows)
        selected, samples = choose_scans(rows, checks, 42)
        rng = random.Random(42)
        expected = [rng.choice(["2", "100"]), rng.choice(["3", "11"])]
        self.assertEqual([r["selected_image_id"] for r in selected], expected)
        random.Random(99).shuffle(rows)
        self.assertEqual(choose_scans(rows, list(reversed(checks)), 42),
                         (selected, samples))
        self.assertEqual(len(samples), 40)

    def test_no_qualified_candidate_blocks_selection_but_saves_checks(self):
        rows = self.scan(count=19)
        source = self.root / "samples.csv"
        with source.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        out = self.root / "output"
        summary = run(source, out, 42)
        self.assertEqual(summary["status"], "blocked")
        self.assertEqual(summary["patients_without_qualified_scans"], 1)
        self.assertTrue((out / "scan_checks.csv").exists())
        self.assertFalse((out / "selected_scans.csv").exists())
        with self.assertRaises(FileExistsError):
            run(source, out, 42)

    def test_good_run_counts_subset_and_blank_dates(self):
        rows = self.scan()
        source = self.root / "samples.csv"
        with source.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        summary = run(source, self.root / "out", 42)
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["selected_patient_count"], 1)
        self.assertEqual(summary["selected_slice_count"], 20)
        self.assertTrue(all(summary["validation"].values()))


if __name__ == "__main__":
    unittest.main()
