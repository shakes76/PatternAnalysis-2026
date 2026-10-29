import copy
import csv
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split_patients import (FIELDS, QUOTAS, build, run, sha256, code_version)


def fixture():
    scans, samples = [], []
    for n in range(680):
        patient, image = f"001_S_{n:04}", str(10000+n)
        label = "AD" if n < 221 else "NC"
        scans.append(dict(patient_id=patient, selected_image_id=image, label=label,
                          candidate_scan_count="1"))
        for i in range(20):
            samples.append(dict(zip(FIELDS, (f"/synthetic/{image}_{i}.jpeg",
                                            "train" if i % 2 else "test",
                                            image, patient, "", label))))
    return scans, samples


class SplitTests(unittest.TestCase):
    def test_normal_exact_quotas_and_complete_join(self):
        scans, samples = fixture()
        patients, slices, checks = build(scans, samples)
        self.assertEqual(len(patients), 680)
        self.assertEqual(len(slices), 13600)
        self.assertTrue(all(checks.values()))
        for label in ("AD", "NC"):
            for split, count in QUOTAS[label].items():
                self.assertEqual(sum(p["label"] == label and p["split"] == split
                                     for p in patients), count)

    def test_repeat_shuffle_and_ignore_original_split(self):
        scans, samples = fixture()
        first = build(scans, samples)
        self.assertEqual(first, build(scans, samples))
        random.Random(3).shuffle(scans)
        random.Random(4).shuffle(samples)
        self.assertEqual(first, build(scans, samples))
        for row in samples:
            row["original_split"] = "train"
        self.assertEqual(first[0], build(scans, samples)[0])
        rng = random.Random(42)
        expected = {}
        for label in ("AD", "NC"):
            group = sorted(r["patient_id"] for r in scans if r["label"] == label)
            rng.shuffle(group)
            start = 0
            for split, count in QUOTAS[label].items():
                expected.update(dict.fromkeys(group[start:start+count], split))
                start += count
        self.assertEqual(expected, {p["patient_id"]: p["split"] for p in first[0]})

    def test_duplicate_patient_rejected(self):
        scans, samples = fixture()
        scans[-1] = dict(scans[0])
        with self.assertRaisesRegex(ValueError, "duplicate patient"):
            build(scans, samples)

    def test_label_or_image_conflict_rejected(self):
        for field, value in (("label", "NC"), ("image_id", "99999")):
            scans, samples = fixture()
            samples[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "mismatch"):
                build(scans, samples)

    def test_missing_association_rejected(self):
        scans, samples = fixture()
        samples[0]["patient_id"] = "999_S_9999"
        with self.assertRaisesRegex(ValueError, "unmatched patient"):
            build(scans, samples)

    def test_abnormal_slice_count_rejected(self):
        scans, samples = fixture()
        samples.pop()
        with self.assertRaisesRegex(ValueError, "20 slices"):
            build(scans, samples)

    def test_duplicate_path_or_slice_number_rejected(self):
        for kind in ("path", "slice"):
            scans, samples = fixture()
            samples[1]["file_path"] = (samples[0]["file_path"] if kind == "path"
                                       else "/another/10000_0.jpeg")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                build(scans, samples)

    def test_hash_guard_and_output_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "selection"
            root.mkdir()
            scans, samples = fixture()
            for name, rows in (("selected_scans", scans), ("selected_samples", samples)):
                with (root / f"{name}.csv").open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader(); w.writerows(rows)
            summary = {"status": "passed", **{
                name + "_sha256": sha256(root / (name + ".csv"))
                for name in ("selected_scans", "selected_samples")}}
            (root / "selection_summary.json").write_text(json.dumps(summary))
            result = run(root, Path(tmp) / "output")
            self.assertTrue(all(result["validation"].values()))
            self.assertEqual(result["counts"]["train"]["patients"], 408)
            self.assertNotIn("001_S_0000", json.dumps(result))
            with self.assertRaises(FileExistsError):
                run(root, Path(tmp) / "output")
            with (root / "selected_scans.csv").open("a") as f:
                f.write("\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                run(root, Path(tmp) / "bad-output")
            self.assertFalse((Path(tmp) / "bad-output").exists())

    def test_version_reflects_actual_git_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); file = root / "script.py"
            def git(*args):
                subprocess.run(["git", "-C", tmp, *args], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            git("init")
            file.write_text("print(1)\n")
            git("add", "script.py")
            git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "commit", "-m", "Fixture")
            self.assertTrue(code_version(file)["committed_script"])
            self.assertFalse(code_version(file)["working_tree_dirty"])
            file.write_text("print(2)\n")
            self.assertFalse(code_version(file)["committed_script"])
            self.assertTrue(code_version(file)["working_tree_dirty"])


if __name__ == "__main__":
    unittest.main()
