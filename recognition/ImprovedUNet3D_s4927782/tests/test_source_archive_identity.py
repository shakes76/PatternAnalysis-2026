"""Opaque synthetic containers exercise identity joins without MRI dependencies."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zlib


SOURCE = Path(__file__).resolve().parents[1] / "tools/verify_source_archive_identity.py"
SPEC = importlib.util.spec_from_file_location("source_archive_identity", SOURCE)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class SourceArchiveIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.manifest = self.root / "manifest.json"
        self.catalog = self.root / "catalog.json"
        self.output = self.root / "new" / "audit.json"
        records, entries = [], []
        for number, patient in ((1, "A001"), (2, "B002")):
            row = {"patient": patient, "visit": "Week0", "split": "train"}
            for role in ("image", "mask"):
                value = f"opaque-{patient}-{role}".encode()
                relative = f"{patient}_{role}.nii.gz"
                (self.data / relative).write_bytes(value)
                row["source_" + role] = relative
                name = (f"semantic_MRs_anon/Case_{number:03d}_Week0_LFOV.nii.gz" if role == "image" else
                        f"semantic_labels_anon/Case_{number:03d}_Week0_SEMANTIC_LFOV.nii.gz")
                entries.append({"filename": name, "uncompressed_bytes": len(value), "crc32": f"{zlib.crc32(value):08x}", "is_directory": False})
            records.append(row)
        self.manifest.write_text(json.dumps({"partial": False, "development_only": False, "records": records}))
        self.catalog.write_text(json.dumps({"entries": entries}))

    def run_audit(self):
        return audit.verify_identity(self.manifest, self.catalog, self.data, self.output)

    def test_all_exact_candidates_and_unique_role_joins_preserve_inputs(self):
        before = {p: p.read_bytes() for p in self.data.iterdir()}
        result = self.run_audit()
        self.assertEqual(result["files_checked"], 4)
        self.assertTrue(result["all_numeric_candidates_match"])
        self.assertTrue(all(x["global_join_status"] == "unique_numeric_candidate" for x in result["files"]))
        self.assertEqual(json.loads(self.output.read_text()), result)
        self.assertTrue(all(p.read_bytes() == value for p, value in before.items()))
        with self.assertRaises(FileExistsError): self.run_audit()

    def test_swapped_images_are_found_by_global_join_without_assuming_numeric_mapping(self):
        a, b = self.data / "A001_image.nii.gz", self.data / "B002_image.nii.gz"
        av, bv = a.read_bytes(), b.read_bytes()
        a.write_bytes(bv); b.write_bytes(av)
        result = self.run_audit()
        self.assertFalse(result["all_numeric_candidates_match"])
        images = [x for x in result["files"] if x["role"] == "image"]
        self.assertTrue(all(x["global_join_status"] == "unique_non_candidate" for x in images))
        self.assertIn("Case_002_", images[0]["unique_role_global_join"])
        self.assertEqual(result["by_role"]["mask"]["numeric_candidate_matches"], 2)

    def test_modified_container_and_duplicate_fingerprint_are_honestly_reported(self):
        (self.data / "A001_image.nii.gz").write_bytes(b"different opaque gzip or NIfTI metadata")
        (self.data / "B002_mask.nii.gz").write_bytes((self.data / "A001_mask.nii.gz").read_bytes())
        result = self.run_audit()
        first_image = next(x for x in result["files"] if x["patient"] == "A001" and x["role"] == "image")
        self.assertEqual(first_image["global_join_status"], "no_match")
        masks = [x for x in result["files"] if x["role"] == "mask"]
        self.assertTrue(all(x["global_join_status"] == "ambiguous" for x in masks))
        self.assertTrue(all(x["unique_role_global_join"] is None for x in masks))
        self.assertIn("do not establish incorrect pixels", " ".join(result["limits"]))

    def test_unsafe_source_catalog_and_symlink_escape_fail_before_output_creation(self):
        original = self.manifest.read_text()
        manifest = json.loads(original)
        for unsafe in ("../outside.nii.gz", "/outside.nii.gz", "./A001_image.nii.gz", "a\\b.nii.gz"):
            with self.subTest(path=unsafe):
                manifest["records"][0]["source_image"] = unsafe
                self.manifest.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError): self.run_audit()
                self.assertFalse(self.output.parent.exists())
        self.manifest.write_text(original)
        catalog = json.loads(self.catalog.read_text()); catalog["entries"][0]["filename"] = "../escape.nii.gz"
        self.catalog.write_text(json.dumps(catalog))
        with self.assertRaises(ValueError): self.run_audit()
        self.assertFalse(self.output.parent.exists())
        catalog["entries"][0]["filename"] = "semantic_MRs_anon/Case_001_Week0_LFOV.nii.gz"
        self.catalog.write_text(json.dumps(catalog))
        outside = self.root / "outside.nii.gz"; outside.write_bytes(b"opaque outside")
        source = self.data / "A001_image.nii.gz"; source.unlink(); source.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes data root"): self.run_audit()
        self.assertFalse(self.output.parent.exists())


if __name__ == "__main__":
    unittest.main()
