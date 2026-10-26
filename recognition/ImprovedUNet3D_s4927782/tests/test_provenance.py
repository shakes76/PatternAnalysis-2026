"""Standard-library checks for fail-before-write provenance verification."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "tools" / "record_provenance.py"
SPEC = importlib.util.spec_from_file_location("record_provenance", SOURCE)
provenance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provenance)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "modules.py").write_bytes(b"# exact source\n")
        (self.root / "manifest.json").write_bytes(b'{"patient_splits":{"train":["A"]}}\n')
        (self.root / "best.pt").write_bytes(b"opaque checkpoint bytes, not a pickle\x00\xff")
        self.expected_path = self.root / "expected.json"
        self.output = self.root / "new-directory" / "provenance.json"
        self.expected = {"training_code_commit": "a" * 40,
                         "source_file_sha256": {"modules.py": self.digest(self.root / "modules.py")},
                         "manifest_sha256": self.digest(self.root / "manifest.json")}
        self.save_expected()

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def save_expected(self):
        self.expected_path.write_text(json.dumps(self.expected), encoding="utf-8")

    def record(self, **kwargs):
        return provenance.record_provenance(self.expected_path, self.root / "manifest.json", self.output,
                                             project_root=self.root, **kwargs)

    def assert_nothing_written(self):
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.parent.exists())

    def test_success_records_measured_source_manifest_and_opaque_checkpoint(self):
        inputs = {name: (self.root / name).read_bytes() for name in ("modules.py", "manifest.json", "best.pt", "expected.json")}
        report = self.record(checkpoint_path=self.root / "best.pt")
        self.assertEqual(report, json.loads(self.output.read_text(encoding="utf-8")))
        self.assertEqual(report["source_file_sha256"], self.expected["source_file_sha256"])
        self.assertEqual(report["manifest_sha256"], self.expected["manifest_sha256"])
        self.assertEqual(report["checkpoint_sha256"], self.digest(self.root / "best.pt"))
        self.assertTrue(report["recorded_at_utc"].endswith("Z"))
        for name, original in inputs.items():
            self.assertEqual((self.root / name).read_bytes(), original)

    def test_source_mismatch_prevents_even_output_directory_creation(self):
        (self.root / "modules.py").write_bytes(b"# changed source\n")
        with self.assertRaisesRegex(ValueError, "Source digest mismatch"):
            self.record()
        self.assert_nothing_written()

    def test_traversal_absolute_and_ambiguous_paths_are_rejected_before_write(self):
        for unsafe in ("../modules.py", "/outside.py", "tools/../modules.py", "./modules.py", "a\\b.py", "a//b.py"):
            with self.subTest(path=unsafe):
                self.expected["source_file_sha256"] = {unsafe: "b" * 64}
                self.save_expected()
                with self.assertRaisesRegex(ValueError, "Unsafe project-relative"):
                    self.record()
                self.assert_nothing_written()

    def test_symlink_escaping_project_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            source = Path(outside) / "outside.py"
            source.write_bytes(b"# outside\n")
            (self.root / "escape.py").symlink_to(source)
            self.expected["source_file_sha256"] = {"escape.py": self.digest(source)}
            self.save_expected()
            with self.assertRaisesRegex(ValueError, "escapes the project root"):
                self.record()
            self.assert_nothing_written()

    def test_manifest_digest_mismatch_prevents_output_creation(self):
        (self.root / "manifest.json").write_bytes(b"{}\n")
        with self.assertRaisesRegex(ValueError, "Manifest digest mismatch"):
            self.record()
        self.assert_nothing_written()

    def test_existing_sidecar_requires_explicit_replace(self):
        self.record()
        original = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.record()
        self.assertEqual(self.output.read_bytes(), original)
        self.record(replace=True)
        self.assertTrue(self.output.is_file())

    def test_replace_cannot_overwrite_manifest_or_source(self):
        for name in ("manifest.json", "modules.py"):
            with self.subTest(output=name):
                self.output = self.root / name
                original = self.output.read_bytes()
                with self.assertRaisesRegex(ValueError, "Output must not replace"):
                    self.record(replace=True)
                self.assertEqual(self.output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
