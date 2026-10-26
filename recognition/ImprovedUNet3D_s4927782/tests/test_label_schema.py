"""Standard-library checks for metadata-only, idempotent label attachment."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "tools" / "attach_label_schema.py"
SPEC = importlib.util.spec_from_file_location("attach_label_schema", SOURCE)
attach = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(attach)


class LabelSchemaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest_path, self.schema_path = self.root / "manifest.json", self.root / "schema.json"
        self.names = dict(zip(map(str, range(6)), ["Background", "Body", "Bones", "Bladder", "Rectum", "Prostate"]))
        self.schema = {"status": "official_source_confirmed", "raw_id_to_name": self.names,
                       "source_url": "https://example.org/official-label-key", "source_sha256": "a" * 64,
                       "dataset_doi": "https://doi.org/10.example/test", "attribution": "Fixture author",
                       "acknowledgement": "Fixture data provider"}
        self.manifest = {"label_ids": list(range(6)), "num_classes": 6,
                         "raw_label_to_index": {str(index): index for index in range(6)},
                         "label_semantics": {"status": "unverified", "mapping": None},
                         "patient_splits": {"train": ["A"], "validation": ["B"], "test": ["C"]},
                         "records": [{"patient": "A", "split": "train", "cache_path": "no-array-read.npz",
                                      "normalization": {"mean": 12.5, "standard_deviation": 2.0}}],
                         "target_shape_dhw": [64, 128, 128], "audit_sha256": "b" * 64}

    def write(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.schema_path.write_text(json.dumps(self.schema), encoding="utf-8")

    def read(self):
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def test_full_official_mapping_and_all_other_fields_unchanged(self):
        self.write()
        original = copy.deepcopy(self.manifest)
        self.assertTrue(attach.attach_label_schema(self.manifest_path, self.schema_path))
        result = self.read()
        semantics = result.pop("label_semantics")
        original.pop("label_semantics")
        self.assertEqual(result, original)
        self.assertEqual(semantics["status"], "official_source_confirmed")
        self.assertEqual(semantics["raw_id_to_name"], self.names)
        self.assertEqual(semantics["class_index_to_name"], self.names)
        for field in attach.SOURCE_FIELDS:
            self.assertEqual(semantics[field], self.schema[field])

    def test_noncontiguous_raw_ids_use_cached_class_indices(self):
        self.manifest.update(label_ids=[0, 2, 5], num_classes=3,
                             raw_label_to_index={"0": 0, "2": 1, "5": 2})
        self.write()
        attach.attach_label_schema(self.manifest_path, self.schema_path)
        self.assertEqual(self.read()["label_semantics"]["class_index_to_name"],
                         {"0": "Background", "1": "Bones", "2": "Prostate"})

    def test_second_attachment_preserves_exact_bytes(self):
        self.write()
        attach.attach_label_schema(self.manifest_path, self.schema_path)
        previous = self.manifest_path.read_bytes()
        self.assertFalse(attach.attach_label_schema(self.manifest_path, self.schema_path))
        self.assertEqual(self.manifest_path.read_bytes(), previous)

    def test_missing_raw_label_or_unconfirmed_schema_rejected_without_writes(self):
        for change in ("missing", "status"):
            with self.subTest(change=change):
                if change == "missing":
                    self.schema["raw_id_to_name"] = {key: value for key, value in self.names.items() if key != "5"}
                else:
                    self.schema["raw_id_to_name"] = self.names
                    self.schema["status"] = "inference_pending_review"
                self.write()
                previous = self.manifest_path.read_bytes()
                with self.assertRaises(ValueError):
                    attach.attach_label_schema(self.manifest_path, self.schema_path)
                self.assertEqual(self.manifest_path.read_bytes(), previous)

    def test_conflicting_confirmed_schema_is_never_replaced(self):
        self.write()
        attach.attach_label_schema(self.manifest_path, self.schema_path)
        previous = self.manifest_path.read_bytes()
        self.schema["attribution"] = "Different attribution"
        self.schema_path.write_text(json.dumps(self.schema), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different source-confirmed"):
            attach.attach_label_schema(self.manifest_path, self.schema_path)
        self.assertEqual(self.manifest_path.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
