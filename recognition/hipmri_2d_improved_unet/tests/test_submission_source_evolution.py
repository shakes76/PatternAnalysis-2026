"""Constructed, stdlib-only archive evolution checks; no model/data execution."""

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))

from verify_submission_archive import (
    Archive, HISTORICAL_SOURCE_REVISIONS,
    EVOLUTION_OLD_PREDICT_GIT_BLOB_SHA1, EVOLUTION_OLD_PREDICT_SHA256,
    EVOLUTION_PARENT, EVOLUTION_RETAINED_SOURCE_REVISION, SOURCE_EVOLUTION,
    current_index_binding, digest, historical_revision_entry, strict_json, validate_source_evolution,
    resolve_evolution_source_revision, verify_source_evolution,
)


class MemoryArchive:
    """Only caller-provided text and one read-only historical Python fixture."""

    def __init__(self, project, old, record, current, helper):
        self.project = project
        self.old = old
        self.files = {SOURCE_EVOLUTION: json.dumps(record).encode(),
                      "predict.py": current, "prediction_panels.py": helper}
        self.index = dict(self.files)
        self.tracked = set(self.files)
        location = project / SOURCE_EVOLUTION
        location.parent.mkdir(parents=True, exist_ok=True)
        location.write_bytes(self.files[SOURCE_EVOLUTION])

    def current(self, name):
        self.safe_relative(name)
        return self.files[name]

    def safe_relative(self, name):
        if name not in self.tracked:
            raise ValueError("Untracked fixture")
        return self.project / name, "project/" + name

    def command(self, *args):
        if len(args) != 3 or args[0] != "show" or args[2] != "--":
            raise ValueError("Only explicitly separated index text is supported")
        return self.index[args[1].removeprefix(":project/")]

    def blob(self, revision, name):
        if revision != EVOLUTION_RETAINED_SOURCE_REVISION or name != "predict.py":
            raise ValueError("Unexpected historical lookup")
        return self.old

    def resolve_revision(self, revision):
        entry = historical_revision_entry(revision)
        return {"requested_revision": revision, "resolved_revision": entry["revision"],
                "verified_project_tree": entry["project_tree"]}


class SourceEvolutionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        git = os.environ.get("HIPMRI_GIT_EXE") or shutil.which("git")
        if not git:
            bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe"
            git = str(bundled) if bundled.is_file() else None
        if not git:
            raise unittest.SkipTest("Git is required for the immutable historical text fixture")
        # This command reads source text only. It never imports the historical predictor.
        cls.old = subprocess.check_output([
            git, "-C", str(PROJECT), "show",
            historical_revision_entry(resolve_evolution_source_revision(EVOLUTION_PARENT, "predict.py"))["revision"]
            + ":recognition/hipmri_2d_improved_unet/predict.py", "--",
        ])
        if digest(cls.old) != EVOLUTION_OLD_PREDICT_SHA256:
            raise AssertionError("Historical predictor fixture differs")
        cls.current = b'"""Constructed display entry, never executed."""\n'
        cls.helper = b'"""Constructed panel helper, never executed."""\n'

    def setUp(self):
        self.record = {
            "schema_version": 1,
            "kind": "post_experiment_prediction_display_source_change",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "parent_revision": EVOLUTION_PARENT,
            "purpose": "Constructed source binding check, not an experiment.",
            "authorization_scope": {
                "test_display_code": True, "constructed_test_branch_checks": True,
                "final_validation_examples": True, "new_training": False,
                "real_test_inference": False,
            },
            "historical_source": {"file": "predict.py", "revision": EVOLUTION_PARENT,
                                  "sha256": EVOLUTION_OLD_PREDICT_SHA256},
            "current_source": {"file": "predict.py", "sha256": digest(self.current)},
            "new_helpers": [{"file": "prediction_panels.py", "sha256": digest(self.helper)}],
            "verification_boundary": {
                "test_entry_validation": "constructed_data_only", "real_test_entry_executed": False,
                "formal_test_results_replaced": False, "model_or_epoch_reselected": False,
            },
        }
        self.test_root = (REPO / "outputs/source_evolution_constructed_checks").resolve()
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="case_", dir=self.test_root)
        self.project = Path(self.temporary.name)

    def tearDown(self):
        target = Path(self.temporary.name).resolve()
        self.assertEqual(target.parent, self.test_root)
        self.assertTrue(target.name.startswith("case_"))
        self.temporary.cleanup()

    def archive(self, record=None):
        return MemoryArchive(self.project, self.old, record or self.record, self.current, self.helper)

    def latest(self):
        return {"predict.py": (EVOLUTION_OLD_PREDICT_SHA256, "2e4ba13a4780f38385fc32d77737d24fd8eae33b")}

    def test_fixed_record_binds_old_current_helper_and_index(self):
        result = verify_source_evolution(self.archive(), self.latest())
        self.assertEqual(result["historical_predictor_sha256"], digest(self.old))
        self.assertEqual(result["current_source"]["current_lf_sha256"], digest(self.current))
        self.assertEqual(result["new_helpers"][0]["current_lf_sha256"], digest(self.helper))
        self.assertTrue(result["old_frozen_source_bindings_preserved"])
        self.assertTrue(result["other_latest_frozen_files_require_unchanged_content"])
        resolution = result["historical_predictor_resolution"]
        self.assertEqual(resolution["requested_legacy_revision"], EVOLUTION_PARENT)
        mapped = historical_revision_entry(EVOLUTION_RETAINED_SOURCE_REVISION)
        self.assertEqual(resolution["recorded_retained_revision"], EVOLUTION_RETAINED_SOURCE_REVISION)
        self.assertEqual(resolution["resolved_retained_revision"], mapped["revision"])
        self.assertEqual(resolution["verified_project_subtree"], mapped["project_tree"])
        self.assertEqual(resolution["resolved_git_blob_sha1"], EVOLUTION_OLD_PREDICT_GIT_BLOB_SHA1)
        self.assertEqual(resolution["resolved_source_sha256"], digest(self.old))
        self.assertEqual(resolution["source_bytes"], 5514)
        self.assertFalse(resolution["original_record_changed"])
        self.assertFalse(resolution["legacy_commit_lookup_required"])

    def test_legacy_resolution_rejects_other_files_and_paths(self):
        for name in ("dataset.py", "modules.py", "../predict.py", "data/predict.py"):
            with self.subTest(file=name), self.assertRaises(ValueError):
                resolve_evolution_source_revision(EVOLUTION_PARENT, name)

    def test_legacy_resolution_rejects_other_revisions(self):
        self.assertEqual(resolve_evolution_source_revision(EVOLUTION_PARENT, "predict.py"),
                         EVOLUTION_RETAINED_SOURCE_REVISION)
        for revision in ("0" * 40, EVOLUTION_RETAINED_SOURCE_REVISION, "cfdf15a", None):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                resolve_evolution_source_revision(revision, "predict.py")

    def test_identity_revision_map_accepts_only_six_exact_original_ids(self):
        self.assertEqual(len(HISTORICAL_SOURCE_REVISIONS), 6)
        for revision, expected in HISTORICAL_SOURCE_REVISIONS.items():
            with self.subTest(revision=revision):
                entry = historical_revision_entry(revision)
                self.assertEqual(entry, expected)
                entry["revision"] = "0" * 40
                self.assertEqual(historical_revision_entry(revision), expected)
        mapped_new = historical_revision_entry(EVOLUTION_RETAINED_SOURCE_REVISION)["revision"]
        for revision in ("0" * 40, EVOLUTION_PARENT, mapped_new, "0211785", None, 5):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                historical_revision_entry(revision)

    def test_archive_maps_blob_through_verified_project_subtree(self):
        archive = Archive.__new__(Archive)
        archive.revision = "f" * 40
        archive.prefix = "project"
        archive.revision_resolutions = {}
        archive.blob_cache = {}
        archive.safe_relative = lambda name: (None, "project/" + name)
        mapped = historical_revision_entry(EVOLUTION_RETAINED_SOURCE_REVISION)
        calls = []

        def command(*args):
            calls.append(args)
            if args == ("rev-parse", mapped["revision"] + ":project"):
                return (mapped["project_tree"] + "\n").encode()
            if args == ("show", mapped["revision"] + ":project/predict.py", "--"):
                return self.old
            raise AssertionError("Unexpected Git access: " + repr(args))

        archive.command = command
        self.assertEqual(archive.blob(EVOLUTION_RETAINED_SOURCE_REVISION, "predict.py"), self.old)
        resolution = archive.resolve_revision(EVOLUTION_RETAINED_SOURCE_REVISION)
        self.assertEqual(resolution["resolved_revision"], mapped["revision"])
        self.assertEqual(resolution["verified_project_tree"], mapped["project_tree"])
        self.assertFalse(resolution["original_record_changed"])
        self.assertFalse(resolution["original_commit_lookup_required"])
        self.assertEqual(len(calls), 2)
        archive.blob(EVOLUTION_RETAINED_SOURCE_REVISION, "predict.py")
        self.assertEqual(len(calls), 2)
        current = archive.resolve_revision(archive.revision)
        self.assertEqual(current["resolution_kind"], "current_head")
        self.assertEqual(current["resolved_revision"], archive.revision)
        self.assertEqual(len(calls), 2)

    def test_archive_rejects_wrong_subtree_and_unknown_legacy_before_blob_read(self):
        archive = Archive.__new__(Archive)
        archive.revision = "f" * 40
        archive.prefix = "project"
        archive.revision_resolutions = {}
        archive.blob_cache = {}
        archive.safe_relative = lambda name: (None, "project/" + name)
        calls = []

        def wrong_tree(*args):
            calls.append(args)
            self.assertEqual(args[0], "rev-parse", "A source blob was read before its tree was verified")
            return b"0000000000000000000000000000000000000000\n"

        archive.command = wrong_tree
        with self.assertRaises(ValueError):
            archive.blob(EVOLUTION_RETAINED_SOURCE_REVISION, "predict.py")
        self.assertEqual(len(calls), 1)
        self.assertEqual(archive.blob_cache, {})
        self.assertEqual(archive.revision_resolutions, {})
        calls.clear()
        with self.assertRaises(ValueError):
            archive.blob("0" * 40, "predict.py")
        self.assertEqual(calls, [])

    def test_unlisted_frozen_file_cannot_be_authorized(self):
        for name in ("dataset.py", "train.py", "modules.py", "evaluate_frozen_test.py"):
            with self.subTest(file=name):
                changed = copy.deepcopy(self.record)
                changed["current_source"]["file"] = name
                with self.assertRaises(ValueError):
                    validate_source_evolution(changed)

    def test_wrong_parent_or_old_source_identity_rejected(self):
        changes = [("parent_revision", "0" * 40), ("historical_source", "file"),
                   ("historical_source", "revision"), ("historical_source", "sha256")]
        for section, field in changes:
            with self.subTest(section=section, field=field):
                changed = copy.deepcopy(self.record)
                if section == "parent_revision":
                    changed[section] = field
                else:
                    changed[section][field] = "0" * 64
                with self.assertRaises(ValueError):
                    validate_source_evolution(changed)

    def test_unapproved_or_traversing_helper_rejected(self):
        for name in ("../prediction_panels.py", "/prediction_panels.py", "other.py", "data/mask.py"):
            with self.subTest(file=name):
                changed = copy.deepcopy(self.record)
                changed["new_helpers"][0]["file"] = name
                with self.assertRaises(ValueError):
                    validate_source_evolution(changed)
        changed = copy.deepcopy(self.record)
        changed["new_helpers"].append(dict(changed["new_helpers"][0]))
        with self.assertRaises(ValueError):
            validate_source_evolution(changed)

    def test_missing_extra_and_duplicate_fields_rejected(self):
        for modifier in (lambda value: value.pop("purpose"),
                         lambda value: value.update({"skip_all_frozen_sources": True}),
                         lambda value: value["current_source"].update({"anything_else": True})):
            changed = copy.deepcopy(self.record)
            modifier(changed)
            with self.assertRaises(ValueError):
                validate_source_evolution(changed)
        with self.assertRaises(ValueError):
            strict_json(b'{"schema_version":1,"schema_version":1}')

    def test_real_test_training_reselection_and_result_replacement_rejected(self):
        paths = [("authorization_scope", "new_training"), ("authorization_scope", "real_test_inference"),
                 ("verification_boundary", "real_test_entry_executed"),
                 ("verification_boundary", "formal_test_results_replaced"),
                 ("verification_boundary", "model_or_epoch_reselected")]
        for section, field in paths:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.record)
                changed[section][field] = True
                with self.assertRaises(ValueError):
                    validate_source_evolution(changed)

    def test_numeric_false_and_missing_timezone_rejected(self):
        changed = copy.deepcopy(self.record)
        changed["authorization_scope"]["new_training"] = 0
        with self.assertRaises(ValueError):
            validate_source_evolution(changed)
        changed = copy.deepcopy(self.record)
        changed["recorded_at"] = "2026-10-26T15:00:00"
        with self.assertRaises(ValueError):
            validate_source_evolution(changed)

    def test_untracked_or_changed_helper_rejected(self):
        archive = self.archive()
        archive.tracked.remove("prediction_panels.py")
        with self.assertRaises(ValueError):
            verify_source_evolution(archive, self.latest())
        archive = self.archive()
        archive.files["prediction_panels.py"] += b"# changed\n"
        archive.index["prediction_panels.py"] = archive.files["prediction_panels.py"]
        with self.assertRaises(ValueError):
            verify_source_evolution(archive, self.latest())

    def test_current_predictor_content_change_is_not_a_newline_change(self):
        archive = self.archive()
        archive.files["predict.py"] += b"# changed\n"
        archive.index["predict.py"] = archive.files["predict.py"]
        with self.assertRaises(ValueError):
            verify_source_evolution(archive, self.latest())

    def test_unstaged_source_or_record_rejected(self):
        archive = self.archive()
        archive.index["predict.py"] += b"# unstaged\n"
        with self.assertRaises(ValueError):
            verify_source_evolution(archive, self.latest())
        archive = self.archive()
        archive.index[SOURCE_EVOLUTION] += b" "
        with self.assertRaises(ValueError):
            verify_source_evolution(archive, self.latest())

    def test_old_git_or_frozen_binding_tampering_rejected(self):
        archive = self.archive()
        archive.old += b"# historical change\n"
        with self.assertRaises(ValueError):
            verify_source_evolution(archive, self.latest())
        latest = self.latest()
        latest["predict.py"] = ("0" * 64, EVOLUTION_PARENT)
        with self.assertRaises(ValueError):
            verify_source_evolution(self.archive(), latest)

    def test_lf_only_current_binding_accepts_checkout_crlf_without_text_change(self):
        archive = self.archive()
        archive.files["predict.py"] = archive.files["predict.py"].replace(b"\n", b"\r\n")
        result = current_index_binding(archive, "predict.py", digest(self.current))
        self.assertEqual(result["current_lf_sha256"], digest(self.current))
        self.assertTrue(result["current_matches_index"])


if __name__ == "__main__":
    unittest.main()
