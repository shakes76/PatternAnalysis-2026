"""Verify saved text evidence without importing project code or reading data/models.

Historical hashes describe original working-tree bytes. Git stored some of those
files with LF endings. A historical digest is accepted only for exact bytes or
for the single LF-to-CRLF reconstruction; JSON values are never reserialized.
The current checkout is compared separately to the historical Git text after
line-ending normalization. No evidence, source file, or Git state is modified.
An explicitly requested --output is a new, non-overwriting JSON audit record.
A fixed, hash-bound post-experiment predictor change may be recorded separately.
Historical records retain their original revision IDs. Six explicitly listed
revisions resolve to identity-rewritten commits only after their project subtree
matches the pinned original subtree; each original per-file hash is then checked.
The old predictor's documentation parent first resolves to its identical report
snapshot source. No original evidence record is modified and unknown historical
revision lookups are rejected, even when an old object happens to exist locally.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit


REPORT_SNAPSHOT = "0211785eaf2b96f4165436cf29ee5addb763078c"
SUMMARY = "evidence/final_report_summary_20261026.json"
VERIFICATION = "evidence/final_report_verification_20261026.json"
SOURCE_EVOLUTION = "evidence/submission_source_evolution_20261026.json"
EVOLUTION_PARENT = "cfdf15a6e5c0691a299cde9f8a6a6fc64328cbd3"
EVOLUTION_OLD_PREDICT_SHA256 = "3c8bda47ecac7392cd53dd8a95823881fd2131952403c451c24f9e8c292dbbef"
EVOLUTION_RETAINED_SOURCE_REVISION = REPORT_SNAPSHOT
EVOLUTION_OLD_PREDICT_GIT_BLOB_SHA1 = "daa54108c09f74790d311b645b30510f80454e2e"
EVOLUTION_OLD_PREDICT_BYTES = 5514
HISTORICAL_SOURCE_REVISIONS = {
    "0211785eaf2b96f4165436cf29ee5addb763078c": {
        "revision": "02d56f946ebb097dd3eaf47052f1290332297865",
        "project_tree": "f27f5e10baf2e50c49c46747f8e30ba2714a803a"},
    "7a8ffc221986e98f664990c1b441684ed4475d1c": {
        "revision": "95de06d1a7a0fc10fe80295c85f9e4fa6c022e98",
        "project_tree": "d34081e96e837c87cfc4a1940d52ccff99cafda0"},
    "caf189540331e4c1b914ba18e194d98d08f01f1c": {
        "revision": "c466dd8105d0650bd864a06e8c01dff15562c352",
        "project_tree": "d90711d80d623f0f5ef47bf5e47ffed43631b874"},
    "8a33707070edb298902ec34f37450228e67541cc": {
        "revision": "eceeee31da033f2963d340e58ef998f156a3929e",
        "project_tree": "19fcb9705bfcde15a892f814ca6d8cea615b6434"},
    "720452518b28c2a85f4bbded4c5802e90982f429": {
        "revision": "be723d76bde3c5440a532c583ccebaae6e4f5f55",
        "project_tree": "b87d2c5c16a8b9059f15bba5061496b3ad1b2d0b"},
    "2e4ba13a4780f38385fc32d77737d24fd8eae33b": {
        "revision": "706f4a2c37b4e0545133c571d18cb175d5c05f5b",
        "project_tree": "b5ab4c82ebd5b52e01bed718680ecc2357fa3129"},
}
TEXT_SUFFIXES = {".py", ".json", ".md", ".txt"}
MODEL_SUFFIXES = (".nii", ".nii.gz", ".pt", ".pth", ".ckpt", ".h5", ".hdf5", ".onnx",
                  ".safetensors", ".npy", ".npz", ".dcm", ".nrrd", ".mha", ".mhd")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def historical_revision_entry(requested_revision):
    """Return one fixed metadata-rewrite mapping; never infer missing revisions."""
    require(isinstance(requested_revision, str) and re.fullmatch(r"[0-9a-f]{40}", requested_revision),
            "Only full Git commit IDs are accepted")
    require(requested_revision in HISTORICAL_SOURCE_REVISIONS,
            "Unknown historical revision; no unlisted source fallback is allowed")
    return dict(HISTORICAL_SOURCE_REVISIONS[requested_revision])


def lf(data):
    """Change CRLF pairs only; reject ambiguous remaining carriage returns."""
    result = data.replace(b"\r\n", b"\n")
    require(b"\r" not in result, "Lone CR is not an accepted line-ending conversion")
    return result


def historical_match(data, expected):
    require(re.fullmatch(r"[0-9a-f]{64}", expected or ""), "Invalid SHA256 binding")
    if digest(data) == expected:
        return "exact_original_bytes"
    # This candidate changes newline bytes only, never whitespace, text or values.
    if b"\r" not in data and digest(data.replace(b"\n", b"\r\n")) == expected:
        return "restored_historical_crlf_from_git_lf"
    raise ValueError("Historical SHA256 mismatch after exact/LF-to-CRLF checks")


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def reject(value):
        raise ValueError("Non-finite JSON value: " + value)

    return json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=reject)


class DocumentLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if value and ((tag == "a" and key == "href") or (tag == "img" and key == "src")):
                self.targets.append(value)


class Archive:
    def __init__(self, project, git, simulate_lf=False):
        self.project = project.resolve()
        self.git = git
        self.root = Path(self.command("rev-parse", "--show-toplevel").decode().strip()).resolve()
        require(self.project.is_relative_to(self.root), "Project is outside Git checkout")
        self.prefix = self.project.relative_to(self.root).as_posix()
        self.revision = self.command("rev-parse", "HEAD").decode().strip()
        self.tracked = set(self.command("ls-files", "--full-name", "-z").decode("utf-8").rstrip("\0").split("\0"))
        # Keep this optional simulation inside the explicitly selected workspace.
        self.temp = (tempfile.TemporaryDirectory(prefix="submission_archive_lf_", dir=self.project)
                     if simulate_lf else None)
        self.simulated_files = set()
        self.records = []
        self.blob_cache = {}
        self.revision_resolutions = {}

    def close(self):
        if self.temp:
            temporary = Path(self.temp.name).resolve()
            require(temporary.parent == self.project and temporary.name.startswith("submission_archive_lf_"),
                    "Temporary cleanup target escapes the selected project")
            self.temp.cleanup()

    def command(self, *args):
        # The command surface is deliberately read-only, with no shell evaluation.
        require(args[0] in {"rev-parse", "ls-files", "show"}, "Unsupported Git operation")
        result = subprocess.run([self.git, "-C", str(self.project), *args], check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout

    def safe_relative(self, name, *, text=True):
        require(isinstance(name, str) and name and "\\" not in name and ":" not in name,
                "Only project-relative POSIX paths are accepted")
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts,
                "Path traversal or absolute path rejected")
        require(relative.parts[0].lower() not in {"data", "outputs", "checkpoints", ".git", ".venv"},
                "Data/model/output locations are outside this text audit")
        require(not name.lower().endswith(MODEL_SUFFIXES), "Data/model path rejected")
        if text:
            require(relative.suffix.lower() in TEXT_SUFFIXES, "Only allowlisted text is readable")
        path = self.project.joinpath(*relative.parts)
        require(path.resolve().is_relative_to(self.project), "Symlink or path escapes project")
        repo_path = self.prefix + "/" + relative.as_posix()
        require(repo_path in self.tracked, "Evidence/source is not tracked: " + name)
        return path, repo_path

    def current(self, name):
        path, _ = self.safe_relative(name)
        data = path.read_bytes()
        require(len(data) <= 16 * 1024 * 1024 and b"\0" not in data, "Non-text or oversized input")
        data.decode("utf-8")
        if self.temp:
            temporary = Path(self.temp.name).joinpath(*PurePosixPath(name).parts)
            temporary.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(lf(data))
            self.simulated_files.add(name)
            return temporary.read_bytes()
        return data

    def resolve_revision(self, revision):
        """Verify the exact historical project tree before reading mapped text."""
        require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision),
                "Only full Git commit IDs are accepted")
        if revision == self.revision:
            return {"requested_revision": revision, "resolved_revision": revision,
                    "verified_project_tree": None, "resolution_kind": "current_head"}
        if revision not in self.revision_resolutions:
            mapped = historical_revision_entry(revision)
            tree = self.command("rev-parse", mapped["revision"] + ":" + self.prefix).decode().strip()
            require(tree == mapped["project_tree"],
                    "Identity-rewritten historical project subtree differs: " + revision)
            self.revision_resolutions[revision] = {
                "requested_revision": revision, "resolved_revision": mapped["revision"],
                "verified_project_tree": tree,
                "resolution_kind": "explicit_identity_rewrite_with_identical_project_subtree",
                "original_record_changed": False, "original_commit_lookup_required": False}
        return dict(self.revision_resolutions[revision])

    def blob(self, revision, name):
        _, relative = self.safe_relative(name)
        resolution = self.resolve_revision(revision)
        key = (revision, name)
        if key not in self.blob_cache:
            # The separator prevents Git from probing a long Windows revision:path as a filename.
            data = self.command("show", resolution["resolved_revision"] + ":" + relative, "--")
            require(len(data) <= 16 * 1024 * 1024 and b"\0" not in data, "Non-text Git blob rejected")
            data.decode("utf-8")
            self.blob_cache[key] = data
        return self.blob_cache[key]

    def bound(self, name, expected, *, revision=REPORT_SNAPSHOT, role="saved_evidence",
              current_must_match=True):
        original_blob = self.blob(revision, name)
        historical_status = historical_match(original_blob, expected)
        current = self.current(name)
        head_blob = self.blob(self.revision, name)
        same_current = lf(current) == lf(original_blob)
        if current_must_match:
            require(same_current, "Current content changed beyond line endings: " + name)
            require(lf(head_blob) == lf(original_blob), "HEAD content differs from snapshot: " + name)
        try:
            current_status = historical_match(current, expected)
        except ValueError:
            current_status = ("checkout_line_endings_only_git_binding_verified" if same_current
                              else "different_from_historical_snapshot")
        row = {"file": name, "role": role, "expected_original_sha256": expected,
               "historical_revision": revision, "historical_git_blob_sha256": digest(original_blob),
               "resolved_historical_revision": self.resolve_revision(revision)["resolved_revision"],
               "verified_historical_project_tree": self.resolve_revision(revision)["verified_project_tree"],
               "historical_git_blob_lf_sha256": digest(lf(original_blob)),
               "historical_binding": historical_status,
               "current_sha256": digest(current), "current_lf_sha256": digest(lf(current)),
               "head_git_blob_sha256": digest(head_blob),
               "head_git_blob_lf_sha256": digest(lf(head_blob)),
               "current_binding": current_status, "current_same_text_as_historical": same_current,
               "current_required_unchanged": current_must_match}
        self.records.append(row)
        return strict_json(original_blob) if name.endswith(".json") else original_blob

    def links(self, name):
        content = self.current(name).decode("utf-8")
        parser = DocumentLinks()
        parser.feed(content)
        targets = parser.targets + re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content)
        checked = []
        skipped = 0
        for target in targets:
            target = target.strip().strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme:
                require(parsed.scheme.lower() in {"https", "http", "mailto"},
                        "Nonportable documentation URL scheme: " + target)
                skipped += 1
                continue
            if parsed.netloc or not parsed.path:
                skipped += 1
                continue
            relative = unquote(parsed.path)
            require("\\" not in relative and ":" not in relative and not relative.startswith("/"),
                    "Nonportable local documentation link: " + target)
            candidate = (self.project / name).parent.joinpath(relative).resolve()
            require(candidate.is_relative_to(self.project), "Documentation link escapes project")
            require(candidate.exists(), "Broken local documentation link: " + target)
            rel = candidate.relative_to(self.root).as_posix()
            require(rel in self.tracked, "Documentation link is not archived in Git: " + target)
            checked.append(target)
        return {"file": name, "checked_local_links": checked, "external_or_anchor_links_skipped": skipped,
                "asset_contents_read": False}


def matcher_checks():
    original = b'{"score": 0.5}\r\n'
    target = digest(original)
    require(historical_match(original, target) == "exact_original_bytes", "Exact-byte check failed")
    require(historical_match(original.replace(b"\r\n", b"\n"), target).startswith("restored_"),
            "Historical CRLF reconstruction failed")
    rejected = 0
    for changed in (b'{"score": 0.6}\n', b'{"score": 0.50}\n', b'{"score":  0.5}\n'):
        try:
            historical_match(changed, target)
        except ValueError:
            rejected += 1
    require(rejected == 3, "Content change was incorrectly accepted as a newline change")
    return {"exact_original_bytes": True, "lf_to_historical_crlf": True,
            "changed_number_rejected": True, "changed_numeric_spelling_rejected": True,
            "changed_whitespace_rejected": True}



def exact_keys(value, keys, label):
    require(isinstance(value, dict) and set(value) == set(keys),
            "Unexpected or missing source-evolution fields: " + label)


def validate_source_evolution(record):
    """Accept only this declared predictor/helper revision and execution scope.

    This record documents a later code capability. It does not authorize running
    test data, revise the historical frozen protocol, or replace any test score.
    """
    exact_keys(record, {"schema_version", "kind", "recorded_at", "parent_revision",
                        "purpose", "authorization_scope", "historical_source",
                        "current_source", "new_helpers", "verification_boundary"}, "record")
    require(type(record["schema_version"]) is int and record["schema_version"] == 1,
            "Unsupported source-evolution schema")
    require(record["kind"] == "post_experiment_prediction_display_source_change",
            "Unexpected source-evolution kind")
    require(record["parent_revision"] == EVOLUTION_PARENT,
            "Source-evolution parent must be the pre-display submission commit")
    require(isinstance(record["purpose"], str) and 0 < len(record["purpose"].strip()) <= 2000,
            "A bounded nonempty source-evolution purpose is required")
    require(isinstance(record["recorded_at"], str), "Source-evolution time must be text")
    recorded_at = datetime.fromisoformat(record["recorded_at"])
    require(recorded_at.tzinfo is not None, "Source-evolution time must include a timezone")
    scope = record["authorization_scope"]
    exact_keys(scope, {"test_display_code", "constructed_test_branch_checks",
                       "final_validation_examples", "new_training", "real_test_inference"}, "scope")
    required_scope = {"test_display_code": True, "constructed_test_branch_checks": True,
                      "final_validation_examples": True, "new_training": False,
                      "real_test_inference": False}
    require(all(type(scope[key]) is bool and scope[key] is expected
                for key, expected in required_scope.items()), "Source-evolution execution scope differs")
    old = record["historical_source"]
    exact_keys(old, {"file", "revision", "sha256"}, "historical_source")
    require(old == {"file": "predict.py", "revision": EVOLUTION_PARENT,
                    "sha256": EVOLUTION_OLD_PREDICT_SHA256},
            "Only the exact historical predictor may evolve")
    current = record["current_source"]
    exact_keys(current, {"file", "sha256"}, "current_source")
    require(current["file"] == "predict.py", "Only predict.py may change after the frozen run")
    require(isinstance(current["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", current["sha256"])
            and current["sha256"] != EVOLUTION_OLD_PREDICT_SHA256,
            "Current predictor requires its distinct actual SHA256")
    helpers = record["new_helpers"]
    require(isinstance(helpers, list) and len(helpers) == 1,
            "Exactly the declared display helper is required")
    exact_keys(helpers[0], {"file", "sha256"}, "helper")
    require(helpers[0]["file"] == "prediction_panels.py", "Unapproved display helper path")
    require(isinstance(helpers[0]["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", helpers[0]["sha256"]), "Invalid display helper SHA256")
    boundary = record["verification_boundary"]
    exact_keys(boundary, {"test_entry_validation", "real_test_entry_executed",
                          "formal_test_results_replaced", "model_or_epoch_reselected"}, "boundary")
    require(boundary["test_entry_validation"] == "constructed_data_only",
            "Only constructed checks may validate the new test display entry")
    require(all(type(boundary[key]) is bool and boundary[key] is False for key in
                ("real_test_entry_executed", "formal_test_results_replaced", "model_or_epoch_reselected")),
            "Source evolution cannot claim a real-test rerun, result replacement or reselection")
    return record


def current_index_binding(archive, name, expected):
    """Bind current LF source to the index as well as its declared SHA256."""
    current = archive.current(name)
    _, repo_path = archive.safe_relative(name)
    index = archive.command("show", ":" + repo_path, "--")
    require(len(index) <= 16 * 1024 * 1024 and b"\0" not in index, "Invalid source index bytes")
    index.decode("utf-8")
    require(lf(current) == lf(index), "Evolved source differs from the Git index: " + name)
    require(digest(lf(current)) == expected, "Evolved source SHA256 mismatch: " + name)
    return {"file": name, "declared_lf_sha256": expected,
            "current_sha256": digest(current), "current_lf_sha256": digest(lf(current)),
            "index_lf_sha256": digest(lf(index)), "current_matches_index": True}


def resolve_evolution_source_revision(requested_revision, name):
    """Resolve one legacy documentation-parent lookup, never other run sources.

    The source record keeps the actual old parent. Its predictor is byte-for-byte
    the same Git blob in REPORT_SNAPSHOT. That recorded snapshot is separately
    resolved to its fixed identity-rewritten commit and verified project subtree.
    No missing-commit fallback or arbitrary path/revision substitution is allowed.
    """
    require(requested_revision == EVOLUTION_PARENT and name == "predict.py",
            "Only the fixed legacy documentation parent and predict.py may be resolved")
    return EVOLUTION_RETAINED_SOURCE_REVISION


def verify_source_evolution(archive, latest):
    """Validate an optional fixed record without weakening other frozen files."""
    if not (archive.project / SOURCE_EVOLUTION).is_file():
        return None
    data = archive.current(SOURCE_EVOLUTION)
    record = validate_source_evolution(strict_json(data))
    _, repo_path = archive.safe_relative(SOURCE_EVOLUTION)
    index = archive.command("show", ":" + repo_path, "--")
    require(lf(data) == lf(index), "Source-evolution record differs from the Git index")
    require("predict.py" in latest and latest["predict.py"][0] == EVOLUTION_OLD_PREDICT_SHA256,
            "Source evolution does not match the latest historically frozen predictor")
    retained_revision = resolve_evolution_source_revision(EVOLUTION_PARENT, "predict.py")
    source_resolution = archive.resolve_revision(retained_revision)
    parent_source = archive.blob(retained_revision, "predict.py")
    require(historical_match(parent_source, EVOLUTION_OLD_PREDICT_SHA256) == "exact_original_bytes",
            "Pre-display predictor binding differs")
    git_blob_sha1 = hashlib.sha1(b"blob " + str(len(parent_source)).encode("ascii")
                                + b"\0" + parent_source).hexdigest()
    require(len(parent_source) == EVOLUTION_OLD_PREDICT_BYTES
            and git_blob_sha1 == EVOLUTION_OLD_PREDICT_GIT_BLOB_SHA1,
            "Resolved predictor must be the identical fixed historical Git blob")
    current = current_index_binding(archive, "predict.py", record["current_source"]["sha256"])
    helper = current_index_binding(archive, "prediction_panels.py", record["new_helpers"][0]["sha256"])
    return {"record_file": SOURCE_EVOLUTION, "record_sha256": digest(data),
            "record_lf_sha256": digest(lf(data)), "parent_revision": EVOLUTION_PARENT,
            "historical_predictor_sha256": EVOLUTION_OLD_PREDICT_SHA256,
            "historical_predictor_resolution": {
                "file": "predict.py", "requested_legacy_revision": EVOLUTION_PARENT,
                "recorded_retained_revision": retained_revision,
                "resolved_retained_revision": source_resolution["resolved_revision"],
                "verified_project_subtree": source_resolution["verified_project_tree"],
                "resolved_source_sha256": digest(parent_source),
                "resolved_git_blob_sha1": git_blob_sha1,
                "expected_git_blob_sha1": EVOLUTION_OLD_PREDICT_GIT_BLOB_SHA1,
                "source_bytes": len(parent_source),
                "resolution_scope": "Only this documentation parent/predict.py pair followed by the fixed snapshot identity mapping; identical pinned project subtree, source bytes and Git blob, not a general revision fallback.",
                "original_record_changed": False, "legacy_commit_lookup_required": False},
            "current_source": current, "new_helpers": [helper],
            "authorization_scope": record["authorization_scope"],
            "verification_boundary": record["verification_boundary"],
            "old_frozen_source_bindings_preserved": True,
            "other_latest_frozen_files_require_unchanged_content": True}


def verify(archive, *, bindings_only=False):
    constructed_checks = matcher_checks()
    verification_blob = archive.blob(REPORT_SNAPSHOT, VERIFICATION)
    verification = strict_json(verification_blob)
    require(lf(archive.current(VERIFICATION)) == lf(verification_blob), "Historical verification was modified")
    summary = archive.bound(SUMMARY, verification["summary_sha256"], role="historical_report_summary")
    require(summary["collector_sha256"] == verification["collector_sha256"],
            "Historical summary/verification collector binding differs")
    require(len(summary["input_sources"]) == 19, "Expected the historical 19-source report")
    documents = {}
    for key, item in summary["input_sources"].items():
        require(PurePosixPath(item["file"]).name == item["file"], "Unexpected evidence basename")
        documents[key] = archive.bound("evidence/" + item["file"], item["sha256"])
    archive.bound(verification["collector_file"], verification["collector_sha256"], role="report_collector")
    archive.bound(verification["report_file"], verification["report_sha256"],
                  role="historical_report_snapshot", current_must_match=False)

    # The old CE plan is not one of the 19 sources; its hash is bound by old_ce.
    formal = archive.bound("evidence/formal_run_plan_20261026.json",
                           documents["old_ce"]["formal_plan_sha256"], role="old_ce_frozen_plan")
    plans = [("old_ce", formal), ("id5", documents["id5_plan"]),
             ("test_recipe", documents["recipe"]), ("original_test", documents["test_plan"]),
             ("authorized_recovery", documents["recovery_plan"])]
    frozen = []
    latest = {}
    for label, plan in plans:
        source = plan["source_sha256"]
        require(isinstance(source, dict) and source, "Missing frozen source mapping")
        for name, expected in source.items():
            require(PurePosixPath(name).name == name and name.endswith((".py", ".json")),
                    "Frozen source must be a direct project text file")
            archive.bound(name, expected, revision=plan["source_revision"],
                          role="frozen_source:" + label, current_must_match=False)
            frozen.append({"plan": label, "file": name, "revision": plan["source_revision"],
                           "original_sha256": expected})
            latest[name] = (expected, plan["source_revision"])
    evolution = verify_source_evolution(archive, latest)
    for name, (expected, revision) in latest.items():
        evolved_predictor = evolution is not None and name == "predict.py"
        archive.bound(name, expected, revision=revision,
                      role=("latest_frozen_predictor_historical_check" if evolved_predictor
                            else "latest_frozen_source_current_check"),
                      current_must_match=not evolved_predictor)

    syntax = []
    for tracked in sorted(archive.tracked):
        if tracked.startswith(archive.prefix + "/") and tracked.endswith(".py"):
            name = tracked[len(archive.prefix) + 1:]
            ast.parse(archive.current(name).decode("utf-8"), filename=name)
            syntax.append(name)
    # This new verifier may not be staged yet; inspect its own source, never execute a project import.
    own_source = Path(__file__).read_bytes()
    own_ast = ast.parse(own_source.decode("utf-8"))
    imported = set()
    for node in ast.walk(own_ast):
        if isinstance(node, ast.Import):
            imported.update(item.name.split(".")[0] for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    require(imported <= set(sys.stdlib_module_names) | {"__future__"}, "Verifier imports non-stdlib code")
    forbidden = [name for name in archive.tracked if name.lower().endswith(MODEL_SUFFIXES)
                 or any(part.lower() in {"data", "checkpoints", ".venv"}
                        for part in PurePosixPath(name).parts)]
    require(not forbidden, "Tracked data/model files found")
    links = [] if bindings_only else [archive.links(name) for name in ("README.md", "FINAL_REPORT.md")]
    current_documents = []
    if not bindings_only:
        for name in ("README.md", "FINAL_REPORT.md"):
            current = archive.current(name)
            head = archive.blob(archive.revision, name)
            current_documents.append({"file": name, "current_sha256": digest(current),
                                      "current_lf_sha256": digest(lf(current)),
                                      "head_git_blob_sha256": digest(head),
                                      "head_git_blob_lf_sha256": digest(lf(head)),
                                      "current_same_text_as_head": lf(current) == lf(head)})
    return {"audit_kind": "submission_archive_text_only_verification",
            "status": "passed_frozen_text_bindings_only" if bindings_only else "passed",
            "created_at": datetime.now(timezone.utc).isoformat(), "performed_by": "Codex stdlib archive audit",
            "current_revision": archive.revision, "historical_report_snapshot_revision": REPORT_SNAPSHOT,
            "resolved_historical_report_snapshot_revision": archive.resolve_revision(REPORT_SNAPSHOT)["resolved_revision"],
            "historical_source_revision_resolutions": [archive.revision_resolutions[key]
                                                       for key in sorted(archive.revision_resolutions)],
            "verifier_sha256": digest(own_source),
            "hash_policy": {"historical_original_bytes": "Exact Git blob bytes or LF-to-CRLF candidate only; no JSON reserialization or numeric tolerance.",
                            "checkout_comparison": "Only CRLF-pair-to-LF normalization is accepted; lone CR is rejected. Frozen files match their historical content, except an explicitly recorded predictor whose current source/helper also match their declared SHA256 and Git index.",
                            "current_report": "Current report may contain marked revisions; old report hash is checked through its explicit identity-rewritten snapshot with the identical pinned project subtree.",
                            "historical_revision_resolution": "Six fixed original-to-rewritten revision mappings only, each pinned to its original project subtree and original per-file hashes; current HEAD is separately readable. Unknown historical revisions are rejected and original records retain their IDs.",
                            "normalized_lf_sha256_is_a_new_audit_value": True,
                            "old_record_or_hash_rewritten": False},
            "verification_record": {"file": VERIFICATION, "snapshot_sha256": digest(verification_blob),
                                    "snapshot_lf_sha256": digest(lf(verification_blob)),
                                    "current_sha256": digest(archive.current(VERIFICATION)),
                                    "current_lf_sha256": digest(lf(archive.current(VERIFICATION)))},
            "summary_source_count": len(documents), "frozen_source_binding_count": len(frozen),
            "latest_frozen_source_file_count": len(latest), "frozen_plans": frozen,
            "post_experiment_source_evolution": evolution,
            "latest_frozen_files_unchanged_in_current_checkout": len(latest) - int(evolution is not None),
            "historical_binding_counts": dict(Counter(row["historical_binding"] for row in archive.records)),
            "file_bindings": archive.records, "local_document_links": links,
            "current_document_snapshots": current_documents,
            "documentation_links_deferred": bindings_only,
            "static_python_syntax": {"passed": True, "tracked_files": syntax, "project_modules_imported": 0},
            "verifier_imports": sorted(imported), "tracked_data_or_models": forbidden,
            "matcher_constructed_checks": constructed_checks,
            "lf_checkout_simulation": {"enabled": bool(archive.temp),
                                       "temporary_text_files_verified": len(archive.simulated_files),
                                       "original_files_changed": False},
            "limits": {"reads": "Allowlisted tracked project text and read-only Git blobs only; link assets are checked by path metadata, not opened.",
                       "not_followed": "No recursive embedded path/hash traversal, absolute evidence paths, external URLs, outputs, MRI/mask, checkpoints or weight bytes.",
                       "not_executed": "No project imports, Torch imports, GPU use, training, model forward, tests with models, profile or Git mutation.",
                       "source_scope": "Five saved frozen plans: original CE, ID5, recipe, initial test and authorized recovery. Original plan revision IDs resolve through six pinned, project-subtree-identical identity mappings, then retain all per-file hash checks. Only predict.py may have the separately declared, index/hash-bound post-experiment display change; other latest files remain strict.",
                       "not_claimed": "No new execution or statistical validation of the scientific results; no tutor check-off, originality approval, PR or submission."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--git", default=shutil.which("git") or "git", help="Git executable; read-only calls only")
    parser.add_argument("--simulate-lf-checkout", action="store_true", help="Additionally verify LF copies in a temporary text-only directory")
    parser.add_argument("--bindings-only", action="store_true", help="Inspect frozen text before new documentation is staged; cannot save the final audit record")
    parser.add_argument("--json", action="store_true", help="Print the full machine-readable audit instead of the concise result")
    parser.add_argument("--output", type=Path, help="Optional new evidence/*.json; existing files are never overwritten")
    args = parser.parse_args()
    require(not (args.bindings_only and args.output), "Bindings-only inspection cannot save a complete archive audit")
    archive = Archive(args.project_dir, args.git)
    try:
        result = verify(archive, bindings_only=args.bindings_only)
        if args.simulate_lf_checkout:
            simulation = Archive(args.project_dir, args.git, simulate_lf=True)
            try:
                simulated = verify(simulation, bindings_only=args.bindings_only)
                historical_fields = ("file", "role", "expected_original_sha256", "historical_revision",
                                     "resolved_historical_revision", "verified_historical_project_tree",
                                     "historical_git_blob_sha256", "historical_binding")
                signature = lambda report: [[row[field] for field in historical_fields]
                                            for row in report["file_bindings"]]
                require(signature(result) == signature(simulated), "LF simulation changed historical bindings")
                require(result["local_document_links"] == simulated["local_document_links"],
                        "Documentation links changed during LF simulation")
                result["lf_checkout_simulation"] = {
                    "enabled": True, "status": simulated["status"],
                    "temporary_text_files_verified": len(simulation.simulated_files),
                    "same_original_historical_bindings": True,
                    "historical_binding_counts": simulated["historical_binding_counts"],
                    "original_files_changed": False,
                    "recorded_current_sha256_scope": "Actual checkout bytes; LF copies are an additional check, not substituted evidence."}
            finally:
                simulation.close()
        if args.output:
            output = args.output if args.output.is_absolute() else archive.project / args.output
            output = output.resolve()
            require(output.parent == archive.project / "evidence" and output.suffix == ".json",
                    "New audit output must be a direct project evidence/*.json file")
            with output.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        if args.json:
            print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
        else:
            concise = {key: result[key] for key in ("status", "summary_source_count",
                       "frozen_source_binding_count", "latest_frozen_source_file_count",
                       "historical_binding_counts", "lf_checkout_simulation")}
            concise["post_experiment_predictor_change_bound"] = result["post_experiment_source_evolution"] is not None
            concise["latest_frozen_files_unchanged_in_current_checkout"] = result["latest_frozen_files_unchanged_in_current_checkout"]
            concise["static_python_syntax_file_count"] = len(result["static_python_syntax"]["tracked_files"])
            concise["checked_local_link_count"] = sum(len(item["checked_local_links"])
                                                      for item in result["local_document_links"])
            if args.output:
                concise["output"] = output.name
            print(json.dumps(concise, ensure_ascii=True, indent=2, allow_nan=False))
    finally:
        archive.close()


if __name__ == "__main__":
    main()
