"""User-authorized, post-test exploratory review of three frozen M015 models.

Only eight clean M015 caches are opened. No training, augmentation, selection,
TTA, postprocessing, saved arrays or replacement six-patient TEST benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from experiments import evaluate_appearance as original

EXPERIMENT = "m015_existing_checkpoint_followup_v1"
MODEL_IDS = ("seed42_reference", "gamma_only", "gamma_polarity")
CASE_KEYS = tuple(f"M015/Week{i}" for i in range(8))
require, sha, read = original.require, original.sha256, original.read_json


def validate_protocol(protocol, appearance):
    require(protocol["experiment_id"] == EXPERIMENT, "Wrong follow-up experiment identity.")
    require(protocol.get("test_informed_exploratory_acknowledged") is True,
            "Explicit post-test exploratory acknowledgement is required.")
    require(tuple(protocol["case_keys"]) == CASE_KEYS, "Exactly M015 Week0..7 in fixed order are required.")
    require(set(protocol["models"]) == set(MODEL_IDS), "Exactly the three declared existing checkpoints are allowed.")
    require(protocol["manifest_sha256"] == original.MANIFEST_SHA
            and protocol["cache_identity_sha256"] == original.IDENTITY_SHA
            and protocol["training_sources_sha256"] == original.SOURCES_SHA, "Frozen data/source identity differs.")
    require(protocol["inference"] == {"split": "test", "device": "cpu", "amp": False, "saved_arrays": False}
            and type(protocol["inference"]["amp"]) is bool and type(protocol["inference"]["saved_arrays"]) is bool,
            "Only clean TEST CPU FP32 inference without saved arrays is allowed.")
    for name in MODEL_IDS:
        spec = protocol["models"][name]
        require(spec["checkpoint"] == appearance["models"][name]["checkpoint"], "Checkpoint path differs from A1.")
        digest = spec["checkpoint_sha256"]
        require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "Invalid checkpoint SHA.")
    require(protocol["models"]["seed42_reference"]["checkpoint_sha256"] == original.REFERENCES["seed42_reference"][2],
            "Original v1 checkpoint must remain frozen.")


def select_records(manifest):
    require(not manifest.get("partial") and not manifest.get("development_only"), "Full frozen manifest required.")
    records = manifest["records"]
    keys = [f"{r['patient']}/{r['visit']}" for r in records]
    require(len(keys) == len(set(keys)), "Duplicate manifest case keys.")
    selected = [r for r in records if r["patient"] == "M015"]
    require(len(selected) == 8 and {f"{r['patient']}/{r['visit']}" for r in selected} == set(CASE_KEYS)
            and all(r["split"] == "test" for r in selected), "Only the complete eight M015 TEST visits may be opened.")
    return sorted(selected, key=lambda r: int(r["visit"].removeprefix("Week")))


def decode_case(manifest_path, manifest, record, identity):
    """Verify decoded identities independently before returning clean arrays."""
    from tools.cache_identity import read_arrays, safe_relative, validate_arrays
    import hashlib
    path = safe_relative(manifest_path.parent, record["cache_path"])
    arrays = read_arrays(path)
    validate_arrays(arrays, record, manifest)
    expected = identity["records"][record["cache_path"]]
    require(all(expected[k] == record[k] for k in ("patient", "visit", "split")), "Cache case identity differs.")
    for key, value in arrays.items():
        header = {"dtype": value.dtype.str, "shape": list(value.shape)}
        digest = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(",", ":")).encode() + b"\0"
                                + value.tobytes(order="C")).hexdigest()
        require(expected["arrays"][key] == {**header, "sha256": digest}, "Decoded M015 cache identity differs.")
    return arrays, path


def add_error_counts(evaluation):
    for case in evaluation["cases"]:
        for row in case["per_class"]:
            row["false_positive_voxels"] = row["predicted_voxels"] - row["intersection"]
            row["false_negative_voxels"] = row["target_voxels"] - row["intersection"]
            require(min(row["false_positive_voxels"], row["false_negative_voxels"]) >= 0, "Invalid overlap counts.")


class VerifiedM015:
    split, augment = "test", False

    def __init__(self, manifest_path, manifest, identity):
        import torch
        self.records, self.items, self.receipts = select_records(manifest), [], []
        for record in self.records:
            arrays, path = decode_case(manifest_path, manifest, record, identity)
            self.items.append({"image": torch.from_numpy(arrays["image"]).unsqueeze(0),
                               "mask": torch.from_numpy(arrays["mask"]).long(), "spacing": torch.from_numpy(arrays["spacing"]),
                               "patient": record["patient"], "visit": record["visit"]})
            self.receipts.append({"case_key": f"{record['patient']}/{record['visit']}", "cache_path": record["cache_path"],
                                  "decoded_identity_verified": True, "cache_file_sha256": sha(path)})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "manifest", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().absolute()
    require(not os.path.lexists(output), "Output directory must be new; refusing overwrite.")
    protocol_path, manifest_path = args.protocol.expanduser().resolve(), args.manifest.expanduser().resolve()
    protocol, protocol_sha = read(protocol_path), sha(protocol_path)
    appearance_path = original.child(protocol["appearance_protocol"])
    appearance, appearance_sha = read(appearance_path), sha(appearance_path)
    require(appearance_sha == protocol["appearance_protocol_sha256"], "Original A1 protocol SHA differs.")
    original.validate_protocol(appearance, sha(manifest_path))
    validate_protocol(protocol, appearance)
    required = protocol["required_file_sha256"]
    require({"tools/evaluate_m015_followup.py", "experiments/evaluate_appearance.py", "tools/cache_identity.py"} <= set(required),
            "Follow-up source guards omit evaluator dependencies.")
    def verify_sources():
        require(sha(protocol_path) == protocol_sha and sha(appearance_path) == appearance_sha, "A protocol changed during execution.")
        for book in (appearance["required_file_sha256"], required):
            require(all(sha(original.child(p)) == digest for p, digest in book.items()), "A protected source/input SHA changed.")
    verify_sources()
    from dataset import load_manifest
    from tools.cache_identity import load_verified_book, safe_relative
    manifest = load_manifest(manifest_path)
    require(sha(manifest_path) == protocol["manifest_sha256"], "Manifest SHA differs.")
    select_records(manifest)  # Scope checked before any pixel is read.
    identity_path = manifest_path.parent / "cache_identity.json"
    require(sha(identity_path) == protocol["cache_identity_sha256"], "Identity book SHA differs.")
    identity = load_verified_book(identity_path, manifest_path, manifest)
    checkpoint_paths = {n: original.child(protocol["models"][n]["checkpoint"]) for n in MODEL_IDS}
    require(not output.resolve().is_relative_to(manifest_path.parent)
            and all(not output.resolve().is_relative_to(p.parent) for p in checkpoint_paths.values()), "Output enters a protected cache/checkpoint directory.")
    import torch
    from modules import ImprovedUNet3D
    from utils import evaluate_volumes, write_json
    threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    require(threads == 4, "This CPU follow-up requires OMP_NUM_THREADS=4.")
    torch.set_num_threads(threads)
    device = torch.device("cpu")
    torch.use_deterministic_algorithms(True, warn_only=True)
    dataset = VerifiedM015(manifest_path, manifest, identity)
    output.mkdir(parents=True, exist_ok=False)
    index = {"schema_version": 1, "experiment_id": EXPERIMENT, "status": "running", "split": "test_M015_only",
             "device": protocol["inference"]["device"], "amp": protocol["inference"]["amp"],
             "exploratory_after_test": True, "replaces_v1_test": False, "saved_arrays": False, "training_performed": False,
             "n_cases": 8, "n_patients": 1, "case_keys": list(CASE_KEYS), "condition": "clean",
             "protocol_sha256": protocol_sha, "appearance_protocol_sha256": appearance_sha,
             "manifest_sha256": sha(manifest_path), "cache_identity_sha256": sha(identity_path),
             "source_file_sha256": required, "cache_verification": dataset.receipts, "models": {}, "results": {},
             "runtime": {"device": protocol["inference"]["device"], "gpu_name": None, "python": platform.python_version(),
                         "torch": str(torch.__version__), "cuda": torch.version.cuda, "amp": protocol["inference"]["amp"],
                         "torch_num_threads": torch.get_num_threads(), "omp_num_threads": threads,
                         "mkldnn_enabled": torch.backends.mkldnn.enabled},
             "limits": ["User-authorized exploratory recheck of one previously examined patient; eight visits are not eight independent patients.",
                        "The separate follow-up protocol narrowly authorizes this recheck; the unchanged A1 protocol did not authorize or preregister it.",
                        "All three models use CPU FP32 here; formal v1 used CUDA AMP. Device/precision can change predictions, so exact voxel equality with that benchmark is not required or assumed.",
                        "Existing clean-VAL-selected checkpoints only; no new training, selection, augmentation, TTA or postprocessing.",
                        "No new six-patient TEST benchmark, independent confirmation, bootstrap or population superiority claim.",
                        "Same downsampled cache grid; no clinical/native-resolution accuracy claim. Original A1 protocol and v1 TEST remain unchanged."]}
    for name in MODEL_IDS:
        path, spec = checkpoint_paths[name], protocol["models"][name]
        require(sha(path) == spec["checkpoint_sha256"], "Declared existing checkpoint SHA differs.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = original.validate_payload(checkpoint, appearance["models"][name], name, manifest, appearance_sha)
        require(config == appearance["training_conditions"], "Original model/training configuration differs.")
        receipt = (original.validate_arm_files(checkpoint, config, name, path, manifest, appearance_sha,
                                              appearance["required_file_sha256"]) if name != "seed42_reference" else {})
        if name != "seed42_reference":
            require(read(path.parent / "provenance.json")["history_sha256"] == receipt["history_sha256"], "Arm provenance/history SHA differs.")
        model = ImprovedUNet3D(6, 16).eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        before = original.state_sha(model)
        model.register_forward_hook(original.finite_main_logits)
        model.to(device)
        evaluation = evaluate_volumes(model, dataset, "improved3d", 6, device, amp=protocol["inference"]["amp"])
        require(len(evaluation["cases"]) == 8 and {f"{c['patient']}/{c['visit']}" for c in evaluation["cases"]} == set(CASE_KEYS)
                and evaluation["aggregate"]["num_patients"] == 1, "Incomplete or out-of-scope follow-up results.")
        add_error_counts(evaluation)
        evaluation.update(schema_version=1, experiment_id=EXPERIMENT, model_id=name, condition="clean", split="test_M015_only",
                          exploratory_after_test=True, replaces_v1_test=False, saved_arrays=False,
                          device=protocol["inference"]["device"], amp=protocol["inference"]["amp"],
                          protocol_sha256=protocol_sha, manifest_sha256=sha(manifest_path), checkpoint_sha256=spec["checkpoint_sha256"])
        after = original.state_sha(model)
        require(before == after and sha(path) == spec["checkpoint_sha256"], "Inference changed checkpoint/model state.")
        relative = f"{name}/evaluation.json"
        write_json(output / relative, evaluation)
        index["models"][name] = {"checkpoint_sha256": spec["checkpoint_sha256"], "checkpoint_epoch": checkpoint["epoch"],
                                 "state_before_sha256": before, "state_after_sha256": after, "state_unchanged": True, **receipt}
        index["results"][name] = {"evaluation_file": relative, "evaluation_sha256": sha(output / relative), "aggregate": evaluation["aggregate"]}
        model.to("cpu")
        del model, checkpoint
    verify_sources()
    require(sha(manifest_path) == protocol["manifest_sha256"] and sha(identity_path) == protocol["cache_identity_sha256"], "Frozen data metadata changed.")
    require(all(sha(checkpoint_paths[n]) == protocol["models"][n]["checkpoint_sha256"] for n in MODEL_IDS), "A checkpoint file changed.")
    require(all(sha(safe_relative(manifest_path.parent, r["cache_path"])) == r["cache_file_sha256"] for r in dataset.receipts), "A selected cache file changed.")
    index.update(status="completed", protected_files_unchanged=True, created_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "m015_followup.json", index)
    print("Completed three existing models x eight clean M015 visits; exploratory only, no arrays saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
