"""Evaluate all four preregistered 3D models on all five VAL conditions.

Only the 38 frozen validation volumes are opened. No TEST access, arrays,
postprocessing, model selection, training, or best-stress selection is provided.
The same six patients are reused across conditions, not thirty independent ones.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

PROJECT = Path(__file__).resolve().parents[1]
MANIFEST_SHA = "c47fe880d353629348786b852e3f699291af1a4d5628b2980bf42a74635d0424"
SOURCES_SHA = "a3cf4ed4eb21c9d79df9a50b53308df797a30cb112655f4ee85d809ad7916e54"
IDENTITY_SHA = "eacb84bf20626a6ec9d7f6d36489a81e2d4e4af9fbe4ade1ce8f508a2022e08a"
EXPERIMENT = "appearance_augmentation_v2"
CONDITIONS = ("clean", "gamma_0.7", "gamma_1.5", "polarity", "spatial_gain")
MODEL_IDS = ("seed42_reference", "seed43_reference", "gamma_only", "gamma_polarity")
REFERENCES = {
    "seed42_reference": (42, 29, "9ea46e13a95617ec4ad80c21ba459087fb62941fb04406c7ffeade22ad014b62"),
    "seed43_reference": (43, 27, "3b9cb27f01004720ebcdda377369c0fe2a488fcfe04d778cf24ff7cac3a4539e"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def child(relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "Evidence paths must be safe project-relative paths.")
    path = (PROJECT / relative).resolve()
    require(path.is_relative_to(PROJECT), "Evidence path escapes project.")
    return path


def validate_protocol(protocol, manifest_sha):
    require(protocol["experiment_id"] == EXPERIMENT, "Wrong experiment identity.")
    require(protocol["manifest_sha256"] == manifest_sha == MANIFEST_SHA, "Frozen manifest identity differs.")
    require(protocol["training_sources_sha256"] == SOURCES_SHA, "Original training sourcebook differs.")
    require(protocol["allowed_splits"] == ["train", "validation"], "Protocol may allow only TRAIN/VAL, never TEST.")
    require(protocol["training_epochs_per_model"] == 30 and protocol["training_seed"] == 42, "Unexpected arm training budget/seed.")
    require(tuple(protocol["conditions"]) == CONDITIONS and set(protocol["models"]) == set(MODEL_IDS), "All four models and all five fixed conditions are required.")
    for name, (seed, epoch, digest) in REFERENCES.items():
        spec = protocol["models"][name]
        require(spec["kind"] == "frozen_v1" and spec["seed"] == seed and spec["selected_epoch"] == epoch and spec["checkpoint_sha256"] == digest, "Frozen reference identity changed.")
    for name in MODEL_IDS[2:]:
        spec = protocol["models"][name]
        require(spec["kind"] == "appearance_arm" and spec["experiment_tag"] == name and spec["seed"] == 42, "Appearance arm identity differs.")
    require(protocol["inference"]["split"] == "validation" and type(protocol["inference"]["amp"]) is bool, "Inference must be validation-only with explicit precision.")


def validate_payload(checkpoint, spec, name, manifest, protocol_sha):
    config = checkpoint["config"]
    require(config["model"] == "improved3d" and config["base_channels"] == 16 and config["num_classes"] == 6,
            "Expected the unchanged base16 six-class 3D model.")
    require(config["manifest_sha256"] == MANIFEST_SHA and config["label_ids"] == manifest["label_ids"]
            and config["target_shape_dhw"] == manifest["target_shape_dhw"] and config["patient_splits"] == manifest["patient_splits"], "Checkpoint data/label/grid identity differs.")
    require(config["seed"] == spec["seed"] and not config.get("development_only") and config.get("max_batches") is None, "Wrong seed or development checkpoint.")
    if name in REFERENCES:
        require(checkpoint["epoch"] == REFERENCES[name][1], "Reference epoch differs.")
    else:
        expected = {"experiment_id": EXPERIMENT, "arm_tag": name, "protocol_sha256": protocol_sha}
        require(checkpoint["experiment"] == expected and checkpoint["epoch"] == checkpoint["best_epoch"], "New checkpoint experiment/tag/best epoch differs.")
    return config


def validate_arm_files(checkpoint, config, name, checkpoint_path, manifest, protocol_sha, required):
    root = checkpoint_path.parent
    run_path, history_path, provenance_path = (root / filename for filename in ("run.json", "history.json", "provenance.json"))
    run, history, provenance = (read_json(p) for p in (run_path, history_path, provenance_path))
    expected = {"experiment_id": EXPERIMENT, "arm_tag": name, "protocol_sha256": protocol_sha}
    require(run["experiment"] == provenance["experiment"] == expected and run["config"] == config, "Arm run/provenance experiment or configuration differs.")
    require(run["status"] == "completed" and run["completed_epochs"] == 30 and len(history) == 30
            and [r["epoch"] for r in history] == list(range(1, 31)), "Arm training is incomplete or not thirty full epochs.")
    require(all(not r.get("development_only") for r in history), "Development history is forbidden.")
    require(all(r["training_examples"] == 143 and r["training_batches"] == 143 for r in history), "Arm history must cover every TRAIN volume each epoch.")
    best = max(history, key=lambda row: row["validation_foreground_dice"])
    require(best["epoch"] == run["best_epoch"] == checkpoint["epoch"] and checkpoint["best_score"] == best["validation_foreground_dice"], "Selected checkpoint/history disagrees.")
    require(provenance["checkpoint_sha256"] == sha256(checkpoint_path) and provenance["run_sha256"] == sha256(run_path)
            and provenance["protocol_sha256"] == protocol_sha and provenance["manifest_sha256"] == MANIFEST_SHA, "Arm receipt file identities differ.")
    require(provenance["source_file_sha256"] == required, "Arm receipt source plan differs from protocol.")
    verification_path = child(provenance["cache_verification_path"])
    require(sha256(verification_path) == provenance["cache_verification_sha256"], "Training/validation cache verification digest differs.")
    verification = read_json(verification_path)
    require(verification["status"] == "verified" and verification["manifest_sha256"] == MANIFEST_SHA
            and verification["counts"] == {"train": 143, "validation": 38}, "Expected the 181-case TRAIN/VAL-only verification.")
    rows = verification["records"]
    expected_keys = {(r["patient"], r["visit"], r["split"]) for r in manifest["records"] if r["split"] in ("train", "validation")}
    require(len(rows) == 181 and {(r["patient"], r["visit"], r["split"]) for r in rows} == expected_keys
            and all(r["decoded_identity_verified"] is True for r in rows), "Verification omits/duplicates cases or includes TEST.")
    return {"run_sha256": sha256(run_path), "history_sha256": sha256(history_path), "provenance_sha256": sha256(provenance_path),
            "cache_verification_sha256": sha256(verification_path)}


class VerifiedValidation:
    """Preload only the 38 decoded-identity-verified VAL cases, about 456MiB."""
    split, augment = "validation", False

    def __init__(self, manifest_path, manifest, identity):
        import numpy as np
        import torch
        self.records = [r for r in manifest["records"] if r["split"] == "validation"]
        require(len(self.records) == 38 and len({r["patient"] for r in self.records}) == 6, "Expected full 38-volume/six-patient validation.")
        self.items = []
        for record in self.records:
            relative = Path(record["cache_path"])
            require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe cache path.")
            path = (manifest_path.parent / relative).resolve()
            require(path.is_relative_to(manifest_path.parent), "Cache escaped manifest directory.")
            with np.load(path, allow_pickle=False) as stored:
                require(set(stored.files) == {"image", "mask", "spacing"}, "Unexpected cache arrays.")
                arrays = {k: stored[k] for k in stored.files}
            expected = identity["records"][record["cache_path"]]
            require(all(expected[k] == record[k] for k in ("patient", "visit", "split")), "Cache case identity differs.")
            for key, array in arrays.items():
                header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)}, sort_keys=True, separators=(",", ":")).encode()
                digest = hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()
                require(expected["arrays"][key] == {"dtype": array.dtype.str, "shape": list(array.shape), "sha256": digest}, "Decoded VAL array identity failed.")
            require(arrays["image"].dtype == np.float32 and arrays["mask"].dtype == np.uint8
                    and arrays["image"].shape == arrays["mask"].shape == tuple(manifest["target_shape_dhw"]), "Invalid VAL cache grid/dtype.")
            self.items.append({"image": torch.from_numpy(arrays["image"]).unsqueeze(0), "mask": torch.from_numpy(arrays["mask"]).long(),
                               "spacing": torch.from_numpy(arrays["spacing"]), "patient": record["patient"], "visit": record["visit"]})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def state_sha(model):
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        result.update(name.encode() + b"\0" + tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def finite_main_logits(_module, _inputs, output):
    """Reject AMP/stress numerical failures before hard argmax can hide them."""
    import torch
    if not bool(torch.isfinite(output["logits"]).all().item()):
        raise RuntimeError("Nonfinite main-head logits during validation stress; evaluation aborted before argmax.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("protocol", "manifest", "output-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().absolute()
    require(not os.path.lexists(output), "Evaluation output must be new; no artifact may be overwritten.")
    protocol_path, manifest_path = args.protocol.expanduser().resolve(), args.manifest.expanduser().resolve()
    protocol, protocol_sha = read_json(protocol_path), sha256(protocol_path)
    validate_protocol(protocol, sha256(manifest_path))
    sys.path.insert(0, str(PROJECT))
    import torch
    from dataset import load_manifest
    from modules import ImprovedUNet3D
    from utils import evaluate_volumes, write_json
    from experiments.appearance import STRESS_CONDITIONS, StressValidationDataset
    require(tuple(STRESS_CONDITIONS) == CONDITIONS, "Appearance stress implementation differs.")
    require(sha256(PROJECT / "training_sources.json") == SOURCES_SHA, "Original sourcebook bytes changed.")
    required = protocol["required_file_sha256"]
    require({"modules.py", "dataset.py", "utils.py", "experiments/appearance.py", "experiments/evaluate_appearance.py"} <= set(required), "Protocol source plan omits evaluation dependencies.")
    require(all(sha256(child(p)) == digest for p, digest in required.items()), "A preregistered source file changed.")
    manifest = load_manifest(manifest_path)
    require(not manifest["partial"] and not manifest["development_only"], "Full frozen manifest required.")
    identity_path = manifest_path.parent / "cache_identity.json"
    require(sha256(identity_path) == IDENTITY_SHA, "Frozen cache identity book differs.")
    identity = read_json(identity_path)
    device = torch.device(args.device)
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    amp = protocol["inference"]["amp"] and device.type == "cuda"
    models, configs, evidence, checkpoint_paths = {}, {}, {}, {}
    for name in MODEL_IDS:
        spec = protocol["models"][name]
        path = child(spec["checkpoint"])
        checkpoint_paths[name] = path
        require(not output.resolve().is_relative_to(path.parent) and not output.resolve().is_relative_to(manifest_path.parent), "Output must not be within checkpoint/cache directories.")
        digest = sha256(path)
        if name in REFERENCES:
            require(digest == REFERENCES[name][2], "Original checkpoint SHA differs.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = validate_payload(checkpoint, spec, name, manifest, protocol_sha)
        receipt = validate_arm_files(checkpoint, config, name, path, manifest, protocol_sha, required) if name not in REFERENCES else {}
        torch.manual_seed(config["seed"])
        model = ImprovedUNet3D(6, 16).eval()
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.register_forward_hook(finite_main_logits)
        models[name] = model
        configs[name] = config
        evidence[name] = {"checkpoint_sha256": digest, "checkpoint_epoch": checkpoint["epoch"], "seed": config["seed"],
                          "state_before_sha256": state_sha(model), "parameters": sum(p.numel() for p in model.parameters()), **receipt}
        del checkpoint
    require(all(configs[name] == configs["seed42_reference"] for name in MODEL_IDS[2:]), "New arms must retain the complete original seed42 training configuration.")
    reference43 = dict(configs["seed43_reference"], seed=42)
    require(reference43 == configs["seed42_reference"], "Reference training settings differ beyond seed.")
    dataset = VerifiedValidation(manifest_path, manifest, identity)
    output.mkdir(parents=True, exist_ok=False)
    index = {"schema_version": 1, "experiment_id": EXPERIMENT, "split": "validation", "protocol_sha256": protocol_sha,
             "manifest_sha256": MANIFEST_SHA, "cache_identity_sha256": IDENTITY_SHA, "models": evidence, "conditions": list(CONDITIONS),
             "n_scans_per_condition": 38, "n_unique_patients": 6, "saved_arrays": False, "results": {},
             "runtime": {"device": str(device), "amp": amp, "python": platform.python_version(), "torch": str(torch.__version__),
                         "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
             "limits": ["All four models and all five predeclared synthetic cache-space conditions; no best-stress selection.",
                        "Repeated conditions reuse the same six validation patients, not thirty independent patients. TRAIN/VAL-only follow-up after v1 TEST was seen.",
                        "No TEST pixels, new TEST results, inference postprocessing or saved arrays. Synthetic polarity is not a clinical acquisition diagnosis."]}
    expected_keys = {(r["patient"], r["visit"]) for r in dataset.records}
    for name, model in models.items():
        model.to(device)
        index["results"][name] = {}
        for condition in CONDITIONS:
            evaluation = evaluate_volumes(model, StressValidationDataset(dataset, condition), "improved3d", 6, device, amp=amp)
            require({(c["patient"], c["visit"]) for c in evaluation["cases"]} == expected_keys
                    and len(evaluation["cases"]) == 38 and evaluation["aggregate"]["num_patients"] == 6, "Incomplete validation evaluation.")
            evaluation.update(schema_version=1, model_id=name, condition=condition, split="validation", saved_arrays=False,
                              protocol_sha256=protocol_sha, manifest_sha256=MANIFEST_SHA, checkpoint_sha256=evidence[name]["checkpoint_sha256"], amp=amp)
            relative = f"{name}/{condition}/evaluation.json"
            write_json(output / relative, evaluation)
            index["results"][name][condition] = {"evaluation_file": relative, "evaluation_sha256": sha256(output / relative),
                                                "aggregate": evaluation["aggregate"], "mean_inference_ms": evaluation["mean_inference_ms"],
                                                "peak_allocated_cuda_bytes": evaluation["peak_allocated_cuda_bytes"]}
        require(state_sha(model) == evidence[name]["state_before_sha256"], "Forward evaluation changed model state.")
        evidence[name]["state_unchanged"] = True
        model.to("cpu")  # Only one model is resident for allocated-memory measurements.
    require(sha256(protocol_path) == protocol_sha and all(sha256(child(p)) == digest for p, digest in required.items()), "Protocol/source changed during evaluation.")
    require(all(sha256(path) == evidence[name]["checkpoint_sha256"] for name, path in checkpoint_paths.items()), "Checkpoint file changed during evaluation.")
    index["status"] = "completed"
    write_json(output / "appearance_evaluation.json", index)
    print("Completed four models x five conditions x 38 validation volumes; no TEST access or saved arrays.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
