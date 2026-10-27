"""Image-only v2 deployment adapter, separate from frozen training sources.

Reads one MRI, selected v2 weights, protocol and source code. Does not read any
mask or data manifest. Reuses the independently checked image preprocessing and
native restoration in infer_native.py; new-cache predictions use the frozen
v2 image-only sliding/slice inference. Native restoration is a geometry service,
not evidence of native-resolution segmentation accuracy.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import platform
import re
import time

import numpy as np

import infer_native as native
import v2_pipeline as v2


def validate_checkpoint_identity(checkpoint, protocol, pipeline_digest, modules_digest):
    """Validate selection/config/source metadata without opening a data manifest."""
    identity = checkpoint.get("identity", {})
    model_name = identity.get("model")
    expected_sources = {**v2.FROZEN_SOURCE_SHA256, **protocol.get("required_original_sha256", {})}
    if (identity.get("protocol_sha256") != protocol["_sha256"]
            or identity.get("experiment_id") != protocol["experiment_id"]
            or identity.get("configuration") != protocol["configuration"]
            or identity.get("v2_pipeline_sha256") != pipeline_digest
            or identity.get("source_hashes") != expected_sources
            or not re.fullmatch(r"[a-f0-9]{64}", identity.get("manifest_sha256", ""))
            or model_name not in v2.MODELS
            or modules_digest != v2.FROZEN_SOURCE_SHA256["modules.py"]):
        raise ValueError("V2 checkpoint protocol, model, source or cache-identity metadata differs.")
    expected_config = {**protocol["configuration"], "model": model_name, "label_ids": list(range(6))}
    if checkpoint.get("config") != expected_config or checkpoint.get("epoch") != checkpoint.get("best_epoch"):
        raise ValueError("Use the selected v2 best checkpoint with the exact frozen six-class configuration.")
    history = checkpoint.get("history", [])
    if not history or history[-1].get("epoch") != checkpoint["epoch"]:
        raise ValueError("Checkpoint history does not reach its saved selection epoch.")
    best_score, best_epoch = None, None
    for row in history:
        score = row.get("validation_prostate_dice")
        if score is None:
            continue
        if type(score) not in (int, float) or not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Checkpoint history contains an invalid validation score.")
        if row.get("epoch") not in protocol["configuration"]["validation_epochs"]:
            raise ValueError("Checkpoint selection used an unscheduled validation epoch.")
        if best_score is None or score > best_score:
            best_score, best_epoch = score, row["epoch"]
    if best_score != checkpoint.get("best_score") or best_epoch != checkpoint.get("best_epoch"):
        raise ValueError("Checkpoint selection does not match strict maximum and earliest tie.")
    return model_name


def validate_selection_freeze(path, checkpoint, checkpoint_digest, protocol, model_name):
    if path is None:
        return None
    receipt = v2.read_json(path)
    if (receipt.get("status") != "frozen"
            or receipt.get("protocol_sha256") != protocol["_sha256"]
            or receipt.get("manifest_sha256") != checkpoint["identity"]["manifest_sha256"]):
        raise ValueError("Selection freeze is not bound to this checkpoint/protocol.")
    for name in v2.MODELS:
        choice = receipt.get("models", {}).get(name, {})
        if choice.get("completed_epochs") != 30 or not re.fullmatch(r"[a-f0-9]{64}", choice.get("checkpoint_sha256", "")):
            raise ValueError("Selection freeze requires both completed formal models.")
    if receipt["models"][model_name]["checkpoint_sha256"] != checkpoint_digest:
        raise ValueError("Requested weights differ from the frozen selection.")
    return v2.sha256(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-freeze", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error("Use a fresh output directory; existing evidence cannot be overwritten.")
    import torch
    import nibabel
    import scipy

    protocol = v2.load_protocol(args.protocol)
    v2.verify_review_sources(protocol, args.protocol.resolve().parent)
    module_path = args.project_root.expanduser().resolve() / "modules.py"
    pipeline_path = Path(v2.__file__).resolve()
    checkpoint_digest, module_digest = v2.sha256(args.checkpoint), v2.sha256(module_path)
    pipeline_digest, native_digest = v2.sha256(pipeline_path), v2.sha256(native.__file__)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_name = validate_checkpoint_identity(checkpoint, protocol, pipeline_digest, module_digest)
    freeze_digest = validate_selection_freeze(args.selection_freeze, checkpoint, checkpoint_digest, protocol, model_name)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable.")
    specification = importlib.util.spec_from_file_location("v2_image_only_frozen_modules", module_path)
    modules = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(modules)
    config = protocol["configuration"]
    factory = modules.UNet2D if model_name == "unet2d" else modules.ImprovedUNet3D
    model = factory(6, config["base_channels"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    v2.seed_everything(config["seed"])
    image_path = args.image.expanduser().resolve(strict=True)
    image_digest = v2.sha256(image_path)
    start = time.perf_counter()
    image, original, metadata = native.preprocess_image(image_path, tuple(config["target_shape_dhw"]))
    prediction = v2.predict_image(model, torch.from_numpy(image).unsqueeze(0), model_name, device, config)
    if (v2.sha256(image_path) != image_digest or v2.sha256(args.checkpoint) != checkpoint_digest
            or v2.sha256(module_path) != module_digest or v2.sha256(pipeline_path) != pipeline_digest
            or v2.sha256(native.__file__) != native_digest or v2.sha256(args.protocol) != protocol["_sha256"]):
        raise RuntimeError("A guarded image/model/protocol/source changed during inference.")
    receipt = {"image": str(image_path), "image_sha256": image_digest,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_digest,
        "checkpoint_epoch": checkpoint["epoch"], "model_config": checkpoint["config"],
        "checkpoint_identity": checkpoint["identity"], "protocol_sha256": protocol["_sha256"],
        "pipeline_source_sha256": pipeline_digest, "native_adapter_source_sha256": native_digest,
        "deployment_adapter_source_sha256": v2.sha256(__file__), "module_source_sha256": module_digest,
        "device": str(device), "amp": config["amp"] and device.type == "cuda",
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python_version": platform.python_version(), "torch_version": str(torch.__version__),
        "numpy_version": np.__version__, "nibabel_version": nibabel.__version__, "scipy_version": scipy.__version__,
        "elapsed_seconds_before_serialization": time.perf_counter() - start,
        "selection_freeze_sha256": freeze_digest}
    native.save_outputs(args.output_dir, prediction.numpy().astype(np.uint8), original, metadata,
                        list(range(6)), receipt)
    # The reused serializer carries v1-specific descriptive defaults. Correct
    # only this freshly created receipt to describe the actual v2 provenance.
    receipt.update({"model_attribution": "Frozen original modules.py model/loss; newly trained finite v2 pipeline checkpoint, selected using validation class-5 Dice.",
        "model_changed_during_inference": False, "mask_or_manifest_read": False,
        "native_resolution_accuracy_claimed": False,
        "assistance_record": "AI-assisted image-only v2 adapter; frozen v2 inference and independently verified native geometry utilities reused by digest. No training, model selection or target-label access during this inference."})
    v2.write_json(args.output_dir / "receipt.json", receipt)
    print(f"Completed image-only {model_name} v2 inference: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
