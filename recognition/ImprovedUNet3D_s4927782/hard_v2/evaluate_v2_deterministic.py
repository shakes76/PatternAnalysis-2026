"""Explicit evaluation execution adapter; frozen model/scientific code is reused.

Static review found that standalone v2 evaluation did not initialise the same
cuDNN/deterministic flags as v2 training. No numerical effect has been measured.
This adapter aligns those flags before the original evaluation entry point and
writes a separately hash-bound runtime receipt, without changing its metrics.
Unsupported CUDA kernels can remain nondeterministic under warn-only settings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import v2_pipeline as v2

HERE = Path(__file__).resolve().parent
DRIVER_NAME = "formal_v2_deterministic_gpu.slurm"
AMENDMENT_NAME = "engineering_execution_amendment.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=("evaluate",), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", choices=("unet2d", "improved3d"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--execution-amendment", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    # Only the adapter-specific argument is removed. Original options and their
    # order are passed unchanged to the frozen pipeline's own full parser.
    forwarded, index = [], 0
    while index < len(argv):
        token = argv[index]
        if token == "--execution-amendment":
            index += 2
            continue
        if token.startswith("--execution-amendment="):
            index += 1
            continue
        forwarded.append(token)
        index += 1
    return args, forwarded


def validate_amendment(path, protocol_path, protocol):
    amendment = read_json(path)
    original_protocol_sha = read_json(HERE / "SOURCE_FREEZE.json")["protocol_sha256"]
    actual_protocol_sha = sha256(protocol_path)
    if (actual_protocol_sha != original_protocol_sha
            or amendment.get("experiment_id") != protocol["experiment_id"]
            or amendment.get("original_protocol_sha256") != original_protocol_sha
            or amendment.get("engineering_only") is not True
            or amendment.get("scientific_configuration_changed") is not False
            or amendment.get("wrapper_sha256") != sha256(__file__)
            or amendment.get("new_driver_sha256") != sha256(HERE / DRIVER_NAME)
            or not isinstance(amendment.get("reason"), str) or not amendment["reason"].strip()):
        raise ValueError("Evaluation execution amendment/source/protocol identity differs.")
    if Path(path).name != AMENDMENT_NAME:
        raise ValueError("Use the explicitly recorded engineering_execution_amendment.json.")
    return amendment


def collect_runtime(torch_api, device, configuration):
    import numpy
    import scipy
    import nibabel

    return {
        "device": str(device),
        "gpu_name": torch_api.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python_version": platform.python_version(), "torch_version": str(torch_api.__version__),
        "cuda_version": torch_api.version.cuda, "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__, "nibabel_version": nibabel.__version__,
        "amp": bool(configuration["amp"] and device.type == "cuda"),
        "seed": configuration["seed"],
        "cudnn_deterministic": bool(torch_api.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch_api.backends.cudnn.benchmark),
        "deterministic_algorithms_enabled": bool(torch_api.are_deterministic_algorithms_enabled()),
        "deterministic_algorithms_warn_only": bool(torch_api.is_deterministic_algorithms_warn_only_enabled()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST"),
    }


def require_matching_flags(runtime):
    if (runtime.get("cudnn_deterministic") is not True
            or runtime.get("cudnn_benchmark") is not False
            or runtime.get("deterministic_algorithms_enabled") is not True
            or runtime.get("deterministic_algorithms_warn_only") is not True
            or runtime.get("seed") != 3711):
        raise RuntimeError("Evaluation did not retain the predeclared training execution flags.")


def execute(argv, pipeline=v2, runtime_collector=collect_runtime):
    args, forwarded = parse_args(list(argv))
    output = args.output_dir.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a fresh evaluation output directory; prior evidence is preserved.")
    protocol = pipeline.load_protocol(args.protocol)
    pipeline.verify_review_sources(protocol, args.protocol.resolve().parent)
    validate_amendment(args.execution_amendment, args.protocol, protocol)
    configuration = protocol["configuration"]
    if configuration["seed"] != 3711:
        raise ValueError("Execution adapter requires the originally frozen seed3711.")
    torch_api = pipeline.torch
    if torch_api is None:
        raise ImportError("Actual evaluation execution requires the measured PyTorch environment.")
    device = torch_api.device("cuda" if args.device == "auto" and torch_api.cuda.is_available()
                              else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch_api.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    checkpoint_digest, manifest_digest = sha256(args.checkpoint), sha256(args.manifest)
    amendment_digest, adapter_digest = sha256(args.execution_amendment), sha256(__file__)
    pipeline.seed_everything(configuration["seed"])
    before = runtime_collector(torch_api, device, configuration)
    require_matching_flags(before)
    result = pipeline.main(forwarded)
    if result not in (None, 0):
        raise RuntimeError("Frozen pipeline evaluation did not finish successfully.")
    after = runtime_collector(torch_api, device, configuration)
    require_matching_flags(after)
    if before != after:
        raise RuntimeError("Evaluation execution environment or flags changed during the call.")
    evaluation_path, metrics_path = output / "evaluation.json", output / "metrics.json"
    evaluation, metrics = read_json(evaluation_path), read_json(metrics_path)
    if evaluation != metrics:
        raise ValueError("Original evaluation and metrics files disagree.")
    identity = evaluation.get("identity", {})
    if (identity.get("model") != args.model
            or identity.get("protocol_sha256") != sha256(args.protocol)
            or identity.get("manifest_sha256") != manifest_digest
            or evaluation.get("checkpoint_sha256") != checkpoint_digest
            or evaluation.get("split") != args.split):
        raise ValueError("Evaluation output identity differs from requested fixed inputs.")
    if (sha256(args.checkpoint) != checkpoint_digest or sha256(args.manifest) != manifest_digest
            or sha256(args.execution_amendment) != amendment_digest or sha256(__file__) != adapter_digest):
        raise RuntimeError("Execution inputs changed during evaluation.")
    receipt = {
        "status": "completed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_sha256": sha256(evaluation_path), "metrics_sha256": sha256(metrics_path),
        "checkpoint_sha256": checkpoint_digest, "protocol_sha256": sha256(args.protocol),
        "manifest_sha256": manifest_digest, "adapter_sha256": adapter_digest,
        "adapter_file": "evaluate_v2_deterministic.py",
        "execution_amendment_file": AMENDMENT_NAME,
        "execution_amendment_sha256": amendment_digest, **after,
        "execution_note": "Same cuDNN/deterministic flags as training; warn-only unsupported CUDA kernels remain a limitation. No measured numerical discrepancy preceded this static correction.",
    }
    target = output / "runtime_receipt.json"
    # The original metrics are not rewritten or enriched: this is a new sidecar.
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(f"Evaluation execution runtime recorded: {target}")
    return 0


def main(argv=None):
    return execute(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
