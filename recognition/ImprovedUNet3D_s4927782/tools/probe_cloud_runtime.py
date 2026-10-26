"""Record a private runtime inventory and optional synthetic GPU acceptance check.

No data cache, dataset split, checkpoint or training history is read or created.
The optional smoke check uses random tensors and one unscaled backward pass per
model. Its memory/loss values are hardware checks, not formal experiment results.
Hostnames, home paths and addresses make this JSON a private connection artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("torch", "numpy", "matplotlib", "nibabel", "scipy", "importlib_resources")


def command_inventory(arguments):
    """Run only a fixed local inventory command, with bounded runtime."""
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=10, check=False)
        return {"command": arguments, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": arguments, "error": f"{type(error).__name__}: {error}"}


def inventory():
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    # These exact keys can populate amendment.environment in the cloud runner.
    environment = {"python_version": platform.python_version(), "torch_version": versions["torch"],
                   "cuda_version": None, "numpy_version": versions["numpy"],
                   "extra_versions": {name: value for name, value in versions.items() if value is not None}}
    measured_platform = {"provider": None, "provider_status": "unverified; requires operator declaration",
                         "cuda_available": False, "cuda_device_count": 0,
                         "gpu_name": None, "gpu_memory_bytes": None, "gpus": []}
    details = {
        "executable": sys.executable,
        "os": {"platform": platform.platform(), "system": platform.system(),
               "release": platform.release(), "machine": platform.machine()},
        "package_inventory": versions,
        "nvidia_smi": command_inventory([
            "nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits"]),
    }
    try:
        import torch
        environment.update(torch_version=str(torch.__version__), cuda_version=torch.version.cuda)
        details["cudnn_version"] = torch.backends.cudnn.version()
        measured_platform["cuda_available"] = torch.cuda.is_available()
        details["backend"] = {
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
        if measured_platform["cuda_available"]:
            measured_platform["cuda_device_count"] = torch.cuda.device_count()
            for index in range(measured_platform["cuda_device_count"]):
                properties = torch.cuda.get_device_properties(index)
                measured_platform["gpus"].append({
                    "index": index, "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": [properties.major, properties.minor],
                })
            measured_platform["gpu_name"] = measured_platform["gpus"][0]["name"]
            measured_platform["gpu_memory_bytes"] = measured_platform["gpus"][0]["total_memory_bytes"]
    except Exception as error:
        details["torch_inventory_error"] = f"{type(error).__name__}: {error}"
    connection = {"private_artifact": True, "hostname": socket.gethostname(),
                  "home": str(Path.home()), "hostname_I": command_inventory(["hostname", "-I"])}
    return environment, measured_platform, details, connection


def gradient_summary(torch, parameters):
    parameters = list(parameters)
    missing, nonfinite = [], []
    for name, parameter in parameters:
        if parameter.grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(parameter.grad).all().item()):
            nonfinite.append(name)
    return {"parameter_tensors": len(parameters), "missing": missing, "nonfinite": nonfinite,
            "all_present_and_finite": bool(parameters) and not missing and not nonfinite}


def smoke_one(torch, model_name, model_class, segmentation_loss):
    """Use the formal batch/grid sizes, without an optimizer or real samples."""
    device = torch.device("cuda:0")
    shape = (16, 1, 128, 128) if model_name == "unet2d" else (1, 1, 64, 128, 128)
    model = images = targets = output = loss = None
    record = {"model": model_name, "status": "failed", "input_shape": list(shape),
              "target_shape": [shape[0], *shape[2:]], "num_classes": 6,
              "base_channels": 16, "amp": True, "autocast_dtype": "float16",
              "backward": "unscaled; no GradScaler, optimizer step or checkpoint"}
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(3710)
        torch.cuda.manual_seed_all(3710)
        model = model_class(num_classes=6, base_channels=16).to(device).train()
        images = torch.randn(shape, device=device)
        targets = torch.randint(0, 6, record["target_shape"], device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(images)
            loss = segmentation_loss(output, targets)
        heads = [output["logits"], *output["aux_logits"]]
        record["head_shapes"] = [list(head.shape) for head in heads]
        record["logits_finite"] = all(bool(torch.isfinite(head).all().item()) for head in heads)
        record["loss_finite"] = bool(torch.isfinite(loss).item())
        record["loss"] = float(loss.detach().item()) if record["loss_finite"] else None
        if not record["loss_finite"] or not record["logits_finite"]:
            raise RuntimeError("Nonfinite synthetic loss or logits; backward not executed.")
        loss.backward()
        torch.cuda.synchronize(device)
        named = list(model.named_parameters())
        gradients = {"all_parameters": gradient_summary(torch, named),
                     "main_head": gradient_summary(torch, [(n, p) for n, p in named if n.startswith("head.")])}
        for index in range(len(output["aux_logits"])):
            gradients[f"aux_head_{index}"] = gradient_summary(
                torch, [(n, p) for n, p in named if n.startswith(f"aux_heads.{index}.")])
        record["gradients"] = gradients
        if not all(group["all_present_and_finite"] for group in gradients.values()):
            raise RuntimeError("Missing or nonfinite synthetic parameter gradients.")
        record["status"] = "passed"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            torch.cuda.synchronize(device)
            record["peak_allocated_cuda_bytes"] = torch.cuda.max_memory_allocated(device)
        except Exception as error:
            record["cuda_cleanup_error"] = f"{type(error).__name__}: {error}"
            record["status"] = "failed"
        if model is not None:
            model.zero_grad(set_to_none=True)
        model = images = targets = output = loss = None
        # Do not retain references from gradient inspection or the forward graph.
        heads = named = gradients = None
        gc.collect()
        torch.cuda.empty_cache()
    return record


def synthetic_smoke():
    result = {"synthetic_only": True, "real_data_read": False,
              "formal_training_or_resource_benchmark": False, "models": []}
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Synthetic AMP hardware acceptance requires a CUDA GPU.")
        sys.path.insert(0, str(PROJECT_ROOT))
        from modules import ImprovedUNet3D, UNet2D, segmentation_loss
        for name, model_class in (("unet2d", UNet2D), ("improved3d", ImprovedUNet3D)):
            result["models"].append(smoke_one(torch, name, model_class, segmentation_loss))
        result["status"] = "passed" if all(r["status"] == "passed" for r in result["models"]) else "failed"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    return result


def write_new_json(path, payload):
    """Publish a complete private JSON atomically, refusing existing outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New private JSON; existing files are never replaced.")
    parser.add_argument("--synthetic-smoke", action="store_true", help="Run one random-data AMP forward/backward per model on CUDA:0.")
    args = parser.parse_args(argv)
    output = args.output.expanduser().absolute()
    if os.path.lexists(output):
        parser.error("Output already exists; choose a new private artifact path.")
    environment, measured_platform, details, connection = inventory()
    payload = {"schema_version": 1, "probe_only": True,
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "environment": environment, "platform": measured_platform,
               "runtime_details": details, "connection": connection,
               "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (Path(__file__).resolve(), PROJECT_ROOT / "modules.py")},
               "synthetic_smoke": synthetic_smoke() if args.synthetic_smoke else {"status": "not_requested"}}
    try:
        write_new_json(output, payload)
    except OSError as error:
        parser.exit(2, f"Could not create new private probe artifact: {error}\n")
    print(f"Private runtime probe saved: {output}")
    return 1 if payload["synthetic_smoke"]["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
