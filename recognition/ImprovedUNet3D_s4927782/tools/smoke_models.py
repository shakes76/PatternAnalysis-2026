"""Synthetic-only model smoke checks; these are NOT accuracy experiments.

No medical data is read. This tool does not install dependencies, request a
cluster allocation, or perform a training experiment. It runs one synthetic
backward pass and an in-memory checkpoint roundtrip per model on the selected
already-available device, then writes the requested JSON report.
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import sys
import time
from pathlib import Path


def run_model(torch, model_class, loss_function, shape, args, device):
    config = {"num_classes": args.num_classes, "base_channels": args.base_channels}
    model = model_class(**config).to(device)
    images = torch.randn((1, 1, *shape), device=device)
    target = torch.randint(args.num_classes, (1, *shape), device=device)
    model.train()

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    output = model(images)
    heads = [output["logits"], *output["aux_logits"]]
    if tuple(output["logits"].shape) != (1, args.num_classes, *shape):
        raise RuntimeError("Main output does not match the synthetic input shape.")
    for head in heads:
        head.retain_grad()
    loss = loss_function(output, target)
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("Synthetic loss is not finite.")
    loss.backward()
    synchronize()
    forward_loss_backward_ms = (time.perf_counter() - start) * 1000
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None

    missing_gradients = []
    invalid_gradients = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is None:
            missing_gradients.append(name)
        elif parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
            invalid_gradients.append(name)
    head_gradient_checks = []
    for head in heads:
        used = head.grad is not None and bool(torch.isfinite(head.grad).all().item())
        nonzero = used and bool((head.grad.abs().sum() > 0).item())
        head_gradient_checks.append({"finite_gradient": used, "nonzero_gradient": nonzero})
    if missing_gradients or invalid_gradients or not all(check["nonzero_gradient"] for check in head_gradient_checks):
        raise RuntimeError("A trainable parameter or output head has missing/invalid gradients.")

    model.eval()
    with torch.inference_mode():
        for _ in range(2):
            model(images)
        synchronize()
        start = time.perf_counter()
        for _ in range(3):
            expected_logits = model(images)["logits"]
        synchronize()
        forward_ms = (time.perf_counter() - start) * 1000 / 3
        expected_prediction = expected_logits.argmax(dim=1)

    buffer = io.BytesIO()
    torch.save({"state_dict": model.state_dict(), "config": config}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, map_location=device, weights_only=True)
    restored = model_class(**saved["config"]).to(device)
    restored.load_state_dict(saved["state_dict"])
    restored.eval()
    with torch.inference_mode():
        restored_logits = restored(images)["logits"]
        torch.testing.assert_close(expected_logits, restored_logits, rtol=1e-5, atol=1e-6)
        actual_prediction = restored_logits.argmax(dim=1)
    roundtrip_equal = bool(torch.equal(expected_prediction, actual_prediction))
    if not roundtrip_equal:
        raise RuntimeError("Checkpoint roundtrip changed categorical predictions.")

    return {
        "model": model_class.__name__,
        "input_shape": list(images.shape),
        "main_output_shape": list(output["logits"].shape),
        "aux_output_shapes": [list(head.shape) for head in output["aux_logits"]],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "synthetic_loss": float(loss.item()),
        "finite_parameter_gradients": not invalid_gradients,
        "all_parameters_used": not missing_gradients,
        "head_gradient_checks_main_then_aux": head_gradient_checks,
        "checkpoint_roundtrip_same_argmax_prediction": roundtrip_equal,
        "checkpoint_roundtrip_logits_close": True,
        "forward_loss_backward_ms": forward_loss_backward_ms,
        "warmed_batch1_forward_ms": forward_ms,
        "timing_warmup_forwards": 2,
        "timing_measured_forwards": 3,
        "peak_allocated_cuda_bytes_forward_loss_backward": peak_memory,
        "aux_weights_half_then_quarter": [0.5, 0.25] if output["aux_logits"] else [],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-channels", type=int, default=4)
    parser.add_argument("--shape3d", type=int, nargs=3, default=(16, 32, 32), metavar=("D", "H", "W"))
    parser.add_argument("--num-classes", type=int, default=3, help="Synthetic classes; no dataset label mapping is implied.")
    args = parser.parse_args(argv)
    if args.base_channels < 1 or args.num_classes < 2 or min(args.shape3d) < 8:
        parser.error("Use positive base channels, at least two classes, and spatial dimensions >= 8.")
    report = {
        "schema_version": 1,
        "synthetic_only": True,
        "medical_data_read": False,
        "accuracy_evaluated": False,
        "status": "blocked",
        "requested_device": args.device,
        "python_version": platform.python_version(),
        "limitations": [
            "Synthetic checks do not establish segmentation accuracy, patient independence, or clinical utility.",
            "Timings use synthetic batch size one and exclude data loading; no end-to-end claim is made.",
        ],
    }
    exit_code = 2
    try:
        import torch
    except ImportError:
        report["error"] = "PyTorch is unavailable. No dependency installation was attempted."
    else:
        try:
            if args.device == "cuda" and not torch.cuda.is_available():
                report["error"] = "CUDA was explicitly requested but is unavailable."
            else:
                sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                from modules import ImprovedUNet3D, UNet2D, segmentation_loss

                device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
                torch.manual_seed(42)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(42)
                    torch.backends.cudnn.benchmark = False
                report.update({
                    "device": str(device),
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                })
                report["models"] = [
                    run_model(torch, UNet2D, segmentation_loss, tuple(args.shape3d[-2:]), args, device),
                    run_model(torch, ImprovedUNet3D, segmentation_loss, tuple(args.shape3d), args, device),
                ]
                report["status"] = "passed"
                exit_code = 0
        except Exception as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            exit_code = 1
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Synthetic-only smoke status: {report['status']}; report: {output.resolve()}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
