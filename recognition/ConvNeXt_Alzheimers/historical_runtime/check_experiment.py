"""One training-only CUDA check for both Step 8 architectures; no test/validation.

Verify accumulated optimisation, finite gradients, weight updates and checkpoint
round-trip. This is not evidence of patient generalisation or expected accuracy.
The Step 7 tiny-overfit result is retained; this check does not repeat that run.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import torch

from dataset import ADNISliceDataset, DataError, make_loader, require
from modules import parameter_count
from train_experiment import (ExperimentConfig, build_model, experiment_sources,
                              load_model, optimizer_for, save_model, train_epoch)
from training_utils import (checked_bundle, choose_device, hardware_info, memory_info,
                            new_private_output, reset_memory_peak, seed_everything,
                            synchronize, utc_now, write_json)


def run_smoke(data_root: Path, splits_dir: Path, pipeline_report: Path,
              output_dir: Path, *, device_name: str = "cuda", batch_size: int = 8,
              accumulation_steps: int = 4, seed: int = 42) -> dict[str, Any]:
    config = ExperimentConfig(batch_size=batch_size, accumulation_steps=accumulation_steps,
                              seed=seed, device=device_name)
    config.validate()
    output = new_private_output(output_dir, data_root, splits_dir)
    report: dict[str, Any] = {"format_version": 1, "created_utc": utc_now(),
        "step8_smoke_passed": False, "batch_size": batch_size,
        "accumulation_steps": accumulation_steps, "seed": seed, "device": device_name,
        "final_test_images_decoded": 0, "validation_images_decoded": 0,
        "generalisation_evidence": False, "smoke_weights_reused": False,
        "models": {}, "errors": []}
    started = time.perf_counter()
    try:
        bundle = checked_bundle(data_root, splits_dir, pipeline_report)
        seed_everything(seed)
        device = choose_device(device_name)
        report.update(manifest_sha256=bundle.manifest_hashes,
            source_code_sha256=experiment_sources(), hardware=hardware_info(device))
        all_records = bundle.records["train"]
        per_class = max(1, batch_size * accumulation_steps // 2)
        rows = [r for r in all_records if r.label == 0][:per_class]
        rows += [r for r in all_records if r.label == 1][:per_class]
        require({r.label for r in rows} == {0, 1}, "Both training classes are required for smoke check.")
        for name in ("baseline", "convnext"):
            print(f"Training-only smoke check: {name}", flush=True)
            seed_everything(seed)
            dataset = ADNISliceDataset(bundle, "train", rotation_degrees=0.0)
            dataset.records = tuple(rows)  # Diagnostic subset; no manifests are altered.
            loader = make_loader(dataset, batch_size=batch_size, num_workers=0,
                                 seed=seed, pin_memory=device.type == "cuda")
            model_config = ExperimentConfig(model=name, batch_size=batch_size,
                accumulation_steps=accumulation_steps, seed=seed, device=device_name)
            model = build_model(name).to(device)
            optimizer = optimizer_for(model, model_config)
            before = next(model.parameters()).detach().clone()
            reset_memory_peak(device)
            trained = train_epoch(model, loader, optimizer, device, model_config)
            require(not torch.equal(before, next(model.parameters()).detach()),
                    "The first model parameter did not update.")
            peak = memory_info(device)
            x = dataset[0]["image"].unsqueeze(0).to(device)
            model.eval()
            with torch.inference_mode():
                expected = model(x).cpu()
            require(expected.shape == (1, 2) and bool(torch.isfinite(expected).all()),
                    "Invalid model output.")
            path = output / f"PRIVATE_{name}_smoke.pt"
            save_model(path, model, model_config, bundle.manifest_hashes, epoch=0)
            count = parameter_count(model)
            del model, optimizer, before
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            restored, _ = load_model(path, device, bundle.manifest_hashes, name)
            with torch.inference_mode():
                observed = restored(x).cpu()
            difference = (expected - observed).abs().max().item()
            require(torch.allclose(expected, observed, rtol=1e-5, atol=1e-6),
                    "Checkpoint reloading changed predictions.")
            report["models"][name] = {"passed": True, "parameters": count,
                "training_step": trained, "finite_gradients": True, "parameters_updated": True,
                "checkpoint_max_absolute_logit_difference": difference,
                "gpu_training_memory": peak}
            del restored, x
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            write_json(output / "step8_smoke_summary.json", report)
        report["step8_smoke_passed"] = True
        synchronize(device)
    except BaseException as exc:
        report["errors"].append({"type": type(exc).__name__,
            "message": str(exc) if isinstance(exc, DataError) else "See private traceback locally."})
        (output / "PRIVATE_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "step8_smoke_summary.json", report)
    print("STEP8_SMOKE_CHECK_PASSED", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "splits-dir", "pipeline-report", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    args = parser.parse_args()
    try:
        run_smoke(args.data_root, args.splits_dir, args.pipeline_report, args.output_dir,
                  device_name=args.device, batch_size=args.batch_size,
                  accumulation_steps=args.accumulation_steps)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"STEP8_SMOKE_STOPPED: {type(exc).__name__}. Preserve the output folder.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
