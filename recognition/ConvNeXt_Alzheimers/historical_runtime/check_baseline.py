"""Step 7 training-only smoke test: gradients, tiny fitting and checkpoint reload.

The tiny fit is a debugging check, not validation/test performance. Its weights
are NEVER used to initialise the actual baseline experiment. No final-test image
is decoded. Outputs/checkpoints must remain outside the public Git repository.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import torch
from torch import nn

from dataset import ADNISliceDataset, DataError, make_loader, require
from modules import BaselineCNN, parameter_count
from training_utils import (classification_loss, checked_bundle, choose_device, hardware_info,
                            load_checkpoint, memory_info, new_private_output,
                            reset_memory_peak, save_checkpoint, seed_everything,
                            select_tiny_training_indices, source_hashes,
                            synchronize, utc_now, write_json)


def run_smoke(data_root: Path, splits_dir: Path, pipeline_report: Path,
              output_dir: Path, *, device_name: str = "cuda", batch_size: int = 32,
              seed: int = 42, max_updates: int = 200) -> dict[str, Any]:
    require(batch_size > 0 and max_updates > 0, "Batch size and update limit must be positive.")
    output = new_private_output(output_dir, data_root, splits_dir)
    report: dict[str, Any] = {"format_version": 1, "created_utc": utc_now(),
                             "smoke_check_passed": False,
                             "final_test_images_decoded": 0,
                             "validation_images_decoded": 0,
                             "generalisation_evidence": False,
                             "tiny_weights_reused_for_full_training": False,
                             "batch_size": batch_size, "seed": seed, "errors": []}
    started = time.perf_counter()
    try:
        bundle = checked_bundle(data_root, splits_dir, pipeline_report)
        seed_everything(seed)
        device = choose_device(device_name)
        report.update(manifest_sha256=bundle.manifest_hashes,
                      source_code_sha256=source_hashes(), hardware=hardware_info(device))
        train_data = ADNISliceDataset(bundle, "train", rotation_degrees=0.0)
        loader = make_loader(train_data, batch_size=batch_size, num_workers=0,
                             seed=seed, pin_memory=device.type == "cuda")
        batch = next(iter(loader))
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        model = BaselineCNN().to(device).train()
        report["parameters"] = parameter_count(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        reset_memory_peak(device)
        before = next(model.parameters()).detach().clone()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        require(logits.shape == (len(y), 2) and bool(torch.isfinite(logits).all()),
                "The baseline produced invalid logits.")
        loss = classification_loss(logits, y)
        require(bool(torch.isfinite(loss)), "Non-finite training loss.")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        changed = not torch.equal(before, next(model.parameters()).detach())
        require(changed, "Parameters did not change after the optimizer step.")
        report["training_step"] = {"samples": len(y), "loss": loss.item(),
                                   "gradient_norm_before_clipping": float(norm),
                                   "finite_gradients": True, "parameters_updated": changed}
        # Test inference after loading the saved state using exactly the same input.
        model.eval()
        with torch.inference_mode():
            expected = model(x[:2]).detach().cpu()
        save_checkpoint(output / "PRIVATE_smoke_checkpoint.pt", model, epoch=0,
                        manifest_hashes=bundle.manifest_hashes,
                        config={"purpose": "debug_only", "seed": seed})
        restored, _ = load_checkpoint(output / "PRIVATE_smoke_checkpoint.pt", device,
                                      bundle.manifest_hashes)
        with torch.inference_mode():
            observed = restored(x[:2]).detach().cpu()
        difference = (expected - observed).abs().max().item()
        require(torch.allclose(expected, observed, rtol=1e-5, atol=1e-6),
                "Checkpoint reload did not reproduce inference.")
        report["checkpoint_reload"] = {"passed": True, "max_absolute_logit_difference": difference}
        report["batch_training_memory"] = memory_info(device)
        del model, restored, optimizer, x, y, before, logits, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # A fresh diagnostic model sees one middle slice from up to four patients
        # per class. Dropout and weight decay are disabled ONLY for this tiny fit.
        indices = select_tiny_training_indices(train_data, per_class=4, seed=seed)
        items = [train_data[index] for index in indices]
        tiny_x = torch.stack([item["image"] for item in items]).to(device)
        tiny_y = torch.tensor([item["label"] for item in items], dtype=torch.long, device=device)
        seed_everything(seed)
        tiny_model = BaselineCNN(dropout=0.0).to(device)
        tiny_optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3, weight_decay=0.0)
        tiny_model.eval()
        with torch.inference_mode():
            initial_loss = classification_loss(tiny_model(tiny_x), tiny_y).item()
        trace: list[dict[str, Any]] = []
        passed = False
        for update in range(1, max_updates + 1):
            tiny_model.train()
            tiny_optimizer.zero_grad(set_to_none=True)
            tiny_logits = tiny_model(tiny_x)
            tiny_loss = classification_loss(tiny_logits, tiny_y)
            require(bool(torch.isfinite(tiny_loss)), "Tiny-subset loss became non-finite.")
            tiny_loss.backward()
            nn.utils.clip_grad_norm_(tiny_model.parameters(), 5.0, error_if_nonfinite=True)
            tiny_optimizer.step()
            if update % 10 == 0 or update == max_updates:
                tiny_model.eval()
                with torch.inference_mode():
                    evaluated = tiny_model(tiny_x)
                    evaluated_loss = classification_loss(evaluated, tiny_y).item()
                    accuracy = (evaluated.argmax(1) == tiny_y).float().mean().item()
                trace.append({"updates": update, "training_subset_eval_mode_loss": evaluated_loss,
                              "training_subset_accuracy": accuracy})
                print(f"Tiny training-only check {update}/{max_updates}: "
                      f"loss={evaluated_loss:.4f}, accuracy={accuracy:.3f}", flush=True)
                if accuracy >= 0.95 and evaluated_loss < 0.10 and evaluated_loss < initial_loss:
                    passed = True
                    break
        report["tiny_fit"] = {"passed": passed, "samples": len(items),
                              "patients": len({item['subject_id'] for item in items}),
                              "initial_loss": initial_loss, "maximum_updates": max_updates,
                              "updates_performed": update, "dropout": 0.0, "weight_decay": 0.0,
                              "criterion": "training-subset accuracy >=0.95 and loss <0.10 and below initial loss",
                              "trace": trace}
        require(passed, "Tiny training-subset fitting criterion was not reached. Inspect before full training.")
        report["smoke_check_passed"] = True
        synchronize(device)
    except BaseException as exc:
        report["errors"].append({"type": type(exc).__name__,
                                 "message": str(exc) if isinstance(exc, DataError) else "See private traceback locally."})
        (output / "PRIVATE_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "baseline_smoke_summary.json", report)
    print("BASELINE_SMOKE_CHECK_PASSED", flush=True)
    print("This is training-only debugging evidence, not validation accuracy.", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--pipeline-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-updates", type=int, default=200)
    args = parser.parse_args()
    try:
        run_smoke(args.data_root, args.splits_dir, args.pipeline_report, args.output_dir,
                  device_name=args.device, batch_size=args.batch_size, seed=args.seed,
                  max_updates=args.max_updates)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"BASELINE_SMOKE_STOPPED: {type(exc).__name__}. Read the saved summary.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
