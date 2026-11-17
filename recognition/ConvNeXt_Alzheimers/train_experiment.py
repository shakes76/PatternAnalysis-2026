"""Step 8: one controlled regularized CNN run and one ConvNeXt-Tiny run.

Entry: python train.py --step8 --model baseline|convnext ...
Only train and validation images are consumed. Shared finalized manifests,
preprocessing, optimiser schedule, physical/effective batch sizes, selection,
aggregation and evaluation are retained for the two runs. Initial Step 7 is
preserved as a separate historical experiment, never overwritten.

No pretrained weights, architecture libraries, mixed precision, threshold tuning,
test evaluation or automatic hyperparameter search. Code assistance: ChatGPT;
student review, local verification and disclosure are required.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
from torch import nn

from dataset import ADNISliceDataset, DataError, counts_for_records, make_loader, preprocessing_spec, require
from modules import BaselineCNN, ConvNeXtTiny, parameter_count
from training_utils import (aggregate_predictions, checked_bundle, choose_device,
    classification_loss, hardware_info, inference_benchmark, memory_info,
    new_private_output, predict_validation, read_json, reset_memory_peak,
    seed_everything, selection_key, sha256_file, synchronize, utc_now, write_json)

SOURCE_FILES = ("dataset.py", "modules.py", "training_utils.py", "train.py",
                "train_experiment.py", "check_experiment.py")
ARCHITECTURES = {"baseline": "BaselineCNN-v1", "convnext": "ConvNeXtTiny-ADNI-v1"}


@dataclass(frozen=True)
class ExperimentConfig:
    model: str = "convnext"
    batch_size: int = 8
    accumulation_steps: int = 4
    epochs: int = 30
    min_epochs: int = 12
    patience: int = 8
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    warmup_epochs: int = 3
    weight_decay: float = 0.05
    gradient_clip_norm: float = 5.0
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda"
    budget_seconds: float = 10800.0

    def validate(self) -> None:
        require(self.model in ARCHITECTURES, "Unknown model.")
        require(self.batch_size > 0 and self.accumulation_steps > 0, "Invalid batch configuration.")
        require(self.epochs >= 1 and 1 <= self.min_epochs <= self.epochs and self.patience > 0,
                "Invalid epoch/patience limits.")
        require(0 <= self.warmup_epochs < self.epochs, "Warmup must be shorter than training.")
        require(0 < self.min_learning_rate <= self.learning_rate and self.weight_decay >= 0,
                "Invalid optimiser settings.")
        require(self.gradient_clip_norm > 0 and self.num_workers >= 0 and self.budget_seconds > 0,
                "Invalid clipping, workers or runtime budget.")
        require(self.device in ("cuda", "cpu"), "Device must be cpu or cuda.")


def experiment_sources() -> dict[str, str]:
    base = Path(__file__).resolve().parent
    return {name: sha256_file(base / name) for name in SOURCE_FILES}


def build_model(name: str) -> nn.Module:
    require(name in ARCHITECTURES, "Unknown model.")
    return BaselineCNN() if name == "baseline" else ConvNeXtTiny()


def optimizer_for(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    # Apply decay to matrix/kernel weights, not biases, norm scales or LayerScale.
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim > 1 else no_decay).append(parameter)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0}], lr=config.learning_rate)


def epoch_learning_rate(epoch: int, config: ExperimentConfig) -> float:
    require(1 <= epoch <= config.epochs, "Epoch outside configured range.")
    if epoch <= config.warmup_epochs:
        return config.learning_rate * epoch / config.warmup_epochs
    remaining = config.epochs - config.warmup_epochs
    fraction = (epoch - config.warmup_epochs - 1) / max(1, remaining - 1)
    return config.min_learning_rate + 0.5 * (config.learning_rate - config.min_learning_rate) * (
        1.0 + math.cos(math.pi * fraction))


def train_epoch(model: nn.Module, loader: Any, optimizer: torch.optim.Optimizer,
                device: torch.device, config: ExperimentConfig) -> dict[str, Any]:
    """Mean CE over each effective batch, including a correctly weighted tail.

    Accumulation is equivalent to a larger batch for independent deterministic
    samples, NOT for BatchNorm statistics or stochastic residual/dropout masks.
    Both real experiments use the same physical size as well as accumulation.
    """
    require(getattr(loader.dataset, "split", None) == "train", "Only training images may update weights.")
    require(loader.batch_size == config.batch_size and not loader.drop_last,
            "Unexpected training loader batch configuration.")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    count, correct, summed_loss, updates = 0, 0, 0.0, 0
    total = len(loader.dataset)
    effective = config.batch_size * config.accumulation_steps
    window_samples = 0
    synchronize(device)
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        require(set(batch["split"]) == {"train"}, "Non-training sample in optimiser input.")
        if batch_index % config.accumulation_steps == 0:
            window_samples = min(effective, total - count)
        x = batch["image"].to(device, non_blocking=device.type == "cuda")
        y = batch["label"].to(device, non_blocking=device.type == "cuda")
        logits = model(x)
        loss = classification_loss(logits, y)
        require(bool(torch.isfinite(loss)), "Non-finite training loss.")
        n = len(y)
        (loss * (n / window_samples)).backward()
        count += n
        summed_loss += loss.item() * n
        correct += int((logits.detach().argmax(1) == y).sum().item())
        if (batch_index + 1) % config.accumulation_steps == 0 or count == total:
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        if (batch_index + 1) % 400 == 0:
            print(f"  train: {count}/{total} images", flush=True)
    synchronize(device)
    require(count == total and updates == math.ceil(total / effective), "Incomplete training epoch.")
    return {"slice_loss": summed_loss / count, "slice_accuracy": correct / count,
            "images": count, "optimizer_updates": updates, "seconds": time.perf_counter() - started}


def save_model(path: Path, model: nn.Module, config: ExperimentConfig,
               hashes: dict[str, str], epoch: int) -> None:
    value = {"format_version": 1, "architecture": ARCHITECTURES[config.model],
             "model_name": config.model, "model_label_map": {"NC": 0, "AD": 1},
             "configuration": asdict(config), "epoch": epoch,
             "manifest_sha256": hashes, "preprocessing": preprocessing_spec(0.0),
             "source_code_sha256": experiment_sources(),
             "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_model(path: Path, device: torch.device, hashes: dict[str, str],
               model_name: str) -> tuple[nn.Module, dict[str, Any]]:
    # Only load checkpoints produced by this project, not downloaded .pt files.
    value = torch.load(path, map_location="cpu", weights_only=True)
    require(value.get("architecture") == ARCHITECTURES.get(model_name)
            and value.get("model_name") == model_name
            and value.get("model_label_map") == {"NC": 0, "AD": 1}
            and value.get("manifest_sha256") == hashes
            and value.get("preprocessing") == preprocessing_spec(0.0),
            "Checkpoint model, labels, preprocessing or partitions differ.")
    model = build_model(model_name)
    model.load_state_dict(value["model_state"], strict=True)
    return model.to(device).eval(), value


def save_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def plot_history(output: Path, history: list[dict[str, Any]], name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [r["epoch"] for r in history]
    for filename, columns, label in (
        ("loss_curve.png", [("train_slice_loss", "Train (online/train mode)"),
                             ("validation_slice_loss", "Validation (eval mode)")], "Slice loss"),
        ("validation_curve.png", [("validation_patient_accuracy", "Patient accuracy"),
                                   ("validation_patient_macro_f1", "Patient macro-F1"),
                                   ("validation_ad_recall", "Patient AD recall")], "Score")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, legend in columns:
            ax.plot(epochs, [r[key] for r in history], label=legend)
        ax.set(xlabel="Epoch", ylabel=label, title=f"{name}: validation only; no final test")
        if label == "Score":
            ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)


def report_markdown(report: dict[str, Any]) -> str:
    metrics = report["selected_validation_metrics"]
    lines = [f"## Step 8: {report['model_name']} regularized experiment", "",
        "This model was trained from random initial weights on the unchanged patient-independent "
        "training partition. Preprocessing and validation aggregation match the initial CNN. "
        "No pretrained weights, image augmentation, calibration, decision-threshold search or final-test "
        "evaluation were performed. The operating threshold remains 0.5.", "",
        f"Configuration: `{report['configuration']}`.", "",
        "AdamW uses stronger weight decay on kernels/matrices, excluding biases and one-dimensional "
        "normalization/LayerScale parameters. Learning rate has linear warmup followed by cosine decay. "
        "Both Step 8 architectures use the same settings, including physical batch size and gradient "
        "accumulation. Accumulation does not make BatchNorm behave like a larger physical batch.", "",
        "Checkpoint selection: validation patient macro-F1, then lower patient log loss, then earliest. "
        "These are selected development results, not independent final-test estimates.", "",
        f"Completed epochs: {report['epochs_completed']}; selected epoch: {report['selected_epoch']}; "
        f"stop reason: {report['stopping_reason']}.", "",
        "| Unit | N | Accuracy | Macro-F1 | AUROC | AD recall |",
        "|---|---:|---:|---:|---:|---:|"]
    for unit, m in metrics.items():
        auc = f"{m['auroc']:.4f}" if m['auroc'] is not None else "N/A"
        lines.append(f"| {unit} | {m['n']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
                     f"{auc} | {m['per_class']['AD']['recall']:.4f} |")
    lines += ["", f"Parameters: {report['parameters']:,}. Hardware: `{report['hardware']}`.",
        f"Training loop time: {report['training_loop_seconds']:.2f} seconds. "
        f"GPU allocator peaks: `{report['training_loop_gpu_memory']}`.",
        f"Batch-1 model-only latency: `{report['model_only_inference']}`. "
        "This excludes JPEG loading, preprocessing and patient aggregation.", "",
        "Limitations: one seed per configuration, repeated use of the development validation set, "
        "slice-weighted training (patients with more scans contribute more examples), and no clinical "
        "validation. The two Step 8 runs share a recipe, but it is not claimed to be optimal for both "
        "architectures. The Step 7-to-Step 8 change is a multi-setting intervention, not an isolated "
        "causal test of weight decay. Data remain restricted; private predictions/checkpoints are not "
        "committed. Confidence referral and final held-out comparison remain outstanding.", "",
        "AI disclosure: ChatGPT assisted with implementation, tests and this draft. Record the actual "
        "source review and local checks performed; do not claim unperformed checks.", ""]
    return "\n".join(lines)


def run_experiment(data_root: Path, splits_dir: Path, pipeline_report: Path,
                   smoke_report: Path, output_dir: Path,
                   config: ExperimentConfig) -> dict[str, Any]:
    config.validate()
    output = new_private_output(output_dir, data_root, splits_dir)
    report: dict[str, Any] = {"format_version": 1, "created_utc": utc_now(),
        "status": "STARTING", "experiment_completed": False, "model_name": config.model,
        "configuration": asdict(config), "errors": [], "epochs_completed": 0,
        "final_test_images_decoded": 0, "final_test_model_evaluation_performed": False,
        "pretrained_weights_used": False, "dataset_or_splits_modified": False}
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    try:
        bundle = checked_bundle(data_root, splits_dir, pipeline_report)
        smoke = read_json(smoke_report)
        sources = experiment_sources()
        require(smoke.get("step8_smoke_passed") is True
                and smoke.get("manifest_sha256") == bundle.manifest_hashes
                and smoke.get("source_code_sha256") == sources,
                "The Step 8 smoke report must pass for these manifests and code.")
        require(smoke.get("batch_size") == config.batch_size
                and smoke.get("accumulation_steps") == config.accumulation_steps
                and smoke.get("device") == config.device
                and smoke.get("seed") == config.seed,
                "Smoke batch/accumulation/device/seed differ from training.")
        seed_everything(config.seed)
        device = choose_device(config.device)
        train_data = ADNISliceDataset(bundle, "train", rotation_degrees=0.0)
        validation_data = ADNISliceDataset(bundle, "validation", rotation_degrees=0.0)
        train_loader = make_loader(train_data, batch_size=config.batch_size,
            num_workers=config.num_workers, seed=config.seed, pin_memory=device.type == "cuda")
        val_loader = make_loader(validation_data, batch_size=config.batch_size,
            num_workers=config.num_workers, seed=config.seed, pin_memory=device.type == "cuda")
        model = build_model(config.model).to(device)
        optimizer = optimizer_for(model, config)
        report.update(status="RUNNING", manifest_sha256=bundle.manifest_hashes,
            source_code_sha256=sources, parameters=parameter_count(model),
            hardware=hardware_info(device), preprocessing=preprocessing_spec(0.0),
            split_counts={s: counts_for_records(bundle.records[s]) for s in ("train", "validation")},
            architecture=ARCHITECTURES[config.model], initialization="fresh random weights; no smoke weight reuse",
            effective_batch_size=config.batch_size * config.accumulation_steps,
            loss="unweighted slice cross-entropy; correct mean over each accumulation window",
            checkpoint_selection="validation patient macro-F1; lower patient log loss; earliest exact tie",
            aggregation="mean slice p(AD) per scan, then mean scan p(AD) per patient",
            budget_note="Soft wall-time budget checked between completed epochs; one epoch can overrun.",
            resume_supported=False)
        write_json(output / "experiment_summary.json", report)
        best_key = None
        best_metrics = None
        selected_epoch, stale = 0, 0
        reset_memory_peak(device)
        synchronize(device)
        loop_start = time.perf_counter()
        stop_reason = "maximum_epochs"
        for epoch in range(1, config.epochs + 1):
            lr = epoch_learning_rate(epoch, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            print(f"{config.model} epoch {epoch}/{config.epochs}; learning rate {lr:.7f}", flush=True)
            trained = train_epoch(model, train_loader, optimizer, device, config)
            probabilities, seconds = predict_validation(model, val_loader, device)
            metrics, _ = aggregate_predictions(validation_data.records, probabilities)
            key = selection_key(metrics)
            improved = best_key is None or key > best_key
            if improved:
                best_key, best_metrics, selected_epoch, stale = key, metrics, epoch, 0
                save_model(output / "PRIVATE_best_model.pt", model, config, bundle.manifest_hashes, epoch)
            else:
                stale += 1
            save_model(output / "PRIVATE_last_model.pt", model, config, bundle.manifest_hashes, epoch)
            patient = metrics["patient"]
            history.append({"epoch": epoch, "learning_rate": lr,
                "train_slice_loss": trained["slice_loss"], "train_slice_accuracy": trained["slice_accuracy"],
                "validation_slice_loss": metrics["slice"]["log_loss"],
                "validation_patient_accuracy": patient["accuracy"],
                "validation_patient_macro_f1": patient["macro_f1"],
                "validation_patient_auroc": patient["auroc"],
                "validation_patient_log_loss": patient["log_loss"],
                "validation_ad_recall": patient["per_class"]["AD"]["recall"],
                "validation_nc_recall": patient["per_class"]["NC"]["recall"],
                "train_seconds": trained["seconds"], "validation_seconds": seconds,
                "optimizer_updates": trained["optimizer_updates"], "selected_as_best": improved})
            save_history(output / "history.csv", history)
            report.update(epochs_completed=epoch, selected_epoch=selected_epoch,
                          selected_validation_metrics=best_metrics, epochs_without_improvement=stale)
            write_json(output / "experiment_summary.json", report)
            print(f"  patient acc={patient['accuracy']:.3f}; F1={patient['macro_f1']:.3f}; "
                  f"AD recall={patient['per_class']['AD']['recall']:.3f}" +
                  (" [selected]" if improved else ""), flush=True)
            if time.perf_counter() - loop_start >= config.budget_seconds:
                stop_reason = "soft_time_budget"
                break
            if epoch >= config.min_epochs and stale >= config.patience:
                stop_reason = "patience_after_minimum_epochs"
                break
        synchronize(device)
        report.update(training_loop_seconds=time.perf_counter() - loop_start,
            training_loop_gpu_memory=memory_info(device), stopping_reason=stop_reason)
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        selected, checkpoint = load_model(output / "PRIVATE_best_model.pt", device,
                                         bundle.manifest_hashes, config.model)
        probabilities, seconds = predict_validation(selected, val_loader, device)
        metrics, private = aggregate_predictions(validation_data.records, probabilities)
        require(checkpoint["epoch"] == selected_epoch and best_metrics is not None
                and np.isclose(metrics["patient"]["macro_f1"], best_metrics["patient"]["macro_f1"], atol=1e-8)
                and np.isclose(metrics["patient"]["log_loss"], best_metrics["patient"]["log_loss"], atol=1e-6),
                "Reloaded selected checkpoint did not reproduce validation results.")
        write_json(output / "PRIVATE_validation_predictions.json", private)
        report.update(status="COMPLETED", experiment_completed=True,
            selected_validation_metrics=metrics, selected_checkpoint_reloaded_and_checked=True,
            selected_checkpoint_sha256=sha256_file(output / "PRIVATE_best_model.pt"),
            validation_end_to_end_seconds=seconds,
            validation_end_to_end_images_per_second=len(validation_data) / seconds,
            model_only_inference=inference_benchmark(selected, validation_data[0]["image"].unsqueeze(0), device))
        plot_history(output, history, config.model)
        (output / "README_experiment_results.md").write_text(report_markdown(report), encoding="utf-8")
    except BaseException as exc:
        report.update(status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED",
                      experiment_completed=False)
        report["errors"].append({"type": type(exc).__name__,
            "message": str(exc) if isinstance(exc, DataError) else "See private traceback locally."})
        (output / "PRIVATE_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        report["total_run_seconds"] = time.perf_counter() - started
        write_json(output / "experiment_summary.json", report)
    print("STEP8_EXPERIMENT_COMPLETED", flush=True)
    print("VALIDATION_ONLY_FINAL_TEST_NOT_EVALUATED", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "splits-dir", "pipeline-report", "smoke-report", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", choices=tuple(ARCHITECTURES), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    args = parser.parse_args()
    config = ExperimentConfig(model=args.model, device=args.device, batch_size=args.batch_size,
                              accumulation_steps=args.accumulation_steps)
    try:
        run_experiment(args.data_root, args.splits_dir, args.pipeline_report,
                       args.smoke_report, args.output_dir, config)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"STEP8_EXPERIMENT_STOPPED: {type(exc).__name__}. Preserve the output folder.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
