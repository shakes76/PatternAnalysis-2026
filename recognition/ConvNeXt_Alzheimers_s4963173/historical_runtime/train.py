"""Train the initial CNN baseline; training + validation only, no test evaluation.

One predeclared experiment: AdamW, 1e-3 learning rate, 1e-4 weight decay,
full float32 precision, batch size 32, seed 42, <=20 epochs, patience 5.
Select on patient-level validation macro-F1, break exact ties by lower patient
log loss, then retain the earliest epoch. No threshold tuning or calibration yet.
The actual Hard-Difficulty ConvNeXt implementation is a subsequent step.

Development assistance: ChatGPT; review, validate and disclose this assistance.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
from torch import nn

from dataset import (ADNISliceDataset, DataError, counts_for_records, make_loader,
                     preprocessing_spec, require)
from modules import BaselineCNN, parameter_count
from training_utils import (classification_loss, aggregate_predictions, binary_metrics, checked_bundle,
                            choose_device, hardware_info, inference_benchmark,
                            load_checkpoint, memory_info, new_private_output,
                            predict_validation, read_json, reset_memory_peak,
                            save_checkpoint, seed_everything, selection_key,
                            sha256_file, source_hashes, synchronize, utc_now, write_json)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    epochs: int = 20
    patience: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 0
    gradient_clip_norm: float = 5.0
    device: str = "cuda"

    def validate(self) -> None:
        require(self.batch_size > 0 and self.epochs > 0 and self.patience > 0,
                "Batch size, epochs and patience must be positive.")
        require(self.num_workers >= 0 and self.learning_rate > 0 and self.weight_decay >= 0,
                "Invalid optimizer/loader settings.")
        require(self.gradient_clip_norm > 0, "Gradient clipping threshold must be positive.")


def train_epoch(model: nn.Module, loader: Any, optimizer: torch.optim.Optimizer,
                device: torch.device, clip: float) -> dict[str, Any]:
    require(getattr(loader.dataset, "split", None) == "train", "Training loader must use training images only.")
    model.train()
    count, correct, summed_loss = 0, 0, 0.0
    synchronize(device)
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        require(set(batch["split"]) == {"train"}, "A training batch has a non-training record.")
        x = batch["image"].to(device, non_blocking=device.type == "cuda")
        y = batch["label"].to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = classification_loss(logits, y)
        require(bool(torch.isfinite(loss)), "Training loss became non-finite.")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
        optimizer.step()
        n = len(y)
        count += n
        summed_loss += loss.item() * n
        correct += int((logits.detach().argmax(1) == y).sum().item())
        if batch_index % 200 == 0:
            print(f"  train: {count}/{len(loader.dataset)} images", flush=True)
    synchronize(device)
    require(count == len(loader.dataset), "A training epoch did not process every image.")
    return {"slice_loss": summed_loss / count, "slice_accuracy": correct / count,
            "images": count, "seconds": time.perf_counter() - started}


def save_history(output: Path, history: list[dict[str, Any]]) -> None:
    with (output / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ("epoch", "train_slice_loss", "train_slice_accuracy", "validation_slice_loss",
                  "validation_patient_accuracy", "validation_patient_macro_f1",
                  "validation_patient_auroc", "validation_patient_log_loss", "train_seconds",
                  "validation_seconds", "selected_as_best")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def plot_history(output: Path, history: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [row["train_slice_loss"] for row in history], label="Training (online, train mode)")
    ax.plot(epochs, [row["validation_slice_loss"] for row in history], label="Validation (eval mode)")
    ax.set(xlabel="Epoch", ylabel="Slice cross-entropy / negative log likelihood",
           title="Baseline loss — no final-test evaluation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "loss_curve.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [row["validation_patient_accuracy"] for row in history], label="Patient accuracy")
    ax.plot(epochs, [row["validation_patient_macro_f1"] for row in history], label="Patient macro-F1")
    ax.set(xlabel="Epoch", ylabel="Score", ylim=(0, 1), title="Baseline validation patient metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "validation_curve.png", dpi=160)
    plt.close(fig)


def majority_reference(train_records: Any, validation_records: Any) -> dict[str, Any]:
    patients = {r.subject_id: r.label for r in train_records}
    majority = int(sum(patients.values()) > len(patients) / 2)  # tie -> NC
    val_patients = {r.subject_id: r.label for r in validation_records}
    metrics = binary_metrics(list(val_patients.values()), [float(majority)] * len(val_patients))
    return {"selected_using": "training patient counts only", "predicted_class": "AD" if majority else "NC",
            "validation_patient_metrics": metrics,
            "note": "Sanity comparator only, not a substitute for the trained CNN baseline."}


def make_report_markdown(report: dict[str, Any]) -> str:
    lines = ["## Initial CNN baseline experiment", "",
             "A four-block convolutional classifier was trained from scratch as the simpler comparison "
             "for the forthcoming ConvNeXt model. It uses the unchanged one-channel, 224-by-224 data "
             "pipeline. The fixed model labels are NC = 0 and AD = 1.", "",
             "### Predeclared training and model selection", "",
             f"Configuration: `{report['configuration']}`. Training uses ordinary unweighted slice-level "
             "cross-entropy and uniformly shuffled images; repeated scans therefore contribute more "
             "training examples. No class weighting, data augmentation, pretrained weights, learning-rate "
             "search, calibration or threshold tuning was used in this initial experiment.", "",
             "Validation aggregation averages slice probabilities within each scan, then averages the "
             "scan probabilities within each patient. The probability threshold is 0.5. Checkpoint selection "
             "uses patient macro-F1, with lower patient log loss breaking exact ties; the earliest exact "
             "tie is retained. Early stopping uses the same rule. These validation results are selected "
             "development results, not an unbiased final-test estimate.", "",
             f"Completed epochs: **{report['epochs_completed']}**; selected epoch: **{report['selected_epoch']}**.", "",
             "| Validation unit | N | Accuracy | Macro-F1 | AUROC |",
             "|---|---:|---:|---:|---:|"]
    for unit, metrics in report["selected_validation_metrics"].items():
        auc = f"{metrics['auroc']:.4f}" if metrics["auroc"] is not None else "Not defined"
        lines.append(f"| {unit} | {metrics['n']} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | {auc} |")
    patient = report["selected_validation_metrics"]["patient"]
    lines += ["", f"Patient AD recall: **{patient['per_class']['AD']['recall']:.4f}**. "
              f"Patient NC recall: **{patient['per_class']['NC']['recall']:.4f}**.",
              f"Training-majority-class reference validation accuracy: "
              f"**{report['majority_class_reference']['validation_patient_metrics']['accuracy']:.4f}**.", "",
              "### Resources and reproducibility", "",
              f"Learnable parameters: **{report['parameters']:,}**. Hardware: `{report['hardware']}`.",
              f"Training-loop wall time (all completed epochs, including validation/checkpoints/history): "
              f"**{report['training_loop_seconds']:.2f} seconds**.",
              f"GPU allocator peaks: `{report['training_loop_gpu_memory']}`.",
              f"Model-only batch-size-1 latency: `{report['model_only_inference']}`. "
              "This excludes image decoding/preprocessing and is not a full-patient latency.", "",
              "CSV hashes, source-code hashes, the exact configuration, per-epoch history and the selected "
              "checkpoint were retained. Checkpoints and patient-linked prediction files are private and "
              "are not committed to GitHub. The final test images have not been evaluated by this step.", "",
              "### Limitations and next experiment", "",
              "This is one initial seed/configuration. The small training-subset fitting check is a "
              "debugging test, not evidence of generalisation. Validation performance may be optimistic "
              "because it selected the checkpoint. The test comparison, confidence-based referral, "
              "failure analysis and comparison with ConvNeXt remain outstanding.", "",
              "### AI assistance for this stage", "",
              "ChatGPT assisted with the baseline architecture, training/evaluation code, synthetic tests "
              "and this report draft. The student must document the checks they personally performed and "
              "must not treat generated code or test results as a substitute for understanding the model.", ""]
    return "\n".join(lines)


def run_training(data_root: Path, splits_dir: Path, pipeline_report: Path,
                 smoke_report: Path, output_dir: Path, config: TrainConfig) -> dict[str, Any]:
    config.validate()
    output = new_private_output(output_dir, data_root, splits_dir)
    report: dict[str, Any] = {"format_version": 1, "created_utc": utc_now(), "status": "STARTING",
                             "baseline_training_completed": False, "final_test_images_decoded": 0,
                             "final_test_model_evaluation_performed": False,
                             "configuration": asdict(config), "errors": [], "epochs_completed": 0,
                             "source_rangpur_copy_hash_verified": False}
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    try:
        bundle = checked_bundle(data_root, splits_dir, pipeline_report)
        smoke = read_json(smoke_report)
        current_sources = source_hashes()
        require(smoke.get("smoke_check_passed") is True, "A passed baseline smoke report is required.")
        require(smoke.get("manifest_sha256") == bundle.manifest_hashes
                and smoke.get("source_code_sha256") == current_sources,
                "Smoke report refers to different manifests or source files.")
        require(smoke.get("batch_size") == config.batch_size
                and smoke.get("hardware", {}).get("device") == config.device,
                "Training batch size/device must match the smoke check.")
        seed_everything(config.seed)
        device = choose_device(config.device)
        train_data = ADNISliceDataset(bundle, "train", rotation_degrees=0.0)
        validation_data = ADNISliceDataset(bundle, "validation", rotation_degrees=0.0)
        train_loader = make_loader(train_data, batch_size=config.batch_size,
                                   num_workers=config.num_workers, seed=config.seed,
                                   pin_memory=device.type == "cuda")
        validation_loader = make_loader(validation_data, batch_size=config.batch_size,
                                        num_workers=config.num_workers, seed=config.seed,
                                        pin_memory=device.type == "cuda")
        # Fresh model and optimizer; never reuse diagnostic/tiny-fit weights.
        model = BaselineCNN().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
        report.update(status="RUNNING", manifest_sha256=bundle.manifest_hashes,
                      source_code_sha256=current_sources, hardware=hardware_info(device),
                      parameters=parameter_count(model), preprocessing=preprocessing_spec(0.0),
                      split_counts={s: counts_for_records(bundle.records[s]) for s in ("train", "validation")},
                      initialization="fresh random weights; smoke weights not reused",
                      checkpoint_selection="highest validation patient macro-F1; then lower patient log loss; then earliest",
                      aggregation="mean slice p(AD) per scan, then mean scan p(AD) per patient",
                      training_weighting="uniform slice shuffling; unweighted cross-entropy; no patient-balanced sampler",
                      majority_class_reference=majority_reference(bundle.records["train"], bundle.records["validation"]))
        write_json(output / "baseline_summary.json", report)
        reset_memory_peak(device)
        best_key: tuple[float, float] | None = None
        best_metrics: dict[str, Any] | None = None
        selected_epoch, stale = 0, 0
        synchronize(device)
        loop_started = time.perf_counter()
        for epoch in range(1, config.epochs + 1):
            print(f"Epoch {epoch}/{config.epochs}", flush=True)
            trained = train_epoch(model, train_loader, optimizer, device, config.gradient_clip_norm)
            probabilities, validation_seconds = predict_validation(model, validation_loader, device)
            metrics, _ = aggregate_predictions(validation_data.records, probabilities)
            key = selection_key(metrics)
            improved = best_key is None or key > best_key
            if improved:
                best_key, best_metrics, selected_epoch, stale = key, metrics, epoch, 0
                save_checkpoint(output / "PRIVATE_best_baseline.pt", model, epoch=epoch,
                                manifest_hashes=bundle.manifest_hashes, config=asdict(config))
            else:
                stale += 1
            save_checkpoint(output / "PRIVATE_last_baseline.pt", model, epoch=epoch,
                            manifest_hashes=bundle.manifest_hashes, config=asdict(config), optimizer=optimizer)
            row = {"epoch": epoch, "train_slice_loss": trained["slice_loss"],
                   "train_slice_accuracy": trained["slice_accuracy"],
                   "validation_slice_loss": metrics["slice"]["log_loss"],
                   "validation_patient_accuracy": metrics["patient"]["accuracy"],
                   "validation_patient_macro_f1": metrics["patient"]["macro_f1"],
                   "validation_patient_auroc": metrics["patient"]["auroc"],
                   "validation_patient_log_loss": metrics["patient"]["log_loss"],
                   "train_seconds": trained["seconds"], "validation_seconds": validation_seconds,
                   "selected_as_best": improved}
            history.append(row)
            save_history(output, history)
            report.update(epochs_completed=epoch, selected_epoch=selected_epoch,
                          selected_validation_metrics=best_metrics, epochs_without_improvement=stale)
            write_json(output / "baseline_summary.json", report)
            print(f"  train loss={trained['slice_loss']:.4f}; validation patient accuracy="
                  f"{metrics['patient']['accuracy']:.3f}, macro-F1={metrics['patient']['macro_f1']:.3f}, "
                  f"AD recall={metrics['patient']['per_class']['AD']['recall']:.3f}" +
                  (" [selected]" if improved else ""), flush=True)
            if stale >= config.patience:
                print("Early stopping: the predeclared patience has been reached.", flush=True)
                break
        synchronize(device)
        report["training_loop_seconds"] = time.perf_counter() - loop_started
        report["training_loop_gpu_memory"] = memory_info(device)
        report["stopping_reason"] = "patience" if stale >= config.patience else "maximum_epochs"
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        selected, checkpoint = load_checkpoint(output / "PRIVATE_best_baseline.pt", device, bundle.manifest_hashes)
        probabilities, validation_seconds = predict_validation(selected, validation_loader, device)
        metrics, private_predictions = aggregate_predictions(validation_data.records, probabilities)
        require(checkpoint["epoch"] == selected_epoch and best_metrics is not None,
                "Selected checkpoint does not match the chosen epoch.")
        require(np.isclose(metrics["patient"]["macro_f1"], best_metrics["patient"]["macro_f1"], atol=1e-8)
                and np.isclose(metrics["patient"]["log_loss"], best_metrics["patient"]["log_loss"], atol=1e-6),
                "Reloaded checkpoint did not reproduce selected validation metrics.")
        write_json(output / "PRIVATE_validation_predictions.json", private_predictions)
        latency = inference_benchmark(selected, validation_data[0]["image"].unsqueeze(0), device)
        report.update(status="COMPLETED", baseline_training_completed=True,
                      selected_validation_metrics=metrics,
                      selected_checkpoint_reloaded_and_checked=True,
                      selected_checkpoint_sha256=sha256_file(output / "PRIVATE_best_baseline.pt"),
                      validation_end_to_end_seconds=validation_seconds,
                      validation_end_to_end_images_per_second=len(validation_data) / validation_seconds,
                      model_only_inference=latency)
        plot_history(output, history)
        (output / "README_baseline_results.md").write_text(make_report_markdown(report), encoding="utf-8")
    except BaseException as exc:
        report["status"] = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
        report["baseline_training_completed"] = False
        report["errors"].append({"type": type(exc).__name__,
                                 "message": str(exc) if isinstance(exc, DataError) else "See private traceback locally."})
        (output / "PRIVATE_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        report["total_run_seconds"] = time.perf_counter() - started
        write_json(output / "baseline_summary.json", report)
    print("BASELINE_TRAINING_COMPLETED", flush=True)
    print("VALIDATION_RESULTS_ONLY_FINAL_TEST_NOT_EVALUATED", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "splits-dir", "pipeline-report", "smoke-report", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = TrainConfig(batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
                         seed=args.seed, num_workers=args.num_workers, device=args.device)
    try:
        run_training(args.data_root, args.splits_dir, args.pipeline_report, args.smoke_report,
                     args.output_dir, config)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"BASELINE_TRAINING_STOPPED: {type(exc).__name__}. Preserve the run folder and read its summary.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    # Step 7's functions/CLI remain available unchanged. Step 8 uses the shared
    # regularized trainer for both architectures through this assignment entry.
    if "--step8" in sys.argv:
        sys.argv.remove("--step8")
        from train_experiment import main as experiment_main
        raise SystemExit(experiment_main())
    raise SystemExit(main())
