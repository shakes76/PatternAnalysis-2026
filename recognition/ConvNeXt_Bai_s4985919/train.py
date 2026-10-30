"""COMP3710 ADNI training: verified CNN/ConvNeXt, train-only augmentation, resumable runs.

The dataset module must provide load_datasets() -> (train, validation, test),
with disjoint patient IDs and ImageFolder class_to_idx == {'AD': 0, 'NC': 1}.

Examples:
    python train.py --model convnext --epochs 3 --batch-size 8 --run-name convnext_run01
    python train.py --model convnext --epochs 3 --batch-size 8 \
        --resume checkpoints/convnext/convnext_run01/last.pth

"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import signal
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import (load_datasets, load_patient_map, get_transform,
                     MRIAugmentedTrainingSet, SPLIT_MODES, INPUT_MODES)
from mri_intensity import intensity_protocol, make_training_input
from predict import (create_records, evaluate, get_sample_info,
                     patient_mapping_fingerprint)
from baseline import build_baseline, MODEL_NAMES
from modules import ConvNeXt

# Signal handlers ask the training loop to save after the current optimizer step.
_STOP_REQUESTED = False


def request_stop(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"\nSignal {signum} received; saving after the current batch.", flush=True)


def calculate_metrics(labels, probabilities_ad, threshold=0.5):
    """Per-slice binary metrics; AD=0 is the positive class for AUROC."""
    if len(labels) != len(probabilities_ad) or not labels:
        raise ValueError("Metrics require equally sized, non-empty labels/scores")
    if any(y not in (0, 1) for y in labels):
        raise ValueError("Expected AD=0 and NC=1 labels")

    # Ties go to class 0, matching torch.argmax on [p_ad, 1-p_ad].
    predictions = [0 if p >= threshold else 1 for p in probabilities_ad]

    tp = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))  # AD -> AD
    fn = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))  # AD -> NC
    fp = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))  # NC -> AD
    tn = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))  # NC -> NC

    def safe_div(a, b):
        return a / b if b else 0.0

    p_ad = safe_div(tp, tp + fp)
    r_ad = safe_div(tp, tp + fn)
    p_nc = safe_div(tn, tn + fn)
    r_nc = safe_div(tn, tn + fp)
    f1_ad = safe_div(2 * p_ad * r_ad, p_ad + r_ad)
    f1_nc = safe_div(2 * p_nc * r_nc, p_nc + r_nc)

    # Mann-Whitney rank formulation; ties receive average rank.
    positive_count = tp + fn
    negative_count = tn + fp
    auroc_ad = None
    if positive_count and negative_count:
        ranked = sorted(zip(probabilities_ad, labels))
        rank_sum = 0.0
        i = 0
        while i < len(ranked):
            j = i + 1
            while j < len(ranked) and ranked[j][0] == ranked[i][0]:
                j += 1
            average_rank = ((i + 1) + j) / 2.0
            rank_sum += average_rank * sum(y == 0 for _, y in ranked[i:j])
            i = j
        auroc_ad = (
            rank_sum - positive_count * (positive_count + 1) / 2
        ) / (positive_count * negative_count)

    return {
        "accuracy": safe_div(tp + tn, len(labels)),
        "precision_ad": p_ad,
        "recall_ad": r_ad,
        "f1_ad": f1_ad,
        "precision_nc": p_nc,
        "recall_nc": r_nc,
        "f1_nc": f1_nc,
        "macro_precision": (p_ad + p_nc) / 2,
        "macro_recall": (r_ad + r_nc) / 2,
        "macro_f1": (f1_ad + f1_nc) / 2,
        "auroc_ad": auroc_ad,
        "tp_ad": tp,
        "fn_ad": fn,
        "fp_ad": fp,
        "tn_ad": tn,
    }


def save_checkpoint_atomic(path, payload):
    """Avoid replacing a good checkpoint with a half-written file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_json_atomic(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    os.replace(tmp, path)



def plot_training_curves(history, output_dir):
    """Save repeatable training/validation plots; safe when matplotlib is unavailable."""
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")  # Rangpur jobs have no display
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib unavailable, history.json is still saved; install matplotlib for plots.", flush=True)
        return
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(epochs, [r["train_loss"] for r in history], marker="o", label="Train CE loss")
    axes[0].plot(epochs, [r["val_loss"] for r in history], marker="o", label="Validation CE loss")
    axes[0].set(title="Training and validation loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    for key, label in [("accuracy", "Slice accuracy"), ("macro_f1", "Slice macro-F1"),
                       ("recall_ad", "AD slice recall"), ("auroc_ad", "AD slice AUROC")]:
        axes[1].plot(epochs, [r[key] if r[key] is not None else float("nan") for r in history],
                     marker=".", label=label)
    axes[1].set(title="Validation performance (slice-level)", xlabel="Epoch", ylabel="Metric", ylim=(0, 1))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    out = Path(output_dir) / "training_curves.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("Saved training curves:", out, flush=True)


def fingerprint_split(dataset):
    """Fail resume if dataset membership, order, or labels have changed."""
    if isinstance(dataset, Subset):
        source = dataset.dataset
        indices = dataset.indices
    else:
        source = dataset
        indices = range(len(dataset))
    if not hasattr(source, "samples"):
        raise ValueError("Expected ImageFolder (or a Subset of ImageFolder)")
    digest = hashlib.sha256()
    for index in indices:
        path, label = source.samples[index]
        digest.update(f"{path}\t{label}\n".encode("utf-8"))
    return digest.hexdigest()


def check_class_mapping(train_dataset, val_dataset, test_dataset):
    expected = {"AD": 0, "NC": 1}
    for name, dataset in [
        ("train", train_dataset), ("validation", val_dataset), ("test", test_dataset)
    ]:
        source = dataset.dataset if isinstance(dataset, Subset) else dataset
        actual = getattr(source, "class_to_idx", None)
        if actual != expected:
            raise ValueError(
                f"{name} class mapping is {actual}; expected {expected}. "
                "Do not proceed with mismatched metrics."
            )
    print("Verified class mapping: AD=0, NC=1")


def learning_rate_at(step, total_steps, base_lr, warmup_steps):
    """Linear warmup, then cosine decay to zero; depends only on the step,
    so an interrupted run resumes on exactly the same schedule."""
    if warmup_steps and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def update_ema(ema_model, model, decay, step):
    """Exponential moving average of weights; short warmup avoids early bias."""
    d = min(decay, (1.0 + step) / (10.0 + step))
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(d).add_(p.detach(), alpha=1.0 - d)
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(b)


def make_training_loader(dataset, epoch, seed, batch_size, start_batch, limit,
                         num_workers=0):
    # Every epoch's order is reproducible, even after an interrupted run.
    generator = torch.Generator().manual_seed(seed + epoch)
    permutation = torch.randperm(len(dataset), generator=generator).tolist()
    if limit is not None:
        permutation = permutation[:limit * batch_size]
    total_batches = math.ceil(len(permutation) / batch_size)
    if start_batch > total_batches:
        raise ValueError("Checkpoint batch index exceeds epoch length")

    remaining_indices = permutation[start_batch * batch_size:]
    # Use a separate generator to avoid perturbing global RNG (DropPath).
    loader_generator = torch.Generator().manual_seed(seed + epoch + 1000000)
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=remaining_indices,
        num_workers=num_workers, generator=loader_generator
    )
    return loader, total_batches



# Train-only augmentation lives in dataset.py, as required by the project structure.


def save_progress(output_dir, model, optimizer, args, fingerprint, epoch,
                  next_batch, loss_sum, seen_samples, history, best_score,
                  ema_model=None):
    checkpoint = {
        "version": 2,
        "model_name": args.model,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer_name": args.optimizer,
        "weight_decay": args.weight_decay,
        "training_augmentation": args.augment,
        "intensity_augmentation": intensity_protocol(args.intensity_augment),
        "seed": args.seed,
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
        "split_fingerprint": fingerprint,
        "split_mode": args.split_mode,
        "input_mode": args.input_mode,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "ema_decay": args.ema_decay,
        "label_smoothing": args.label_smoothing,
        "target_epochs": args.epochs,
        "selection_level": args.selection_level,
        "selection_threshold": args.selection_threshold,
        "epoch": epoch,                  # zero-indexed, next epoch/batch to run
        "next_batch": next_batch,
        "train_loss_sum": loss_sum,
        "train_seen_samples": seen_samples,
        "best_score": best_score,
        "history": history,
        "model_state": model.state_dict(),
        "ema_state": ema_model.state_dict() if ema_model is not None else None,
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    save_checkpoint_atomic(output_dir / "last.pth", checkpoint)


@torch.no_grad()
def validate(model, loader, criterion, device, max_batches=None,
             samples=None, patient_map=None, selection_level="slice",
             selection_threshold=0.5, allow_mixed_patients=False):
    """Score every available validation slice and, when complete, scans/patients.

    All checkpoints are selected using the same metric and threshold that
    predict.py uses. Short smoke tests must not report partial-scan scores.
    """
    model.eval()
    total_loss, total_samples = 0.0, 0
    labels_all, probabilities_ad = [], []
    for batch_number, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_number >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        prob_ad = torch.softmax(outputs, dim=1)[:, 0]
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        labels_all.extend(labels.cpu().tolist())
        probabilities_ad.extend(prob_ad.cpu().tolist())
    # Keep the historical 0.50 slice diagnostics for apples-to-apples reporting.
    metrics = calculate_metrics(labels_all, probabilities_ad, 0.5)
    metrics["val_loss"] = total_loss / total_samples
    metrics["val_images"] = total_samples
    if samples is None or patient_map is None:
        if selection_level != "slice":
            raise ValueError("Scan/patient selection requires complete validation data")
        chosen = calculate_metrics(labels_all, probabilities_ad, selection_threshold)
    else:
        if len(samples) != len(probabilities_ad):
            raise RuntimeError("Validation dataset is incomplete; cannot aggregate scans/patients")
        if labels_all != [label for _, label in samples]:
            raise RuntimeError("Validation order/labels changed between DataLoader and ImageFolder")
        records = create_records(samples, probabilities_ad, patient_map)
        if records["excluded_mixed_label_patients"]:
            if not allow_mixed_patients or selection_level == "patient":
                raise ValueError("Mixed-label patients detected; review data before model selection")
            # Official split only: such subjects still count at slice/scan level.
            print(f"WARNING: {records['excluded_mixed_label_patients']} validation subjects have "
                  "mixed AD/NC labels and are left out of patient-level metrics.", flush=True)
        for level in ("slice", "scan", "patient"):
            score = evaluate(records[level], selection_threshold)
            if score is not None:
                metrics[f"{level}_accuracy_at_selection_threshold"] = score["accuracy"]
                metrics[f"{level}_macro_f1_at_selection_threshold"] = score["macro_f1"]
                metrics[f"{level}_recall_ad_at_selection_threshold"] = score["recall_ad"]
        chosen = evaluate(records[selection_level], selection_threshold)
        if chosen is None:
            raise RuntimeError(f"No valid {selection_level} records for model selection")
    metrics["selection_level"] = selection_level
    metrics["selection_threshold"] = selection_threshold
    metrics["selection_accuracy"] = chosen["accuracy"]
    metrics["selection_macro_f1"] = chosen["macro_f1"]
    metrics["selection_recall_ad"] = chosen["recall_ad"]
    metrics["selection_auroc_ad"] = chosen["auroc_ad"]
    return metrics


def main():
    global _STOP_REQUESTED
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-only", action="store_true",
                        help="Call predict.py for a pre-selected checkpoint, without training")
    parser.add_argument("--test-checkpoint", type=Path, default=None,
                        help="Model weights for --test-only (raw best.pth)")
    parser.add_argument("--test-threshold", type=float, default=None,
                        help="Optional threshold explicitly locked on development data")
    parser.add_argument("--test-model", choices=list(MODEL_NAMES), default="convnext",
                        help="Architecture used for --test-only (legacy checkpoints)")
    parser.add_argument("--confirm-final-test", action="store_true",
                        help="Acknowledge official test should not be used for tuning")
    parser.add_argument("--model", choices=list(MODEL_NAMES), default="cnn",
                        help="cnn: small SimpleCNN; cnn_matched: CNN with a ConvNeXt-like parameter count")
    parser.add_argument("--split-mode", choices=list(SPLIT_MODES), default="strict",
                        help="strict: remove subjects shared with official test (default); "
                             "official: use the course train/test folders as provided")
    parser.add_argument("--input-mode", choices=list(INPUT_MODES), default="single",
                        help="single: one slice copied to 3 channels; "
                             "context3: previous/current/next slice of the same scan")
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none",
                        help="cosine: linear warmup then per-step cosine decay")
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="0 disables; e.g. 0.999 validates/saves EMA weights")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3,
                        help="Total target epochs, not additional epochs on resume")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--augment", action="store_true",
                        help="Enable mild geometric/noise augmentation on TRAIN only")
    parser.add_argument("--intensity-augment", action="store_true",
                        help="TRAIN only: fixed scale/Gamma 0.9-1.1 before original augmentation")
    parser.add_argument("--selection-level", choices=["slice", "scan", "patient"],
                        default="slice", help="Level used to choose best.pth on validation")
    parser.add_argument("--selection-threshold", type=float, default=0.5,
                        help="AD cutoff used for model selection; fixed before training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--save-every-batches", type=int, default=50)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to last.pth of this same run")
    args = parser.parse_args()

    if args.test_only:
        # Standalone test driver is predict.py, but train.py also exposes a
        # guarded testing entry point as required by the assignment.
        if not args.confirm_final_test or args.test_checkpoint is None:
            parser.error("--test-only requires --test-checkpoint and --confirm-final-test")
        if args.test_threshold is not None and not 0 < args.test_threshold < 1:
            parser.error("--test-threshold must be between 0 and 1")
        import subprocess
        import sys
        cmd = [sys.executable, str(Path(__file__).with_name("predict.py")),
               "--split", "test", "--checkpoint", str(args.test_checkpoint),
               "--model", args.test_model, "--confirm-final-test"]
        if args.test_threshold is not None:
            cmd += ["--threshold", str(args.test_threshold)]
        subprocess.run(cmd, check=True)
        return

    if not (0 < args.selection_threshold < 1):
        parser.error("--selection-threshold must be between 0 and 1")
    if args.selection_level != "slice" and args.max_val_batches is not None:
        parser.error("Scan/patient selection requires full validation; omit --max-val-batches")
    if args.epochs <= 0 or args.batch_size <= 0 or args.save_every_batches <= 0:
        parser.error("epochs, batch-size, save-every-batches must be positive")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")
    if not 0.0 <= args.ema_decay < 1.0:
        parser.error("ema-decay must be in [0, 1)")
    if not 0.0 <= args.label_smoothing < 0.5:
        parser.error("label-smoothing must be in [0, 0.5)")
    if args.warmup_epochs < 0 or args.num_workers < 0:
        parser.error("warmup-epochs and num-workers cannot be negative")
    if args.weight_decay < 0:
        parser.error("weight-decay cannot be negative")
    if args.intensity_augment and (not args.augment or args.model != "convnext"):
        parser.error("--intensity-augment requires --model convnext --augment")
    if any(v is not None and v <= 0 for v in
           (args.max_train_batches, args.max_val_batches)):
        parser.error("max-train-batches and max-val-batches must be positive")
    if args.run_name and (
        args.run_name in (".", "..") or
        Path(args.run_name).name != args.run_name or
        "/" in args.run_name or "\\" in args.run_name
    ):
        parser.error("run-name must be a single folder name")
    if args.resume and args.run_name:
        parser.error("use --resume alone; the run folder comes from the checkpoint")

    is_smoke_test = (args.max_train_batches is not None or
                     args.max_val_batches is not None)
    if args.model == "convnext" and not is_smoke_test and not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for full ConvNeXt training")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    train_dataset, val_dataset, test_dataset = load_datasets(args.split_mode, args.input_mode)
    transform = get_transform(args.input_mode)
    check_class_mapping(train_dataset, val_dataset, test_dataset)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Testing images:", len(test_dataset))
    fingerprint = {
        "train": fingerprint_split(train_dataset),
        "validation": fingerprint_split(val_dataset),
        "test": fingerprint_split(test_dataset),
    }
    patient_map_all = load_patient_map()
    patient_fingerprint = patient_mapping_fingerprint(
        (train_dataset, val_dataset, test_dataset), patient_map_all)

    val_samples = get_sample_info(val_dataset) if args.max_val_batches is None else None
    val_patient_map = patient_map_all if val_samples is not None else None
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, generator=torch.Generator().manual_seed(args.seed + 2000000)
    )
    model = (ConvNeXt(num_classes=2, in_channels=3) if args.model == "convnext"
             else build_baseline(args.model))
    model = model.to(device)
    ema_model = None
    if args.ema_decay > 0:
        ema_model = copy.deepcopy(model)
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    print("Optimizer:", args.optimizer, "weight decay:", args.weight_decay)
    print("Train-only augmentation:", args.augment)
    print("Train-only intensity protocol:", intensity_protocol(args.intensity_augment))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    val_criterion = nn.CrossEntropyLoss()  # plain CE keeps val loss comparable across runs
    print("Selected model:", args.model)
    print(f"Input mode: {args.input_mode}; scheduler: {args.scheduler} "
          f"(warmup {args.warmup_epochs} epochs); EMA decay: {args.ema_decay}; "
          f"label smoothing: {args.label_smoothing}")
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.resume:
        output_dir = args.resume.resolve().parent
        if not args.resume.is_file() or args.resume.name != "last.pth":
            raise FileNotFoundError("--resume must point to an existing last.pth")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("version") != 2:
            raise ValueError("Unsupported checkpoint format")
        for key, current in {
            "model_name": args.model, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "seed": args.seed,
            "optimizer_name": args.optimizer, "weight_decay": args.weight_decay,
            "training_augmentation": args.augment,
            "intensity_augmentation": intensity_protocol(args.intensity_augment),
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
            "split_fingerprint": fingerprint,
            "split_mode": args.split_mode,
            "input_mode": args.input_mode,
            "scheduler": args.scheduler,
            "warmup_epochs": args.warmup_epochs,
            "ema_decay": args.ema_decay,
            "label_smoothing": args.label_smoothing,
            "selection_level": args.selection_level,
            "selection_threshold": args.selection_threshold,
        }.items():
            previous = checkpoint.get(key, {
                "optimizer_name": "adam",
                "weight_decay": 0.0,
                "training_augmentation": False,
                "selection_level": "slice",
                "selection_threshold": 0.5,
                "split_mode": "strict",
                "input_mode": "single",
                "scheduler": "none",
                "warmup_epochs": 1.0,
                "ema_decay": 0.0,
                "label_smoothing": 0.0,
            }.get(key))
            if previous != current:
                raise ValueError(f"Resume mismatch for {key}: saved={checkpoint.get(key)}, now={current}")
        if args.scheduler == "cosine" and checkpoint.get("target_epochs") != args.epochs:
            raise ValueError("A cosine schedule depends on --epochs; resume with the "
                             f"original value {checkpoint.get('target_epochs')}")
        model.load_state_dict(checkpoint["model_state"])
        if ema_model is not None:
            if checkpoint.get("ema_state") is None:
                raise ValueError("Checkpoint has no EMA weights")
            ema_model.load_state_dict(checkpoint["ema_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"]
        start_batch = checkpoint["next_batch"]
        loss_sum = checkpoint["train_loss_sum"]
        seen_samples = checkpoint["train_seen_samples"]
        history = checkpoint["history"]
        best_score = tuple(checkpoint["best_score"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        if device.type == "cuda" and checkpoint["cuda_rng_states"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        print(f"Resumed run: {output_dir}, epoch {start_epoch + 1}, next batch {start_batch + 1}",
              flush=True)
    else:
        run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_root = "smoke_outputs" if is_smoke_test else "checkpoints"
        output_dir = Path(output_root) / args.model / run_name
        if output_dir.exists():
            raise FileExistsError(f"Run folder already exists: {output_dir}. Choose --resume or a new --run-name")
        output_dir.mkdir(parents=True)
        start_epoch, start_batch, loss_sum, seen_samples = 0, 0, 0.0, 0
        history = []
        best_score = (-1.0, -1.0, float("-inf"))  # macro F1, AD recall, -val loss
        save_json_atomic(output_dir / "config.json", {
            "model": args.model, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "seed": args.seed,
            "optimizer_name": args.optimizer,
            "weight_decay": args.weight_decay,
            "training_augmentation": args.augment,
            "intensity_augmentation": intensity_protocol(args.intensity_augment),
            "target_epochs": args.epochs,
            "train_images": len(train_dataset), "val_images": len(val_dataset),
            "test_images": len(test_dataset),
            "class_to_idx": {"AD": 0, "NC": 1},
            "metric_positive_class": "AD (class 0)",
            "selection_metric": "macro_f1 (tie: recall_ad, then lower val_loss)",
            "selection_level": args.selection_level,
            "selection_threshold": args.selection_threshold,
            "preprocessing_repr": repr(transform),
            "split_mode": args.split_mode,
            "input_mode": args.input_mode,
            "scheduler": args.scheduler,
            "warmup_epochs": args.warmup_epochs,
            "ema_decay": args.ema_decay,
            "label_smoothing": args.label_smoothing,
            "num_workers": args.num_workers,
            "split_fingerprint": fingerprint,
            "smoke_test": is_smoke_test,
        })
        save_progress(output_dir, model, optimizer, args, fingerprint,
                      0, 0, 0.0, 0, history, best_score, ema_model)

    if args.epochs < start_epoch or (args.epochs == start_epoch and start_batch):
        raise ValueError("--epochs must be larger than the saved progress")

    # SIGTERM is sometimes sent shortly before Slurm kills a job; periodic
    # checkpoints also protect against abrupt kills with no warning.
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)

    print("Outputs:", output_dir, flush=True)
    print("Split mode:", args.split_mode, flush=True)
    print(f"Best model: {args.selection_level} macro-F1 at AD threshold "
          f"{args.selection_threshold:.2f} (tie: AD recall, then validation loss)", flush=True)
    print("Checkpoint every", args.save_every_batches, "training batches", flush=True)

    # Keep the original dataset unchanged for patient-level split auditing,
    # fingerprinting, and validation. Augmentation wraps only training images.
    training_input = make_training_input(train_dataset, args.augment,
                                         args.intensity_augment, MRIAugmentedTrainingSet)
    samples_per_epoch = len(training_input)
    if args.max_train_batches is not None:
        samples_per_epoch = min(samples_per_epoch, args.max_train_batches * args.batch_size)
    steps_per_epoch = math.ceil(samples_per_epoch / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(round(args.warmup_epochs * steps_per_epoch))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        batch_offset = start_batch if epoch == start_epoch else 0
        if epoch != start_epoch:
            loss_sum, seen_samples = 0.0, 0
        loader, total_batches = make_training_loader(
            training_input, epoch, args.seed, args.batch_size,
            batch_offset, args.max_train_batches, args.num_workers
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for local_index, (images, labels) in enumerate(loader):
            next_batch = batch_offset + local_index + 1
            global_step = epoch * steps_per_epoch + next_batch - 1
            if args.scheduler == "cosine":
                current_lr = learning_rate_at(global_step, total_steps,
                                              args.learning_rate, warmup_steps)
                for group in optimizer.param_groups:
                    group["lr"] = current_lr
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if ema_model is not None:
                update_ema(ema_model, model, args.ema_decay, global_step)

            n = labels.size(0)
            loss_sum += loss.item() * n
            seen_samples += n

            if (next_batch % args.save_every_batches == 0 or
                    next_batch == total_batches or _STOP_REQUESTED):
                save_progress(output_dir, model, optimizer, args, fingerprint,
                              epoch, next_batch, loss_sum, seen_samples,
                              history, best_score, ema_model)
                print(f"Epoch {epoch + 1}/{args.epochs}: "
                      f"batch {next_batch}/{total_batches}, "
                      f"train loss {loss_sum / seen_samples:.4f}; checkpoint saved",
                      flush=True)
            if _STOP_REQUESTED:
                print(f"Stopped safely. Resume using --resume {output_dir / 'last.pth'}",
                      flush=True)
                return

        # Also handles a saved checkpoint at the end of training, before validation.
        if seen_samples == 0:
            raise ValueError("Empty training epoch")
        train_loss = loss_sum / seen_samples
        eval_model = ema_model if ema_model is not None else model
        metrics = validate(eval_model, val_loader, val_criterion, device, args.max_val_batches,
                           samples=val_samples, patient_map=val_patient_map,
                           selection_level=args.selection_level,
                           selection_threshold=args.selection_threshold,
                           allow_mixed_patients=(args.split_mode == "official"))
        candidate = (metrics["selection_macro_f1"],
                     metrics["selection_recall_ad"], -metrics["val_loss"])
        if candidate > best_score:
            best_score = candidate
            save_checkpoint_atomic(output_dir / "best.pth", eval_model.state_dict())
            save_json_atomic(output_dir / "best_meta.json", {
                "model": args.model,
                "checkpoint_format": "raw_state_dict_v1",
                "best_epoch": epoch + 1,
                "selection_level": args.selection_level,
                "selection_threshold": args.selection_threshold,
                "selection_macro_f1": metrics["selection_macro_f1"],
                "selection_recall_ad": metrics["selection_recall_ad"],
                "selection_accuracy": metrics["selection_accuracy"],
                "split_mode": args.split_mode,
                "input_mode": args.input_mode,
                "intensity_augmentation": intensity_protocol(args.intensity_augment),
                "weights": "ema" if ema_model is not None else "raw",
                "scheduler": args.scheduler,
                "label_smoothing": args.label_smoothing,
                "split_fingerprint": fingerprint,
                "patient_mapping_fingerprint": patient_fingerprint,
                "preprocessing_repr": repr(transform),
                "class_to_idx": {"AD": 0, "NC": 1},
                "smoke_test": is_smoke_test,
            })
            print("Best model updated using validation "
                  f"{args.selection_level} macro-F1 at {args.selection_threshold:.2f}.", flush=True)

        record = {"epoch": epoch + 1, "train_loss": train_loss,
                  "learning_rate": optimizer.param_groups[0]["lr"], **metrics}
        if device.type == "cuda":
            record["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        history.append(record)
        save_json_atomic(output_dir / "history.json", history)
        plot_training_curves(history, output_dir)

        # The next checkpoint starts at the beginning of the following epoch.
        # The saved state includes Adam's momentum/variance estimates and RNG.
        save_progress(output_dir, model, optimizer, args, fingerprint,
                      epoch + 1, 0, 0.0, 0, history, best_score, ema_model)
        print(f"\nEpoch {epoch + 1}/{args.epochs} completed: "
              f"loss={train_loss:.4f}, val_acc={metrics['accuracy']:.4f}, "
              f"macro_F1={metrics['macro_f1']:.4f}, "
              f"AD_recall={metrics['recall_ad']:.4f}, "
              f"AUROC_AD={metrics['auroc_ad']}, "
              f"selection_{args.selection_level}_macro_F1={metrics['selection_macro_f1']:.4f}",
              flush=True)

    print("\nTraining finished. Results saved to:", output_dir, flush=True)
    if is_smoke_test:
        print("DEVELOPMENT SMOKE TEST - not a final result.", flush=True)


if __name__ == "__main__":
    main()
