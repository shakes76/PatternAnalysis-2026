"""Train a 2D baseline or residual/deep-supervised 3D U-Net.

Validation always reconstructs complete cached volumes. Test data is never
evaluated here. --max-batches marks a development smoke run, not final evidence.
"""

from __future__ import annotations

import argparse
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import HipMRISliceDataset, HipMRIVolumeDataset, load_manifest
from modules import ImprovedUNet3D, UNet2D, segmentation_loss
from utils import evaluate_volumes, file_sha256, plot_training, synchronize, write_json


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                  "position": numpy_state[2], "has_gauss": numpy_state[3], "cached_gaussian": numpy_state[4]},
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["algorithm"], np.asarray(numpy_state["keys"], dtype=np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))


def save_checkpoint(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def resolve_training_class_weights(manifest_path, manifest, method, device):
    """Count training volumes once; validation/test never inform CE weights.

    sqrt_inverse_frequency: inverse square-root frequency, divide by its class
    mean, clamp to [0.05, 10], then divide by its class mean again. The final
    normalization means the clamp is an intermediate stabilizer, not a bound
    on the final weights. Counts include background in class-index order.
    """
    num_classes = int(manifest["num_classes"])
    records = [record for record in manifest["records"] if record["split"] == "train"]
    if not records:
        raise ValueError("The training split is empty.")
    has_counts = ["class_voxel_counts" in record for record in records]
    counts = torch.zeros(num_classes, dtype=torch.int64)
    if all(has_counts):
        for record in records:
            values = record["class_voxel_counts"]
            if (not isinstance(values, list) or len(values) != num_classes
                    or any(type(value) is not int or value < 0 for value in values)):
                raise ValueError("Training record class_voxel_counts must be nonnegative integers in class-index order.")
            if sum(values) != int(np.prod(manifest["target_shape_dhw"])):
                raise ValueError("Training record class_voxel_counts do not sum to the cached volume size.")
            counts += torch.tensor(values, dtype=torch.int64)
        source = "training records' cached class_voxel_counts; no validation/test records used"
    else:
        # Do not use the slice dataset: each cached training volume is read once.
        volumes = HipMRIVolumeDataset(manifest_path, "train", augment=False)
        for index in range(len(volumes)):
            counts += torch.bincount(volumes[index]["mask"].reshape(-1), minlength=num_classes)
        source = "one unaugmented pass over training volume masks; no validation/test volumes loaded"
    missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
    if missing:
        raise ValueError(f"Training split contains no voxels for class indices {missing}; audit the split/labels before training.")
    if method == "none":
        return None, counts.tolist(), None, source
    frequency = counts.to(torch.float64) / counts.sum()
    weights = frequency.rsqrt()
    weights = weights / weights.mean()
    weights = weights.clamp(min=0.05, max=10.0)
    weights = (weights / weights.mean()).to(torch.float32)
    return weights.to(device), counts.tolist(), weights.tolist(), source


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", choices=("unet2d", "improved3d"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30, help="Total epochs, including epochs completed before --resume.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--class-weighting", choices=("none", "sqrt_inverse_frequency"),
                        default="sqrt_inverse_frequency", help="CE weights derived exclusively from training-volume voxel counts.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Number of 2D slices per inference batch.")
    parser.add_argument("--max-batches", type=int, help="Development smoke: cap training batches and validation cases per epoch.")
    parser.add_argument("--resume", type=Path, help="Resume an own checkpoint with identical data/model/training settings.")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.base_channels, args.eval_batch_size) < 1 or args.workers < 0:
        parser.error("Epochs, batch sizes and channels must be positive; workers must be nonnegative.")
    if not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Learning rate must be positive and weight decay nonnegative, both finite.")
    if not np.isfinite(args.clip_grad) or args.clip_grad <= 0:
        parser.error("Gradient clip must be positive and finite.")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive.")
    if not 0 <= args.seed < 2**32:
        parser.error("Seed must be in [0, 2**32).")
    return args


def main(argv=None):
    args = parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    partial = bool(manifest.get("partial") or manifest.get("development_only"))
    development = partial or args.max_batches is not None
    if partial and args.max_batches is None:
        raise ValueError("Partial/development-only manifests require an explicit --max-batches smoke run.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires CUDA; omit it for CPU runs.")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and args.resume.expanduser().resolve().parent != output_dir:
        raise ValueError("Resume into the checkpoint's existing output directory so the previous best.pt remains available.")
    if not args.resume and any((output_dir / name).exists() for name in ("latest.pt", "best.pt", "history.json")):
        raise ValueError("Output already contains a run. Use --resume or choose a new --output-dir.")
    seed_everything(args.seed)
    dataset_class = HipMRISliceDataset if args.model == "unet2d" else HipMRIVolumeDataset
    training = dataset_class(manifest_path, "train", augment=True)
    if not len(training):
        raise ValueError("The training split is empty.")
    class_weights, class_counts, resolved_weights, weight_source = resolve_training_class_weights(
        manifest_path, manifest, args.class_weighting, device)
    validation_records = [record for record in manifest["records"] if record["split"] == "validation"]
    if not validation_records and not development:
        raise ValueError("A complete training run requires a nonempty validation split.")
    validation = HipMRIVolumeDataset(manifest_path, "validation", augment=False) if validation_records else None
    config = {
        "model": args.model, "num_classes": int(manifest["num_classes"]),
        "base_channels": args.base_channels, "seed": args.seed,
        "workers": args.workers, "eval_batch_size": args.eval_batch_size,
        "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
        "clip_grad": args.clip_grad, "amp": args.amp,
        "class_weighting": args.class_weighting,
        "training_class_voxel_counts": class_counts,
        "resolved_class_weights": resolved_weights,
        "max_batches": args.max_batches, "development_only": development,
        "manifest_sha256": file_sha256(manifest_path),
        "audit_sha256": manifest.get("audit_sha256"), "label_ids": manifest["label_ids"],
        "target_shape_dhw": manifest["target_shape_dhw"], "patient_splits": manifest["patient_splits"],
    }
    model_class = UNet2D if args.model == "unet2d" else ImprovedUNet3D
    model = model_class(num_classes=config["num_classes"], base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    start_epoch, best_score, best_epoch, history = 1, None, None, []
    if args.resume:
        checkpoint = torch.load(args.resume.expanduser(), map_location="cpu", weights_only=True)
        if checkpoint.get("config") != config:
            raise ValueError("Resume config differs from the checkpoint (data, labels, model, seed or training settings).")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        history = checkpoint["history"]
        best_score, best_epoch = checkpoint["best_score"], checkpoint["best_epoch"]
        start_epoch = checkpoint["epoch"] + 1
        restore_rng(checkpoint["rng_state"])
        if args.epochs < start_epoch:
            raise ValueError("--epochs must exceed the checkpoint epoch; it is the total desired epoch count.")
    run = {
        "status": "running", "config": config, "requested_total_epochs": args.epochs,
        "manifest": str(manifest_path), "workers": args.workers, "device": str(device),
        "torch_version": str(torch.__version__), "python_version": platform.python_version(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "class_weight_source": weight_source,
        "class_weight_formula": "inverse sqrt(training voxel frequency) -> class-mean normalization -> clamp [0.05,10] -> class-mean normalization; CE only" if args.class_weighting != "none" else "unweighted CE",
        "selection_metric": "Whole-volume foreground Dice: mean classes per case, mean visits per patient, then mean patients; both-empty class excluded.",
        "accuracy_evidence": "development smoke only" if development else "validation only; test remains unevaluated",
        "determinism": "Seeded Python/NumPy/PyTorch and epoch-specific shuffle; cuDNN deterministic; deterministic algorithms warn on unsupported operations.",
    }
    write_json(output_dir / "run.json", run)
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            generator = torch.Generator().manual_seed(args.seed + epoch)
            loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                pin_memory=device.type == "cuda", worker_init_fn=seed_worker, generator=generator)
            model.train()
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            loss_sum, examples, batches = 0.0, 0, 0
            for batch_index, batch in enumerate(loader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["mask"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=args.amp):
                    output = model(images)
                    loss = segmentation_loss(output, targets, class_weights=class_weights)
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError("Nonfinite training loss; checkpoint not advanced.")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                count = int(images.shape[0])
                loss_sum += float(loss.item()) * count
                examples += count
                batches += 1
            synchronize(device)
            train_seconds = time.perf_counter() - start
            train_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            # Free the last training graph/gradients before measuring validation.
            optimizer.zero_grad(set_to_none=True)
            del output, loss, images, targets
            evaluation = evaluate_volumes(model, validation, args.model, config["num_classes"], device,
                                          args.eval_batch_size, args.amp, max_cases=args.max_batches) if validation is not None else None
            score = evaluation["aggregate"]["foreground_mean_dice"] if evaluation is not None else None
            if score is None and not development:
                raise RuntimeError("Validation has no evaluable foreground score; inspect the labels before selecting a checkpoint.")
            improved = score is not None and (best_score is None or score > best_score)
            if improved:
                best_score, best_epoch = score, epoch
            row = {"epoch": epoch, "train_loss": loss_sum / examples, "training_batches": batches,
                   "training_examples": examples, "train_seconds": train_seconds,
                   "peak_allocated_cuda_bytes_training": train_peak,
                   "last_gradient_norm_before_clipping": float(gradient_norm.item()),
                   "validation_foreground_dice": score,
                   "validation_foreground_iou": evaluation["aggregate"]["foreground_mean_iou"] if evaluation else None,
                   "validation": evaluation, "development_only": development}
            history.append(row)
            payload = {"schema_version": 1, "config": config, "state_dict": model.state_dict(),
                       "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                       "epoch": epoch, "best_score": best_score, "best_epoch": best_epoch,
                       "history": history, "rng_state": rng_state()}
            save_checkpoint(output_dir / "latest.pt", payload)
            # A partial manifest may have no validation and therefore no best.
            if improved:
                save_checkpoint(output_dir / "best.pt", payload)
            write_json(output_dir / "history.json", history)
            # Plotting must not change the RNG sequence relative to a resume.
            try:
                plot_training(history, output_dir / "curves.png")
            finally:
                restore_rng(payload["rng_state"])
            run.update({"completed_epochs": epoch, "best_epoch": best_epoch, "best_validation_foreground_dice": best_score})
            write_json(output_dir / "run.json", run)
            score_text = f"{score:.4f}" if score is not None else "unavailable"
            print(f"Epoch {epoch}/{args.epochs}: train loss={row['train_loss']:.5f}; validation patient-balanced Dice={score_text}", flush=True)
        run["status"] = "completed_development_smoke" if development else "completed"
        run["completed_epochs"] = history[-1]["epoch"] if history else start_epoch - 1
    except BaseException as error:
        run.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "error": f"{type(error).__name__}: {error}"})
        write_json(output_dir / "run.json", run)
        raise
    write_json(output_dir / "run.json", run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
