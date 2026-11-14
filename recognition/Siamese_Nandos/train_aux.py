"""Train the auxiliary-head Siamese model using the notebook's loss and schedule."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from dataset import ImageCache, ISICImageDataset, PairDataset, index_images, make_transform
from modules import SiameseNetWithClassifier, contrastive_loss
from split_utils import load_splits
from runtime import select_device
from training_utils import (prepare_output, atomic_save, atomic_csv, capture_rng,
                            restore_rng, manifest_hashes, validate_resume, Progress, epoch_complete)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", required=True, help="Directory containing train.csv and val.csv")
    parser.add_argument("--images", required=True, help="Directory containing extracted JPEGs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64, help="Pairs per batch")
    parser.add_argument("--pairs-per-epoch", type=int, default=24000)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--duplicates", help="Duplicate-pair CSV; defaults to the parent of --splits")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-loss-weight", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-images", action="store_true", help="Preload resized images into RAM")
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=25, help="Print progress every N batches")
    args = parser.parse_args()
    if args.log_every < 1:
        raise ValueError("log-every must be positive")

    if min(args.batch_size, args.pairs_per_epoch, args.image_size, args.epochs, args.embedding_dim) < 1 or args.workers < 0:
        raise ValueError("Batch, pair, image, epoch and embedding sizes must be positive; workers cannot be negative.")
    if args.pairs_per_epoch % args.batch_size:
        raise ValueError("pairs-per-epoch must be divisible by batch-size.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = select_device(args.device)
    splits, output = Path(args.splits), Path(args.output)
    prepare_output(output, args.resume)
    hashes = manifest_hashes(splits)
    frames = load_splits(splits, args.duplicates)
    train_df, val_df = frames["train"], frames["val"]
    paths = index_images(args.images)
    required = {Path(str(name).strip()).stem
                for frame in (train_df, val_df) for name in frame.image_name}
    missing = required - paths.keys()
    if missing:
        raise FileNotFoundError(f"{len(missing)} split images missing; examples: {list(missing)[:5]}")

    image_cache = None
    if args.cache_images:
        image_cache = ImageCache(list(train_df.image_name) + list(val_df.image_name), paths, args.image_size)
        print(f"Cached {len(image_cache):,} images ({image_cache.memory_gb:.2f} GB).")

    pair_ds = PairDataset(
        train_df, paths, args.pairs_per_epoch, args.seed,
        transform=make_transform(
            args.image_size, train=True, resize=image_cache is None
        ),
        image_cache=image_cache,
    )
    train_loader = DataLoader(
        pair_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    val_ds = ISICImageDataset(
        val_df,
        paths,
        make_transform(args.image_size, resize=image_cache is None),
        image_cache=image_cache,
    )
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )

    model = SiameseNetWithClassifier(args.embedding_dim, pretrained=not args.no_pretrained and not args.resume).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda"
    # Prefer BF16 where available; FP16 uses a scaler to protect small gradients.
    # On CPU, autocast and scaling are disabled.
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    best_ap, start_epoch, history = -1.0, 0, []
    last_path, best_path = output / "last.pt", output / "best.pt"
    if args.resume:
        # Resume the optimiser and scaler as well as the weights so the next
        # update continues from the saved training state.
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        validate_resume(checkpoint, args, hashes)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_ap, history = checkpoint["best_ap"], checkpoint["history"]
        restore_rng(checkpoint["rng"])
        print(f"Resuming after epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        pair_ds.set_epoch(epoch)
        model.train()
        totals = np.zeros(3, dtype=np.float64)
        started = time.time()
        progress = Progress("train", epoch + 1, args.epochs, len(train_loader), args.log_every)
        for batch, (a, b, same, y_a, y_b) in enumerate(train_loader, 1):
            a, b = a.to(device), b.to(device)
            same, y_a, y_b = same.to(device), y_a.to(device), y_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                z_a, z_b, logit_a, logit_b = model(a, b)
                # The metric term shapes the embedding space; the BCE term
                # trains the auxiliary head to classify each endpoint directly.
                metric_loss = contrastive_loss(z_a, z_b, same, args.margin)
                class_loss = 0.5 * (criterion(logit_a, y_a) + criterion(logit_b, y_b))
                loss = metric_loss + args.class_loss_weight * class_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            progress.update(batch, loss.item())
            totals += np.array([loss.item(), metric_loss.item(), class_loss.item()]) * len(same)

        model.eval()
        logits, labels, names = [], [], []
        progress = Progress("validation", epoch + 1, args.epochs, len(val_loader), args.log_every)
        with torch.inference_mode():
            for batch, (x, y, image_names) in enumerate(val_loader, 1):
                x = x.to(device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    score = model.classify(x).float()
                logits.extend(score.cpu().numpy().tolist())
                labels.extend(y.numpy().tolist())
                names.extend(image_names)
                progress.update(batch)
        # AP is useful for this highly imbalanced validation set; AUROC is also
        # recorded to describe ranking performance across class thresholds.
        ap = float(average_precision_score(labels, logits))
        auc = float(roc_auc_score(labels, logits))
        row = {
            "epoch": epoch + 1,
            "train_total_loss": totals[0] / len(pair_ds),
            "train_contrastive_loss": totals[1] / len(pair_ds),
            "train_classification_loss": totals[2] / len(pair_ds),
            "val_ap": ap,
            "val_auroc": auc,
            "seconds": time.time() - started,
        }
        history.append(row)
        probabilities = 1 / (1 + np.exp(-np.clip(np.asarray(logits), -40, 40)))
        predictions = pd.DataFrame({
            "image_name": names, "target": labels, "score_logit": logits,
            "score_probability": probabilities,
        })
        print("Saving validation results and checkpoints...", flush=True)
        # Keep the weights and predictions from the same best-AP epoch.
        if ap > best_ap:
            best_ap = ap
            atomic_save({
                "model": model.state_dict(), "epoch": epoch, "val_ap": ap,
                "config": vars(args),
            }, best_path)
            atomic_csv(predictions, output / "best_val_scores.csv")
        print("Saving epoch recovery checkpoint...", flush=True)
        atomic_save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch, "best_ap": best_ap,
            "history": history, "config": vars(args),
            "rng": capture_rng(), "manifest_hashes": hashes,
        }, last_path)
        atomic_csv(pd.DataFrame(history), output / "history.csv")
        epoch_complete(row, history, args.epochs)
    print(f"Best validation AP: {best_ap:.6f}; checkpoint: {best_path}")


if __name__ == "__main__":
    main()
