"""Train the ResNet-18 baseline and save evaluator-compatible checkpoints."""

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

from dataset import ImageCache, ISICImageDataset, index_images, make_transform
from modules import ResNet18Baseline
from split_utils import load_splits
from runtime import select_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--duplicates", help="Duplicate-pair CSV; defaults to the parent of --splits")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--cache-images",
        action="store_true",
        help="Preload resized train and validation images into host RAM (about 4.2 GB at 224 px).",
    )
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--pos-weight", type=float, default=20.0)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    if min(args.batch_size, args.image_size, args.epochs) < 1 or args.workers < 0:
        raise ValueError("Batch size, image size and epochs must be positive; workers cannot be negative.")

    # Fix the random ordering and initialisation so a run can be reproduced
    # from the same split files and settings.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = select_device(args.device)
    splits, output = Path(args.splits), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frames = load_splits(splits, args.duplicates)
    train_df, val_df = frames["train"], frames["val"]
    paths = index_images(args.images)
    required = {Path(str(name).strip()).stem
                for frame in (train_df, val_df) for name in frame.image_name}
    missing = required - paths.keys()
    if missing:
        raise FileNotFoundError(f"{len(missing)} split images missing; examples: {list(missing)[:5]}")

    # One shared cache avoids decoding and resizing the same JPEGs every epoch.
    # It contains train and validation images only; random augmentation remains
    # active for training because pixels are cached before the transform runs.
    image_cache = None
    if args.cache_images:
        image_names = list(train_df.image_name) + list(val_df.image_name)
        image_cache = ImageCache(image_names, paths, args.image_size)
        print(
            f"Cached {len(image_cache):,} train/validation images "
            f"({image_cache.memory_gb:.2f} GB in RGB uint8 pixels)."
        )

    # Augmentation is only used for training. The validation transform is fixed
    # so that a change in its score reflects the model, not a random crop/flip.
    # Cached images were resized during preload, so skip Resize in their
    # transforms; without caching, the ordinary on-demand path resizes each JPEG.
    train_ds = ISICImageDataset(
        train_df,
        paths,
        make_transform(args.image_size, train=True, resize=image_cache is None),
        image_cache=image_cache,
    )
    val_ds = ISICImageDataset(
        val_df,
        paths,
        make_transform(args.image_size, resize=image_cache is None),
        image_cache=image_cache,
    )
    loader_args = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, **loader_args)
    model = ResNet18Baseline(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    # Supply raw logits to BCEWithLogitsLoss; it combines sigmoid and BCE in a
    # numerically stable operation. pos_weight can increase the melanoma loss.
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.pos_weight], dtype=torch.float32, device=device)
    )
    best_ap, history = -1.0, []
    best_path = output / "best.pt"

    for epoch in range(args.epochs):
        model.train()
        loss_sum, seen, started = 0.0, 0, time.time()
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            seen += len(y)

        model.eval()
        logits_all, labels, names = [], [], []
        with torch.inference_mode():
            for x, y, image_names in val_loader:
                logits = model(x.to(device, non_blocking=True)).float()
                logits_all.extend(logits.cpu().numpy().tolist())
                labels.extend(y.numpy().tolist())
                names.extend(image_names)
        # Average precision is the selection metric. Use logits directly for
        # ranking metrics; sigmoid probabilities are saved for later reporting.
        ap = float(average_precision_score(labels, logits_all))
        auc = float(roc_auc_score(labels, logits_all))
        row = {"epoch": epoch + 1, "train_loss": loss_sum / max(seen, 1),
               "val_ap": ap, "val_auroc": auc, "seconds": time.time() - started}
        history.append(row)
        probabilities = 1 / (1 + np.exp(-np.clip(np.asarray(logits_all), -40, 40)))
        predictions = pd.DataFrame({
            "image_name": names, "target": labels,
            "logit": logits_all, "probability": probabilities,
        })
        # Preserve the checkpoint format expected by the baseline evaluator.
        if ap > best_ap:
            best_ap = ap
            torch.save({"architecture": "resnet18", "state_dict": model.state_dict(),
                        "epoch": epoch + 1, "val_ap": ap, "config": vars(args)}, best_path)
            predictions.to_csv(output / "best_val_predictions.csv", index=False)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(row)
    print(f"Best validation AP: {best_ap:.6f}; checkpoint: {best_path}")


if __name__ == "__main__":
    main()
