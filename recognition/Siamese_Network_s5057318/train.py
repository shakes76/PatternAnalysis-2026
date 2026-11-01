"""Train the baseline classifier or the Siamese network on the ISIC melanoma data.

The checkpoint is chosen by validation loss only; the test split is evaluated
once at the end, with the best checkpoint, and never used to pick anything.

Besides the loss, every epoch also records a metric on the train and validation
splits (baseline: AUROC and PR-AUC of the melanoma score; Siamese: AUROC and
accuracy of the pair distance at telling similar from dissimilar pairs). They are
printed, and plotted in `<model>_metrics.png` next to `<model>_loss.png`. They are
read from the same forward passes that give the loss, so they do not change the
training, the random numbers or the choice of best checkpoint.
"""
import argparse
import os
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import dataset as d
from modules import BaselineCNN, ContrastiveLoss, SiameseEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
DEBUG_TRAIN, DEBUG_VAL, DEBUG_EPOCHS = 256, 128, 2


def set_seed(seed):
    """Seed Python, NumPy and torch (CPU and CUDA) and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def subsample(splits, seed):
    """Debug mode: small random subsets of each split (still within that split)."""
    sizes = {"train": DEBUG_TRAIN, "val": DEBUG_VAL, "test": DEBUG_VAL}
    out = {name: df.sample(n=min(sizes[name], len(df)), random_state=seed).reset_index(drop=True)
           for name, df in splits.items()}
    for name, df in out.items():
        if (df["target"] == 1).sum() < 2:
            raise RuntimeError(f"debug {name} subset has fewer than 2 malignant images")
    return out


METRIC_NAMES = {"baseline": ("auroc", "pr_auc"), "siamese": ("pair_auroc", "pair_acc")}


def auroc(scores, labels):
    """Area under the ROC curve (rank formula, tied scores share their average rank)."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels) == 1
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    _, inverse, counts = np.unique(scores[order], return_inverse=True, return_counts=True)
    average_rank = np.cumsum(counts) - (counts - 1) / 2  # 1-based, ties averaged
    ranks = np.empty(len(scores))
    ranks[order] = average_rank[inverse]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores, labels):
    """Average precision (PR-AUC): mean of the precision at each positive, going down the ranking."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels) == 1
    if labels.sum() == 0:
        return float("nan")
    hits = labels[np.argsort(-scores, kind="mergesort")]
    precision_at_hit = np.cumsum(hits)[hits] / (np.flatnonzero(hits) + 1)
    return float(precision_at_hit.mean())


def epoch_metrics(model_name, outputs, margin):
    """Metrics from the (score, label) pairs collected during one pass over a split.

    Baseline: score = logit, label = 1 for melanoma. Siamese: score = distance between
    the two embeddings, label = 1 for a dissimilar pair; a pair is called dissimilar
    when its distance is above margin / 2.
    """
    scores = torch.cat([s for s, _ in outputs]).numpy()
    labels = torch.cat([y for _, y in outputs]).numpy()
    if model_name == "baseline":
        return auroc(scores, labels), average_precision(scores, labels)
    return auroc(scores, labels), float(((scores > margin / 2) == (labels == 1)).mean())


def make_step(model_name, device, args, splits):
    """Return (net, step) where step(batch) -> (loss, batch_size).

    step also leaves `step.last = (scores, labels)` for the batch it just ran (on the CPU,
    detached), which run_epoch and evaluate collect for the per-epoch metrics.
    """
    if model_name == "baseline":
        net = BaselineCNN().to(device)
        train_targets = splits["train"]["target"]  # pos_weight from the TRAIN split only
        pos_weight = (train_targets == 0).sum() / (train_targets == 1).sum()
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([float(pos_weight)], device=device))

        def step(batch):
            """Run one baseline batch: return the weighted BCE loss and batch size, and store logits and labels in step.last."""
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)  # float, same shape as logits [N, 1]
            logits = net(images)
            step.last = (logits.detach().squeeze(1).cpu(), labels.detach().squeeze(1).cpu())
            return criterion(logits, labels), len(images)
    else:
        net = SiameseEncoder().to(device)
        criterion = ContrastiveLoss(margin=args.margin)

        def step(batch):
            """Run one pair batch through the shared encoder: return the contrastive loss and batch size, and store pair distances and labels in step.last."""
            img1, img2, labels = batch  # label 0 = similar, 1 = dissimilar
            n = len(img1)
            emb = net(torch.cat([img1, img2]).to(device))  # same encoder for both images
            dist = torch.linalg.vector_norm(emb[:n].detach() - emb[n:].detach(), dim=1)
            step.last = (dist.cpu(), labels.detach().cpu())
            return criterion(emb[:n], emb[n:], labels.to(device)), n
    return net, step


def run_epoch(net, step, loader, optimizer, outputs=None):
    """One training pass over `loader`. Returns mean loss per sample.

    If `outputs` is a list, the (scores, labels) of every batch are appended to it.
    """
    net.train()
    total, count = 0.0, 0
    for batch in loader:
        loss, n = step(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * n
        count += n
        if outputs is not None:
            outputs.append(step.last)
    return total / count


def evaluate(net, step, loader, outputs=None):
    """Mean loss per sample with no gradients and the model in eval mode.

    If `outputs` is a list, the (scores, labels) of every batch are appended to it.
    """
    net.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            loss, n = step(batch)
            total += loss.item() * n
            count += n
            if outputs is not None:
                outputs.append(step.last)
    return total / count


def main():
    """Train with a validation pass every epoch, keep the lowest-val-loss checkpoint, test it once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="folder with cleaned_metadata.csv and combined_images/")
    parser.add_argument("--model", choices=["baseline", "siamese"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=None,
                        help="seed of the train/val/test split and the val/test pairs "
                             "(default: same as --seed)")
    parser.add_argument("--augment", choices=["basic", "strong"], default="basic",
                        help="train-image augmentation (val/test are never augmented)")
    parser.add_argument("--preprocess", choices=["none", "grayworld"], default="none",
                        help="colour-constancy step applied to every image of every split")
    parser.add_argument("--pair_source", choices=["any", "same"], default="any",
                        help="same: both images of every TRAIN pair come from one source (IP_ or HAM_)")
    parser.add_argument("--no_cudnn", action="store_true", help="disable cuDNN")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", default="outputs",
                        help="relative paths are taken from this script's folder")
    parser.add_argument("--debug", action="store_true",
                        help=f"{DEBUG_TRAIN} train / {DEBUG_VAL} val images, {DEBUG_EPOCHS} epochs")
    args = parser.parse_args()
    if args.debug:
        args.epochs = DEBUG_EPOCHS
    if args.split_seed is None:
        args.split_seed = args.seed
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(HERE, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    set_seed(args.seed)
    if args.no_cudnn:
        torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"device: {device} | model: {args.model} | epochs: {args.epochs} | seed: {args.seed}"
          f"{' | DEBUG' if args.debug else ''}")

    image_dir = os.path.join(args.data_dir, "combined_images")
    splits = d.split_by_id(d.load_clean_metadata(args.data_dir), seed=args.split_seed)
    d.check_no_id_overlap(splits)
    if args.debug:
        splits = subsample(splits, args.seed)
    if args.model == "baseline":
        train_loader, val_loader, test_loader = d.get_dataloaders(
            splits, image_dir, args.batch_size, args.num_workers, seed=args.seed,
            augment=args.augment, preprocess=args.preprocess)
    else:
        train_loader, val_loader, test_loader = d.get_pair_dataloaders(
            splits, image_dir, args.batch_size, args.num_workers, seed=args.seed,
            split_seed=args.split_seed, augment=args.augment,
            preprocess=args.preprocess, pair_source=args.pair_source)

    net, step = make_step(args.model, device, args, splits)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    best_val, ckpt_path = float("inf"), os.path.join(out_dir, f"{args.model}_best.pt")
    train_losses, val_losses = [], []
    metric_names = METRIC_NAMES[args.model]
    train_metrics, val_metrics = [], []  # one tuple of metric values per epoch
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_out, val_out = [], []
        # Train metrics come from the (augmented) training batches while the weights change,
        # so they are running numbers; the validation metrics use the final weights of the epoch.
        train_loss = run_epoch(net, step, train_loader, optimizer, train_out)
        val_loss = evaluate(net, step, val_loader, val_out)  # val split only
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics.append(epoch_metrics(args.model, train_out, args.margin))
        val_metrics.append(epoch_metrics(args.model, val_out, args.margin))
        saved = ""
        if val_loss < best_val:  # best checkpoint is chosen by validation loss alone
            best_val = val_loss
            torch.save({"model": net.state_dict(), "epoch": epoch, "val_loss": val_loss,
                        "args": vars(args)}, ckpt_path)
            saved = "  (saved best)"
        shown = " ".join(f"{name} {value:.4f}" for name, value in zip(metric_names, val_metrics[-1]))
        print(f"epoch {epoch:3d} | train loss {train_loss:.6f} | val loss {val_loss:.6f} "
              f"| val {shown} | {time.time() - t0:.1f}s{saved}")
    total_seconds = time.time() - start

    net.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    test_loss = evaluate(net, step, test_loader)  # evaluated once, after training
    print(f"test loss (best checkpoint, epoch {val_losses.index(best_val) + 1}): {test_loss:.6f}")

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, marker="o", label="train")
    plt.plot(epochs, val_losses, marker="o", label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"{args.model} loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{args.model}_loss.png"), dpi=120)
    plt.close()

    fig, axes = plt.subplots(1, len(metric_names), figsize=(10, 4))
    for i, (ax, name) in enumerate(zip(axes, metric_names)):
        ax.plot(epochs, [m[i] for m in train_metrics], marker="o", label="train (running)")
        ax.plot(epochs, [m[i] for m in val_metrics], marker="o", label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel(name)
        ax.set_title(f"{args.model} {name}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{args.model}_metrics.png"), dpi=120)
    plt.close(fig)

    peak = (f"{torch.cuda.max_memory_allocated() / 2**20:.1f} MB" if device.type == "cuda"
            else "n/a (cpu)")
    print(f"parameters: {n_params} | total training time: {total_seconds:.1f}s "
          f"| peak GPU memory: {peak}")


if __name__ == "__main__":
    main()
