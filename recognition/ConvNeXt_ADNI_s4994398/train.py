"""
Train, validate, test and save a ConvNeXt (or the baseline CNN) on ADNI AD vs NC.

Examples:
    python train.py --arch convnext --epochs 30
    python train.py --arch baseline --epochs 30 --lr 1e-3

Outputs:
    checkpoints/<arch>_best.pt      best weights by validation accuracy (+ normalisation stats)
    checkpoints/<arch>_logits.npz   val/test logits of the best model (used by evaluate.py)
    results/<arch>_history.json     per-epoch losses and metrics
    results/<arch>_curves.png       training curves
    results/<arch>_test_metrics.json
"""

import argparse
import json
import math
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import (TensorBatcher, compute_stats, gpu_augment, list_split, load_split,
                     preprocess, subject_split, DEFAULT_ROOT)
from modules import build_model
from utils import classification_report, fit_temperature, softmax_np, subject_level

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=["convnext", "baseline"], default="convnext")
    ap.add_argument("--variant", default="tiny", help="ConvNeXt size: tiny/small/base")
    ap.add_argument("--no-pretrained", action="store_true", help="train ConvNeXt from scratch")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--drop-path", type=float, default=0.2)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None, help="name for outputs (defaults to --arch)")
    return ap.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def predict_logits(model, images, labels, mean, std, device, batch_size=128):
    """Deterministic (no augmentation) forward pass over a whole split."""
    model.eval()
    out = []
    for xb, _, _ in TensorBatcher(images, labels, batch_size, False, device):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out.append(model(preprocess(xb, mean, std)).float().cpu())
    return torch.cat(out)


def plot_history(hist, path, title):
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in hist], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in hist], label="val")
    ax[0].set(xlabel="epoch", ylabel="cross-entropy", title=f"{title}: loss")
    ax[1].plot(ep, [h["train_acc"] for h in hist], label="train acc")
    ax[1].plot(ep, [h["val_acc"] for h in hist], label="val acc")
    ax[1].plot(ep, [h["val_auroc"] for h in hist], label="val AUROC", ls="--")
    ax[1].set(xlabel="epoch", ylabel="score", title=f"{title}: metrics", ylim=(0.4, 1.0))
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    tag = args.tag or args.arch
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir, res_dir = os.path.join(HERE, "checkpoints"), os.path.join(HERE, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # ---- data: subject-level train/val split of the official train set; test untouched ----
    images, labels, paths = load_split("train", args.root)
    tr_idx, va_idx = subject_split(list_split(args.root, "train")[0], labels.tolist(), args.val_frac, args.seed)
    x_tr, y_tr = images[tr_idx], labels[tr_idx]
    x_va, y_va = images[va_idx], labels[va_idx]
    x_te, y_te, te_paths = load_split("test", args.root)
    mean, std = compute_stats(x_tr)
    print(f"train {len(y_tr)} | val {len(y_va)} | test {len(y_te)} | mean {mean:.4f} std {std:.4f}")

    model = build_model(args.arch, args.variant, pretrained=not args.no_pretrained,
                        drop_path_rate=args.drop_path).to(device).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.arch}: {n_params / 1e6:.2f}M parameters")

    # no weight decay on norms, biases and layer-scale parameters
    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)

    loader = TensorBatcher(x_tr, y_tr, args.batch_size, True, device, drop_last=True)
    total_steps = args.epochs * len(loader)
    warmup = int(args.warmup_epochs * len(loader))

    hist, best_acc, step = [], -1.0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for xb, yb, _ in loader:
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total_steps, warmup, args.lr)
            x = gpu_augment(xb, mean, std).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits.float(), yb, label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            tl += loss.item() * len(yb)
            tc += (logits.argmax(1) == yb).sum().item()
            tn += len(yb)

        val_logits = predict_logits(model, x_va, y_va, mean, std, device)
        val_loss = F.cross_entropy(val_logits, y_va).item()
        rep = classification_report(val_logits.softmax(1).numpy(), y_va.numpy())
        hist.append({"epoch": epoch, "train_loss": tl / tn, "train_acc": tc / tn, "val_loss": val_loss,
                     "val_acc": rep["accuracy"], "val_auroc": rep["auroc"], "lr": opt.param_groups[0]["lr"],
                     "time_s": time.time() - t0})
        print(f"ep {epoch:3d} | train loss {tl / tn:.4f} acc {tc / tn:.4f} | val loss {val_loss:.4f} "
              f"acc {rep['accuracy']:.4f} auroc {rep['auroc']:.4f} | {time.time() - t0:.0f}s", flush=True)

        if rep["accuracy"] > best_acc:
            best_acc = rep["accuracy"]
            torch.save({"model": model.state_dict(), "arch": args.arch, "variant": args.variant,
                        "mean": mean, "std": std, "epoch": epoch, "val_acc": best_acc},
                       os.path.join(ckpt_dir, f"{tag}_best.pt"))
        plot_history(hist, os.path.join(res_dir, f"{tag}_curves.png"), tag)
        with open(os.path.join(res_dir, f"{tag}_history.json"), "w") as f:
            json.dump(hist, f, indent=1)

    train_peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
    train_time = time.time() - t0

    # ---- test the best checkpoint (selected on validation only) ----
    ckpt = torch.load(os.path.join(ckpt_dir, f"{tag}_best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    val_logits = predict_logits(model, x_va, y_va, mean, std, device).numpy()
    test_logits = predict_logits(model, x_te, y_te, mean, std, device).numpy()
    np.savez(os.path.join(ckpt_dir, f"{tag}_logits.npz"), val_logits=val_logits, val_labels=y_va.numpy(),
             test_logits=test_logits, test_labels=y_te.numpy(), test_paths=np.array(te_paths))

    T = fit_temperature(val_logits, y_va.numpy())
    test_probs = softmax_np(test_logits)
    report = {
        "arch": args.arch, "tag": tag, "best_epoch": ckpt["epoch"], "best_val_acc": best_acc,
        "params": n_params, "train_time_s": train_time, "train_peak_vram_mb": train_peak_mb,
        "temperature": T, "args": vars(args),
        "test_slice": classification_report(test_probs, y_te.numpy()),
        "test_slice_temp_scaled": classification_report(softmax_np(test_logits, T), y_te.numpy()),
        "test_subject": classification_report(*subject_level(test_probs, y_te.numpy(), te_paths)[:2]),
    }
    with open(os.path.join(res_dir, f"{tag}_test_metrics.json"), "w") as f:
        json.dump(report, f, indent=1)
    s = report["test_slice"]
    print(f"\nTEST ({tag}, best epoch {ckpt['epoch']}): acc {s['accuracy']:.4f} | macro-F1 {s['macro_f1']:.4f} "
          f"| AUROC {s['auroc']:.4f} | ECE {s['ece']:.4f} | subject acc {report['test_subject']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
