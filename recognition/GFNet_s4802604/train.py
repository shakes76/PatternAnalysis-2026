"""Train, validate and test a model on ADNI AD vs NC.

Usage: python train.py --model baseline [--leaky] [--epochs 15]
Checkpoints and per-scan test predictions are saved to ~/adni_work (not in git).
The loss/accuracy plot is saved next to this file for the README.
"""
import argparse
import json
import math
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dataset import WORK, get_loaders
from modules import BaselineCNN, GFNet


def evaluate(model, loader, device):
    """Slice-level and scan-level metrics. Scan prob = mean of its slice probs."""
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, n, correct = 0.0, 0, 0
    probs, labels = defaultdict(list), {}
    with torch.no_grad():
        for x, y, sid in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += crit(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            n += y.numel()
            p = out.softmax(1)[:, 1].cpu().tolist()
            for s, pi, yi in zip(sid, p, y.cpu().tolist()):
                probs[s].append(pi)
                labels[s] = yi
    scan_p = {s: sum(v) / len(v) for s, v in probs.items()}
    scan_acc = sum((scan_p[s] > 0.5) == bool(labels[s]) for s in scan_p) / len(scan_p)
    return {"loss": loss_sum / n, "slice_acc": correct / n, "scan_acc": scan_acc}, scan_p, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="baseline", choices=["baseline", "gfnet"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--leaky", action="store_true", help="use the provided (leaky) split")
    ap.add_argument("--tag", default="", help="suffix for output file names")
    ap.add_argument("--dim", type=int, default=128, help="GFNet embedding width")
    ap.add_argument("--depth", type=int, default=4, help="GFNet number of blocks")
    ap.add_argument("--drop", type=float, default=0.1, help="GFNet dropout")
    ap.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay")
    ap.add_argument("--patch", type=int, default=16, help="GFNet patch size")
    ap.add_argument("--smooth", type=float, default=0.0, help="label smoothing")
    ap.add_argument("--select", default="loss", choices=["loss", "scan_acc"], help="checkpoint rule")
    ap.add_argument("--strong-aug", action="store_true", help="stronger training augmentation")
    ap.add_argument("--drop-path", type=float, default=0.0, help="GFNet stochastic depth rate")
    ap.add_argument("--conv-stem", action="store_true", help="GFNet with a convolutional stem")
    ap.add_argument("--warmup", type=int, default=0, help="linear LR warmup epochs")
    ap.add_argument("--cosine", action="store_true", help="cosine LR decay after warmup")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--mixup", type=float, default=0.0, help="mixup Beta alpha (0 = off)")
    args = ap.parse_args()

    name = f"{args.model}_{'leaky' if args.leaky else 'clean'}{args.tag}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    train_dl, val_dl, test_dl = get_loaders(args.batch_size, leaky=args.leaky, strong_aug=args.strong_aug)
    model = (GFNet(dim=args.dim, depth=args.depth, drop=args.drop, patch=args.patch, drop_path=args.drop_path, conv_stem=args.conv_stem) if args.model == "gfnet" else BaselineCNN()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = nn.CrossEntropyLoss(label_smoothing=args.smooth)

    def lr_at(ep):  # ep counts from 0
        if args.warmup and ep < args.warmup:
            return args.lr * (ep + 1) / args.warmup
        if args.cosine:
            frac = (ep - args.warmup) / max(1, args.epochs - args.warmup)
            return args.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * frac)))
        return args.lr

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {n_params:,} parameters, device={device}")

    hist = {"train_loss": [], "val_loss": [], "val_slice_acc": [], "val_scan_acc": []}
    best_score, ckpt = float("-inf"), f"{WORK}/{name}.pth"
    for epoch in range(1, args.epochs + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(epoch - 1)
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for x, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if args.mixup > 0:
                lam = float(torch.distributions.Beta(args.mixup, args.mixup).sample())
                idx = torch.randperm(x.size(0), device=device)
                out = model(lam * x + (1 - lam) * x[idx])
                loss = lam * crit(out, y) + (1 - lam) * crit(out, y[idx])
            else:
                loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.numel()
            seen += y.numel()
        val, _, _ = evaluate(model, val_dl, device)
        hist["train_loss"].append(run_loss / seen)
        hist["val_loss"].append(val["loss"])
        hist["val_slice_acc"].append(val["slice_acc"])
        hist["val_scan_acc"].append(val["scan_acc"])
        score = -val["loss"] if args.select == "loss" else val["scan_acc"]
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), ckpt)
        print(f"epoch {epoch:02d} train_loss {run_loss / seen:.4f} val_loss {val['loss']:.4f} "
              f"val_slice_acc {val['slice_acc']:.3f} val_scan_acc {val['scan_acc']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # Test once, with the best-validation checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=device))
    test, scan_p, labels = evaluate(model, test_dl, device)
    print("TEST", json.dumps(test))
    json.dump({"metrics": test, "scan_prob": scan_p, "scan_label": labels},
              open(f"{WORK}/{name}_test_preds.json", "w"))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[1].plot(hist["val_slice_acc"], label="val slice acc")
    ax[1].plot(hist["val_scan_acc"], label="val scan acc")
    ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.suptitle(name); fig.tight_layout()
    fig.savefig(f"curves_{name}.png", dpi=120)


if __name__ == "__main__":
    main()
