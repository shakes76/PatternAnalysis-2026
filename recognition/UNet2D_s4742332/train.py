"""
Containing the source code for training, validating, testing, and saving your model.
The model should be imported from modules.py and the data loader should be imported from
dataset.py. Make sure to plot the losses and metrics during training.
"""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots without a window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from dataset import full_load_dataset, slice_name, N_LABELS
from modules import UNet2D

LABELS = {"background": 0, "body": 1, "femoral hip":2, "bladder":3, "rectum":4, "prostate":5}
REGIONS = ("base", "central", "apex")

def parse_args():
    p = argparse.ArgumentParser(description="Train 2D U-Net on HipMRI prostate slices")
    p.add_argument("--data-root", type=Path, required=True,
                   help="HipMRI keras_slices_data folder (holds the keras_slices_<split> and keras_slices_seg_<split> folders)")
    p.add_argument("--cache-dir", type=Path, required=True,
                   help="folder for the slice cache (~2 GB, built on the first run), keep it outside the repo")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--n-val", type=int, default=5)
    p.add_argument("--n-test", type=int, default=5)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--no-skip", dest="skip_connected", action="store_false",
                   help="baseline model without skip connections")
    p.add_argument("--patient-weighting", choices=["sqrt", "inverse"], default=None)
    p.add_argument("--conv-depths", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    p.add_argument("--out-dir", type=Path, default=Path("runs"))
    p.add_argument("--loss", choices=["ce", "ce_dice"], default="ce",
                   help="ce_dice adds a soft Dice term on the foreground classes so small structures like the apex count")
    p.add_argument("--lr-schedule", choices=["none", "cosine"], default="none",
                   help="cosine decays the lr from --lr to 0 over --epochs")
    p.add_argument("--context", type=int, default=0,
                   help="2.5D: neighbouring slices on each side fed as extra input channels, 1 -> slices k-1, k, k+1")
    p.add_argument("--original-split", action="store_true",
                   help="keep the dataset's own train/validate/test patients, --n-val/--n-test are ignored")
    p.add_argument("--oversample-ends", type=float, default=None, metavar="FACTOR",
                   help="draw base and apex prostate slices FACTOR times as often as the rest, epochs stay the same length")
    return p.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    loaders = full_load_dataset(seed=args.seed, batch_size=args.batch_size, augment=args.augment,
                                num_workers=args.num_workers, n_val=args.n_val, n_test=args.n_test,
                                root=args.data_root, cache_dir=args.cache_dir, patient_weighting=args.patient_weighting,
                                slice_weights=(lambda ds: end_weights(ds, args.oversample_ends)) if args.oversample_ends else None,
                                resplit=not args.original_split, context=args.context)

    model = UNet2D(1 + 2 * args.context, N_LABELS, conv_depths=tuple(args.conv_depths), skip_connected=args.skip_connected).to(device)
    loss_fn = CEDiceLoss() if args.loss == "ce_dice" else nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_name = f"{'skip' if args.skip_connected else 'noskip'}_pw-{args.patient_weighting}_seed{args.seed}"
    if args.loss != "ce":
        run_name += f"_{args.loss}"
    if args.lr_schedule != "none":
        run_name += f"_{args.lr_schedule}"
    if args.oversample_ends:
        run_name += f"_ends{args.oversample_ends:g}"
    if args.original_split:
        run_name += "_origsplit"
    if args.context:
        run_name += f"_ctx{args.context}"
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{run_name} on {device}, {n_params:.2f}M params, "
          + ", ".join(f"{s}: {len(l)} batches" for s, l in loaders.items()))
    return device, loaders, model, loss_fn, optimiser, run_dir

class CEDiceLoss(nn.Module):
    def __init__(self, n_classes=N_LABELS, smooth=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, logits, mask):
        probs = logits.softmax(dim=1)[:, 1:] 
        target = nn.functional.one_hot(mask, self.n_classes).permute(0, 3, 1, 2)[:, 1:].float()
        inter = (probs * target).sum(dim=(0, 2, 3))
        total = probs.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
        dice = (2 * inter + self.smooth) / (total + self.smooth)
        return self.ce(logits, mask) + (1 - dice.mean())

def dice_counts(logits, mask, n_classes=N_LABELS):
    pred = logits.argmax(dim=1)
    inter = torch.zeros(n_classes, device=mask.device)
    total = torch.zeros(n_classes, device=mask.device)
    for c in range(n_classes):
        p, t = pred == c, mask == c
        inter[c] = (p & t).sum()
        total[c] = p.sum() + t.sum()
    return inter, total

def dice_from_counts(inter, total, eps=1e-6):
    return ((2 * inter + eps) / (total + eps)).cpu()

def prostate_regions(dataset):
    scans = {}
    for k, sid in enumerate(dataset.ids):
        m = np.asarray(dataset.masks[dataset.indices[k]])
        scans.setdefault((sid.patient, sid.week), []).append((sid.slice, (m == LABELS["prostate"]).any(), (m == LABELS["bladder"]).any()))
    regions = {}
    for (patient, week), rows in scans.items():
        prostate = sorted(s for s, has_p, _ in rows if has_p)
        bladder = [s for s, _, has_b in rows if has_b]
        if not prostate:
            continue
        if bladder and np.mean(bladder) > np.mean(prostate):
            prostate.reverse()  # bladder above the high slices, so base starts there
        for rank, s in enumerate(prostate):
            regions[f"case_{patient:03d}_week_{week}_slice_{s}"] = 3 * rank // len(prostate)
    return regions

def end_weights(dataset, factor):
    """sampling weight per slice: factor for base and apex prostate slices, 1 for the rest"""
    regions = prostate_regions(dataset)
    ends = (REGIONS.index("base"), REGIONS.index("apex"))
    return torch.tensor([factor if regions.get(slice_name(sid), -1) in ends else 1.0 for sid in dataset.ids], dtype=torch.double)

def run_epoch(model, loader, loss_fn, device, optimiser=None, regions=None):
    training = optimiser is not None
    model.train(training)
    loss_sum, n_seen = 0.0, 0
    inter = torch.zeros(N_LABELS, device=device)
    total = torch.zeros(N_LABELS, device=device)
    reg_inter = torch.zeros(len(REGIONS), device=device)
    reg_total = torch.zeros(len(REGIONS), device=device)
    with torch.set_grad_enabled(training):
        for image, mask, names in loader:
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            logits = model(image)
            loss = loss_fn(logits, mask)
            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            loss_sum += loss.item() * image.shape[0]
            n_seen += image.shape[0]
            i, t = dice_counts(logits.detach(), mask)
            inter += i
            total += t
            if regions is not None:
                r = torch.tensor([regions.get(n, -1) for n in names], device=device)
                keep = r >= 0
                pred_p = (logits.detach().argmax(dim=1) == LABELS["prostate"]).flatten(1)
                true_p = (mask == LABELS["prostate"]).flatten(1)
                reg_inter.index_add_(0, r[keep], (pred_p & true_p).sum(1)[keep].float())
                reg_total.index_add_(0, r[keep], (pred_p.sum(1) + true_p.sum(1))[keep].float())
    region_dice = dice_from_counts(reg_inter, reg_total) if regions is not None else None
    return loss_sum / n_seen, dice_from_counts(inter, total), region_dice

def plot_history(history, run_dir):
    fig, (ax_loss, ax_dice, ax_region) = plt.subplots(1, 3, figsize=(18, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="validate")
    ax_loss.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax_loss.legend()
    val_dice = np.array(history["val_dice"])
    for name, c in LABELS.items():
        ax_dice.plot(epochs, val_dice[:, c], label=name, linewidth=2.5 if c == LABELS["prostate"] else 1)
    ax_dice.axhline(0.75, color="grey", linestyle="--", linewidth=0.8)
    ax_dice.set(xlabel="epoch", ylabel="dice", title="Validation Dice", ylim=(0, 1))
    ax_dice.legend(fontsize=8)
    region_dice = np.array(history["val_region_dice"])
    ax_region.plot(epochs, val_dice[:, LABELS["prostate"]], label="whole", linewidth=2.5, color="black")
    for r, name in enumerate(REGIONS):
        ax_region.plot(epochs, region_dice[:, r], label=name)
    ax_region.axhline(0.75, color="grey", linestyle="--", linewidth=0.8)
    ax_region.set(xlabel="epoch", ylabel="dice", title="Validation prostate Dice by region", ylim=(0, 1))
    ax_region.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=120)
    plt.close(fig)

def train(args, device, loaders, model, loss_fn, optimiser, run_dir):
    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": [], "val_region_dice": [], "lr": []}
    best_dice = -1.0
    regions = prostate_regions(loaders["validate"].dataset)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs) if args.lr_schedule == "cosine" else None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lr = optimiser.param_groups[0]["lr"]
        train_loss, train_dice, _ = run_epoch(model, loaders["train"], loss_fn, device, optimiser)
        val_loss, val_dice, val_region_dice = run_epoch(model, loaders["validate"], loss_fn, device, regions=regions)
        if scheduler is not None:
            scheduler.step()
        for key, value in zip(history, (train_loss, val_loss, train_dice.tolist(), val_dice.tolist(), val_region_dice.tolist(), lr)):
            history[key].append(value)

        prostate = val_dice[LABELS["prostate"]].item()
        improved = prostate > best_dice
        if improved:
            best_dice = prostate
            torch.save({"epoch": epoch, "model": model.state_dict(), "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "val_dice": val_dice.tolist()},
                       run_dir / "best.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimiser": optimiser.state_dict()}, run_dir / "last.pt")

        print(f"epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
              f"val dice {' '.join(f'{name} {d:.3f}' for name, d in zip(LABELS, val_dice.tolist()))}"
              f"{' *' if improved else ''} | prostate {' '.join(f'{name} {d:.3f}' for name, d in zip(REGIONS, val_region_dice.tolist()))}"
              f" | lr {lr:.2e} | {time.time() - t0:.0f}s")
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=1)
        plot_history(history, run_dir)
    print(f"best val prostate dice {best_dice:.4f}, saved to {run_dir / 'best.pt'}")
    return history

if __name__ == "__main__":
    args = parse_args()
    device, loaders, model, loss_fn, optimiser, run_dir = setup(args)
    train(args, device, loaders, model, loss_fn, optimiser, run_dir)
