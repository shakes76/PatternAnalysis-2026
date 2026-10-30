import argparse
import json
import os
import random
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset import CLASS_NAMES, get_dataloaders, load_splits, save_splits
from modules import build_model
from utils import DiceCELoss, count_parameters, evaluate

PROSTATE = CLASS_NAMES.index("prostate")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", default="runs/improved")
    p.add_argument("--model", choices=["improved", "baseline"], default="improved")
    p.add_argument("--split-file", default=None, help="re-use an existing splits.json")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--target-size", type=int, nargs=3, default=[128, 128, 64])
    p.add_argument("--slice-axis", type=int, default=2, help="axis along which slices are stacked")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--max-train", type=int, default=None, help="limit #train volumes (smoke test)")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_history(hist, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["train_loss"], label="train")
    ax[0].plot(hist["val_loss"], label="val")
    ax[0].set(xlabel="epoch", ylabel="loss (Dice+CE)", title="Loss")
    ax[0].legend()
    ax[1].plot(hist["val_fg_dice"], label="val mean foreground Dice")
    ax[1].plot(hist["val_prostate_dice"], label="val prostate Dice")
    ax[1].set(xlabel="epoch", ylabel="Dice", title="Validation Dice")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASS_NAMES)
    use_amp = device.type == "cuda" and not args.no_amp

    splits = load_splits(args.split_file) if args.split_file else None
    if splits is not None and args.max_train:
        splits["train"] = splits["train"][:args.max_train]
    loaders, splits = get_dataloaders(
        args.data_root, splits, tuple(args.target_size), args.batch_size,
        args.num_workers, num_classes, args.seed)
    if args.max_train and not args.split_file:
        loaders["train"].dataset.pairs = loaders["train"].dataset.pairs[:args.max_train]
    save_splits(splits, os.path.join(args.out_dir, "splits.json"))

    model = build_model(args.model, 1, num_classes, args.base_channels, args.dropout).to(device)
    n_params = count_parameters(model)
    print(f"Model: {args.model} | trainable parameters: {n_params:,} | device: {device}", flush=True)

    criterion = DiceCELoss(num_classes)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hist = {"train_loss": [], "val_loss": [], "val_fg_dice": [], "val_prostate_dice": []}
    best, ckpt_path = -1.0, os.path.join(args.out_dir, "best.pt")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t_start = time.time()
    print(f"Starting training: {len(loaders['train'])} train batches/epoch", flush=True)
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        model.train()
        running = 0.0
        for img, lab in loaders["train"]:
            img, lab = img.to(device, non_blocking=True), lab.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(img)
            # compute the loss in fp32 for numerical safety
            if isinstance(outputs, (list, tuple)):
                outputs = [o.float() for o in outputs]
            else:
                outputs = outputs.float()
            loss = criterion(outputs, lab)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            running += loss.item()
        scheduler.step()

        val = evaluate(model, loaders["val"], device, num_classes, PROSTATE,
                       args.slice_axis, criterion, boundary=False)
        hist["train_loss"].append(running / len(loaders["train"]))
        hist["val_loss"].append(val["loss"])
        hist["val_fg_dice"].append(val["mean_fg_dice"])
        hist["val_prostate_dice"].append(val["dice_per_class"][PROSTATE])
        print(f"epoch {epoch:3d}/{args.epochs} | train loss {hist['train_loss'][-1]:.4f} | "
              f"val loss {val['loss']:.4f} | val fg Dice {val['mean_fg_dice']:.4f} | "
              f"val prostate Dice {val['dice_per_class'][PROSTATE]:.4f} | "
              f"{time.time() - t_epoch:.0f}s (total {(time.time() - t_start) / 60:.1f} min)", flush=True)

        score = val["dice_per_class"][PROSTATE]
        if score == score and score > best:   # NaN-safe
            best = score
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch,
                        "val_prostate_dice": best}, ckpt_path)

    train_peak_gb = (torch.cuda.max_memory_allocated() / 1024 ** 3) if device.type == "cuda" else None
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(hist, f)
    plot_history(hist, os.path.join(args.out_dir, "curves.png"))

    # ---- test with the best checkpoint ------------------------------------ #
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    test = evaluate(model, loaders["test"], device, num_classes, PROSTATE, args.slice_axis,
                    criterion, return_per_case=True)
    infer_peak_gb = (torch.cuda.max_memory_allocated() / 1024 ** 3) if device.type == "cuda" else None

    print("\n=== TEST RESULTS (best val checkpoint, epoch %d) ===" % ckpt["epoch"])
    for name, d, i, h, a, m, n in zip(CLASS_NAMES, test["dice_per_class"], test["iou_per_class"],
                                     test["hd95_per_class"], test["asd_per_class"],
                                     test["missed_per_class"], test["gt_present_per_class"]):
        print(f"{name:>10}: Dice {d:.4f} | IoU {i:.4f} | HD95 {h:.2f} vox | ASD {a:.2f} vox | "
              f"missed {m}/{n}")
    print("prostate Dice by thirds of GT extent:", [f"{x:.4f}" for x in test["prostate_dice_thirds"]])
    print(f"params {n_params:,} | train peak VRAM {train_peak_gb} GB | inference peak VRAM "
          f"{infer_peak_gb} GB | mean inference {test['mean_inference_s']:.3f} s/volume")

    test.update({"model": args.model, "parameters": n_params, "train_peak_vram_gb": train_peak_gb,
                 "inference_peak_vram_gb": infer_peak_gb, "best_epoch": ckpt["epoch"],
                 "class_names": CLASS_NAMES})
    with open(os.path.join(args.out_dir, "test_results.json"), "w") as f:
        json.dump(test, f, indent=2)


if __name__ == "__main__":
    main()