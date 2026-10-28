"""Train, validate, test and save a HipMRI 2D segmentation model.

Both models use the same data pipeline, Dice loss (Isensee et al., Eq. 1),
optimiser, learning-rate schedule and epoch budget, so the comparison only
changes the architecture.

* Optimiser: Adam, learning rate 5e-4, L2 weight decay 1e-5 (as in the paper).
* Schedule: lr = 5e-4 * 0.985 ** (e * 300 / epochs) for epoch index e = 0,
  1, ... The paper decays by 0.985 per "epoch" of 100 batches for 300 such
  epochs; scaling the exponent keeps a similar overall decay for any number
  of full passes over the training data (last-epoch lr is about 1.2% of the
  initial value for 40 epochs; 1.1% in the paper).
* Reproducibility: the shuffle order and augmentation of epoch e are seeded
  with ``seed + e``, so a resumed run sees the same data as an
  uninterrupted one (cuDNN autotuning still makes results not bit-for-bit).
* Checkpoint selection: after every epoch the mean per-series 3D prostate
  Dice on the validation split is computed; the best epoch is kept
  (ties go to the lower validation loss) along with the last epoch.
* Test: only with ``--test``, after the full epoch budget, once for the best
  and once for the last checkpoint. It is refused for shortened runs and
  never overwrites earlier test results, so test scores cannot feed back
  into any setting.
* 2.5D experiment: ``--context-slices K`` feeds the K neighbouring slices on
  each side as extra input channels (see ``dataset.py``). Everything else
  stays the same; the default 0 is the 2D setting of the main results.

Examples::

    # short timing run: 100 training batches + validation (test is never touched)
    python train.py --model unet --epochs 1 --max-train-batches 100
    # full run with the final test evaluation
    python train.py --model improved --epochs 40 --out-dir runs/improved --test
    # 2.5D experiment: three adjacent slices as input
    python train.py --model improved --context-slices 1 --out-dir runs/improved_25d
"""

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from dataset import make_dataloaders
from metrics import SegmentationMetrics
from modules import MulticlassSoftDiceLoss, build_model, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=["unet", "improved"], required=True)
    parser.add_argument("--data-root", default="keras_slices_data")
    parser.add_argument("--out-dir", default=None, help="default: runs/<model>")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lr-decay", type=float, default=0.985)
    parser.add_argument("--paper-epochs", type=int, default=300,
                        help="number of paper epochs the decay is spread over")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-slices", type=int, default=0,
                        help="neighbouring slices on each side as extra channels (0 = 2D)")
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="stop each epoch early (timing and smoke tests only)")
    parser.add_argument("--test", action="store_true",
                        help="evaluate the test split once after the full training budget")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt in out-dir")
    return parser.parse_args()


WARMUP_BATCHES = 5  # excluded from the steady-state timing (cuDNN autotuning, worker start-up)
# Arguments that must not change when a run is resumed.
RESUME_KEYS = ("model", "epochs", "batch_size", "lr", "weight_decay", "lr_decay",
               "paper_epochs", "seed", "max_train_batches", "context_slices")
# Values for keys added after the first runs, whose checkpoints do not store them.
RESUME_DEFAULTS = {"context_slices": 0}


def run_epoch(model, loader, loss_fn, device, use_amp, optimizer=None, scaler=None,
              max_batches=None, metrics=None):
    """Run one pass over ``loader``; trains when an optimizer is given.

    Returns a dict with the mean loss per batch, the number of batches, the
    total seconds and the steady-state seconds per batch (after warm-up).
    """
    training = optimizer is not None
    model.train(training)
    total_loss, batches = 0.0, 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = warm_start = time.perf_counter()

    with torch.set_grad_enabled(training):
        for batch in loader:
            if max_batches is not None and batches >= max_batches:
                break
            if batches == WARMUP_BATCHES:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                warm_start = time.perf_counter()
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
            loss = loss_fn(logits, masks)  # computed in float32 inside the loss

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            if metrics is not None:
                metrics.update(logits.argmax(dim=1), batch["label"].to(device),
                               batch["case"], batch["week"], batch["slice"])
            total_loss += loss.item()
            batches += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    steady_batches = batches - WARMUP_BATCHES
    return {
        "loss": total_loss / max(batches, 1),
        "batches": batches,
        "seconds": end - start,
        "steady_seconds_per_batch": ((end - warm_start) / steady_batches if steady_batches > 0
                                     else (end - start) / max(batches, 1)),
    }


def evaluate(model, loader, loss_fn, device, use_amp):
    """Return (mean loss, metric summary) for a validation or test loader."""
    metrics = SegmentationMetrics()
    result = run_epoch(model, loader, loss_fn, device, use_amp, metrics=metrics)
    summary = metrics.summary()
    summary["seconds"] = result["seconds"]
    return result["loss"], summary


def save_curves(history, path):
    """Plot training/validation loss and validation prostate Dice per epoch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], label="validation")
    ax_loss.set(xlabel="epoch", ylabel="soft Dice loss (Eq. 1)", title="Loss")
    ax_loss.legend()
    ax_dice.plot(epochs, [h["val_prostate_dice"] for h in history], label="per-series mean (primary)")
    ax_dice.plot(epochs, [h["val_prostate_dice_case_weighted"] for h in history],
                 label="case-weighted mean")
    ax_dice.plot(epochs, [h["val_foreground_dice"] for h in history], label="foreground classes 1-5")
    ax_dice.set(xlabel="epoch", ylabel="Dice", title="Validation Dice", ylim=(0, 1))
    ax_dice.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    if args.context_slices < 0:
        raise SystemExit("--context-slices must be 0 or more")
    if args.test and args.max_train_batches is not None:
        raise SystemExit("--test is not allowed with --max-train-batches (shortened runs)")
    out_dir = Path(args.out_dir or f"runs/{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_dir)
    if args.test and any(out_dir.glob("test_metrics_*.json")):
        raise SystemExit(f"{out_dir} already has test results; the test split is evaluated once")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    torch.backends.cudnn.benchmark = True  # faster convolutions; not bit-for-bit reproducible

    loaders = make_dataloaders(args.data_root, batch_size=args.batch_size,
                               num_workers=args.num_workers, seed=args.seed,
                               context_slices=args.context_slices)
    model = build_model(args.model, in_channels=2 * args.context_slices + 1).to(device)
    loss_fn = MulticlassSoftDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: args.lr_decay ** (e * args.paper_epochs / args.epochs))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    config = vars(args) | {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        # str(): torch.__version__ is a TorchVersion object, which torch.load
        # (weights_only=True) refuses to unpickle from a checkpoint.
        "torch": str(torch.__version__), "python": platform.python_version(), "amp": use_amp,
        "parameters": count_parameters(model),
        "train_batches_per_epoch": len(loaders["train"]),
    }
    history, start_epoch = [], 1
    best = {"score": -1.0, "val_loss": float("inf"), "epoch": None}
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    if args.resume:
        if not last_path.is_file():
            raise SystemExit(f"--resume given but {last_path} does not exist (check --out-dir)")
        state = torch.load(last_path, map_location=device)
        changed = [k for k in RESUME_KEYS
                   if state["config"].get(k, RESUME_DEFAULTS.get(k)) != getattr(args, k)]
        if changed:
            raise SystemExit(f"Cannot resume: {changed} differ from the saved run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, best, start_epoch = state["history"], state["best"], state["epoch"] + 1
        print(f"Resumed after epoch {state['epoch']}")
    # Written only after the resume check, so a refused resume leaves config.json as it was.
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2))

    for epoch in range(start_epoch, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        lr = optimizer.param_groups[0]["lr"]
        # Seed this epoch's shuffle and augmentation from the epoch number.
        loaders["train"].generator.manual_seed(args.seed + epoch)
        torch.manual_seed(args.seed + epoch)
        train = run_epoch(model, loaders["train"], loss_fn, device, use_amp, optimizer, scaler,
                          max_batches=args.max_train_batches)
        val_loss, val = evaluate(model, loaders["validation"], loss_fn, device, use_amp)
        scheduler.step()

        train_loss = train["loss"]
        record = {
            "epoch": epoch, "lr": lr, "train_loss": train_loss, "val_loss": val_loss,
            "val_prostate_dice": val["prostate_dice_series_mean"],
            "val_prostate_dice_case_weighted": val["prostate_dice_case_weighted"],
            "val_prostate_dice_global": val["prostate_dice_global"],
            "val_foreground_dice": val["foreground_dice_series_mean"],
            "train_batches": train["batches"], "train_seconds": train["seconds"],
            "seconds_per_train_batch": train["steady_seconds_per_batch"],
            "val_seconds": val["seconds"],
            "peak_gpu_memory_gib": (torch.cuda.max_memory_allocated() / 2 ** 30
                                    if device.type == "cuda" else None),
        }
        history.append(record)

        score = record["val_prostate_dice"] or 0.0  # every validation series has prostate
        if score > best["score"] or (score == best["score"] and val_loss < best["val_loss"]):
            best = {"score": score, "val_loss": val_loss, "epoch": epoch}
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, best_path)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "history": history, "best": best, "config": config},
                   last_path)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        save_curves(history, out_dir / "curves.png")

        full_epoch_estimate = (record["seconds_per_train_batch"] * len(loaders["train"])
                               + record["val_seconds"])
        print(f"epoch {epoch:3d}/{args.epochs} lr {lr:.2e} train_loss {train_loss:.4f} "
              f"val_loss {val_loss:.4f} val_prostate_dice {score:.4f} (best {best['score']:.4f} "
              f"@ {best['epoch']}) train {train['seconds']:.0f}s val {val['seconds']:.0f}s "
              f"est_full_epoch {full_epoch_estimate:.0f}s", flush=True)

    if not args.test:
        return

    # Final test evaluation: once for the best and once for the last checkpoint.
    for tag, path in (("best", best_path), ("last", last_path)):
        model.load_state_dict(torch.load(path, map_location=device)["model"])
        test_loss, test = evaluate(model, loaders["test"], loss_fn, device, use_amp)
        test["loss"] = test_loss
        test["checkpoint_epoch"] = best["epoch"] if tag == "best" else history[-1]["epoch"]
        (out_dir / f"test_metrics_{tag}.json").write_text(json.dumps(test, indent=2))
        print(f"test ({tag}, epoch {test['checkpoint_epoch']}): prostate Dice per-series mean "
              f"{test['prostate_dice_series_mean']:.4f}, case-weighted "
              f"{test['prostate_dice_case_weighted']:.4f}, global {test['prostate_dice_global']:.4f}")


if __name__ == "__main__":
    main()
