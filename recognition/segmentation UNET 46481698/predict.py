import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from tqdm import tqdm

from dataset import get_dataloaders
from modules import UNet2D, ImprovedUNet, dice_counts

PROSTATE_LABEL_ID = 5  # label value of the prostate in the segmentation files
# RGB values from 0 to 1: background black, then primaries, then secondaries
CLASS_COLOURS = [(0, 0, 0),  # class 0: black (background)
                 (0, 0, 1),  # class 1: blue
                 (0, 1, 0),  # class 2: green
                 (1, 0, 0),  # class 3: red
                 (0, 1, 1),  # class 4: cyan
                 (1, 0, 1)]  # class 5: magenta


# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------

def load_model(weights, use_improved, num_classes, device):
    """Build the model and load a checkpoint saved by train.py (or a plain state dict)."""
    model_class = ImprovedUNet if use_improved else UNet2D
    model = model_class(in_channels=1, num_classes=num_classes).to(device)
    checkpoint = torch.load(weights, map_location=device)
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_slices(model, dataset, indices, device):
    """Run the model on chosen dataset indices together as one batch.
    Returns numpy arrays: images (N,H,W), true class map (N,H,W), predicted class map (N,H,W)."""
    images = torch.stack([dataset[i][0] for i in indices]).to(device)
    labels = torch.stack([dataset[i][1] for i in indices])
    logits = model(images)
    return (images.cpu().numpy()[:, 0],
            labels.argmax(dim=1).numpy(),
            logits.argmax(dim=1).cpu().numpy())


# ----------------------------------------------------------------------------
# Metrics for ONE slice
# ----------------------------------------------------------------------------

def dice_per_class(pred, true, num_classes):
    """Dice of each class for one slice (nan if the class is in neither prediction nor truth)."""
    scores = []
    for c in range(num_classes):
        both = ((pred == c) & (true == c)).sum()
        total = (pred == c).sum() + (true == c).sum()
        scores.append(float("nan") if total == 0 else float(2 * both / total))
    return scores


def contour_errors(pred, true, prostate):
    """(false positive px, false negative px, over-contoured %, under-contoured %) for the prostate.
    Percentages are relative to the true prostate size, nan if the label has no prostate."""
    p, t = pred == prostate, true == prostate
    fp, fn, size = int((p & ~t).sum()), int((~p & t).sum()), int(t.sum())
    if size == 0:
        return fp, fn, float("nan"), float("nan")
    return fp, fn, 100 * fp / size, 100 * fn / size


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

def draw_rows(rows, num_classes, prostate):
    """One row per slice: input | ground truth | prediction | prostate errors | numbers.
    rows = list of (title, image, true mask, predicted mask)"""
    cmap = ListedColormap(CLASS_COLOURS[:num_classes])
    error_cmap = ListedColormap(["black", "green", "yellow", "red"])
    n = len(rows)
    height = 4.5 * n
    fig = plt.figure(figsize=(14, height))
    name_map = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]

    for r, (title, image, true, pred) in enumerate(rows):
        image, true, pred = (np.rot90(a, k=-1) for a in (image, true, pred))
        ax = fig.add_subplot(n, 5, 5 * r + 1)
        ax.imshow(image, cmap="gray")
        ax.set_title("Image")

        ax = fig.add_subplot(n, 5, 5 * r + 2)
        ax.imshow(true, cmap=cmap, vmin=0, vmax=num_classes - 1, interpolation="bilinear")
        ax.set_title("Segmentation Mask")

        ax = fig.add_subplot(n, 5, 5 * r + 3)
        ax.imshow(pred, cmap=cmap, vmin=0, vmax=num_classes - 1, interpolation="bilinear")
        ax.set_title("Prediction")

        # 1 = correct (green), 2 = false positive (yellow), 3 = missed (red), 0 = hidden
        p, t = pred == prostate, true == prostate
        errors = 1 * (p & t) + 2 * (p & ~t) + 3 * (~p & t)
        ax = fig.add_subplot(n, 5, 5 * r + 4)
        ax.imshow(image, cmap="gray")
        ax.imshow(np.ma.masked_equal(errors, 0), cmap=error_cmap, vmin=0, vmax=3,
                  alpha=0.8, interpolation="bilinear")
        ax.set_title("Prostate: Mask/Prediction Overlap\n G = Prostate, Y = FP, "
                     "Red = FN")

        ax = fig.add_subplot(n, 5, 5 * r + 5)
        text = title + "\n\nDice per class\n"
        for c, d in enumerate(dice_per_class(pred, true, num_classes)):
            text += f"class {c} {name_map[c]}: {d:.3f}" + "\n"
        fp, fn, over, under = contour_errors(pred, true, prostate)
        text += (f"\nFalse Positive:\n  {fp} px\n  = {over:.2f}%"
                 f"\nFalse Negative:\n  {fn} px\n  = {under:.2f}%\n")
        ax.text(0, 1, text, va="top", family="monospace", fontsize=10, transform=ax.transAxes)

    for ax in fig.axes:
        ax.axis("off")
    fig.legend(handles=[Patch(color=col, label=f"class {c} ({name_map[c]})") for c, col in enumerate(CLASS_COLOURS[:num_classes])],
               loc="lower center", ncol=num_classes)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.5 / height, bottom=0.9 / height, wspace=0.05, hspace=0.25)
    return fig


def save_figure(fig, path, show):
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    if show:
        plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------------
# Test 1: whole test set
# ----------------------------------------------------------------------------

@torch.no_grad()
def full_test(model, loader, num_classes, prostate, device, num_show, out_dir, show):
    """Dice per class over the whole test set, over/under-contouring of the prostate,
    and a figure of the best and worst slices (by prostate Dice)."""
    inter = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)  # predicted + true pixels per class
    prostate_true = 0.0  # true prostate pixels in the test set
    records = []  # (prostate Dice, dataset index) per slice with prostate
    offset = 0  # loader is not shuffled, so offset = dataset index
    name_map = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]

    for image, label in tqdm(loader, desc="test", leave=False):
        image, label = image.to(device), label.to(device)
        logits = model(image)

        i, u = dice_counts(logits, label)
        inter += i
        union += u
        prostate_true += label[:, prostate].sum().item()

        # per-slice prostate Dice, to find the best and worst slices
        p = (logits.argmax(dim=1) == prostate).flatten(1)
        t = (label.argmax(dim=1) == prostate).flatten(1)
        both, total, size = (p & t).sum(1), p.sum(1) + t.sum(1), t.sum(1)
        for k in range(len(image)):
            if size[k] > 0:  # no prostate in label -> Dice is meaningless, skip
                records.append((2 * both[k].item() / total[k].item(), offset + k))
        offset += len(image)

    # -- report --
    dice = ((2 * inter + 1e-6) / (union + 1e-6)).cpu()
    print("\nDice per class (pooled over the test set):")
    for c, d in enumerate(dice.tolist()):
        print(f"class {c} {name_map[c]}: {d:.4f}")
    print(f"Mean Dice: {dice.mean().item():.4f}")
    print(f"Prostate Dice: {dice[prostate].item():.4f}")

    if prostate_true > 0:
        predicted = union[prostate].item() - prostate_true  # union = predicted + true
        fp = predicted - inter[prostate].item()
        fn = prostate_true - inter[prostate].item()
        print(f"False Positive:  {100 * fp / prostate_true:.2f}% of the true prostate")
        print(f"False Negative: {100 * fn / prostate_true:.2f}% of the true prostate")

    if not records:
        print("No test slice contained prostate, nothing to visualise")
        return
    scores = [d for d, _ in records]
    print(f"Per-slice prostate Dice (slices with prostate): mean {np.mean(scores):.4f}, "
          f"best {max(scores):.4f}, worst {min(scores):.4f}")

    # -- visualise the best and worst slices --
    records.sort()
    worst, best = records[:num_show], records[::-1][:num_show]
    rows = []
    for name, chosen in (("BEST", best), ("WORST", worst)):
        indices = [idx for _, idx in chosen]
        images, trues, preds = predict_slices(model, loader.dataset, indices, device)
        for k, (d, idx) in enumerate(chosen):
            rows.append((f"{name} Slice: Index {idx}\nprostate Dice {d:.3f}", images[k], trues[k], preds[k]))
    save_figure(draw_rows(rows, num_classes, prostate), out_dir / "best_and_worst.png", show)


# ----------------------------------------------------------------------------
# Test 2: effect of prostate size / class balance
# ----------------------------------------------------------------------------

@torch.no_grad()
def class_balance_test(model, dataset, num_classes, prostate, device, num_slices, indexes, out_dir, show):
    """Compare slices with the most, the least (but not zero) and no prostate.
    By default searches the first num_slices test slices (the first 60 are one patient).
    indexes = optional {"name": slice index} to hard code which slices are used."""
    if indexes is None:
        n = min(num_slices, len(dataset))
        sizes = [int(dataset[s][1][prostate].sum()) for s in range(n)]
        print("Prostate pixels in each slice index:", sizes)
        with_prostate = [s for s in range(n) if sizes[s] > 0]
        if not with_prostate or 0 not in sizes:
            print("Need slices with and without prostate - pass --indexes instead")
            return
        indexes = {"MAX prostate": max(with_prostate, key=lambda s: sizes[s]),
                   "MIN prostate": min(with_prostate, key=lambda s: sizes[s]),
                   "NO prostate": sizes.index(0)}

    images, trues, preds = predict_slices(model, dataset, list(indexes.values()), device)
    rows = []
    for k, (name, s) in enumerate(indexes.items()):
        fp, fn, over, under = contour_errors(preds[k], trues[k], prostate)
        dice = [round(d, 3) for d in dice_per_class(preds[k], trues[k], num_classes)]
        print(f"{name} (slice index {s}):")
        print("   Dice per class:", dice)
        print(f"   Over-contoured: {fp} px = {over:.1f}%   Under-contoured: {fn} px = {under:.1f}%")
        rows.append((f"{name}\nslice index {s}", images[k], trues[k], preds[k]))
    save_figure(draw_rows(rows, num_classes, prostate), out_dir / "class_balance_test.png", show)


# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained HipMRI segmentation model")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Dataset root"),
    parser.add_argument("--weights", default="checkpoints/best_model.pth", help="Checkpoint saved by train.py")
    parser.add_argument("--unet", action="store_true", help="Checkpoint is an ImprovedUNet (same flag as train.py)")
    parser.add_argument("--mode", choices=["full", "balance"], default="full")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_show", type=int, default=3, help="Best and worst slices to plot (full test)")
    parser.add_argument("--num_slices", type=int, default=60, help="Slices to search (class balance test)")
    parser.add_argument("--indexes", type=int, nargs="+",
                        help="Hard coded slice indexes for class mode, e.g. --indexes 1 20 50")
    parser.add_argument("--out_dir", default="./plots", help="Where to save the figures")
    parser.add_argument("--show", action="store_true", help="Also open the figure in a window")
    args = parser.parse_args()
    if args.data_path is None:
        parser.error("give --data_path")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: CUDA not found. Using CPU")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # full=True skips the prostate-only subsetting of the training set (not needed for testing)
    _, _, test_loader, id_channels, num_classes = get_dataloaders(
        args.data_path, num_workers=args.num_workers, batch_size=args.batch_size, full=True)
    prostate = id_channels[PROSTATE_LABEL_ID]  # channel index of the prostate label

    model = load_model(args.weights, args.unet, num_classes, device)

    if args.mode == "full":
        full_test(model, test_loader, num_classes, prostate, device, args.num_show, out_dir, args.show)
    else:
        indexes = None
        if args.indexes:
            indexes = {f"slice {s}": s for s in args.indexes}
        class_balance_test(model, test_loader.dataset, num_classes, prostate, device,
                       args.num_slices, indexes, out_dir, args.show)


if __name__ == "__main__":
    main()