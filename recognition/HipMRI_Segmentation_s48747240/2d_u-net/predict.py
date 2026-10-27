"""
predict.py

Example usage of the trained 2D U-Net: loads saved model weights, scores every
test slice, prints per-class Dice, then visualises the WORST failure cases
(input / ground truth / prediction / difference overlay) for the failure-case
autopsy.

Slice selection: find the worst-performing classes overall, then pick slices
where those classes are present and score lowest, spaced apart so the figures
are not near-duplicates from the same scan.

NumPy is used here for visualization/array handling, as permitted by the
assignment spec for predict.py specifically.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from modules import dice_coefficient, combined_loss
from dataset import load_hipmri_2d

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "unet"
CKPT_PREFIX = {"unet": "unet2d", "baseline": "baseline2d",
               "improved": "improved2d",
               "improved_pcd": "improved2d_pcd"}[MODEL_NAME]
CHECKPOINT_PATH = f"checkpoints/{CKPT_PREFIX}_best.keras"
OUTPUT_DIR = "predictions" if MODEL_NAME == "unet" else f"predictions_{MODEL_NAME}"
NUM_EXAMPLES_TO_VISUALIZE = 5
NUM_WORST_CLASSES = 2   # how many of the weakest classes to target
MIN_SLICE_GAP = 10      # min index distance between picks (avoids near-duplicates)


def load_trained_model(checkpoint_path: str) -> tf.keras.Model:
    """Load a saved model, registering the custom loss/metric used in training."""
    return tf.keras.models.load_model(checkpoint_path, compile=False)


def per_class_dice(y_true, y_pred, num_classes, smooth=1e-6):
    """Compute Dice score separately for each class channel."""
    scores = []
    for c in range(num_classes):
        true_c = y_true[..., c].flatten()
        pred_c = y_pred[..., c].flatten()
        intersection = np.sum(true_c * pred_c)
        denom = np.sum(true_c) + np.sum(pred_c)
        scores.append((2.0 * intersection + smooth) / (denom + smooth))
    return scores


def score_test_set(model, test_ds):
    """
    Pass 1: run inference over the whole test set (no plotting).
    Returns per-slice, per-class Dice and a boolean array saying which classes
    are present in each ground-truth slice. Both have shape (N, num_classes).
    """
    dice_rows, present_rows, area_rows = [], [], []
    for images, true_masks in test_ds:
        preds = model(images, training=False).numpy()
        true_np = true_masks.numpy()
        for i in range(true_np.shape[0]):
            num_classes = true_np.shape[-1]
            dice_rows.append(per_class_dice(true_np[i], preds[i], num_classes))
            areas = true_np[i].reshape(-1, num_classes).sum(axis=0)
            area_rows.append(areas)
            present_rows.append(areas > 0)
    return np.array(dice_rows), np.array(present_rows), np.array(area_rows)


def select_failure_slices(dice, present, areas, k, n_worst, min_gap):
    """
    Pick k slices to inspect.
    1. Find the n_worst classes (excluding background class 0) by mean Dice over
       the slices where that class is actually present.
    2. Rank slices containing those classes by their mean Dice on them.
    3. Greedily take the worst, skipping any within min_gap of one already taken.
    Falls back to worst overall slices if too few candidates exist.
    Returns (selected_indices, worst_classes).
    """
    num_classes = dice.shape[1]
    class_means = {}
    for c in range(1, num_classes):
        if present[:, c].any():
            class_means[c] = dice[present[:, c], c].mean()
    worst_classes = sorted(class_means, key=class_means.get)[:n_worst]

    # Slice score: mean Dice over the worst classes present in that slice
    slice_scores = np.full(dice.shape[0], np.inf)
    for i in range(dice.shape[0]):
        cs = [c for c in worst_classes if present[i, c]]
        if cs:
            slice_scores[i] = (np.mean([dice[i, c] for c in cs])
                               - 1e-6 * sum(areas[i, c] for c in cs))

    # Fallback ranking: mean Dice over all classes present in the slice
    overall = np.array([dice[i, present[i]].mean() if present[i].any() else np.inf
                        for i in range(dice.shape[0])])

    selected = []

    def take(order, gap):
        for i in order:
            if len(selected) >= k:
                return
            if not np.isfinite(order_scores[i]):
                continue
            if i in selected:
                continue
            if all(abs(i - j) >= gap for j in selected):
                selected.append(int(i))

    order_scores = slice_scores
    take(np.argsort(slice_scores), min_gap)
    order_scores = overall
    take(np.argsort(overall), min_gap)   # fill with worst overall, still spaced
    take(np.argsort(overall), 1)         # last resort: ignore spacing
    return selected, worst_classes


def select_contrast_slices(dice, present, cls, taken, min_gap):
    """
    Pick two extra slices for class `cls` to contrast with the failures:
    - good:    a typical success (median Dice among slices with Dice >= 0.8)
    - partial: a partial overlap (Dice in [0.2, 0.6], closest to 0.4)
    Both must have the class present and be >= min_gap from slices already taken.
    Returns (good, partial); either may be None if no slice qualifies.
    """
    idx = np.where(present[:, cls])[0]

    def far(i, others):
        return all(abs(int(i) - j) >= min_gap for j in others)

    good = None
    good_c = [i for i in idx if dice[i, cls] >= 0.8 and far(i, taken)]
    if good_c:
        vals = np.array([dice[i, cls] for i in good_c])
        good = int(good_c[int(np.argmin(np.abs(vals - np.median(vals))))])

    partial = None
    others = list(taken) + ([good] if good is not None else [])
    partial_c = [i for i in idx if 0.2 <= dice[i, cls] <= 0.6 and far(i, others)]
    if partial_c:
        partial = int(min(partial_c, key=lambda i: abs(dice[i, cls] - 0.4)))
    return good, partial


def visualize_prediction(image, true_mask, pred_mask, index, output_dir,
                         dice_scores, present):
    """
    Save a 4-panel figure: input image, ground truth mask, predicted mask,
    and a difference overlay (false positives/negatives). The title lists the
    test-set index and per-class Dice so each figure can be traced back.
    """
    num_classes = true_mask.shape[-1]
    true_labels = np.argmax(true_mask, axis=-1)
    pred_labels = np.argmax(pred_mask, axis=-1)
    difference = (true_labels != pred_labels).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))

    axes[0].imshow(image[..., 0], cmap="gray")
    axes[0].set_title("Input MRI Slice")

    # Fixed colour range so a class has the same colour in every figure
    axes[1].imshow(true_labels, cmap="tab10", vmin=0, vmax=num_classes - 1)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(pred_labels, cmap="tab10", vmin=0, vmax=num_classes - 1)
    axes[2].set_title("Prediction")

    axes[3].imshow(difference, cmap="Reds")
    axes[3].set_title("Difference (errors in red)")

    for ax in axes:
        ax.axis("off")

    caption = " | ".join(
        f"c{c}: {dice_scores[c]:.2f}" + ("" if present[c] else " (absent)")
        for c in range(num_classes)
    )
    fig.suptitle(f"Test slice #{index}   Dice  {caption}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"prediction_{index:04d}.png"))
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading trained model from {CHECKPOINT_PATH}...")
    model = load_trained_model(CHECKPOINT_PATH)

    print("Loading test data...")
    _, _, test_ds = load_hipmri_2d(batch_size=1, early_stop=False)

    print("Pass 1: scoring every test slice...")
    dice, present, areas = score_test_set(model, test_ds)
    mean_per_class = dice.mean(axis=0)

    print("\n--- Per-class mean Dice score on test set ---")
    for c, score in enumerate(mean_per_class):
        if present[:, c].any():
            when_present = dice[present[:, c], c].mean()
            print(f"  Class {c}: {score:.4f} (only slices where present: "
                  f"{when_present:.4f}, present in {present[:, c].sum()}/{len(dice)})")
        else:
            print(f"  Class {c}: {score:.4f} (never present in test set)")
    print(f"  Overall mean Dice: {mean_per_class.mean():.4f}")

    selected, worst_classes = select_failure_slices(
        dice, present, areas, NUM_EXAMPLES_TO_VISUALIZE, NUM_WORST_CLASSES, MIN_SLICE_GAP
    )
    print(f"\nWorst classes targeted: {worst_classes}")
    print(f"Selected test-slice indices: {selected}")
    for i in selected:
        print(f"  slice {i}: GT present c4/c5 = {present[i, 4]}/{present[i, 5]}, "
              f"GT pixels c4/c5 = {int(areas[i, 4])}/{int(areas[i, 5])}")

    cls = worst_classes[0]
    good, partial = select_contrast_slices(
        dice, present, cls, selected, MIN_SLICE_GAP
    )
    print(f"\nContrast slices for class {cls}: good={good}, partial={partial}")
    for tag, i in (("good", good), ("partial", partial)):
        if i is not None:
            print(f"  {tag} slice {i}: class {cls} Dice={dice[i, cls]:.2f}, "
                  f"GT pixels={int(areas[i, cls])}")
    selected = selected + [i for i in (good, partial) if i is not None]
    print("\nPass 2: re-running inference on selected slices for figures...")
    selected_set = set(selected)
    saved = 0
    idx = 0
    for images, true_masks in test_ds:
        true_np = true_masks.numpy()
        for b in range(true_np.shape[0]):
            if idx in selected_set:
                img = images.numpy()[b]
                pred = model(images[b:b + 1], training=False).numpy()[0]
                # Sanity check: the dataset must be in the same order as pass 1
                check = per_class_dice(true_np[b], pred, true_np.shape[-1])
                if not np.allclose(check, dice[idx], atol=1e-4):
                    print(f"  WARNING: slice {idx} differs from pass 1. "
                          "Is the test set being shuffled?")
                visualize_prediction(img, true_np[b], pred, idx, OUTPUT_DIR,
                                     dice[idx], present[idx])
                saved += 1
            idx += 1

    print(f"\nSaved {saved} failure-case visualizations to {OUTPUT_DIR}/")
    print("Use these for the failure-case autopsy (3-5 representative cases).")


if __name__ == "__main__":
    main()
