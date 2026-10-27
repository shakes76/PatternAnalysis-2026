"""Exact validation-only checkpoint ordering; no training or test-set access."""

import math


SELECTION_RULES = {
    "ce": "minimum full-validation cross_entropy; exact ties choose the earliest epoch",
    "id5_dice": "maximum full-validation pixel-pooled ID5 Dice; exact ties choose lower cross_entropy, then the earliest epoch",
}


def checkpoint_candidate(epoch, validation):
    """Copy the scalar validation values that determine checkpoint ordering."""
    dice = validation.get("dice_per_class", [])
    candidate = {
        "epoch": epoch,
        "validation_cross_entropy": validation["cross_entropy"],
        "validation_id5_dice": dice[5] if len(dice) > 5 else None,
    }
    checkpoint_key(candidate, "ce")
    return candidate


def checkpoint_key(candidate, metric):
    """Smaller tuples are better. Ties are exact, without a tolerance band."""
    if metric not in SELECTION_RULES:
        raise ValueError(f"Unknown checkpoint selection metric: {metric}")
    epoch = candidate["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("Checkpoint epoch must be a positive integer.")
    ce, dice = candidate["validation_cross_entropy"], candidate["validation_id5_dice"]
    if ce is None or not math.isfinite(ce) or ce < 0:
        raise ValueError("Checkpoint selection requires finite, nonnegative validation CE.")
    if dice is not None and (not math.isfinite(dice) or not 0 <= dice <= 1):
        raise ValueError("Validation ID5 Dice must be finite in [0,1] or undefined (None).")
    if metric == "id5_dice":
        if dice is None:
            raise ValueError("ID5 checkpoint selection requires a defined validation ID5 Dice.")
        return (-dice, ce, epoch)
    return (ce, epoch)


def is_better_checkpoint(candidate, incumbent, metric):
    """Check a new candidate against the saved best using one shared rule."""
    key = checkpoint_key(candidate, metric)
    return incumbent is None or key < checkpoint_key(incumbent, metric)
