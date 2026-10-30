"""Verify metric edge cases and aggregation across unequal batches."""

import math
import torch

from metrics import positive_slice_stats, summarize_stats


def main():
    target = torch.tensor([
        [[5, 5], [0, 0]],
        [[5, 0], [0, 0]],
        [[5, 5], [0, 0]],
        [[0, 0], [0, 0]],
    ], dtype=torch.int64)

    prediction = torch.tensor([
        [[5, 5], [0, 0]],
        [[0, 0], [0, 0]],
        [[5, 0], [0, 0]],
        [[5, 0], [0, 0]],
    ], dtype=torch.int64)

    stats = positive_slice_stats(prediction, target)
    result = summarize_stats(stats)

    assert result["eligible_slices"] == 3
    assert result["excluded_slices"] == 1
    assert result["negative_slices_with_fp"] == 1
    assert result["negative_slice_fp_pixels"] == 1
    assert math.isclose(result["mean_dice"], 5 / 9, abs_tol=1e-12)
    assert math.isclose(result["mean_iou"], 0.5, abs_tol=1e-12)

    missed = summarize_stats(
        positive_slice_stats(prediction[1:2], target[1:2])
    )
    assert missed["mean_dice"] == 0.0

    empty = summarize_stats(
        positive_slice_stats(prediction[3:], target[3:])
    )
    assert empty["mean_dice"] is None

    # Batch partitioning must not change the final slice-weighted result.
    first = positive_slice_stats(prediction[:1], target[:1])
    second = positive_slice_stats(prediction[1:], target[1:])
    combined = {key: first[key] + second[key] for key in first}
    combined_result = summarize_stats(combined)
    assert math.isclose(
        combined_result["mean_dice"], result["mean_dice"], abs_tol=1e-12
    )

    print("Eligible slices:", result["eligible_slices"])
    print("Excluded slices:", result["excluded_slices"])
    print("Mean prostate Dice:", result["mean_dice"])
    print("Mean prostate IoU:", result["mean_iou"])
    print("Missed positive slice Dice:", missed["mean_dice"])
    print("No eligible slices:", empty["mean_dice"])
    print("Unequal batch aggregation: PASS")
    print("Metric checks: PASS")


if __name__ == "__main__":
    main()
