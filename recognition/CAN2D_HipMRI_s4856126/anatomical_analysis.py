"""Additional HipMRI analysis: prostate-to-organ leakage and base/apex candidates.

Put next to dataset.py and evaluate.py inside the recognition model folder.
Run from that folder, inside a Rangpur GPU compute allocation:
    python anatomical_analysis.py --device cuda --split test

This script DOES NOT modify models/checkpoints or train anything.
Base/apex are never inferred solely from slice number: it only locates the
lowest/highest prostate-containing slice numbers per patient/week.  Confirm
image orientation and anatomy from original 3D NIfTI before naming them.

Patient identifiers and medical-derived images/CSVs may be sensitive.
Do not upload output CSVs or any patient-level medical images to public GitHub.
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import HipMRIDataset
from evaluate import load_model

CHECKPOINTS = {
    "baseline": Path("checkpoints/baseline_unet.pt"),
    "can": Path("checkpoints/can_best.pt"),
}

# Example: img_040_week_0_slice_24.nii.gz or seg_040_week_0_slice_24.nii.gz
FILE_PATTERN = re.compile(
    r"(?P<patient>\d+)[_-]week[_-]?(?P<week>\d+)[_-]slice[_-]?(?P<slice>\d+)",
    flags=re.IGNORECASE,
)


def filename_for(dataset, index):
    """Use dataset.pairs if available; never invent a real patient or slice ID."""
    pairs = getattr(dataset, "pairs", None)
    if pairs is None:
        return ""
    try:
        pair = pairs[index]
        image_path = pair[0] if isinstance(pair, (tuple, list)) else pair
        return Path(str(image_path)).name
    except (IndexError, KeyError, TypeError, ValueError, AttributeError):
        return ""


def identify_scan(filename):
    m = FILE_PATTERN.search(filename)
    if not m:
        return None
    return (m.group("patient"), m.group("week"), int(m.group("slice")))


def prostate_dice_and_fn(gt, pred):
    target = gt == 5
    positive = pred == 5
    tp = int(np.count_nonzero(target & positive))
    fp = int(np.count_nonzero(~target & positive))
    fn = int(np.count_nonzero(target & ~positive))
    den = 2 * tp + fp + fn
    # No prostate in GT and prediction: undefined, not automatically a perfect hit.
    dice = round(2 * tp / den, 6) if den > 0 else ""
    return dice, fn


def organ_leakage(gt, pred):
    """Count GT bladder/rectum pixels that were predicted as prostate."""
    predicted_prostate = pred == 5
    return (
        int(np.count_nonzero((gt == 3) & predicted_prostate)),
        int(np.count_nonzero((gt == 4) & predicted_prostate)),
    )


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyse(dataset, baseline, can, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    records = []
    offset = 0
    baseline.eval()
    can.eval()
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            bp = baseline(images).argmax(dim=1).cpu().numpy()
            cp = can(images).argmax(dim=1).cpu().numpy()
            truth = targets.numpy()
            for j, gt in enumerate(truth):
                index = offset + j
                filename = filename_for(dataset, index)
                parsed = identify_scan(filename)
                b_dice, b_fn = prostate_dice_and_fn(gt, bp[j])
                c_dice, c_fn = prostate_dice_and_fn(gt, cp[j])
                b_bladder, b_rectum = organ_leakage(gt, bp[j])
                c_bladder, c_rectum = organ_leakage(gt, cp[j])
                records.append({
                    "index": index,
                    "filename": filename,
                    "patient_id": parsed[0] if parsed else "",
                    "week": parsed[1] if parsed else "",
                    "slice_number": parsed[2] if parsed else "",
                    "gt_prostate_pixels": int(np.count_nonzero(gt == 5)),
                    "baseline_prostate_dice": b_dice,
                    "can_prostate_dice": c_dice,
                    "baseline_prostate_fn": b_fn,
                    "can_prostate_fn": c_fn,
                    "baseline_bladder_leak": b_bladder,
                    "can_bladder_leak": c_bladder,
                    "baseline_rectum_leak": b_rectum,
                    "can_rectum_leak": c_rectum,
                })
            offset += len(truth)
    return records


def leakage_summary(records):
    return [
        {"model": model,
         "bladder_gt_pixels_predicted_prostate": sum(r[f"{model}_bladder_leak"] for r in records),
         "rectum_gt_pixels_predicted_prostate": sum(r[f"{model}_rectum_leak"] for r in records),
         "test_slices": len(records)}
        for model in ("baseline", "can")
    ]


def base_apex_candidates(records, end_count=3):
    """Extract low/high index ends, NOT clinically confirmed base/apex labels.

    Separate patient + week studies.  Keep only slices with GT prostate.
    Do not assume slice ordering represents superior-inferior anatomy.
    """
    groups = defaultdict(list)
    for r in records:
        if (r["patient_id"] and r["week"] != "" and
                r["slice_number"] != "" and r["gt_prostate_pixels"] > 0):
            groups[(r["patient_id"], r["week"])].append(r)

    candidates = []
    for (patient, week), group in sorted(groups.items()):
        group.sort(key=lambda r: r["slice_number"])
        # If there are fewer than 2*end_count slices, avoid duplicate records.
        low_ids = {r["index"] for r in group[:end_count]}
        high_ids = {r["index"] for r in group[-end_count:]}
        for row in group:
            in_low = row["index"] in low_ids
            in_high = row["index"] in high_ids
            if not (in_low or in_high):
                continue
            end = "both_ends" if in_low and in_high else (
                "lower_slice_number" if in_low else "higher_slice_number")
            candidates.append({
                **row,
                "candidate_end": end,
                "anatomical_region": "UNVERIFIED - confirm 3D orientation",
            })
    return candidates


def selected_case_rows(records, selection_csv):
    if not selection_csv.is_file():
        return []
    with selection_csv.open(newline="", encoding="utf-8") as f:
        ids = {int(row["index"]) for row in csv.DictReader(f)}
    return [r for r in records if r["index"] in ids]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--split", choices=["test", "validate"], default="test")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--end-count", type=int, default=3,
                   help="candidate slices from each end of a prostate-containing run")
    args = p.parse_args()
    if args.batch_size < 1 or args.end_count < 1:
        p.error("--batch-size and --end-count must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Use a Slurm GPU allocation, or --device cpu")
    for path in CHECKPOINTS.values():
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path} (run from model folder)")

    dataset = HipMRIDataset(split=args.split)
    baseline = load_model("baseline", CHECKPOINTS["baseline"], device)
    can = load_model("can", CHECKPOINTS["can"], device)
    rows = analyse(dataset, baseline, can, device, args.batch_size)

    output = Path("outputs/anatomical_analysis")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / f"{args.split}_all_slices.csv", rows)
    summary = leakage_summary(rows)
    write_csv(output / f"{args.split}_organ_leakage_summary.csv", summary)
    candidates = base_apex_candidates(rows, args.end_count)
    write_csv(output / f"{args.split}_base_apex_CANDIDATES.csv", candidates)
    selected = selected_case_rows(
        rows, Path("outputs/failure_cases_can") / f"{args.split}_selected_cases.csv")
    write_csv(output / f"{args.split}_selected_cases_leakage.csv", selected)

    print("\nOrgan leakage on complete evaluation split:")
    for s in summary:
        print(f"  {s['model']}: bladder={s['bladder_gt_pixels_predicted_prostate']} "
              f"pixels, rectum={s['rectum_gt_pixels_predicted_prostate']} pixels")
    print(f"\nBase/apex END CANDIDATES: {len(candidates)} slices")
    if not candidates:
        print("WARNING: No source patient/slice IDs were parsed from dataset.pairs. "
              "Check dataset.py metadata; never infer base/apex from DataLoader indices.")
    else:
        print("WARNING: lower/higher slice numbers are NOT verified anatomical "
              "base/apex. Check original 3D NIfTI affine/orientation and anatomy.")
    print(f"\nSaved private analysis CSVs to {output}")
    print("Do not commit patient-level CSVs or medical images to public GitHub.")


if __name__ == "__main__":
    main()
