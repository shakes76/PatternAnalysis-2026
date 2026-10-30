"""Qualitative failure autopsy on the official test split (post hoc).

For the frozen model(s) and the frozen threshold from best_meta.json, this
script picks wrong test scans (scan = mean AD probability of its slices):

* "confident" errors: the most confidently wrong scan of each error type
  (missed AD = false negative, false alarm on CN = false positive);
* "typical" errors: the wrong scan whose confidence is closest to the median
  confidence of all wrong scans of that type, so the figure is not only made
  of extreme cases;
* one confidently correct AD and one CN scan as a reference.

For every case it shows the slice the model got most wrong (for the reference
rows: most right), the Grad-CAM map of the ensemble for the predicted class
on that slice, and runs the checks the README asked for:

* how many of the scan's slices were wrong (one bad slice vs the whole scan);
* the subject's other visits anywhere in the course folders (train and test),
  with their labels and years, to see whether the subject's diagnosis
  changes between visits (label noise / conversion) and whether the subject
  was seen in training.

Nothing here changes the model or threshold; it is analysis after the final
test run. Outputs:
    <output-dir>/failure_autopsy_LOCAL_ONLY.csv  raw scan IDs, keep local
    <output-dir>/failure_autopsy.json            anonymous case numbers only
    <figure>                                     slices + Grad-CAM, no IDs

Example (Rangpur, official split, final ensemble):
    python -u failure_autopsy.py --confirm-final-test \
        --checkpoint checkpoints/convnext/official_cosine_ema_context3_s4{2,3,4}/best.pth \
        --output-dir eval_autopsy_convnext --figure figures/failure_autopsy_convnext.png
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from analyze import _plt, parse_raw_name, run_inference
from baseline import build_baseline
from dataset import DATA_ROOT, METADATA_PATH, get_image_id, load_datasets, load_patient_map
from modules import ConvNeXt
from predict import check_checkpoint_compatibility, get_sample_info, load_best_metadata


# --------------------------------------------------------------------------
# Case selection (pure NumPy, no model needed)
# --------------------------------------------------------------------------

def scan_table(samples, prob_ad, threshold):
    """One row per test scan: mean P(AD), prediction and per-slice details."""
    scans = defaultdict(list)
    for index, ((path, label), p) in enumerate(zip(samples, prob_ad)):
        scans[get_image_id(path)].append((index, float(p), label))
    rows = []
    for scan_id, entries in scans.items():
        label = entries[0][2]
        p = float(np.mean([e[1] for e in entries]))
        predicted = 0 if p >= threshold else 1
        slice_wrong = [(0 if q >= threshold else 1) != label for _, q, _ in entries]
        rows.append({
            "scan_id": scan_id,
            "true": "AD" if label == 0 else "NC",
            "predicted": "AD" if predicted == 0 else "NC",
            "scan_prob_ad": p,
            "confidence": p if predicted == 0 else 1 - p,
            "slices": len(entries),
            "slices_wrong": int(sum(slice_wrong)),
            # (dataset index, P(AD)) of every slice, used to pick the shown slice
            "slice_scores": [(i, q) for i, q, _ in entries],
        })
    return rows


def select_cases(rows):
    """Confident and typical error of each type, plus two correct references."""
    cases = []
    for true, predicted, kind in (("AD", "NC", "missed AD"), ("NC", "AD", "false alarm")):
        wrong = sorted((r for r in rows if r["true"] == true and r["predicted"] == predicted),
                       key=lambda r: -r["confidence"])
        if not wrong:
            continue
        cases.append(dict(wrong[0], kind=f"{kind}, most confident"))
        median = float(np.median([r["confidence"] for r in wrong]))
        typical = min(wrong[1:] or wrong, key=lambda r: abs(r["confidence"] - median))
        if typical["scan_id"] != wrong[0]["scan_id"]:
            cases.append(dict(typical, kind=f"{kind}, typical (median confidence)"))
    for true in ("AD", "NC"):
        right = [r for r in rows if r["true"] == true and r["predicted"] == true]
        if right:
            best = max(right, key=lambda r: r["confidence"])
            cases.append(dict(best, kind=f"correct {true} (reference)"))
    return cases


def shown_slice(case):
    """Dataset index of the slice that pushes hardest towards the prediction."""
    pick = max if case["predicted"] == "AD" else min
    return pick(case["slice_scores"], key=lambda item: item[1])


def course_folder_labels(root):
    """scan ID -> (folder, label) for every slice file in the course folders.

    Uses the raw train/test folders, not the training split, so that visits
    removed from training by the strict split are still found.
    """
    found = {}
    for folder in ("train", "test"):
        for label in ("AD", "NC"):
            directory = Path(root) / folder / label
            if not directory.is_dir():
                continue
            for path in directory.iterdir():
                found.setdefault(get_image_id(path), (folder, label))
    return found


def other_visits(scan_id, patient_map, folder_labels, metadata):
    """The subject's other scans: folder, label and year (no IDs returned)."""
    subject = patient_map[scan_id]
    visits = []
    for other_id, (folder, label) in folder_labels.items():
        if other_id == scan_id or patient_map.get(other_id) != subject:
            continue
        year = parse_raw_name(metadata[other_id]["raw"])["year"] if other_id in metadata else None
        visits.append({"folder": folder, "label": label, "year": year})
    return sorted(visits, key=lambda v: (v["year"] or 0, v["folder"]))


# --------------------------------------------------------------------------
# Grad-CAM
# --------------------------------------------------------------------------

def cam_layer(model):
    """Last spatial feature map: ConvNeXt stage 4 (7x7) or the CNN features."""
    return model.stages[-1] if isinstance(model, ConvNeXt) else model.features


def grad_cam(models, image, target_class):
    """Grad-CAM averaged over ensemble members, upsampled to the input size.

    For each member: weight each channel of the last feature map by the mean
    gradient of the target-class logit, sum over channels, keep positive
    evidence (ReLU) and normalise to [0, 1]; then average the members.
    """
    maps = []
    for model in models:
        store = {}

        def hook(_module, _inputs, output):
            output.retain_grad()
            store["features"] = output

        handle = cam_layer(model).register_forward_hook(hook)
        try:
            with torch.enable_grad():
                logits = model(image.unsqueeze(0))       # (1, 2)
                model.zero_grad(set_to_none=True)
                logits[0, target_class].backward()
        finally:
            handle.remove()
        features = store["features"]                     # (1, C, h, w)
        weights = features.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * features).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear",
                            align_corners=False)[0, 0]
        cam = cam - cam.min()
        maps.append((cam / cam.max().clamp_min(1e-8)).detach().cpu().numpy())
    return np.mean(maps, axis=0)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def plot_autopsy(panels, path):
    """One row per case: input slice, Grad-CAM overlay, text with the checks."""
    plt = _plt()
    fig, axes = plt.subplots(len(panels), 3, figsize=(9.5, 2.6 * len(panels)),
                             gridspec_kw={"width_ratios": [1, 1, 1.5]}, squeeze=False)
    for row, panel in enumerate(panels):
        image_ax, cam_ax, text_ax = axes[row]
        image_ax.imshow(panel["image"], cmap="gray")
        cam_ax.imshow(panel["image"], cmap="gray")
        cam_ax.imshow(panel["cam"], cmap="jet", alpha=0.4)
        for ax in axes[row]:
            ax.axis("off")
        colour = "green" if panel["true"] == panel["predicted"] else "red"
        image_ax.set_title(f"case {row + 1}: true {panel['true']}, pred {panel['predicted']}",
                           fontsize=8, color=colour)
        cam_ax.set_title(f"Grad-CAM for '{panel['predicted']}'", fontsize=8)
        text_ax.text(0, 0.5, panel["text"], fontsize=7.5, va="center", family="monospace")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def describe(case, visits, seen):
    other = ", ".join(f"{v['label']}({v['year'] or '?'},{v['folder']})" for v in visits) or "none"
    labels = {v["label"] for v in visits} | {case["true"]}
    return "\n".join([
        case["kind"],
        f"scan P(AD) = {case['scan_prob_ad']:.2f}",
        f"wrong slices: {case['slices_wrong']}/{case['slices']}",
        f"shown slice P(AD) = {case['shown_prob_ad']:.2f}",
        f"{case['sequence']}, {case['year']}",
        f"subject in training folder: {'yes' if seen else 'no'}",
        f"other visits: {other}",
        f"label changes across visits: {'YES' if len(labels) > 1 else 'no'}",
    ])


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--model", choices=["convnext", "cnn", "cnn_matched"],
                        default="convnext")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True,
                        help="Where to save the figure (e.g. figures/failure_autopsy.png)")
    parser.add_argument("--confirm-final-test", action="store_true",
                        help="Required: this reads the official test split")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if not args.confirm_final_test:
        parser.error("Reads the official test split; pass --confirm-final-test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_meta = [load_best_metadata(c) for c in args.checkpoint]
    if any(m is None for m in all_meta):
        raise ValueError("Every checkpoint needs best_meta.json")
    for key in ("split_mode", "input_mode", "selection_threshold", "model"):
        if len({str(m.get(key)) for m in all_meta}) != 1:
            raise ValueError(f"Checkpoints disagree on {key}")
    meta = all_meta[0]
    if meta["model"] != args.model:
        raise ValueError(f"Checkpoints are {meta['model']!r}; pass --model {meta['model']}")
    split_mode, input_mode = meta.get("split_mode", "strict"), meta.get("input_mode", "single")
    threshold = meta["selection_threshold"]

    train_ds, val_ds, test_ds = load_datasets(split_mode, input_mode)
    patient_map = load_patient_map()
    check = argparse.Namespace(split="test", model=args.model, input_mode=input_mode)
    for m in all_meta:
        check_checkpoint_compatibility(m, check, train_ds, val_ds, test_ds, patient_map)

    models = []
    for checkpoint in args.checkpoint:
        model = (ConvNeXt(num_classes=2, in_channels=3) if args.model == "convnext"
                 else build_baseline(args.model)).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
        model.eval()
        models.append(model)

    test_samples = get_sample_info(test_ds)
    prob_ad, labels, _, _ = run_inference(models, test_ds, device, args.batch_size,
                                          args.num_workers)
    if labels.tolist() != [label for _, label in test_samples]:
        raise RuntimeError("Test order changed")

    metadata = json.loads(Path(METADATA_PATH).read_text())
    folder_labels = course_folder_labels(DATA_ROOT)
    seen_subjects = {patient_map[get_image_id(p)] for ds in (train_ds, val_ds)
                     for p, _ in get_sample_info(ds)}

    cases = select_cases(scan_table(test_samples, prob_ad, threshold))
    panels, report = [], []
    for number, case in enumerate(cases, 1):
        index, case["shown_prob_ad"] = shown_slice(case)
        info = parse_raw_name(metadata[case["scan_id"]]["raw"])
        case.update(sequence=info["sequence"], year=info["year"])
        visits = other_visits(case["scan_id"], patient_map, folder_labels, metadata)
        seen = patient_map[case["scan_id"]] in seen_subjects

        image, _ = test_ds[index]                         # (3, 224, 224), normalised
        target = 0 if case["predicted"] == "AD" else 1
        cam = grad_cam(models, image.to(device), target)
        # Middle channel = the slice itself (context3: previous/current/next).
        shown = (image[1].cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        panels.append({"image": shown, "cam": cam, "true": case["true"],
                       "predicted": case["predicted"], "text": describe(case, visits, seen)})

        public = {k: case[k] for k in ("kind", "true", "predicted", "scan_prob_ad",
                                       "confidence", "slices", "slices_wrong",
                                       "shown_prob_ad", "sequence", "year")}
        public.update(case=number, subject_in_training_folder=seen, other_visits=visits,
                      label_changes_across_visits=len({v["label"] for v in visits}
                                                      | {case["true"]}) > 1)
        report.append(public)
        case["number"] = number
        print(f"case {number}: {public}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "failure_autopsy_LOCAL_ONLY.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "scan_id", "kind", "true", "predicted", "scan_prob_ad"])
        for c in cases:
            writer.writerow([c["number"], c["scan_id"], c["kind"], c["true"], c["predicted"],
                             f"{c['scan_prob_ad']:.3f}"])
    (args.output_dir / "failure_autopsy.json").write_text(json.dumps(
        {"model": args.model, "ensemble_size": len(models), "split_mode": split_mode,
         "threshold": threshold, "cases": report}, indent=2))
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    plot_autopsy(panels, args.figure)
    print("Saved:", args.figure, "and", args.output_dir)


if __name__ == "__main__":
    main()
