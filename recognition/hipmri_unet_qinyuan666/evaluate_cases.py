"""Evaluate a fixed validation or test split, pooling slices within each Case.

Validation is the default. Test evaluation requires explicit --split test after
checkpoint selection is frozen. No fitting or model selection is performed.
This does not estimate physical boundaries. Public aliases omit source Case IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch
from torch.utils.data import DataLoader

from dataset import HipMRIDataset
from metrics import confusion_matrix, scores_from_confusion
from modules import build_model
from predict import _device, _validate_checkpoint


def summarize_cases(matrices: dict, slice_counts: dict, prostate_channel: int) -> dict:
    """Pool pixels within Case first, then give each defined Case equal weight."""
    if not matrices or set(matrices) != set(slice_counts):
        raise ValueError("Need matching nonempty Case matrices and slice counts")
    first = next(iter(matrices.values()))
    if first.ndim != 2 or not 0 < prostate_channel < first.shape[0]:
        raise ValueError("Invalid prostate channel")
    pooled = torch.zeros_like(first)
    cases = []
    for number, case in enumerate(sorted(matrices), 1):
        matrix = matrices[case]
        if matrix.shape != first.shape or slice_counts[case] < 1:
            raise ValueError("Case matrices must have matching shapes and positive slice counts")
        scores = scores_from_confusion(matrix)
        pooled += matrix
        cases.append({"case_alias": f"Case {number:02d}", "slices": slice_counts[case], **scores})
    macro = []
    for channel in range(first.shape[0]):
        entry = {"class_index": channel}
        for metric in ("dice", "iou"):
            values = [case["per_class"][channel][metric] for case in cases
                      if case["per_class"][channel][metric] is not None]
            entry[f"mean_{metric}"] = statistics.mean(values) if values else None
            entry[f"defined_{metric}_cases"] = len(values)
        macro.append(entry)
    prostate = [case["per_class"][prostate_channel]["dice"] for case in cases
                if case["per_class"][prostate_channel]["dice"] is not None]
    return {
        "case_count": len(cases), "slice_count": sum(slice_counts.values()),
        "cases": cases, "pixel_pooled": scores_from_confusion(pooled),
        "case_macro_per_class": macro,
        "prostate_case_dice": {
            "mean": statistics.mean(prostate) if prostate else None,
            "median": statistics.median(prostate) if prostate else None,
            "std_population": statistics.pstdev(prostate) if prostate else None,
            "min": min(prostate) if prostate else None,
            "max": max(prostate) if prostate else None,
            "defined_cases": len(prostate), "undefined_cases": len(cases) - len(prostate),
        },
    }


def evaluate(checkpoint: Path, manifest: Path, output: Path,
             device: str = "auto", batch_size: int = 4, *, split: str = "val") -> dict:
    if split not in ("val", "test"):
        raise ValueError("split must be val or test")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    checkpoint, manifest, output = Path(checkpoint), Path(manifest), Path(output)
    if output.exists():
        raise FileExistsError("Choose a new output directory to preserve prior evaluation")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config, channel = _validate_checkpoint(payload, synthetic=False)
    manifest_hash = sha(manifest)
    if manifest_hash != config.get("manifest_sha256"):
        raise ValueError("Evaluation manifest does not match checkpoint manifest_sha256")
    selected_device = _device(device)
    data = HipMRIDataset(manifest, split, config["image_size"], config["label_values"],
                        augment=False, resize_mode=config["resize_mode"])
    model = build_model(config["model"], num_classes=config["num_classes"],
                        base_channels=config["base_channels"]).to(selected_device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    matrices, counts = {}, {}
    with torch.inference_mode():
        for batch in DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0):
            logits = model(batch["image"].to(selected_device))
            if not torch.isfinite(logits).all():
                raise ValueError("Checkpoint produced non-finite logits")
            predictions = logits.argmax(1).cpu()
            for case, prediction, target in zip(batch["patient_id"], predictions, batch["mask"]):
                matrix = confusion_matrix(prediction, target, config["num_classes"])
                if case not in matrices:
                    matrices[case] = torch.zeros_like(matrix)
                    counts[case] = 0
                matrices[case] += matrix
                counts[case] += 1
    result = summarize_cases(matrices, counts, channel)
    result.update({
        "model": config["model"], "epoch": payload["epoch"], "split": split,
        "label_values": config["label_values"], "prostate_raw_label": config["prostate_label"],
        "prostate_class_index": channel, "image_size": config["image_size"],
        "resize_mode": config["resize_mode"], "batch_size": batch_size,
        "device": str(selected_device), "test_evaluated": split == "test",
        "manifest_sha256": manifest_hash, "checkpoint_sha256": sha(checkpoint),
        "evaluator_sha256": sha(Path(__file__)),
        "dependency_sha256": {name: sha(Path(__file__).parent / name) for name in
                              ("dataset.py", "modules.py", "metrics.py", "predict.py")},
        "aggregation": "Sum confusion matrices across every slice and week of each manifest patient_id (Case), then average defined Case scores equally.",
        "empty_policy": "Both truth and prediction empty: null, excluded from mean with count reported. One empty: zero.",
        "limitations": [
            "Case grouping follows the adopted pseudonymous dataset convention.",
            "Weeks are pooled within each Case; not a per-scan average or reconstructed 3D volume metric.",
            "Metrics include the padded input grid; not physical-space boundary measurements.",
            ("Validation used for model development; not final test evidence or a clinical claim."
             if split == "val" else
             "Held-out test evaluation of a previously selected checkpoint; few Cases and one seed, not a clinical claim."),
        ],
    })
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    private = {f"Case {i:02d}": {"patient_id": case, "confusion_matrix": matrices[case].tolist()}
               for i, case in enumerate(sorted(matrices), 1)}
    (output / "case_details_private.json").write_text(json.dumps(private, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New private directory, usually under runs/")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--split", choices=("val", "test"), default="val",
                        help="Use test only after freezing checkpoints and evaluation protocol")
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = evaluate(**vars(args))
    print(json.dumps({"cases": result["case_count"], "slices": result["slice_count"],
                      "prostate_case_dice": result["prostate_case_dice"]}, indent=2))


if __name__ == "__main__":
    main()
