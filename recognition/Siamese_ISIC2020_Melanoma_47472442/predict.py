"""Score a patient group using a trained checkpoint and fixed rules.

Rules are frozen on calibration patients. Output melanoma if classifier
probability reaches its decision threshold; otherwise output review if the
reference score reaches its review threshold; otherwise output benign. The
classifier's own review threshold is also scored for comparison. The
command accepts a checkpoint path and an optional partition, defaulting to
calibration. Predictions, metrics and SVGs go under ``<run>/<partition>/``.

Author: Alex Anderson (47472442), COMP3710 Pattern Recognition, 2026.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .dataset import load_splits, read_samples
from .modules import MelanomaModel, nearest_references
from .utils import (
    accelerator_memory_gib,
    image_loader,
    melanoma_probability,
    peak_process_memory_gib,
    score_images,
    select_device,
    threshold_metrics,
)
from .visualisation import plot_cases, plot_curves, plot_embeddings, plot_pair_distances


@torch.no_grad()
def batch_latency_ms(model: MelanomaModel, config: dict[str, Any], device: torch.device) -> float:
    """Measure mean forward time for a preloaded batch.

    Parameters
    ----------
    model : MelanomaModel
        Model already in evaluation mode and on the selected device.
    config : dict
        Settings containing batch_size and image_size.
    device : torch.device
        Device used for the forward passes and timing synchronisation.

    Returns
    -------
    float
        Mean batch latency in milliseconds over ten passes after three
        warm-up passes. Uses random inputs and excludes image loading.
    """
    size = config["image_size"]
    batch = torch.randn(config["batch_size"], 3, size, size, device=device)
    started = time.perf_counter()
    for repeat in range(13):
        if repeat == 3:
            getattr(torch, device.type).synchronize()  # torch.cuda / torch.mps / torch.cpu
            started = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.bfloat16):
            model(batch)
    getattr(torch, device.type).synchronize()
    return (time.perf_counter() - started) * 100  # 10 timed passes, in milliseconds


def main() -> None:
    """Score a checkpoint and patient group from command-line arguments.

    Print predictions to standard output and summary metrics to standard
    error. Save predictions.csv, metrics.json and SVG figures in the
    checkpoint's partition folder, replacing files with matching names.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--partition", choices=("selection", "calibration", "final"), default="calibration")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config, rule = checkpoint["config"], checkpoint["rule"]
    device = select_device()
    model = MelanomaModel()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    samples = load_splits(read_samples())[args.partition]
    out_dir = args.checkpoint.parent / args.partition
    out_dir.mkdir(exist_ok=True)

    started = time.perf_counter()
    margins, embeddings, labels = score_images(model, image_loader(samples, config), device)
    images_per_second = len(samples) / (time.perf_counter() - started)
    probabilities = melanoma_probability(margins, rule["platt"])
    nearest, reference_scores = nearest_references(
        embeddings, checkpoint["gallery_embeddings"], checkpoint["gallery_labels"], config["neighbours"]
    )
    reference_scores = reference_scores.numpy()
    decisions = np.where(
        probabilities >= rule["decision_threshold"], "melanoma",
        np.where(reference_scores >= rule["reference_review_threshold"], "review", "benign"),
    )

    columns = ("image_name", "patient_id", "actual", "probability", "reference_score", "decision")
    rows = [
        (s.image_name, s.patient_id, s.label, float(p), float(r), d)
        for s, p, r, d in zip(samples, probabilities, reference_scores, decisions, strict=True)
    ]
    with (out_dir / "predictions.csv").open("w", newline="") as stream:
        for output in (stream, sys.stdout):
            csv.writer(output).writerows([columns, *rows])

    # Does uncertainty find wrong decisions? Confidence = distance from the threshold in log-odds;
    # reference disagreement = how far the reference score leans away from the decision.
    flagged = probabilities >= rule["decision_threshold"]
    errors = flagged != labels
    log_odds = rule["platt"][0] * margins + rule["platt"][1]
    decision_log_odds = np.log(rule["decision_threshold"] / (1 - rule["decision_threshold"]))
    # Every unordered image pair: 0 benign-benign, 1 mixed, 2 melanoma-melanoma
    first, second = np.triu_indices(len(labels), 1)
    pair_distances = torch.cdist(embeddings, embeddings).numpy()[first, second]
    pair_types = labels[first] + labels[second]
    melanoma_pairs = pair_types > 0
    metrics = {
        "partition": args.partition,
        "images": len(samples),
        "melanomas": int(labels.sum()),
        "classifier": {
            "auroc": float(roc_auc_score(labels, probabilities)),
            "average_precision": float(average_precision_score(labels, probabilities)),
            "brier": float(brier_score_loss(labels, probabilities)),
        },
        "reference": {
            "auroc": float(roc_auc_score(labels, reference_scores)),
            "average_precision": float(average_precision_score(labels, reference_scores)),
        },
        "decision": threshold_metrics(labels, probabilities, rule["decision_threshold"]),
        "reference_review": threshold_metrics(labels, reference_scores, rule["reference_review_threshold"]),
        "classifier_review": threshold_metrics(labels, probabilities, rule["review_threshold"]),
        "triage": {
            "workload": float((decisions != "benign").mean()),
            "melanoma_capture": float((decisions != "benign")[labels == 1].mean()),
        },
        "embedding_pairs": {
            "benign_benign": float(pair_distances[pair_types == 0].mean()),
            "mixed": float(pair_distances[pair_types == 1].mean()),
            "melanoma_melanoma": float(pair_distances[pair_types == 2].mean()),
            # Chance that a melanoma-melanoma pair is closer than a melanoma-benign pair
            "melanoma_pair_auroc": float(roc_auc_score(
                pair_types[melanoma_pairs] == 2, -pair_distances[melanoma_pairs]
            )),
        },
        "error_detection_auroc": {
            "confidence": float(roc_auc_score(errors, -np.abs(log_odds - decision_log_odds))),
            "reference_disagreement": float(roc_auc_score(errors, np.abs(reference_scores - flagged))),
        },
        "profile": {
            "parameters": sum(p.numel() for p in model.parameters()),
            "batch_latency_ms": batch_latency_ms(model, config, device),
            "images_per_second_end_to_end": images_per_second,
            "peak_gpu_memory_gib": accelerator_memory_gib(device),
            "peak_process_memory_gib": peak_process_memory_gib(),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # Failure cases: the two lowest-scored melanomas and three highest-scored benign images
    missed = [i for i in np.argsort(probabilities) if labels[i] == 1][:2]
    false_alarms = [i for i in np.argsort(-probabilities) if labels[i] == 0][:3]
    all_samples = {s.image_name: s for s in read_samples()}
    gallery = [all_samples[name] for name in checkpoint["gallery_names"]]
    cases = []
    for i in missed + false_alarms:
        reference = gallery[nearest[i]]
        cases.append({
            "image_name": samples[i].image_name, "path": samples[i].path,
            "actual": ("benign", "melanoma")[labels[i]],
            "probability": probabilities[i], "decision": decisions[i], "reference_score": reference_scores[i],
            "reference_name": reference.image_name, "reference_path": reference.path,
            "reference_actual": ("benign", "melanoma")[reference.label],
        })
    title = f"{args.checkpoint.parent.name} · {args.partition}"
    plot_curves(labels, probabilities, reference_scores, rule, out_dir / "curves.svg", title)
    plot_embeddings(
        checkpoint["gallery_embeddings"].numpy(), checkpoint["gallery_labels"].numpy(), embeddings.numpy(), labels,
        out_dir / "embeddings.svg", args.partition,
    )
    plot_pair_distances(pair_distances, pair_types, out_dir / "pair_distances.svg", f"{title} · embedding pairs")
    plot_cases(cases, out_dir / "cases.svg")
    summary = ("classifier", "reference", "decision", "reference_review", "triage", "embedding_pairs")
    print(json.dumps({key: metrics[key] for key in summary}, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
