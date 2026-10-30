"""Explicitly guarded evaluation of frozen checkpoints on the final test split.

Freeze a hash manifest BEFORE releasing test labels. Reuse all existing metric,
probability and decision functions. Never train, calibrate or search thresholds.
"""

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from confidence import (apply_policy, failure_cases, plot_distributions, plot_neighbors,
                        plot_reliability, plot_risk, probabilities, reliability_metrics,
                        risk_curve)
from dataset import DEFAULT_ROOT, load_facebook, sha256_file
from evaluate import (PROJECT, benchmark_forward, classification_metrics, load_model,
                      plot_confusion, write_csv)
from predict import DEFAULT_POLICY, read_policy
from train import TrainConfig, configure_reproducibility

FROZEN_SOURCES = ("dataset.py", "modules.py", "train.py", "evaluate.py",
                  "confidence.py", "predict.py", "final_evaluate.py")
DEFAULT_MANIFEST = PROJECT / "outputs/frozen_evaluation_manifest.json"


def artifact_record(dataset, checkpoints, policy_path):
    """Hash and validate artifacts without requesting any target labels."""
    configs = {}
    for name, path in checkpoints.items():
        read_policy(policy_path, path, dataset, name)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        configs[name] = {"checkpoint_sha256": sha256_file(path),
                         "model_config": checkpoint["model_config"],
                         "training_config": checkpoint["training_config"],
                         "selected_epoch": checkpoint["epoch"]}
    return {
        "models": configs, "policy_sha256": sha256_file(policy_path),
        "frozen_policy": json.loads(policy_path.read_text()),
        "data_sha256": dataset.provenance["sha256"],
        "split_mask_sha256": {name: row["mask_sha256"] for name, row in dataset.audit["splits"].items()},
        "source_sha256": {name: sha256_file(PROJECT / name) for name in FROZEN_SOURCES},
        "protocol": {
            "classification": "argmax; per-class precision/recall/F1, macro-F1, macro OvR AUROC",
            "probabilities": "raw and saved-temperature softmax on CPU float64",
            "calibration": "existing ECE 15 equal-width bins, stable NLL, summed multiclass Brier",
            "referral": "existing >= saved threshold, whole ties; no search/refitting",
            "latency": "full graph; first call, 20 warmup then 100 synchronized calls",
            "failures": "same five-case ordering as Stage 6; training-labelled neighbors only",
        },
    }


def freeze_manifest(dataset, checkpoints, policy_path, destination):
    destination = Path(destination)
    record = artifact_record(dataset, checkpoints, Path(policy_path))
    record.update(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
                  test_labels_released=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    return record


def verify_manifest(dataset, checkpoints, policy_path, manifest_path):
    saved = json.loads(Path(manifest_path).read_text())
    actual = artifact_record(dataset, checkpoints, Path(policy_path))
    for key, value in actual.items():
        if saved.get(key) != value:
            raise ValueError(f"Frozen manifest mismatch: {key}; evaluation refused")
    return saved


@torch.inference_mode()
def frozen_forward(dataset, checkpoint_path, name, device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model, _ = load_model(checkpoint_path, dataset, device, name)
    x = dataset.graph.x.to(device)
    edge = dataset.graph.edge_index.to(device) if name == "GCN" else None

    def forward():
        return model(x) if edge is None else model(x, edge)

    timing, samples = benchmark_forward(forward, device)
    logits = forward().cpu().double().numpy()
    resources = {**timing, "parameter_count": sum(p.numel() for p in model.parameters()),
                 "inference_nodes": dataset.graph.num_nodes,
                 "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
                 "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == "cuda" else None}
    return logits, resources, samples


def run_final(dataset, checkpoints, policy_path, manifest_path, output_dir,
              device="auto", *, release_test_labels=False, partition="test"):
    if partition not in ("test", "selection"):
        raise ValueError("Only final test or development smoke evaluation is supported")
    if partition == "test" and not release_test_labels:
        raise PermissionError("Final test labels require --release-test-labels")
    manifest = verify_manifest(dataset, checkpoints, policy_path, manifest_path)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Choose a fresh final-evaluation output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    chosen_device = configure_reproducibility(TrainConfig(device=device))
    # Durable audit written BEFORE the explicitly guarded target request.
    release = {"partition": partition, "released_at_utc": datetime.now(timezone.utc).isoformat(),
               "manifest_sha256": sha256_file(Path(manifest_path)),
               "manifest_created_at_utc": manifest["created_at_utc"]}
    (output_dir / "label_release_audit.json").write_text(json.dumps(release, indent=2) + "\n")
    targets = dataset.targets(partition, allow_test=release_test_labels).numpy()
    mask = dataset.masks[partition].numpy()
    settings = manifest["frozen_policy"]["models"]
    results, logits_by_model, curves, raw, comparison, per_class = {}, {}, {}, {}, [], []
    for name, checkpoint in checkpoints.items():
        all_logits, resources, samples = frozen_forward(dataset, checkpoint, name, chosen_device)
        logits = logits_by_model[name] = all_logits[mask]
        temperature = settings[name]["temperature"]
        threshold = settings[name]["threshold"]
        raw_probs, scaled_probs = probabilities(logits), probabilities(logits, temperature)
        classes = {"raw": classification_metrics(targets, raw_probs),
                   "scaled": classification_metrics(targets, scaled_probs)}
        reliability = {"before": reliability_metrics(logits, targets),
                       "after": reliability_metrics(logits, targets, temperature)}
        referral = apply_policy(scaled_probs, targets, threshold)
        results[name] = {"classification": classes, "metrics": {partition: reliability},
                         "referral": referral, "temperature": temperature, "threshold": threshold,
                         "resources": resources,
                         "policy": {**referral, "feasible": referral["accepted"] > 0}}
        curves[name] = risk_curve(scaled_probs, targets)  # diagnostic only; never used to choose a rule
        plot_confusion(classes["raw"], name, output_dir / f"{name.lower()}_confusion_{partition}", partition)
        raw[name + "_logits"] = logits
        raw[name + "_latency_ms"] = np.asarray(samples)
        for scale, stats in classes.items():
            comparison.append({"model": name, "probabilities": scale,
                               **{key: stats[key] for key in ("nodes", "accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_ovr_auroc")}})
            per_class.extend({"model": name, "probabilities": scale, **row} for row in stats["per_class"])
        print(f"{name} {partition}: accuracy={classes['raw']['accuracy']:.6f}, "
              f"F1={classes['raw']['macro_f1']:.6f}, referral={referral}", flush=True)
    cases = failure_cases(dataset, logits_by_model,
                          {name: settings[name]["temperature"] for name in settings},
                          {name: settings[name]["threshold"] for name in settings},
                          partition=partition, allow_test=release_test_labels)
    plot_reliability(results, output_dir / f"reliability_{partition}", partition)
    plot_distributions(logits_by_model, targets,
                       {name: settings[name]["temperature"] for name in settings},
                       output_dir / f"confidence_{partition}", partition)
    plot_risk(curves, results, output_dir / f"risk_coverage_{partition}", partition)
    plot_neighbors(cases, output_dir / f"failure_neighbors_{partition}", partition)
    verify_manifest(dataset, checkpoints, policy_path, manifest_path)
    report = {"partition": partition, "label_release": release, "models": results,
              "failure_cases": cases, "device": str(chosen_device),
              "device_name": torch.cuda.get_device_name(chosen_device) if chosen_device.type == "cuda" else "CPU",
              "manifest_sha256": release["manifest_sha256"],
              "resource_scope": "full graph inference only; setup through final forward, no metric tensors on GPU",
              "no_fitting_or_threshold_search": True}
    (output_dir / "final_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(output_dir / "node_outputs.npz", node_ids=np.flatnonzero(mask), **raw)
    write_csv(output_dir / "classification.csv", comparison)
    write_csv(output_dir / "per_class.csv", per_class)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--mlp-checkpoint", type=Path, default=PROJECT / "outputs/mlp_seed42/best.pt")
    parser.add_argument("--gcn-checkpoint", type=Path, default=PROJECT / "outputs/gcn_seed42/best.pt")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/final_test_seed42")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda:0"), default="auto")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-only", action="store_true")
    group.add_argument("--release-test-labels", action="store_true")
    args = parser.parse_args()
    dataset = load_facebook(args.data_root)
    checkpoints = {"MLP": args.mlp_checkpoint, "GCN": args.gcn_checkpoint}
    if args.freeze_only:
        freeze_manifest(dataset, checkpoints, args.policy, args.manifest)
        print(f"Frozen manifest written to {args.manifest}; no test labels released")
    else:
        run_final(dataset, checkpoints, args.policy, args.manifest, args.output_dir,
                  args.device, release_test_labels=True)


if __name__ == "__main__":
    main()
