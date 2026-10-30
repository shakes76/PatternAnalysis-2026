"""Temperature calibration, empirical referral policy and development failures.

Temperature fitting uses calibration labels only; thresholds use policy labels
only. Selection nodes illustrate reliability and errors. Test labels are never
requested. See Guo et al. (2017), https://proceedings.mlr.press/v70/guo17a.html.
"""

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import CLASS_NAMES, DEFAULT_ROOT, load_facebook, sha256_file
from evaluate import PROJECT, load_model, save_plot, write_csv
from train import TrainConfig, configure_reproducibility, source_record

ECE_BINS = 15
TARGET_ERROR = .05
MIN_ACCEPTED = 100
TEMPERATURE_BOUNDS = (.05, 20.)
TEMPERATURE_ITERATIONS = 60
ROLES = ("selection", "calibration", "policy")


def validate_logits(logits, targets=None):
    logits = torch.as_tensor(logits, dtype=torch.float64).detach().cpu()
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("Expected nonempty [nodes, classes] logits")
    if not torch.isfinite(logits).all():
        raise ValueError("Logits must be finite")
    if targets is None:
        return logits
    targets = torch.as_tensor(targets).detach().cpu()
    if (targets.dtype != torch.long or targets.shape != (len(logits),)
            or targets.min() < 0 or targets.max() >= logits.shape[1]):
        raise ValueError("Targets must be int64 class IDs aligned with logits")
    return logits, targets


def probabilities(logits, temperature=1.):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    return (validate_logits(logits) / temperature).softmax(dim=1).numpy()


def fit_temperature(calibration_logits, calibration_targets):
    """Minimise calibration NLL in inverse temperature beta=1/T.

    NLL is convex in beta; its derivative is monotone. Fixed 60-step bisection
    solves it on beta in [0.05, 20], equivalent to T in [0.05, 20]. Endpoints
    handle boundary optima. Include T=1 as a fallback against numerical worsening.
    No optimizer, neural weight updates or model-selection labels are involved.
    """
    logits, targets = validate_logits(calibration_logits, calibration_targets)
    lower, upper = TEMPERATURE_BOUNDS
    true_logits = logits[torch.arange(len(targets)), targets]

    def derivative(beta):
        return float(((logits * (logits * beta).softmax(dim=1)).sum(1) - true_logits).mean())

    def nll(beta):
        return float(torch.nn.functional.cross_entropy(logits * beta, targets))

    left, right = 1 / upper, 1 / lower
    if derivative(left) >= 0:
        beta = left
    elif derivative(right) <= 0:
        beta = right
    else:
        for _ in range(TEMPERATURE_ITERATIONS):
            middle = (left + right) / 2
            if derivative(middle) < 0:
                left = middle
            else:
                right = middle
        beta = (left + right) / 2
    beta = min((1., beta), key=nll)
    temperature = 1 / beta
    return {
        "temperature": temperature, "calibration_nodes": len(targets),
        "nll_before": nll(1.), "nll_after": nll(beta),
        "temperature_bounds": list(TEMPERATURE_BOUNDS),
        "at_boundary": bool(np.isclose(temperature, lower) or np.isclose(temperature, upper)),
        "method": "bounded inverse-temperature NLL derivative bisection, 60 iterations",
    }


def reliability_metrics(logits, targets, temperature=1., bins=ECE_BINS):
    """Top-label ECE, NLL and summed multiclass Brier score (range 0..2).

    ECE bins are [i/B, (i+1)/B), last bin includes 1; empty bins have zero weight.
    NLL is evaluated from log-softmax rather than clipped probabilities.
    """
    logits, targets = validate_logits(logits, targets)
    if bins < 1:
        raise ValueError("Need at least one calibration bin")
    probs = probabilities(logits, temperature)
    targets = targets.numpy()
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == targets
    memberships = np.minimum((confidence * bins).astype(int), bins - 1)
    records = []
    ece = 0.
    for index in range(bins):
        selected = memberships == index
        count = int(selected.sum())
        mean = float(confidence[selected].mean()) if count else None
        accuracy = float(correct[selected].mean()) if count else None
        if count:
            ece += count / len(targets) * abs(mean - accuracy)
        records.append({"lower": index / bins, "upper": (index + 1) / bins,
                        "count": count, "confidence": mean, "accuracy": accuracy})
    groups = {}
    for name, mask in (("correct", correct), ("incorrect", ~correct)):
        groups[name] = {"count": int(mask.sum()),
                        "mean_confidence": float(confidence[mask].mean()) if mask.any() else None,
                        "median_confidence": float(np.median(confidence[mask])) if mask.any() else None}
    return {
        "nodes": len(targets), "accuracy": float(correct.mean()), "ece": float(ece),
        "nll": float(torch.nn.functional.cross_entropy(logits / temperature, torch.from_numpy(targets))),
        "brier": float(np.square(probs - np.eye(probs.shape[1])[targets]).sum(axis=1).mean()),
        "mean_confidence": float(confidence.mean()), "confidence_groups": groups,
        "bins": records,
    }


def risk_curve(probs, targets):
    """All attainable >= thresholds, treating tied confidences as one group."""
    probs = np.asarray(probs)
    targets = np.asarray(targets)
    if probs.ndim != 2 or len(probs) == 0 or targets.shape != (len(probs),):
        raise ValueError("Expected aligned nonempty probabilities and targets")
    if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(1), 1):
        raise ValueError("Invalid probabilities")
    confidence = probs.max(axis=1)
    order = np.argsort(-confidence, kind="stable")
    sorted_confidence = confidence[order]
    errors = np.cumsum(probs.argmax(1)[order] != targets[order])
    # Only the end of a tie group is attainable with a single >= threshold.
    ends = np.r_[np.flatnonzero(sorted_confidence[:-1] != sorted_confidence[1:]), len(probs) - 1]
    return [{"threshold": float(sorted_confidence[i]), "accepted": int(i + 1),
             "errors": int(errors[i]), "coverage": float((i + 1) / len(probs)),
             "referral_rate": float(1 - (i + 1) / len(probs)),
             "accepted_error": float(errors[i] / (i + 1))} for i in ends]


def choose_threshold(policy_probabilities, policy_targets,
                     target_error=TARGET_ERROR, min_accepted=MIN_ACCEPTED):
    if not 0 <= target_error < 1 or min_accepted < 1:
        raise ValueError("Invalid risk objective or minimum count")
    candidates = risk_curve(policy_probabilities, policy_targets)
    feasible = [row for row in candidates if row["accepted"] >= min_accepted
                and row["accepted_error"] <= target_error]
    chosen = max(feasible, key=lambda row: row["accepted"]) if feasible else {
        "threshold": None, "accepted": 0, "errors": 0, "coverage": 0.,
        "referral_rate": 1., "accepted_error": None,
    }
    return {**chosen, "feasible": bool(feasible), "target_error": target_error,
            "min_accepted": min_accepted, "selection_nodes": len(policy_targets)}


def acceptance_mask(probs, threshold):
    return (np.zeros(len(probs), dtype=bool) if threshold is None
            else np.asarray(probs).max(axis=1) >= threshold)


def apply_policy(probs, targets, threshold):
    accepted = acceptance_mask(probs, threshold)
    errors = np.asarray(probs).argmax(axis=1) != np.asarray(targets)
    count = int(accepted.sum())
    return {"nodes": len(probs), "accepted": count, "errors": int(errors[accepted].sum()),
            "coverage": float(accepted.mean()), "referral_rate": float(1 - accepted.mean()),
            "accepted_error": float(errors[accepted].mean()) if count else None,
            "total_errors": int(errors.sum()),
            "error_referral_fraction": float((errors & ~accepted).sum() / errors.sum()) if errors.any() else None}


@torch.inference_mode()
def collect_logits(dataset, checkpoint_path, name, device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model, _ = load_model(checkpoint_path, dataset, device, name)
    features = dataset.graph.x.to(device)
    logits = (model(features, dataset.graph.edge_index.to(device))
              if name == "GCN" else model(features))
    return logits.cpu().double().numpy()


def failure_cases(dataset, selection_logits, temperatures, thresholds, limit=5,
                  *, partition="selection", allow_test=False):
    """Deterministic illustrative errors; inspect train-labelled neighbors only.

    Cases intentionally illustrate confident errors, uncertain errors, and model
    disagreements. They are not a random sample or an estimate of error prevalence.
    """
    if partition not in ("selection", "test"):
        raise ValueError("Failure analysis is restricted to selection or final test")
    if partition == "test" and not allow_test:
        raise PermissionError("Final test failure analysis requires explicit release")
    ids = np.flatnonzero(dataset.masks[partition].numpy())
    truth = dataset.targets(partition, allow_test=allow_test).numpy()
    raw = {name: probabilities(logits) for name, logits in selection_logits.items()}
    scaled = {name: probabilities(logits, temperatures[name]) for name, logits in selection_logits.items()}
    predictions = {name: probs.argmax(1) for name, probs in scaled.items()}
    confidences = {name: probs.max(1) for name, probs in scaled.items()}
    wrong = {name: predictions[name] != truth for name in predictions}
    descending = {name: np.lexsort((ids, -confidence)) for name, confidence in confidences.items()}
    ascending_gcn = np.lexsort((ids, confidences["GCN"]))
    chosen = []

    def add(order, eligible, reason):
        for index in order:
            if eligible[index] and index not in [row[0] for row in chosen]:
                chosen.append((int(index), reason))
                break

    add(descending["GCN"], wrong["GCN"], "highest-confidence GCN error")
    add(ascending_gcn, wrong["GCN"], "lowest-confidence GCN error")
    add(descending["GCN"], wrong["GCN"] & ~wrong["MLP"], "GCN wrong, MLP correct")
    add(descending["MLP"], wrong["MLP"] & ~wrong["GCN"], "MLP wrong, GCN correct")
    while len(chosen) < limit:
        before = len(chosen)
        add(descending["GCN"], wrong["GCN"], "next confident GCN error")
        if len(chosen) == before:
            break
    train_labels = np.full(dataset.graph.num_nodes, -1, dtype=int)
    train_labels[dataset.masks["train"].numpy()] = dataset.targets("train").numpy()
    edges = dataset.graph.edge_index.numpy()
    cases = []
    for index, reason in chosen[:limit]:
        node = int(ids[index])
        neighbors = np.unique(edges[1, edges[0] == node])
        neighbors = neighbors[neighbors != node]
        known = train_labels[neighbors]
        counts = np.bincount(known[known >= 0], minlength=len(CLASS_NAMES))
        row = {"node_id": node, "partition": partition, "reason": reason,
               "true_class": CLASS_NAMES[truth[index]], "true_class_id": int(truth[index]),
               "unique_nonself_degree": len(neighbors),
               "training_neighbor_class_counts": dict(zip(CLASS_NAMES, map(int, counts))),
               "training_labeled_neighbors": int(counts.sum()),
               "neighbors_without_used_labels": int(len(neighbors) - counts.sum())}
        for name in ("MLP", "GCN"):
            row[name] = {"predicted_class": CLASS_NAMES[predictions[name][index]],
                         "predicted_class_id": int(predictions[name][index]),
                         "raw_confidence": float(raw[name][index].max()),
                         "calibrated_confidence": float(confidences[name][index]),
                         "referred": bool(not acceptance_mask(scaled[name][index:index+1], thresholds[name])[0])}
        cases.append(row)
    return cases


def plot_reliability(results, stem, partition="selection"):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for axis, name in zip(axes, ("MLP", "GCN")):
        axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect reliability")
        for stage, label in (("before", "Raw"), ("after", "Temperature scaled")):
            bins = [row for row in results[name]["metrics"][partition][stage]["bins"] if row["count"]]
            axis.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins],
                      marker="o", markersize=4, label=label)
        axis.set(title=name, xlabel="Mean maximum softmax probability", xlim=(0, 1), ylim=(0, 1))
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Observed accuracy")
    figure.suptitle(f"{partition.title()} reliability: 15 equal-width confidence bins")
    figure.tight_layout()
    save_plot(figure, stem)


def plot_distributions(selection_logits, targets, temperatures, stem, partition="selection"):
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for row, name in enumerate(("MLP", "GCN")):
        for column, temperature in enumerate((1., temperatures[name])):
            probs = probabilities(selection_logits[name], temperature)
            correct = probs.argmax(1) == targets
            for flag, label in ((True, "Correct"), (False, "Incorrect")):
                values = probs.max(1)[correct == flag]
                if len(values):
                    # Density normalises each group separately despite unequal counts.
                    axes[row, column].hist(values, bins=np.linspace(0, 1, 21), density=True,
                                           histtype="step", linewidth=2, label=f"{label} (n={len(values)})")
            axes[row, column].set(title=f"{name}: {'raw' if column == 0 else 'temperature scaled'}",
                                  xlabel="Maximum softmax probability", ylabel="Within-group density")
            axes[row, column].legend(fontsize=8)
    figure.suptitle(f"{partition.title()} confidence: correct versus incorrect predictions")
    figure.tight_layout()
    save_plot(figure, stem)


def plot_risk(curves, results, stem, partition="policy"):
    figure, axis = plt.subplots(figsize=(7, 5))
    for name in ("MLP", "GCN"):
        curve = curves[name]
        line, = axis.plot([r["coverage"] for r in curve], [r["accepted_error"] for r in curve], label=name)
        chosen = results[name]["policy"]
        if chosen["feasible"]:
            axis.scatter(chosen["coverage"], chosen["accepted_error"], s=65, color=line.get_color(), marker="x")
    axis.axhline(TARGET_ERROR, color="gray", linestyle="--", label="5% observed-error objective")
    axis.set(xlabel="Accepted fraction (coverage)", ylabel="Observed accepted-case error",
             title=f"{partition.title()} partition: calibrated-confidence risk versus coverage", xlim=(0, 1))
    axis.legend()
    axis.grid(alpha=.2)
    figure.tight_layout()
    save_plot(figure, stem)


def plot_neighbors(cases, stem, partition="selection"):
    figure, axis = plt.subplots(figsize=(9, 4))
    bottom = np.zeros(len(cases))
    for label in (*CLASS_NAMES, "Labels not used"):
        counts = np.array([case["training_neighbor_class_counts"][label] if label in CLASS_NAMES
                           else case["neighbors_without_used_labels"] for case in cases])
        axis.bar([str(c["node_id"]) for c in cases], counts, bottom=bottom, label=label)
        bottom += counts
    axis.set(xlabel=f"{partition.title()} node ID", ylabel="Unique non-self neighbors",
             title="Failure cases: graph degree and training-labelled neighborhood")
    axis.legend(fontsize=8)
    figure.tight_layout()
    save_plot(figure, stem)


def run_analysis(dataset, mlp_checkpoint, gcn_checkpoint, output_dir, device="auto"):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Choose a fresh output directory")
    device = configure_reproducibility(TrainConfig(device=device))
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {role: dataset.targets(role).numpy() for role in ROLES}
    masks = {role: dataset.masks[role].numpy() for role in ROLES}
    results, selection_logits, temperatures, thresholds, curves, raw = {}, {}, {}, {}, {}, {}
    for name, checkpoint in (("MLP", mlp_checkpoint), ("GCN", gcn_checkpoint)):
        logits = collect_logits(dataset, checkpoint, name, device)
        role_logits = {role: logits[mask] for role, mask in masks.items()}
        fit = fit_temperature(role_logits["calibration"], targets["calibration"])
        temperature = temperatures[name] = fit["temperature"]
        metrics = {role: {"before": reliability_metrics(values, targets[role]),
                          "after": reliability_metrics(values, targets[role], temperature)}
                   for role, values in role_logits.items()}
        policy_probs = probabilities(role_logits["policy"], temperature)
        chosen = choose_threshold(policy_probs, targets["policy"])
        thresholds[name] = chosen["threshold"]
        curves[name] = risk_curve(policy_probs, targets["policy"])
        selection_logits[name] = role_logits["selection"]
        # Applying the already chosen policy here does not refit it.
        selection_policy = apply_policy(probabilities(role_logits["selection"], temperature),
                                        targets["selection"], chosen["threshold"])
        results[name] = {"checkpoint_sha256": sha256_file(Path(checkpoint)), "temperature_fit": fit,
                         "metrics": metrics, "policy": chosen, "selection_policy": selection_policy}
        for role, values in role_logits.items():
            raw[name + "_" + role + "_logits"] = values
        print(f"{name}: T={temperature:.6f}, policy={chosen}", flush=True)
    cases = failure_cases(dataset, selection_logits, temperatures, thresholds)
    plot_reliability(results, output_dir / "reliability_selection")
    plot_distributions(selection_logits, targets["selection"], temperatures, output_dir / "confidence_selection")
    plot_risk(curves, results, output_dir / "risk_coverage_policy")
    plot_neighbors(cases, output_dir / "failure_neighbors_selection")
    policy = {
        "schema_version": 1, "test_evaluated": False,
        "temperature_fitting_partition": "calibration", "threshold_selection_partition": "policy",
        "target_observed_error": TARGET_ERROR, "minimum_accepted": MIN_ACCEPTED,
        "confidence_rule": "accept iff max softmax(logits / T) >= threshold; null threshold refers all",
        "threshold_search": "maximum coverage over unique observed policy confidences; include entire ties",
        "ece_bins": ECE_BINS, "temperature_bounds": list(TEMPERATURE_BOUNDS),
        "data_sha256": dataset.provenance["sha256"],
        "split_mask_sha256": {name: row["mask_sha256"] for name, row in dataset.audit["splits"].items()},
        "models": {name: {"checkpoint_sha256": results[name]["checkpoint_sha256"],
                          "temperature": temperatures[name], "threshold": thresholds[name]}
                   for name in results},
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "device": str(device),
        "source": {**source_record(), "confidence_py_sha256": sha256_file(Path(__file__))},
        "evaluated_partitions": list(ROLES), "test_evaluated": False,
        "models": results, "failure_cases": cases,
        "limitations": "Policy error is adaptively selected empirical error, not a bound on future risk. Neighbor labels use training nodes only. Selection cases are illustrative, not representative.",
    }
    for name, value in (("confidence_summary.json", report), ("referral_policy.json", policy)):
        (output_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    comparison = []
    for name, result in results.items():
        for role in ROLES:
            for stage in ("before", "after"):
                metrics = result["metrics"][role][stage]
                comparison.append({"model": name, "partition": role, "calibration": stage,
                                   **{key: metrics[key] for key in ("nodes", "accuracy", "ece", "nll", "brier", "mean_confidence")}})
    write_csv(output_dir / "calibration_comparison.csv", comparison)
    write_csv(output_dir / "failure_cases.csv", [
        {"node_id": c["node_id"], "true_class": c["true_class"], "degree": c["unique_nonself_degree"],
         **{name.lower() + "_" + key: value for name in ("MLP", "GCN") for key, value in c[name].items()}}
        for c in cases])
    np.savez_compressed(output_dir / "development_logits.npz", **raw)
    write_csv(output_dir / "risk_coverage.csv", [{"model": name, **row} for name, rows in curves.items() for row in rows])
    return report, policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--mlp-checkpoint", type=Path, default=PROJECT / "outputs/mlp_seed42/best.pt")
    parser.add_argument("--gcn-checkpoint", type=Path, default=PROJECT / "outputs/gcn_seed42/best.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/confidence_seed42")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda:0"), default="auto")
    args = parser.parse_args()
    run_analysis(load_facebook(args.data_root), args.mlp_checkpoint, args.gcn_checkpoint,
                 args.output_dir, args.device)


if __name__ == "__main__":
    main()
