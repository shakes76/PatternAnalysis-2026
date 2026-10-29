"""Precommitted within-graph robustness study with five disjoint label roles.

This uses a previously studied graph, not new external data. Final-role labels
are withheld from this study's fitting and all policies are saved before their
evaluation. Repeating a split does not restore project-level independence.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from dataset import (CLASS_NAMES, DATA_SHA256, DEFAULT_PATH, canonical_edges,
                     load_arrays, sha256, standardize_features)
from evaluation import (choose_threshold, classification_metrics, fit_temperature,
                        plot_history, selective_metrics)
from modules import make_model, normalized_adjacency

HERE = Path(__file__).resolve().parent
SEEDS = [42, 43, 44]
SPLIT_SEED = 20261026
ROLES = ["train", "selection", "temperature", "policy", "final"]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_sha(path):
    return sha256(path)


def role_split(y):
    """Stratify one fixed split without searching over seeds or final scores."""
    random = np.random.default_rng(SPLIT_SEED)
    pieces = {name: [] for name in ROLES}
    for label in range(len(CLASS_NAMES)):
        ids = random.permutation(np.flatnonzero(y == label))
        sizes = [int(.60 * len(ids)), *[int(.10 * len(ids))] * 3]
        sizes.append(len(ids) - sum(sizes))
        for name, group in zip(ROLES, np.split(ids, np.cumsum(sizes)[:-1])):
            pieces[name].append(group)
    splits = {name: np.sort(np.concatenate(groups)) for name, groups in pieces.items()}
    assert np.array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(len(y)))
    return splits


def fit_class_thresholds(probabilities, labels, target=.98, minimum=50):
    """Fit by predicted class; never select a route using a case's true class."""
    confidence, prediction = probabilities.max(1)
    thresholds, details = [], []
    for label, name in enumerate(CLASS_NAMES):
        group = prediction == label
        required = max(minimum, math.ceil(.05 * int(group.sum())))
        candidates = []
        for index in range(101):
            threshold = index / 100
            chosen = group & (confidence >= threshold)
            count = int(chosen.sum())
            if count >= required:
                accuracy = float((prediction[chosen] == labels[chosen]).double().mean())
                if accuracy >= target:
                    candidates.append((count, -threshold, accuracy))
        if candidates:
            count, minus_threshold, accuracy = max(candidates)
            threshold = -minus_threshold
        else:
            count, accuracy, threshold = 0, None, 1.01
        thresholds.append(threshold)
        details.append({"predicted_class": name, "policy_cases": int(group.sum()),
                        "minimum_accepted": required, "accepted": count,
                        "accepted_accuracy": accuracy, "threshold": threshold,
                        "feasible_on_fitting_data": bool(candidates)})
    return thresholds, details


def class_decisions(probabilities, thresholds):
    """Return label-free acceptance and class prediction from four cutoffs."""
    if len(thresholds) != len(CLASS_NAMES) or any(not math.isfinite(t) or t < 0 for t in thresholds):
        raise ValueError("Four finite nonnegative thresholds required")
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError("Expected four-class probabilities")
    if not torch.isfinite(probabilities).all() or bool((probabilities < 0).any()) or bool((probabilities > 1).any()):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if not torch.allclose(probabilities.sum(1), torch.ones(len(probabilities), device=probabilities.device), atol=1e-5, rtol=1e-5):
        raise ValueError("Probability rows must sum to one")
    confidence, prediction = probabilities.max(1)
    cutoffs = probabilities.new_tensor(thresholds)[prediction]
    return confidence >= cutoffs


def policy_metrics(probabilities, labels, accepted):
    prediction = probabilities.argmax(1)
    correct = prediction == labels
    total, count = len(labels), int(accepted.sum())
    result = {"total": total, "accepted": count, "referred": total - count,
              "coverage": count / total,
              "accepted_accuracy": float(correct[accepted].double().mean()) if count else None,
              "accepted_errors": int((~correct & accepted).sum()),
              "referred_errors": int((~correct & ~accepted).sum()),
              "whole_population_accuracy": float(correct.double().mean())}
    for kind, groups in [("predicted_class", prediction), ("true_class", labels)]:
        result[kind] = {}
        for label, name in enumerate(CLASS_NAMES):
            group = groups == label
            chosen = group & accepted
            n = int(chosen.sum())
            result[kind][name] = {"total": int(group.sum()), "accepted": n,
                                  "accepted_accuracy": float(correct[chosen].double().mean()) if n else None,
                                  "accepted_errors": int((~correct & chosen).sum())}
    return result


def train_model(name, seed, x, y, masks, adjacency, destination):
    torch.manual_seed(seed)
    config = {"inputs": x.shape[1], "hidden": 64, "classes": 4, "dropout": .5}
    model = make_model(name, **config)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.0005)
    graph_matrix = adjacency if name == "gcn" else None
    best_score, best_epoch, best_state = -1., 0, None
    history = []
    started = time.perf_counter()
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(x, graph_matrix)
        loss = F.cross_entropy(logits[masks["train"]], y[masks["train"]])
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise ValueError("Nonfinite gradient")
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits, _ = model(x, graph_matrix)
        selection = classification_metrics(logits[masks["selection"]], y[masks["selection"]])
        history.append({"epoch": epoch, "train_loss": float(loss.detach()),
                        "selection_loss": selection["negative_log_likelihood"],
                        "selection_macro_f1": selection["macro_f1"]})
        if selection["macro_f1"] > best_score:
            best_score, best_epoch = selection["macro_f1"], epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 50 == 0:
            print(f"{name} seed={seed} epoch={epoch} selection_macro_f1={selection['macro_f1']:.4f}", flush=True)
        if epoch - best_epoch >= 30:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, _ = model(x, graph_matrix)
    temperature = fit_temperature(logits[masks["temperature"]], y[masks["temperature"]])
    probabilities = (logits / temperature).softmax(1)
    global97 = choose_threshold(logits[masks["policy"]], y[masks["policy"]], temperature, .97)
    thresholds, fitting = fit_class_thresholds(probabilities[masks["policy"]], y[masks["policy"]])
    destination.mkdir(parents=True)
    torch.save({"model": name, "model_config": config, "model_state": best_state,
                "initialization_seed": seed, "split_seed": SPLIT_SEED,
                "dataset_sha256": DATA_SHA256, "temperature": temperature,
                "class_thresholds": thresholds, "global97_threshold": global97,
                "phase": "within_graph_five_role_study", "best_epoch": best_epoch}, destination / "model.pt")
    with (destination / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, destination / "training.png")
    policy = {"model": name, "seed": seed, "temperature": temperature,
              "global97_threshold": global97, "class98_thresholds": thresholds,
              "class98_fitting": fitting, "best_epoch": best_epoch,
              "epochs": len(history), "training_seconds_cpu": time.perf_counter() - started,
              "checkpoint_sha256": file_sha(destination / "model.pt"),
              "evaluation_status": "Prospectively frozen within-graph resplit; previously studied data, not external validation"}
    save(destination / "policy.json", policy)
    print(f"Saved policy for {name} seed={seed}; final-role labels not evaluated yet.", flush=True)
    return logits, probabilities, policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new, nonexistent output folder; partial studies are preserved too.")
    if sha256(args.data) != DATA_SHA256:
        raise ValueError("Pinned dataset required")
    torch.set_num_threads(4)
    protocol = {"date_brisbane": "2026-10-26", "id": "five-role-class-aware-v1",
                "split_seed": SPLIT_SEED, "initialization_seeds": SEEDS,
                "roles": "60% train, 10% selection, 10% temperature, 10% policy, remainder final; stratified flooring",
                "architecture": "Matched 128-64-4 MLP/GCN, 8516 parameters, original Adam/dropout/early-stopping settings",
                "policy_control": "Global 97% empirical accepted policy accuracy, maximum coverage",
                "policy_candidate": "Per predicted class: 98% empirical accepted policy accuracy; min max(50, ceil(5% of predicted class)); maximum coverage, lower-threshold tie; infeasible class refers all",
                "buffer_reason": "One percentage point above the desired 97%, chosen before this final-role evaluation; not a guarantee",
                "all_six_policies_saved_before_final_role_evaluation": True,
                "previously_studied_graph": True, "external_independent_validation": False,
                "no_split_or_threshold_search_after_final_results": True,
                "new_cpu_training": True, "original_gpu_runs_and_colab_preserved": True,
                "human_correction_accuracy_assumed": False}
    save(args.output / "FROZEN_PROTOCOL.json", protocol)
    source_hashes = {name: file_sha(HERE / name) for name in ["validation_study.py", "dataset.py", "modules.py", "evaluation.py"]}
    x, y, edges = load_arrays(args.data)
    splits = role_split(y)
    masks = {name: torch.from_numpy(np.isin(np.arange(len(y)), ids)) for name, ids in splits.items()}
    x, mean, scale = standardize_features(x, splits["train"])
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    adjacency = normalized_adjacency(torch.from_numpy(canonical_edges(edges, len(y))), len(y))
    np.savez_compressed(args.output / "split_ids.npz", **splits, feature_mean=mean, feature_scale=scale)
    save(args.output / "SPLIT_MANIFEST.json", {"split_sha256": file_sha(args.output / "split_ids.npz"),
          "counts": {name: len(ids) for name, ids in splits.items()}, "dataset_sha256": DATA_SHA256,
          "source_sha256": source_hashes})
    cached = {}
    for seed in SEEDS:
        for name in ["mlp", "gcn"]:
            cached[(name, seed)] = train_model(name, seed, x, y, masks, adjacency, args.output / f"{name}_seed{seed}")
    hashes = {f"{name}_seed{seed}": file_sha(args.output / f"{name}_seed{seed}/policy.json") for name, seed in cached}
    save(args.output / "ALL_POLICIES_FROZEN.json", hashes)
    print("All six policies frozen. Evaluating the study's final role once.", flush=True)
    result = {"protocol": protocol, "role_counts": {name: len(ids) for name, ids in splits.items()}, "runs": []}
    final = masks["final"]
    for (name, seed), (logits, probabilities, policy) in cached.items():
        assert file_sha(args.output / f"{name}_seed{seed}/policy.json") == hashes[f"{name}_seed{seed}"]
        p, labels = probabilities[final], y[final]
        global_accept = p.max(1).values >= policy["global97_threshold"]
        class_accept = class_decisions(p, policy["class98_thresholds"])
        row = {"model": name, "seed": seed, "temperature": policy["temperature"],
               "global97_threshold": policy["global97_threshold"], "class98_thresholds": policy["class98_thresholds"],
               "raw_classification": classification_metrics(logits[final], labels),
               "scaled_classification": classification_metrics(logits[final], labels, policy["temperature"]),
               "all_accept": policy_metrics(p, labels, torch.ones(len(labels), dtype=torch.bool)),
               "global97": policy_metrics(p, labels, global_accept),
               "class98": policy_metrics(p, labels, class_accept)}
        result["runs"].append(row)
        save(args.output / f"{name}_seed{seed}/final_evaluation.json", row)
        with (args.output / f"{name}_seed{seed}/private_final_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["node_id", "true_label", "predicted_label", "confidence", "global97_accepted", "class98_accepted"])
            writer.writerows(zip(splits["final"].tolist(), labels.tolist(), p.argmax(1).tolist(), p.max(1).values.tolist(), global_accept.tolist(), class_accept.tolist()))
    assert all(file_sha(HERE / name) == checksum for name, checksum in source_hashes.items())
    save(args.output / "STUDY_RESULTS.json", result)
    print("Study complete. Preserve every result; no tuning using final-role labels.", flush=True)


if __name__ == "__main__":
    main()
