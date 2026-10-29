"""Post-hoc controlled comparison of exact thresholds and class-weighted GCN.

Yuhan Liao - 49839991. The existing graph and final role have been inspected.
All new policies are saved before this comparison, without external-validation claims.
"""
from pathlib import Path
import argparse
import csv
import importlib.metadata
import json
import platform
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from dataset import CLASS_NAMES, DATA_SHA256, DEFAULT_PATH, canonical_edges, load_arrays, sha256, standardize_features
from evaluation import classification_metrics, fit_temperature, plot_history
from exact_policy import exact_decisions, fit_exact_class_thresholds
from modules import make_model, normalized_adjacency
from validation_study import ROLES, SPLIT_SEED, class_decisions, policy_metrics

HERE = Path(__file__).resolve().parent
SEEDS = [42, 43, 44]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def training_class_weights(y, train_mask):
    counts = torch.bincount(y[train_mask], minlength=len(CLASS_NAMES))
    if bool((counts == 0).any()):
        raise ValueError("Each class needs training examples")
    return counts, counts.sum().float() / (len(CLASS_NAMES) * counts.float())


def train_weighted(seed, x, y, masks, adjacency, weights, output):
    torch.manual_seed(seed)
    config = {"inputs": x.shape[1], "hidden": 64, "classes": 4, "dropout": .5}
    model = make_model("gcn", **config)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.0005)
    best_f1, best_epoch, best_state = -1., 0, None
    history = []
    started = time.perf_counter()
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(x, adjacency)
        loss = F.cross_entropy(logits[masks["train"]], y[masks["train"]], weight=weights)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite weighted loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Nonfinite gradient")
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits, _ = model(x, adjacency)
        selected = classification_metrics(logits[masks["selection"]], y[masks["selection"]])
        history.append({"epoch": epoch, "train_loss": float(loss.detach()),
                        "selection_loss": selected["negative_log_likelihood"], "selection_macro_f1": selected["macro_f1"]})
        if selected["macro_f1"] > best_f1:
            best_f1, best_epoch = selected["macro_f1"], epoch
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        if epoch == 1 or epoch % 50 == 0:
            print(f"weighted GCN seed={seed} epoch={epoch} selection_macro_f1={selected['macro_f1']:.4f}", flush=True)
        if epoch - best_epoch >= 30:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, _ = model(x, adjacency)
    temperature = fit_temperature(logits[masks["temperature"]], y[masks["temperature"]])
    checkpoint = {"model": "gcn", "model_config": config, "model_state": best_state,
                  "initialization_seed": seed, "split_seed": SPLIT_SEED, "dataset_sha256": DATA_SHA256,
                  "temperature": temperature, "class_weights": weights, "best_epoch": best_epoch,
                  "phase": "posthoc_weighted_refinement"}
    output.mkdir(parents=True)
    torch.save(checkpoint, output / "model.pt")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, output / "training.png")
    save(output / "training_summary.json", {"epochs": len(history), "best_epoch": best_epoch,
         "best_selection_macro_f1": best_f1, "training_seconds_cpu": time.perf_counter() - started,
         "class_weights": weights.tolist(), "selection_objective_unchanged": True})
    return logits, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-study", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new output folder; completed and partial experiments are preserved.")
    if sha256(args.data) != DATA_SHA256:
        raise ValueError("Pinned dataset required")
    torch.set_num_threads(4)
    base_manifest = read(args.base_study / "SPLIT_MANIFEST.json")
    if sha256(args.base_study / "split_ids.npz") != base_manifest["split_sha256"]:
        raise ValueError("Base split/scaling artifact mismatch")
    raw, labels, edges = load_arrays(args.data)
    with np.load(args.base_study / "split_ids.npz", allow_pickle=False) as saved:
        roles = {name: saved[name].copy() for name in ROLES}
        old_mean, old_scale = saved["feature_mean"].copy(), saved["feature_scale"].copy()
    if not np.array_equal(np.sort(np.concatenate(list(roles.values()))), np.arange(len(raw))):
        raise ValueError("Label roles must be disjoint and exhaustive")
    x, mean, scale = standardize_features(raw, roles["train"])
    if not np.array_equal(mean, old_mean) or not np.array_equal(scale, old_scale):
        raise ValueError("Training-only scaling mismatch")
    x, y = torch.from_numpy(x), torch.from_numpy(labels)
    masks = {name: torch.from_numpy(np.isin(np.arange(len(labels)), ids)) for name, ids in roles.items()}
    counts, weights = training_class_weights(y, masks["train"])
    protocol = {"id": "exact-threshold-weighted-gcn-v1", "date_brisbane": "2026-10-26",
                "status": "Post-hoc controlled refinement; graph and final-role outcomes were previously inspected",
                "external_independent_validation": False, "seeds": SEEDS, "split_seed": SPLIT_SEED,
                "three_variants": ["unweighted grid Class98", "unweighted exact Class98", "weighted exact Class98"],
                "weighted_change": "Only training CE weights n_train/(4*n_class), derived from training labels",
                "training_class_counts": counts.tolist(), "class_weights": weights.tolist(),
                "architecture_and_optimizer": "Original 128-64-4 GCN, dropout .5, Adam .01/weight decay .0005, max200/patience30, selection Macro-F1",
                "temperature": "Unweighted NLL fit on temperature role; original temperatures retained for unweighted models",
                "exact_policy": "Distinct confidence boundaries with whole ties; >=98% empirical accuracy and max(50,ceil(5% class)) cases; max coverage; all-accept cutoff0; fallback1.01",
                "all_six_new_policies_saved_before_this_comparison": True,
                "no_target_weight_or_threshold_retuning_after_comparison": True,
                "no_true_labels_or_error_IDs_in_route_decisions": True,
                "random_audit": "Optional label-free sample of eligible predictions; no human correction assumed"}
    save(args.output / "FROZEN_PROTOCOL.json", protocol)
    shutil.copy2(args.base_study / "split_ids.npz", args.output / "split_ids.npz")
    shutil.copy2(args.base_study / "SPLIT_MANIFEST.json", args.output / "BASE_SPLIT_MANIFEST.json")
    source_hashes = {name: sha256(HERE / name) for name in ["refinement_study.py", "exact_policy.py", "validation_study.py", "dataset.py", "modules.py", "evaluation.py"]}
    save(args.output / "EXECUTED_SOURCE_HASHES.json", source_hashes)
    adjacency = normalized_adjacency(torch.from_numpy(canonical_edges(edges, len(raw))), len(raw))
    cached = {}
    frozen_base = read(args.base_study / "ALL_POLICIES_FROZEN.json")
    for seed in SEEDS:
        base_folder = args.base_study / f"gcn_seed{seed}"
        old = read(base_folder / "policy.json")
        if sha256(base_folder / "policy.json") != frozen_base[f"gcn_seed{seed}"] or sha256(base_folder / "model.pt") != old["checkpoint_sha256"]:
            raise ValueError("Base policy/checkpoint binding mismatch")
        base_checkpoint = torch.load(base_folder / "model.pt", map_location="cpu", weights_only=True)
        model = make_model("gcn", **base_checkpoint["model_config"])
        model.load_state_dict(base_checkpoint["model_state"])
        model.eval()
        with torch.no_grad():
            baseline_logits, _ = model(x, adjacency)
        for variant in ["unweighted_exact", "weighted_exact"]:
            folder = args.output / f"{variant}_seed{seed}"
            if variant == "unweighted_exact":
                folder.mkdir()
                shutil.copy2(base_folder / "model.pt", folder / "model.pt")
                logits, checkpoint = baseline_logits, base_checkpoint
            else:
                logits, checkpoint = train_weighted(seed, x, y, masks, adjacency, weights, folder)
            probabilities = (logits / checkpoint["temperature"]).softmax(1)
            thresholds, fitting = fit_exact_class_thresholds(probabilities[masks["policy"]], y[masks["policy"]])
            policy = {"id": f"{variant}-seed{seed}", "variant": variant, "seed": seed,
                      "temperature": checkpoint["temperature"], "thresholds": thresholds, "fitting": fitting,
                      "checkpoint_sha256": sha256(folder / "model.pt"), "dataset_sha256": DATA_SHA256,
                      "split_sha256": base_manifest["split_sha256"], "protocol_sha256": sha256(args.output / "FROZEN_PROTOCOL.json"),
                      "evaluation_status": protocol["status"]}
            save(folder / "policy.json", policy)
            cached[(variant, seed)] = (logits, probabilities, policy, old)
            print(f"Saved {variant} seed={seed}; politician threshold={thresholds[3]:.17g}", flush=True)
    frozen = {f"{variant}_seed{seed}": sha256(args.output / f"{variant}_seed{seed}/policy.json") for variant, seed in cached}
    save(args.output / "ALL_POLICIES_FROZEN.json", frozen)
    print("All six new policies frozen. Starting explicitly post-hoc comparison.", flush=True)
    results = {"protocol": protocol, "runs": []}
    final = masks["final"]
    for (variant, seed), (logits, probabilities, policy, old) in cached.items():
        folder = args.output / f"{variant}_seed{seed}"
        assert sha256(folder / "policy.json") == frozen[f"{variant}_seed{seed}"]
        p, truth = probabilities[final], y[final]
        accepted = exact_decisions(p, policy["thresholds"])
        row = {"variant": variant, "seed": seed, "thresholds": policy["thresholds"],
               "classification_raw": classification_metrics(logits[final], truth),
               "classification_scaled": classification_metrics(logits[final], truth, policy["temperature"]),
               "policy": policy_metrics(p, truth, accepted),
               "accepted_errors_confidence_ge_099": int(((p.argmax(1) != truth) & accepted & (p.max(1).values >= .99)).sum())}
        if variant == "unweighted_exact":
            coarse = class_decisions(p, old["class98_thresholds"])
            row["grid_policy_control"] = policy_metrics(p, truth, coarse)
            original = read(args.base_study / f"gcn_seed{seed}/final_evaluation.json")
            assert row["classification_raw"]["confusion_matrix"] == original["raw_classification"]["confusion_matrix"]
        save(folder / "evaluation.json", row)
        with (folder / "private_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["node_id", "true_label", "predicted_label", "confidence", "accepted"])
            writer.writerows(zip(roles["final"].tolist(), truth.tolist(), p.argmax(1).tolist(), p.max(1).values.tolist(), accepted.tolist()))
        results["runs"].append(row)
    assert all(sha256(HERE / name) == value for name, value in source_hashes.items())
    save(args.output / "ENVIRONMENT.json", {"python": platform.python_version(), "platform": platform.platform(),
         "device": "cpu", "threads": 4, "packages": {name: importlib.metadata.version(name) for name in ["torch", "torch-geometric", "numpy", "scikit-learn"]}})
    save(args.output / "REFINEMENT_RESULTS.json", results)
    print("Refinement complete; retain failures and do not retune using these outcomes.", flush=True)


if __name__ == "__main__":
    main()
