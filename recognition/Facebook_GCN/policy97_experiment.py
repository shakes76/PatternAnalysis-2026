"""Fit a frozen 97% empirical acceptance policy and report a post-hoc follow-up.

Original checkpoints and project files are read-only inputs. All three policy
files are saved before any original-test labels are used for this follow-up.
The original test was inspected previously; this is not an untouched test.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def accepted_mask(probabilities, threshold):
    """Make label-free decisions; threshold > 1 explicitly refers every case."""
    import torch
    if probabilities.ndim != 2 or probabilities.shape[0] == 0 or probabilities.shape[1] < 2 or not torch.isfinite(probabilities).all():
        raise ValueError("Finite row-by-class probabilities required")
    if bool((probabilities < 0).any()) or bool((probabilities > 1).any()):
        raise ValueError("Probabilities must be in [0, 1]")
    if not torch.allclose(probabilities.sum(1), torch.ones_like(probabilities.sum(1)), atol=1e-5, rtol=1e-5):
        raise ValueError("Probability rows must sum to one")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Finite nonnegative threshold required")
    return probabilities.max(dim=1).values >= threshold


def verify_policy_binding(policy, checkpoint_path, dataset_sha256, seed):
    if policy["checkpoint_sha256"] != digest(checkpoint_path):
        raise ValueError("Policy is bound to a different checkpoint")
    if policy["dataset_sha256"] != dataset_sha256 or policy["initialization_seed"] != seed:
        raise ValueError("Policy dataset or initialization seed mismatch")


def quick_semantic_checks():
    import torch
    p = torch.tensor([[0.75, 0.25], [0.50, 0.50], [0.95, 0.05]])
    assert accepted_mask(p, 0.75).tolist() == [True, False, True]
    assert not bool(accepted_mask(p, 1.01).any())
    try:
        accepted_mask(p, float("nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid policy threshold must be rejected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs/final")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/policy97_followup")
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new --output; partial experiment folders are preserved too.")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source_dir.resolve()))

    import numpy as np
    import torch
    from dataset import CLASS_NAMES, DATA_SHA256, DEFAULT_PATH, make_pyg_graph, sha256
    from evaluation import choose_threshold, classification_metrics, selective_metrics, split_validation_roles
    from modules import make_model, normalized_adjacency

    torch.set_num_threads(4)
    quick_semantic_checks()
    original_hashes = {str(p): digest(p) for p in args.source_dir.rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts}
    checkpoint_hashes = {seed: digest(args.runs_dir / f"gcn_seed{seed}/model.pt") for seed in [42, 43, 44]}
    protocol = {
        "id": "normal-policy97-followup-v1", "decision_date_brisbane": "2026-10-26",
        "selected_policy": "Maximum calibration coverage subject to empirical accepted accuracy >= 0.97",
        "target_accuracy": 0.97, "threshold_grid": "0.00 to 1.00 inclusive in steps of 0.01",
        "minimum_accepted_count": "max(1, floor(0.05 * calibration_count))",
        "tie_break": "lower threshold", "no_feasible_threshold": "1.01, refer all; accepted accuracy undefined",
        "initialization_seeds": [42, 43, 44], "data_split_seed": 42, "validation_role_seed": 43,
        "weights": "Original GCN checkpoints unchanged", "temperature": "Original calibration-fitted values unchanged",
        "threshold_fit_labels": "Original calibration subset only",
        "all_policies_frozen_before_this_followup_test_metrics": True,
        "original_test_previously_inspected": True,
        "evaluation_status": "Post-hoc original-test reuse; not independent confirmation",
        "parameters_will_not_be_retuned_after_followup_test": True,
        "target_is_not_course_requirement_or_population_guarantee": True,
        "target_97_does_not_require_threshold_0_97": True,
        "human_review_accuracy_assumed": False,
    }
    save(args.output / "FROZEN_PROTOCOL.json", protocol)
    protocol_sha = digest(args.output / "FROZEN_PROTOCOL.json")
    if sha256(args.data or DEFAULT_PATH) != DATA_SHA256:
        raise ValueError("Dataset differs from the pinned course archive")
    graph = make_pyg_graph(args.data or DEFAULT_PATH, 42)
    _, calibration = split_validation_roles(graph.y, graph.validation_mask, 43)
    adjacency = normalized_adjacency(graph.edge_index, graph.num_nodes)
    saved = {}
    development = []

    # Stage A: only calibration labels are used. Save all policies before Stage B.
    for seed in [42, 43, 44]:
        folder = args.runs_dir / f"gcn_seed{seed}"
        checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
        assert checkpoint["phase"] == "final" and checkpoint["dataset_sha256"] == DATA_SHA256
        assert checkpoint["model"] == "gcn" and checkpoint["initialization_seed"] == seed
        assert checkpoint["split_seed"] == 42 and checkpoint["validation_role_seed"] == 43
        if not math.isfinite(checkpoint["temperature"]) or checkpoint["temperature"] <= 0:
            raise ValueError("A finite positive temperature is required")
        torch.testing.assert_close(graph.feature_mean, checkpoint["feature_mean"])
        torch.testing.assert_close(graph.feature_scale, checkpoint["feature_scale"])
        model = make_model("gcn", **checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        with torch.no_grad():
            logits, _ = model(graph.x, adjacency)
        temperature = checkpoint["temperature"]
        threshold = choose_threshold(logits[calibration], graph.y[calibration], temperature, 0.97)
        metrics = selective_metrics(logits[calibration], graph.y[calibration], temperature, threshold)
        assert metrics["accepted"] == 0 or metrics["accepted_accuracy"] >= 0.97
        policy = {
            "policy_id": f"normal-policy97-gcn-seed{seed}", "status": "development_fitted_frozen_for_posthoc_followup",
            "protocol_sha256": protocol_sha, "dataset_sha256": DATA_SHA256,
            "checkpoint_sha256": checkpoint_hashes[seed], "initialization_seed": seed,
            "model_config": checkpoint["model_config"], "split_seed": 42, "validation_role_seed": 43,
            "temperature": temperature, "original_threshold": checkpoint["confidence_threshold"],
            "new_threshold": threshold, "target_calibration_accepted_accuracy": 0.97,
            "calibration_metrics": metrics,
        }
        save(args.output / f"policies/gcn_seed{seed}_policy97.json", policy)
        development.append({"seed": seed, **metrics})
        saved[seed] = (logits, policy)
    policy_hashes = {seed: digest(args.output / f"policies/gcn_seed{seed}_policy97.json") for seed in saved}
    save(args.output / "POLICIES_FROZEN_BEFORE_FOLLOWUP.json", {
        "protocol_sha256": protocol_sha,
        "policy_sha256": {str(seed): value for seed, value in policy_hashes.items()},
        "next_stage": "Original-test post-hoc evaluation; no policy refitting allowed",
    })
    print("All three 97% policies frozen. Starting labelled post-hoc original-test follow-up.", flush=True)

    # Stage B: reuse the historically inspected test; never feed these labels to policy fitting.
    truth = graph.y[graph.test_mask]
    test_ids = torch.where(graph.test_mask)[0]
    results = []
    for seed in [42, 43, 44]:
        logits, policy = saved[seed]
        verify_policy_binding(policy, args.runs_dir / f"gcn_seed{seed}/model.pt", DATA_SHA256, seed)
        assert digest(args.output / f"policies/gcn_seed{seed}_policy97.json") == policy_hashes[seed]
        test_logits = logits[graph.test_mask]
        original = json.loads((args.runs_dir / f"gcn_seed{seed}/summary.json").read_text())
        all_metrics = classification_metrics(test_logits, truth)
        assert all_metrics["accuracy"] == original["test_raw"]["accuracy"]
        assert all_metrics["confusion_matrix"] == original["test_raw"]["confusion_matrix"]
        old_policy = selective_metrics(test_logits, truth, policy["temperature"], policy["original_threshold"])
        new_policy = selective_metrics(test_logits, truth, policy["temperature"], policy["new_threshold"])
        probabilities = (test_logits / policy["temperature"]).softmax(1)
        accept = accepted_mask(probabilities, policy["new_threshold"])
        prediction = probabilities.argmax(1)
        wrong = prediction != truth
        per_class = {}
        for i, name in enumerate(CLASS_NAMES):
            group = truth == i
            selected = group & accept
            per_class[name] = {
                "total": int(group.sum()), "accepted": int(selected.sum()),
                "reviewed_fraction": float((group & ~accept).sum() / group.sum()),
                "accepted_accuracy": float((prediction[selected] == truth[selected]).float().mean()) if selected.any() else None,
            }
        row = {
            "seed": seed, "threshold": policy["new_threshold"], "model_accuracy_unchanged": all_metrics["accuracy"],
            "original_policy": old_policy, "new_policy_posthoc": new_policy,
            "reviewed": int((~accept).sum()), "referred_original_errors": int((wrong & ~accept).sum()),
            "accepted_errors": int((wrong & accept).sum()), "total_model_errors": int(wrong.sum()),
            "error_capture_fraction": float((wrong & ~accept).sum() / wrong.sum()),
            "per_true_class_posthoc": per_class,
            "target_97_met_on_reused_test": new_policy["accepted_accuracy"] is not None and new_policy["accepted_accuracy"] >= 0.97,
        }
        destination = args.output / f"posthoc_predictions/gcn_seed{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "predictions_policy97_original_test.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["node_id", "true_label", "predicted_label", "confidence", "policy97_accepted", "evaluation_status"])
            for node, label, pred, confidence, decision in zip(test_ids.tolist(), truth.tolist(), prediction.tolist(), probabilities.max(1).values.tolist(), accept.tolist()):
                writer.writerow([node, label, pred, confidence, decision, "posthoc_original_test_reuse"])
        save(destination / "posthoc_evaluation.json", row)
        results.append(row)

    for name, checksum in original_hashes.items():
        assert digest(name) == checksum
    assert all(digest(args.runs_dir / f"gcn_seed{seed}/model.pt") == value for seed, value in checkpoint_hashes.items())
    aggregate = {}
    for key, function in {
        "test_review_fraction": lambda row: row["reviewed"] / row["new_policy_posthoc"]["total"],
        "test_coverage": lambda row: row["new_policy_posthoc"]["coverage"],
        "test_accepted_accuracy": lambda row: row["new_policy_posthoc"]["accepted_accuracy"],
        "error_capture_fraction": lambda row: row["error_capture_fraction"],
        "original_all_test_accuracy": lambda row: row["model_accuracy_unchanged"],
    }.items():
        values = [function(row) for row in results]
        aggregate[key] = {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}
    record = {
        "date_brisbane": "2026-10-26", "protocol": protocol, "development": development,
        "posthoc_original_test": results, "aggregate_posthoc": aggregate,
        "independent_validation_completed": False, "original_model_sources_and_weights_unchanged": True,
        "policy_files_unchanged_after_followup": True,
        "next_action": "Keep this fixed rule; report all results without further tuning. Independent confirmation remains outstanding.",
    }
    save(args.output / "FOLLOWUP_RESULTS.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
