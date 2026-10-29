"""Fixed post-hoc six-GCN averaging and disagreement-routing ablation.

Yuhan Liao - 49839991. Reuses saved models; does not train or retune them.
The final labels have already been inspected. Separate fitting roles avoid
direct refitting on final labels but cannot restore independent validation.
"""
from pathlib import Path
import argparse
import csv
import importlib.metadata
import platform
import shutil
import time

import numpy as np
import torch

from consensus_policy import (MEMBERS, average_and_agreement, calibrated_average,
                              confidence_diagnostics, fit_rule, load_members,
                              member_probabilities, prepare_inputs, read, route)
from dataset import DATA_SHA256, DEFAULT_PATH, load_arrays, sha256
from evaluation import classification_metrics, fit_temperature
from exact_policy import exact_decisions
from refinement_study import save
from validation_study import policy_metrics

HERE = Path(__file__).resolve().parent


def timed_calls(operation, repeats=10):
    for _ in range(3):
        operation()
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        durations.append(1000 * (time.perf_counter() - started))
    return {"median_ms": float(np.median(durations)), "p95_ms": float(np.percentile(durations, 95)),
            "repeats": repeats, "three_warmups": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-study", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Choose a new output folder; preserve completed and partial runs.")
    torch.set_num_threads(4)
    x, adjacency, roles = prepare_inputs(args.base_study, args.data)
    members, identities = load_members(args.base_study)
    protocol = {"id": "six-gcn-consensus-v1", "date_brisbane": "2026-10-26",
                "status": "Post-hoc on previously inspected graph and final cases; not independent confirmation",
                "members": identities, "member_selection": "All six existing GCNs, fixed without final-score selection",
                "averaging": "Equal arithmetic mean of six member-temperature-scaled probability vectors",
                "ensemble_temperature": "Positive scalar fit to NLL on temperature role only",
                "rules": ["mean_only", "unanimous"],
                "unanimity": "All six argmax predictions must agree; no reference labels or known-error IDs",
                "thresholds": "Exact boundaries; >=98% empirical fitting accuracy; at least max(50,ceil(5% eligible class)); maximum coverage",
                "fit_unanimity_thresholds": "Policy-role eligible cases only; independent threshold-only comparator also saved",
                "external_validation": False, "no_retraining": True,
                "no_final_result_tuning": True, "no_human_correction_assumed": True,
                "resource_scope": "CPU inference, prepared graph/input, no GPU profiling of ensemble"}
    args.output.mkdir(parents=True)
    save(args.output / "FROZEN_PROTOCOL.json", protocol)
    source_names = ["consensus_policy.py", "consensus_study.py", "exact_policy.py", "evaluation.py", "dataset.py", "modules.py"]
    source_hashes = {name: sha256(HERE / name) for name in source_names}
    save(args.output / "EXECUTED_SOURCE_HASHES.json", source_hashes)
    destination = args.output / "members"
    destination.mkdir()
    for name in ["FROZEN_PROTOCOL.json", "ALL_POLICIES_FROZEN.json", "BASE_SPLIT_MANIFEST.json", "split_ids.npz"]:
        shutil.copy2(args.base_study / name, destination / name)
    for name in MEMBERS:
        (destination / name).mkdir()
        for filename in ["policy.json", "model.pt"]:
            shutil.copy2(args.base_study / name / filename, destination / name / filename)
    raw, labels, _ = load_arrays(args.data)
    y = torch.from_numpy(labels)
    probabilities = member_probabilities(members, x, adjacency)
    mean, unanimous, votes = average_and_agreement(probabilities)
    log_mean = mean.clamp_min(torch.finfo(mean.dtype).tiny).log()
    temp_ids, policy_ids = torch.from_numpy(roles["temperature"]), torch.from_numpy(roles["policy"])
    temperature = fit_temperature(log_mean[temp_ids], y[temp_ids])
    calibrated, _, _ = calibrated_average(probabilities, temperature)
    policies = {}
    for name, require_unanimity in [("mean_only", False), ("unanimous", True)]:
        eligible = unanimous[policy_ids] if require_unanimity else torch.ones(len(policy_ids), dtype=torch.bool)
        thresholds, fitting = fit_rule(calibrated[policy_ids], y[policy_ids], eligible)
        policy = {"id": f"six-gcn-{name}-v1", "name": name, "require_unanimity": require_unanimity,
                  "temperature": temperature, "thresholds": thresholds, "fitting": fitting,
                  "dataset_sha256": DATA_SHA256, "split_sha256": sha256(destination / "split_ids.npz"),
                  "protocol_sha256": sha256(args.output / "FROZEN_PROTOCOL.json"),
                  "members_manifest_sha256": sha256(destination / "ALL_POLICIES_FROZEN.json"),
                  "member_identities": identities, "status": protocol["status"]}
        save(args.output / f"{name}_policy.json", policy)
        policies[name] = policy
    frozen = {name: sha256(args.output / f"{name}_policy.json") for name in policies}
    save(args.output / "ALL_POLICIES_FROZEN.json", frozen)
    print("Both consensus policies saved before this explicitly post-hoc final comparison.", flush=True)

    ids = torch.from_numpy(roles["final"])
    truth = y[ids]
    final_p, final_agreement = calibrated[ids], unanimous[ids]
    records = {"protocol": protocol, "temperature": temperature, "final_cases": len(ids), "comparisons": {}}
    baseline = None
    for index, key in enumerate(MEMBERS):
        p = probabilities[index, ids]
        thresholds = read(destination / key / "policy.json")["thresholds"]
        accepted = exact_decisions(p, thresholds)
        records["comparisons"][key] = {
            "classification": classification_metrics(p.clamp_min(torch.finfo(p.dtype).tiny).log(), truth),
            "confidence": confidence_diagnostics(p, truth), "policy": policy_metrics(p, truth, accepted),
            "accepted_errors_ge_099": int((accepted & (p.argmax(1) != truth) & (p.max(1).values >= .99)).sum())}
        if key == "unweighted_exact_seed42":
            baseline = (p, accepted)
    for name, policy in policies.items():
        accepted = route(final_p, final_agreement, policy["thresholds"], policy["require_unanimity"])
        result = {"classification": classification_metrics(final_p.clamp_min(torch.finfo(final_p.dtype).tiny).log(), truth),
                  "confidence": confidence_diagnostics(final_p, truth),
                  "policy": policy_metrics(final_p, truth, accepted),
                  "accepted_errors_ge_099": int((accepted & (final_p.argmax(1) != truth) & (final_p.max(1).values >= .99)).sum()),
                  "disagreement_cases": int((~final_agreement).sum()),
                  "disagreement_true_politicians": int((~final_agreement & (truth == 3)).sum())}
        records["comparisons"][name] = result
    records["uncalibrated_mean_confidence"] = confidence_diagnostics(mean[ids], truth)
    base_p, base_accepted = baseline
    wrong_pol = base_accepted & (truth == 3) & (base_p.argmax(1) != truth)
    consensus_accepted = route(final_p, final_agreement, policies["unanimous"]["thresholds"], True)
    records["baseline_politician_error_tracking"] = {
        "baseline_accepted_missed_politicians": int(wrong_pol.sum()),
        "new_rule_refers": int((wrong_pol & ~consensus_accepted).sum()),
        "new_rule_still_automatically_wrong": int((wrong_pol & consensus_accepted & (final_p.argmax(1) != truth)).sum()),
        "classification_changed_to_correct": int((wrong_pol & (final_p.argmax(1) == truth)).sum()),
        "warning": "Referring or changing a prediction does not prove human correction"}
    records["case_tracking"] = []
    for local in torch.where(wrong_pol)[0].tolist()[:5]:
        records["case_tracking"].append({"node_id": int(ids[local]), "true_label": int(truth[local]),
            "baseline_prediction": int(base_p[local].argmax()), "baseline_confidence": float(base_p[local].max()),
            "six_votes": votes[:, ids[local]].tolist(), "mean_prediction": int(final_p[local].argmax()),
            "mean_confidence": float(final_p[local].max()), "unanimous": bool(final_agreement[local]),
            "accepted": bool(consensus_accepted[local])})
    with (args.output / "private_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["node_id", "true_label", "predicted_label", "confidence", "unanimous", "accepted_mean_only", "accepted_unanimous"])
        plain = route(final_p, final_agreement, policies["mean_only"]["thresholds"], False)
        writer.writerows(zip(ids.tolist(), truth.tolist(), final_p.argmax(1).tolist(), final_p.max(1).values.tolist(),
                             final_agreement.tolist(), plain.tolist(), consensus_accepted.tolist()))
    np.savez_compressed(args.output / "private_probabilities.npz", node_id=roles["final"],
                        member_probabilities=probabilities[:, ids].numpy(), probabilities=final_p.numpy())
    with torch.no_grad():
        single = timed_calls(lambda: (members[0][0](x, adjacency)[0] / members[0][1]).softmax(1))
        ensemble = timed_calls(lambda: calibrated_average(member_probabilities(members, x, adjacency), temperature))
    records["cpu_profile"] = {"single_gcn": single, "six_member_average_and_agreement": ensemble,
        "latency_ratio": ensemble["median_ms"] / single["median_ms"], "threads": 4, "nodes_per_forward": len(raw),
        "single_parameters": identities[0]["parameters"], "six_member_parameters": sum(item["parameters"] for item in identities),
        "scope": "CPU full-graph inference with prepared input/adjacency and probability calculations; no file IO, GPU, training or human review"}
    records["environment"] = {"python": platform.python_version(), "platform": platform.platform(), "device": "cpu",
                               "packages": {name: importlib.metadata.version(name) for name in ["torch", "numpy", "scikit-learn"]}}
    assert all(sha256(HERE / key) == value for key, value in source_hashes.items())
    assert all(sha256(args.output / f"{key}_policy.json") == value for key, value in frozen.items())
    save(args.output / "CONSENSUS_RESULTS.json", records)
    print("Consensus comparison complete. Preserve all outcomes; do not choose a new rule from final scores.", flush=True)
    for key in ["unweighted_exact_seed42", "mean_only", "unanimous"]:
        row = records["comparisons"][key]
        print(key, row["policy"]["coverage"], row["policy"]["accepted_accuracy"], row["policy"]["accepted_errors"],
              row["policy"]["true_class"]["politician"]["accepted_errors"], flush=True)


if __name__ == "__main__":
    main()
