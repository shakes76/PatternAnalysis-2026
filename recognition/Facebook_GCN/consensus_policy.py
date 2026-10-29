"""Probability averaging and label-free disagreement checks for saved GCNs.

Yuhan Liao - 49839991. Averaging is inspired by Lakshminarayanan et al.,
NeurIPS 2017, https://arxiv.org/abs/1612.01474. The fixed six-member,
mixed-loss unanimity gate is a project-specific empirical ablation, not a
reproduction of that paper or a statistical safety guarantee.
"""
from pathlib import Path
import json
import math

import numpy as np
import torch

from dataset import (CLASS_NAMES, DATA_SHA256, canonical_edges, load_arrays,
                     sha256, standardize_features)
from exact_policy import exact_decisions, fit_exact_class_thresholds, validate_probabilities
from modules import make_model, normalized_adjacency
from validation_study import ROLES, SPLIT_SEED

MEMBERS = [f"{variant}_seed{seed}" for variant in ["unweighted_exact", "weighted_exact"] for seed in [42, 43, 44]]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare_inputs(study, data):
    """Verify frozen split/scaling and build a label-free inference input."""
    study, data = Path(study), Path(data)
    if sha256(data) != DATA_SHA256:
        raise ValueError("Pinned data checksum mismatch")
    manifest = read(study / "BASE_SPLIT_MANIFEST.json")
    if sha256(study / "split_ids.npz") != manifest["split_sha256"]:
        raise ValueError("Split/scaling checksum mismatch")
    raw, _, edges = load_arrays(data)
    with np.load(study / "split_ids.npz", allow_pickle=False) as saved:
        roles = {key: saved[key].copy() for key in ROLES}
        mean, scale = saved["feature_mean"].copy(), saved["feature_scale"].copy()
    if any(ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids) for ids in roles.values()):
        raise ValueError("Nonempty integer label roles required")
    if not np.array_equal(np.sort(np.concatenate(list(roles.values()))), np.arange(len(raw))):
        raise ValueError("Roles must be disjoint and exhaustive")
    x, actual_mean, actual_scale = standardize_features(raw, roles["train"])
    if not np.array_equal(actual_mean, mean) or not np.array_equal(actual_scale, scale):
        raise ValueError("Training-only scaling mismatch")
    adjacency = normalized_adjacency(torch.from_numpy(canonical_edges(edges, len(raw))), len(raw))
    return torch.from_numpy(x), adjacency, roles


def load_members(study):
    """Bind all six existing policies to dataset, protocol, split and weights."""
    study = Path(study)
    frozen = read(study / "ALL_POLICIES_FROZEN.json")
    if set(frozen) != set(MEMBERS):
        raise ValueError("Exactly the six predefined members are required")
    protocol_hash = sha256(study / "FROZEN_PROTOCOL.json")
    split_hash = sha256(study / "split_ids.npz")
    protocol = read(study / "FROZEN_PROTOCOL.json")
    members, identities = [], []
    for key in MEMBERS:
        folder = study / key
        if sha256(folder / "policy.json") != frozen[key]:
            raise ValueError("Frozen member policy mismatch")
        policy = read(folder / "policy.json")
        if sha256(folder / "model.pt") != policy["checkpoint_sha256"]:
            raise ValueError("Member checkpoint mismatch")
        if policy["dataset_sha256"] != DATA_SHA256 or policy["split_sha256"] != split_hash or policy["protocol_sha256"] != protocol_hash:
            raise ValueError("Member data/protocol/split binding mismatch")
        checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
        phase = "posthoc_weighted_refinement" if key.startswith("weighted") else "within_graph_five_role_study"
        if key != f"{policy['variant']}_seed{policy['seed']}" or checkpoint["initialization_seed"] != policy["seed"]:
            raise ValueError("Member identity mismatch")
        if checkpoint["phase"] != phase or checkpoint["split_seed"] != SPLIT_SEED or checkpoint["dataset_sha256"] != DATA_SHA256:
            raise ValueError("Member checkpoint protocol mismatch")
        if checkpoint["model"] != "gcn" or checkpoint["model_config"] != {"inputs": 128, "hidden": 64, "classes": 4, "dropout": .5}:
            raise ValueError("Member architecture mismatch")
        temperature = policy["temperature"]
        if not math.isfinite(temperature) or temperature <= 0 or temperature != checkpoint["temperature"]:
            raise ValueError("Member temperature mismatch")
        if key.startswith("weighted") and not torch.equal(checkpoint["class_weights"], torch.tensor(protocol["class_weights"])):
            raise ValueError("Member class-weight mismatch")
        model = make_model("gcn", **checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        members.append((model, temperature))
        identities.append({"run": key, "policy_sha256": frozen[key], "checkpoint_sha256": policy["checkpoint_sha256"],
                           "temperature": temperature, "parameters": sum(p.numel() for p in model.parameters())})
    return members, identities


def validate_members(probabilities):
    if probabilities.ndim != 3 or probabilities.shape[0] < 2:
        raise ValueError("At least two member probability matrices required")
    for member in probabilities:
        validate_probabilities(member)


def average_and_agreement(probabilities):
    """Use model outputs only; equal votes never imply ground-truth correctness."""
    validate_members(probabilities)
    mean = probabilities.mean(0)
    votes = probabilities.argmax(2)
    unanimous = (votes == votes[0:1]).all(0)
    return mean, unanimous, votes


def calibrated_average(probabilities, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("A finite positive ensemble temperature is required")
    mean, unanimous, votes = average_and_agreement(probabilities)
    logits = mean.clamp_min(torch.finfo(mean.dtype).tiny).log()
    return (logits / temperature).softmax(1), unanimous, votes


def member_probabilities(members, x, adjacency):
    with torch.no_grad():
        return torch.stack([(model(x, adjacency)[0] / temperature).softmax(1) for model, temperature in members])


def fit_rule(probabilities, labels, eligible):
    """Fit only supplied policy-role cases; never accept a subset of score ties."""
    validate_probabilities(probabilities)
    if eligible.dtype != torch.bool or eligible.shape != (len(probabilities),):
        raise ValueError("Aligned boolean eligibility mask required")
    if eligible.any():
        return fit_exact_class_thresholds(probabilities[eligible], labels[eligible], target=.98, minimum=50)
    return [1.01] * 4, [{"predicted_class": name, "policy_cases": 0, "minimum_accepted": 50,
                        "accepted": 0, "threshold": 1.01, "accepted_accuracy": None,
                        "feasible_on_fitting_data": False} for name in CLASS_NAMES]


def route(probabilities, unanimous, thresholds, require_unanimity):
    """Predict labels, then check thresholds and optional disagreement rejection."""
    if unanimous.dtype != torch.bool or unanimous.shape != (len(probabilities),):
        raise ValueError("Aligned boolean unanimity mask required")
    threshold_pass = exact_decisions(probabilities, thresholds)
    return threshold_pass & unanimous if require_unanimity else threshold_pass


def confidence_diagnostics(probabilities, labels):
    """Descriptive groups distinguish predicted politicians from missed true ones."""
    validate_probabilities(probabilities)
    prediction, confidence = probabilities.argmax(1), probabilities.max(1).values
    correct = prediction == labels
    groups = {}
    for name, mask in [("all", torch.ones_like(labels, dtype=torch.bool)),
                       ("predicted_politician", prediction == 3), ("true_politician", labels == 3)]:
        selected = confidence[mask]
        groups[name] = {"count": int(mask.sum()), "correct": int((mask & correct).sum()),
                        "accuracy": float(correct[mask].double().mean()) if mask.any() else None,
                        "mean_confidence": float(selected.double().mean()) if len(selected) else None,
                        "errors_confidence_ge_099": int((mask & ~correct & (confidence >= .99)).sum())}
    scores, binary = probabilities[:, 3], labels == 3
    bins, ece = [], 0.
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        chosen = (scores >= low) & (scores < high if index < 9 else scores <= high)
        count = int(chosen.sum())
        frequency = float(binary[chosen].double().mean()) if count else None
        certainty = float(scores[chosen].double().mean()) if count else None
        if count:
            ece += count / len(labels) * abs(frequency - certainty)
        bins.append({"lower": low, "upper": high, "count": count,
                     "observed_politician_frequency": frequency, "mean_politician_probability": certainty})
    return {"groups": groups, "politician_one_vs_rest_ece_10bins": ece,
            "politician_one_vs_rest_brier": float(((scores.double() - binary.double()) ** 2).mean()),
            "politician_probability_bins": bins,
            "missed_true_politicians": int(((labels == 3) & (prediction != 3)).sum()),
            "scope": "Descriptive on reused final cases; small bins and correlated graph nodes limit interpretation"}
