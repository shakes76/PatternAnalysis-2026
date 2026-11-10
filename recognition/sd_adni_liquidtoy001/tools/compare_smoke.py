"""Accept only matching continuous and restarted real training trajectories."""

import argparse
import copy
import json
from pathlib import Path

import torch
from train_smoke import validate_contract


def compare(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        if (
            not isinstance(right, torch.Tensor)
            or left.shape != right.shape
            or left.dtype != right.dtype
        ):
            raise AssertionError(f"Tensor schema differs: {path}")
        if left.is_floating_point():
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5, msg=path)
        elif not torch.equal(left, right):
            raise AssertionError(f"Discrete tensor differs: {path}")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError(f"Keys differ: {path}")
        for key in left:
            compare(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"Sequence differs: {path}")
        for i, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f"{path}.{i}")
    elif isinstance(left, float):
        if abs(left - right) > 1e-6 + 1e-5 * abs(right):
            raise AssertionError(f"Scalar differs: {path}")
    elif left != right:
        raise AssertionError(f"Value differs: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    states = [
        torch.load(
            args.root / name / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        for name in ("continuous", "resumed")
    ]
    compare(*states)
    corrupt = copy.deepcopy(states[0]["contract"])
    corrupt["manifest"] = "wrong-manifest"
    try:
        validate_contract(corrupt, states[0]["contract"])
    except ValueError:
        pass
    else:
        raise AssertionError("Corrupted contract was accepted")
    # Also prove the comparator rejects a changed learned adapter parameter.
    changed = copy.deepcopy(states[1])
    next(iter(changed["adapter"].values())).view(-1)[0] += 1
    try:
        compare(states[0], changed)
    except AssertionError:
        pass
    else:
        raise AssertionError("Changed adapter was accepted")
    maximum = max(
        (states[0]["adapter"][k] - states[1]["adapter"][k]).abs().max().item()
        for k in states[0]["adapter"]
    )
    report = {
        "status": "passed",
        "steps": states[0]["step"],
        "max_adapter_difference": maximum,
        "atol": 1e-6,
        "rtol": 1e-5,
        "checks": [
            "adapters",
            "optimizer",
            "scheduler",
            "RNG",
            "input_noise_loss_traces",
            "mismatched_contract_rejected",
            "changed_adapter_rejected",
        ],
    }
    (args.root / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
