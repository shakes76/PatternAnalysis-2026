"""Verify the saved held-out results without raw data, checkpoints or generation."""
import hashlib
import json
import math
from pathlib import Path
import statistics

PROJECT = Path(__file__).resolve().parents[1]
ARCHIVE = PROJECT / "evaluation/final_test_20261106"
EXPECTED = {
    "plan.json": "14f2c2fa6c82de4ef3a1a45f242220e40639dd28f9acf2455e817a35f2ec82d6",
    "evaluate_frozen_test.py": "80069ec82a6ab2c46d13b47925a6c5fe7097954e656673557f3d6dd9c6ed5adc",
    "report": "a6b9c0f3d9636703aeee4d57bcf630ecd172a77049a4d1088059fb1a32f184a4",
}
METRICS = ("invalid_rate", "crossed_rate", "kl_spread", "kl_mid_return",
           "abs_return_acf_mae", "depth_ssim")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, label):
    require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), label)


def main():
    report_path = PROJECT / "reports/test/final_test_20261106.json"
    for name in ("plan.json", "evaluate_frozen_test.py"):
        require(sha(ARCHIVE / name) == EXPECTED[name], f"Frozen {name} hash differs")
    require(sha(report_path) == EXPECTED["report"], "Saved report hash differs")
    plan = json.loads((ARCHIVE / "plan.json").read_text(encoding="utf-8"))
    result = json.loads(report_path.read_text(encoding="utf-8"))
    require(result["protocol"] == plan, "Embedded plan differs")
    require(result["plan_sha256"] == EXPECTED["plan.json"], "Recorded plan hash differs")
    require(plan["helper_sha256"] == EXPECTED["evaluate_frozen_test.py"], "Recorded evaluator hash differs")
    require(result["test_evaluated"] and not result["train_or_validation_reused_for_test_selection"], "Test scope differs")
    require(result["test_windows"] == 1838 and result["snapshots_per_generation_seed"] == 58816, "Test size differs")
    models = result["model_results"]
    require(len(models) == 15 and len(plan["models"]) == 15, "Fit count differs")
    for frozen, model in zip(plan["models"], models):
        for key in ("id", "group", "training_seed", "checkpoint", "checkpoint_sha256", "information_budget"):
            require(model[key] == frozen[key], f"{model['id']}: {key} differs")
        require([r["seed"] for r in model["runs"]] == plan["generation_seeds"] == [42, 43, 44], "Generation seeds differ")
        for metric in model["generation_mean"]:
            values = [run["scores"][metric] for run in model["runs"]]
            require(all(math.isfinite(v) for v in values), "Non-finite score")
            close(model["generation_mean"][metric], statistics.mean(values), f"{model['id']}: mean {metric}")
            close(model["generation_sd"][metric], statistics.stdev(values), f"{model['id']}: SD {metric}")
    require(set(result["group_results"]) == {m["group"] for m in models}, "Groups differ")
    for name, group in result["group_results"].items():
        rows = [m for m in models if m["group"] == name]
        require(group["fitted_models"] == len(rows), f"{name}: fit count differs")
        require({r["information_budget"] for r in rows} == {group["information_budget"]}, f"{name}: mixed information budgets")
        for metric in METRICS:
            values = [r["generation_mean"][metric] for r in rows]
            close(group["mean"][metric], statistics.mean(values), f"{name}: mean {metric}")
            if len(rows) > 1:
                close(group["across_fit_sd"][metric], statistics.stdev(values), f"{name}: fit SD {metric}")
            else:
                require(group["across_fit_sd"] is None, f"{name}: single-fit SD")
                close(group["generation_sd"][metric], rows[0]["generation_sd"][metric], f"{name}: generation SD {metric}")
        mean = group["mean"]
        targets = dict(invalidity=mean["invalid_rate"] == 0., spread_kl=mean["kl_spread"] <= .1,
                       return_kl=mean["kl_mid_return"] <= .1, depth_ssim=mean["depth_ssim"] > .6)
        require(group["absolute_targets"] == targets, f"{name}: target flags differ")
    zero = result["group_results"]["Volume zero"]["mean"]
    matched = result["group_results"]["Volume matched"]["mean"]
    for metric in METRICS:
        close(result["volume_pair_difference_matched_minus_zero"][metric], matched[metric] - zero[metric], metric)
    print("PASS: frozen hashes, 15 fits / 45 generation-seed records, means, sample SDs and targets.")
    print("This verifies saved evidence only; no model was trained or run and no files were written.")
    source_matches = sum(sha(PROJECT / name) == digest for name, digest in plan["source_hashes"].items())
    print(f"Current scientific sources: {source_matches}/{len(plan['source_hashes'])} recorded byte hashes match.")
    if source_matches != len(plan["source_hashes"]):
        print("Source bytes differ from the evaluated snapshot; Git line-ending conversion can also change byte hashes.")
    print("Group                  Invalid %  Spread KL  Return KL  ACF error  SSIM")
    for name, group in result["group_results"].items():
        mean = group["mean"]
        print(f"{name:<22} {mean['invalid_rate'] * 100:9.2f} {mean['kl_spread']:10.4f} "
              f"{mean['kl_mid_return']:10.4f} {mean['abs_return_acf_mae']:10.4f} {mean['depth_ssim']:7.4f}")
    print("Historical and volume-conditioned groups have different information budgets; do not pool them.")


if __name__ == "__main__":
    main()
