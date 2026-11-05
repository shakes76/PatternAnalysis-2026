"""One frozen, inference-only test assessment. Never edits existing reports or READMEs."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

PROJECT = Path(r"C:/Users/franz/OneDrive/Documents/FinMath/Sem4/COMP3710/Project/2.7/PatternAnalysis-2026/recognition/timegan_spagheddy7")
DATA = PROJECT.parents[2] / "LOBSTER_SampleFile_AMZN_2012-06-21_10"
STUDY = PROJECT / "outputs/timegan/final_test_20261106"
DEST = PROJECT / "reports/test/final_test_20261106.json"
SOURCE_NAMES = ["dataset.py", "evaluation.py", "modules.py", "predict.py", "protocol.py", "train.py", "tune_timegan.py", "timegan_report.py"]
METRICS = ["invalid_rate", "crossed_rate", "kl_spread", "kl_mid_return", "abs_return_acf_mae", "depth_ssim"]

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")

def update_progress(value):
    p = STUDY / "progress.json"
    temp = p.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, p)

def prepare():
    if DEST.exists() or (STUDY / "plan.json").exists():
        raise RuntimeError("A frozen plan/report already exists; inspect it, do not redefine the cohort.")
    ar = read(PROJECT / "reports/ar/results.json")
    ledger = read(PROJECT / "reports/timegan/tuning.json")
    volume = read(PROJECT / "reports/timegan/volume_recovery_study.json")
    ar_selected = next(x for x in ar["candidates"] if x["candidate"] == ar["selected_candidate"])
    models = [dict(id="var2", group="VAR(2)", family="ar", checkpoint="outputs/ar/tuning/selected_model.pt",
                   expected_hash=ar["selected_checkpoint_sha256"], checkpoint_rule="Previously validation-selected VAR(2), ridge 0.01",
                   information_budget="anchor_only_v1", training_seed=None, expected_step=None,
                   validation_reference={k: ar_selected[k + "_mean"] for k in METRICS})]
    for prefix, label in [("night_reference", "Reference"), ("night_gentle", "Gentle penalty"),
                          ("order_mid", "Moderate penalty"), ("no_supervision", "No supervision")]:
        for seed in (2026, 2027, 2028):
            name = f"{prefix}_s{seed}"
            entry = ledger["candidates"][name]
            models.append(dict(id=name, group=label, family="timegan", checkpoint=entry["checkpoint"],
                               expected_hash=entry["selected_checkpoint_sha256"], training_seed=seed,
                               expected_step=entry["selected"]["step"], expected_config=entry["config"],
                               checkpoint_rule="Previously validation-selected checkpoint; no test selection",
                               information_budget="anchor_only_v1",
                               validation_reference={k: entry["selected"][k + "_mean"] for k in METRICS}))
    for mode in ("zero", "matched"):
        audited = volume["audit"]["models"][mode]
        final = audited["checkpoints"]["final"]
        print("FINAL AUDIT KEYS", mode, list(final), flush=True)
        # Derive no selection from test: the recorded primary is joint 4000.
        ck = f"outputs/timegan/experiments/volume_recovery_{mode}_s2026/joint_last.pt"
        expected = final.get("checkpoint_sha256", final.get("sha256"))
        if expected is None:
            raise ValueError("Inspect the existing final-checkpoint audit hash field before freezing.")
        validation = final.get("metrics", final.get("summary", final))
        models.append(dict(id=f"volume_{mode}", group=f"Volume {mode}", family="timegan", checkpoint=ck,
                           expected_hash=expected, training_seed=2026, expected_step=4000,
                           expected_config=audited["config"], checkpoint_rule="Predetermined final joint 4000; selected checkpoint excluded",
                           information_budget="anchor_price_volume_conditioned_v1",
                           validation_reference=validation))
    for model in models:
        model["checkpoint_sha256"] = sha(PROJECT / model["checkpoint"])
        assert model["checkpoint_sha256"] == model.pop("expected_hash"), model["id"]
    assert len(models) == 15
    for item in ar["input_files"]:
        f = DATA / item["name"]
        assert f.stat().st_size == item["bytes"] and sha(f) == item["sha256"]
    plan = dict(created_utc=now(), authorization="User explicitly requested final test assessment on 6 November Brisbane; all READMEs must remain untouched.",
                scope="One assessment of all prespecified reported model groups, not a test-based model search.",
                models=models, generation_seeds=[42, 43, 44], device="cpu", threads=1,
                split="test", session_interval="15:01:30–16:00 (last 15% of elapsed session time)",
                sequence_length=32, stride=32, minimum_within_split_history=4,
                input_files=ar["input_files"], metric_specification=ar["metric_specification"],
                preprocessing_scaler=ar["preprocessing_scaler"],
                source_hashes={n: sha(PROJECT / n) for n in SOURCE_NAMES},
                helper_sha256=sha(__file__),
                protected_readmes={str(f.relative_to(PROJECT)): sha(f) for f in PROJECT.glob("*README.md")},
                report_hashes={n: sha(PROJECT / n) for n in ["reports/ar/results.json", "reports/timegan/tuning.json", "reports/timegan/volume_recovery_study.json"]},
                aggregation="Average three generation seeds within each fit, then equal-weight fit means within each historical group. Across-fit SD and generation-seed SD are distinct.",
                exclusions=["No training or optimizer calls", "No checkpoint selection on test", "No clipping, sorting, repair or invalid-sample removal", "No pooling across information budgets", "No README or historical-report edits", "No claim that the validation fixed-context diversity screen was retested"],
                targets={"kl_spread_max": .1, "kl_mid_return_max": .1, "depth_ssim_strict_min": .6, "invalid_rate": 0.0})
    write_new(STUDY / "plan.json", plan)
    print(json.dumps({"status": "frozen", "fits": len(models), "generation_calls": 45, "plan_sha256": sha(STUDY / "plan.json")}), flush=True)

def run():
    import numpy as np
    import torch
    import psutil
    sys.path.insert(0, str(PROJECT))
    from dataset import prepare_datasets, PRICE_COLUMNS, VOLUME_COLUMNS
    from evaluation import validation_windows, metric_specification, ValidationEvaluator, decode_batch, validity_rates
    from protocol import sample_training_prior, anchor_context, fit_anchor_context
    from predict import load_timegan
    from modules import VectorAR

    plan = read(STUDY / "plan.json")
    assert sha(__file__) == plan["helper_sha256"], "Evaluator changed after plan freeze"
    assert not DEST.exists(), "Final report already exists"
    for n, h in plan["source_hashes"].items():
        assert sha(PROJECT / n) == h, n
    for m in plan["models"]:
        assert sha(PROJECT / m["checkpoint"]) == m["checkpoint_sha256"], m["id"]
    proc = psutil.Process()
    identities = [dict(pid=proc.pid, create_time=proc.create_time(), cmdline=proc.cmdline())]
    parent = proc.parent()
    if parent and str(Path(__file__).resolve()) in " ".join(parent.cmdline()):
        identities.append(dict(pid=parent.pid, create_time=parent.create_time(), cmdline=parent.cmdline()))
    write_new(STUDY / "started.json", dict(started_utc=now(), plan_sha256=sha(STUDY / "plan.json"), identities=identities))
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = time.perf_counter()
    datasets, scaler, audit = prepare_datasets(DATA, 32, 32)
    training, test = datasets["train"], datasets["test"]
    assert audit["files"] == plan["input_files"]
    assert scaler.to_dict() == plan["preprocessing_scaler"]
    spec = metric_specification(training)
    assert spec == plan["metric_specification"], "Training-derived evaluation specification changed"
    starts, anchors, real = validation_windows(test, 4)
    raw_starts = starts + test.raw_offset
    assert np.all(starts >= 4) and np.all(np.diff(starts) == 32)
    assert raw_starts.min() >= audit["splits"]["test"]["raw_row_interval_half_open"][0]
    assert int(raw_starts[-1] + 32) < audit["splits"]["test"]["raw_row_interval_half_open"][1]
    reference = ValidationEvaluator(real, anchors, spec)
    identity, _ = reference.depth.score(real)
    assert np.max(np.abs(identity - 1.0)) < 1e-12
    priors = {s: sample_training_prior(training, len(starts), 4, s) for s in plan["generation_seeds"]}
    volume_anchors = test.book[starts][:, VOLUME_COLUMNS].astype(np.float64)
    fitted_context = fit_anchor_context(training.book[training.starts][:, PRICE_COLUMNS] / 10000.,
                                       training.book[training.starts][:, VOLUME_COLUMNS].astype(np.float64))
    result = dict(created_utc=now(), plan_sha256=sha(STUDY / "plan.json"), protocol=plan,
                  test_evaluated=True, train_or_validation_reused_for_test_selection=False,
                  test_windows=len(starts), snapshots_per_generation_seed=int(len(starts) * 32),
                  raw_test_anchor_indices=raw_starts.tolist(), splits=audit["splits"],
                  actual_scored_future_seconds=[float(test.times[starts[0] + 1]), float(test.times[starts[-1] + 32])],
                  real_validity=validity_rates(real), identity_ssim=float(identity.mean()),
                  model_results=[], group_results={},
                  environment=dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__))
    for item in plan["models"]:
        checkpoint_path = PROJECT / item["checkpoint"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        np.testing.assert_array_equal(checkpoint["scaler"]["minimum"].numpy(), scaler.minimum)
        np.testing.assert_array_equal(checkpoint["scaler"]["maximum"].numpy(), scaler.maximum)
        context = None
        if item["family"] == "ar":
            assert checkpoint["fit_report"]["candidate"] == "p2_ridge0.01"
            assert checkpoint["fit_report"]["evaluation_protocol"] == "anchor_only_v1"
            model = VectorAR(**checkpoint["state_dict"]).eval()
            assert model.lags == 2
        else:
            model, restored_scaler, loaded = load_timegan(checkpoint_path, "cpu")
            assert checkpoint["stage"] == "joint" and checkpoint["step"] == item["expected_step"]
            assert checkpoint["config"] == item["expected_config"]
            cfg = checkpoint["config"]
            mode = cfg.get("volume_conditioning", "none")
            if mode != "none":
                assert checkpoint["anchor_context_scaler"] == fitted_context
                assert cfg["anchor_conditioning"] == "matched" and cfg["price_decoder"] == "ladder"
            else:
                assert cfg.get("anchor_conditioning", "none") == "none"
            context = anchor_context(anchors, checkpoint.get("anchor_context_scaler"), cfg.get("anchor_conditioning", "none"),
                                     volumes=volume_anchors if mode == "matched" else None, volume_mode=mode)
        runs = []
        with torch.inference_mode():
            for seed in plan["generation_seeds"]:
                tick = time.perf_counter()
                if item["family"] == "ar":
                    features = model.generate(priors[seed][0][:, -model.lags:], 32, seed)
                else:
                    features = model.generate(len(starts), 32, seed, checkpoint["config"]["batch_size"], context=context, anchors=anchors)
                generation_seconds = time.perf_counter() - tick
                assert np.isfinite(features.numpy()).all()
                books = decode_batch(features, scaler, anchors)
                scores, details = reference.evaluate(books)
                assert all(np.isfinite(v) for v in scores.values())
                runs.append(dict(seed=seed, generation_seconds=generation_seconds, scores=scores,
                                 generated_features_sha256=hashlib.sha256(features.contiguous().numpy().tobytes()).hexdigest(),
                                 abs_return_acf=details["acf"].tolist()))
                del features, books
        keys = list(runs[0]["scores"])
        means = {k: float(np.mean([r["scores"][k] for r in runs])) for k in keys}
        sds = {k: float(np.std([r["scores"][k] for r in runs], ddof=1)) for k in keys}
        model_result = dict(id=item["id"], group=item["group"], training_seed=item["training_seed"], checkpoint=item["checkpoint"],
                            checkpoint_sha256=item["checkpoint_sha256"], information_budget=item["information_budget"],
                            runs=runs, generation_mean=means, generation_sd=sds)
        write_new(STUDY / "models" / f"{item['id']}.json", model_result)
        result["model_results"].append(model_result)
        print(json.dumps({"completed_fit": item["id"], "n": len(result["model_results"]), "of": 15}), flush=True)
        update_progress(dict(status="running", completed_models=len(result["model_results"]), last_model=item["id"], updated_utc=now()))
        del model, checkpoint
    for group in dict.fromkeys(x["group"] for x in result["model_results"]):
        rows = [x for x in result["model_results"] if x["group"] == group]
        means = {k: float(np.mean([x["generation_mean"][k] for x in rows])) for k in METRICS}
        sd = {k: float(np.std([x["generation_mean"][k] for x in rows], ddof=1)) for k in METRICS} if len(rows) > 1 else None
        result["group_results"][group] = dict(information_budget=rows[0]["information_budget"], fitted_models=len(rows),
            mean=means, across_fit_sd=sd, generation_sd=rows[0]["generation_sd"] if len(rows) == 1 else None,
            absolute_targets=dict(invalidity=means["invalid_rate"] == 0., spread_kl=means["kl_spread"] <= .1,
                return_kl=means["kl_mid_return"] <= .1, depth_ssim=means["depth_ssim"] > .6))
    zero = result["group_results"]["Volume zero"]["mean"]
    matched = result["group_results"]["Volume matched"]["mean"]
    result["volume_pair_difference_matched_minus_zero"] = {k: matched[k] - zero[k] for k in METRICS}
    result["volume_test_screen_scope"] = "Core metrics only; the validation fixed-context diversity screen was not recomputed and no full-screen pass is claimed."
    result["verification"] = dict(source_hashes_unchanged=all(sha(PROJECT / n) == h for n,h in plan["source_hashes"].items()),
        checkpoint_hashes_unchanged=all(sha(PROJECT / m["checkpoint"]) == m["checkpoint_sha256"] for m in plan["models"]),
        historical_reports_unchanged=all(sha(PROJECT / n) == h for n,h in plan["report_hashes"].items()),
        readme_current_hashes={n:sha(PROJECT / n) for n in plan["protected_readmes"]}, readme_write_operations=0,
        weights_fitted=False, model_generation_calls=45, test_checkpoint_selection=False)
    assert result["verification"]["source_hashes_unchanged"] and result["verification"]["checkpoint_hashes_unchanged"] and result["verification"]["historical_reports_unchanged"]
    result.update(completed_utc=now(), total_seconds=time.perf_counter() - started)
    write_new(DEST, result)
    write_new(STUDY / "completion.json", dict(completed_utc=now(), report=str(DEST), report_sha256=sha(DEST), total_seconds=result["total_seconds"]))
    update_progress(dict(status="complete", completed_models=15, updated_utc=now()))
    print(json.dumps({"status":"complete", "test_windows":len(starts), "seconds":result["total_seconds"], "groups":result["group_results"]}), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run"])
    args = parser.parse_args()
    try:
        prepare() if args.action == "prepare" else run()
    except Exception:
        if args.action == "run" and (STUDY / "started.json").exists():
            error = dict(failed_utc=now(), traceback=traceback.format_exc())
            if not (STUDY / "failure.json").exists():
                write_new(STUDY / "failure.json", error)
            update_progress(dict(status="failed", updated_utc=now()))
        raise
