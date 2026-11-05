"""Fit a small VAR grid on training data and rank common validation rollouts."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import torch

from dataset import PROJECT_DIR, PRICE_COLUMNS, PRICE_DIVISOR, prepare_datasets, sha256_file
from evaluation import (ValidationEvaluator, conditioning_history, decode_batch, metric_specification,
                        validation_windows, validity_rates)
from modules import VectorAR, VARFit, lagged_pairs
from train import make_checkpoint, write_json
from ar_report import source_fingerprint
from protocol import (PRIOR_SEED_OFFSET, comparison_protocol, comparison_signature, sample_training_prior,
                      fit_anchor_context, anchor_context, conditional_training_pairs)


DEFAULT_GRID = {"lags": [1, 2, 4], "ridges": [1e-6, 1e-4, 1e-2], "seeds": [42, 43, 44]}
RANK_KEYS = ("invalid_rate_mean", "mean_kl_mean", "abs_return_acf_mae_mean", "lags", "ridge")


def select_candidate(rows):
    eligible = [row for row in rows if row["status"] == "ok"]
    if not eligible:
        raise RuntimeError("Every candidate failed; inspect per-seed errors before changing the grid.")
    return min(eligible, key=lambda row: tuple(row[key] for key in RANK_KEYS))


def experiment_config(lags, ridges, seeds, sequence_length, threads):
    return {"lags": list(lags), "ridges": list(ridges), "seeds": list(seeds),
            "sequence_length": sequence_length, "threads": threads}


def evaluate_candidate(model, base_row, histories, seeds, length, scaler, anchors, reference, context=None):
    """Evaluate a fitted model without changing the information used to fit it."""
    row, runs, diagnostics, preview = dict(base_row), [], [], None
    for seed in seeds:
        run = {"candidate": row["candidate"], "seed": seed, "status": "ok"}
        try:
            generated_at = perf_counter()
            features = model.generate(histories[seed], length, seed, context=context)
            run["generation_seconds"] = perf_counter() - generated_at
            scores, details = reference.evaluate(decode_batch(features, scaler, anchors))
            run.update(scores)
            diagnostics.append(details)
            if seed == seeds[0]:
                preview = decode_batch(features[:3], scaler, anchors[:3])
        except (FloatingPointError, RuntimeError) as error:
            run.update(status="failed", error=str(error))
            row["status"] = "failed"
        runs.append(run)
    curves, distributions = None, None
    if row["status"] == "ok":
        for name in (k for k in runs[0] if k not in ("candidate", "seed", "status")):
            values = [run[name] for run in runs]
            row[f"{name}_mean"], row[f"{name}_sd"] = float(np.mean(values)), float(np.std(values, ddof=1))
        curves = np.mean([d["acf"] for d in diagnostics], axis=0).tolist()
        distributions = {name: np.mean([d[name] for d in diagnostics], axis=0).tolist() for name in reference.probabilities}
    return row, runs, curves, distributions, preview


def run_grid(data_dir, lags=(1, 2, 4), ridges=(1e-6, 1e-4, 1e-2), seeds=(42, 43, 44),
             sequence_length=32, threads=4, report_path=None, output_dir=None, render=True):
    if not lags or len(set(lags)) != len(lags) or any(not isinstance(p, int) or p < 1 for p in lags):
        raise ValueError("Provide positive integer lag orders.")
    if not ridges or len(set(ridges)) != len(ridges) or any(not np.isfinite(r) or r < 0 for r in ridges):
        raise ValueError("Provide finite non-negative ridge penalties.")
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**63 - PRIOR_SEED_OFFSET for s in seeds):
        raise ValueError("Provide at least two distinct valid generation seeds.")
    if threads < 1:
        raise ValueError("threads must be positive.")
    if sequence_length < 7:
        raise ValueError("Depth SSIM requires at least seven updates per sequence.")
    if 1 not in lags or 1e-4 not in ridges:
        raise ValueError("Keep the original VAR(1), ridge=1e-4, in the comparison grid.")
    torch.set_num_threads(threads)
    report_path = Path(report_path or PROJECT_DIR / "reports/ar/results.json")
    output_dir = Path(output_dir or PROJECT_DIR / "outputs/ar/tuning")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    datasets, scaler, audit = prepare_datasets(data_dir, sequence_length, sequence_length)
    train, validation = datasets["train"], datasets["validation"]
    # Existing full-day loading/audit also prepares test data. No test outcomes are evaluated or used below.
    del datasets
    loading_seconds = perf_counter() - started
    starts, anchors, real = validation_windows(validation, max(lags))
    spec = metric_specification(train, max_lag=min(10, sequence_length - 1))
    train_values = torch.as_tensor(train.features, dtype=torch.float64)
    validation_values = torch.as_tensor(validation.features, dtype=torch.float64)
    reference = ValidationEvaluator(real, anchors, spec)
    protocol = comparison_protocol(sequence_length, max(lags), seeds)
    priors, prior_rows = {}, {}
    for seed in seeds:
        priors[seed], sampled_rows = sample_training_prior(train, len(starts), max(lags), seed)
        prior_rows[str(seed)] = sampled_rows.tolist()
    experiments = {mode: {"candidates": [], "runs": [], "generated_abs_return_acf": {}, "generated_histograms": {}}
                   for mode in ("anchor_only_v1", "observed_history_v1")}
    selected_model, best_row = None, None
    # Small previews are always the first three eligible windows and first declared seed.
    previews = {mode: {"real": real[:3], "anchors": anchors[:3], "raw_start_indices": starts[:3] + validation.raw_offset}
                for mode in experiments}
    for order in lags:
        fitted_at = perf_counter()
        fitter = VARFit(train_values, order)
        setup_seconds = perf_counter() - fitted_at
        history = conditioning_history(validation, starts, order)
        histories = {"anchor_only_v1": {seed: priors[seed][:, -order:] for seed in seeds},
                     "observed_history_v1": {seed: history for seed in seeds}}
        x_val, y_val = lagged_pairs(validation_values, order)
        offset = max(lags) - order  # Identical one-step target rows for every order.
        for ridge in ridges:
            key = f"p{order}_ridge{ridge:g}"
            fitted_at = perf_counter()
            model = fitter.fit(ridge)
            solve_seconds = perf_counter() - fitted_at
            with torch.no_grad():
                validation_mse = float((model(x_val[offset:]) - y_val[offset:]).square().mean())
            row = {"candidate": key, "lags": order, "ridge": ridge, "status": "ok",
                   "fit_seconds": setup_seconds / len(ridges) + solve_seconds, "ridge_solve_seconds": solve_seconds,
                   "decomposition_seconds": setup_seconds, "training_one_step_mse": float(model.residuals.square().mean()),
                   "training_pairs": len(train_values) - order, "validation_one_step_mse": validation_mse,
                   "coefficient_count": model.weight.numel() + model.bias.numel(), "spectral_radius": model.spectral_radius}
            for mode, experiment in experiments.items():
                result, runs, curve, distribution, preview = evaluate_candidate(
                    model, row, histories[mode], seeds, sequence_length, scaler, anchors, reference)
                experiment["candidates"].append(result)
                experiment["runs"].extend(runs)
                if result["status"] == "ok":
                    experiment["generated_abs_return_acf"][key] = curve
                    experiment["generated_histograms"][key] = distribution
                    previews[mode][key] = preview
                    if mode == protocol["id"] and (best_row is None or tuple(result[k] for k in RANK_KEYS) < tuple(best_row[k] for k in RANK_KEYS)):
                        best_row, selected_model = result, model
                print(f"{mode} {key}: {result['status']}; invalid={result.get('invalid_rate_mean', float('nan')):.2%}; SSIM={result.get('depth_ssim_mean', float('nan')):.3f}", flush=True)
        del fitter  # Do not retain the previous order's tall design during the next SVD.

    for mode, experiment in experiments.items():
        selected = select_candidate(experiment["candidates"])
        original = next(row for row in experiment["candidates"] if row["lags"] == 1 and row["ridge"] == 1e-4)
        if original["status"] != "ok":
            raise RuntimeError(f"Original candidate failed in {mode}; cannot build a complete comparison report.")
        experiment.update(protocol_id=mode, selected_candidate=selected["candidate"], original_candidate=original["candidate"])
    primary = experiments[protocol["id"]]
    checkpoint = make_checkpoint(selected_model, train, scaler, {**best_row, "fit_split": "train", "sequence_length": sequence_length,
                                                               "evaluation_protocol": protocol["id"]})
    torch.save(checkpoint, output_dir / "selected_model.pt")
    np.savez_compressed(output_dir / "validation_examples.npz", **previews[protocol["id"]])
    np.savez_compressed(output_dir / "conditional_examples.npz", **previews["observed_history_v1"])
    shuffled = np.random.default_rng(2026).permutation(len(real))
    identity_scores, _ = reference.depth.score(real)
    shuffled_scores, _ = reference.depth.score(real[shuffled])
    report = {
        "format_version": 3, "config": experiment_config(lags, ridges, seeds, sequence_length, threads),
        **primary, "comparison_protocol": protocol, "conditional_validation": experiments["observed_history_v1"],
        "preprocessing_scaler": scaler.to_dict(), "prior_training_anchor_indices": prior_rows,
        "comparison_signature": comparison_signature(protocol, spec, starts + validation.raw_offset, audit["files"], scaler.to_dict()),
        "ssim_controls": {"identity": float(identity_scores.mean()), "shuffled_real": float(shuffled_scores.mean()),
                          "shuffled_seed": 2026, "meaning": "Identity checks the metric; shuffled real diagnoses pairing sensitivity, not a ceiling or selection criterion."},
        "source_fingerprint": source_fingerprint(), "session": audit["session"], "splits": audit["splits"],
        "input_files": audit["files"], "torch_version": str(torch.__version__), "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "fit_split": "train", "selection_split": "validation", "test_evaluated": False, "device": "cpu",
        "validation_windows": len(starts), "snapshots_per_seed": len(starts) * sequence_length,
        "raw_validation_anchor_indices": (starts + validation.raw_offset).tolist(),
        "validation_one_step_target_rows": len(validation.features) - max(lags), "metric_specification": spec,
        "selection_rule": "Lexicographic: lowest seed-mean invalid rate, then mean KL, then ACF MAE, then p and ridge.",
        "real_validity": validity_rates(real), "real_abs_return_acf": reference.acf.tolist(),
        "real_histograms": {name: values.tolist() for name, values in reference.probabilities.items()},
        "decompositions": len(lags), "data_loading_seconds": loading_seconds,
        "total_seconds": perf_counter() - started, "selected_checkpoint_sha256": sha256_file(output_dir / "selected_model.pt"),
        "limitations": [
            "Main anchor_only_v1 comparison supplies only observed starting prices to the decoder; AR initial histories come from training.",
            "observed_history_v1 supplies actual validation histories and is a separate conditional experiment, not the TimeGAN comparison.",
            "All weights, residuals, scaling and histogram boundaries use training only; validation selects hyperparameters.",
            "Existing preprocessing reads/audits the full day; no test model metrics or test-based choices are computed.",
            "Invalid finite books remain in all distribution and ACF metrics. No clipping, sorting, rounding or repair.",
            "Seed standard deviations measure innovation sampling variability, not independent days or confidence intervals.",
            "KL depends on the declared binning and smoothing; 32-update ACF only measures short event-time dependence.",
            "Depth SSIM uses fixed chronological pairing and the declared 7x7 uniform-window variant. It measures volumes, not price validity.",
            "Teacher-forced one-step validation MSE is a separate observed-history diagnostic, not main generation performance.",
            "Final test comparison is pending. TimeGAN validation experiments are recorded separately in timegan_results.ipynb.",
            "The lowest invalid rate is ranked first even when another candidate has lower KL or one-step MSE.",
            "Candidate fit time includes an equal share of the order's SVD setup plus its own ridge solve/residual calculation.",
            "Rollout timing excludes decoding and IO. Historical refactor timings cover one observed-history grid without SSIM, not the new two-protocol run.",
        ],
    }
    # Keep the earlier measured smoke test with the report, rather than refitting or relabelling its numbers.
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        for key in ("initial_experiment", "historical_validation", "refactor_verification"):
            if key in previous:
                report[key] = previous[key]
    if render:
        from ar_report import example_diagnostics, save_figures
        report["example_diagnostics"] = example_diagnostics(report, previews[protocol["id"]])
        report["figures"] = save_figures(report, previews[protocol["id"]], report_path.parent)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def run_conditioned_baseline(data_dir, report_path=None, output_dir=None, threads=4):
    """Fixed VARX(2), ridge 0.01, with a zero-context control; no new grid or test scoring."""
    torch.set_num_threads(threads)
    report_path = Path(report_path or PROJECT_DIR / "reports/ar/anchor_conditioned.json")
    output_dir = Path(output_dir or PROJECT_DIR / "outputs/ar/anchor_conditioned")
    output_dir.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        raise ValueError("Conditional AR report exists; preserve it and choose a new destination.")
    datasets, scaler, audit = prepare_datasets(data_dir, 32, 32)
    training, validation = datasets["train"], datasets["validation"]
    del datasets
    normalisation = fit_anchor_context(training.book[training.starts][:, PRICE_COLUMNS] / PRICE_DIVISOR)
    protocol = comparison_protocol(context_scaler=normalisation)
    starts, anchors, real = validation_windows(validation, 4)
    spec = metric_specification(training)
    reference = ValidationEvaluator(real, anchors, spec)
    previous, following, contexts = conditional_training_pairs(training, normalisation)
    histories = {seed: sample_training_prior(training, len(starts), 4, seed)[0][:, -2:] for seed in (42, 43, 44)}
    rows, all_runs, replays = [], [], {}
    for mode in ("zero", "matched"):
        fitted_at = perf_counter()
        inputs = np.zeros_like(contexts) if mode == "zero" else contexts
        fitter = VARFit.from_pairs(torch.from_numpy(previous), torch.from_numpy(following), torch.from_numpy(inputs))
        model = fitter.fit(.01)
        fit_seconds = perf_counter() - fitted_at
        del fitter
        key = f"p2_ridge0.01_{mode}"
        context = anchor_context(anchors, normalisation, mode)
        base = dict(candidate=key, status="ok", lags=2, ridge=.01, anchor_conditioning=mode,
            fit_seconds=fit_seconds, coefficient_count=sum(v.numel() for k,v in model.state_dict().items() if k != "residuals"),
            training_pairs=len(following), training_one_step_mse=float(model.residuals.square().mean()), spectral_radius=model.spectral_radius)
        row, runs, _, _, preview = evaluate_candidate(model, base, histories, (42,43,44), 32, scaler, anchors, reference, context)
        if row["status"] != "ok":
            raise RuntimeError(f"Conditional AR rollout failed: {runs}")
        checkpoint = make_checkpoint(model, training, scaler, dict(base, sequence_length=32))
        checkpoint["anchor_context_scaler"] = normalisation
        path = output_dir / f"{mode}.pt"
        torch.save(checkpoint, path)
        restored = VectorAR(**torch.load(path, weights_only=True)["state_dict"])
        replay = decode_batch(restored.generate(histories[42], 32, 42, context)[:3], scaler, anchors[:3])
        difference = float(np.abs(replay-preview).max())
        if difference != 0:
            raise RuntimeError("Conditional AR checkpoint did not replay exactly.")
        row.update(checkpoint=path.relative_to(PROJECT_DIR).as_posix(), checkpoint_sha256=sha256_file(path), reload_max_abs_difference=difference)
        np.savez_compressed(output_dir/f"{mode}_examples.npz", real=real[:3], generated=preview, anchors=anchors[:3])
        rows.append(row); all_runs.extend(runs); replays[mode] = difference
        print(mode, "invalid", row["invalid_rate_mean"], "return KL", row["kl_mid_return_mean"], flush=True)
    report = dict(format_version=1, candidates=rows, runs=all_runs, original_candidate=rows[0]["candidate"],
        selected_candidate=rows[1]["candidate"], selection_rule="Fixed previous VAR(2)/ridge=0.01; matched context is the predeclared baseline, zero is its ablation.",
        comparison_protocol=protocol, comparison_signature=comparison_signature(protocol,spec,starts+validation.raw_offset,audit["files"],scaler.to_dict()),
        source_fingerprint=source_fingerprint(), preprocessing_scaler=scaler.to_dict(), context_scaler=normalisation,
        metric_specification=spec, validation_windows=len(starts), test_evaluated=False, checkpoint_replays=replays,
        limitations=["Training pairs cover complete windows with four past rows; static context is fixed within each window.",
            "Both controls use the same pairs, uniform training-history bootstrap and generation seeds; only context values differ.",
            "This is a new conditioning protocol. Historical anchor-only scores remain supplementary, not an equal-information comparison.",
            "Context does not guarantee valid prices; no clipping, sorting, rounding or repair is used."])
    write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--lags", type=int, nargs="+", default=DEFAULT_GRID["lags"])
    parser.add_argument("--ridges", type=float, nargs="+", default=DEFAULT_GRID["ridges"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_GRID["seeds"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--anchor-conditioned", action="store_true", help="Run the fixed VARX baseline and its zero-context control.")
    args = parser.parse_args()
    if args.anchor_conditioned:
        run_conditioned_baseline(args.data_dir, threads=args.threads)
        return
    report = run_grid(args.data_dir, args.lags, args.ridges, args.seeds, threads=args.threads)
    print(f"Selected {report['selected_candidate']} under {report['protocol_id']}; saved reports/ar/results.json (validation only).")


if __name__ == "__main__":
    main()
