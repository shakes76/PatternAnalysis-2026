"""Reconstruction diagnostics and reproducible conditioning reports; no model training."""

import json
from pathlib import Path

import numpy as np
import torch

from dataset import PRICE_COLUMNS, VOLUME_COLUMNS, PROJECT_DIR, prepare_datasets, sha256_file
from evaluation import decode_batch, validation_windows, validity_rates, ValidationEvaluator, depth_images
from predict import load_timegan
from protocol import anchor_context, fit_anchor_context
from train import timegan_code_hashes, write_json


def reconstruction_scores(features, recovered, training_mean, scaler, anchors, real):
    """Observed-input diagnostics; the constant predictor is fitted on training only."""
    if features.shape != recovered.shape or features.ndim != 3 or features.shape[-1] != 40:
        raise ValueError("Expected matching (windows, updates, 40) reconstruction arrays.")
    if training_mean.shape != (40,) or not all(torch.isfinite(x).all() for x in (features, recovered, training_mean)):
        raise ValueError("Reconstruction inputs and training means must be finite.")
    mse = (features - recovered).square().mean((0, 1)).double().numpy()
    baseline = (features - training_mean).square().mean((0, 1)).double().numpy()
    variance = features.double().var((0, 1), unbiased=False).numpy()
    groups = {}
    for name, columns in (("price", PRICE_COLUMNS), ("volume", VOLUME_COLUMNS)):
        error, constant, var = (float(a[columns].mean()) for a in (mse, baseline, variance))
        if min(constant, var) <= 0:
            raise ValueError("Reconstruction groups need positive baseline error and variance.")
        groups[name] = dict(mse=error, training_mean_mse=constant, mse_over_training_mean=error / constant,
                            variance=var, mse_over_variance=error / var)
    decoded = decode_batch(recovered, scaler, anchors)
    if not np.isfinite(decoded).all():
        raise FloatingPointError("Non-finite decoded reconstruction.")
    price_error = decoded[..., PRICE_COLUMNS] - real[..., PRICE_COLUMNS]
    return dict(feature_groups=groups, reconstruction=validity_rates(decoded), real=validity_rates(real),
                price_rmse_usd=float(np.sqrt(np.square(price_error).mean())),
                endpoint_price_rmse_usd=float(np.sqrt(np.square(price_error[:, -1]).mean())),
                mean_price_error_by_update_usd=price_error.mean((0, 2)).tolist(),
                invalid_by_update=[validity_rates(decoded[:, i:i + 1])["invalid_rate"] for i in range(features.shape[1])],
                finite=True, windows=len(features))


def reconstruction_screen(candidate, control):
    """Fixed-final pilot gate, declared before fitting; not a simulator quality claim."""
    candidate, control = candidate["validation"], control["validation"]
    price = candidate["feature_groups"]["price"]["mse"] / control["feature_groups"]["price"]["mse"]
    volume = candidate["feature_groups"]["volume"]["mse"] / control["feature_groups"]["volume"]["mse"]
    invalid = candidate["reconstruction"]["invalid_rate"] - control["reconstruction"]["invalid_rate"]
    endpoint = candidate["endpoint_price_rmse_usd"] / control["endpoint_price_rmse_usd"]
    values = (price, volume, invalid, endpoint)
    if not np.isfinite(values).all():
        raise ValueError("Screen values must be finite.")
    return [dict(criterion=label, passed=bool(passed), value=float(value)) for label, passed, value in (
        ("Price MSE at least 20% lower", price <= .8, price),
        ("Volume MSE increase at most 5%", volume <= 1.05, volume),
        ("Reconstructed invalidity increase at most 2 percentage points", invalid <= .02, invalid),
        ("Endpoint price RMSE increase at most 5%", endpoint <= 1.05, endpoint))]


def save_reconstruction_figures(report, audit_folder, destination):
    """Fixed cases and common scales for the reconstruction-to-generation study."""
    import matplotlib.pyplot as plt
    destination, audit_folder = Path(destination), Path(audit_folder)
    def save(fig, key):
        path = destination / f"reconstruction_{key}.png"
        fig.savefig(path, dpi=145)
        plt.close(fig)
        report["figures"][key] = dict(file=path.name, sha256=sha256_file(path))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for name, run in sorted(report["reconstruction_runs"].items()):
        label = name.replace("er_", "").replace("_s2026", "").replace("_", " ")
        rows = [r for r in run["evaluations"] if r["step"] > 0]
        for ax, group in zip(axes[0], ("price", "volume")):
            ax.plot([r["step"] for r in rows], [r["validation"]["feature_groups"][group]["mse_over_training_mean"] for r in rows], marker="o", label=label)
            ax.set(title=f"Validation {group} MSE / training-mean baseline", xlabel="Embedding update")
        axes[1, 0].plot([r["step"] for r in rows], [100*r["validation"]["reconstruction"]["invalid_rate"] for r in rows], marker="o", label=label)
        axes[1, 1].plot(range(1, 33), run["final"]["validation"]["mean_price_error_by_update_usd"], label=label)
    axes[1, 0].set(title="Reconstructed invalid snapshots (%)", xlabel="Embedding update")
    axes[1, 1].set(title="Fixed-final mean price error (USD)", xlabel="Within-window update")
    axes[1, 1].axhline(0, color="black", linewidth=.7)
    for ax in axes.ravel(): ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Observed-input E/R probes; one training seed, no generator training")
    save(fig, "probes")

    audit = report["full_audit"]
    metrics = [("invalid_rate", "Invalid (%)", 100), ("kl_spread", "Spread KL", 1), ("kl_mid_return", "Return KL", 1),
               ("abs_return_acf_mae", "ACF error", 1), ("depth_ssim", "Depth SSIM", 1)]
    rows = [("control", "selected"), ("candidate", "selected"), ("candidate", "final")]
    fig, axes = plt.subplots(1, 5, figsize=(14, 4), layout="constrained")
    for ax, (key, title, scale) in zip(axes, metrics):
        scores = [audit["models"][label]["checkpoints"][stage]["score"] for label, stage in rows]
        ax.bar(range(3), [r[key+"_mean"]*scale for r in scores], yerr=[r[key+"_sd"]*scale for r in scores], capsize=3,
               color=["#8291a3", "#187caa", "#dc8b32"])
        ax.set(title=title, xticks=range(3), xticklabels=["Control\n5000", "Candidate\nselected 3000", "Candidate\nfinal 5000"])
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Full TimeGAN pilot; error bars show generation-seed SD, not retraining uncertainty")
    save(fig, "full_metrics")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for label, model in audit["models"].items():
        for ax, stage, key, title in zip(axes[0], ("embedding", "supervised", "joint"),
                ("reconstruction_mse", "supervised_mse", "generator_adversarial"),
                ("Unweighted reconstruction warm-up MSE", "Supervisor warm-up MSE", "Joint generator adversarial loss")):
            records = [r for r in model["training_history"] if r["stage"] == stage]
            ax.plot([r["step"] for r in records], [r[key] for r in records], label=label)
            ax.set(title=title, xlabel="Stage update")
        records = model["validation_history"]
        for ax, key, title, scale in zip(axes[1], ("invalid_rate", "kl_mid_return", "depth_ssim"),
                ("Validation invalid (%)", "Validation return KL", "Validation depth SSIM"), (100, 1, 1)):
            ax.plot([r["step"] for r in records], [r[key+"_mean"]*scale for r in records], marker="o", label=label)
            ax.set(title=title, xlabel="Joint update")
    for ax in axes.ravel(): ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle("Training diagnostics and validation trajectories; lower losses do not certify quality")
    save(fig, "full_training")

    samples = {}
    for label, stage in rows:
        with np.load(audit_folder / f"{label}_{stage}_cases.npz") as saved:
            samples[label+" "+stage] = saved["generated"]
            samples["Real"] = saved["real"]
    order = ["Real", "control selected", "candidate selected", "candidate final"]
    spec = report["depth_specification"]
    images = {label: depth_images(samples[label], spec) for label in order}
    fig, axes = plt.subplots(3, 4, figsize=(14, 8), layout="constrained")
    for i, window in enumerate((1, 540, 1080)):
        for j, label in enumerate(order):
            picture = np.concatenate([images[label][i, 0, ::-1], images[label][i, 1]], axis=0)
            heat = axes[i, j].imshow(picture, aspect="auto", vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
            axes[i, j].set(title=f"Window {window}: {label}", xlabel="Update", yticks=[0, 9.5, 19], yticklabels=["Ask 10", "Best", "Bid 10"])
    fig.colorbar(heat, ax=axes.ravel().tolist(), shrink=.8, label="log(1 + volume) / training maximum")
    save(fig, "depth")
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), layout="constrained")
    for i, window in enumerate((1, 540, 1080)):
        for label in order:
            book = samples[label][i]
            gaps = np.c_[np.diff(book[:, 0::4], axis=1), -np.diff(book[:, 2::4], axis=1), book[:, 0]-book[:, 2]]
            axes[i, 0].plot(range(1, 33), gaps.min(1), label=label)
            axes[i, 1].plot(range(1, 33), book[:, 0]-book[:, 2], label=label)
        for ax, title in zip(axes[i], ("Minimum of 19 price gaps", "Best spread")):
            ax.set(title=f"Window {window}: {title}", xlabel="Update", ylabel="USD")
            ax.axhline(0, color="black", linewidth=.7)
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    save(fig, "gaps")


def shifted_anchors(anchors, seed):
    """Move every window index by a seeded nonzero shift; prices can still repeat."""
    if len(anchors) < 2:
        raise ValueError("Anchor diagnostic requires at least two windows.")
    shift = int(np.random.default_rng(seed).integers(1, len(anchors)))
    return np.roll(anchors, shift, axis=0), shift


def check_decoder_pair(runs):
    """Require paired fresh fits differing only in the configured price decoder."""
    settings = []
    for mode in ("ordinary", "ladder"):
        run = runs[mode]
        config = json.loads(json.dumps(run["config"]))
        expected = "none" if mode == "ordinary" else "ladder"
        if config.get("price_decoder", "none") != expected or config["architecture"].get("price_decoder", "none") != expected:
            raise ValueError("Incorrect decoder pair labels.")
        if config.get("anchor_conditioning") != "matched" or run.get("test_evaluated") is not False:
            raise ValueError("Use matched-anchor validation fits only.")
        for key in ("price_decoder", "ladder_gap_scale", "objective"):
            config.pop(key, None)
        config["architecture"].pop("price_decoder", None)
        settings.append(config)
    if settings[0] != settings[1]:
        raise ValueError("Decoder controls must have matching budgets and settings.")
    for key in ("source_files", "comparison_signature", "metric_specification", "parameter_count",
                "initial_parameters_sha256", "minibatch_stream_sha256", "noise_stream_sha256"):
        if key not in runs["ordinary"] or runs["ordinary"][key] != runs["ladder"].get(key):
            raise ValueError("Decoder pair mismatch: " + key)


def audit_generation_runs(data_dir, run_dirs, destination):
    """Replay selected/final checkpoints, fixed-anchor noise and observed-input recovery.

    No training or test scoring. Generation-seed SD describes one fitted model,
    and the extra correlation diagnostic is descriptive, not a selection rule.
    """
    torch.set_num_threads(1)
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Audit output already exists; preserve it and choose a fresh folder.")
    destination.mkdir(parents=True, exist_ok=True)
    prepared = prepare_datasets(data_dir, 32, 32)
    validation, fitted_scaler = prepared[0]["validation"], prepared[1]
    starts, anchors, real = validation_windows(validation, 4)
    actual = np.stack([validation.features[s:s + 32] for s in starts])
    real_sd = actual.reshape(-1, 40).std(0)
    cases = np.array([0, (len(starts) - 1) // 2, len(starts) - 1])
    current_sources = timegan_code_hashes()
    result = dict(test_evaluated=False, source_files=current_sources, analysis_sha256=sha256_file(Path(__file__)),
                  scope="One training seed per mode; noise-seed SD is not retraining uncertainty.", models={}, cases=[], noise_rows=[], resources=[])
    signature = None
    for label, run_dir in run_dirs.items():
        folder = Path(run_dir)
        run = json.loads((folder / "results.json").read_text(encoding="utf-8"))
        if run["test_evaluated"] or run["source_files"] != current_sources:
            raise ValueError("Use validation fits with their unchanged model sources.")
        signature = signature or run["comparison_signature"]
        if signature != run["comparison_signature"]:
            raise ValueError("Run comparison protocols differ.")
        reference = ValidationEvaluator(real, anchors, run["metric_specification"])
        result["models"][label] = dict(name=folder.name, config=run["config"], training_history=run["history"],
                                       validation_history=run["validation_history"], checkpoints={})
        result["resources"].append(dict(model=label, parameters=run["parameter_count"], fit_minutes=run["total_seconds"] / 60,
                                        stage_training_seconds=run["stage_training_seconds"], device=run["device"]))
        for stage, filename, score in (("selected", "selected.pt", run["selected"]), ("final", "joint_last.pt", run["validation_history"][-1])):
            path = folder / filename
            model, scaler, checkpoint = load_timegan(path)
            assert checkpoint["source_files"] == current_sources and checkpoint["config"] == run["config"]
            assert checkpoint["step"] == score["step"] and checkpoint["comparison_signature"] == signature
            np.testing.assert_array_equal(scaler.minimum, fitted_scaler.minimum)
            np.testing.assert_array_equal(scaler.maximum, fitted_scaler.maximum)
            if stage == "selected":
                assert sha256_file(path) == run["selected_checkpoint_sha256"]
            volume_mode = run["config"].get("volume_conditioning", "none")
            if volume_mode != "none":
                training = prepared[0]["train"]
                training_prices = training.book[training.starts][:, PRICE_COLUMNS] / 10000
                training_volumes = training.book[training.starts][:, VOLUME_COLUMNS]
                expected_scaler = fit_anchor_context(training_prices, training_volumes)
                assert checkpoint["anchor_context_scaler"] == run["anchor_context_scaler"] == expected_scaler
                np.testing.assert_array_equal(checkpoint["training_volume_anchors"].numpy(), training_volumes)
            volume_anchors = None if volume_mode == "none" else validation.book[starts][:, VOLUME_COLUMNS]
            context = anchor_context(anchors, checkpoint.get("anchor_context_scaler"), run["config"].get("anchor_conditioning", "none"), volumes=volume_anchors, volume_mode=volume_mode)
            noise = []
            shifted_context_runs = []
            covariance = []
            for recorded in score["runs"]:
                seed = recorded["seed"]
                features = model.generate(len(starts), 32, seed, 128, context=context, anchors=anchors)
                assert torch.isfinite(features).all()
                books = decode_batch(features, scaler, anchors)
                assert np.isfinite(books).all()
                scores, details = reference.evaluate(books)
                for key, value in scores.items():
                    assert np.isclose(value, recorded[key], rtol=1e-12, atol=1e-12), (label, stage, seed, key)
                for index in cases:
                    fixed_anchors = np.repeat(anchors[index:index+1], 128, axis=0)
                    fixed_context = None if context is None else np.repeat(context[index:index+1], 128, axis=0)
                    generated = model.generate(128, 32, seed, 128, context=fixed_context, anchors=fixed_anchors).numpy()
                    sd = generated.std(0).mean(0)
                    noise.append(dict(seed=seed, validation_window=int(index+1), price_sd=float(sd[::2].mean()), volume_sd=float(sd[1::2].mean()),
                                      price_ratio=float(sd[::2].mean() / real_sd[::2].mean()),
                                      volume_ratio=float(sd[1::2].mean() / real_sd[1::2].mean()),
                                      unique_sequences=int(len(np.unique(generated.reshape(128, -1), axis=0)))))
                if context is not None:
                    moved, shift = shifted_anchors(anchors, seed)
                    # Volume pilots shift only the volume context; price information stays fixed.
                    moved_context = anchor_context(moved if volume_mode == "none" else anchors, checkpoint.get("anchor_context_scaler"), run["config"]["anchor_conditioning"], volumes=None if volume_anchors is None else np.roll(volume_anchors, shift, axis=0), volume_mode=volume_mode)
                    moved_features = model.generate(len(starts), 32, seed, 128, context=moved_context, anchors=anchors)
                    moved_scores, _ = reference.evaluate(decode_batch(moved_features, scaler, anchors))
                    shifted_context_runs.append(dict(seed=seed, cyclic_shift=shift, mean_feature_change=float((moved_features-features).abs().mean()), **moved_scores))
                price_real, price_fake = actual[..., ::2].reshape(-1, 20), features.numpy()[..., ::2].reshape(-1, 20)
                mask = np.triu(np.ones((20, 20), dtype=bool), 1)
                covariance.append(dict(seed=seed, price_correlation_mae=float(np.abs(np.corrcoef(price_real.T) - np.corrcoef(price_fake.T))[mask].mean())))
                if seed == 42:
                    if stage == "selected":
                        with np.load(folder / "validation_examples.npz") as saved:
                            np.testing.assert_array_equal(books[:3], saved["generated"])
                    np.savez_compressed(destination / f"{label}_{stage}_cases.npz", real=real[cases], generated=books[cases], anchors=anchors[cases])
                    for index in cases:
                        book = books[index]
                        gaps = np.c_[np.diff(book[:, 0::4], axis=1), -np.diff(book[:, 2::4], axis=1), book[:, 0] - book[:, 2]]
                        result["cases"].append(dict(model=label, checkpoint=stage, validation_window=int(index + 1), raw_anchor_row=int(starts[index] + validation.raw_offset),
                            **validity_rates(book[None]), depth_ssim=float(details["depth_ssim_per_sequence"][index]), minimum_gap_usd=float(gaps.min()),
                            generated_volume_changes=int(np.count_nonzero(np.diff(book[:, VOLUME_COLUMNS], axis=0))),
                            real_volume_changes=int(np.count_nonzero(np.diff(real[index][:, VOLUME_COLUMNS], axis=0)))))
            for index in cases:
                case_noise = [r for r in noise if r["validation_window"] == int(index+1)]
                result["noise_rows"].append({"Model": label, "Checkpoint": stage, "Validation window": int(index + 1),
                    "Price SD / real": float(np.mean([r["price_ratio"] for r in case_noise])),
                    "Volume SD / real": float(np.mean([r["volume_ratio"] for r in case_noise])),
                    "Minimum unique sequences": min(r["unique_sequences"] for r in case_noise)})
            diagnostic = diagnose(data_dir, path, destination / f"{label}_{stage}_reconstruction.json", prepared=prepared)
            result["models"][label]["checkpoints"][stage] = dict(score=score, checkpoint_sha256=sha256_file(path),
                replayed_generation_seeds=[r["seed"] for r in score["runs"]], exact_saved_examples=stage == "selected",
                finite=True, reconstruction=diagnostic, fixed_anchor_noise_runs=noise, shifted_context_runs=shifted_context_runs, price_correlation=covariance)
    result["noise_scope"] = "128 sequences per generation seed at each of three fixed anchors/contexts. SD across noise at fixed updates divided by global real-validation feature SD, separately for prices and volumes. Noise-seed SD is not retraining uncertainty. Shifted-context checks keep decoder anchors fixed and change only network context."
    result["shifted_context_scope"] = "Volume-context fits shift only starting volumes while preserving price context and decoder anchors; legacy fits shift price-gap context."
    result["comparison_signature"] = signature
    write_json(destination / "audit.json", result)
    return result


# Preserve the former report entry point for archived local study scripts.
audit_unconditional_runs = audit_generation_runs


@torch.inference_mode()
def diagnose(data_dir, checkpoint_path, destination, train_count=512, prepared=None):
    torch.set_num_threads(1)
    model, scaler, checkpoint = load_timegan(checkpoint_path)
    datasets, fitted_scaler, audit = prepare_datasets(data_dir, 32, 32) if prepared is None else prepared
    np.testing.assert_array_equal(scaler.minimum, fitted_scaler.minimum)
    np.testing.assert_array_equal(scaler.maximum, fitted_scaler.maximum)
    result = dict(scope="Observed-input reconstruction and anchor controls, not noise-only generation scores.",
                  checkpoint_sha256=sha256_file(Path(checkpoint_path)), architecture=checkpoint["architecture"],
                  test_evaluated=False, source_files=timegan_code_hashes(), splits={})
    for split in ("train", "validation"):
        dataset = datasets[split]
        starts, anchors, real = validation_windows(dataset, 4)
        if split == "train":
            take = np.linspace(0, len(starts) - 1, min(train_count, len(starts)), dtype=int)
            starts, anchors, real = starts[take], anchors[take], real[take]
        features = torch.from_numpy(np.stack([dataset.features[s:s + 32] for s in starts]))
        volume_mode = checkpoint["config"].get("volume_conditioning", "none")
        context = anchor_context(anchors, checkpoint.get("anchor_context_scaler"), checkpoint["config"].get("anchor_conditioning", "none"), volumes=None if volume_mode == "none" else dataset.book[starts][:, VOLUME_COLUMNS], volume_mode=volume_mode)
        context = None if context is None else torch.from_numpy(context)
        recovered = torch.cat([model.reconstruct(features[i:i+128], None if context is None else context[i:i+128], torch.from_numpy(anchors[i:i+128])) for i in range(0, len(features), 128)])
        decoded = decode_batch(recovered, scaler, anchors)
        roundtrip = decode_batch(features, scaler, anchors)
        mse = (features - recovered).square().mean((0, 1)).numpy()
        variance = features.var((0, 1), unbiased=False).numpy()
        groups = {name: dict(scaled_mse=float(mse[columns].mean()), scaled_variance=float(variance[columns].mean()),
                            mse_over_variance=float(mse[columns].mean() / variance[columns].mean()))
                  for name, columns in (("price", PRICE_COLUMNS), ("volume", VOLUME_COLUMNS))}
        controls = []
        for seed in (42, 43, 44):
            moved, shift = shifted_anchors(anchors, seed)
            controls.append(dict(seed=seed, cyclic_shift=shift, **validity_rates(decode_batch(features, scaler, moved))))
        result["splits"][split] = dict(windows=len(starts), raw_start_indices=(starts + dataset.raw_offset).tolist(),
            real=validity_rates(real), roundtrip=validity_rates(roundtrip), reconstruction=validity_rates(decoded),
            feature_groups=groups, feature_mse=mse.tolist(), feature_variance=variance.tolist(),
            price_rmse_usd=float(np.sqrt(np.square(decoded[..., PRICE_COLUMNS] - real[..., PRICE_COLUMNS]).mean())),
            roundtrip_max_price_error_usd=float(np.abs(roundtrip[..., PRICE_COLUMNS] - real[..., PRICE_COLUMNS]).max()),
            reconstruction_invalid_by_update=[validity_rates(decoded[:, i:i + 1])["invalid_rate"] for i in range(32)],
            shuffled_anchor_runs=controls,
            shuffled_anchor_mean={key: float(np.mean([c[key] for c in controls])) for key in validity_rates(real)})
    write_json(destination, result)
    return result


def check_conditioning_pair(runs):
    """Reject comparisons that change more than the supplied context values."""
    settings = []
    for mode in ("zero", "matched"):
        run = runs[mode]
        if run["config"].get("anchor_conditioning") != mode or run.get("test_evaluated") is not False:
            raise ValueError("Use completed matched/zero validation runs, with no test scoring.")
        settings.append({k: v for k, v in run["config"].items() if k not in ("anchor_conditioning", "objective")})
    if settings[0] != settings[1]:
        raise ValueError("Paired runs must share architecture, seed, losses and stage budgets.")
    for key in ("comparison_signature", "metric_specification", "source_files", "parameter_count"):
        if runs["zero"][key] != runs["matched"][key]:
            raise ValueError(f"Paired runs differ in {key}.")


def check_volume_conditioning_pair(runs):
    """Isolate access to starting volumes, retaining identical price context."""
    settings = []
    for mode in ("zero", "matched"):
        run = runs[mode]
        config = run["config"]
        if (config.get("volume_conditioning") != mode or config.get("anchor_conditioning") != "matched"
                or config["architecture"].get("context_dim") != 39 or run.get("test_evaluated") is not False):
            raise ValueError("Use matched-price, 39-context, zero/matched-volume validation fits.")
        settings.append({k: v for k, v in config.items() if k not in ("volume_conditioning", "objective")})
    if settings[0] != settings[1]:
        raise ValueError("Volume pair changes settings other than volume conditioning.")
    for key in ("comparison_signature", "metric_specification", "source_files", "parameter_count",
                "anchor_context_scaler", "initial_parameters_sha256", "minibatch_stream_sha256", "noise_stream_sha256"):
        if not runs["zero"].get(key) or runs["zero"][key] != runs["matched"].get(key):
            raise ValueError(f"Volume pair differs in {key}.")


def volume_conditioning_screen(matched, zero, noise_rows):
    """New SSIM-focused screen for the volume pilot; previous study gates stay intact."""
    criteria = [("Depth SSIM gain at least 0.05", matched["depth_ssim_mean"] - zero["depth_ssim_mean"] >= .05,
                 matched["depth_ssim_mean"] - zero["depth_ssim_mean"]),
                ("No increased invalidity", matched["invalid_rate_mean"] <= zero["invalid_rate_mean"],
                 matched["invalid_rate_mean"] - zero["invalid_rate_mean"])]
    for metric in ("kl_spread", "kl_mid_return", "abs_return_acf_mae"):
        criteria.append((metric + " increase at most 5%", matched[metric + "_mean"] <= 1.05 * zero[metric + "_mean"],
                         matched[metric + "_mean"] - zero[metric + "_mean"]))
    if len(noise_rows) != 3:
        raise ValueError("Use all three predetermined noise cases.")
    ratios = [r[k] for r in noise_rows for k in ("Price SD / real", "Volume SD / real")]
    if not np.isfinite(ratios).all():
        raise ValueError("Noise ratios must be finite.")
    criteria.append(("Fixed-context noise SD ratios at least 0.1 in every case", min(ratios) >= .1, min(ratios)))
    if not all(np.isfinite(value) for _, _, value in criteria):
        raise ValueError("Screen metrics must be finite.")
    return [dict(Criterion=label, Passed=bool(passed), Value=float(value)) for label, passed, value in criteria]


def conditioning_screen(matched, zero, noise_rows):
    """The predeclared pilot screen; it does not certify simulator quality."""
    gain = 100 * (zero["invalid_rate_mean"] - matched["invalid_rate_mean"])
    criteria = [("Invalidity gain at least 5 percentage points", gain >= 5, gain)]
    for metric in ("kl_spread", "kl_mid_return", "abs_return_acf_mae"):
        difference = matched[metric + "_mean"] - zero[metric + "_mean"]
        criteria.append((metric + " increase at most 5%", matched[metric + "_mean"] <= 1.05 * zero[metric + "_mean"], difference))
    difference = matched["depth_ssim_mean"] - zero["depth_ssim_mean"]
    criteria.append(("Depth SSIM drop at most 0.01", difference >= -.01, difference))
    if len(noise_rows) != 3:
        raise ValueError("The noise screen needs all three predetermined cases.")
    ratios = [r[key] for r in noise_rows for key in ("Price SD / real", "Volume SD / real")]
    if not np.isfinite(ratios).all():
        raise ValueError("Noise ratios must be finite.")
    criteria.append(("Fixed-context noise SD ratios at least 0.1 in every case", min(ratios) >= .1, min(ratios)))
    return [dict(Criterion=label, Passed=bool(passed), Value=float(value)) for label, passed, value in criteria]


def analyse_conditioning(data_dir, matched_run, zero_run, destination, baseline_path=None):
    """Replay saved fits and regenerate validation diagnostics in a new folder.

    Read each CSV pair once and reuse the prepared data for reconstruction.
    Saved training timings remain historical; this function never calls fit().
    """
    baseline_path = baseline_path or PROJECT_DIR / "reports/ar/anchor_conditioned.json"
    torch.set_num_threads(1)
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Report already exists; choose a new report directory to preserve it.")
    folders = {"zero": Path(zero_run), "matched": Path(matched_run)}
    runs = {mode: json.loads((folder / "results.json").read_text(encoding="utf-8")) for mode, folder in folders.items()}
    check_conditioning_pair(runs)
    if runs["zero"]["source_files"] != timegan_code_hashes():
        raise ValueError("Model sources differ from these runs; use their archived source or new fits.")
    previous_path = PROJECT_DIR / "reports/timegan/conditioning.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else {}
    plan = dict(protocol="anchor_conditioned_v1", training_seeds=[runs["zero"]["config"]["training_seed"]],
                generation_seeds=runs["zero"]["config"]["generation_seeds"], settings=runs["zero"]["config"],
                decision="Screen invalidity, both KL scores, ACF, SSIM and fixed-context noise together; one fitted seed is exploratory.")
    prepared = prepare_datasets(data_dir, 32, 32)
    datasets, scaler, audit = prepared
    training, validation = datasets["train"], datasets["validation"]
    starts, anchors, real = validation_windows(validation, 4)
    actual_features = np.stack([validation.features[s:s + 32] for s in starts])
    real_sd = actual_features.reshape(-1, 40).std(0)
    case_indices = np.array([0, (len(starts) - 1) // 2, len(starts) - 1])
    ar = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    result = dict(scope=f"Pilot: one training seed ({plan['training_seeds'][0]}) per mode; generation-seed variation is not retraining uncertainty.",
        plan=plan, comparison=[], final_comparison=[], reconstruction_rows=[], noise_rows=[], shifted_context_rows=[], cases=[], resources=[], figures={},
        verification=dict(checkpoint_replays={}, final_checkpoints_replayed=True, finite_outputs=True, test_evaluated=False),
        source_files=timegan_code_hashes(), analysis_sha256=sha256_file(Path(__file__)), models={})
    all_books = {}
    for mode in ('zero','matched'):
        folder = folders[mode]
        name, run = folder.name, runs[mode]
        assert run['comparison_signature']==ar['comparison_signature']
        assert run['source_files'] == timegan_code_hashes()
        assert run['reload_max_abs_difference'] == 0
        assert sha256_file(folder/'selected.pt')==run['selected_checkpoint_sha256']
        model,loaded_scaler,checkpoint=load_timegan(folder/'selected.pt')
        np.testing.assert_array_equal(loaded_scaler.minimum,scaler.minimum)
        np.testing.assert_array_equal(loaded_scaler.maximum,scaler.maximum)
        assert checkpoint['anchor_context_scaler']==fit_anchor_context(training.book[training.starts][:,PRICE_COLUMNS]/10000)
        assert checkpoint['anchor_context_scaler']==ar['context_scaler']
        assert checkpoint['source_files']==run['source_files'] and checkpoint['config']==run['config']
        result['verification']['context_statistics_match_training_and_ar']=True
        context=anchor_context(anchors,checkpoint['anchor_context_scaler'],mode)
        features=model.generate(len(starts),32,plan['generation_seeds'][0],128,context=context)
        assert torch.isfinite(features).all()
        books=decode_batch(features,scaler,anchors);all_books[mode]=books
        sample=np.load(folder/'validation_examples.npz')
        assert np.array_equal(books[:3],sample['generated'])
        reference=ValidationEvaluator(real,anchors,run['metric_specification'])
        scores,details=reference.evaluate(books)
        for key,value in scores.items():
            assert np.isclose(value,run['selected']['runs'][0][key],rtol=1e-12,atol=1e-12),(name,key)
        label='TimeGAN: '+mode+' context'
        result['comparison'].append(dict(label=label,score=run['selected']))
        result['final_comparison'].append(dict(label=label,score=run['validation_history'][-1]))
        result['verification']['checkpoint_replays'][name]=0.
        diagnostic=diagnose(data_dir,folder/'selected.pt',destination/f'{mode}_reconstruction.json',prepared=prepared)
        for split,diag in diagnostic['splits'].items():
            for group,row in diag['feature_groups'].items():
                result['reconstruction_rows'].append({'Model':label,'Split':split,'Feature group':group,'MSE / variance':row['mse_over_variance'],
                    'Reconstructed invalid (%)':100*diag['reconstruction']['invalid_rate']})
        noise_details=[]
        for index in case_indices:
            constant_context=np.repeat(context[index:index+1],128,axis=0)
            checks=[]
            for seed in plan['generation_seeds']:
                generated=model.generate(128,32,seed,128,context=constant_context).numpy()
                noise_sd=generated.std(axis=0).mean(axis=0)
                checks.append(dict(seed=seed,price_sd=float(noise_sd[::2].mean()),volume_sd=float(noise_sd[1::2].mean()),
                    price_ratio=float(noise_sd[::2].mean()/real_sd[::2].mean()),volume_ratio=float(noise_sd[1::2].mean()/real_sd[1::2].mean()),
                    unique_sequences=int(len(np.unique(generated.reshape(128,-1),axis=0)))))
            noise_details.append(dict(validation_window=int(index+1),runs=checks))
            result['noise_rows'].append({'Model':label,'Validation window':int(index+1),'Price noise SD':np.mean([r['price_sd'] for r in checks]),
                'Volume noise SD':np.mean([r['volume_sd'] for r in checks]),'Price SD / real':np.mean([r['price_ratio'] for r in checks]),
                'Volume SD / real':np.mean([r['volume_ratio'] for r in checks])})
        shifted=[]
        for seed in plan['generation_seeds']:
            moved,shift=shifted_anchors(anchors,seed)
            moved_context=anchor_context(moved,checkpoint['anchor_context_scaler'],mode)
            sampled=model.generate(len(starts),32,seed,128,context=moved_context)
            moved_books=decode_batch(sampled,scaler,anchors)
            moved_scores,_=reference.evaluate(moved_books)
            original=model.generate(len(starts),32,seed,128,context=context)
            original_scores,_=reference.evaluate(decode_batch(original,scaler,anchors))
            recorded=next(r for r in run['selected']['runs'] if r['seed']==seed)
            for key,value in original_scores.items():
                assert np.isclose(value,recorded[key],rtol=1e-12,atol=1e-12),(name,seed,key)
            delta=float((sampled-original).abs().max())
            if mode=='zero': assert delta==0
            shifted.append(dict(seed=seed,shift=shift,max_feature_difference=delta,**moved_scores))
        result['shifted_context_rows'].append({'Model':label,'Correct context invalid (%)':100*run['selected']['invalid_rate_mean'],
            'Shifted context invalid (%)':100*np.mean([r['invalid_rate'] for r in shifted]),
            'Max feature difference':max(r['max_feature_difference'] for r in shifted)})
        for index in case_indices:
            row=validity_rates(books[index:index+1]);book=books[index]
            gaps=np.c_[np.diff(book[:,0::4],axis=1),-np.diff(book[:,2::4],axis=1),book[:,0]-book[:,2]]
            result['cases'].append({'Validation window':int(index+1),'Model':mode,'Raw anchor row':int(starts[index]+validation.raw_offset),
                'Invalid (%)':100*row['invalid_rate'],'Crossed (%)':100*row['crossed_rate'],
                'Depth SSIM':float(details['depth_ssim_per_sequence'][index]),'Minimum gap (USD)':float(gaps.min()),
                'Generated volume changes':int(np.count_nonzero(np.diff(book[:,VOLUME_COLUMNS],axis=0))),
                'Real volume changes':int(np.count_nonzero(np.diff(real[index][:,VOLUME_COLUMNS],axis=0)))})
        result['resources'].append({'Model':label,'Parameters':run['parameter_count'],'Fit minutes':run['total_seconds']/60,
            'Generation ms / 1080 windows':1000*run['selected']['generation_seconds_mean'],'Device':run['device'],'Selected joint step':run['selected']['step']})
        final_model,final_scaler,final_checkpoint=load_timegan(folder/'joint_last.pt')
        assert final_checkpoint['step']==run['config']['joint_steps'] and final_checkpoint['source_files']==run['source_files']
        assert final_checkpoint['config']==run['config'] and final_checkpoint['anchor_context_scaler']==checkpoint['anchor_context_scaler']
        np.testing.assert_array_equal(final_scaler.minimum,scaler.minimum)
        np.testing.assert_array_equal(final_scaler.maximum,scaler.maximum)
        for recorded in run['validation_history'][-1]['runs']:
            final_features=final_model.generate(len(starts),32,recorded['seed'],128,context=context)
            assert torch.isfinite(final_features).all()
            final_scores,_=reference.evaluate(decode_batch(final_features,scaler,anchors))
            for key,value in final_scores.items():
                assert np.isclose(value,recorded[key],rtol=1e-12,atol=1e-12),(name,'final',key)
        del final_model
        result['models'][mode]=dict(name=name,checkpoint_sha256=run['selected_checkpoint_sha256'],diagnostics=diagnostic,
            training_history=run['history'],validation_history=run['validation_history'],
            fixed_context_noise=noise_details,shifted_context=shifted,selected=run['selected'],final=run['validation_history'][-1])
    for row in ar['candidates']:
        result['comparison'].append(dict(label='VARX(2): '+row['anchor_conditioning']+' context',score=row))
        result['resources'].append({'Model':'VARX(2): '+row['anchor_conditioning']+' context','Parameters':row['coefficient_count'],
            'Fit minutes':row['fit_seconds']/60,'Generation ms / 1080 windows':1000*row['generation_seconds_mean'],'Device':'cpu','Selected joint step':None})

    matched,zero=(result['models'][mode]['selected'] for mode in ('matched','zero'))
    gain=100*(zero['invalid_rate_mean']-matched['invalid_rate_mean'])
    noise_rows=[r for r in result['noise_rows'] if r['Model']=='TimeGAN: matched context']
    result['pilot_screen']=conditioning_screen(matched,zero,noise_rows)
    passed=all(row['Passed'] for row in result['pilot_screen'])
    result['pilot_screen_passed']=passed
    result['conclusion']=(f'Matched context gives {100*matched["invalid_rate_mean"]:.2f}% invalid snapshots versus {100*zero["invalid_rate_mean"]:.2f}% for the same-size zero-context control (gain {gain:.2f} percentage points). '
        f'Return KL is {matched["kl_mid_return_mean"]:.3f} versus {zero["kl_mid_return_mean"]:.3f}; ACF error {matched["abs_return_acf_mae_mean"]:.3f} versus {zero["abs_return_acf_mae_mean"]:.3f}; depth SSIM {matched["depth_ssim_mean"]:.3f} versus {zero["depth_ssim_mean"]:.3f}. '
        'This is a single training-seed pilot, not confirmation across retraining seeds.')
    result['next_step']='Pilot screen met: repeat both modes at matched training seeds before making a reliability claim.' if passed else 'Pilot screen not met: retain the negative/trade-off evidence; do not expand this configuration into a blind grid or call it a validated simulator.'

    result['performance_check'] = dict(scope="No training or timing benchmark is run during report generation.", rows=[])
    if all(previous.get('models', {}).get(mode, {}).get('checkpoint_sha256') == runs[mode]['selected_checkpoint_sha256'] for mode in folders):
        result['performance_check'] = previous['performance_check']
        result['performance_check_origin'] = "Historical timing measurement retained for the identical checkpoint pair; not remeasured."
    save_conditioning_figures(result, real, all_books, case_indices, ar['metric_specification']['depth_ssim'], destination)
    write_json(destination / 'conditioning.json', result)
    return result


def save_conditioning_figures(result, real, all_books, case_indices, depth_spec, folder):
    """Use fixed cases and common scales; never repair or select generated examples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    def save(fig,key):
        path=folder/f'conditioning_{key}.png';fig.savefig(path,dpi=145);plt.close(fig)
        result['figures'][key]=dict(file=path.name,sha256=sha256_file(path))

    metrics=[('invalid_rate','Invalid (%)',100),('kl_spread','Spread KL',1),('kl_mid_return','Return KL',1),('abs_return_acf_mae','ACF error',1),('depth_ssim','Depth SSIM',1)]
    fig,axes=plt.subplots(1,5,figsize=(15,4),layout='constrained')
    for ax,(key,label,scale) in zip(axes,metrics):
        rows=result['comparison']; ax.bar(range(len(rows)),[r['score'][key+'_mean']*scale for r in rows],yerr=[r['score'][key+'_sd']*scale for r in rows],capsize=3,color=['#b1bac5','#2c7fb8','#d2d5b1','#6e8b3d'])
        ax.set(title=label,xticks=range(len(rows)),xticklabels=['TG zero','TG matched','AR zero','AR matched']);ax.tick_params(axis='x',rotation=65);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Starting-book conditioning pilot; error bars = generation-seed SD, one fitted model per bar')
    save(fig,'metrics')
    fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
    for mode,model in result['models'].items():
        for ax,stage,key,title in zip(axes.ravel()[:4],['embedding','supervised','joint','joint'],
                ['reconstruction_mse','supervised_mse','generator_loss','discriminator_loss'],
                ['Reconstruction warm-up MSE','Supervisor warm-up MSE','Joint generator objective','Joint discriminator objective']):
            rows=[r for r in model['training_history'] if r['stage']==stage]
            ax.plot([r['step'] for r in rows],[r[key] for r in rows],label=mode)
            ax.set(title=title,xlabel='Stage update');ax.grid(alpha=.2)
        rows=model['validation_history']
        for ax,key,title,scale in [(axes[1,1],'invalid_rate_mean','Validation invalid (%)',100),(axes[1,2],'kl_mid_return_mean','Validation return KL',1)]:
            ax.plot([r['step'] for r in rows],[r[key]*scale for r in rows],marker='o',label=mode)
            ax.set(title=title,xlabel='Joint update');ax.grid(alpha=.2)
    axes[0,0].legend();save(fig,'training')
    images={k:depth_images(v[case_indices],depth_spec) for k,v in dict(real=real,**all_books).items()}
    fig,axes=plt.subplots(3,3,figsize=(12,8),layout='constrained')
    for i,index in enumerate(case_indices):
        for j,mode in enumerate(('real','zero','matched')):
            picture=np.concatenate([images[mode][i,0,::-1],images[mode][i,1]],axis=0)
            heat=axes[i,j].imshow(picture,aspect='auto',vmin=0,vmax=1,cmap='viridis',interpolation='nearest')
            axes[i,j].set(title=f'Window {index+1}: {mode}',xlabel='Update',yticks=[0,9.5,19],yticklabels=['Ask 10','Best levels','Bid 10'])
    fig.colorbar(heat,ax=axes.ravel().tolist(),shrink=.8,label='log(1 + volume) / training maximum')
    save(fig,'depth')
    fig,axes=plt.subplots(3,2,figsize=(11,7),layout='constrained')
    for i,index in enumerate(case_indices):
        for mode,books in dict(real=real,**all_books).items():
            book=books[index]
            gap=np.c_[np.diff(book[:,0::4],axis=1),-np.diff(book[:,2::4],axis=1),book[:,0]-book[:,2]]
            axes[i,0].plot(np.arange(1,33),gap.min(1),label=mode)
            axes[i,1].plot(np.arange(1,33),book[:,0]-book[:,2],label=mode)
        for ax in axes[i]: ax.axhline(0,color='black',linewidth=.7);ax.set_xlabel('Update');ax.grid(alpha=.2)
        axes[i,0].set(title=f'Window {index+1}: minimum of 19 gaps',ylabel='USD')
        axes[i,1].set(title=f'Window {index+1}: best spread',ylabel='USD')
    axes[0,0].legend();save(fig,'gaps')
