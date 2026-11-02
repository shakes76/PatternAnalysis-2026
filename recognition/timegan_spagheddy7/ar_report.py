"""Presentation and integrity checks for the single saved AR report bundle."""

import hashlib
import json
from pathlib import Path

import numpy as np


def source_fingerprint(project=None):
    project = Path(project or Path(__file__).resolve().parent)
    names = ("dataset.py", "modules.py", "evaluation.py", "protocol.py", "train.py", "tune_ar.py", "ar_report.py")
    hashes = {name: hashlib.sha256((project / name).read_text(encoding="utf-8").encode()).hexdigest() for name in names}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def load_report(project, historical_sha256=None):
    """Read current or explicitly pinned historical results; normalise text line endings."""
    folder = Path(project) / "reports/ar"
    content = (folder / "results.json").read_text(encoding="utf-8").encode()
    report = json.loads(content)
    historical_match = historical_sha256 is not None and hashlib.sha256(content).hexdigest() == historical_sha256
    if report.get("format_version") != 3 or (report["source_fingerprint"] != source_fingerprint(project) and not historical_match):
        raise ValueError("Saved report is from different code. Recompute the grid explicitly before reporting current results.")
    for figure in report.get("figures", {}).values():
        content = (folder / figure["file"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != figure["sha256"]:
            raise ValueError(f"Figure does not match this report: {figure['file']}")
    return report


def example_diagnostics(report, samples):
    from evaluation import DepthSSIM, validity_rates
    rows = []
    depth = DepthSSIM(samples["real"], report["metric_specification"]["depth_ssim"])
    scores = {key: depth.score(samples[key])[0] for key in ("real", report["original_candidate"], report["selected_candidate"])}
    for window, raw_index in enumerate(samples["raw_start_indices"]):
        for label, key in (("Real", "real"), ("Original", report["original_candidate"]),
                           ("Selected", report["selected_candidate"])):
            book = samples[key][window:window + 1]
            rows.append({"window": window + 1, "anchor_raw_row": int(raw_index), "model": label,
                         **validity_rates(book), "minimum_spread_usd": float((book[..., 0] - book[..., 2]).min()),
                         "depth_ssim": float(scores[key][window])})
    return rows


def save_figures(report, samples, folder):
    """Render report figures once during recomputation, using fixed common examples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {}

    def save(fig, name):
        path = folder / f"{name}.png"
        fig.savefig(path, dpi=145)
        plt.close(fig)
        manifest[name] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    config, candidates = report["config"], report["candidates"]
    by_params = {(r["lags"], r["ridge"]): r for r in candidates}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), layout="constrained")
    metrics = [("invalid_rate_mean", "Invalid snapshots (%)", 100), ("mean_kl_mean", "Mean KL", 1),
               ("abs_return_acf_mae_mean", "Absolute-return ACF error", 1), ("depth_ssim_mean", "Depth SSIM (higher is better)", 1)]
    for ax, (metric, title, scale) in zip(axes, metrics):
        values = np.array([[by_params[p, r].get(metric, np.nan) for r in config["ridges"]] for p in config["lags"]]) * scale
        heat = ax.imshow(values, cmap="YlGn" if metric == "depth_ssim_mean" else "YlOrRd", aspect="auto")
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
        ax.set(title=title, xlabel="Ridge", ylabel="Past updates")
        ax.set_xticks(range(len(config["ridges"])), [f"{r:.0e}" for r in config["ridges"]])
        ax.set_yticks(range(len(config["lags"])), config["lags"])
        fig.colorbar(heat, ax=ax, shrink=0.8)
    save(fig, "grid")

    series = [("Real validation", "real"), ("Original VAR(1)", report["original_candidate"]),
              ("Selected VAR", report["selected_candidate"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout="constrained")
    for ax, quantity, title in zip(axes, ("spread", "mid_return"), ("Spread distribution", "Mid-price return distribution")):
        for label, key in series:
            histogram = report["real_histograms"][quantity] if key == "real" else report["generated_histograms"][key][quantity]
            ax.step(np.arange(len(histogram)), histogram, where="mid", label=label)
        boundaries = np.asarray(report["metric_specification"]["boundaries"][quantity])
        regular = len(boundaries) - 1
        middle = np.linspace(regular // 4, 3 * regular // 4, 3).astype(int)
        labels = [f"<{boundaries[0]:.2g}", *[f"{(boundaries[i - 1] + boundaries[i]) / 2:.2g}" for i in middle],
                  f"≥{boundaries[-1]:.2g}"]
        ax.set_xticks([0, *middle, regular + 1], labels)
        xlabel = "Spread (USD; end bins contain tails)" if quantity == "spread" else "Log return (end bins contain tails)"
        ax.set(title=title, xlabel=xlabel, ylabel="Probability", yscale="log")
        ax.grid(alpha=0.2)
    axes[0].legend()
    save(fig, "distributions")

    fig, ax = plt.subplots(figsize=(8, 3.2), layout="constrained")
    lags = np.arange(1, report["metric_specification"]["max_lag"] + 1)
    for label, key in series:
        acf = report["real_abs_return_acf"] if key == "real" else report["generated_abs_return_acf"][key]
        ax.plot(lags, acf, marker=".", label=label)
    ax.set(xlabel="Lag (updates)", ylabel="Absolute-return ACF", title="Validation temporal dependence; seed means")
    ax.grid(alpha=0.2); ax.legend()
    save(fig, "acf")

    count, length = len(samples["real"]), config["sequence_length"]
    fig, axes = plt.subplots(count, 2, figsize=(11, 2.5 * count), layout="constrained", squeeze=False)
    for i, row in enumerate(axes):
        for label, key in series:
            book = samples[key][i]
            row[0].plot(np.arange(1, length + 1), book[:, 0] - book[:, 2], label=label)
            row[1].plot(np.arange(1, length + 1), book[:, 1], label=label)
        row[0].set(ylabel="Spread (USD)", title=f"Window {i + 1}: anchor row {samples['raw_start_indices'][i]}")
        row[1].set(ylabel="Best ask shares", title=f"Window {i + 1}: outstanding volume")
        for ax in row:
            ax.set_xlabel("Future update"); ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    save(fig, "examples")

    # Signed log preserves negative-volume failures; one colour scale for every panel.
    depth = {}
    for label, key in series:
        books = samples[key]
        volumes = np.concatenate((books[..., 1::4][..., ::-1], books[..., 3::4]), axis=-1)
        depth[key] = np.sign(volumes) * np.log1p(np.abs(volumes))
    bound = max(float(np.abs(values).max()) for values in depth.values()) or 1
    fig, axes = plt.subplots(count, 3, figsize=(11, 2.2 * count), layout="constrained", squeeze=False)
    for i in range(count):
        for j, (label, key) in enumerate(series):
            ax = axes[i, j]
            heat = ax.imshow(depth[key][i].T, aspect="auto", origin="upper", cmap="coolwarm", vmin=-bound, vmax=bound,
                             extent=(0.5, length + 0.5, 19.5, -0.5), interpolation="nearest")
            ax.set(title=f"Window {i + 1}: {label}", xlabel="Future update")
            ax.set_yticks([0, 9.5, 19], ["Ask 10", "Best levels", "Bid 10"])
            ax.axhline(9.5, color="black", linewidth=0.5)
    fig.colorbar(heat, ax=axes.ravel().tolist(), shrink=0.8, label="sign(volume) × log(1 + |shares|)")
    save(fig, "depth")

    initial = folder / "initial_preview.png"
    if initial.exists():
        manifest["initial_preview"] = {"file": initial.name, "sha256": hashlib.sha256(initial.read_bytes()).hexdigest()}
    return manifest
