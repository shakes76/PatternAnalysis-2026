"""Reload AR or TimeGAN and generate examples without CSV access."""

import argparse
import json
import hashlib
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import FEATURE_NAMES, FeatureScaler, PROJECT_DIR, sha256_file
from evaluation import decode_batch
from modules import VectorAR, TimeGAN
from protocol import anchor_context


def save_preview(book, initial_prices, destination, lags=1, model_name=None):
    """Show the first generated sequence without filtering or repairing it."""
    prices = np.vstack((initial_prices[[0, 1]], book[:, [0, 2]]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
    axes[0].plot(np.arange(len(prices)), prices[:, 0], label="Best ask")
    axes[0].plot(np.arange(len(prices)), prices[:, 1], label="Best bid")
    axes[0].set(xlabel="Generated update (0 = supplied anchor)", ylabel="Price (USD)", title="Generated best prices")
    axes[1].plot(np.arange(1, len(book) + 1), book[:, 1], label="Best ask volume")
    axes[1].plot(np.arange(1, len(book) + 1), book[:, 3], label="Best bid volume")
    axes[1].set(xlabel="Generated update", ylabel="Shares", title="Generated level-1 volumes")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    fig.suptitle(f"{model_name or f'VAR({lags})'}: first generated example; training anchor, no structural repair")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def generate_examples(checkpoint_path, output_path, report_path, figure_path, count=64, seed=42, threads=4):
    if count < 1 or seed < 0 or seed >= 2**63 - 1 or threads < 1:
        raise ValueError("Use a positive count/threads and a seed between 0 and 2**63 - 2.")
    torch.set_num_threads(threads)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    supported = {(1, "vector_ar1"), (2, "vector_ar")}
    if (checkpoint.get("format_version"), checkpoint.get("model_type")) not in supported:
        raise ValueError("Unsupported checkpoint format or model type.")
    if checkpoint["feature_names"] != FEATURE_NAMES:
        raise ValueError("Checkpoint features do not match the current data pipeline.")
    model = VectorAR(**checkpoint["state_dict"]).eval()
    scaler = FeatureScaler(checkpoint["scaler"]["minimum"].numpy(), checkpoint["scaler"]["maximum"].numpy())
    contexts = checkpoint["contexts"]
    rng = torch.Generator().manual_seed(seed)
    selected = torch.randint(len(contexts["features"]), (count,), generator=rng)
    initial_features, initial_prices = contexts["features"][selected], contexts["initial_prices"][selected]
    sequence_length = checkpoint["fit_report"]["sequence_length"]

    started = perf_counter()
    # Separate deterministic streams for choosing contexts and innovations.
    context = anchor_context(initial_prices.numpy(), checkpoint.get("anchor_context_scaler"), checkpoint["fit_report"].get("anchor_conditioning", "none"))
    features = model.generate(initial_features, sequence_length=sequence_length, seed=seed + 1, context=context)
    generation_seconds = perf_counter() - started
    books = decode_batch(features, scaler, initial_prices.numpy())

    output_path, report_path = Path(output_path), Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, scaled_features=features.numpy(), orderbooks=books,
                        initial_features=initial_features.numpy(), initial_prices=initial_prices.numpy(),
                        context_raw_start_indices=contexts["raw_start_indices"][selected].numpy())
    save_preview(books[0], initial_prices[0].numpy(), figure_path, model.lags)
    report = {
        "model": f"vector_ar{model.lags}", "checkpoint_sha256": sha256_file(Path(checkpoint_path)),
        "device": "cpu", "torch_version": str(torch.__version__), "threads": threads,
        "context_split": "train", "context_seed": seed, "innovation_seed": seed + 1,
        "sequences": count, "sequence_length": sequence_length, "features": len(FEATURE_NAMES),
        "generated_shape": list(features.shape), "all_features_finite": True, "all_decoded_values_finite": True,
        "generation_seconds": generation_seconds, "sequences_per_second": count / generation_seconds,
        "timing_scope": "Feature rollout only; excludes model loading, context selection, decoding, file IO and plotting.",
        "first_context_raw_start_index": int(contexts["raw_start_indices"][selected[0]]),
        "initial_state_policy": "Previous p features and matching price anchor sampled from training; all returned updates generated.",
        "structural_repair": "none",
        "status": "Fit/save/reload/generate smoke test only; no held-out quality evaluation.",
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def load_timegan(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint.get("format_version"), checkpoint.get("model_type")) != (1, "timegan"):
        raise ValueError("Unsupported TimeGAN checkpoint.")
    if checkpoint["feature_names"] != FEATURE_NAMES:
        raise ValueError("Checkpoint features differ from the current pipeline.")
    model = TimeGAN(**checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    scaler = FeatureScaler(checkpoint["scaler"]["minimum"].numpy(), checkpoint["scaler"]["maximum"].numpy())
    return model.eval(), scaler, checkpoint


def save_timegan_report_figures(examples, destination):
    """Cache fixed example plots, keeping the underlying book arrays out of Git."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    real, generated = np.asarray(examples["real"]), np.asarray(examples["generated"])
    figures = {}
    def save(fig, name):
        path = destination / f"{name}.png"
        fig.savefig(path, dpi=145)
        plt.close(fig)
        figures[name] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    fig, axes = plt.subplots(3, 2, figsize=(11, 7), layout="constrained")
    for i, row in enumerate(axes):
        for label, books in (("Real", real), ("TimeGAN", generated)):
            row[0].plot(np.arange(1, 33), books[i, :, 0] - books[i, :, 2], label=label)
            row[1].plot(np.arange(1, 33), books[i, :, 1], label=label)
        row[0].set(title=f"Window {i+1}: spread", ylabel="USD")
        row[0].axhline(0, color="grey", linewidth=.7)
        row[1].set(title=f"Window {i+1}: best ask volume", ylabel="Shares")
        for ax in row:
            ax.set_xlabel("Future update"); ax.grid(alpha=.2)
    axes[0, 0].legend()
    save(fig, "examples")
    depth = [np.concatenate((books[..., 1::4][..., ::-1], books[..., 3::4]), axis=-1) for books in (real, generated)]
    depth = [np.sign(v) * np.log1p(np.abs(v)) for v in depth]
    bound = max(float(np.abs(v).max()) for v in depth) or 1
    fig, axes = plt.subplots(3, 2, figsize=(10, 6), layout="constrained")
    for i, row in enumerate(axes):
        for j, (label, volumes) in enumerate(zip(("Real", "TimeGAN"), depth)):
            heat = row[j].imshow(volumes[i].T, aspect="auto", cmap="coolwarm", vmin=-bound, vmax=bound, interpolation="nearest")
            row[j].set(title=f"Window {i+1}: {label}", xlabel="Future update")
            row[j].set_yticks([0, 9.5, 19], ["Ask 10", "Best levels", "Bid 10"])
    fig.colorbar(heat, ax=axes.ravel().tolist(), shrink=.8, label="Signed log(1 + |volume|)")
    save(fig, "depth")
    return figures


def generate_timegan_examples(checkpoint_path, output_path, report_path, figure_path, count=64, seed=42, device="cpu", threads=4):
    if count < 1 or seed < 0 or seed >= 2**63 - 10000 or threads < 1:
        raise ValueError("Invalid count, seed or thread count.")
    torch.set_num_threads(threads)
    model, scaler, checkpoint = load_timegan(checkpoint_path, device)
    rng = torch.Generator().manual_seed(seed + 10000)
    pool = checkpoint["training_price_anchors"]
    indices = torch.randint(len(pool), (count,), generator=rng)
    anchors = pool[indices].numpy()
    volumes = checkpoint.get("training_volume_anchors")
    volumes = None if volumes is None else volumes[indices].numpy()
    started = perf_counter()
    context = anchor_context(anchors, checkpoint.get("anchor_context_scaler"), checkpoint["config"].get("anchor_conditioning", "none"), volumes=volumes, volume_mode=checkpoint["config"].get("volume_conditioning", "none"))
    features = model.generate(count, 32, seed, checkpoint["config"]["batch_size"], context=context, anchors=anchors)
    elapsed = perf_counter() - started
    books = decode_batch(features, scaler, anchors)
    output_path, report_path = Path(output_path), Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, scaled_features=features.numpy(), orderbooks=books, initial_prices=anchors)
    save_preview(books[0], anchors[0], figure_path, model_name="TimeGAN")
    report = dict(model="timegan", checkpoint_sha256=sha256_file(Path(checkpoint_path)), seed=seed, sequences=count,
                  generated_shape=list(features.shape), all_decoded_values_finite=bool(np.isfinite(books).all()),
                  generation_seconds=elapsed, sequences_per_second=count / elapsed, device=device,
                  anchor_conditioning=checkpoint["config"].get("anchor_conditioning", "none"),
                  volume_conditioning=checkpoint["config"].get("volume_conditioning", "none"),
                  scope="Seeded generation with sampled training anchors and the checkpoint's conditioning mode; not validation quality results.")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("ar", "timegan"), default="ar")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--num-sequences", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    folder = PROJECT_DIR / "outputs" / args.model
    checkpoint = args.checkpoint or folder / ("selected.pt" if args.model == "timegan" else "model.pt")
    output, report_path, figure = args.output or folder / "generated.npz", args.report or folder / "generation.json", args.figure or folder / "preview.png"
    if args.model == "timegan":
        report = generate_timegan_examples(checkpoint, output, report_path, figure, args.num_sequences, args.seed, args.device, args.threads)
    else:
        report = generate_examples(checkpoint, output, report_path, figure, args.num_sequences, args.seed, args.threads)
    print(json.dumps(report, indent=2))
    print(f"Saved generated arrays: {output}\nSaved preview: {figure}")


if __name__ == "__main__":
    main()
