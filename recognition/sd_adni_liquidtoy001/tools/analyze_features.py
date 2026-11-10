"""Descriptive VAE distances and t-SNE; never interpret 2D distances as coverage."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def stats(values):
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    x = np.load(args.input / "private-features.npz")["features"]
    rows = json.loads((args.input / "private-rows.json").read_text())
    groups = np.array([r["group"] for r in rows])
    real = int(np.isin(groups, ("train", "validation")).sum())
    if x.shape != (len(rows), 256) or real != 204 or not np.isfinite(x).all():
        raise ValueError("Invalid feature matrix")
    # Generated groups in the order extract_features wrote them.
    generated_groups = list(
        dict.fromkeys(g for g in groups if g not in ("train", "validation"))
    )
    labels = np.array([r["label"] for r in rows])
    report = {"metric": "unstandardized pooled VAE feature RMSE", "labels": {}}
    for label in ("all", "AD", "NC"):
        mask = np.ones(len(rows), bool) if label == "all" else labels == label
        train = x[(groups == "train") & mask]
        val = x[(groups == "validation") & mask]
        matched = np.concatenate(
            [
                x[(groups == "train") & (labels == c)][:32]
                for c in ("AD", "NC")
                if label in ("all", c)
            ]
        )
        distances = {
            "real_validation_to_matched_train": stats(cdist(val, train).min(1) / 16),
            "real_validation_to_count_matched_train": stats(
                cdist(val, matched).min(1) / 16
            ),
        }
        for group in generated_groups:
            generated = x[(groups == group) & mask]
            distances[group] = {
                "n": len(generated),
                "to_train": stats(cdist(generated, train).min(1) / 16),
                "to_validation": stats(cdist(generated, val).min(1) / 16),
                "validation_to_generated": stats(cdist(val, generated).min(1) / 16),
            }
        report["labels"][label] = distances
    with threadpool_limits(limits=2):
        projection = TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
            random_state=3710,
        ).fit_transform(x)
    np.save(args.input / "private-tsne.npy", projection)
    fig, ax = plt.subplots(figsize=(11, 8))
    shapes = ("x", "^", "D", "v", "P", "*", "X", "h", "<", ">")
    if len(generated_groups) > len(shapes):
        raise ValueError(
            f"Only {len(shapes)} marker shapes for {len(generated_groups)} groups"
        )
    markers = dict(zip(generated_groups, shapes))
    for group, marker in [("train", "o"), ("validation", "s"), *markers.items()]:
        for label, color in (("AD", "#d55e00"), ("NC", "#0072b2")):
            mask = (groups == group) & (labels == label)
            ax.scatter(
                *projection[mask].T,
                marker=marker,
                color=color,
                s=32,
                alpha=0.65,
                label=f"{group} / {label}",
            )
    ax.set(
        title=f"Frozen VAE features: joint t-SNE ({len(rows)} samples)",
        xlabel="t-SNE 1",
        ylabel="t-SNE 2",
    )
    ax.legend(loc="best", fontsize=8)
    fig.text(
        0.02,
        0.02,
        "Generated colors denote requested prompts, not verified disease labels. 2D distances are not coverage metrics.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.input / "tsne.png", dpi=180)
    plt.close(fig)
    (args.input / "aggregate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["labels"]["all"], indent=2))


if __name__ == "__main__":
    main()
