"""Save SVG figures for model structure, metrics and image comparisons.

Plotting functions create parent directories and replace existing SVGs.
Run this module to redraw the architecture figure in the package's figures
directory. No model training or inference is performed here.

Author: Alex Anderson (47472442), COMP3710 Pattern Recognition, 2026.
"""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .dataset import MEAN, STD, ISICDataset, Sample

FIGURES = Path(__file__).parent / "figures"
COLOURS = {0: "#1f77b4", 1: "#c45c35"}  # benign, melanoma


def _save(figure: Figure, path: Path) -> None:
    """Save an SVG figure and close it.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to lay out, save and close.
    path : pathlib.Path
        Destination SVG file. Parent directories are created, and any
        existing file is replaced.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(h_pad=2)
    figure.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(figure)


def _thumbnail(path: Path) -> Image.Image:
    """Read an oriented RGB thumbnail without cropping the image.

    Parameters
    ----------
    path : pathlib.Path
        Image file to open.

    Returns
    -------
    PIL.Image.Image
        RGB image at most 512 pixels per side, keeping its aspect ratio.
    """
    with Image.open(path) as image:
        image.draft("RGB", (512, 512))
        image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((512, 512))
    return image


def plot_architecture(path: Path) -> None:
    """Draw the forward pass, combined loss and weight updates.

    Parameters
    ----------
    path : pathlib.Path
        Destination SVG file.
    """
    figure, axis = plt.subplots(figsize=(12.5, 4.5))
    axis.set(xlim=(-0.04, 1.10), ylim=(-0.10, 1.80))
    axis.axis("off")
    nodes = {
        "images": (0.06, 0.45, "RGB batch\n32 images\n3 × 384 × 384 each"),
        "encoder": (0.27, 0.45, "Shared ResNet-18\nGlobal average pool\n512 features / image"),
        "logits": (0.50, 0.75, "Linear 512 → 2\nClass logits"),
        "embedding": (0.50, 0.15, "L2 normalisation\n512-d embedding"),
        "classification": (0.72, 0.75, "Both models\nCross-entropy (CE)"),
        "contrastive": (0.72, 0.15, "Siamese training only\nContrastive loss\n496 pairs / batch"),
        "total": (0.95, 0.45, "Total training loss\nCE + λ × contrastive\nλ = 0: baseline\nλ = 0.5: Siamese"),
    }
    training_colours = {
        "classification": ("#fff3df", "#a56b1f"),
        "contrastive": ("#f1e7fb", "#7d4db2"),
        "total": ("#e8f4ed", "#2b7050"),
    }
    labels = {}
    for name, (x, y, label) in nodes.items():
        fill, edge = training_colours.get(name, ("#edf3f8", "#2878b5"))
        box = {"boxstyle": "round,pad=0.5", "fc": fill, "ec": edge}
        labels[name] = axis.text(x, y, label, ha="center", va="center", fontsize=12.5, bbox=box)
    for start, end in (
        ("images", "encoder"), ("encoder", "logits"), ("encoder", "embedding"),
        ("logits", "classification"), ("embedding", "contrastive"),
        ("classification", "total"), ("contrastive", "total"),
    ):
        colour = training_colours.get(start, training_colours.get(end, ("", "#34414d")))[1]
        axis.annotate("", xy=nodes[end][:2], xytext=nodes[start][:2], arrowprops={
            "arrowstyle": "->", "color": colour,
            "patchA": labels[start].get_bbox_patch(), "patchB": labels[end].get_bbox_patch(),
            "shrinkA": 5, "shrinkB": 5,
        })
    feedback: dict[str, Any] = {"color": "#2b7050", "linestyle": "--", "linewidth": 1.2}
    axis.annotate("", xy=(0.95, 1.25), xytext=nodes["total"][:2], arrowprops={
        **feedback, "arrowstyle": "-", "patchA": labels["total"].get_bbox_patch(), "shrinkA": 5, "shrinkB": 0,
    })
    axis.plot([0.95, 0.27], [1.25, 1.25], **feedback)
    for name in ("encoder", "logits"):
        axis.annotate("", xy=nodes[name][:2], xytext=(nodes[name][0], 1.25), arrowprops={
            **feedback, "arrowstyle": "->", "patchB": labels[name].get_bbox_patch(), "shrinkA": 0, "shrinkB": 5,
        })
    axis.text(
        0.61, 1.53, "Backpropagation + AdamW weight updates\nEncoder: total loss · classifier: cross-entropy only",
        ha="center", va="center", fontsize=13, color="#2b7050",
    )
    _save(figure, path)


def plot_training_history(history: list[dict[str, float]], path: Path) -> None:
    """Save training losses and selection ranking scores by epoch.

    Parameters
    ----------
    history : list of dict of str to float
        Epoch records containing epoch, train_cross_entropy,
        train_contrastive, selection_cross_entropy, selection_auroc and
        selection_average_precision.
    path : pathlib.Path
        Destination SVG file.
    """
    epochs = [h["epoch"] for h in history]
    figure, (losses, ranking) = plt.subplots(1, 2, figsize=(11, 4))
    for key, label in (
        ("train_cross_entropy", "Training cross-entropy"),
        ("train_contrastive", "Training contrastive"),
        ("selection_cross_entropy", "Selection cross-entropy"),
    ):
        losses.plot(epochs, [h[key] for h in history], "o-", markersize=3, label=label)
    for key, label in (("selection_auroc", "Selection AUROC"), ("selection_average_precision", "Selection AP")):
        ranking.plot(epochs, [h[key] for h in history], "o-", markersize=3, label=label)
    losses.set(title="Losses", xlabel="Epoch", ylabel="Loss")
    ranking.set(title="Selection ranking", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    for axis in (losses, ranking):
        axis.grid(alpha=0.3)
        axis.legend(fontsize="small")
    _save(figure, path)


def reliability_bins(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Summarise eight quantile bins, retaining bins with zero melanomas.

    Parameters
    ----------
    labels : numpy.ndarray of shape (images,)
        Class labels: 0 for benign and 1 for melanoma.
    probabilities : numpy.ndarray of shape (images,)
        Calibrated melanoma probabilities.

    Returns
    -------
    predicted : numpy.ndarray of shape (occupied_bins,)
        Mean probability in each nonempty bin.
    observed : numpy.ndarray of shape (occupied_bins,)
        Melanoma fraction in each nonempty bin.
    positives : numpy.ndarray of shape (occupied_bins,)
        Melanoma count in each nonempty bin.
    counts : numpy.ndarray of shape (occupied_bins,)
        Image count in each nonempty bin.

    Notes
    -----
    Quantiles define up to eight occupied bins. Equal probabilities stay in
    the same bin, and bins with no melanomas are retained if nonempty.
    """
    edges = np.quantile(probabilities, np.linspace(0, 1, 9))
    bins = np.searchsorted(edges[1:-1], probabilities)
    counts = np.bincount(bins, minlength=8)
    positives = np.bincount(bins, weights=labels, minlength=8)
    totals = np.bincount(bins, weights=probabilities, minlength=8)
    occupied = counts > 0
    predicted, observed = totals[occupied] / counts[occupied], positives[occupied] / counts[occupied]
    return predicted, observed, positives[occupied].astype(int), counts[occupied]


def plot_curves(
    labels: np.ndarray,
    probabilities: np.ndarray,
    reference_scores: np.ndarray,
    rule: dict[str, Any],
    path: Path,
    title: str,
) -> None:
    """Save ROC, precision-recall and probability reliability plots.

    Parameters
    ----------
    labels : numpy.ndarray of shape (images,)
        Class labels: 0 for benign and 1 for melanoma.
    probabilities : numpy.ndarray of shape (images,)
        Calibrated classifier probabilities aligned with the labels.
    reference_scores : numpy.ndarray of shape (images,)
        Reference similarity scores aligned with the labels.
    rule : dict
        Fixed decision_threshold and reference_review_threshold values
        used to mark operating points on ROC and precision-recall plots.
    path : pathlib.Path
        Destination SVG file.
    title : str
        Figure title identifying the model, seed or patient partition.
    """
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), gridspec_kw={"height_ratios": (3.8, 1.4)})
    roc, pr, reliability, bins_axis = axes[0]
    for axis in axes[1]:
        axis.axis("off")
    for scores, name, colour in ((probabilities, "Classifier", "#087e83"), (reference_scores, "References", "#9467bd")):
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        auroc, ap = roc_auc_score(labels, scores), average_precision_score(labels, scores)
        roc.plot(fpr, tpr, color=colour, linewidth=2, label=f"{name} AUC {auroc:.3f}")
        pr.plot(recall, precision, color=colour, linewidth=2, label=f"{name} AP {ap:.3f}")
    for scores, key, marker in (
        (probabilities, "decision_threshold", "o"), (reference_scores, "reference_review_threshold", "s")
    ):
        flagged = scores >= rule[key]
        recall = flagged[labels == 1].mean()
        cutoff = "Binary cutoff" if key == "decision_threshold" else "Review cutoff"
        roc.scatter(flagged[labels == 0].mean(), recall, marker=marker, color="black", zorder=3, label=cutoff)
        pr.scatter(recall, labels[flagged].mean(), marker=marker, color="black", zorder=3, label=cutoff)
    roc.plot([0, 1], [0, 1], "--", color="grey")
    pr.axhline(labels.mean(), ls="--", color="grey", label=f"Prevalence = {labels.mean():.2%}")
    predicted, observed, positives, counts = reliability_bins(labels, probabilities)
    reliability.plot(predicted, observed, "o-", color="#087e83", label="Classifier")
    bins_axis.axis("off")
    bins_axis.set_title("Reliability bins (8)", fontsize=18)
    counts_table = bins_axis.table(
        cellText=[
            [f"{p:.4g}", f"{positive}/{count}"] for p, positive, count in zip(predicted, positives, counts, strict=True)
        ],
        colLabels=["Mean p", "Mel./n"], bbox=(0, 0.08, 1, 0.84),
    )
    counts_table.auto_set_font_size(False)
    counts_table.set_fontsize(16)
    for (row, _), cell in counts_table.get_celld().items():
        cell.set_edgecolor("#cbd5df")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#edf3f8")
            cell.set_text_props(weight="bold")
    reliability.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    reliability.set(xscale="log", xlim=(1e-4, 1), ylim=(-0.01, max(0.15, float(observed.max()) * 1.25)))
    roc.set(title="ROC", xlabel="Benign false-positive rate", ylabel="Melanoma recall")
    pr.set(title="Precision-recall", xlabel="Melanoma recall", ylabel="Melanoma precision", ylim=(0, 1))
    reliability.set(
        title="Reliability", xlabel="Mean predicted probability\n(log scale)", ylabel="Observed melanoma rate"
    )
    for axis, legend_axis in zip((roc, pr, reliability), axes[1], strict=False):  # fourth axis holds the table
        axis.grid(alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=14)
        axis.xaxis.label.set_size(16)
        axis.yaxis.label.set_size(16)
        axis.title.set_size(18)
        handles, names = axis.get_legend_handles_labels()
        legend_axis.legend(handles, names, loc="upper left", borderaxespad=0, fontsize=14, frameon=False)
    figure.suptitle(f"{title} · {len(labels):,} images / {labels.sum()} melanomas", fontsize=18)
    _save(figure, path)


def plot_embeddings(
    gallery: np.ndarray, gallery_labels: np.ndarray, queries: np.ndarray, query_labels: np.ndarray,
    path: Path, partition: str,
) -> None:
    """Save a two-dimensional PCA view of references and query embeddings.

    Parameters
    ----------
    gallery : numpy.ndarray of shape (references, features)
        Training reference embeddings used to fit PCA.
    gallery_labels : numpy.ndarray of shape (references,)
        Reference labels: 0 for benign and 1 for melanoma.
    queries : numpy.ndarray of shape (images, features)
        Query embeddings projected using the fitted reference PCA.
    query_labels : numpy.ndarray of shape (images,)
        Query labels: 0 for benign and 1 for melanoma.
    path : pathlib.Path
        Destination SVG file.
    partition : str
        Patient partition name shown in the title and legend.
    """
    pca = PCA(n_components=2).fit(gallery)
    references, projected = pca.transform(gallery), pca.transform(queries)
    figure, axis = plt.subplots(figsize=(8, 6))
    for label, name in ((0, "benign"), (1, "melanoma")):
        axis.scatter(
            *references[gallery_labels == label].T, s=30, facecolors="none", edgecolors=COLOURS[label],
            alpha=0.6, label=f"Training references · {name}",
        )
        axis.scatter(
            *projected[query_labels == label].T, s=14 + 30 * label, marker="x", color=COLOURS[label],
            alpha=0.35 + 0.5 * label, label=f"{partition} · {name}",
        )
    variance = pca.explained_variance_ratio_
    axis.set(
        title=f"Embeddings · {partition}",
        xlabel=f"PC1 ({variance[0]:.0%} of reference variance)", ylabel=f"PC2 ({variance[1]:.0%})",
    )
    axis.legend(fontsize=8)
    _save(figure, path)


def plot_pair_distances(distances: np.ndarray, pair_types: np.ndarray, path: Path, title: str) -> None:
    """Save embedding-distance histograms for each kind of image pair.

    Parameters
    ----------
    distances : numpy.ndarray of shape (pairs,)
        Euclidean distances between unit-length embeddings, from 0 to 2.
    pair_types : numpy.ndarray of shape (pairs,)
        Melanomas in each pair: 0 benign-benign, 1 mixed, 2 melanoma-melanoma.
    path : pathlib.Path
        Destination SVG file.
    title : str
        Figure title identifying the model and patient partition.
    """
    figure, axis = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 2, 81).tolist()
    for kind, name, colour in (
        (0, "Benign–benign", COLOURS[0]), (1, "Melanoma–benign", "#7f7f7f"), (2, "Melanoma–melanoma", COLOURS[1])
    ):
        values = distances[pair_types == kind]
        label = f"{name} · {len(values):,} pairs · mean {values.mean():.2f}"
        axis.hist(values, bins, density=True, histtype="step", linewidth=2, color=colour, label=label)
    axis.set(title=title, xlabel="Embedding distance (unit-length embeddings, 0–2)", ylabel="Density")
    axis.legend(fontsize=9)
    _save(figure, path)


def plot_cases(cases: list[dict[str, Any]], path: Path) -> None:
    """Save query images beside their nearest training references.

    Parameters
    ----------
    cases : list of dict
        Nonempty case records containing path, image_name, actual,
        probability, decision and reference_score for each query, plus
        reference_path, reference_name and reference_actual for its match.
    path : pathlib.Path
        Destination SVG file.

    Notes
    -----
    Images are thumbnails of the originals, not the model's input crops.
    """
    figure, axes = plt.subplots(len(cases), 2, figsize=(7.5, 3.9 * len(cases)), squeeze=False)
    for (query, reference), case in zip(axes, cases, strict=True):
        query.imshow(_thumbnail(case["path"]))
        query.set_title(
            f"{case['image_name']} · actual {case['actual']}\np = {case['probability']:.3g} → {case['decision']}"
            f"\nreference score = {case['reference_score']:.2f}",
            fontsize=9,
        )
        reference.imshow(_thumbnail(case["reference_path"]))
        reference.set_title(
            f"Nearest reference {case['reference_name']}\nactual {case['reference_actual']}", fontsize=9
        )
        query.axis("off")
        reference.axis("off")
    _save(figure, path)


def plot_case_comparison(
    samples: list[Sample], predictions: dict[str, dict[str, np.ndarray]], image_size: int, path: Path
) -> None:
    """Show original images, evaluation crops and both models' results.

    Parameters
    ----------
    samples : list of Sample
        The same selected cases for both models, in display order.
    predictions : dict of str to dict of str to numpy.ndarray
        Baseline and Siamese entries keyed by baseline and siamese. Each
        contains probability, reference_score and decision arrays aligned
        with the selected samples.
    image_size : int
        Evaluation crop size from the saved training configuration.
    path : pathlib.Path
        Destination SVG file.

    Notes
    -----
    Both models receive the same evaluation crop shown on the right. The
    left caption gives baseline results; the right gives Siamese results.
    """
    dataset = ISICDataset(samples, image_size)
    figure, axes = plt.subplots(len(samples), 2, figsize=(8, 4.6 * len(samples)), squeeze=False)
    for index, (sample, (original, cropped)) in enumerate(zip(samples, axes, strict=True)):
        normalised, _ = dataset[index]
        image = normalised.permute(1, 2, 0).numpy() * np.asarray(STD) + np.asarray(MEAN)
        original.imshow(_thumbnail(sample.path))
        cropped.imshow(image.clip(0, 1))
        actual = ("benign", "melanoma")[sample.label]
        original.set_title(f"{sample.image_name} · actual {actual}\nOriginal image", fontsize=11)
        cropped.set_title(f"Evaluation input (both models)\n{image_size} × {image_size} centre crop", fontsize=11)
        for axis, model in ((original, "baseline"), (cropped, "siamese")):
            values = predictions[model]
            axis.set_xlabel(
                f"{model.capitalize()}: p={values['probability'][index]:.6g}, "
                f"r={values['reference_score'][index]:.3f}\nOutput: {values['decision'][index]}",
                fontsize=10,
            )
            axis.set(xticks=[], yticks=[])
            for spine in axis.spines.values():
                spine.set_visible(False)
    _save(figure, path)


if __name__ == "__main__":
    plot_architecture(FIGURES / "architecture.svg")
