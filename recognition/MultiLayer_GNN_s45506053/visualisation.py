"""
visualisation.py
Author: Derek McKinney (dm-uq), SN: 45506053

Contains utilities used to visualise learned model representations.

References:

    05_oasis_vae.ipynb (unpublished code, COMP3710 Assignment 2) [Sem. 2, 2026]

"""

# IMPORTS

from pathlib import Path
from typing import Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from matplotlib.figure import Figure
from torch import Tensor

from dataset import CLASS_NAMES

# CONSTANTS

sns.set_theme(context="notebook", style="whitegrid")

ReductionMethod: TypeAlias = Literal["umap, tsne"]


def reduce_embeddings(
        embeddings: Tensor,
        random_state: int = 808,
) -> np.ndarray:
    """
    Reduce learned node embeddings to two dimensions using UMAP.

    The standard UMAP values are retained for neighbourhood size, minimum
    distance, output dimensionality, and distance metric. A fixed random
    state and single-worker execution are used for reproducibility.

    Args:
        embeddings: Learned node representation of shape
            `(num_nodes, hidden_channels)`.
        random_state: Random seed used for reproducible UMAP reduction.

    Returns:
        Two-dimensional node embedding of shape `(num_nodes, 2)`.

    """
    # move the learned representation to CPU for UMAP reduction
    z = embeddings.detach().cpu().numpy()

    # construct a reproducible two-dimensional UMAP reducer
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=random_state,
        n_jobs=1,
    )

    # reduce the latent representation to two dimensions
    embedding_2d = reducer.fit_transform(z)

    if not np.isfinite(embedding_2d).all():
        raise ValueError(
            "UMAP produced non-finite embedding coordinates."
        )

    return embedding_2d


def plot_embeddings(
        embedding_2d: np.ndarray,
        labels: Tensor,
        model_name: str,
        output_path: Path | None = None,
) -> Figure:
    """
    Plot two-dimensional node embeddings coloured by ground-truth class.

    The plotting workflow is adapted from my unpublished COMP3710 Assignment 2
    VAE latent-space visualisation work. [Semester 2, 2026]

    Args:
        embedding_2d: Two-dimensional node embedding of shape
            `(num_nodes, 2)`.
        labels: Ground-truth class labels for each graph node.
        model_name: Display name of the model being visualised.
        output_path: Optional path at which to save the completed figure.

    Returns:
        Completed Matplotlib figure containing the embedding plot.

    """
    # convert class indices to categorical page-class labels
    class_labels = [
        CLASS_NAMES[label]
        for label in labels.detach().cpu().tolist()
    ]

    # plot the two-dimensional latent representation
    fig_embedding, ax_embedding = plt.subplots(
        figsize=(8, 6),
    )

    # plot embeddings using ground-truth class as categorical hue
    sns.scatterplot(
        x=embedding_2d[:, 0],
        y=embedding_2d[:, 1],
        hue=class_labels,
        hue_order=list(CLASS_NAMES.values()),
        s=14,
        alpha=0.65,
        linewidth=0,
        ax=ax_embedding,
    )

    # label the reduced embedding space
    ax_embedding.set(
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        title=f"{model_name} Learned Node Embeddings",
    )

    ax_embedding.legend(
        title="Page class",
    )

    sns.despine(ax=ax_embedding)
    fig_embedding.tight_layout()

    # save completed figure if path was provided
    if output_path is not None:
        fig_embedding.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
        )

    return fig_embedding
