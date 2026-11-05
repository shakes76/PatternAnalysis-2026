"""
train.py
Author: Derek McKinney (dm-uq), SN: 45506053

Contains training and evaluation utilities used by the project experiments.

References: None

"""
from typing import TypeAlias

from collections.abc import Callable

import copy
import time

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from modules import (
    MLPNetwork,
    MLPNetwork3Layer,
    MLPNetwork4Layer,
    MLPNetwork5Layer,
    MLPNetwork3LayerELU,
    MLPNetwork3LayerLReLU,
    GCNNetwork,
    GCNNetwork3Layer,
    GCNNetwork4Layer,
    GCNNetwork5Layer,
    GCNNetwork4LayerELU,
    GCNNetwork4LayerLReLU,
)

# define MLP model type alias
MLPModel: TypeAlias = (
        MLPNetwork
        | MLPNetwork3Layer
        | MLPNetwork4Layer
        | MLPNetwork5Layer
        | MLPNetwork3LayerELU
        | MLPNetwork3LayerLReLU
)

# define GCN model type alias
GCNModel: TypeAlias = (
        GCNNetwork
        | GCNNetwork3Layer
        | GCNNetwork4Layer
        | GCNNetwork5Layer
        | GCNNetwork4LayerELU
        | GCNNetwork4LayerLReLU
)


def _train_epoch(
        model: nn.Module,
        forward_pass: Callable[[], Tensor],
        labels: Tensor,
        train_mask: Tensor,
        criterion: Callable[[Tensor, Tensor], Tensor],
        optimizer: Optimizer,
) -> float:
    """
    Perform shared optimisation steps for one training epoch.

    Args:
        model: Model being trained.
        forward_pass: Callable that performs the model-specific forward pass
            and returns raw class logits.
        labels: Node class-label tensor.
        train_mask: Boolean mask identifying training nodes.
        criterion: Loss function used for optimisation.
        optimizer: Optimizer used to update model parameters.

    Returns:
        Training loss for the epoch.

    """
    model.train()
    optimizer.zero_grad()  # clear gradients from the previous step

    logits = forward_pass()  # perform the model-specific forward pass

    # compute the loss using only the training nodes
    loss = criterion(
        logits[train_mask],
        labels[train_mask],
    )

    loss.backward()  # compute gradients via backpropagation
    optimizer.step()  # update model parameters

    return float(loss.item())


@torch.no_grad()
def _evaluate(
        model: nn.Module,
        forward_pass: Callable[[], Tensor],
        labels: Tensor,
        mask: Tensor,
        criterion: Callable[[Tensor, Tensor], Tensor],
) -> tuple[float, float]:
    """
    Perform shared evaluation steps for nodes selected by the provided mask.

    Args:
        model: Model to evaluate.
        forward_pass: Callable that performs the model-specific forward pass
            and returns raw class logits.
        labels: Node class-label tensor.
        mask: Boolean mask identifying nodes to evaluate.
        criterion: Loss function used for evaluation.

    Returns:
        Evaluation loss and classification accuracy.

    """
    model.eval()  # switch to evaluation mode

    logits = forward_pass()  # perform the model-specific forward pass

    # select only the nodes included in the evaluation split
    masked_logits = logits[mask]
    masked_labels = labels[mask]

    loss = criterion(
        masked_logits,
        masked_labels,
    )

    # select the highest-scoring class for each node
    predictions = masked_logits.argmax(dim=1)

    # calculate the proportion of correctly classified nodes
    accuracy = (
            predictions == masked_labels
    ).float().mean()

    return float(loss.item()), float(accuracy.item())


def train_mlp_epoch(
        model: MLPModel,
        features: Tensor,
        labels: Tensor,
        train_mask: Tensor,
        criterion: Callable[[Tensor, Tensor], Tensor],
        optimizer: Optimizer,
) -> float:
    """
    Train an MLP for one epoch using the selected training nodes.

    Args:
        model: MLP model to train.
        features: Node-feature tensor.
        labels: Node class-label tensor.
        train_mask: Boolean mask identifying training nodes.
        criterion: Loss function used for optimisation.
        optimizer: Optimizer used to update model parameters.

    Returns:
        Training loss for the epoch.

    """
    return _train_epoch(
        model=model,
        forward_pass=lambda: model(features),
        labels=labels,
        train_mask=train_mask,
        criterion=criterion,
        optimizer=optimizer,
    )


def evaluate_mlp(
        model: MLPModel,
        features: Tensor,
        labels: Tensor,
        mask: Tensor,
        criterion: Callable[[Tensor, Tensor], Tensor],
) -> tuple[float, float]:
    """
    Evaluate an MLP on the nodes selected by a mask.

    Args:
        model: MLP model to evaluate.
        features: Node-feature tensor.
        labels: Node class-label tensor.
        mask: Boolean mask identifying nodes to evaluate.
        criterion: Loss function used for evaluation.

    Returns:
        Evaluation loss and classification accuracy.

    """
    return _evaluate(
        model=model,
        forward_pass=lambda: model(features),
        labels=labels,
        mask=mask,
        criterion=criterion,
    )


def train_mlp_model(
        model: MLPModel,
        features: Tensor,
        labels: Tensor,
        train_mask: Tensor,
        val_mask: Tensor,
        criterion: Callable[[Tensor, Tensor], Tensor],
        optimizer: Optimizer,
        epochs: int,
        print_every: int = 100,
):
    """
    Train an MLP model and retain the state with the lowest validation loss.

    Runs full-batch training for the requested number of epochs, evaluating
    the model on the validation split after each epoch. Training loss,
    validation loss, and validation accuracy are recorded for later
    analysis and visualisation.

    :param model: MLP model to train.
    :param features: Node feature tensor.
    :param labels: Ground-truth node class labels.
    :param train_mask: Boolean mask selecting training nodes.
    :param val_mask: Boolean mask selecting validation nodes.
    :param criterion: Loss function used for training and validation.
    :param optimizer: Optimizer used to update the model parameters.
    :param epochs: Number of training epochs to run.
    :param print_every: Interval between printed training updates.
    :return: Dictionary containing the training history, best validation
        loss, best epoch, best model state, and elapsed training time.
    """
    # guard against div by 0
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    if print_every < 1:
        raise ValueError("print_every must be at least 1")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None

    print("> Training MLP")

    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss = train_mlp_epoch(
            model=model,
            features=features,
            labels=labels,
            train_mask=train_mask,
            criterion=criterion,
            optimizer=optimizer,
        )

        val_loss, val_accuracy = evaluate_mlp(
            model=model,
            features=features,
            labels=labels,
            mask=val_mask,
            criterion=criterion,
        )

        # record training history for later visualisation
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        # retain the model with the lowest validation loss
        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )

        if (
                epoch == 1
                or epoch % print_every == 0
                or epoch == epochs
        ):
            print(
                f"Epoch [{epoch:4}/{epochs}] "
                f"Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_loss:.5f} | "
                f"Val Acc: {val_accuracy:.2%}"
            )

    elapsed = time.perf_counter() - start

    print(
        f"\nTraining took {elapsed:.2f} secs "
        f"or {elapsed / 60:.2f} mins in total."
    )

    print(
        f"Best validation loss: {best_val_loss:.5f} "
        f"at epoch {best_epoch}."
    )

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_state": best_state,
        "elapsed": elapsed,
    }


def train_gcn_epoch(
        model: GCNModel,
        features: Tensor,
        edge_index: Tensor,
        labels: Tensor,
        train_mask: Tensor,
        criterion: Callable[[Tensor, Tensor], Tensor],
        optimizer: Optimizer,
) -> float:
    """
    Train a GCN for one epoch using the selected training nodes.

    Args:
        model: GCN model to train.
        features: Node-feature tensor.
        edge_index: Edge index tensor.
        labels: Node class-label tensor.
        train_mask: Boolean mask identifying training nodes.
        criterion: Loss function used for optimisation.
        optimizer: Optimizer used to update model parameters.

    Returns:
        Training loss for the epoch.

    """
    return _train_epoch(
        model=model,
        forward_pass=lambda: model(features, edge_index),
        labels=labels,
        train_mask=train_mask,
        criterion=criterion,
        optimizer=optimizer,
    )


def evaluate_gcn(
        model: GCNModel,
        features: Tensor,
        edge_index: Tensor,
        labels: Tensor,
        mask: Tensor,
        criterion: Callable[[Tensor, Tensor], Tensor],
) -> tuple[float, float]:
    """
    Evaluate a GCN on the nodes selected by a mask.

    Args:
        model: GCN model to evaluate.
        features: Node-feature tensor.
        edge_index: Edge index tensor.
        labels: Node class-label tensor.
        mask: Boolean mask identifying nodes to evaluate.
        criterion: Loss function used for evaluation.

    Returns:
        Evaluation loss and classification accuracy.

    """
    return _evaluate(
        model=model,
        forward_pass=lambda: model(features, edge_index),
        labels=labels,
        mask=mask,
        criterion=criterion,
    )


def train_gcn_model(
        model: GCNModel,
        features: Tensor,
        edge_index: Tensor,
        labels: Tensor,
        train_mask: Tensor,
        val_mask: Tensor,
        criterion: Callable[[Tensor, Tensor], Tensor],
        optimizer: Optimizer,
        epochs: int,
        print_every: int = 100,
):
    """
    Train a GCN model and retain the state with the lowest validation loss.

    Runs full-batch training for the requested number of epochs, evaluating
    the model on the validation split after each epoch. Training loss,
    validation loss, and validation accuracy are recorded for later
    analysis and visualisation.

    :param model: GCN model to train.
    :param features: Node feature tensor.
    :param edge_index: Graph connectivity in PyG edge-index format.
    :param labels: Ground-truth node class labels.
    :param train_mask: Boolean mask selecting training nodes.
    :param val_mask: Boolean mask selecting validation nodes.
    :param criterion: Loss function used for training and validation.
    :param optimizer: Optimizer used to update the model parameters.
    :param epochs: Number of training epochs to run.
    :param print_every: Interval between printed training updates.
    :return: Dictionary containing the training history, best validation
        loss, best epoch, best model state, and elapsed training time.
    """
    # guard against div by 0
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    if print_every < 1:
        raise ValueError("print_every must be at least 1")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None

    print("> Training GCN")

    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss = train_gcn_epoch(
            model=model,
            features=features,
            edge_index=edge_index,
            labels=labels,
            train_mask=train_mask,
            criterion=criterion,
            optimizer=optimizer,
        )

        val_loss, val_accuracy = evaluate_gcn(
            model=model,
            features=features,
            edge_index=edge_index,
            labels=labels,
            mask=val_mask,
            criterion=criterion,
        )

        # record training history for later visualisation
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        # retain the model with the lowest validation loss
        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )

        if (
                epoch == 1
                or epoch % print_every == 0
                or epoch == epochs
        ):
            print(
                f"Epoch [{epoch:4}/{epochs}] "
                f"Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_loss:.5f} | "
                f"Val Acc: {val_accuracy:.2%}"
            )

    elapsed = time.perf_counter() - start

    print(
        f"\nTraining took {elapsed:.2f} secs "
        f"or {elapsed / 60:.2f} mins in total."
    )

    print(
        f"Best validation loss: {best_val_loss:.5f} "
        f"at epoch {best_epoch}."
    )

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_state": best_state,
        "elapsed": elapsed,
    }


def plot_training_validation_loss(
        history: dict[str, list[float]],
        best_epoch: int,
        title: str,
) -> tuple[Figure, Axes]:
    """
    Plot training and validation loss across training epochs.

    :param history: Training history containing training and validation losses.
    :param best_epoch: Epoch with the lowest validation loss.
    :param title: Figure title.
    :return: Matplotlib figure and axes containing the loss plot.
    """
    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        epochs,
        history["train_loss"],
        label="Training loss",
    )
    ax.plot(
        epochs,
        history["val_loss"],
        label="Validation loss",
    )
    ax.axvline(
        best_epoch,
        linestyle="--",
        label=f"Best epoch ({best_epoch})",
    )

    ax.set(
        xlabel="Epoch",
        ylabel="Loss",
        title=title,
    )
    ax.legend()

    sns.despine(ax=ax)
    fig.tight_layout()

    return fig, ax


def plot_training_validation_accuracy(
        history: dict[str, list[float]],
        best_epoch: int,
        title: str,
) -> tuple[Figure, Axes]:
    """
    Plot validation accuracy across training epochs.

    :param history: Training history containing validation accuracy.
    :param best_epoch: Epoch with the lowest validation loss.
    :param title: Figure title.
    :return: Matplotlib figure and axes containing the accuracy plot.
    """
    epochs = range(
        1,
        len(history["val_accuracy"]) + 1
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        epochs,
        history["val_accuracy"],
        label="Validation accuracy",
    )
    ax.axvline(
        best_epoch,
        linestyle="--",
        label=f"Best epoch ({best_epoch})",
    )

    ax.set(
        xlabel="Epoch",
        ylabel="Accuracy",
        title=title,
    )
    ax.yaxis.set_major_formatter(
        PercentFormatter(1.0)
    )
    ax.legend()

    sns.despine(ax=ax)
    fig.tight_layout()

    return fig, ax


def classification_metrics(
        logits: Tensor,
        labels: Tensor,
        mask: Tensor,
) -> dict:
    """
    Calculate classification metrics for nodes selected by a mask.

    :param logits: Raw class logits produced by the model.
    :param labels: Ground-truth node class labels.
    :param mask: Boolean mask selecting nodes to evaluate.
    :return: Dictionary containing aggregate and per-class metrics.
    """
    selected_logits = logits[mask]
    selected_labels = labels[mask]

    # convert raw logits and get prediction results
    probabilities = selected_logits.softmax(dim=1)
    predictions = probabilities.argmax(dim=1)

    y_true = selected_labels.detach().cpu().numpy()
    y_pred = predictions.detach().cpu().numpy()
    y_prob = probabilities.detach().cpu().numpy()

    # compute metrics
    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=range(selected_logits.shape[1]),
            zero_division=0,
        )
    )

    accuracy = (
            predictions == selected_labels
    ).float().mean().item()

    macro_auroc = roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr",
        average="macro",
    )

    return {
        "accuracy": accuracy,
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "macro_auroc": float(macro_auroc),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
    }
