"""
predict.py
Author: Derek McKinney (dm-uq), SN: 45506053

Contains prediction and evaluation utilities used to conduct project analysis.

Frozen Models:

    MLP:
        layers: 3
        hidden: 64
        dropout: 0.5
        activation: ReLU
        weight_decay: n/a for inference
        checkpoint: mlp_final.pt

    GCN:
        layers: 4
        hidden: 64
        dropout: 0.55
        activation: ReLU
        weight_decay: n/a for inference
        checkpoint: gcn_final.pt

References: None

"""

# IMPORTS

from pathlib import Path
from typing import Final, Literal, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import FacebookPagePage

from dataset import add_stratified_masks, load_facebook_data, CLASS_NAMES
from modules import MLPNetwork3Layer, GCNNetwork4Layer
from profiling import profile_model, print_profile
from train import classification_metrics
from visualisation import reduce_embeddings, plot_embeddings

# CONSTANTS

ModelName: TypeAlias = Literal["mlp", "gcn"]

PredictionModel: TypeAlias = (MLPNetwork3Layer | GCNNetwork4Layer)

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parent
CHECKPOINTS_DIR: Final[Path] = PROJECT_DIR / "checkpoints"
IMAGES_DIR: Final[Path] = PROJECT_DIR / "images" / "evaluation"

MLP_CHECKPOINT: Final[Path] = CHECKPOINTS_DIR / "mlp_final.pt"
GCN_CHECKPOINT: Final[Path] = CHECKPOINTS_DIR / "gcn_final.pt"


# device setup
def select_device(
        requested_device: str | torch.device | None = None,
) -> torch.device:
    """
    Select the device used for model inference.

    Args:
        requested_device: Optional device requested for inference.

    Returns:
        requested device, or CUDA when available, otherwise CPU.

    """
    if requested_device is not None:
        return torch.device(requested_device)

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def load_prediction_data(
        train_pct: int = 60,
        val_pct: int = 20,
        test_pct: int = 20,
        seed: int = 808,
) -> tuple[FacebookPagePage, Data]:
    """
    Load the Facebook Page-Page data and reconstruct the project data splits.

    Args:
        train_pct: Percentage of nodes assigned to training.
        val_pct: Percentage of nodes assigned to validation.
        test_pct: Percentage of nodes assigned to testing.
        seed: Random seed used for the stratified splits.

    Returns:
        Loaded dataset and graph data containing the reproducible split masks.

    """
    # load the Facebook Page-Page dataset
    dataset = load_facebook_data()

    # retrieve the graph data from the single-graph dataset
    data = cast(  # narrow the Typing scope to satisfy checker
        Data,
        dataset[0],
    )

    # reconstruct the stratified train, validation, and test masks
    data = add_stratified_masks(
        data,
        train_pct=train_pct,
        val_pct=val_pct,
        test_pct=test_pct,
        seed=seed,
    )

    return dataset, data


def load_mlp(
        checkpoint_path: Path,
        in_channels: int,
        out_channels: int,
        device: torch.device,
) -> MLPNetwork3Layer:
    """
    Load the selected final MLP model and trained weights.

    Args:
        checkpoint_path: Path to the saved MLP model state.
        in_channels: Number of input node features.
        out_channels: Number of output classes.
        device: Device on which to load the model.

    Returns:
        Loaded MLP model in evaluation mode.

    """
    # construct the selected final MLP architecture
    model = MLPNetwork3Layer(
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=out_channels,
        dropout=0.5,
    ).to(device)

    # load the selected trained model state
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_gcn(
        checkpoint_path: Path,
        in_channels: int,
        out_channels: int,
        device: torch.device,
) -> GCNNetwork4Layer:
    """
    Load the selected final GCN model and trained weights.

    Args:
        checkpoint_path: Path to the saved GCN model state.
        in_channels: Number of input node features.
        out_channels: Number of output classes.
        device: Device on which to load the model.

    Returns:
        Loaded GCN model in evaluation mode.

    """
    # construct the selected final GCN architecture
    model = GCNNetwork4Layer(
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=out_channels,
        dropout=0.55,
    ).to(device)

    # load the selected trained model state
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_model(
        model_name: ModelName,
        in_channels: int,
        out_channels: int,
        checkpoint_path: str | Path | None = None,
        requested_device: str | torch.device | None = None,
) -> tuple[PredictionModel, torch.device]:
    """
    Load a selected trained model for prediction.

    The model type determines the final architecture and default checkpoint.
    A checkpoint path or device may be provided to override the defaults.

    Args:
        model_name: Model to load - accepts either `"mlp"` or `"gcn"`.
        in_channels: Number of input node features.
        out_channels: Number of output classes.
        checkpoint_path: Optional path overriding the default model checkpoint.
        requested_device: Optional device on which to perform inference.

    Returns:
        A loaded model in evaluation mode, and its selected inference device.

    """
    # select the requested device, otherwise prefer CUDA when available
    device = select_device(requested_device)

    # select the model-specific default checkpoint unless overridden
    if model_name == "mlp":
        selected_checkpoint = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else MLP_CHECKPOINT
        )

    elif model_name == "gcn":
        selected_checkpoint = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else GCN_CHECKPOINT
        )

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Expected 'mlp' or 'gcn'."
        )

    # ensure selected checkpoint is available prior to model construction
    if not selected_checkpoint.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {selected_checkpoint}"
        )

    # perform model construction and state loading using the appropriate loader
    if model_name == "mlp":
        model = load_mlp(
            checkpoint_path=selected_checkpoint,
            in_channels=in_channels,
            out_channels=out_channels,
            device=device,
        )
    else:
        model = load_gcn(
            checkpoint_path=selected_checkpoint,
            in_channels=in_channels,
            out_channels=out_channels,
            device=device,
        )

    return model, device


@torch.inference_mode()
def run_inference(
        model: PredictionModel,
        data: Data,
) -> Tensor:
    """
    Run inference using a loaded prediction model.

    Args:
        model: Loaded MLP or GCN model in evaluation mode.
        data: Graph data containing node features and connectivity.

    Returns:
        Raw class logits for all graph nodes.

    """
    # ensure node features are available for prediction
    features = data.x

    if features is None:
        raise ValueError("Graph data does not contain node features.")

    # perform the model-specific forward pass
    if isinstance(model, MLPNetwork3Layer):  # mlp
        return model(features)

    if isinstance(model, GCNNetwork4Layer):  # cgn
        edge_index = data.edge_index

        if edge_index is None:
            raise ValueError("Graph data does not contain an edge index.")

        return model(
            features,
            edge_index,
        )

    # provided model arg is not of the correct type
    raise TypeError(
        f"Unsupported prediction model: {type(model).__name__}"
    )


def print_results(
        model_name: str,
        metrics: dict,
) -> None:
    """
    Print formatted classification results for a prediction model.

    Args:
        model_name: Display name of the evaluated model.
        metrics: Classification metrics produced for the model.

    Returns:
        None.

    """
    # aggregate results table
    print(f"\n{model_name} Aggregate Test Results")
    result_width = 26  # fixed width
    print("-" * result_width)  # title separator

    print(f"{'Accuracy:':<18}{metrics['accuracy']:.2%}")
    print(f"{'Macro Precision:':<18}{metrics['macro_precision']:.2%}")
    print(f"{'Macro Recall:':<18}{metrics['macro_recall']:.2%}")
    print(f"{'Macro F1:':<18}{metrics['macro_f1']:.2%}")
    print(f"{'Macro AUROC:':<18}{metrics['macro_auroc']:.2%}")

    # per-class results table
    print(f"\n{model_name} Per-Class Results")

    # get column widths for table separators
    class_width = 14
    precision_width = 12
    recall_width = 10
    f1_width = 10
    support_width = 10

    # compute total table width
    table_width = (
            class_width
            + precision_width
            + recall_width
            + f1_width
            + support_width
    )

    print("-" * table_width)  # title separator

    # print per-class table headings
    print(
        f"{'Class':<14}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'Support':>10}"
    )

    print("-" * table_width)  # header separator

    # loop over the class names to generate results output
    for class_id, class_name in CLASS_NAMES.items():
        print(
            f"{class_name:<14}"
            f"{metrics['precision'][class_id]:>12.2%}"
            f"{metrics['recall'][class_id]:>10.2%}"
            f"{metrics['f1'][class_id]:>10.2%}"
            f"{metrics['support'][class_id]:>10}"
        )
    print()


def compute_confusion_matrix(
        logits: Tensor,
        labels: Tensor,
        mask: Tensor,
) -> np.ndarray:
    """
    Compute the confusion matrix for a selected data split.

    Args:
        logits: Raw class logits for all graph nodes.
        labels: Ground-truth class labels for all graph nodes.
        mask: Boolean mask selecting the nodes to evaluate.

    Returns:
        Confusion matrix with true classes as rows and predicted classes
        as columns.

    """
    # select predictions and targets for the requested evaluation split
    predictions = logits[mask].argmax(dim=1)
    targets = labels[mask]

    # convert tensors for scikit-learn confusion-matrix calculation
    return confusion_matrix(
        targets.cpu().numpy(),
        predictions.cpu().numpy(),
    )


def print_confusion_matrix(
        model_name: str,
        matrix: np.ndarray,
) -> None:
    """
    Print a formatted confusion matrix for a prediction model.

    Args:
        model_name: Display name of the evaluated model.
        matrix: Confusion matrix with true classes as rows and predicted
            classes as columns.

    Returns:
        None.

    """
    # set column widths for class labels and prediction counts
    class_width = 14
    value_width = 12

    # print table title and separator
    print(f"\n{model_name} Confusion Matrix")
    print("-" * (class_width + value_width * len(CLASS_NAMES)))

    # print predicted classes as column headings
    axis_label = "Actual \\ Pred."
    print(f"{axis_label:<{class_width}}", end="")
    for class_name in CLASS_NAMES.values():
        print(f"{class_name:>{value_width}}", end="")
    print()

    print("-" * (class_width + value_width * len(CLASS_NAMES)))

    # print each actual class and its predicted-class counts
    for class_id, class_name in CLASS_NAMES.items():
        print(f"{class_name:<{class_width}}", end="")

        for count in matrix[class_id]:
            print(f"{count:>{value_width}}", end="")

        print()
    print()


def main():
    """
    Main function
    :return: None
    """

    # data
    dataset, data = load_prediction_data()

    # models
    mlp, mlp_device = load_model(
        model_name="mlp",
        in_channels=data.num_node_features,
        out_channels=dataset.num_classes,
    )

    # device
    gcn, gcn_device = load_model(
        model_name="gcn",
        in_channels=data.num_node_features,
        out_channels=dataset.num_classes,
    )

    # move graph tensors to the selected inference device
    data = data.to(mlp_device)

    # inference
    mlp_logits = run_inference(
        model=mlp,
        data=data,
    )

    gcn_logits = run_inference(
        model=gcn,
        data=data,
    )

    # metrics
    mlp_metrics = classification_metrics(
        logits=mlp_logits,
        labels=data.y,
        mask=data.test_mask,
    )

    gcn_metrics = classification_metrics(
        logits=gcn_logits,
        labels=data.y,
        mask=data.test_mask,
    )

    mlp_confusion = compute_confusion_matrix(
        logits=mlp_logits,
        labels=data.y,
        mask=data.test_mask,
    )

    gcn_confusion = compute_confusion_matrix(
        logits=gcn_logits,
        labels=data.y,
        mask=data.test_mask,
    )

    # profile final models
    mlp_profile = profile_model(
        model=mlp,
        data=data,
        device=mlp_device,
    )

    gcn_profile = profile_model(
        model=gcn,
        data=data,
        device=gcn_device,
    )

    # extract embedding data
    mlp.eval()
    gcn.eval()

    with torch.inference_mode():
        mlp_embeddings = mlp.extract_embeddings(
            data.x,
        )

        gcn_embeddings = gcn.extract_embeddings(
            data.x,
            data.edge_index,
        )

    # reduce embeddings
    mlp_embedding_2d = reduce_embeddings(
        embeddings=mlp_embeddings,
    )

    gcn_embedding_2d = reduce_embeddings(
        embeddings=gcn_embeddings,
    )

    # data check
    print(f"\nDataset: {dataset}\n")
    print(f"Data: {data}\n")

    # model check
    print(f"MLP Model: {mlp}\n")
    print(f"GCN Model: {gcn}\n")

    # device check
    print(f"Selected MLP Device: {mlp_device.type}")
    print(f"Selected GCN Device: {gcn_device.type}\n")

    # inference check
    print(f"MLP Logits Shape: {mlp_logits.shape}")
    print(f"GCN Logits Shape: {gcn_logits.shape}\n")

    # formatted classification metrics check
    print_results(
        model_name="MLP",
        metrics=mlp_metrics,
    )

    print_results(
        model_name="GCN",
        metrics=gcn_metrics,
    )

    # confusion matrices
    print_confusion_matrix(
        model_name="MLP",
        matrix=mlp_confusion,
    )

    print_confusion_matrix(
        model_name="GCN",
        matrix=gcn_confusion,
    )

    # profiling
    print_profile("MLP", mlp_profile)
    print_profile("GCN", gcn_profile)

    # embedding data check
    print(f"MLP Embeddings: {mlp_embeddings.shape}")
    print(f"GCN Embeddings: {gcn_embeddings.shape}\n")

    # reduced embeddings check
    print(f"MLP UMAP: {mlp_embedding_2d.shape}")
    print(f"GCN UMAP: {gcn_embedding_2d.shape}")

    # create save location for the embedding plots
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # UMAP plotting check
    # Note: plot saving/over-writing is currently 'on'!
    mlp_figure = plot_embeddings(
        embedding_2d=mlp_embedding_2d,
        labels=data.y,
        model_name="MLP",
        output_path=IMAGES_DIR / "mlp_final_umap.png",
    )

    gcn_figure = plot_embeddings(
        embedding_2d=gcn_embedding_2d,
        labels=data.y,
        model_name="GCN",
        output_path=IMAGES_DIR / "gcn_final_umap.png",
    )

    # display UMAP plots
    plt.show()

    plt.close(mlp_figure)
    plt.close(gcn_figure)


if __name__ == "__main__":
    main()
