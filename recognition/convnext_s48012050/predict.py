"""
Shows example usage of the trained model

Modes:
    test   - Standard evaluation on the full test set.
    reject - Confidence-threshold demonstration similar to how the model would be applied in practice.
    both   - Run both analyses.

Examples:
    python predict.py --model convnext --mode test
    python predict.py --model convnext --mode reject --confidence-threshold 0.8
    python predict.py --model convnext --mode both --confidence-threshold 0.8
    python predict.py --model baseline --mode both --reject-mode flag


Arguments:
    --mode                  Evaluation mode: test, reject, or both (default: both)
    --model                 Model architecture used during training (convnext or baseline)
    --weights               Path to model weights; defaults to the best checkpoint for the selected model
    --data-dir              Directory containing the dataset (default: datasets)
    --output-dir            Directory to save predictions and visualizations (default: 
                            prediction_results)
    --reject-mode           How to handle low-confidence predictions: flag (send to human review) or 
                            mask(default low confidence preds to AD)
    --confidence-threshold  Confidence threshold for reject-mode evaluation (default: 0.8)
    --batch-size            Batch size for inference (default: 32)
    --num-workers           Number of worker processes for data loading (default: 0)
    --max-visualizations    Maximum number of visualized predictions to save (default: 16)
    --seed                  Random seed for reproducibility (default: 42)
    --pin-memory            Whether to pin memory for DataLoader (default: None, which uses True if CUDA 
                            is available)
    --val-fraction          Fraction of training data to use for validation (default: 0.2)

Outputs:
    predictions.csv          Per-image predictions and class probabilities
    prediction_summary.json  Parameter counts and evaluation metrics
    prediction_examples.png  Visual examples of model predictions
    confusion_matrix.png     Test-set confusion matrix
    roc_curve.png            ROC curve for AD as the positive class
    reject_predictions.csv   (Optional) Per-image predictions with confidence-based rejection


Outputs are saved in the selected output directory.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Subset

from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from dataset import create_dataloaders
from modules import (
    BaselineCNN,
    ConvNeXt,
    ConvNeXtBlock
)
from dataset import CLASS_TO_IDX


def get_class_names(dataset):
    """Return class names ordered by their model output indices."""
    if isinstance(dataset, Subset):
        return get_class_names(dataset.dataset)

    mapping = getattr(dataset, "class_to_idx", None)
    if mapping:
        names = [""] * len(mapping)
        for name, index in mapping.items():
            names[index] = name
        return names

    return ["AD", "NC"]


def get_sample_paths(dataset):
    """Recover image paths, including for a Subset dataset."""
    if isinstance(dataset, Subset):
        parent_paths = get_sample_paths(dataset.dataset)
        return [parent_paths[i] for i in dataset.indices]

    samples = getattr(dataset, "samples", None)
    if samples is not None:
        return [str(sample[0]) for sample in samples]

    for attribute in ("image_paths", "paths", "files"):
        paths = getattr(dataset, attribute, None)
        if paths is not None and len(paths) == len(dataset):
            return [str(path) for path in paths]

    return [f"image_{i:05d}" for i in range(len(dataset))]


def count_parameters(model):
    """Return total and trainable parameter counts."""
    return {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }


def load_weights(model, weights_path, device):
    """Load weights saved by train.py, or a state-dict checkpoint."""
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    checkpoint = torch.load(
        weights_path, map_location=device, weights_only=True
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    return model


def save_confusion_matrix(matrix, class_names, output_path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Number of images")

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="Actual class",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_roc_curve(labels, ad_probabilities, ad_index, output_path):
    """Save the binary ROC curve using AD as the positive class."""
    binary_labels = (labels == ad_index).astype(int)

    if len(set(binary_labels.tolist())) != 2:
        print("Warning: ROC curve/AUROC unavailable: test set has one class.")
        return None

    fpr, tpr, _ = roc_curve(binary_labels, ad_probabilities)
    roc_auc = roc_auc_score(binary_labels, ad_probabilities)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUROC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC curve (AD positive class)",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )

    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return float(roc_auc)


def save_prediction_examples(examples, class_names, output_path):
    """Save a grid of grayscale images and their predictions."""
    if not examples:
        return

    columns = min(4, len(examples))
    rows = (len(examples) + columns - 1) // columns

    fig, axes = plt.subplots(
        rows, columns, figsize=(4 * columns, 3.5 * rows), squeeze=False
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, item in zip(axes.flat, examples):
        image = item["image"].squeeze(0)
        ax.imshow(image, cmap="gray")
        ax.set_title(
            f"Actual: {class_names[item['actual']]}\n"
            f"Predicted: {class_names[item['predicted']]}\n"
            f"Confidence: {item['confidence']:.1%}",
            fontsize=9,
        )
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def evaluate_predictions(
    labels,
    predictions,
    probabilities,
    class_names,
    output_dir,
):
    """Calculate full-sweep metrics, confusion matrix and ROC curve."""
    labels = labels.astype(int)
    predictions = predictions.astype(int)

    num_classes = len(class_names)
    matrix = confusion_matrix(
        labels, predictions, labels=list(range(num_classes))
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(num_classes)),
        zero_division=0,
    )

    per_class = {}
    for index, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[index]), # type: ignore
            "recall": float(recall[index]), # type: ignore
            "f1": float(f1[index]), # type: ignore
            "support": int(support[index]), # type: ignore
        }
    ad_metrics = per_class["AD"]
    nc_metrics = per_class["NC"]


    ad_index = class_names.index("AD")
    ad_probabilities = probabilities[:, ad_index]

    roc_auc = save_roc_curve(
        labels,
        ad_probabilities,
        ad_index,
        output_dir / "roc_curve.png",
    )

    save_confusion_matrix(
        matrix,
        class_names,
        output_dir / "confusion_matrix.png",
        "Test-set confusion matrix",
    )

    return {
        "num_examples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_recall_fscore_support(
            labels, predictions, average="weighted", zero_division=0
        )[0]),
        "recall": float(precision_recall_fscore_support(
            labels, predictions, average="weighted", zero_division=0
        )[1]),
        "macro_precision": float(precision_recall_fscore_support(
            labels, predictions, average="macro", zero_division=0
        )[0]),
        "macro_recall": float(precision_recall_fscore_support(
            labels, predictions, average="macro", zero_division=0
        )[1]),
        "macro_f1": float(
            precision_recall_fscore_support(
                labels, predictions, average="macro", zero_division=0
            )[2]
        ),
        "per_class": per_class,
        "auroc_ad_positive": roc_auc,
        "confusion_matrix": matrix.tolist(),
        "class_names": class_names,
    }

def save_gradcam_examples(model, examples, device, output_path, target_layer):
    """Save Grad-CAM overlays for selected misclassified images."""
    if not examples:
        return

    columns = min(4, len(examples))
    rows = (len(examples) + columns - 1) // columns

    fig, axes = plt.subplots(
        rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False
    )

    for ax in axes.flat:
        ax.axis("off")

    activations = {}
    gradients = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        for ax, item in zip(axes.flat, examples):
            image = item["image"].unsqueeze(0).to(device)

            model.zero_grad(set_to_none=True)

            # Forward pass with gradients enabled.
            logits = model(image)

            # Explain the predicted class, including when the prediction is wrong.
            class_index = int(item["predicted"])
            logits[0, class_index].backward()

            activation = activations["value"]
            gradient = gradients["value"]

            if activation.ndim != 4 or gradient.ndim != 4:
                raise ValueError(
                    "Grad-CAM target layer must produce BCHW feature maps."
                )

            # Weight each feature map by its mean gradient.
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activation).sum(dim=1, keepdim=True).relu()

            # Resize the heatmap to the original image dimensions.
            cam = torch.nn.functional.interpolate(
                cam,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            cam = cam.detach().cpu()
            cam_max = cam.max()

            if cam_max > 0:
                cam = cam / cam_max

            original = item["image"].squeeze(0).detach().cpu().numpy()

            ax.imshow(original, cmap="gray")
            ax.imshow(
                cam.numpy(),
                cmap="jet",
                alpha=0.4,
                vmin=0,
                vmax=1,
            )

            ax.set_title(
                f"{item['filename']}\n"
                f"Actual: {item['actual_name']} | "
                f"Predicted: {item['predicted_name']}",
                fontsize=8,
            )
            ax.axis("off")

    finally:
        # Remove hooks so they do not affect subsequent model operations.
        forward_handle.remove()
        backward_handle.remove()
        model.zero_grad(set_to_none=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ADNI CNN/ConvNeXt predictions."
    )
    parser.add_argument(
        "--model",
        choices=["convnext", "baseline"],
        default="convnext",
    )
    parser.add_argument("--weights", default=None)
    parser.add_argument("--data-dir", default="datasets")
    parser.add_argument("--output-dir", default="prediction_results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["test", "reject", "both"],
        default="both",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--reject-mode",
        choices=["flag", "mask"],
        default="mask",
        help="flag sends low-confidence cases to human review (-1); "
             "mask defaults them to AD.",
    )
    parser.add_argument("--max-visualizations", type=int, default=16)
    args = parser.parse_args()

    if not 0.0 <= args.confidence_threshold <= 1.0:
        parser.error("--confidence-threshold must be between 0 and 1")
    if args.max_visualizations < 0:
        parser.error("--max-visualizations must be non-negative")

    model_classes = {
        "convnext": ConvNeXt,
        "baseline": BaselineCNN,
    }
    model_class = model_classes[args.model]
    model = model_class(num_classes=2)

    weights_path = (
        Path(args.weights)
        if args.weights
        else Path("models") / f"{model_class.__name__}_best.pt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_weights(model, weights_path, device).to(device)
    model.eval()

    loaders = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    test_loader = loaders.get("test")
    if test_loader is None:
        raise ValueError("No test loader returned by create_dataloaders().")

    class_names = get_class_names(test_loader.dataset)
    if len(class_names) != 2 or set(class_names) != {"AD", "NC"}:
        raise ValueError(
            f"Expected classes AD and NC; received {class_names}"
        )

    # The module-level reject function uses CLASS_TO_IDX["AD"].
    # Verify that its class indices agree with the dataset.

    if any(CLASS_TO_IDX[name] != i for i, name in enumerate(class_names)):
        raise ValueError(
            "modules.CLASS_TO_IDX does not match the dataset class mapping: "
            f"{CLASS_TO_IDX} vs {class_names}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = get_sample_paths(test_loader.dataset)
    all_labels = []
    all_predictions = []
    all_probabilities = []
    records = []
    reject_records = []
    examples = []
    incorrect_examples = []
    patient_results = {}

    total = 0
    inference_start = time.perf_counter()

    # One forward pass per batch is shared by the full and reject analyses.
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            if logits.ndim != 2 or logits.shape[1] != len(class_names):
                raise ValueError(
                    f"Expected (batch, 2) logits; got {tuple(logits.shape)}"
                )

            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)

            # Use the supplied modules.py method for the reject-option result.
            # This currently performs its own forward pass, so it is only
            # needed when the reject analysis has been requested.
            if args.mode in ("reject", "both"):
                rejected_predictions, rejected_confidences = model.predict(
                    images,
                    confidence_threshold=args.confidence_threshold,
                    mode=args.reject_mode,
                )
                rejected_predictions = rejected_predictions.cpu().tolist()
                rejected_confidences = rejected_confidences.cpu().tolist()

            labels_cpu = labels.cpu().tolist()
            predictions_cpu = predictions.cpu().tolist()
            probabilities_cpu = probabilities.cpu().tolist()
            confidences_cpu = confidences.cpu().tolist()

            for i, (actual, predicted, probs, confidence) in enumerate(
                zip(
                    labels_cpu,
                    predictions_cpu,
                    probabilities_cpu,
                    confidences_cpu,
                )
            ):
                image_path = paths[total]
                all_labels.append(actual)
                all_predictions.append(predicted)
                all_probabilities.append(probs)

                record = {
                    "image": image_path,
                    "actual_class": class_names[actual],
                    "predicted_class": class_names[predicted],
                    "confidence": confidence,
                    "correct": actual == predicted,
                }
                for class_index, name in enumerate(class_names):
                    record[f"probability_{name}"] = probs[class_index]
                records.append(record)

                if args.mode in ("reject", "both"):
                    final_prediction = int(rejected_predictions[i])
                    is_rejected = final_prediction == -1
                    reject_records.append({
                        "image": image_path,
                        "actual_class": class_names[actual],
                        "original_predicted_class": class_names[predicted],
                        "confidence": float(rejected_confidences[i]),
                        "confidence_threshold": args.confidence_threshold,
                        "low_confidence": (
                            float(rejected_confidences[i])
                            < args.confidence_threshold
                        ),
                        "final_decision": (
                            "HUMAN_REVIEW"
                            if is_rejected
                            else class_names[final_prediction]
                        ),
                        "final_class_index": final_prediction,
                        "correct_if_accepted": (
                            actual == final_prediction if not is_rejected else ""
                        ),
                    })

                filename = Path(image_path).name
                stem = Path(filename).stem
                # Slice filenames end in _<slice number>; retain the patient prefix.
                patient_id = stem.rsplit("_", 1)[0] if "_" in stem else stem
                patient = patient_results.setdefault(patient_id, {
                    "num_images": 0, "num_correct": 0, "num_errors": 0, "errors": []
                })
                patient["num_images"] += 1
                if actual == predicted:
                    patient["num_correct"] += 1
                else:
                    patient["num_errors"] += 1
                    patient["errors"].append({
                        "image": image_path,
                        "actual_class": class_names[actual],
                        "predicted_class": class_names[predicted],
                        "confidence": float(confidence),
                    })
                    if len(incorrect_examples) < args.max_visualizations:
                        incorrect_examples.append({
                            "image": images[i].detach().cpu(),
                            "actual": actual, "predicted": predicted,
                            "confidence": confidence, "filename": filename,
                            "actual_name": class_names[actual],
                            "predicted_name": class_names[predicted],
                        })
                total += 1

    elapsed = time.perf_counter() - inference_start
    if total == 0:
        raise ValueError("Test DataLoader yielded no examples.")

    # The full sweep is always computed so that both modes use the same
    # underlying test predictions.
    import numpy as np

    labels_array = np.asarray(all_labels, dtype=int)
    predictions_array = np.asarray(all_predictions, dtype=int)
    probabilities_array = np.asarray(all_probabilities, dtype=float)

    summary = {
        "model_name": model_class.__name__,
        "weights_path": str(weights_path),
        "device": str(device),
        "parameter_counts": count_parameters(model),
        "inference_seconds": elapsed,
        "images_per_second": total / elapsed if elapsed > 0 else 0.0,
    }

    if args.mode in ("test", "both"):
        full_metrics = evaluate_predictions(
            labels_array,
            predictions_array,
            probabilities_array,
            class_names,
            output_dir,
        )
        summary["full_evaluation"] = full_metrics
        summary["per_patient_test_results"] = {
            patient_id: {
                **values,
                "accuracy": values["num_correct"] / values["num_images"],
                "error_rate": values["num_errors"] / values["num_images"],
            }
            for patient_id, values in sorted(patient_results.items())
        }

        with (output_dir / "predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

        save_prediction_examples(
            incorrect_examples,
            class_names,
            output_dir / "incorrect_prediction_examples.png",
        )

        target_layer = next(
            (layer for layer in reversed(list(model.modules()))
            if isinstance(layer, ConvNeXtBlock)),
            None
        )
        if target_layer is not None and incorrect_examples:
            save_gradcam_examples(
                model, incorrect_examples, device,
                output_dir / "gradcam_incorrect_examples.png", target_layer,
            )
        elif incorrect_examples:
            print("Warning: no Conv2d layer found; Grad-CAM was not generated.")

        print("\nFull test-set evaluation")
        print(f"Accuracy: {full_metrics['accuracy']:.2%}")
        print(f"Precision (weighted): {full_metrics['precision']:.4f}")
        print(f"Recall (weighted): {full_metrics['recall']:.4f}")
        print(f"Macro precision: {full_metrics['macro_precision']:.4f}")
        print(f"Macro recall: {full_metrics['macro_recall']:.4f}")
        print(f"Macro-F1: {full_metrics['macro_f1']:.4f}")
        print(f"AUROC (AD positive): {full_metrics['auroc_ad_positive']}")
        for name, metrics in full_metrics["per_class"].items():
            print(
                f"{name}: precision={metrics['precision']:.4f}, "
                f"recall={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"support={metrics['support']}"
            )

    if args.mode in ("reject", "both"):
        reject_csv = output_dir / "reject_predictions.csv"
        with reject_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=list(reject_records[0].keys())
            )
            writer.writeheader()
            writer.writerows(reject_records)

        actual = np.asarray(all_labels, dtype=int)
        decisions = np.asarray(
            [row["final_class_index"] for row in reject_records], dtype=int
        )
        accepted = decisions != -1
        accepted_count = int(accepted.sum())
        rejected_count = int((~accepted).sum())

        accepted_accuracy = (
            float((actual[accepted] == decisions[accepted]).mean())
            if accepted_count else None
        )

        reject_summary = {
            "confidence_threshold": args.confidence_threshold,
            "reject_mode": args.reject_mode,
            "num_examples": total,
            "accepted_predictions": accepted_count,
            "rejected_for_human_review": rejected_count,
            "coverage": accepted_count / total,
            "rejection_rate": rejected_count / total,
            "accuracy_on_accepted": accepted_accuracy,
        }
        summary["reject_option"] = reject_summary

        print("\nReject-option evaluation")
        print(f"Threshold: {args.confidence_threshold:.3f}")
        print(f"Mode: {args.reject_mode}")
        print(f"Accepted: {accepted_count}/{total}")
        print(f"Human review: {rejected_count}/{total}")
        print(f"Coverage: {reject_summary['coverage']:.2%}")
        print(f"Rejection rate: {reject_summary['rejection_rate']:.2%}")
        print(f"Accuracy on accepted: {accepted_accuracy}")

    with (output_dir / "prediction_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, allow_nan=False)

    print(f"\nTotal parameters: {summary['parameter_counts']['total_parameters']:,}")
    print(f"Trainable parameters: {summary['parameter_counts']['trainable_parameters']:,}")
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
