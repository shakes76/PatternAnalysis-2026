
"""
COMP3710 - ADNI Alzheimer's Disease Classification
Final Test Evaluation

Models:
    - baseline
    - convnext

Evaluates the selected best.pth checkpoint on the held-out
test dataset. No training or weight updates are performed.

Outputs:
    metrics.json
    classification_report.txt
    predictions.csv
    confusion_matrix.png
    roc_curve.png
    misclassified_examples.png
"""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from dataset import DATA_DIR, SPLITS_CSV, get_dataloaders
from modules import build_model


CLASS_NAMES = ["NC", "AD"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def save_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title("Confusion Matrix - Test Set")
    fig.tight_layout()

    fig.savefig(
        output_dir / "confusion_matrix.png",
        dpi=200,
    )

    plt.close(fig)

    return cm


def save_roc_curve(y_true, probabilities, output_dir):
    fpr, tpr, _ = roc_curve(
        y_true, probabilities
    )

    auc = roc_auc_score(y_true, probabilities)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        fpr, tpr,
        label=f"ROC curve (AUC = {auc:.4f})",
    )

    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - AD vs NC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        output_dir / "roc_curve.png",
        dpi=200,
    )

    plt.close(fig)

    return float(auc)


def denormalize_image(image_tensor):
    image = image_tensor.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))

    image = image * IMAGENET_STD + IMAGENET_MEAN

    return np.clip(image, 0, 1)


def save_misclassified_examples(examples, output_dir):
    if not examples:
        print("No misclassified examples available.")
        return

    n = len(examples)

    fig, axes = plt.subplots(
        1, n,
        figsize=(3.5 * n, 4),
        squeeze=False,
    )

    for ax, example in zip(axes[0], examples):
        image = denormalize_image(example["image"])

        ax.imshow(image)
        ax.axis("off")

        true_name = CLASS_NAMES[example["true"]]
        pred_name = CLASS_NAMES[example["pred"]]

        ax.set_title(
            f"True: {true_name}\n"
            f"Pred: {pred_name}\n"
            f"Confidence: {example['confidence']:.2%}"
        )

    fig.suptitle("Misclassified MRI Slices")
    fig.tight_layout()

    fig.savefig(
        output_dir / "misclassified_examples.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def evaluate(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []
    probabilities_ad = []

    examples = []
    elapsed = 0.0
    total_images = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(
                device,
                non_blocking=device.type == "cuda",
            )

            synchronize_device(device)
            start = time.perf_counter()

            logits = model(images)

            synchronize_device(device)
            elapsed += time.perf_counter() - start

            probs = torch.softmax(logits, dim=1)
            predictions = probs.argmax(dim=1)

            probs_cpu = probs.cpu()
            predictions_cpu = predictions.cpu()
            labels_cpu = labels.cpu()

            y_true.extend(labels_cpu.tolist())
            y_pred.extend(predictions_cpu.tolist())
            probabilities_ad.extend(
                probs_cpu[:, 1].tolist()
            )

            total_images += labels_cpu.size(0)

            # Save up to five examples of incorrect predictions.
            for i in range(labels_cpu.size(0)):
                true_label = int(labels_cpu[i])
                pred_label = int(predictions_cpu[i])

                if (
                    true_label != pred_label
                    and len(examples) < 5
                ):
                    examples.append({
                        "image": images[i].detach().cpu(),
                        "true": true_label,
                        "pred": pred_label,
                        "confidence": float(
                            probs_cpu[i, pred_label]
                        ),
                    })

    if total_images == 0:
        raise RuntimeError("Test DataLoader is empty.")

    return (
        np.asarray(y_true),
        np.asarray(y_pred),
        np.asarray(probabilities_ad),
        examples,
        elapsed,
        total_images,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Final ADNI test evaluation"
    )

    parser.add_argument(
        "--model",
        choices=["baseline", "convnext"],
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--csv-path",
        type=Path,
        default=SPLITS_CSV,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
    )

    parser.add_argument(
        "--require-cuda",
        action="store_true",
    )

    args = parser.parse_args()

    if args.batch_size < 1 or args.num_workers < 0:
        parser.error(
            "Batch size must be positive and "
            "num-workers must be nonnegative."
        )

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.require_cuda and device.type != "cuda":
        raise RuntimeError(
            "CUDA unavailable. Run within a GPU Slurm job."
        )

    print(f"Device: {device}", flush=True)

    if device.type == "cuda":
        print(
            "GPU:", torch.cuda.get_device_name(0),
            flush=True,
        )
        torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    # Use only checkpoints produced by your own training.
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("model_name") != args.model:
        raise ValueError(
            "Checkpoint model_name does not match --model"
        )

    expected_mapping = {"NC": 0, "AD": 1}

    if checkpoint.get("class_mapping") != expected_mapping:
        raise ValueError(
            "Unexpected checkpoint class mapping."
        )

    if checkpoint.get("image_size") != 224:
        raise ValueError(
            "Expected checkpoint image_size=224."
        )

    # No ImageNet download is needed here:
    # checkpoint weights replace model initialization.
    model = build_model(
        model_name=args.model,
        num_classes=2,
        pretrained=False,
        freeze_backbone=False,
        partial_finetune=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print(
        f"Model: {args.model}\n"
        f"Checkpoint epoch: {checkpoint['epoch']}\n"
        f"Total parameters: {total_params:,}",
        flush=True,
    )

    # --------------------------------------------------------
    # Test DataLoader
    # --------------------------------------------------------

    loaders = get_dataloaders(
        csv_path=args.csv_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=224,
    )

    test_loader = loaders["test"]

    print(
        f"Test images: {len(test_loader.dataset)}",
        flush=True,
    )

    print(
        "Evaluating selected checkpoint on "
        "the held-out test split...",
        flush=True,
    )

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    (
        y_true,
        y_pred,
        prob_ad,
        examples,
        inference_seconds,
        n_images,
    ) = evaluate(
        model=model,
        loader=test_loader,
        device=device,
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0,
        )
    )

    macro_f1 = float(np.mean(f1))

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    cm = save_confusion_matrix(
        y_true, y_pred, args.output_dir
    )

    # AUROC requires both classes in the test set.
    if len(np.unique(y_true)) == 2:
        auc = save_roc_curve(
            y_true, prob_ad, args.output_dir
        )
    else:
        auc = None
        print(
            "ROC-AUC unavailable: test data contains "
            "only one class."
        )

    save_misclassified_examples(
        examples, args.output_dir
    )

    metrics = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test_images": n_images,
        "accuracy": float(accuracy),
        "macro_f1": macro_f1,
        "roc_auc": auc,
        "NC": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "AD": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
        "confusion_matrix": cm.tolist(),
        "total_parameters": total_params,
        "inference_seconds": inference_seconds,
        "ms_per_image": (
            1000 * inference_seconds / n_images
        ),
        "images_per_second": (
            n_images / inference_seconds
        ),
    }

    if device.type == "cuda":
        metrics["peak_gpu_memory_gib"] = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    with (
        args.output_dir / "metrics.json"
    ).open("w") as f:
        json.dump(metrics, f, indent=2)

    with (
        args.output_dir / "classification_report.txt"
    ).open("w") as f:
        f.write(report)

    with (
        args.output_dir / "predictions.csv"
    ).open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "test_index",
            "true_label",
            "predicted_label",
            "probability_NC",
            "probability_AD",
            "confidence",
            "correct",
        ])

        for i, (true, pred, p_ad) in enumerate(
            zip(y_true, y_pred, prob_ad)
        ):
            p_ad = float(p_ad)
            p_nc = 1.0 - p_ad

            writer.writerow([
                i,
                int(true),
                int(pred),
                p_nc,
                p_ad,
                max(p_nc, p_ad),
                int(true == pred),
            ])

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 50)
    print(f"FINAL TEST RESULTS - {args.model.upper()}")
    print("=" * 50)

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Macro-F1: {macro_f1:.4f}")

    if auc is not None:
        print(f"ROC-AUC: {auc:.4f}")

    print("\nClassification report:")
    print(report)

    print("Confusion matrix:")
    print(cm)

    print(
        f"\nInference time: {inference_seconds:.2f}s"
    )
    print(
        f"Average latency: "
        f"{metrics['ms_per_image']:.3f} ms/image"
    )

    if device.type == "cuda":
        print(
            f"Peak GPU allocated memory: "
            f"{metrics['peak_gpu_memory_gib']:.2f} GiB"
        )

    print(
        f"\nResults saved to: "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()

