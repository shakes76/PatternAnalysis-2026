"""
Evaluation of the trained ADNI classifier.

Usage:
    python3 predict.py --model convnext                    # full evaluation
    python3 predict.py --model convnext --image scan.jpeg  # one image
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from dataset import get_dataloaders, get_transforms
from train import get_device, build_model

CLASS_NAMES = ["NC", "AD"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained ADNI model")
    parser.add_argument("--model", default="convnext", choices=["baseline", "convnext"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--data_root", default=os.path.expanduser("~/comp3710_data/AD_NC")
    )
    parser.add_argument("--image", default=None)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--fig_dir", default="images")
    parser.add_argument("--out_dir", default="outputs")
    return parser.parse_args()


def load_model(name, checkpoint, device):
    """Build the model and load the trained weights."""
    model = build_model(name)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_logits(model, loader, device):
    """
    Run the model over a loader and collect its logits for the temperature scaling
    Returns (logits [N, 2], labels [N])
    """
    all_logits, all_labels = [], []
    for images, labels in loader:
        all_logits.append(model(images.to(device)).cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def compute_metrics(probs, labels):
    """Computes the Classification metrics from the AD probs"""
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_AD": precision_score(labels, preds),
        "recall_AD": recall_score(labels, preds),
        "recall_NC": recall_score(labels, preds, pos_label=0),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "auroc": roc_auc_score(labels, probs),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


def plot_confusion_and_roc(probs, labels, save_path, title):
    """Produces confusion matrix and ROC curves side by side."""
    preds = (probs >= 0.5).astype(int)
    false_pos_rate, true_pos_rate, _ = roc_curve(labels, probs)
    auroc = roc_auc_score(labels, probs)
    figure, (confusion_panel, roc_panel) = plt.subplots(1, 2, figsize=(10, 4))

    ConfusionMatrixDisplay(
        confusion_matrix(labels, preds), display_labels=CLASS_NAMES
    ).plot(ax=confusion_panel, colorbar=False)
    confusion_panel.set_title("Confusion matrix")

    roc_panel.plot(false_pos_rate, true_pos_rate, label=f"AUROC {auroc:.3f}")
    roc_panel.plot([0, 1], [0, 1], "k--")
    roc_panel.set_xlabel("False positive rate")
    roc_panel.set_ylabel("True positive rate")
    roc_panel.set_title("ROC curve")
    roc_panel.legend()

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(save_path)
    plt.close(figure)


def wait_for_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def measure_resources(model, device, n_runs=100):
    """Measures the number of parameters, latency per slices in ms, and the peak VRAM in GB"""

    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 1, 240, 256, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(10):  # warm up gpu resources
            model(x)
        wait_for_device(device)
        start = time.time()
        for _ in range(n_runs):
            model(x)
        wait_for_device(device)
        latency_ms = (time.time() - start) / n_runs * 1000

    peak_vram_gb = 0.0
    if device.type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

    return {"params": n_params, "latency_ms": latency_ms, "peak_vram_gb": peak_vram_gb}


def fit_temperature(logits, labels):
    """
    Temperature scaling finds a number T that minimises the cross-entropy
    of softmax(logits / T) on the validation set.
    """
    best_t, best_loss = 1.0, float("inf")
    for t in torch.arange(0.5, 5.01, 0.01):
        loss = torch.nn.functional.cross_entropy(logits / t, labels).item()
        if loss < best_loss:
            best_t, best_loss = t.item(), loss
    return best_t


def expected_calibration_error(confidences, correct, n_bins=10):
    """
    Computes the average gap between confidence and accuracy which is weighted by how many
    predictions fall in each confidence interval
    """

    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > low) & (confidences <= high)
        if in_bin.sum() > 0:
            gap = abs(correct[in_bin].mean() - confidences[in_bin].mean())
            ece += in_bin.mean() * gap
    return ece


def plot_reliability(conf_before, conf_after, correct, path, title):
    """Produces reliability diagrams before and after temperature scaling."""
    edges = np.linspace(0.5, 1.0, 11)
    centres = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    panels = [
        (axes[0], conf_before, "Before temperature scaling"),
        (axes[1], conf_after, "After temperature scaling"),
    ]
    for ax, conf, name in panels:
        accs = []
        for low, high in zip(edges[:-1], edges[1:]):
            in_bin = (conf > low) & (conf <= high)
            accs.append(correct[in_bin].mean() if in_bin.any() else 0.0)
        ece = expected_calibration_error(conf, correct)

        ax.bar(centres, accs, width=0.05, edgecolor="black", label="Accuracy")
        ax.plot([0.5, 1], [0.5, 1], "k--", label="Perfect calibration")
        ax.set_xlim(0.5, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{name} (ECE {ece:.3f})")
        ax.legend(loc="upper left")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def reject_option(confidences, correct, threshold):
    """Report predications >= to the threshold, otherwise advice referral"""
    accepted = confidences >= threshold
    return {
        "threshold": threshold,
        "referral_rate": 1 - accepted.mean(),
        "accepted_accuracy": correct[accepted].mean() if accepted.any() else 0.0,
    }


def plot_referral_tradeoff(confidences, correct, chosen, path, title):
    """Accepted vs referred scans at different thresholds"""
    thresholds = np.arange(0.5, 1.0, 0.01)
    referred, accuracy = [], []
    for t in thresholds:
        result = reject_option(confidences, correct, t)
        if result["referral_rate"] < 1:  # skip thresholds that accept nothing
            referred.append(result["referral_rate"])
            accuracy.append(result["accepted_accuracy"])

    chosen_result = reject_option(confidences, correct, chosen)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(referred, accuracy, marker=".")
    ax.scatter(
        chosen_result["referral_rate"],
        chosen_result["accepted_accuracy"],
        color="red",
        zorder=3,
        label=f"Threshold {chosen:.2f}",
    )
    ax.axvline(0.3, color="grey", linestyle="--", label="30% referral limit")
    ax.set_xlabel("Fraction of slices referred to a clinician")
    ax.set_ylabel("Accuracy of accepted slices")
    ax.set_title(title)
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_failures(samples, probs, labels, path, title, n=5):
    """Show the most confident slies that are wrong (at most n), 1 per scan"""
    preds = (probs >= 0.5).astype(int)
    confidences = np.maximum(probs, 1 - probs)
    wrong = np.where(preds != labels)[0]

    worst, seen_scans = [], set()
    for idx in wrong[np.argsort(-confidences[wrong])]:
        scan = os.path.basename(samples[idx][0]).split("_")[0]
        if scan not in seen_scans:
            seen_scans.add(scan)
            worst.append(idx)
        if len(worst) == n:
            break

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    for ax, idx in zip(np.atleast_1d(axes), worst):
        path_i = samples[idx][0]
        ax.imshow(Image.open(path_i).convert("L"), cmap="gray")
        ax.set_title(
            f"{os.path.basename(path_i)}\n"
            f"true {CLASS_NAMES[labels[idx]]}, pred {CLASS_NAMES[preds[idx]]}\n"
            f"confidence {confidences[idx]:.2f}",
            fontsize=9,
        )
        ax.axis("off")

    fig.suptitle(title, y=1.05)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return [samples[i][0] for i in worst]


def predict_image(model, image_path, device, temperature, threshold):
    """Predict one slice and decide whether to accept or refer it."""
    image = Image.open(image_path).convert("L")
    x = get_transforms(train=False)(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x) / temperature, dim=1)[0].cpu()

    label = CLASS_NAMES[probs.argmax().item()]
    confidence = probs.max().item()
    decision = "ACCEPT" if confidence >= threshold else "REFER to clinician"
    print(f"{image_path}: {label} (confidence {confidence:.3f}) -> {decision}")


if __name__ == "__main__":
    args = parse_args()
    device = get_device()
    checkpoint = args.checkpoint or os.path.join(args.out_dir, f"{args.model}_best.pt")
    model = load_model(args.model, checkpoint, device)
    eval_path = os.path.join(args.out_dir, f"{args.model}_eval.json")

    if args.image is not None:
        temperature = 1.0
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                temperature = json.load(f)["temperature"]
        predict_image(model, args.image, device, temperature, args.threshold)
        raise SystemExit

    os.makedirs(args.fig_dir, exist_ok=True)
    _, val_loader, test_loader = get_dataloaders(
        args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Logits for validation for fitting T and testing
    val_logits, val_labels = collect_logits(model, val_loader, device)
    test_logits, test_labels = collect_logits(model, test_loader, device)
    temperature = fit_temperature(val_logits, val_labels)

    labels = test_labels.numpy()
    probs_before = torch.softmax(test_logits, dim=1).numpy()
    probs_after = torch.softmax(test_logits / temperature, dim=1).numpy()
    conf_before = probs_before.max(axis=1)
    conf_after = probs_after.max(axis=1)
    correct = (probs_before.argmax(axis=1) == labels).astype(float)

    metrics = compute_metrics(probs_before[:, 1], labels)
    resources = measure_resources(model, device)
    reject = reject_option(conf_after, correct, args.threshold)

    # ECE on val and test, confident mistakes
    # Added here which I forgot to add earlier.
    val_probs = torch.softmax(val_logits, dim=1).numpy()
    val_correct = (val_probs.argmax(axis=1) == val_labels.numpy()).astype(float)
    confident_wrong = (correct == 0) & (conf_after >= args.threshold)
    calibration = {
        "val_ece": expected_calibration_error(val_probs.max(axis=1), val_correct),
        "test_ece_before": expected_calibration_error(conf_before, correct),
        "test_ece_after": expected_calibration_error(conf_after, correct),
        "wrong_slices": int((correct == 0).sum()),
        "confident_wrong_slices": int(confident_wrong.sum()),
        "confident_wrong_AD_predicted_NC": int((confident_wrong & (labels == 1)).sum()),
    }

    # Reject options at multiple thresholds.
    reject_table = []
    for t in (0.8, 0.9, 0.95):
        reject_table.append(reject_option(conf_after, correct, t))
    threshold_30 = float(np.quantile(conf_after, 0.30))
    reject_table.append(reject_option(conf_after, correct, threshold_30))

    results = {
        "model": args.model,
        "temperature": temperature,
        "metrics": metrics,
        "resources": resources,
        "calibration": calibration,
        "reject_option": reject,
        "reject_table": reject_table,
    }

    # Figures for the README
    name = args.model
    plot_confusion_and_roc(
        probs_before[:, 1],
        labels,
        os.path.join(args.fig_dir, f"{name}_confusion_roc.png"),
        f"{name} test set",
    )
    plot_reliability(
        conf_before,
        conf_after,
        correct,
        os.path.join(args.fig_dir, f"{name}_reliability.png"),
        f"{name} calibration",
    )
    plot_referral_tradeoff(
        conf_after,
        correct,
        args.threshold,
        os.path.join(args.fig_dir, f"{name}_referral.png"),
        f"{name} reject option",
    )
    results["failure_cases"] = plot_failures(
        test_loader.dataset.samples,
        probs_before[:, 1],
        labels,
        os.path.join(args.fig_dir, f"{name}_failures.png"),
        f"{name}: most confident mistakes",
    )

    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
