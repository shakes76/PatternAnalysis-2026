"""ROUGE evaluation, dataset overlap auditing, and result visualisation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rouge_score import rouge_scorer

from dataset import load_radiology_dataset

def normalize_report(report):
    """Normalize case and whitespace for overlap comparison."""
    return " ".join(str(report).lower().split())

def plot_dataset_overlap(seen_count, unseen_count, output_dir):
    """
    Visualize the proportion of seen and unseen validation reports.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = ["Seen", "Unseen"]
    counts = [seen_count, unseen_count]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(labels, counts)

    ax.set_title("Validation Dataset Overlap Distribution")
    ax.set_ylabel("Number of Samples")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 80,
            f"{count:,}",
            ha="center",
        )

    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    plt.tight_layout()

    figure_path = output_dir / "dataset_overlap.png"

    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Dataset overlap figure saved to: {figure_path}")

def audit_dataset_overlap():
    """
    Audit exact report overlap between training and validation splits.
    """
    dataset = load_radiology_dataset()

    train_reports = dataset["train"]["radiology_report"]
    val_reports = dataset["validation"]["radiology_report"]

    unique_train_reports = {
        normalize_report(report)
        for report in train_reports
    }

    seen_count = sum(
        normalize_report(report) in unique_train_reports
        for report in val_reports
    )

    unseen_count = len(val_reports) - seen_count

    print("\n=== Dataset Overlap Audit ===")
    print(f"Training samples: {len(train_reports):,}")
    print(f"Unique training reports: {len(unique_train_reports):,}")
    print(f"Validation samples: {len(val_reports):,}")
    print(f"Seen validation samples: {seen_count:,}")
    print(f"Unseen validation samples: {unseen_count:,}")
    print(f"Overlap percentage: {seen_count / len(val_reports) * 100:.2f}%")

    plot_dataset_overlap(
        seen_count=seen_count,
        unseen_count=unseen_count,
        output_dir=Path("figures"),
    )

def calculate_rouge(predictions):
    """
    Calculate average ROUGE F1 scores across evaluation samples.

    Args:
        predictions (list): Generated predictions and references.

    Returns:
        dict: Average ROUGE scores.
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer=True,
    )

    scores = {
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "rougeLsum": [],
    }

    for prediction in predictions:
        reference = prediction["reference_summary"]
        generated = prediction["generated_summary"]

        result = scorer.score(reference, generated)

        for metric in scores:
            scores[metric].append(result[metric].fmeasure)

    return {
        metric: float(np.mean(values))
        for metric, values in scores.items()
    }

def plot_rouge_scores(rouge_scores, output_dir, method):
    """
    Generate a bar chart of ROUGE F1 scores.

    Args:
        rouge_scores (dict): ROUGE evaluation results.
        output_dir (Path): Directory for evaluation outputs.
        method (str): Fine-tuning method.
    """
    metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "ROUGE-Lsum"]

    values = [rouge_scores[metric] for metric in metrics]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(labels, values)

    ax.set_title(f"{method.upper()} — ROUGE Evaluation")
    ax.set_ylabel("ROUGE F1 Score")
    ax.set_ylim(0, 1)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )

    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    plt.tight_layout()

    figure_path = output_dir / "rouge_scores.png"

    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"ROUGE figure saved to: {figure_path}")

def get_training_reports():
    """
    Build a set of radiology reports from the training split.

    Exact string matching is used to identify overlapping reports.
    """
    dataset = load_radiology_dataset()
    training_reports = {
        normalize_report(report)
        for report in dataset["train"]["radiology_report"]
    }

    print("\n=== Training Report Index ===")
    print(f"Unique training reports: {len(training_reports):,}")

    return training_reports

def calculate_grouped_rouge(predictions):
    """
    Calculate ROUGE scores separately for seen and unseen reports.
    """
    results = {}

    for group in ["seen", "unseen"]:
        group_predictions = [
            prediction
            for prediction in predictions
            if prediction["split_group"] == group
        ]

        results[group] = {
            "num_samples": len(group_predictions),
            "scores": (
                calculate_rouge(group_predictions)
                if group_predictions
                else None
            ),
        }

    return results

def plot_seen_unseen_rouge(grouped_rouge, output_dir):
    """
    Compare ROUGE scores between seen and unseen validation reports.
    """
    import numpy as np

    metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "ROUGE-Lsum"]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    for offset, group in [(-width / 2, "seen"), (width / 2, "unseen")]:
        result = grouped_rouge[group]

        if result["scores"] is None:
            continue

        values = [result["scores"][metric] for metric in metrics]

        ax.bar(
            x + offset,
            values,
            width,
            label=f"{group.capitalize()} (n={result['num_samples']})",
        )

    ax.set_title("Seen vs Unseen Validation ROUGE")
    ax.set_ylabel("ROUGE F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    plt.tight_layout()

    figure_path = output_dir / "seen_unseen_rouge.png"

    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Seen/unseen ROUGE figure saved to: {figure_path}")
