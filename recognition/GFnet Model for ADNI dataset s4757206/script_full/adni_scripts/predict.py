"""Evaluate a saved CNN or GFNet checkpoint on ADNI test slices."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from dataset import (
    MRIDataset,
    build_transforms,
    load_split_csvs,
    resolve_image_paths,
)
from modules import CNNBaseline, GFNet


def parse_args():
    """Read evaluation paths; extended analyses run unless explicitly skipped."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--model", choices=["cnn", "gfnet"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )

    parser.add_argument('--skip-analysis', action='store_true',
                        help='Only compute core test metrics and plots.')
    parser.add_argument('--benchmark', action='store_true',
                        help='Also run the notebook CUDA inference benchmark.')
    return parser.parse_args()


def evaluate(model, loader, device):
    """Collect labels, predictions, probabilities, and average loss."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_confidences = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if not torch.isfinite(outputs).all() or not torch.isfinite(loss):
                raise RuntimeError("Non-finite logits or loss during evaluation.")
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predictions = probabilities.max(dim=1)

            total_loss += loss.item() * labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(
                probabilities[:, 1].cpu().tolist()
            )
            all_confidences.extend(confidence.cpu().tolist())

    if not all_labels:
        raise ValueError("The test dataset is empty.")

    return {
        "loss": total_loss / len(all_labels),
        "labels": np.asarray(all_labels, dtype=int),
        "predictions": np.asarray(all_predictions, dtype=int),
        "ad_probabilities": np.asarray(all_probabilities, dtype=float),
        "confidence": np.asarray(all_confidences, dtype=float),
    }


def save_plots(labels, predictions, ad_probabilities, model_name, output_dir):
    """Save a confusion matrix and ROC curve."""
    cm = confusion_matrix(labels, predictions, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["NC", "AD"],
    ).plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )
    ax.set_title(f"{model_name} Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(
        output_dir / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    fpr, tpr, _ = roc_curve(labels, ad_probabilities)
    auroc = roc_auc_score(labels, ad_probabilities)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{model_name} (AUROC = {auroc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} Test ROC Curve")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(
        output_dir / "roc_curve.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    """Load one checkpoint and write a reproducible evaluation directory."""
    args = parse_args()

    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid batch size or worker count.")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} is not empty. Choose a new output directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Audit all split memberships; evaluation uses validation/test images only.
    _, val_df, test_df = load_split_csvs(args.split_dir)

    test_df = resolve_image_paths(
        test_df,
        source_root=args.source_root,
        image_root=args.image_root,
    )

    _, evaluation_transform = build_transforms(args.model)

    test_dataset = MRIDataset(
        test_df,
        transform=evaluation_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    if args.model == "cnn":
        model = CNNBaseline()
        model_name = "CNN"
    else:
        model = GFNet(
            image_size=256,
            patch_size=16,
            dim=128,
            depth=8,
            dropout=0.1,
        )
        model_name = "GFNet"

    # Load the plain state_dict saved by the notebooks or train.py.
    state_dict = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    if args.benchmark and device.type != 'cuda':
        raise ValueError('--benchmark requires CUDA.')
    with (output_dir / 'environment.json').open('w') as file:
        json.dump(environment_info(device), file, indent=2)
    with (output_dir / 'evaluation_config.json').open('w') as file:
        json.dump(vars(args), file, indent=2)

    # Validation is evaluated independently; test predictions cannot influence
    # threshold selection. Training images are not loaded for evaluation.
    validation_results = None
    if not args.skip_analysis:
        val_df = resolve_image_paths(val_df, args.source_root, args.image_root)
        val_dataset = MRIDataset(val_df, evaluation_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=(device.type == 'cuda'))
        validation_results = evaluate(model, val_loader, device)
        if not np.array_equal(validation_results['labels'], val_df.label.map({'NC': 0, 'AD': 1}).to_numpy()):
            raise RuntimeError('Validation predictions do not match CSV order.')
        validation_table = val_df[['patient_id', 'image_id', 'path', 'label']].copy()
        validation_table['true_label'] = validation_results['labels']
        validation_table['predicted_label'] = validation_results['predictions']
        validation_table['ad_probability'] = validation_results['ad_probabilities']
        validation_table['confidence'] = validation_results['confidence']
        validation_table.to_csv(output_dir / 'validation_predictions.csv', index=False)

    results = evaluate(model, test_loader, device)

    labels = results["labels"]
    predictions = results["predictions"]
    ad_probabilities = results["ad_probabilities"]

    expected_labels = (
        test_df["label"].map({"NC": 0, "AD": 1}).to_numpy()
    )
    if not np.array_equal(labels, expected_labels):
        raise RuntimeError("Predictions do not match test CSV order.")

    if not np.isfinite(ad_probabilities).all():
        raise RuntimeError("Non-finite prediction probabilities.")

    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Both NC and AD are required to calculate AUROC.")

    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["NC", "AD"],
        output_dict=True,
        zero_division=0,
    )

    summary = {
        "model": model_name,
        "checkpoint": str(checkpoint.resolve()),
        "test_slices": len(labels),
        "test_patients": int(test_df["patient_id"].nunique()),
        "test_loss": float(results["loss"]),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "auroc": float(roc_auc_score(labels, ad_probabilities)),
    }

    with (output_dir / "metrics.json").open("w") as file:
        json.dump(summary, file, indent=2)

    pd.DataFrame(report).T.to_csv(
        output_dir / "classification_report.csv"
    )

    # Preserve image identifiers and original CSV order for error analysis.
    prediction_table = test_df[
        ["patient_id", "image_id", "path", "label"]
    ].copy()

    prediction_table["true_label"] = labels
    prediction_table["predicted_label"] = predictions
    prediction_table["predicted_class"] = [
        "NC" if value == 0 else "AD" for value in predictions
    ]
    prediction_table["ad_probability"] = ad_probabilities
    prediction_table["confidence"] = results["confidence"]
    prediction_table["correct"] = predictions == labels

    prediction_table.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
    )

    save_plots(
        labels,
        predictions,
        ad_probabilities,
        model_name,
        output_dir,
    )

    print("Loaded checkpoint:", checkpoint)
    print("Model:", model_name)
    print("Device:", device)
    print("Test slices:", summary["test_slices"])
    print(f"Test loss: {summary['test_loss']:.4f}")
    print(f"Test accuracy: {summary['accuracy']:.2%}")
    print(f"Balanced accuracy: {summary['balanced_accuracy']:.2%}")
    print(f"Macro F1: {summary['macro_f1']:.4f}")
    print(f"AUROC: {summary['auroc']:.4f}")

    print("\nClassification report:")
    print(classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["NC", "AD"],
        digits=4,
        zero_division=0,
    ))

    if args.model == 'gfnet':
        print('80% test accuracy target met:', summary['accuracy'] >= 0.80)
    if not args.skip_analysis:
        confidence_analysis(results, output_dir)
        referral_analysis(validation_results, results, output_dir)
        save_failure_cases(test_dataset, results, model_name, output_dir)
    if args.benchmark:
        benchmark_model(model, model_name, output_dir)
    print("Evaluation files saved to:", output_dir)

def confidence_analysis(results, output_dir):
    """Summarise confidence by correctness and the descriptive >=0.90 subset.

    Maximum softmax probability is a model score, not a calibrated probability
    of correctness. This diagnostic does not select the referral threshold.
    """
    confidence = results['confidence']
    correct = results['predictions'] == results['labels']
    rows = []
    for name, mask in [('Correct predictions', correct), ('Incorrect predictions', ~correct)]:
        values = confidence[mask]
        rows.append({'Group': name, 'Count': int(mask.sum()),
                     'Mean confidence': float(values.mean()) if len(values) else None,
                     'Median confidence': float(np.median(values)) if len(values) else None})
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / 'confidence_summary.csv', index=False)
    high = confidence >= 0.90
    diagnostic = {'threshold': 0.90, 'count': int(high.sum()),
                  'coverage': float(high.mean()),
                  'errors': int((high & ~correct).sum()),
                  'accuracy': float(correct[high].mean()) if high.any() else None}
    with (output_dir / 'high_confidence_diagnostic.json').open('w') as file:
        json.dump(diagnostic, file, indent=2)
    print('\nConfidence summary:\n', summary.to_string(index=False))
    print('Descriptive >=0.90 subset:', diagnostic)


def select_referral_threshold(validation_results):
    """Select a threshold using validation predictions ONLY.

    Preserve the notebook's prespecified grid and constraints: retained
    accuracy >=90%, coverage >=10%; maximise coverage, then prefer the lower
    threshold. Returning None is a valid outcome, not a reason to relax criteria.
    """
    confidence = validation_results['confidence']
    correct = validation_results['labels'] == validation_results['predictions']
    rows = []
    for threshold in np.arange(50, 100) / 100:
        retained = confidence >= threshold
        rows.append({'threshold': float(threshold), 'retained_count': int(retained.sum()),
                     'coverage': float(retained.mean()),
                     'retained_accuracy': float(correct[retained].mean()) if retained.any() else np.nan})
    table = pd.DataFrame(rows)
    eligible = table[(table.coverage >= 0.10) & (table.retained_accuracy >= 0.90)]
    if eligible.empty:
        return table, None
    selected = eligible.sort_values(['coverage', 'threshold'], ascending=[False, True]).iloc[0]
    return table, float(selected.threshold)


def referral_analysis(validation_results, test_results, output_dir):
    """Freeze the validation-selected rule, then assess retained test errors.

    Referral is a simulated decision rule. We do not assume human review fixes
    referred errors and do not claim that the retained predictions are safe.
    """
    table, threshold = select_referral_threshold(validation_results)
    table.to_csv(output_dir / 'validation_thresholds.csv', index=False)
    result = {'target_validation_accuracy': 0.90, 'minimum_validation_coverage': 0.10,
              'selected_threshold': threshold, 'test_evaluated': threshold is not None}
    if threshold is None:
        result['reason'] = 'No candidate met both validation criteria.'
    else:
        selected = table.loc[table.threshold == threshold].iloc[0]
        result.update(validation_coverage=float(selected.coverage),
                      validation_retained_accuracy=float(selected.retained_accuracy))
        labels, predictions = test_results['labels'], test_results['predictions']
        retained = test_results['confidence'] >= threshold
        correct = labels == predictions
        ad, nc = labels == 1, labels == 0
        result.update(test_slices=len(labels), retained_slices=int(retained.sum()),
                      referred_slices=int((~retained).sum()), coverage=float(retained.mean()),
                      referral_rate=float((~retained).mean()),
                      retained_accuracy=float(correct[retained].mean()) if retained.any() else None,
                      retained_errors=int((retained & ~correct).sum()),
                      retained_false_negatives=int((retained & ad & (predictions == 0)).sum()),
                      retained_false_positives=int((retained & nc & (predictions == 1)).sum()),
                      ad_coverage=float(retained[ad].mean()) if ad.any() else None,
                      nc_coverage=float(retained[nc].mean()) if nc.any() else None)
    with (output_dir / 'referral_results.json').open('w') as file:
        json.dump(result, file, indent=2)
    print('\nReferral analysis:\n', json.dumps(result, indent=2))


def save_failure_cases(dataset, results, model_name, output_dir):
    """Select four contrasting errors, prioritising distinct patients.

    Match the notebook ordering: highest-confidence FN, highest-confidence FP,
    lowest-confidence FN, lowest-confidence FP. Fewer cases are saved if an
    error category has no remaining eligible patients. These selected examples
    illustrate failure modes; they do not estimate population frequencies.
    """
    audit = dataset.dataframe.reset_index(drop=True).copy()
    audit['dataset_index'] = np.arange(len(audit))
    audit['true_label'] = results['labels']
    audit['predicted_label'] = results['predictions']
    audit['ad_probability'] = results['ad_probabilities']
    audit['confidence'] = results['confidence']
    fn = audit[(audit.true_label == 1) & (audit.predicted_label == 0)]
    fp = audit[(audit.true_label == 0) & (audit.predicted_label == 1)]
    rules = [('High-confidence FN', fn, False), ('High-confidence FP', fp, False),
             ('Lower-confidence FN', fn, True), ('Lower-confidence FP', fp, True)]
    selected, used = [], set()
    for name, candidates, ascending in rules:
        candidates = candidates[~candidates.patient_id.isin(used)].sort_values('confidence', ascending=ascending)
        if candidates.empty:
            continue
        row = candidates.iloc[0].copy()
        row['case_type'] = name
        selected.append(row)
        used.add(row.patient_id)
    if not selected:
        print('No eligible failure cases; no failure-case figure generated.')
        return
    cases = pd.DataFrame(selected).reset_index(drop=True)
    if 'filename' not in cases:
        cases['filename'] = cases.path.map(lambda p: Path(p).name)
    columns = ['case_type', 'patient_id', 'filename', 'true_label',
               'predicted_label', 'ad_probability', 'confidence']
    cases[columns].to_csv(output_dir / 'failure_cases.csv', index=False)
    fig, axes = plt.subplots(1, len(cases), figsize=(4 * len(cases), 4.5), squeeze=False)
    names = {0: 'NC', 1: 'AD'}
    for i, row in cases.iterrows():
        tensor, label = dataset[int(row.dataset_index)]
        if label != int(row.true_label):
            raise RuntimeError('Failure-case labels do not match dataset order.')
        # Undo only the deterministic normalization for a faithful display.
        image = (tensor.squeeze(0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        axes[0, i].imshow(image, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f"Case {i+1}: {row.case_type}\n"
                             f"True: {names[int(row.true_label)]} | Predicted: {names[int(row.predicted_label)]}\n"
                             f"Confidence: {row.confidence:.2%}", fontsize=10)
        axes[0, i].axis('off')
    fig.suptitle(f'{model_name} Failure Cases')
    fig.tight_layout()
    fig.savefig(output_dir / 'failure_cases.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def benchmark_model(model, model_name, output_dir):
    """Measure batch-one CUDA forward latency and process allocated memory.

    Matches the notebook protocol: FP32 tensors, 30 warmups, 200 CUDA-event
    measurements, then a separate memory measurement. Excludes loading,
    preprocessing and host-to-device transfer. TF32 flags are recorded.
    A standalone process has no training optimizer state, so absolute memory
    allocations need not reproduce the notebook's process-wide measurements.
    """
    import gc
    device = next(model.parameters()).device
    if device.type != 'cuda':
        raise ValueError('--benchmark requires a CUDA device.')
    model.eval()
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    sample = torch.randn(1, 1, 256, 256, device=device, dtype=torch.float32)
    times = []
    with torch.inference_mode():
        for _ in range(30):
            output = model(sample)
        del output
        torch.cuda.synchronize(device)
        for _ in range(200):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(sample)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
            del output
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        output = model(sample)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        del output
    row = {'Model': model_name, 'Parameters': sum(p.numel() for p in model.parameters()),
           'Batch size': 1, 'Mean latency (ms)': float(np.mean(times)),
           'Median latency (ms)': float(np.median(times)),
           'P95 latency (ms)': float(np.percentile(times, 95)),
           'Baseline allocated (MiB)': baseline / 1024**2,
           'Peak allocated (MiB)': peak / 1024**2,
           'Incremental peak (MiB)': (peak - baseline) / 1024**2}
    pd.DataFrame([row]).to_csv(output_dir / 'benchmark.csv', index=False)
    pd.DataFrame({'latency_ms': times}).to_csv(output_dir / 'benchmark_timings.csv', index=False)
    print('\nForward-only inference benchmark:\n', pd.DataFrame([row]).round(3).to_string(index=False))


def environment_info(device):
    """Record actual package versions and GPU flags rather than assuming Colab defaults."""
    import platform
    from importlib.metadata import version
    return {'python': platform.python_version(),
            'packages': {name: version(name) for name in
                         ['torch', 'torchvision', 'numpy', 'pandas', 'scikit-learn', 'Pillow', 'matplotlib']},
            'cuda_build': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32}



if __name__ == "__main__":
    main()