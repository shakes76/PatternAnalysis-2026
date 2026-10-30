"""Visualise U-Net predictions and representative prostate segmentation failures."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import test_dataset
from evaluate import PROSTATE_CLASS, calculate_prostate_metrics, device, load_model


RESULTS_DIR = Path('results')
FAILURE_DIR = RESULTS_DIR / 'failure_cases'

CHECKPOINTS = {
    'baseline': 'checkpoints/unet_baseline_best.pth',
    'improved': 'checkpoints/unet_improved_best.pth',
}


def read_csv(path, required_columns):
    """Read a CSV file and validate its required columns."""
    if not path.is_file():
        raise FileNotFoundError(
            f'Missing {path}. Run evaluate.py and regional_analysis.py first.'
        )

    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        missing = set(required_columns) - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f'{path} is missing columns: {sorted(missing)}'
            )

        return list(reader)


def plot_mask(ax, image, mask, color, title):
    """Display the prostate segmentation over the original MRI."""
    ax.imshow(image, cmap='gray')

    overlay = np.ma.masked_where(
        mask != PROSTATE_CLASS,
        mask == PROSTATE_CLASS
    )

    ax.imshow(
        overlay,
        cmap=color,
        vmin=0,
        vmax=1,
        alpha=0.6
    )

    ax.set_title(title)
    ax.axis('off')


def plot_error_overlay(ax, image, ground_truth, prediction, title):
    """Display false positives in red and false negatives in blue."""
    false_positives = (
        (prediction == PROSTATE_CLASS)
        & (ground_truth != PROSTATE_CLASS)
    )

    false_negatives = (
        (prediction != PROSTATE_CLASS)
        & (ground_truth == PROSTATE_CLASS)
    )

    ax.imshow(image, cmap='gray')

    ax.imshow(
        np.ma.masked_where(~false_positives, false_positives),
        cmap='Reds',
        vmin=0,
        vmax=1,
        alpha=0.8
    )

    ax.imshow(
        np.ma.masked_where(~false_negatives, false_negatives),
        cmap='Blues',
        vmin=0,
        vmax=1,
        alpha=0.8
    )

    ax.set_title(title)
    ax.axis('off')


def predict_one(model, image_tensor):
    """Generate a predicted segmentation mask for one MRI slice."""
    with torch.inference_mode():
        logits = model(image_tensor.unsqueeze(0).to(device))

        return (
            logits.argmax(dim=1)
            .squeeze(0)
            .cpu()
            .numpy()
        )


def find_failure_cases(num_cases=5):
    """Select representative failure cases using regional evaluation results."""
    regional = read_csv(
        RESULTS_DIR / 'regional_per_slice.csv',
        [
            'model', 'index', 'region', 'dice',
            'tp', 'fp', 'fn', 'si_lower', 'si_upper'
        ]
    )

    components = read_csv(
        RESULTS_DIR / 'disconnected_components.csv',
        [
            'model', 'patient', 'week', 'slice',
            'filename', 'extra_components'
        ]
    )

    # Match component results to their test dataset indices.
    filename_to_index = {
        Path(path).name: index
        for index, path in enumerate(test_dataset.image_paths)
    }

    region_by_index = {}

    # Organise regional results by image, region and model.
    for row in regional:
        index = int(row['index'])

        if index < 0 or index >= len(test_dataset):
            raise ValueError(
                f'Regional CSV contains invalid test index: {index}'
            )

        if row['model'] not in CHECKPOINTS:
            continue

        region_by_index.setdefault(index, {}).setdefault(
            row['region'], {}
        )[row['model']] = row

    extra = {}

    # Store the number of extra disconnected components.
    for row in components:
        index = filename_to_index.get(row['filename'])

        if index is not None and row['model'] in CHECKPOINTS:
            extra[(index, row['model'])] = int(
                row['extra_components']
            )

    area = {}

    # Load whole-slice prostate areas to identify possible under-segmentation.
    for model_name in CHECKPOINTS:
        rows = read_csv(
            RESULTS_DIR / f'{model_name}_prostate_slices.csv',
            ['index', 'ground_truth_pixels', 'predicted_pixels']
        )

        for row in rows:
            area[(int(row['index']), model_name)] = (
                int(row['ground_truth_pixels']),
                int(row['predicted_pixels'])
            )

    # Only consider slices with regional results for both models.
    candidates = sorted(
        index
        for index in region_by_index
        if any(
            all(
                model in region_by_index[index].get(region, {})
                for model in CHECKPOINTS
            )
            for region in ('Apex', 'Mid-gland', 'Base')
        )
    )

    if not candidates:
        raise ValueError(
            'No common regional test cases found for both models.'
        )

    def region_score(index, region):
        """Return the lowest Dice score for a specified region."""
        rows = region_by_index[index].get(region, {})

        return (
            min(float(row['dice']) for row in rows.values())
            if rows else float('inf')
        )

    def model_difference(index):
        """Return the largest regional Dice difference between models."""
        return max(
            (
                abs(
                    float(rows['baseline']['dice'])
                    - float(rows['improved']['dice'])
                )
                for rows in region_by_index[index].values()
                if 'baseline' in rows and 'improved' in rows
            ),
            default=0.0
        )

    selected = []

    def choose(indices, reason, region=None):
        """Select the highest-ranked unused failure case."""
        for index in indices:
            if index not in [entry[0] for entry in selected]:
                if len(selected) < num_cases:
                    selected.append((index, reason, region))

                    print(
                        f'{reason}: '
                        f'{Path(test_dataset.image_paths[index]).name}'
                    )

                    return

        print(f'No additional case available for: {reason}')

    # Select poorly segmented apex and base regions.
    for region in ('Apex', 'Base'):
        eligible = [
            index
            for index in candidates
            if region in region_by_index[index]
            and any(
                int(row['tp']) + int(row['fn']) > 0
                for row in region_by_index[index][region].values()
            )
        ]

        choose(
            sorted(
                eligible,
                key=lambda i: region_score(i, region)
            ),
            f'Poor {region.lower()} segmentation',
            region
        )

    # Identify cases with disconnected prostate predictions.
    disconnected = [
        i for i in candidates
        if extra.get((i, 'improved'), 0) > 0
    ]

    choose(
        sorted(
            disconnected,
            key=lambda i: extra[(i, 'improved')],
            reverse=True
        ),
        'Improved U-Net disconnected components'
    )

    def area_deficit(index):
        """Calculate the largest predicted prostate area deficit."""
        return max(
            (
                gt - pred
                for model in CHECKPOINTS
                for gt, pred in [
                    area.get((index, model), (0, 0))
                ]
            ),
            default=0
        )

    # Select a possible under-segmentation failure.
    choose(
        sorted(
            [
                i for i in candidates
                if area_deficit(i) > 0
            ],
            key=area_deficit,
            reverse=True
        ),
        'Predicted area deficit'
    )

    # Select a case with a large performance difference between models.
    choose(
        sorted(
            candidates,
            key=model_difference,
            reverse=True
        ),
        'Largest regional Dice difference'
    )

    def worst_dice(index):
        """Return the lowest regional Dice across both models."""
        return min(
            float(row['dice'])
            for rows in region_by_index[index].values()
            for row in rows.values()
        )

    # Fill remaining slots with other poor regional predictions.
    for index in sorted(candidates, key=worst_dice):
        if len(selected) >= num_cases:
            break

        if index not in [entry[0] for entry in selected]:
            selected.append((
                index,
                'Additional poor regional prediction',
                None
            ))

    return selected[:num_cases], region_by_index


def metric_text(metrics):
    """Format Dice, IoU and HD95 for figure titles."""
    if metrics is None:
        return 'No prostate in either mask'

    return (
        f"Dice={metrics['dice']:.3f}, "
        f"IoU={metrics['iou']:.3f}, "
        f"HD95={metrics['hd95_pixels']:.2f}px"
    )


def error_counts(pred, truth):
    """Count false-positive and false-negative prostate pixels."""
    fp = int(np.count_nonzero(
        (pred == PROSTATE_CLASS)
        & (truth != PROSTATE_CLASS)
    ))

    fn = int(np.count_nonzero(
        (pred != PROSTATE_CLASS)
        & (truth == PROSTATE_CLASS)
    ))

    return fp, fn


def generate_failure_cases(models, num_cases=5):
    """Generate six-panel visualisations of selected segmentation failures."""
    selected, regional = find_failure_cases(num_cases)

    FAILURE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for index, reason, region in selected:
        image_tensor, mask_tensor = test_dataset[index]

        image = image_tensor.squeeze().cpu().numpy()
        truth = mask_tensor.cpu().numpy()

        # Generate predictions from both trained models.
        predictions = {
            name: predict_one(model, image_tensor)
            for name, model in models.items()
        }

        if any(
            pred.shape != truth.shape
            for pred in predictions.values()
        ):
            raise ValueError(
                f'Prediction/ground truth shape mismatch at index {index}'
            )

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        # Display the MRI, ground truth and both predictions.
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original MRI')
        axes[0, 0].axis('off')

        plot_mask(
            axes[0, 1],
            image,
            truth,
            'Greens',
            'Ground truth (prostate)'
        )

        plot_mask(
            axes[0, 2],
            image,
            predictions['baseline'],
            'Reds',
            'Baseline prediction'
        )

        plot_mask(
            axes[1, 0],
            image,
            predictions['improved'],
            'Blues',
            'Improved prediction'
        )

        # Display false-positive and false-negative errors.
        plot_error_overlay(
            axes[1, 1],
            image,
            truth,
            predictions['baseline'],
            'Baseline errors (FP red / FN blue)'
        )

        plot_error_overlay(
            axes[1, 2],
            image,
            truth,
            predictions['improved'],
            'Improved errors (FP red / FN blue)'
        )

        descriptions = []

        # Calculate accuracy and error counts for each model.
        for name in CHECKPOINTS:
            metrics = calculate_prostate_metrics(
                predictions[name],
                truth
            )

            fp, fn = error_counts(
                predictions[name],
                truth
            )

            descriptions.append(
                f'{name.title()}: {metric_text(metrics)}, '
                f'FP={fp}, FN={fn}'
            )

        filename = Path(
            test_dataset.image_paths[index]
        ).name

        fig.suptitle(
            f'{filename} | {reason}\n'
            + '\n'.join(descriptions),
            fontsize=11
        )

        fig.tight_layout(rect=[0, 0, 1, 0.89])

        destination = (
            FAILURE_DIR / f'failure_{index:04d}.png'
        )

        # Save the failure-case figure.
        fig.savefig(destination, dpi=200)
        plt.close(fig)

        print(f'Saved {destination}')

        manifest.append({
            'index': index,
            'filename': filename,
            'reason': reason,
            'region': region or '',
            'figure': str(destination)
        })

    # Save the selected cases and their selection reasons.
    with (
        FAILURE_DIR / 'selected_cases.csv'
    ).open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'index',
                'filename',
                'reason',
                'region',
                'figure'
            ]
        )

        writer.writeheader()
        writer.writerows(manifest)

    print(f'Saved {len(manifest)} failure-case figures')


def generate_overview(models):
    """Generate general prediction and error comparison figures."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    image_tensor, mask_tensor = test_dataset[0]

    image = image_tensor.squeeze().cpu().numpy()
    truth = mask_tensor.cpu().numpy()

    preds = {
        name: predict_one(model, image_tensor)
        for name, model in models.items()
    }

    # Compare the original MRI, ground truth and model predictions.
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original MRI')
    axes[0].axis('off')

    plot_mask(
        axes[1],
        image,
        truth,
        'Greens',
        'Ground truth'
    )

    plot_mask(
        axes[2],
        image,
        preds['baseline'],
        'Reds',
        'Baseline U-Net'
    )

    plot_mask(
        axes[3],
        image,
        preds['improved'],
        'Blues',
        'Improved U-Net'
    )

    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / 'prediction_comparison.png',
        dpi=200
    )
    plt.close(fig)

    # Compare false-positive and false-negative errors.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original MRI')
    axes[0].axis('off')

    plot_error_overlay(
        axes[1],
        image,
        truth,
        preds['baseline'],
        'Baseline errors'
    )

    plot_error_overlay(
        axes[2],
        image,
        truth,
        preds['improved'],
        'Improved errors'
    )

    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / 'error_comparison.png',
        dpi=200
    )
    plt.close(fig)

    print(
        'Saved prediction_comparison.png and error_comparison.png'
    )


def main():
    """Load both models and generate segmentation comparison figures."""
    if len(test_dataset) == 0:
        raise ValueError('Test dataset is empty')

    models = {}

    # Load the best checkpoints and compare trainable parameters.
    for name, checkpoint in CHECKPOINTS.items():
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(
                f'Missing checkpoint: {checkpoint}'
            )

        models[name] = load_model(name, checkpoint)

        parameters = sum(
            p.numel()
            for p in models[name].parameters()
            if p.requires_grad
        )

        print(
            f'{name.title()} trainable parameters: {parameters:,}'
        )

    # Generate overview figures and five representative failure cases.
    generate_overview(models)
    generate_failure_cases(models, num_cases=5)


if __name__ == '__main__':
    main()