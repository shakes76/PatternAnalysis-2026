"""Plot orthogonal training-only images with class-ID contours for alignment QA."""

from pathlib import Path
import argparse
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import HipMRIVolumeDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-class', type=int, default=5)
    args = parser.parse_args()
    dataset = HipMRIVolumeDataset(args.manifest, 'train', augment=False)
    if not 1 <= args.target_class < dataset.num_classes:
        parser.error('target-class must be a declared foreground class index.')
    item = dataset[0]
    image, mask = item['image'][0].numpy(), item['mask'].numpy()
    coordinates = np.argwhere(mask == args.target_class)
    center = np.round(coordinates.mean(axis=0)).astype(int) if len(coordinates) else np.array(mask.shape) // 2

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    figure, axes = plt.subplots(3, 2, figsize=(8, 10))
    palette = plt.get_cmap('tab10')
    limits = np.percentile(image, [1, 99])
    for dimension, name in enumerate(('Axial (D)', 'Coronal (H)', 'Sagittal (W)')):
        plane = np.take(image, int(center[dimension]), axis=dimension)
        labels = np.take(mask, int(center[dimension]), axis=dimension)
        for axis in axes[dimension]:
            axis.imshow(plane, cmap='gray', vmin=limits[0], vmax=limits[1], interpolation='nearest')
            axis.axis('off')
        for label in range(1, dataset.num_classes):
            selected = labels == label
            if selected.any() and not selected.all():
                axes[dimension, 1].contour(selected, levels=[0.5], colors=[palette(label)], linewidths=0.7)
        axes[dimension, 0].set_title(f'{name}: image', fontsize=10)
        axes[dimension, 1].set_title('Same grid with ground-truth contours', fontsize=10)
    legend = [Line2D([0], [0], color=palette(label), label=f'Class {label}') for label in range(1, dataset.num_classes)]
    figure.legend(handles=legend, loc='lower center', ncol=dataset.num_classes - 1, fontsize=9)
    figure.suptitle(f"Training-only alignment QA: {item['patient']} / {item['visit']}\nCanonical RAS / resampled grid; anatomical label names unverified", fontsize=11)
    figure.tight_layout(rect=(0, 0.035, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)
    print(args.output)


if __name__ == '__main__':
    main()
