"""Load actual saved weights; save native-grid predictions and FP/FN panels."""
from pathlib import Path
import argparse
import logging
import nibabel as nib
import numpy as np
import torch
from dataset import ROOT, HipMRISlices, load_pair, normalize_image
from modules import build_model, load_checkpoint


def panel(image: np.ndarray, truth: np.ndarray | None, prediction: np.ndarray,
          destination: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 4 if truth is not None else 2,
                                figsize=(12 if truth is not None else 6, 4))
    for axis in axes:
        axis.imshow(image.T, cmap='gray', origin='lower')
        axis.axis('off')
    prediction_axis = axes[2] if truth is not None else axes[1]
    prediction_axis.imshow(np.ma.masked_where(prediction.T == 0, prediction.T), cmap='tab10',
                   vmin=0, vmax=5, alpha=0.6, origin='lower')
    titles = ['Input', 'Prediction']
    if truth is not None:
        axes[1].imshow(np.ma.masked_where(truth.T == 0, truth.T), cmap='tab10',
                       vmin=0, vmax=5, alpha=0.6, origin='lower')
        overlay = np.zeros((*truth.T.shape, 4))
        overlay[(prediction.T == 5) & (truth.T != 5)] = [1, 0, 0, 0.8]
        overlay[(prediction.T != 5) & (truth.T == 5)] = [0, 0.6, 1, 0.8]
        axes[3].imshow(overlay, origin='lower')
        titles = ['Input', 'Ground truth (IDs 0-5)', 'Prediction',
                  'Label 5: red FP / blue FN']
    for axis, title in zip(axes, titles):
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--input', type=Path, help='single 2D NIfTI or directory; no labels required')
    parser.add_argument('--labels', type=Path, help='optional matching mask directory for --input')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'validate', 'test'], default='validate')
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    return parser.parse_args()


def prediction_paths(args: argparse.Namespace) -> list[tuple[Path, Path | None]]:
    if args.count <= 0:
        raise ValueError('count must be positive')
    if args.input is None:
        if args.labels is not None:
            raise ValueError('--labels requires --input')
        return HipMRISlices(args.root, args.split, args.count).pairs
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    paths = [args.input] if args.input.is_file() else sorted(
        path for path in args.input.iterdir() if path.is_file()
        and path.name.endswith(('.nii', '.nii.gz')))
    if not paths or any(not path.name.endswith(('.nii', '.nii.gz')) for path in paths):
        raise ValueError('input must contain NIfTI files')
    indices = np.linspace(0, len(paths)-1, min(args.count, len(paths)), dtype=int)
    pairs = [(paths[index], None) for index in indices]
    if args.labels is not None:
        pairs = [(image, args.labels/image.name.replace('case_', 'seg_', 1))
                 for image, _ in pairs]
        for _, label in pairs:
            if not label.is_file():
                raise FileNotFoundError(label)
    return pairs


def save_prediction(image: torch.Tensor, truth: np.ndarray | None,
                    predicted: np.ndarray, source: nib.Nifti1Image,
                    image_path: Path, output: Path) -> None:
    header = source.header.copy()
    header.set_data_dtype(np.uint8)
    name = image_path.name
    mask_name = name.replace('case_', 'pred_', 1) if name.startswith('case_') else 'pred_' + name
    # Retain singleton axes from a source stored as (H, W, 1).
    mask = predicted.astype(np.uint8).reshape(source.shape)
    nib.save(nib.Nifti1Image(mask, source.affine, header), output/mask_name)
    stem = name.removesuffix('.gz').removesuffix('.nii')
    panel(image[0].numpy(), truth, predicted, output/f'{stem}.png')


def main() -> None:
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; pass --device cpu')
    pairs = prediction_paths(args)
    checkpoint = load_checkpoint(args.checkpoint, args.device)
    config = checkpoint['config']
    model = build_model(config['model'], config['width']).to(args.device).eval()
    model.load_state_dict(checkpoint['state_dict'])
    args.output.mkdir(parents=True, exist_ok=False)
    for image_path, label_path in pairs:
        source = nib.load(image_path)
        if label_path is None:
            image, truth = normalize_image(np.asarray(source.dataobj)), None
        else:
            image, labels = load_pair(image_path, label_path)
            truth = labels.numpy()
        with torch.inference_mode():
            predicted = model(image.unsqueeze(0).to(args.device)).argmax(1)[0].cpu().numpy()
        save_prediction(image, truth, predicted, source, image_path, args.output)
        logging.info('Predicted %s', image_path.name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
