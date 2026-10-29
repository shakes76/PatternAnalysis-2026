"""Load a saved architecture and export one aligned validation prediction."""
import argparse
import io
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from dataset import HipMRI2DDataset
from metrics import confusion_matrix, dice_iou_from_confusion
from model_factory import model_from_checkpoint
from training_io import load_checkpoint


def load_predictor(checkpoint_path, device='cpu', allow_partial=False):
    saved = load_checkpoint(checkpoint_path)
    config = saved['config']
    if not isinstance(config.get('partial_run'), bool):
        raise ValueError('Checkpoint must explicitly record partial_run')
    if config['partial_run'] and not allow_partial:
        raise ValueError('Partial checkpoint: use --allow-partial only for a smoke check')
    model_config = saved['model_config']
    if model_config['in_channels'] != 1 or model_config['num_classes'] != 6:
        raise ValueError('Expected one MRI input channel and six numerical label IDs')
    normalize = config['preprocessing']['normalize']
    if not isinstance(normalize, bool):
        raise ValueError('Checkpoint normalization setting must be boolean')
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable on this node')
    model = model_from_checkpoint(saved)
    model.to(device).eval()
    return model, saved


def infer_mask(model, image, device='cpu'):
    if image.ndim != 3 or image.shape[0] != 1 or not image.is_floating_point():
        raise ValueError('Expected one float MRI image with shape [1, H, W]')
    if not torch.isfinite(image).all():
        raise ValueError('MRI image must be finite')
    model.eval()
    with torch.inference_mode():
        logits = model(image.unsqueeze(0).to(device))
        if tuple(logits.shape) != (1, 6, *image.shape[-2:]) or not torch.isfinite(logits).all():
            raise RuntimeError('Invalid prediction logits')
        return logits.argmax(dim=1)[0].cpu()


def overlap_map(prediction, target, label_id):
    """Codes: 0=neither, 1=TP, 2=FP, 3=FN for one numerical label."""
    if type(label_id) is not int or not 0 <= label_id <= 5:
        raise ValueError('label_id must be an integer in range 0-5')
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError('Expected matching 2D prediction and target')
    if prediction.dtype != torch.long or target.dtype != torch.long:
        raise ValueError('Prediction and target must contain torch.long IDs')
    if prediction.device != target.device:
        raise ValueError('Prediction and target must share a device')
    predicted = prediction == label_id
    actual = target == label_id
    codes = torch.zeros_like(target)
    codes[predicted & actual] = 1
    codes[predicted & ~actual] = 2
    codes[~predicted & actual] = 3
    return codes


def save_figure(image, target, prediction, codes, label_id, destination, title, normalized):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    colors = plt.get_cmap('tab10').colors[:6]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)
    overlay_colors = [(0, 0, 0, 0), (0.1, 0.75, 0.2, 0.6), (0.9, 0.15, 0.1, 0.65), (0.1, 0.4, 0.95, 0.65)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), layout='constrained')
    fig.suptitle(title, fontsize=12)
    mri = image[0].numpy()
    axes[0].imshow(mri, cmap='gray', origin='lower', interpolation='nearest')
    axes[0].set_title('MRI (z-score)' if normalized else 'MRI (raw)')
    target_image = axes[1].imshow(target.numpy(), cmap=cmap, norm=norm, origin='lower', interpolation='nearest')
    axes[1].set_title('Ground truth')
    axes[2].imshow(prediction.numpy(), cmap=cmap, norm=norm, origin='lower', interpolation='nearest')
    axes[2].set_title('Prediction')
    axes[3].imshow(mri, cmap='gray', origin='lower', interpolation='nearest')
    axes[3].imshow(codes.numpy(), cmap=ListedColormap(overlay_colors), vmin=-0.5, vmax=3.5,
                   origin='lower', interpolation='nearest')
    axes[3].set_title(f'Label {label_id} overlap')
    axes[3].legend([Patch(facecolor=color) for color in overlay_colors[1:]], ['TP', 'FP', 'FN'],
                   loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8, frameon=False)
    for axis in axes:
        axis.set_axis_off()
    fig.colorbar(target_image, ax=list(axes[1:3]), ticks=range(6), shrink=0.7,
                 label='Numeric label ID', fraction=0.035, pad=0.02)
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=160, facecolor='white', metadata={'Description': title})
        with Path(destination).open('xb') as handle:
            handle.write(buffer.getvalue())
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--mask-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory for this prediction')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--label-id', type=int, choices=range(6), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--allow-partial', action='store_true', help='Explicitly permit a smoke-check checkpoint')
    args = parser.parse_args(argv)
    if args.index < 0:
        parser.error('index must be non-negative')
    destination = args.output_dir.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f'Prediction directory already exists: {destination}')
    torch.set_num_threads(1)
    model, saved = load_predictor(args.checkpoint, args.device, args.allow_partial)
    normalize = saved['config']['preprocessing']['normalize']
    dataset = HipMRI2DDataset(args.image_dir, args.mask_dir, normalize=normalize)
    if args.index >= len(dataset):
        parser.error('index is outside the dataset')
    image, target = dataset[args.index]
    prediction = infer_mask(model, image, args.device)
    matrix = confusion_matrix(prediction, target)
    scores = dice_iou_from_confusion(matrix)
    codes = overlap_map(prediction, target, args.label_id)
    image_path, mask_path = dataset.pairs[args.index]
    partial = saved['config']['partial_run']
    status = 'PARTIAL CHECKPOINT - smoke only' if partial else 'Checkpoint prediction'
    title = f'{saved["architecture"]} | epoch {saved["epoch"]} | {status}\n{image_path.name}'
    report = {
        'checkpoint': str(args.checkpoint.expanduser().resolve()), 'epoch': saved['epoch'],
        'partial_checkpoint': partial, 'image': str(image_path), 'target': str(mask_path),
        'architecture': saved['architecture'], 'model_config': saved['model_config'],
        'normalization': 'per-slice z-score' if normalize else 'none',
        'device': args.device, 'shape': list(prediction.shape), 'label_id': args.label_id,
        'pixels': target.numel(), 'confusion': matrix.tolist(),
        'dice': [None if math.isnan(value) else value for value in scores['dice'].tolist()],
        'iou': [None if math.isnan(value) else value for value in scores['iou'].tolist()],
        'selected_label_counts': {name: int((codes == code).sum())
                                  for code, name in ((1, 'tp'), (2, 'fp'), (3, 'fn'))},
    }
    destination.mkdir(parents=True, exist_ok=False)
    save_figure(image, target, prediction, codes, args.label_id,
                destination / 'prediction.png', title, normalize)
    reference = nib.load(str(image_path))
    nifti = nib.Nifti1Image(prediction.numpy().astype(np.uint8), reference.affine,
                          header=reference.header.copy())
    nifti.set_data_dtype(np.uint8)
    nib.save(nifti, str(destination / 'prediction.nii.gz'))
    with (destination / 'prediction.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print('Architecture:', saved['architecture'])
    print('Checkpoint epoch:', saved['epoch'])
    print('Partial checkpoint:', partial)
    print('Normalization:', report['normalization'])
    print('Prediction shape:', tuple(prediction.shape))
    print('Selected numeric label:', args.label_id, report['selected_label_counts'])
    print('Output directory:', destination)
    print('Prediction export: PASS')
    return report


if __name__ == '__main__':
    main()
