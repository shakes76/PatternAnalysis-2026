"""Shared baseline inference, hard metrics and experiment-file helpers.

These are implementation utilities, not pretrained models. Evaluations are
per complete resized volume, never a mean of slice Dice scores.
"""

import hashlib
import json
import os
from pathlib import Path
import random
import socket
import time

import torch

from dataset import LABEL_NAMES, volume_to_slices
from modules import CAN3D, UNet2D, SoftDiceLoss, segmentation_loss


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_device(name):
    if socket.gethostname().split('.')[0].startswith('login'):
        raise RuntimeError('Use a compute allocation, not a shared login node.')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use an allocated GPU.')
    return torch.device(name)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def save_checkpoint(path, model, config, **extra):
    """Save only tensors and ordinary Python values; replace after writing."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save({'model_state': model.state_dict(), 'config': config, **extra}, temporary)
    temporary.replace(path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if config['model'] == 'unet2d':
        model = UNet2D(base_channels=config['base_channels']).to(device)
    elif config['model'] == 'can3d':
        model = CAN3D().to(device)
    else:
        raise ValueError('Unknown checkpoint architecture.')
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.eval()
    return model, checkpoint


def hard_overlap(predicted, truth):
    """Per-class hard Dice/IoU for one whole case.

    Both empty => None (excluded from that class's mean). Only one empty
    => zero. Nonempty masks use exact counts with no smoothing. Background
    is reported separately; it is never substituted for prostate code 5.
    """
    if predicted.shape != truth.shape:
        raise ValueError('Prediction and truth shapes differ.')
    for labels in (predicted, truth):
        if not torch.all((labels >= 0) & (labels < 6) & (labels == labels.round())):
            raise ValueError('Hard masks must contain integer codes 0..5.')
    table = torch.bincount((truth.long()*6+predicted.long()).flatten(), minlength=36).reshape(6, 6)
    intersection = table.diag()
    actual, detected = table.sum(1), table.sum(0)
    dice, iou = [], []
    for c in range(6):
        denominator = int(actual[c]+detected[c])
        union = int(actual[c]+detected[c]-intersection[c])
        dice.append(2*int(intersection[c])/denominator if denominator else None)
        iou.append(int(intersection[c])/union if union else None)
    return {'dice': dice, 'iou': iou, 'true_voxels': actual.tolist(),
            'predicted_voxels': detected.tolist()}


def mean_present(values):
    valid = [v for v in values if v is not None]
    return sum(valid)/len(valid) if valid else None


def aggregate_cases(cases):
    """Report scan macro and patient macro; latter weights each patient once."""
    patients = sorted({case['patient_id'] for case in cases})
    report = {'case_count': len(cases), 'patient_count': len(patients)}
    for metric in ('dice', 'iou'):
        report['case_macro_'+metric] = [mean_present([case[metric][c] for case in cases]) for c in range(6)]
        report['patient_macro_'+metric] = [mean_present([
            mean_present([case[metric][c] for case in cases if case['patient_id'] == patient])
            for patient in patients]) for c in range(6)]
    return report


@torch.no_grad()
def predict_volume(model, image, device, batch_size=8):
    """Six logits x D x H x W on CPU: reassembled 2D slices or one 3D volume."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    model.eval()
    if isinstance(model, CAN3D):
        # One whole volume: no slicing or patch stitching for CAN3D.
        return model(image[None].to(device))[0].cpu()
    slices = image.permute(1, 0, 2, 3)
    outputs = []
    for first in range(0, len(slices), batch_size):
        outputs.append(model(slices[first:first+batch_size].to(device)).cpu())
    return torch.cat(outputs).permute(1, 0, 2, 3).contiguous()


def evaluate_cases(model, dataset, device, batch_size=8, limit=None, loss_kind='soft-dice'):
    """Evaluate whole volumes; timing excludes data loading and metric work.

    Validation timing is descriptive, not a controlled final benchmark.
    A fresh prediction process has no training optimizer occupying VRAM.
    """
    cases = []
    dice_criterion = SoftDiceLoss()
    criterion = segmentation_loss(loss_kind)
    for i in range(len(dataset) if limit is None else min(limit, len(dataset))):
        sample = dataset[i]
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = predict_volume(model, sample['image'], device, batch_size)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        result = hard_overlap(logits.argmax(0), sample['mask'])
        dice_loss = dice_criterion(logits[None], sample['target'][None]).item()
        objective_loss = (dice_loss if loss_kind == 'soft-dice' else
                          criterion(logits[None], sample['target'][None]).item())
        result.update(case_id=sample['case_id'], patient_id=sample['patient_id'],
                      soft_dice_loss=dice_loss, objective_loss=objective_loss,
                      inference_seconds=seconds)
        cases.append(result)
    if not cases:
        raise ValueError('No cases to evaluate.')
    report = aggregate_cases(cases)
    report.update(cases=cases, label_names=LABEL_NAMES,
                  mean_soft_dice_loss=mean_present([case['soft_dice_loss'] for case in cases]),
                  mean_objective_loss=mean_present([case['objective_loss'] for case in cases]),
                  loss_kind=loss_kind,
                  mean_inference_seconds=mean_present([case['inference_seconds'] for case in cases]),
                  timing_scope='preprocessed CPU input to reassembled CPU logits; no disk I/O or metrics')
    return report


def plot_history(history, output_path, mode, loss_kind='soft-dice'):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(output_path).parent / '.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xkey = 'step' if mode == 'debug' else 'epoch'
    x = [entry[xkey] for entry in history]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(x, [entry['train_loss'] for entry in history], label='training loss')
    if mode != 'debug':
        axes[0].plot(x, [entry['validation_loss'] for entry in history], label='validation loss')
    axes[1].plot(x, [entry['prostate_dice'] for entry in history], label=(
        'selected training slices (debug only)' if mode == 'debug' else 'validation patient macro prostate'))
    for ax in axes:
        ax.set_xlabel(xkey)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('soft Dice + CE loss' if loss_kind == 'dice-ce' else 'soft Dice loss')
    axes[1].set_ylabel('hard Dice')
    axes[1].set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
