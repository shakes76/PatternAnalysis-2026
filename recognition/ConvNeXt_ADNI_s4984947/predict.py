"""Inference driver for the locked ConvNeXt V7 ADNI classifier.

Examples:
    python predict.py --split val --examples-per-class 2
    python predict.py --image /path/to/an_ADNI_slice.jpeg

Uses only the already fine-tuned checkpoint; official ImageNet weights
are NOT required for inference. Saves sample figures to ignored runs/.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from PIL import Image, ImageOps
from torchvision import transforms

from dataset_v2 import build_splits
from modules import ConvNeXtClassifier

NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def read_mri(path):
    """Match the final train/validation/test preprocessing exactly."""
    with Image.open(path) as img:
        rgb = ImageOps.pad(img.convert('RGB'), (224, 224),
                           method=Image.Resampling.BILINEAR, color=0)
    return NORMALIZE(rgb), rgb


def load_locked_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError('Expected a V7 best.pt checkpoint containing model_state_dict.')
    model = ConvNeXtClassifier(num_classes=2)
    # V7 uses the official ConvNeXt-Tiny LayerNorm epsilon = 1e-6.
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.eps = 1e-6
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device).eval()
    return model, checkpoint.get('epoch', 'unknown')


def choose_examples(split, per_class):
    """Choose reproducible examples without revealing patient or scan identifiers."""
    rows = build_splits(seed=3710)[split]
    picked = []
    for label in (0, 1):
        candidates = sorted((r for r in rows if r[1] == label), key=lambda r: str(r[0]))
        if not candidates:
            raise RuntimeError('No examples found for class ' + str(label))
        # Spread selected examples across the available filename order, not by confidence.
        k = min(per_class, len(candidates))
        for i in range(k):
            idx = min(len(candidates)-1, int((i+0.5) * len(candidates)/k))
            picked.append((candidates[idx][0], label))
    return picked


def main():
    p = argparse.ArgumentParser(description='Predict AD/NC from MRI slices with the locked V7 checkpoint.')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--image', type=Path, help='A single local JPEG MRI slice')
    group.add_argument('--split', choices=['val', 'test'], default='val',
                       help='Use example slices from a patient-disjoint split (default: val)')
    p.add_argument('--examples-per-class', type=int, default=2)
    p.add_argument('--checkpoint', type=Path,
                   default=Path('runs/convnext_v7_pretrained20_newsplit/best.pt'))
    p.add_argument('--save-dir', type=Path, default=Path('runs/predict_examples'))
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = p.parse_args()
    if args.examples_per_class < 1:
        p.error('--examples-per-class must be >= 1')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else ('cpu' if args.device == 'auto' else args.device))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but no GPU allocated; use Slurm or --device cpu.')
    if not args.checkpoint.exists():
        raise FileNotFoundError(f'{args.checkpoint} not found. Checkpoint files are not stored in Git.')
    model, epoch = load_locked_model(args.checkpoint, device)
    print(f'Loaded ConvNeXt V7 checkpoint epoch={epoch}; device={device}', flush=True)
    samples = [(args.image, None)] if args.image else choose_examples(args.split, args.examples_per_class)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    for index, (path, truth) in enumerate(samples, start=1):
        image, rgb = read_mri(path)
        with torch.inference_mode():
            if device.type == 'cuda' and torch.cuda.is_bf16_supported():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(image.unsqueeze(0).to(device))
            else:
                logits = model(image.unsqueeze(0).to(device))
            p_ad = float(logits.float().softmax(1)[0, 1].item())
        predicted = 'AD' if p_ad >= 0.5 else 'NC'
        confidence = max(p_ad, 1.0-p_ad)
        t = 'unlabelled' if truth is None else ('AD' if truth else 'NC')
        print(f'Example {index}: predicted={predicted}, P(AD)={p_ad:.4f}, '
              f'confidence={confidence:.4f}, true={t}', flush=True)
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.imshow(rgb.convert('L'), cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'Example {index} | true: {t} | predicted: {predicted}\n'
                     f'P(AD)={p_ad:.3f}; confidence={confidence:.3f}')
        ax.axis('off')
        fig.tight_layout()
        output = args.save_dir / f'example_{index:02d}.png'
        fig.savefig(output, dpi=140)
        plt.close(fig)
    print(f'Figures saved under {args.save_dir}/ (do not commit raw MRI case images).')


if __name__ == '__main__':
    main()
