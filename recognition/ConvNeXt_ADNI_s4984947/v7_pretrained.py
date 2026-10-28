"""Strictly map torchvision ConvNeXt-Tiny weights to hand-coded ConvNeXt.

Every backbone tensor must match; classifier weights are freshly initialized
for two-class AD/NC classification. No test data is accessed.
"""
import argparse
import re
from pathlib import Path

import torch
from torch import nn
from modules import ConvNeXtClassifier

BLOCKS = {
    **{i: (1, i) for i in range(3)},
    **{i: (3, i - 5) for i in range(5, 8)},
    **{i: (5, i - 10) for i in range(10, 19)},
    **{i: (7, i - 21) for i in range(21, 24)},
}
DOWN_NORM = {3: 2, 8: 4, 19: 6}
DOWN_CONV = {4: 2, 9: 4, 20: 6}
BLOCK_PARTS = {
    'depthwise': 'block.0',
    'norm': 'block.2',
    'expand': 'block.3',
    'project': 'block.5',
    'gamma': 'layer_scale',
}


def official_name(custom_name):
    if custom_name.startswith('stem.0.'):
        return custom_name.replace('stem.0.', 'features.0.0.', 1)
    if custom_name.startswith('stem.1.norm.'):
        return custom_name.replace('stem.1.norm.', 'features.0.1.', 1)
    if custom_name.startswith('norm.'):
        return custom_name.replace('norm.', 'classifier.0.', 1)
    if custom_name.startswith('classifier.'):
        return custom_name.replace('classifier.', 'classifier.2.', 1)

    match = re.fullmatch(r'features\.(\d+)\.(.+)', custom_name)
    if not match:
        raise KeyError(f'Unexpected custom key: {custom_name}')
    idx = int(match.group(1))
    suffix = match.group(2)
    if idx in BLOCKS:
        stage, block_idx = BLOCKS[idx]
        part, dot, tail = suffix.partition('.')
        if part not in BLOCK_PARTS or not dot:
            raise KeyError(f'Unexpected block key: {custom_name}')
        if part == 'gamma':
            raise KeyError(f'Unexpected gamma suffix: {custom_name}')
        return f'features.{stage}.{block_idx}.{BLOCK_PARTS[part]}.{tail}'
    if idx in DOWN_NORM:
        if not suffix.startswith('norm.'):
            raise KeyError(f'Unexpected downsample norm: {custom_name}')
        return f'features.{DOWN_NORM[idx]}.0.{suffix[len("norm."):]}'
    if idx in DOWN_CONV:
        return f'features.{DOWN_CONV[idx]}.1.{suffix}'
    raise KeyError(f'Unknown feature index: {idx}')


def load_custom_pretrained(weights_file, num_classes=2):
    weights_file = Path(weights_file)
    if not weights_file.is_file():
        raise FileNotFoundError(weights_file)
    official = torch.load(weights_file, map_location='cpu', weights_only=True)
    model = ConvNeXtClassifier(num_classes=num_classes)

    # TorchVision uses eps=1e-6 in all its ConvNeXt LayerNorm layers.
    for layer in model.modules():
        if isinstance(layer, nn.LayerNorm):
            layer.eps = 1e-6

    target = model.state_dict()
    new_state = {}
    matched_source = set()
    for custom_key, current_value in target.items():
        if num_classes != 1000 and custom_key.startswith('classifier.'):
            new_state[custom_key] = current_value  # new AD/NC classification head
            continue
        if custom_key.endswith('.gamma'):
            idx = int(custom_key.split('.')[1])
            stage, block_idx = BLOCKS[idx]
            source_key = f'features.{stage}.{block_idx}.layer_scale'
        else:
            source_key = official_name(custom_key)
        if source_key not in official:
            raise KeyError(f'Missing in pretrained weights: {source_key} (for {custom_key})')
        source_tensor = official[source_key]
        if custom_key.endswith('.gamma'):
            source_tensor = source_tensor.reshape(-1)
        if source_tensor.shape != current_value.shape:
            raise RuntimeError(f'Shape mismatch: {custom_key}: '
                               f'{tuple(current_value.shape)} != {tuple(source_tensor.shape)}')
        if source_key in matched_source:
            raise RuntimeError(f'Duplicate source mapping: {source_key}')
        matched_source.add(source_key)
        new_state[custom_key] = source_tensor

    expected_sources = set(official)
    if num_classes != 1000:
        expected_sources -= {'classifier.2.weight', 'classifier.2.bias'}
    if matched_source != expected_sources:
        raise RuntimeError(f'Incomplete mapping: missing {sorted(expected_sources-matched_source)}, '
                           f'extra {sorted(matched_source-expected_sources)}')

    model.load_state_dict(new_state, strict=True)
    print(f'Pretrained tensors mapped: {len(matched_source)}/{len(expected_sources)}', flush=True)
    print('Classifier:', 'new 2-class head' if num_classes != 1000 else 'ImageNet head', flush=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='pretrained/convnext_tiny-983f1562.pth')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = load_custom_pretrained(args.weights)
    print('Custom architecture ready:', sum(p.numel() for p in model.parameters()), 'parameters')
    if args.verify:
        from torchvision.models import convnext_tiny
        print('Checking equivalence of official and custom 1000-class outputs on CPU...')
        official = convnext_tiny(weights=None).eval()
        official.load_state_dict(torch.load(args.weights, map_location='cpu', weights_only=True), strict=True)
        custom = load_custom_pretrained(args.weights, num_classes=1000).eval()
        with torch.no_grad():
            torch.manual_seed(3710)
            sample = torch.randn(1, 3, 64, 64)
            diff = (official(sample) - custom(sample)).abs().max().item()
        print('Maximum output absolute difference:', diff)
        if diff >= 1e-3:
            raise RuntimeError(f'Network parity check failed; max diff={diff}')
        print('PASS: Custom ConvNeXt matches official pretrained model in eval mode.')


if __name__ == '__main__':
    main()
