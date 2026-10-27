"""Map official torchvision ImageNet weights into our custom architecture.

Source: https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
Weight release: ConvNeXt_Tiny_Weights.IMAGENET1K_V1.
Only the 1000-class head is discarded; the custom binary head stays random.
"""
import hashlib
from pathlib import Path
import torch

WEIGHTS_URL = 'https://download.pytorch.org/models/convnext_tiny-983f1562.pth'


def parameter_mapping(model):
    mapping = {}
    def pair(a, b):
        for suffix in ('weight', 'bias'):
            mapping[a + '.' + suffix] = b + '.' + suffix
    pair('stem.0', 'features.0.0')
    pair('stem.1.norm', 'features.0.1')
    for stage, blocks in enumerate(model.stages):
        for block in range(len(blocks)):
            a, b = f'stages.{stage}.{block}', f'features.{2*stage+1}.{block}'
            for ours, theirs in [('depthwise','0'),('norm','2'),('expand','3'),('project','5')]:
                pair(f'{a}.{ours}',f'{b}.block.{theirs}')
            mapping[a+'.layer_scale'] = b+'.layer_scale'
        if stage < 3:
            pair(f'downsamples.{stage}.0.norm',f'features.{2*stage+2}.0')
            pair(f'downsamples.{stage}.1',f'features.{2*stage+2}.1')
    pair('final_norm','classifier.0')
    pair('classifier','classifier.2')
    return mapping


def load_backbone(model, state):
    mapping = parameter_mapping(model)
    target = model.state_dict()
    if set(mapping) != set(target):
        raise ValueError('Incomplete custom parameter mapping')
    if set(state) != set(mapping.values()):
        raise ValueError('Unexpected source keys')
    if tuple(state['classifier.2.weight'].shape) != (1000, 768) or tuple(state['classifier.2.bias'].shape) != (1000,):
        raise ValueError('Expected the official 1000-class source head')
    for name, source_name in mapping.items():
        if name.startswith('classifier.'):
            continue
        value = state[source_name]
        expected = target[name].shape
        if name.endswith('.layer_scale'):
            if tuple(value.shape) != (expected[0], 1, 1):
                raise ValueError('Unexpected Layer Scale shape')
            value = value.reshape(expected)
        if value.shape != expected or not torch.isfinite(value).all():
            raise ValueError('Invalid source tensor: '+source_name)
        target[name] = value
    model.load_state_dict(target, strict=True)
    return len(mapping)-2


def load_official_weights(model, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / WEIGHTS_URL.rsplit('/', 1)[1]
    state = torch.hub.load_state_dict_from_url(WEIGHTS_URL, model_dir=str(cache_dir),
                                              map_location='cpu', check_hash=True, weights_only=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith('983f1562'):
        raise ValueError('Cached official weight checksum mismatch; do not use this file')
    count = load_backbone(model, state)
    return {'source':WEIGHTS_URL, 'sha256':digest, 'loaded_tensors':count,
            'discarded':'ImageNet 1000-class head', 'binary_head':'random initialization',
            'all_layers_trainable':True}
