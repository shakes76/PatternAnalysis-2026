"""Explicit model construction shared by training and checkpoint consumers."""
import math

from modules import ImprovedUNet2D, UNet2D


MODEL_TYPES = {'UNet2D': UNet2D, 'ImprovedUNet2D': ImprovedUNet2D}
ARCHITECTURES = tuple(MODEL_TYPES)


def build_model(architecture, model_config):
    """Build a whitelisted architecture with explicit, validated channel widths."""
    if not isinstance(architecture, str) or architecture not in MODEL_TYPES:
        raise ValueError(f'Unsupported architecture: {architecture!r}')
    if not isinstance(model_config, dict):
        raise ValueError('model_config must be a dictionary')
    required = {'in_channels', 'num_classes', 'base_channels'}
    allowed = required | ({'dropout'} if architecture == 'ImprovedUNet2D' else set())
    if not required.issubset(model_config) or set(model_config) - allowed:
        raise ValueError('Missing or unsupported model_config fields')
    for key in required:
        if type(model_config[key]) is not int or model_config[key] < 1:
            raise ValueError(f'model_config {key} must be a positive integer')
    if 'dropout' in model_config:
        dropout = model_config['dropout']
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError('model_config dropout must be finite and in range [0, 1)')
    return MODEL_TYPES[architecture](**model_config)


def model_from_checkpoint(saved):
    """Restore saved architecture/width strictly; never guess from state keys.

    Early baseline checkpoints omitted architecture/model_config from the nested
    config dictionary. Their explicit top-level fields remain authoritative.
    When nested fields exist, they must agree with the top-level metadata.
    """
    if not isinstance(saved, dict):
        raise ValueError('Checkpoint must be a dictionary')
    for key in ('architecture', 'model_config', 'model_state'):
        if key not in saved:
            raise ValueError(f'Checkpoint is missing {key}')
    config = saved.get('config', {})
    if not isinstance(config, dict):
        raise ValueError('Checkpoint config must be a dictionary')
    for key in ('architecture', 'model_config'):
        if key in config and config[key] != saved[key]:
            raise ValueError(f'Checkpoint {key} metadata mismatch')
    model = build_model(saved['architecture'], saved['model_config'])
    model.load_state_dict(saved['model_state'], strict=True)
    return model
