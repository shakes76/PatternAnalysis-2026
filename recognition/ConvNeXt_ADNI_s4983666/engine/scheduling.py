"""Optional epoch-level warmup/cosine schedule; legacy constant LR is unchanged."""
import math


def learning_rate(epoch: int, epochs: int, base: float, name: str,
                  warmup: int = 2, min_ratio: float = 0.01) -> float:
    """Return the declared learning rate before one epoch's optimizer updates."""
    if name not in ('constant', 'warmup_cosine', 'cosine') or not 1 <= epoch <= epochs:
        raise ValueError('Invalid schedule name or epoch.')
    if name == 'constant':
        return base
    if name == 'cosine':
        warmup = 0
    if type(warmup) is not int or not 0 <= warmup < epochs or not 0 <= min_ratio <= 1:
        raise ValueError('Warmup must be below epoch cap and min LR ratio must be in [0, 1].')
    if epoch <= warmup:
        return base * epoch / warmup
    decay_epochs = epochs - warmup
    progress = (epoch - warmup - 1) / max(1, decay_epochs - 1)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
