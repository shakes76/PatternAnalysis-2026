import torch


class MRINormalise:
    """Normalise MRI image values, excluding pixels under threshold"""

    def __init__(self, thresh: float = 0.02):
        self.thresh = thresh

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        mask = image > self.thresh

        if not mask.any():
            return image

        fg = image[mask]

        mean = fg.mean()
        std = fg.std(unbiased=False)
        image = (image - mean) / std
        image = image.masked_fill(~mask, 0.0).clamp_min(1e-6)

        return image
    