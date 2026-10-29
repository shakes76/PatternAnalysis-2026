"""Print trainable parameter counts for the three models (README resource table)."""
from train import build_model

for name in ("unet2d", "improved2d", "improved3d"):
    model = build_model(name, 16)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name:12s} {n:>12,d} parameters ({n / 1e6:.2f} M)")
