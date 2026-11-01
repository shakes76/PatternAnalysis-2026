"""Smoke test for BaselineCNN and SiameseEncoder: output shape, parameter count and gradient flow."""
import torch
import torch.nn as nn

from modules import BaselineCNN, SiameseEncoder

torch.manual_seed(0)
model = BaselineCNN()
x = torch.randn(4, 3, 224, 224)
logits = model(x)
print("output shape:", tuple(logits.shape))
assert logits.shape == (4, 1), "expected one logit per image"

trainable = [p for p in model.parameters() if p.requires_grad]
print("trainable parameters:", sum(p.numel() for p in trainable))

labels = torch.tensor([0.0, 1.0, 0.0, 1.0]).unsqueeze(1)
loss = nn.BCEWithLogitsLoss()(logits, labels)
loss.backward()
print("loss:", round(loss.item(), 4))
with_grad = sum(p.grad is not None for p in trainable)
print(f"parameters with a gradient: {with_grad} / {len(trainable)}")
assert with_grad == len(trainable), "some parameters received no gradient"

# ---- SiameseEncoder ----
print("\nSiameseEncoder")
encoder = SiameseEncoder()
emb = encoder(x)
print("output shape:", tuple(emb.shape))
assert emb.shape == (4, 128), "expected a 128-d embedding per image"

norms = emb.norm(dim=1)
print("embedding norms:", [round(n, 4) for n in norms.tolist()])
assert torch.allclose(norms, torch.ones(4), atol=1e-5), "embeddings are not unit length"

enc_params = [p for p in encoder.parameters() if p.requires_grad]
print("trainable parameters:", sum(p.numel() for p in enc_params))

# Contrastive-style loss on pairs (0,1) and (2,3): pull pair 0 together, push pair 1 apart.
dist = (emb[:2] - emb[2:]).pow(2).sum(dim=1).sqrt()
print("pair distances (must be within [0, 2]):", [round(v, 4) for v in dist.tolist()])
assert (dist >= 0).all() and (dist <= 2 + 1e-5).all()
loss = dist[0] ** 2 + torch.clamp(1.0 - dist[1], min=0) ** 2
loss.backward()
with_grad = sum(p.grad is not None and p.grad.abs().sum() > 0 for p in enc_params)
print(f"parameters with a gradient: {with_grad} / {len(enc_params)}")
assert with_grad == len(enc_params), "some parameters received no gradient"
