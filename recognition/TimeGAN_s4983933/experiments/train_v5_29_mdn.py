import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset_v4 import prepare_dataset_v4
from depth_model_v5_29 import (
    ProbabilisticDepthNet,
    mixture_nll,
)

SEED = 371029
EPOCHS = 15
BATCH = 2048
LR = 0.001
THRESHOLD = 1e-5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

out = Path("outputs_v5_29_mdn")
out.mkdir(exist_ok=True)

data = prepare_dataset_v4("data")
norm = data["normalizer"]


def extract(x):
    return np.maximum(
        np.expm1(x[..., 1:40:2]), 0
    )


train = extract(
    norm.inverse_transform(
        np.asarray(data["train"], dtype=np.float64)
    )
)

val = extract(
    norm.inverse_transform(
        np.asarray(data["val"][:512], dtype=np.float64)
    )
)


def transitions(x):
    log = np.log1p(x)

    current = log[:, :-1]
    following = log[:, 1:]

    changed = (
        np.abs(np.diff(x, axis=1)) >= THRESHOLD
    ).astype(np.float32)

    return (
        current.reshape(-1, 20).astype(np.float32),
        following.reshape(-1, 20).astype(np.float32),
        changed.reshape(-1, 20),
    )


xtr, ytr, mtr = transitions(train)
xva, yva, mva = transitions(val)

# Fit normalization using TRAIN only.
mu = xtr.mean(axis=0)
sd = xtr.std(axis=0) + 1e-6

xtr = (xtr - mu) / sd
ytr = (ytr - mu) / sd

xva = (xva - mu) / sd
yva = (yva - mu) / sd

X = torch.from_numpy(xtr)
Y = torch.from_numpy(ytr)
M = torch.from_numpy(mtr)

XV = torch.from_numpy(xva)
YV = torch.from_numpy(yva)
MV = torch.from_numpy(mva)

model = ProbabilisticDepthNet(
    levels=20,
    hidden=128,
    components=3,
)

optimizer = torch.optim.Adam(
    model.parameters(), lr=LR
)


def objective(state, target, mask):
    logits, weights, means, log_scales = model(state)

    mask_loss = F.binary_cross_entropy_with_logits(
        logits, mask
    )

    per_level_nll = mixture_nll(
        target, weights, means, log_scales
    )

    # Train update-value distribution on ACTUAL
    # changes, not on copied/unchanged positions.
    value_loss = (
        per_level_nll * mask
    ).sum() / mask.sum().clamp_min(1)

    # Mixture NLL is a density loss and can be
    # negative; this is mathematically valid.
    total = mask_loss + 0.5 * value_loss

    return total, mask_loss, value_loss


best_loss = float("inf")
best_epoch = None
history = []

for epoch in range(1, EPOCHS + 1):
    model.train()

    order = torch.randperm(len(X))
    training_losses = []

    for ids in order.split(BATCH):
        loss, _, _ = objective(
            X[ids],
            Y[ids],
            M[ids]
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=5.0
        )
        optimizer.step()

        training_losses.append(float(loss.item()))

    model.eval()

    with torch.no_grad():
        val_loss, mask_loss, value_loss = objective(
            XV, YV, MV
        )

    record = {
        "epoch": epoch,
        "train_loss": float(
            np.mean(training_losses)
        ),
        "val_loss": float(val_loss),
        "val_mask_loss": float(mask_loss),
        "val_value_nll": float(value_loss),
    }
    history.append(record)

    print(
        f"Epoch {epoch:02d}"
        f" | train={record['train_loss']:.4f}"
        f" | val={record['val_loss']:.4f}"
        f" | mask={record['val_mask_loss']:.4f}"
        f" | mdn_nll={record['val_value_nll']:.4f}",
        flush=True
    )

    if record["val_loss"] < best_loss:
        best_loss = record["val_loss"]
        best_epoch = epoch

        torch.save({
            "model_state_dict": model.state_dict(),
            "mu": mu,
            "sd": sd,
            "epoch": epoch,
            "components": 3,
            "hidden": 128,
        }, out / "best_mdn.pt")

with open(out / "history.json", "w") as f:
    json.dump(history, f, indent=2)

print("\n=== V5.29 TRAINING COMPLETE ===")
print("Best epoch:", best_epoch)
print("Best validation loss:", best_loss)
print("Checkpoint:", out / "best_mdn.pt")
