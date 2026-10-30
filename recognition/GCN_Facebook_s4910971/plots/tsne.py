"""Plot test embeddings with ground-truth colours."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
RUNS = {
    "MLP": ROOT / "training_results" / "mlp_20261030_000036",
    "GCN": ROOT / "training_results" / "gcn_20261030_002758",
}
SPLIT = "test"
SEED = 0

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, name in zip(axes, RUNS):
    # Read hidden features and true labels.
    path = RUNS[name] / "prediction" / "embeddings.pt"
    data = torch.load(path, map_location="cpu", weights_only=True)
    mask = data[f"{SPLIT}_mask"]
    x = data["x"][mask].numpy()
    y = data["y"][mask].numpy()

    # Reduce hidden features to two dimensions.
    xy = TSNE(perplexity=30, random_state=SEED).fit_transform(x)



    for i, label in enumerate(data["classes"]):
        selected = y == i
        ax.scatter(xy[selected, 0], xy[selected, 1], s=3,label=label)

    ax.set_title(f"{name}: last hidden layer")
    ax.set_xticks([])
    ax.set_yticks([])

fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4)
fig.tight_layout(rect=[0, 0.08, 1, 1])
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "tsne.png")
plt.close(fig)