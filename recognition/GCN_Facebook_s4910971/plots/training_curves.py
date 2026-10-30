"""Plot training and validation curves."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)


RUNS = {
    "MLP": ROOT / "training_results" / "mlp_20261030_000036",
    "GCN": ROOT / "training_results" / "gcn_20261030_002758",
}


METRICS = [
    ("loss", "Loss"),
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro F1"),
]

fig, axes = plt.subplots(2, 3, figsize=(12, 6))


for row, model_name in enumerate(RUNS):
    run_folder = RUNS[model_name]


    history_path = run_folder / "history.csv"
    with history_path.open(newline="", encoding="utf-8") as f:
        data = list(csv.DictReader(f))


    epochs = []
    for row_data in data:
        epochs.append(int(row_data["epoch"]))

    # Read the selected checkpoint epoch.
    best_epoch = 0
    params_path = run_folder / "params.txt"
    with params_path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("Best epoch:"):
                best_epoch = int(line.split(":")[1])

    print(f"  Best epoch = {best_epoch}")

    for col, (key, title) in enumerate(METRICS):
        train_values = []
        val_values = []
        for row_data in data:
            train_values.append(float(row_data[f"train_{key}"]))
            val_values.append(float(row_data[f"val_{key}"]))

        ax = axes[row, col]
        ax.plot(epochs, train_values, label="Train")
        ax.plot(epochs, val_values, label="Validation")
        ax.axvline(
            best_epoch,
            color="black",
            linestyle="--",
            label=f"Best epoch: {best_epoch}",
        )
        ax.set_title(f"{model_name}: {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()

fig.tight_layout()
fig.savefig(OUT / "training_curves.png")
plt.close(fig)
