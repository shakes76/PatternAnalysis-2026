"""Train the MLP and save results."""

import csv
import platform
from copy import deepcopy
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset import load_graph
from experiments.baseline.model import MLP



# Local paths.
DATA = ROOT / "data" / "facebook.npz"
PARAM_FILE = ROOT / "experiments" / "baseline" / "search_results" / "best_params.txt"
OUT = ROOT / "training_results"


# True: search parameters. False: PARAMS.
USE_SEARCH = True
PARAMS = {
    "hidden_dim": 64,
    "learning_rate": 0.01,
    "dropout": 0.2,
    "weight_decay": 0.0005,
}

SEED = 0
EPOCHS = 300



def metrics(out, y):
    """Calculate loss, accuracy and macro F1."""
    pred = out.argmax(dim=1)
    loss = F.cross_entropy(out, y).item()
    acc = (pred == y).float().mean().item()
    f1 = f1_score(y.cpu(), pred.cpu(), labels=[0, 1, 2, 3],
                  average="macro", zero_division=0)
    return loss, acc, f1



def main():
    """Train and save the best validation checkpoint."""
    params = PARAMS.copy()
    source = "Custom values in experiments/baseline/train.py"
    if USE_SEARCH:
        source = str(PARAM_FILE.relative_to(ROOT))
        for line in PARAM_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                name, value = line.split("=", 1)
                name = name.strip()
                if name in params:
                    params[name] = float(value)
        params["hidden_dim"] = int(params["hidden_dim"])



    # Load the graph and fixed split.
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_graph(DATA, seed=SEED).to(device)
    run = OUT / f"mlp_{datetime.now():%Y%m%d_%H%M%S}"
    run.mkdir(parents=True)



    # Build model and optimiser.
    model = MLP(data.num_node_features, params["hidden_dim"], 4,
                params["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["learning_rate"],
                           weight_decay=params["weight_decay"])
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)




    # Save run settings.
    with (run / "params.txt").open("w", encoding="utf-8") as f:
        f.write(f"Source: {source}\n")

        for name, value in params.items():
            f.write(f"{name}: {value}\n")

        f.write(f"\nEpochs: {EPOCHS}\n")
        f.write(f"Seed: {SEED}\n")
        f.write("Batch: full batch\n")
        f.write("Optimiser: Adam\n")
        f.write("Loss: cross-entropy (unweighted)\n")

        f.write("\nLayers: 3\nInput: node features only\n")
        f.write("Split: 70/10/20 (train/val/test)\n")
        f.write(f"Train nodes: {data.train_mask.sum().item()}\n")
        f.write(f"Val nodes: {data.val_mask.sum().item()}\n")
        f.write(f"Test nodes: {data.test_mask.sum().item()}\n")

        f.write("\nCheckpoint: best validation macro F1; ties use lower loss\n")
        f.write("Early stopping: no\n")

    # Measure time and GPU memory.
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = perf_counter()



    # Train and record each epoch.
    best_f1 = -1
    best_loss = float("inf")
    columns = ["epoch", "train_loss", "train_accuracy", "train_macro_f1",
               "val_loss", "val_accuracy", "val_macro_f1", "epoch_seconds"]
    with (run / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for epoch in range(1, EPOCHS + 1):
            step = perf_counter()
            model.train()
            opt.zero_grad()
            out = model(data.x)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            opt.step()


            # Evaluate with dropout off.
            model.eval()
            with torch.no_grad():
                out = model(data.x)
                train = metrics(out[data.train_mask], data.y[data.train_mask])
                val = metrics(out[data.val_mask], data.y[data.val_mask])

            # Keep the best validation checkpoint.
            if val[2] > best_f1 or (val[2] == best_f1 and val[0] < best_loss):
                best_f1 = val[2]
                best_loss = val[0]
                best_epoch = epoch
                best_weights = deepcopy(model.state_dict())

            if device == "cuda":
                torch.cuda.synchronize()
            seconds = perf_counter() - step
            writer.writerow([epoch, *train, *val, seconds])
            f.flush()

    seconds = perf_counter() - start




    # Save weights and model settings.
    torch.save({
        "weights": best_weights,
        "params": params,
        "seed": SEED,
        "epochs": EPOCHS,
        "best_epoch": best_epoch,
    }, run / "model.pt")

    with (run / "params.txt").open("a", encoding="utf-8") as f:
        f.write(f"Best epoch: {best_epoch}\n")



    # Save cost and versions; MB = 1,000,000 bytes.
    memory = 0.0
    name = platform.processor()
    if device == "cuda":
        memory = torch.cuda.max_memory_allocated() / 1_000_000
        name = torch.cuda.get_device_name()

    with (run / "train_info.txt").open("w", encoding="utf-8") as f:
        f.write(f"Device: {device} ({name})\n")
        f.write(f"Trainable parameters: {count}\n")
        f.write(f"Training and validation time (seconds): {seconds:.4f}\n")
        f.write(f"Peak PyTorch GPU allocation (MB): {memory:.2f}\n")
        f.write(f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}\n")
        f.write(f"CUDA build: {torch.version.cuda}\n")
        for package in ["torch-geometric", "numpy", "scikit-learn"]:
            f.write(f"{package}: {version(package)}\n")


if __name__ == "__main__":
    main()
