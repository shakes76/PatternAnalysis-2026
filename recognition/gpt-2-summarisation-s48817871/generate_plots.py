import os
import matplotlib.pyplot as plt
import json

from numpy import append

# get all folders starting with 'gpt2_'
IMAGERY_FOLDER = "./imagery"
TO_PLOT = "to_plot.json"
VALID_LOSS_HEADER = 'validation_loss'
TRAIN_LOSS_HEADER = 'training_losses'

model_folders = []
for e in os.scandir("."):
    if e.is_dir() and e.name.startswith("gpt2_"):
        model_folders.append(e.name)

for m in model_folders:
    with open(os.path.join(m, TO_PLOT), "r") as f:
        data = json.load(f)
    plt.clf()
    fig, ax = plt.subplots(1, 1)
    fig.tight_layout()
    fig.set_size_inches(18, 10)
    x = []
    x_plus_one = []
    train_loss_mean = []
    valid_loss_mean = []
    train_loss_std = []
    valid_loss_std = []
    # keys in data are strings, and we need them to be integers
    for step in data[VALID_LOSS_HEADER].keys():
        x.append(int(step))
        x_plus_one.append(int(step)+1)
        tmean, tstd = data[TRAIN_LOSS_HEADER][step]
        vmean, vstd = data[VALID_LOSS_HEADER][step]
        train_loss_mean.append(tmean)
        train_loss_std.append(tstd)
        valid_loss_mean.append(vmean)
        valid_loss_std.append(vstd)
    ax.set_ylim(0, 1)
    #ax.set_yscale("log")
    ax.errorbar(x, train_loss_mean, yerr=train_loss_std, label="Train Loss", elinestyle="--")
    ax.errorbar(x_plus_one, valid_loss_mean, yerr=valid_loss_std, label="Validation Loss", elinestyle=":")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross Entropy Loss")
    ax.legend(loc = "upper right")
    fig.savefig(os.path.join(IMAGERY_FOLDER, f"{m}_loss.png"))
