import random
import torch
import torch.nn as nn
import math
import time
import numpy as np

from dataset import get_dataloaders
from modules import ResNet
from modules import ConvNeXt
from predict import evaluate


EPOCHS = 40
WARMUP_EPOCHS = 4
LR = 5e-4
WD = 0.05
LABEL_SMOOTHING = 0.05   
BATCH=64
SEED=123
CUTOFF = 0.5
DROP_PATH = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"

MIXUP_ALPHA = 0.2   
MIXUP_PROB = 0.15

import matplotlib.pyplot as plt


def plot_curves(history, best_epoch, path="curves.png"):
    """Plots loss per epoch and accuracy per epoch over train and val sets"""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, key, title in ((ax[0], "loss", "Loss per epoch"), (ax[1], "acc", "Accuracy per epoch")):
        a.plot(epochs, history[f"train_{key}"], label="train")
        a.plot(epochs, history[f"val_{key}"], label="val")
        a.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"best val (epoch {best_epoch})")
        a.set_xlabel("epoch")
        a.set_ylabel(key)
        a.set_title(title)
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mixup(x, y, alpha):
    """Synthetically combines two images to form a new image"""
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x = lam * x + (1 - lam) * x[perm]
    y = lam * y + (1 - lam) * y[perm]
    return x, y, lam, perm


def main(model_name):
    random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH)
    if model_name == 'resnet50':
        model = ResNet(num_classes=1, drop_path_rate=DROP_PATH).to(device)
        torch.backends.cudnn.enabled = False # Fixes bug with running on AMD rocm with batchnorms
    elif model_name == 'convnext':
        model = ConvNeXt(num_classes=1, drop_path_rate=DROP_PATH).to(device)
        torch.backends.cudnn.enabled = True
        print("hello")
    else:
        return
    criterion = nn.BCEWithLogitsLoss()

    # no weight decay on normal paramters
    decay, no_decay = [], []
    for p in model.parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WD},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=LR, betas=(0.9, 0.999),
    )


    steps_per_epoch = len(train_loader)
    total = EPOCHS * steps_per_epoch
    warm = WARMUP_EPOCHS * steps_per_epoch
    
    # linear warmup, then cosine decay
    def lr_lambda(step):                            
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * t))

    on_gpu = device=="cuda"
    if (on_gpu):
        print("YAY!")
    scaler = torch.amp.GradScaler("cuda", enabled=on_gpu)

    

    best_acc, best_epoch = -1.0, 0
    total = EPOCHS * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(EPOCHS):
        model.train()
        run_loss = 0
        seen = 0
        correct = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Custom label smoothing as using BCELogitsLoss
            y = y.float()
            y_orig = y.clone()
            y_smooth = y * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING

            # Randomly decides whether or not to use mixup
            use_mix = MIXUP_ALPHA > 0 and np.random.rand() < MIXUP_PROB
            if use_mix:
                x, y_smooth, lam, perm = mixup(x, y_smooth, MIXUP_ALPHA)

            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=on_gpu):
                logits = model(x).squeeze(1)
            loss = criterion(logits.float(), y_smooth)

            scaler.scale(loss).backward()
           
            scaler.step(opt)
            scaler.update()
            sched.step()

            run_loss += loss.item() * y.size(0)
            seen += y.size(0)

            if use_mix:
                # score against whichever source image dominates the mix
                target = y_orig if lam >= 0.5 else y_orig[perm]
            else:
                target = y_orig

            correct += ((logits > 0).long() == target.long()).sum().item()
        val = evaluate(model, val_loader)
        v = val["slice"]
        print(f"epoch: {epoch+1} | train loss {run_loss/seen} acc: {correct/seen} | time: {time.time()-start_time}")
        print(v)

        # tracks loss and acc over epochs
        history["train_loss"].append(run_loss / seen)
        history["val_loss"].append(val["loss"])
        history["train_acc"].append(correct / seen)
        history["val_acc"].append(v["acc"])

        if v["acc"] > best_acc:
            best_acc = v["acc"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"convnext3.pt")
        alloc = torch.cuda.max_memory_allocated() / 1024**3

    plot_curves(history, best_epoch, path="curves.png")
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"overall peak mem: {alloc:.2f} GiB allocated | {reserved:.2f} GiB reserved | {total_mem:.2f} GiB total")
    model.load_state_dict(torch.load(f"convnext3.pt"))
    print(f"best val acc {best_acc:.4f} @ epoch {best_epoch}")
    print("test:", evaluate(model, test_loader)["slice"])

if __name__ == "__main__":
    # main('resnet50')
    main('convnext')
