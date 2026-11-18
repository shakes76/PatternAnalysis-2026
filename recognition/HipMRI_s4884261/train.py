# Containing the source code for training, validating, testing, and saving your model.
# The model should be imported from modules.py and the data loader should be imported from dataset.py.
# Make sure to plot the losses and metrics during training.

import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import get_pairs, split_pairs, get_loaders, check_shapes, augment
from modules import Improved3DUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("WARNING: CUDA unavailable")

# find fastest conv algo for input shape
torch.backends.cudnn.benchmark = True
# tensor float 32 on newer GPUs, speedup barely any accuracy loss
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# hyperparams
# data_root = "/home/twig/Desktop/Uni/Year 3 - Sem 2/COMP3710/Proj/data"
data_root = "/home/groups/comp3710/HipMRI_Study_open/"
num_classes = 6
class_names = ("Background", "Body", "Bone", "Bladder", "Rectum", "Prostate")
base_filters = 16
batch_size = 2 # only able to fit a few whole volumes on GPU at once
num_epochs = 30
lr = 5e-4
weight_decay = 1e-5
lr_gamma = 0.985 # lr multiplied by this every epoch

class DiceLoss(nn.Module):
    # multiclass soft dice loss

    def __init__(self, num_classes, smooth=1e-5):
        # store num_classes and the small smoothing constant
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, target):
        # logits: (batch, num_classes, D, H, W) raw network output
        # target: (batch, D,H,W) integer labels
        # softmax the logits. one-hot encode target. Per class dice = 2*sum(pred*target)/(sum(pred)+sum(target)).
        # return 1 - mean dice over classes
        probs = F.softmax(logits.float(), dim=1)
        one_hot = F.one_hot(target, self.num_classes).permute(0, 4, 1, 2, 3).float() # (batch, D, H, W, B) -> (batch, C, D, H, W)
        dims = (0, 2, 3, 4) # sum over batch and voxels, keeping each class separate
        intersection = (probs * one_hot).sum(dims)
        total = probs.sum(dims) + one_hot.sum(dims)
        dice = (2*intersection+self.smooth)/(total+self.smooth)
        return 1 - dice.mean()# every class counts equally, so small prostate class isnt drowned out by background or body

def dice_score(logits, target, num_classes):
    # Hard Dice per class for reporting (not traiing), computes dice for each class
    preds = logits.argmax(dim=1) # most likely class per voxel
    classes = torch.arange(num_classes, device=preds.device).view(-1, 1, 1, 1, 1) # compares againstevery class at once, same as onehot
    pred_one_hot = preds.unsqueeze(0) == classes # (C, batch, D, H, W)
    target_one_hot = target.unsqueeze(0) == classes
    dims = (1, 2, 3, 4)
    intersection = (pred_one_hot & target_one_hot).sum(dims).float()
    total = (pred_one_hot.sum(dims) + target_one_hot.sum(dims)).float()
    # a class missing from both the prediction and label counts as perfect instead of Nan
    return torch.where(total > 0, 2 * intersection / total.clamp(min=1), torch.ones_like(total))

def train_one_epoch(model, loader, loss_fn, optimizer, device, scaler):
    # Train one pass over loader. Move to device -> forward -> loss -> backward -> step. Return avg loss
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long() # onehot needs int64
        images, labels = augment(images, labels)
        optimizer.zero_grad(set_to_none=True)

        # mized precision, halves the memory
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)
        loss = loss_fn(logits, labels)

        # scale the loss up so fp16 gradients dont underflow, then unscale before the optimizer step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.detach()
    return (running_loss / len(loader)).item()

def validate(model, loader, loss_fn, device, num_classes):
    # evaluate on validation set (model.eval, torch.nograd). Avg loss, and avg dice per class
    model.eval()
    total_loss = 0.0
    total_dice = torch.zeros(num_classes, device=device)
    with torch.no_grad():
        for images, labels in loader: # batch size 2, so dice is per volume then averaged
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(images)
            total_loss += loss_fn(logits, labels).item()
            total_dice += dice_score(logits, labels, num_classes)
    return total_loss / len(loader), (total_dice/ len(loader)).cpu()

def plot_curves(train_losses, val_losses, val_dice, save_path="training_curves.json"):
    # plot loss and dice against epoch and save the figure. If on rangpur, print values out so can plot later.
    val_dice = torch.stack(val_dice) # (epochs, classes)
    plot_data = {
        "epoch": list(range(1, len(train_losses) + 1)),
        "train_loss": [round(v, 4) for v in train_losses],
        "val_loss": [round(v, 4) for v in val_losses],
        "val_dice": {name: [round(v, 4) for v in val_dice[:, c].tolist()] for c, name in enumerate(class_names)},
    }
    with open(save_path, 'w') as file:
        json.dump(plot_data, file)

    print(f"\nPlot data (also saved in {os.path.basename(save_path)})")
    print("Loss plot: x=epoch, lines = train_loss and val_loss (Diceloss)")
    print("DSC plot: x=epoch, one line per class in val_dice, dashed horizontal line at the 0.9 target")
    for key, values in plot_data.items():
        print(f"{key}={values}")

def main(model=None, name="unet3d", root=data_root, epochs=num_epochs, out_dir="."):
    """
        Set up and run training

        1. BUild the train an validation dataloaders
        2. Build improved unet 3d (or use the model passed in, UNet_2D.py passes its 2D model) and move to gpu
        3. loss = diceloss; optimizer = adam(lr=5e-4, weight_decay=1e-5); scheduler=exponentialLr(gamma=0.985).
        4. For each epoch: train_one_epoch->validate-> scheduler.step()->save model if val dice is best so far.
        5. Plot curves at end.
    """
    model_path = os.path.join(out_dir, f"{name}_hipmri.pt")

    # dataloaders
    start = time.time()
    pairs = get_pairs(root)
    if not pairs:
        print("What?")
        return

    check_shapes(pairs)
    splits = split_pairs(pairs)
    del splits["test"] # test set is only loaded for the final evaluation
    loaders = get_loaders(splits, batch_size)
    train_loader, val_loader = loaders["train"], loaders["val"]
    print(f"Train volumes: {len(splits['train'])} | validation volumes: {len(splits['val'])}")
    print(f"Dataset setup time: {time.time()-start}")

    # model
    if model is None:
        model = Improved3DUNet(in_channels=1, num_classes=num_classes, base_filters=base_filters)
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")


    # loss, optimiser, scheduler
    loss_fn = DiceLoss(num_classes)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=lr_gamma)
    scaler = torch.amp.GradScaler("cuda")

    # train
    train_losses, val_losses, val_dice = [], [], []
    best_min_dice = -1 # below any real dsc so the first epoch always saved
    torch.cuda.reset_peak_memory_stats(device) # so the peak only counts training
    start = time.time()
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimiser, device, scaler)
        val_loss, dice = validate(model, val_loader, loss_fn, device, num_classes)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dice.append(dice)

        # keeps the epoch where the worst class is best, since every label needs > 0.9
        if dice.min().item() > best_min_dice:
            best_min_dice = dice.min().item()
            torch.save(model.state_dict(), model_path)

        dice_str = " ".join(f"{name}: {d:.4f}" for name, d in zip(class_names, dice))
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val DSC {dice_str} | Peak VRAM: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB | Elapsed: {time.time() - start:.1f}s")

    elapsed = time.time() - start
    print(f"Training took {str(elapsed)} secs or {str(elapsed / 60)} mins in total")
    print(f"best validation min class DSC: {best_min_dice} saved to {model_path}")
    print(f"Peak VRAM allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB | Peak VRAM reserved: {torch.cuda.max_memory_reserved(device) / 1024**3:.2f} GB")

    # curves
    plot_curves(train_losses, val_losses, val_dice, os.path.join(out_dir, f"{name}_plot_data.json"))

if __name__ == "__main__":
    main()
