import os, sys
import torch
from tqdm import tqdm #to help visualise progress

from dataset import get_dataloaders, NUM_CLASSES, PROSTATE_CLASS, pad_collate
from modules import UNet2D, DiceLoss, dice_counts

# ---- settings ----
ROOT = os.environ.get("HIPMRI_ROOT", "/Users/alyxgoller/data/HipMRI_data")  #changes per machine
BATCH_SIZE = 16
NUM_EPOCHS = 20 
LR = 5e-4
WEIGHT_DECAY = 1e-5
LR_DECAY = 0.985                       # lr = LR * LR_DECAY ** epoch (paper)
SAVE_PATH = "best_model.pth"



#set up gpu access
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("Warning: CUDA not found. Using CPU")



def train_one_epoch(model, loader, criterion, optimizer):
    """process one training epoch.

    Args:
        model (nn.Module): 2D Unet model with set criteria
        loader (DataLoader): an iterable dataset of training images and their labels
        criterion (nn.Module): Loss function which take inout logits from training and their target
        optimizer (torch.optim.adam.Adam): Optimizer for updating model weights

    Returns:
        Average DICE loss over the training epoch (total loss / number of slices processed)
    """
    model.train()  # train flag in nn.module set as TRUE = dropout2D ON
    total = 0.0
    #iterate through batches of image labels pairs for entire dataset
    for image, label in tqdm(loader, desc="train", leave=False, disable=not sys.stdout.isatty()):
        image, label = image.to(device), label.to(device)
        #Forward Pass
        loss = criterion(model(image), label)

        #Backward and Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total += loss.item() * image.size(0)           
        #.item extract number from tensor
        # * num of slices in batch to bring average found in loss back to a total

        #returning total loss / number of slices processed
    return total / len(loader.dataset)

# Validate: check training progress
@torch.no_grad()
def validate(model, loader, criterion):
    model.eval() ## train flag in nn.module set as FALSE = dropout2D OFF
    total = 0.0
    inter = torch.zeros(NUM_CLASSES, device=device)
    denom = torch.zeros(NUM_CLASSES, device=device)
    #iterate through batches of image labels pairs for entire dataset
    for image, label in tqdm(loader, desc="val", leave=False, disable=not sys.stdout.isatty()):
        image, label = image.to(device), label.to(device)

        #predict
        logits = model(image)

        #evaluate
        total += criterion(logits, label).item() * image.size(0)
        i, d = dice_counts(logits, label)
        inter += i
        denom += d
        
    dice = (2 * inter + 1e-5) / (denom + 1e-5)     # Dice per class over the whole val set
    return total / len(loader.dataset), dice.cpu()



def main():
    # --fetch data--
    train_loader, val_loader, _ = get_dataloaders(ROOT, batch_size=BATCH_SIZE)

    

    # --Set up model and modules for training--
    
    #unet model: for train, validate and test
    model = UNet2D(in_channels=1, num_classes=NUM_CLASSES).to(device)
    
    #loss module function: for weight update (gradient descent minimising loss) and performance evaluation
    criterion = DiceLoss(epsilon=1)
    
    #optimizer object: set up weights and biases(model.para), initial learning rate and weight decay 
    # - in order for optimizer to update values at end of each batch during training.
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    #learning rate rule set - decrease exponetially per epoch - can be changed if recommended
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: LR_DECAY ** epoch)

    # --Run training and evaluation--
    best = 0.0
    for epoch in range(1, NUM_EPOCHS + 1): # use this range to ensure epoch values are correct for scheduler

        #train, validate and update learning rate
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, dice = validate(model, val_loader, criterion)
        scheduler.step()

        #show progress and track best prostate DICE value
        prostate = dice[PROSTATE_CLASS].item()
        print(f"Epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} "
              f"| prostate Dice {prostate:.4f} | all classes {[round(d, 3) for d in dice.tolist()]}")

        if prostate > best:
            best = prostate
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  saved new best model (prostate Dice {best:.4f})")

# ===================== QUICK TEST: delete this whole block when finished =====================
def initial_test(n_train=2000, n_val=400, epochs=50, max_minutes=15, workers=4, batch_size=16):
    """Short run on a random subset of the full HPC data.
    Run with:  python train.py quick
    Stops after max_minutes; saves to quick_test.pth (does not touch best_model.pth)."""
    import time
    from torch.utils.data import Subset, DataLoader


    train_loader, val_loader, _ = get_dataloaders(ROOT, batch_size=batch_size, num_workers=workers)

    
    g = torch.Generator().manual_seed(0)
    def subset(loader, n, shuffle):
        ds = loader.dataset
        idx = torch.randperm(len(ds), generator=g)[:min(n, len(ds))].tolist()
        return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=shuffle, num_workers=workers, collate_fn=pad_collate)

    train_loader = subset(train_loader, n_train, True)
    val_loader = subset(val_loader, n_val, False)
    print(f"quick test: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val slices, "
          f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}", flush=True)

    model = UNet2D(in_channels=1, num_classes=NUM_CLASSES).to(device)
    criterion = DiceLoss(epsilon=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: LR_DECAY ** e)

    start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, dice = validate(model, val_loader, criterion)
        scheduler.step()
        epoch_time = time.time() - t0
        print(f"Epoch {epoch:3d} ({epoch_time:.0f}s) | train loss {train_loss:.4f} | val loss {val_loss:.4f} "
              f"| prostate Dice {dice[PROSTATE_CLASS].item():.4f} "
              f"| all classes {[round(d, 3) for d in dice.tolist()]}", flush=True)
        if (time.time() - start) + epoch_time > max_minutes * 60:
            print(f"Stopping: another epoch would pass the {max_minutes}-minute limit.", flush=True)
            break

    torch.save(model.state_dict(), "quick_test.pth")


# =============================================================================================


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quick":    # delete these two lines too
        initial_test()                                  # when removing the quick test
    else:
        main()

