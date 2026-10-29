import torch
import torch.nn as nn
import numpy as np
import os

import torch_directml

from modules import *
from dataset import get_oasis_dataloaders

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

DATA_DIR = os.path.join(PROJECT_ROOT, "OASIS")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(MODELS_DIR, "training_log.md")

BATCH_SIZE = 30 # 32 cuts it a little too close.
EPOCHS = 10 # Number of epochs to be run at once. Still saves the checkpoints and logs between epochs.
LEARNING_RATE = 5e-5

# IMPORTANT: CHANGE THIS VALUE TO EITHER "can" or "unet". Sets the model to be run for testing.
MODEL_NAME = "unet"

if MODEL_NAME.lower() == "unet":
    SAVE_PATH = os.path.join(MODELS_DIR, "best_unet_model.pth")
    CHECKPOINT_PATH = os.path.join(MODELS_DIR, "checkpoint_unet.pth")
else:
    SAVE_PATH = os.path.join(MODELS_DIR, "best_can_model.pth")
    CHECKPOINT_PATH = os.path.join(MODELS_DIR, "checkpoint_can.pth")

def calculate_dice_score(outputs, targets, num_classes=4, smoothing=1e-6) -> float:    
    """
    Computes macro-averaged Dice score across active classes.
    """
    
    # Ensure variables are on CPU to stay immune to DirectML backend bugs.
    if outputs.device.type != 'cpu':
        outputs = outputs.cpu()
    if targets.device.type != 'cpu':
        targets = targets.cpu()
    
    predicted_classes = torch.argmax(outputs, dim=1)
    
    # One-hot encode predictions and targets: [B, H, W] -> [B, H, W, C] -> [B, C, H, W]
    prediction_one_hot = torch.nn.functional.one_hot(predicted_classes, num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Calculate overlap and cardinality per class across batch, height, and width
    intersection = (prediction_one_hot * target_one_hot).sum(dim=(0, 2, 3))
    cardinality = prediction_one_hot.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))
    
    # Compute dice per class with smoothing
    dice_per_class = (2.0 * intersection + smoothing) / (cardinality + smoothing)

    return dice_per_class.mean().item()

def train_one_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # --- NAN GUARD --- 
        # Had a lot of issues with gradient explosion so this was added
        if torch.isnan(loss):
            print(f"WARN: NaN loss detected at batch {batch_idx+1}. Skipping batch to prevent corruption.")
            optimizer.zero_grad()
            continue
        # -----------------
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, foreach=False)
        optimizer.step()

        running_loss += loss.item() * batch_size
        total_samples += batch_size
                
        # Only print every 20 batches to minimize synchronous overhead
        if (batch_idx + 1) % 20 == 0:
            print(f"Batch [{batch_idx+1}/{len(dataloader)}] | Loss: {loss.item():.4f}")
        
    total_samples = len(dataloader.dataset)
    
    dice = calculate_dice_score(outputs, targets, num_classes=num_classes)
    return running_loss / total_samples, dice

def validate_one_epoch(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * images.size(0)
            
            # Move to cpu for this stuff. critical. Lots of lost hair over this.
            outputs_cpu = outputs.detach().cpu()
            targets_cpu = targets.detach().cpu()
            
            # DICE. Not included in the training loop because it slows us down.
            dice = calculate_dice_score(outputs_cpu, targets_cpu, num_classes=num_classes)
            running_dice += dice * batch_size
            
            total_samples += batch_size
            
            print(f"Val Batch [{batch_idx+1}/{len(dataloader)}] | Loss: {loss.item():.4f} | Batch Dice: {dice:.4f}")
                
        total_samples = len(dataloader.dataset)
        return running_loss / total_samples, running_dice / total_samples
    
def evaluate_test_set(model, test_loader, criterion, device, num_classes=4):
    model.eval() # Not necessary currently but just in case I call this sometime elsewhere I want it here.
    
    print("\nRunning final test set evaluation...")
    test_loss, test_dice = validate_one_epoch(model, test_loader, criterion, device, num_classes) 
    print(f"Final Test Loss: {test_loss:.4f} | Final Test Dice: {test_dice:.4f}")
      
def save_checkpoint(epoch, model, optimizer, scheduler, path=CHECKPOINT_PATH, best_dice_val=0.0):
    """Saves a complete snapshot of the model to be resumed later. Prevents the need for long compute sessions, enabling incremental improvement of the model."""
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_dice_val': best_dice_val
    }
    torch.save(checkpoint, path)
    print(f"--> Checkpoint successfully saved to {path}")
    
def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):   
    """
    Loads the checkpoint states into the model, optimizer, and scheduler. 
    Returns the next epoch index to start training from and current best dice score on the validation set.
    """
    
    # No checkpoint. Start from scratch. Epoch 0, 1.0 Dice score.
    if not os.path.exists(path):
        print("No checkpoint found. Starting fresh from Epoch 0!")
        return 0, 0
    
    # Load the checkpoint
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location='cpu')
    
    # Inject the checkpoint into the model.
    model.load_state_dict(checkpoint['model_state'])
    
    # Restore optimizer
    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
    # Restore learning rate scheduler position
    if scheduler is not None and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        
    best_dice_val = checkpoint.get('best_dice_val', 0.0)
    start_epoch = checkpoint['epoch'] + 1
    
    print(f"Resuming seamlessly! Next epoch will be Epoch {start_epoch + 1}.")
    return start_epoch, best_dice_val

def log_to_markdown_table(epoch, model_name, train_loss, train_dice, val_loss, val_dice, path=LOG_FILE_PATH):
    """Appends training and validation metrics to a Markdown table log file."""
    file_exists = os.path.exists(path)
    
    with open(path, "a", encoding="utf-8") as f:
        # If file doesn't exist, write the Markdown header and divider
        if not file_exists:
            f.write("# Model Training and Validation Log\n\n")
            f.write("| Epoch | Model | Train Loss | Train Dice | Val Loss | Val Dice |\n")
            f.write("|:---:|:---|:---:|:---:|:---:|:---:|\n")        
        # Append the current epoch's row
        f.write(f"| {epoch + 1} | {model_name} | {train_loss:.4f} | {train_dice:.4f} | {val_loss:.4f} | {val_dice:.4f} |\n")    
    
def get_device():
    """Fetches the optimal device to run the computation on, checks for rangpur CUDA gpu, if not, decides between my local AMD gpu and the cpu"""
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"INFO: Oh! We are on Rangpur! AMAZING: {device}")
    else:
        try:
            # index 1 is my graphics card and 0 is my integrated graphics.
            device = torch_directml.device(1) # Finally... a hardware acceleration fix that isn't rocm?
            print(f"INFO: Home AMD GPU Active via DirectML: {device}")
        except (ImportError, RuntimeError):
            device = torch.device("cpu")
            print(f"WARN: Falling back to CPU: {device}")
            
    return device

def select_model(in_channels, num_classes, C, device):
    """Selects model to run based on the value of MODEL_NAME"""
    can = ContextAggregationModule(in_channels=1, num_classes=num_classes, C=64).to(device=device)
    unet = Simple2DUNet(in_channels=1, num_classes=4, C=64).to(device=device)
    model = None
    
    # Flexible model selection.
    if MODEL_NAME.lower() == "unet":
        print("Initializing Raw 2D U-Net Baseline Model...")
        model = unet
    else:
        print("Initializing 2D Context Aggregation Network (CAN)...")
        model = can
        
    return model
    
    
def main():    
    # Make the correct device selection according to what hardware this code is being run on
    device = get_device()
                
    # Data loaders
    train_loader, val_loader, test_loader, num_classes = get_oasis_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    
    # Based on user configuration, select either the 2D Can model or the 2D unet model.
    model = select_model(in_channels=1, num_classes=num_classes, C=64, device=device)
    
    # Optimization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4, foreach=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    
    # Load the checkpoint for iterative improvement
    start_epoch, best_dice_val = load_checkpoint(CHECKPOINT_PATH, model, optimizer=optimizer, scheduler=scheduler, device=device)
    end_epoch = start_epoch + EPOCHS
    print(f"Starting Training Loop from Epoch {start_epoch + 1}")
                
    for epoch in range(start_epoch, end_epoch):
        
        # Train and Validate
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device, num_classes)
        val_loss, val_dice = validate_one_epoch(model, val_loader, criterion, device, num_classes)
        
        scheduler.step(val_dice)
        
        # LOGGING
        log_to_markdown_table(epoch, MODEL_NAME, train_loss, train_dice, val_loss, val_dice)
        print(f"Epoch [{epoch}/{end_epoch - 1}] finished | "
              f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
                
        # Save best model checkpoint as the best model for this run. Just in case the checkpoint explodes :(
        if val_dice > best_dice_val:
            best_dice_val = val_dice
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"--> Saved new best model checkpoint to {SAVE_PATH} (Dice: {best_dice_val:.4f})")
            
        save_checkpoint(epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler, path=CHECKPOINT_PATH, best_dice_val=best_dice_val)

    print("Training complete!")
    
    print("Beginning Testing!")
    evaluate_test_set(model, test_loader, criterion, device, num_classes=4)
    
if __name__ == "__main__":
    main()