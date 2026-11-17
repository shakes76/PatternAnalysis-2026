import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import random
import matplotlib.pyplot as plt

try:
    import torch_directml
    HAS_DIRECTML = True
except ImportError:
    HAS_DIRECTML = False

from modules import ContextAggregationModule, Simple2DUNet
from dataset import get_oasis_dataloaders

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DATA_DIR = os.path.join(SCRIPT_DIR, "OASIS")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
VIS_DIR = os.path.join(SCRIPT_DIR, "visualisations")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(MODELS_DIR, "training_log.md")

BATCH_SIZE = 30 # 32 cuts it a little too close.
LEARNING_RATE = 5e-5
NUM_CLASSES = 4

def calculate_dice_score(outputs, targets, num_classes=NUM_CLASSES, smoothing=1e-6) -> float:    
    """
    Per-class DICE score over the batch, then averaged over tissue classes only (no background) with absent classes treated as NaN
    so they do not inflate the score. Mirrors predict.py::evaluate() so training, 
    validation and test Dice are on the same scale as teh reported tissue mean.
        
    detach() before cpu() is deliberate: DirectML's .cpu() on an autograd-connected tensor 
    can silently corrupt values, and we never back-prop through this metric anyways.
    """
    
    outputs = outputs.detach().cpu()
    targets = targets.detach().cpu()   
    
    predicted_classes = torch.argmax(outputs, dim=1)
    
    # One-hot encode predictions and targets: [B, H, W] -> [B, H, W, C] -> [B, C, H, W]
    prediction_one_hot = torch.nn.functional.one_hot(predicted_classes, num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Calculate overlap and cardinality per class across batch, height, and width
    intersection = (prediction_one_hot * target_one_hot).sum(dim=(0, 2, 3))
    cardinality = prediction_one_hot.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))

    absent = cardinality == 0
    dice_per_class = (2.0 * intersection + smoothing) / (cardinality + smoothing)
    dice_per_class = dice_per_class.masked_fill(absent, float("nan"))
    
    # Exclude background (0)
    tissue_dice = dice_per_class[1:]
    if torch.isnan(tissue_dice).all():
        return 0.0
    return torch.nanmean(tissue_dice).item()

def train_one_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    total_samples = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images, targets = images.to(device), targets.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Gradient explosion issues were encountered during testing. This is some mitigation
        if torch.isnan(loss):
            print(f"WARN: NaN loss detected at batch {batch_idx+1}. Skipping batch to prevent corruption.")
            optimizer.zero_grad()
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, foreach=False)
        optimizer.step()

        running_loss += loss.item() * batch_size
        running_dice += calculate_dice_score(outputs, targets, num_classes) * batch_size
        total_samples += batch_size
                
        # Only print every 20 batches to minimize synchronous overhead
        if (batch_idx + 1) % 20 == 0:
            print(f"Batch [{batch_idx + 1}/{len(dataloader)}] | "
                  f"loss={loss.item():.4f} | "
                  f"epoch-avg loss={running_loss / total_samples:.4f} | "
                  f"epoch-avg dice={running_dice / total_samples:.4f}")
                    
    if total_samples == 0:
        return 0.0, 0.0
    return running_loss / total_samples, running_dice / total_samples

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
            
            running_loss += loss.item() * batch_size
            running_dice += calculate_dice_score(outputs, targets, num_classes=num_classes) * batch_size
            total_samples += batch_size
            
            print(f"Val Batch [{batch_idx + 1}/{len(dataloader)}] | "
                  f"loss={loss.item():.4f}")
                
        if total_samples == 0:
            return 0.0, 0.0
        return running_loss / total_samples, running_dice / total_samples
    
def evaluate_test_set(model, test_loader, criterion, device, num_classes=4):
    model.eval()
    print("\nRunning final test set evaluation...")
    test_loss, test_dice = validate_one_epoch(model, test_loader, criterion, device, num_classes) 
    print(f"Final Test Loss: {test_loss:.4f} | "
          f"Final Test Dice (tissue mean, excl. background): {test_dice:.4f}")
      
def save_checkpoint(epoch, model, optimizer, scheduler, path, best_dice_val=0.0):
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
    
    # No checkpoint. Start from scratch.
    if not os.path.exists(path):
        print("No checkpoint found. Starting fresh from Epoch 0")
        return 0, 0.0
    
    # Load the checkpoint
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    
    # Restore optimizer and learning rate scheduler position
    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state']) 
    if scheduler is not None and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        
    best_dice_val = checkpoint.get('best_dice_val', 0.0)
    start_epoch = checkpoint['epoch'] + 1
    
    print(f"Resuming seamlessly! Next epoch will be Epoch {start_epoch}.")
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
    
    
def plot_training_metrics(model_name, log_path=LOG_FILE_PATH):
    """
    Parse the Markdown training log and write loss + Dice curves for a single
    model to visualisations/training_metrics_{model_name}.png
    """
    if not os.path.exists(log_path):
        print(f"WARN: {log_path} not found; skipping training-metrics plot.")
        return
    
    epochs: list[int] = []
    train_loss: list[float] = []
    train_dice: list[float] = []
    val_loss: list[float] = []
    val_dice: list[float] = []
        
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            # Skip blanks, the "|:---:|" divider, and the header row.
            if not line.startswith("|") or line.startswith("|:") or "Epoch" in line:
                continue
            
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) != 6:
                continue
            
            try:
                ep          = int(cells[0])
                row_model   = cells[1]
                t_loss      = float(cells[2])
                t_dice      = float(cells[3])
                v_loss      = float(cells[4])
                v_dice      = float(cells[5])
            except ValueError:
                continue
            
            if row_model.lower() != model_name.lower():
                continue
            
            epochs.append(ep)
            train_loss.append(t_loss)
            train_dice.append(t_dice)
            val_loss.append(v_loss)
            val_dice.append(v_dice)

    if not epochs:
        print(f"WARN: No log rows found for model '{model_name}'; skipping plot.")
        return

    fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, train_loss, marker="o", label="Train")
    ax_loss.plot(epochs, val_loss,   marker="s", label="Validation")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(f"{model_name.upper()} — Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_dice.plot(epochs, train_dice, marker="o", label="Train")
    ax_dice.plot(epochs, val_dice,   marker="s", label="Validation")
    ax_dice.axhline(0.90, color="red", linestyle="--", alpha=0.6, label="Target 0.90")
    ax_dice.set_xlabel("Epoch")
    ax_dice.set_ylabel("Dice")
    ax_dice.set_title(f"{model_name.upper()} — Dice")
    ax_dice.legend()
    ax_dice.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(VIS_DIR, f"training_metrics_{model_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"--> Training metrics plot saved to: {out_path}")
    
    
def get_device():
    """Fetches the optimal device to run the computation on, checks for rangpur CUDA gpu, if not, decides between my local AMD gpu and the cpu"""
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"INFO: Rangpur CUDA GPU active: {device}")
        
    elif HAS_DIRECTML:
        try:
            # index 1 is my graphics card and 0 is my integrated graphics.
            device = torch_directml.device(1)
            print(f"INFO: Home AMD GPU Active via DirectML: {device}")
        except Exception as e:
            device = torch.device("cpu")
            print(f"WARN: DirectML failed ({e}), falling back to CPU.")  
    else:
        device = torch.device("cpu")
        print(f"WARN: Falling back to CPU: {device}")
            
    return device

def get_peak_vram_mb():
    """
    Return peak GPU VRAM in MiB since the last torch.cuda.reset_peak_memory_stats()
    call, or None when the backend exposes no memory introspection.

    CUDA: torch.cuda.max_memory_allocated() reports the high-water mark of the
    caching allocator.
    DirectML / CPU: torch_directml has no allocator API, so this returns None and
    the caller should skip the print. This is a backend limitation, not a missing
    measurement — see README Resource Profiling for the Task-Manager fallback used
    on DirectML.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return None

def set_seed(seed: int):
    """
    Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    Caveats:
      - DirectML (AMD GPU) is not bit-exact. Two runs with the same seed can
        still diverge slightly. The seed controls initialisation and batch
        order, not hardware-level FP reduction order.
      - Resuming from a checkpoint does NOT restore RNG state — only model,
        optimizer, and scheduler state are saved. For a seeded run to be
        meaningful, start from a fresh checkpoint.
      - CUDA determinism knobs only apply when torch.cuda.is_available().
    """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def select_model(model_name, in_channels, num_classes, C, device):
    """Selects model to run based on the value of MODEL_NAME"""
    if model_name.lower() == "unet":
        print("Initializing Raw 2D U-Net Baseline Model...")
        return Simple2DUNet(in_channels=1, num_classes=num_classes, C=64).to(device=device)
    else:
        print("Initializing 2D Context Aggregation Network (CAN)...")
        return ContextAggregationModule(in_channels=1, num_classes=num_classes, C=64).to(device=device)
        
    
def main():    
    parser = argparse.ArgumentParser(description="Train OASIS Segmentation Models")    
    parser.add_argument("--model", type=str, default="can", choices=["can", "unet"], help="Model to train (can or unet)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to run")
    parser.add_argument("--seed", type=int, default=67, help="Random seed for reproducibility (default: 67)")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any existing checkpoint and train from scratch")
    args = parser.parse_args()
    
    # Seed.
    set_seed(args.seed)
    print(f"INFO: Global seed set to {args.seed}")
    
    model_name = args.model
    save_path = os.path.join(MODELS_DIR, f"best_{model_name}_model.pth")
    checkpoint_path = os.path.join(MODELS_DIR, f"checkpoint_{model_name}.pth")
        
    # Make the correct device selection according to what hardware this code is being run on
    device = get_device()
    
    # Reset the peak VRAM measurements so it covers this run only
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
                
    # Data loaders
    train_loader, val_loader, test_loader, num_classes = get_oasis_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    
    # Based on user configuration, select either the 2D Can model or the 2D unet model.
    model = select_model(model_name=model_name, in_channels=1, num_classes=num_classes, C=64, device=device)
    
    # Optimization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4, foreach=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    
    # Load the checkpoint for iterative improvement, unless a fresh run is requested.
    if args.no_resume:
        if os.path.exists(checkpoint_path):
            print(f"INFO: --no-resume set; ignoring existing checkpoint at {checkpoint_path}.")
        start_epoch, best_dice_val = 0, 0.0
    else:            
        start_epoch, best_dice_val = load_checkpoint(checkpoint_path, model, optimizer=optimizer, scheduler=scheduler, device=device)
    
    end_epoch = start_epoch + args.epochs
    print(f"Starting Training Loop from Epoch {start_epoch + 1}")
                
    for epoch in range(start_epoch, end_epoch):
        
        # Train and Validate
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device, num_classes)
        val_loss, val_dice = validate_one_epoch(model, val_loader, criterion, device, num_classes)
        
        scheduler.step(val_dice)
        
        # LOGGING
        log_to_markdown_table(epoch, model_name, train_loss, train_dice, val_loss, val_dice)
        print(f"Epoch [{epoch}/{end_epoch - 1}] finished | "
              f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
                
        # Save best model checkpoint as the best model for this run.
        if val_dice > best_dice_val:
            best_dice_val = val_dice
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model checkpoint to {save_path} (Dice: {best_dice_val:.4f})")
            
        save_checkpoint(epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler, path=checkpoint_path, best_dice_val=best_dice_val)

    plot_training_metrics(model_name)
    
    print("Training complete!")
    
    print("Beginning Testing!")
    print(f"\nLoading best model from {save_path} for final testing...")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        evaluate_test_set(model, test_loader, criterion, device, num_classes=num_classes)
    else:
        print("ERROR: No best model found. Skipping test evaluation.")
        
    peak_vram = get_peak_vram_mb()
    if peak_vram is not None:
        print(f"INFO: Peak CUDA VRAM (whole session): {peak_vram:.1f} MiB")
    else:
        print("INFO: Peak VRAM not available — DirectML/CPU backend exposes no "
              "memory API. See README Resource Profiling for the DirectML "
              "measurement method.")
    
if __name__ == "__main__":
    main()