import torch
import copy
import os
import time
import argparse
import torch.optim as optim
from modules import ResNet18, GFNet, get_criterion, load_checkpoint, get_device, WEIGHT, LABEL_SMOOTHING
from dataset import data_loaders

# Global variable definitions for file paths
PATH_LATEST_MODEL = "Models/latest"
PATH_BEST_MODEL = "Models/best"

# Hyperparameters
EPOCHS = 250
WARMUP_EPOCHS = 10
PATIENCE = 40
ETA_MIN = 1e-6
LR = 0.0002           # Stabilized learning rate to prevent optimizer overshoot and loss volatility
WD = 0.05
EMA_DECAY = 0.995     # Exponential moving average decay to smooth validation loss and enhance test generalization

class ModelEMA:
    """
    Exponential Moving Average (EMA) of model weights.
    Maintains smooth parameter trajectories to eliminate high-variance validation loss spikes
    and improve generalization on the test dataset.
    """
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
            for ema_buf, buf in zip(self.ema_model.buffers(), model.buffers()):
                ema_buf.copy_(buf)

class EarlyStopping:
    """
    Early stopping monitor to halt training when validation metric stops improving,
    preventing overfitting and restoring the optimal model weights.
    Monitors validation accuracy (with validation loss as tie-breaker).
    """
    def __init__(self, patience=PATIENCE, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_acc = -1.0
        self.best_loss = torch.inf
        self.early_stop = False
        self.best_weights = None
        self.best_epoch = 0

    def __call__(self, val_acc, val_loss, model, epoch):
        improved = (val_acc > self.best_acc + self.min_delta) or (
            abs(val_acc - self.best_acc) <= self.min_delta and val_loss < self.best_loss
        )
        if improved:
            if self.verbose:
                print(f" -> Validation metric improved (Acc: {self.best_acc:.4f} -> {val_acc:.4f}, Loss: {self.best_loss:.4f} -> {val_loss:.4f}). Stashing best weights.")
            self.best_acc = val_acc
            self.best_loss = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f" -> EarlyStopping counter: {self.counter}/{self.patience} (Best Val Acc: {self.best_acc:.4f}, Val Loss: {self.best_loss:.4f} at epoch {self.best_epoch})")
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print(f"Restored best model weights from epoch {self.best_epoch} (Val Acc: {self.best_acc:.4f}, Val Loss: {self.best_loss:.4f}).")

def save_checkpoint(path, model, information):
    """
    Saves model state dictionary and hyperparameter dictionary locally.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "information": information,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Successfully saved checkpoint to '{path}'.")

def train(
    model,
    criterion,
    optimizer,
    scheduler,
    device, 
    train_loader, 
    val_loader,
    latest_path=None,
    best_path=None,
    best_val_accuracy=-1.0
):
    """
    Trains the provided model on the ADNI dataset with Automatic Mixed Precision (AMP),
    gradient clipping, early stopping, and CUDA/A100 hardware acceleration.
    """
    print(f"\nBeginning training on device '{device}' for {EPOCHS} epochs...")

    early_stopping = EarlyStopping(patience=PATIENCE)
    ema = ModelEMA(model, decay=EMA_DECAY)
    use_cuda = device.type == "cuda"
    device_type = "cuda" if use_cuda else "cpu"
    best_val_loss = torch.inf

    # Configure CUDA / Nvidia A100 performance optimizations
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

    # Use native bfloat16 on Ampere A100 architecture for high dynamic range & throughput; fallback to float16
    amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_cuda and amp_dtype == torch.float16))

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()

    train_losses = []
    val_losses = []
    epochs = []

    for epoch in range(1, EPOCHS + 1):      
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device, non_blocking=use_cuda)
            batch_y = batch_y.to(device, non_blocking=use_cuda)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_cuda):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Update Exponential Moving Average of weights
            ema.update(model)

            train_loss += loss.item() * batch_X.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # --- Validation Phase (Evaluated on smoothed EMA Model to eliminate loss spikes) ---
        eval_model = ema.ema_model
        eval_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device, non_blocking=use_cuda)
                batch_y = batch_y.to(device, non_blocking=use_cuda)
                with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_cuda):
                    outputs = eval_model(batch_X)
                    loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_X.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        epochs.append(epoch)

        # Step Learning Rate Scheduler
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | LR: {current_lr:.6f} | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        # Save latest model at each epoch (saving the smoothed EMA weights for evaluation)
        if latest_path is not None:
            current_time = time.time() - start_time
            latest_information = consolidate_information(
                model=eval_model,
                val_accuracy=epoch_val_acc,
                criterion=criterion,
                optimiser=optimizer,
                scheduler=scheduler,
                device=device,
                training_time=current_time,
                losses=(train_losses, val_losses, epochs)
            )
            save_checkpoint(path=latest_path, model=eval_model, information=latest_information)

        # Update best model if validation metric improves (prioritizing accuracy and loss calibration)
        improved = (epoch_val_acc > best_val_accuracy + 1e-4) or (
            abs(epoch_val_acc - best_val_accuracy) <= 1e-4 and epoch_val_loss < best_val_loss
        )
        if best_path is not None and improved:
            print(f" -> Validation metric improved (Acc: {best_val_accuracy:.4f} -> {epoch_val_acc:.4f}, Loss: {best_val_loss:.4f} -> {epoch_val_loss:.4f}). Saving best EMA model to '{best_path}'.")
            best_val_accuracy = epoch_val_acc
            best_val_loss = epoch_val_loss
            current_time = time.time() - start_time
            best_information = consolidate_information(
                model=eval_model,
                val_accuracy=epoch_val_acc,
                criterion=criterion,
                optimiser=optimizer,
                scheduler=scheduler,
                device=device,
                training_time=current_time,
                losses=(train_losses, val_losses, epochs)
            )
            save_checkpoint(path=best_path, model=eval_model, information=best_information)

        # Check Early Stopping
        if early_stopping is not None:
            early_stopping(epoch_val_acc, epoch_val_loss, eval_model, epoch)
            if early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch}. Stopping training early.")
                break

    if early_stopping is not None:
        early_stopping.restore_best_weights(model)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_training_time = time.time() - start_time

    print("Training finished.")
    print(f"Total Training Time: {total_training_time:.2f} seconds ({total_training_time / 60:.2f} minutes)")
    return model, total_training_time, (train_losses, val_losses, epochs)

def consolidate_information(model, val_accuracy, criterion, optimiser, scheduler, device, training_time, losses):
    information = {
        "model": str(model),
        "epochs": EPOCHS,
        "criterion": f"Cross-Entropy: (weight: {WEIGHT}), (label smoothing: {LABEL_SMOOTHING})",
        "optimiser": "ADAMW: " + str(optimiser.state_dict()["param_groups"]),
        "val_accuracy": val_accuracy,
        "scheduler": f"Linear Warmup ({WARMUP_EPOCHS} epochs) + Cosine Annealing (T_max: {max(EPOCHS - WARMUP_EPOCHS, 1)}, eta_min: {ETA_MIN})",
        "device": str(device),
        "training_time": training_time,
        "losses": losses
    }
    return information

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet-18 model and GFNet model training")
    parser.add_argument("--model", default="RN18", choices=["RN18", "GFNET"], help="Model to train: 'RN18' or 'GFNET'")
    args = parser.parse_args()
    mode = args.model

    print("Selecting Model...")
    if mode == "RN18":
        model = ResNet18()
    else:
        model = GFNet()
    print(f"Model is {mode}...")

    print("Constructing paths...")
    latest_path = PATH_LATEST_MODEL + "_" + str(mode) + ".pt"
    best_path = PATH_BEST_MODEL + "_" + str(mode) + ".pt"

    print("Constructing data loaders...")
    train_loader, val_loader, _ = data_loaders()

    print("Constructing optimiser, scheduler, and criterion...")
    if hasattr(model, "get_parameter_groups"):
        param_groups = model.get_parameter_groups(weight_decay=WD)
        optimiser = optim.AdamW(params=param_groups, lr=LR)
    else:
        optimiser = optim.AdamW(params=model.parameters(), lr=LR, weight_decay=WD)

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimiser, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(EPOCHS - WARMUP_EPOCHS, 1), eta_min=ETA_MIN
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimiser, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS]
    )
    criterion = get_criterion()

    print("Obtaining device...")
    device = get_device()
    model.to(device=device)

    best_model_info = load_checkpoint(best_path, device=device)
    if best_model_info is not None:
        best_model_accuracy = float(best_model_info.get("information", dict()).get("val_accuracy", -1.0))
        print(f"Existing best model found with validation accuracy: {best_model_accuracy:.4f}.")
    else:
        best_model_accuracy = -1.0
        print("No existing best model found.")

    trained_model, training_time, losses = train(
        model=model,
        criterion=criterion,
        optimizer=optimiser,
        scheduler=scheduler,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        latest_path=latest_path,
        best_path=best_path,
        best_val_accuracy=best_model_accuracy
    )
    print("Training run completed. Latest and best models have been saved.")
