import argparse
import numpy as np
import torch
import random
import time
from pathlib import Path

from dataset import SEED, LABELS, get_dataloader_2d, get_dataloader_3d, verify_splits
from modules import dsc_per_class, DiceCELoss, ImprovedUnet, ImprovedUnet3D
from predict import metrics_to_string, compute_metrics
from utils import VRAMTracker, plot_training, write_report

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)


# ======================================================================================================================
# =                                             Hyperparameter definitions                                             =
# ======================================================================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IN_CHANNELS = 1 # Greyscale
OUT_CHANNELS = len(LABELS)

model_dir = Path("trained_model/")
model_dir.mkdir(parents=True, exist_ok=True)

# Clear cache and reset peak stats before training
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


# ======================================================================================================================
# =                                                    Train Model                                                     =
# ======================================================================================================================
def train_model(
        model: ImprovedUnet | ImprovedUnet3D,
        train_dataloader: torch.utils.data.DataLoader,
        validate_dataloader: torch.utils.data.DataLoader,
        num_epochs: int,
        criterion: DiceCELoss,
        optimiser,
        scheduler,
        scaler,
        model_type: str
):
    training_losses = list()
    validation_losses = list()
    training_scores = list()
    validation_scores = list()
    predictions_all, targets_all = list(), list()

    train_report = str()
    validate_report = str()
    vram_tracker = VRAMTracker()

    print("Training...")
    start_time = time.time()
    training_times = list()
    training_subtime_cum = start_time

    for epoch in range(num_epochs):
        training_subtime = time.time()
        epoch_loss = validation_loss = 0
        losses_per_batch = torch.Tensor().to(device)
        losses_per_batch_temp = torch.Tensor().to(device)

        # ==============================================================================================================
        # =                                               Training Loop                                                =
        # ==============================================================================================================
        model.train()
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (images, masks) in enumerate(train_dataloader):
            # print(f"\tBatch {batch_idx + 1}/{len(train_dataloader)}")
            images = images.to(device)
            masks = masks.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad()

            vram_tracker.update()

            losses_per_batch = torch.cat(
                [
                    losses_per_batch,
                    torch.mean(dsc_per_class(outputs, masks), dim=0).reshape(-1, len(LABELS))
                ]
            )
            epoch_loss = epoch_loss + loss.item()

        train_report = train_report + f"Mean DSC per class at epoch {epoch + 1}/{num_epochs}\n"
        losses_per_batch = torch.mean(losses_per_batch, dim=0)
        for label_idx, label in enumerate(LABELS):
            train_report = train_report + f"{label}: {losses_per_batch[label_idx]:.3f}\n"
        train_report = train_report + f"\n"

        avg_loss = epoch_loss / len(train_dataloader)
        training_losses.append(avg_loss)
        training_scores.append(1 - avg_loss)
        out_str = f"\tAverage Loss: {avg_loss:.3f}"

        training_subtime_cum = time.time() - training_subtime_cum
        training_subtime = time.time() - training_subtime
        training_times.append(training_subtime)

        # ==============================================================================================================
        # =                                              Validation Loop                                               =
        # ==============================================================================================================
        model.eval()
        with torch.no_grad():
            for batch_idx, (val_images, val_masks) in enumerate(validate_dataloader):
                val_images = val_images.to(device)
                val_masks = val_masks.to(device)

                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = model(val_images)
                    val_loss = criterion(outputs, val_masks)
                validation_loss = validation_loss + val_loss.item()
                losses_per_batch_temp = torch.cat(
                    [
                        losses_per_batch_temp,
                        torch.mean(dsc_per_class(outputs, val_masks), dim=0).reshape(-1, len(LABELS))
                    ]
                )
                predictions_all.append(outputs.argmax(1).to(torch.uint8).cpu())
                targets_all.append(val_masks.argmax(1).to(torch.uint8).cpu())

            validate_report = validate_report + (f"Mean DSC per class for validation set at "
                                                                         f"epoch {epoch + 1}/{num_epochs}\n")
            losses_per_batch_temp = torch.mean(losses_per_batch_temp, dim=0)
            for label_idx, label in enumerate(LABELS):
                validate_report = validate_report + f"{label}: {losses_per_batch_temp[label_idx]:.3f}\n"
            validate_report = validate_report + f"\n"

        scheduler.step()
        avg_validation_loss = validation_loss / len(validate_dataloader)
        validation_losses.append(avg_validation_loss)
        validation_scores.append(1 - avg_validation_loss)
        out_str = out_str + (f" | Validation Loss: {avg_validation_loss:.3f}\t"
                             f"Time: {training_subtime:.3f}s\tTime Cum: {training_subtime_cum}")
        training_subtime_cum = time.time() - training_subtime_cum
        print(out_str)

        del outputs, loss
        torch.cuda.empty_cache()

    max_peak_allocated, max_peak_reserved, mean_peak_allocated, mean_peak_reserved = vram_tracker.evaluate()
    resource_report = (
            f"Average time per epoch: {np.mean(training_times):.3f}s\n"
            f"Max peak VRAM allocated: {max_peak_allocated}:.3f MB\n" 
            f"Max peak VRAM reserved: {max_peak_reserved} MB\n" 
            f"Mean peak VRAM allocated: {mean_peak_allocated} MB\n"
            f"Mean peak VRAM reserved: {mean_peak_reserved} MB\n"
    )
    print(resource_report)


    # ==================================================================================================================
    # =                                          Save model & Training Plots                                           =
    # ==================================================================================================================
    outpath = model_dir / fr"unet_improved_{model_type}.pth"
    torch.save(model.state_dict(), outpath)
    print(f"Saved checkpoint to {outpath}")
    print("")

    outpath = model_dir / fr"{model_type} UNET Segmentation Training Curve.png"
    plot_training(outpath, training_losses, validation_losses, training_scores, validation_scores)

    metrics = compute_metrics(torch.cat(predictions_all), torch.cat(targets_all))
    table = metrics_to_string(metrics)

    # Write to file
    # 1. Training set per epoch
    # 2. Validation set per epoch
    # 3. Testing set per batch and mean across set
    unet_report_path = model_dir / rf"unet_report_{model_type}.txt"
    report_metrics = (
        train_report,
        validate_report,
        table,
        resource_report,
    )
    write_report(unet_report_path, report_metrics)


# ======================================================================================================================
# =                                                     Main Guard                                                     =
# ======================================================================================================================
def main(argsv):
    print("Using device:", device.type)
    model_type = argsv.model
    num_epochs = argsv.epochs

    criterion = DiceCELoss(w=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    if model_type == "3D":
        dropout = 0.31
        learning_rate = 0.000434
        model = ImprovedUnet3D(IN_CHANNELS, OUT_CHANNELS, dropout)
        model = model.to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=num_epochs, eta_min=4.9e-8)

        train_dataloader = get_dataloader_3d("train")
        validate_dataloader = get_dataloader_3d("validate")
        train_model(model, train_dataloader, validate_dataloader, num_epochs, criterion, optimiser, scheduler, scaler, model_type)
    elif model_type == "2D":
        dropout = 0.23
        learning_rate = 0.0001
        model = ImprovedUnet(IN_CHANNELS, OUT_CHANNELS, dropout_p = dropout)
        model.to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=num_epochs, eta_min=1.2e-5)

        train_dataloader = get_dataloader_2d("train")
        validate_dataloader = get_dataloader_2d("validate")
        train_model(model, train_dataloader, validate_dataloader, num_epochs, criterion, optimiser, scheduler, scaler, model_type)
    else:
        raise ValueError(f"Invalid model type: {model_type}! Must be '3D' or '2D'.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type = str, required = True)
    ap.add_argument("--epochs", type = int, required = False, default = 80)
    args = ap.parse_args()
    verify_splits(args.model)   # We don't need to care about this since we're re-training with (new) splits anyway
    main(args)