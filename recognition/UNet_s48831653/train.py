import torch
import torch.nn.functional as F
import torch.optim as optim
import time

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os

import modules
import dataset

def soft_dice_loss(data, mask, eps=1e-6):
    """
    Uses probability values to calculate average channel dice loss.
    data (tensor): [B, C, H, W] shape containing the model's predictions
    mask (tensor): [B, C, H, W] shape containing the true segmentation mask
    eps (Float): used for gradient stabilisation
    """
    probabilities = torch.sigmoid(data)

    # flatten to [B, C, HxW]
    pred = probabilities.view(probabilities.size(0), probabilities.size(1), -1)
    target = mask.view(mask.size(0), mask.size(1), -1)

    intersection = (pred * target).sum(dim=(0, 2)) #|X int Y|
    total = pred.sum(dim=(0, 2)) + target.sum(dim=(0, 2)) #|X| + |Y|

    # +eps is stabilisation for empty slices
    dice = (2.0 * intersection + eps) / (total + eps)
    avg_loss = 1.0 - dice.mean()
    return avg_loss

def loss(data, mask, bce_weight=0.3):
    """Combination of BCE and average dice loss used for training and validation.

    data (tensor): [B, C, H, W] shape containing the model's predictions
    mask (tensor): [B, C, H, W] shape containing the true segmentation mask
    bce_weight (Float): BCE's weighting in the overall loss function
    """
    # take sigmoid of data and compute BCE between data and mask
    bce_loss = F.binary_cross_entropy_with_logits(data, mask)
    # take sigmoid of data and compute soft dice loss for the batch of slices
    avg_dice = soft_dice_loss(data, mask)
    return bce_loss * bce_weight + (1-bce_weight) * avg_dice

def hard_dice_loss(data, mask, threshold=0.5):
    """
    Uses prediction results to intersection and union of the ground truth and
    prediction masks for each channel.
    data (tensor): [B, C, H, W] shape containing the model's predictions
    mask (tensor): [B, C, H, W] shape containing the true segmentation mask
    eps (Float): used for gradient stabilisation
    """
    with torch.no_grad():
        probabilities = torch.sigmoid(data) # convert values to [0, 1]
        predictions = (probabilities >= threshold).float() # convert values to {0, 1}

        # flatten to [B, C, HxW]
        pred = predictions.view(predictions.size(0), predictions.size(1), -1)
        target = mask.view(mask.size(0), mask.size(1), -1)

        intersection = (pred * target).sum(dim=2) #|X int Y|
        total = pred.sum(dim=2) + target.sum(dim=2) #|X| + |Y|

        return intersection, total # maintain array structure

def train_epoch(model, dataloader, device, optimiser):
    """
    Execute one epoch of training for the given model.
    """
    model.train()
    running_loss = 0.0
    total_intersection = torch.zeros(5, device=device)
    total_pixels = torch.zeros(5, device=device)

    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)

        optimiser.zero_grad()
        probabilities = model(images)

        # loss per batch
        batch_loss = loss(probabilities, masks)
        batch_loss.backward()
        optimiser.step()

        intersection, total = hard_dice_loss(probabilities, masks)
        total_intersection += intersection.sum(dim=0)
        total_pixels += total.sum(dim=0)

        # extract loss value, scale by batch size
        running_loss += batch_loss.item() * images.size(0)
    avg_running_loss = running_loss / len(dataloader.dataset) # average (BCE + Dice) loss per slice
    eps = 1e-6
    channel_dice = (2.0 * total_intersection + eps) / (total_pixels + eps)
    return avg_running_loss, channel_dice[4].item()

def validate_epoch(model, dataloader, device):
    """
    Evaluates a model prediction and compares to the ground truth with dice loss
    """
    model.eval()
    running_batch_loss = 0
    total_intersection = torch.zeros(5, device=device)
    total_pixels = torch.zeros(5, device=device)

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            probabilities = model(images)
            # batch's average (BCE + Dice) loss
            batch_loss = loss(probabilities, masks)
            running_batch_loss += batch_loss.item() * images.size(0)

            # batch's ground truth and prediction intersection and total values
            intersection, total = hard_dice_loss(probabilities, masks)
            total_intersection += intersection.sum(dim=0)
            total_pixels += total.sum(dim=0)

    # average (BCE + Dice) loss per slice
    avg_loss = running_batch_loss / len(dataloader.dataset)

    # dice
    eps = 1e-6 # eps for gradient stablisation
    channel_dice = (2.0 * total_intersection + eps) / (total_pixels + eps)
    channel_iou = (total_intersection + eps) / (total_pixels - total_intersection + eps)
    return channel_dice, channel_iou, avg_loss

def plot_loss_curves(train_loss, val_loss, train_prostate_loss, val_prostate_loss, num_epochs, lossplot_path):
    """
    Plots the training loss and validation loss over each epoch.
    """
    e = range(1, num_epochs+1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(e, train_loss, label="Training loss", color="blue")
    axes[0].plot(e, val_loss, label="Validation loss", color="orange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE + Soft Dice Loss")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Average Channel BCE + Dice Loss per Sample")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(e, train_prostate_loss, label="Training Dice", color="blue")
    axes[1].plot(e, val_prostate_loss, label="Validation Dice", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Similarity Coefficient")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Prostate Channel Dice Value per Sample")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    fig.savefig(lossplot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return

def count_parameters(model):
    """
    Print the number of model parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return

def train_and_validate(UNet=True, num_epochs=30):
    """
    Prepare the dataloaders, model, optimiser and scheduler. Trains and validates
    the model, reporting performance per epoch and overall results.
    """
    # Set up
    train_loader, val_loader, test_loader = dataset.get_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if UNet:
        model = modules.UNet_2D(in_channels=1, num_categories=5).to(device)
        checkpoint_path = "best_2DUnet.pth"
        lossplot_path = "2DUnet_loss_plot.png"
        print("Training 2D UNet...")
    else:
        model = modules.CAN_2D().to(device)
        checkpoint_path = "best_2DCAN.pth"
        lossplot_path = "2DCAN_loss_plot.png"
        print("Training 2D CAN...")
    count_parameters(model)
    optimiser = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # alters learning rate by factor 0.1 for optimiser when validation dice doesn't increase
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='max', factor=0.1, patience=4)

    train_loss = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    train_prostate_dice = np.zeros(num_epochs)
    val_prostate_dice = np.zeros(num_epochs)

    best_dice = 0.0
    total_time = 0.0

    # Train Model
    for e in range(num_epochs):
        # begin timing current epoch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()

        # training + validation
        train_loss[e], train_prostate_dice[e] = train_epoch(model, train_loader, device, optimiser)
        all_channel_dice, all_channel_iou, val_loss[e] = validate_epoch(model, val_loader, device)
        val_prostate_dice[e] = all_channel_dice[4].item()
        scheduler.step(val_prostate_dice[e])

        # End timing current epoch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start
        total_time += epoch_time

        # Report per channel dice
        print(f"Epoch {e+1}:")
        print(f"    Elapsed time = {epoch_time:.3f}")
        print(f"    DICE VALUES...")
        print(f"        C0 Background/Rectum: {all_channel_dice[0].item()}")
        print(f"        C1  Body/Soft Tissue: {all_channel_dice[1].item()}")
        print(f"        C2              Bone: {all_channel_dice[2].item()}")
        print(f"        C3           Bladder: {all_channel_dice[3].item()}")
        print(f"        C4          Prostate: {all_channel_dice[4].item()}")
        print(f"    IOU VALUES...")
        print(f"        C0 Background/Rectum: {all_channel_iou[0].item()}")
        print(f"        C1  Body/Soft Tissue: {all_channel_iou[1].item()}")
        print(f"        C2              Bone: {all_channel_iou[2].item()}")
        print(f"        C3           Bladder: {all_channel_iou[3].item()}")
        print(f"        C4          Prostate: {all_channel_iou[4].item()}\n")

        # save model if prostate dice improved
        if val_prostate_dice[e] > best_dice:
            best_dice = val_prostate_dice[e]
            torch.save(model.state_dict(), checkpoint_path)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"NEW MODEL SAVED: prostate dice = {val_prostate_dice[e]:.4f}\n")

    # Report elapsed time and peak VRAM for training + validation
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        peak_vram_gb = peak_vram_bytes / (1024.0 * 1024.0 * 1024.0)
    print(f"Average Training and Validation Time = {(total_time/num_epochs):.3f}s per epoch")
    print(f"Peak VRAM during Training and Validation = {peak_vram_gb:.4f} GB")
    print(f"Saved model's validation prostate dice = {best_dice}")

    plot_loss_curves(train_loss, val_loss, train_prostate_dice, val_prostate_dice, num_epochs, lossplot_path)
    return

def save_prediction_image(img, gt_masks, pred_masks, i, image_name, image_directory):
    """
    Save figure with original MRI, ground truth segmentation, predicted
    segmentation and the corresponding error map.
    """
    # assign colours to channels
    label_colours = ["black", "green", "brown", "cyan", "red"]
    label_cmap = ListedColormap(label_colours)

    # Retrieve map of the segmented values
    gt_map = np.argmax(gt_masks, axis=0)
    pred_map = np.argmax(pred_masks, axis=0)

    # Construct RGB error map for prostate segmentation
    gt_prostate = gt_masks[4]
    pred_prostate = pred_masks[4]
    error_map = np.zeros((*gt_map.shape, 3))
    error_map[(pred_prostate == 1) & (gt_prostate == 0)]  = [1, 0, 0] # False positive (Red)
    error_map[(pred_prostate == 0) & (gt_prostate == 1)]  = [0, 0, 1] # False negative (Blue)
    error_map[(pred_prostate == 1) & (gt_prostate == 1)] = [0, 1, 0] # True positive (Green)

    tp = np.sum((pred_prostate == 1) & (gt_prostate == 1))
    fp = np.sum((pred_prostate == 1) & (gt_prostate == 0))
    fn = np.sum((pred_prostate == 0) & (gt_prostate == 1))
    
    eps = + 1e-6
    dice = (2*tp + eps)/(2*tp + fp + fn + eps)
    iou = (tp + eps)/(tp + fp + fn + eps)

    # Construct image
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Dice={dice:.4f}, IOU={iou:.4g}")

    # original MRI slice
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"Batch {i:03d}: Input MRI Slice")
    axes[0].axis("off")

    # Ground Truth Segmentation Map
    axes[1].imshow(gt_map, cmap=label_cmap, vmin=0, vmax=4)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    #  Model Predicted Segmentation Map
    axes[2].imshow(pred_map, cmap=label_cmap, vmin=0, vmax=4)
    axes[2].set_title(f"Predicted Mask ({i:03d})")
    axes[2].axis("off")

    # Error overlay
    axes[3].imshow(img, cmap="gray")
    axes[3].imshow(error_map, alpha=0.7)
    axes[3].set_title("Error map")
    axes[3].axis("off")

    # Save image
    plt.tight_layout()
    save_path = os.path.join(image_directory, f"{image_name}_{i:03}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    return

def test(model, dataloader, device, image_name, image_directory):
    """
    Determines model predictions of the given test set, reports on and saves
    result images in the given image_directory.
    """
    model.eval()
    total_intersection = torch.zeros(5, device=device)
    total_pixels = torch.zeros(5, device=device)

    # create path to store prediction result images
    os.makedirs(image_directory, exist_ok=True)
    i = 0

    with torch.no_grad():
        for images, masks in dataloader:
            # testing
            images, masks = images.to(device), masks.to(device)
            probabilities = model(images)

            intersection, total = hard_dice_loss(probabilities, masks)
            total_intersection += intersection.sum(dim=0)
            total_pixels += total.sum(dim=0)
            preds = (torch.sigmoid(probabilities) >= 0.5).float()

            # Save result images
            for j in range(images.size(0)):
                # Retrieve relevant maps
                img = images[j, 0].cpu().numpy() if images.ndim == 4 else images[j].cpu().numpy()
                gt_masks = masks[j].cpu().numpy()
                pred_masks = preds[j].cpu().numpy()
                save_prediction_image(img, gt_masks, pred_masks, i, image_name, image_directory)
                i += 1

    eps = 1e-6 # eps for gradient stablisation
    channel_dice = (2.0 * total_intersection + eps) / (total_pixels + eps)
    channel_iou = (total_intersection + eps) / (total_pixels - total_intersection + eps)
    return channel_dice, channel_iou

if __name__ == "__main__":
    train_and_validate(UNet=True, num_epochs=20) # train 2D U-Net
    train_and_validate(UNet=False, num_epochs=20) # train 2D CAN
