import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re

from dataset import *
from modules import UNet2D, UNet3D
from metrics import calculate_dice_iou


def train_2d(epochs=5, batch_size=1, learning_rate=0.001):
    """
    Train the 2D U-Net baseline on the HipMRI training slices.
    NOT USED ANYMORE
    """

    # Use GPU if available
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Training dataset
    train_dataset = HipMRI2DDataset(
        SLICE_DATA_DIR / "keras_slices_train",
        SLICE_DATA_DIR / "keras_slices_seg_train"
    )
    
    # Validation dataset
    val_dataset = HipMRI2DDataset(
        SLICE_DATA_DIR / "keras_slices_validate",
        SLICE_DATA_DIR / "keras_slices_seg_validate"
    )

    # Load training data in batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    # Load validationdata in batches
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Create model
    model = UNet2D(
        in_channels=1,
        num_classes=6
    ).to(device)

    # Multi-class segmentation loss
    criterion = nn.CrossEntropyLoss()

    # Optimiser
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )
    
    # Keep track of best validation loss
    best_val_loss = float("inf")
            
    # Training epochs
    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for images, masks, mask_one_hot in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}"
        ):

            images = images.to(device)
            masks = masks.to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, masks)

            # Backpropagation
            loss.backward()

            # Update model weights
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        #store validation Dice and IoU values
        dice_scores = {
            class_id: []
            for class_id in range(1,6)
        }
        
        iou_scores = {
            class_id: []
            for class_id in range(1,6)
        }

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                # forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                #Choose the predicted class at each pixel
                predictions = torch.argmax(outputs, dim=1)
                
                # Calc Dice and IoU
                results = calculate_dice_iou(predictions, masks, num_classes=6)
                
                for class_id, scores in results.items():
                    dice_scores[class_id].append(scores["dice"])
                    iou_scores[class_id].append(scores["iou"])
        average_val_loss = val_loss / len(val_loader)
        
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Training Loss: {average_loss:.4f}"
            f"Validation Loss: {average_val_loss:.4f}"
        )
        
        #display average validation metrics
        for class_id in range(1, 6):
            if dice_scores[class_id]:
                average_dice = sum(dice_scores[class_id]) / len(dice_scores[class_id])
                average_iou = sum(iou_scores[class_id]) / len(iou_scores[class_id])
                
                print(
                    f"Class: {class_id}"
                    f"Dice: {average_dice:.4f}"
                    f"IoU: {average_iou:.4f}"
                )
                
        # Save model with the best validation loss
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            torch.save(model.state_dict(), "best_2d_unet.pth")
            print ("best model saved")
    return model

def train_2d_aligned(epochs=20, slice_batch_size=8, learning_rate=0.001):
    """
    Train a 2D U-Net on the same patients and preprocessed
    MRI volumes used by the 3D U-Net.
    
    Assisted by ChatGPT 6 model 6.0 for plotting
    """

    torch.manual_seed(67)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Use the same patient split as the 3D model
    pairs = find_volume_pairs()
    train_patients, val_patients, test_patients = split_patients(pairs)

    assert set(train_patients).isdisjoint(val_patients)
    assert set(train_patients).isdisjoint(test_patients)
    assert set(val_patients).isdisjoint(test_patients)

    train_pairs = [pair for pair in pairs if pair["patient"] in train_patients]

    val_pairs = [pair for pair in pairs if pair["patient"] in val_patients]

    # Reuse the 3D preprocessing to keep inputs consistent
    train_dataset = HipMRI3DDataset(train_pairs)
    val_dataset = HipMRI3DDataset(val_pairs)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"Training volumes: {len(train_dataset)}")
    print(f"Validation volumes: {len(val_dataset)}")

    # Use the existing 2D U-Net architecture
    model = UNet2D(in_channels=1, num_classes=6).to(device)

    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {parameter_count:,}")

    # Same training class weights as the weighted 3D U-Net
    class_counts = torch.tensor(
        [87254835, 53391820, 4918408, 855653, 217516, 162408],
        dtype=torch.float32,
        device=device
    )

    class_weights = torch.sqrt(class_counts.sum() / class_counts)
    class_weights = class_weights / class_weights.mean()

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")

    train_history = []
    val_history = []
    dice_history = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):

        # Training
        model.train()
        running_loss = 0.0
        train_slices = 0

        for images, masks, _ in tqdm(
            train_loader,
            desc=f"2D Epoch {epoch + 1}/{epochs}"
        ):

            # Convert one volume into a batch of 2D slices
            # [1, 1, D, H, W] -> [D, 1, H, W]
            images = images[0].permute(1, 0, 2, 3)
            masks = masks[0]

            images = images.to(device)
            masks = masks.to(device)

            # Process the slices in small batches
            for start in range(0, len(images), slice_batch_size):
                x = images[start:start + slice_batch_size]
                y = masks[start:start + slice_batch_size]

                optimizer.zero_grad()

                outputs = model(x)
                loss = criterion(outputs, y)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len(x)
                train_slices += len(x)

        average_train_loss = running_loss / train_slices

        # Validation
        model.eval()

        validation_loss = 0.0
        validation_slices = 0

        # Count actual and predicted voxels for each class
        confusion = torch.zeros((6, 6), dtype=torch.long, device=device)

        with torch.inference_mode():

            for images, masks, _ in val_loader:

                images = images[0].permute(1, 0, 2, 3)
                masks = masks[0]

                images = images.to(device)
                masks = masks.to(device)

                for start in range(0, len(images), slice_batch_size):
                    x = images[start:start + slice_batch_size]
                    y = masks[start:start + slice_batch_size]

                    outputs = model(x)
                    loss = criterion(outputs, y)

                    validation_loss += loss.item() * len(x)
                    validation_slices += len(x)

                    predictions = outputs.argmax(dim=1)

                    counts = torch.bincount(
                        (y * 6 + predictions).reshape(-1),
                        minlength=36
                    ).reshape(6, 6)

                    confusion += counts

        average_val_loss = validation_loss / validation_slices

        # Calculate Dice across all validation voxels
        dice_scores = []

        for class_id in range(1, 6):
            intersection = confusion[class_id, class_id].item()

            actual_count = confusion[class_id, :].sum().item()
            predicted_count = confusion[:, class_id].sum().item()

            if actual_count + predicted_count > 0:
                dice = (
                    2 * intersection / (actual_count + predicted_count)
                )

                dice_scores.append(dice)

                print(f"Class {class_id} Dice: {dice:.4f}")

        mean_dice = sum(dice_scores) / len(dice_scores)

        train_history.append(average_train_loss)
        val_history.append(average_val_loss)
        dice_history.append(mean_dice)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {average_train_loss:.4f} | "
            f"Val Loss: {average_val_loss:.4f} | "
            f"Mean Dice: {mean_dice:.4f}"
        )

        # Save the model with the best validation loss
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss

            torch.save(
                model.state_dict(),
                PROJECT_DIR / "best_2d_unet_aligned.pth"
            )

            print("Best aligned 2D model saved")

    # Save training graphs for the README
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epoch_numbers = range(1, epochs + 1)

    axes[0].plot(epoch_numbers, train_history, label="Training")
    axes[0].plot(epoch_numbers, val_history, label="Validation")
    axes[0].set_title("2D U-Net Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Weighted Cross-Entropy")
    axes[0].legend()

    axes[1].plot(epoch_numbers, dice_history)
    axes[1].set_title("2D U-Net Validation Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Foreground Dice")
    axes[1].set_ylim(0, 1)

    fig.tight_layout()

    output_dir = PROJECT_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    fig.savefig(
        output_dir / "2d_aligned_training_curves.png",
        dpi=180
    )

    plt.close(fig)

    if device.type == "cuda":
        peak_memory = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
        )
        print(f"Peak training GPU memory: {peak_memory:.3f} GB")

    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Aligned 2D training complete")

    return model


def train_3d(epochs=5, batch_size=1, learning_rate=0.001):
    """
    Train the 3D U-Net on the HipMRI MRI volumes.
    """

    # Use GPU if available
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Find all matching MRI and segmentation volumes
    pairs = find_volume_pairs()

    # Split patients into training, validation and testing groups
    train_patients, val_patients, test_patients = split_patients(pairs)

    # Only use volumes belonging to training patients
    train_pairs = [pair for pair in pairs if pair["patient"] in train_patients]

    print(f"Training volumes: {len(train_pairs)}")
    
    # Create 3D training dataset
    train_dataset = HipMRI3DDataset(train_pairs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Use separate patients for validation
    val_pairs = [pair for pair in pairs if pair["patient"] in val_patients]
    
    # Create validation dataset and loader
    val_dataset = HipMRI3DDataset(val_pairs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Validation volumes: {len(val_dataset)}")

    # Create 3D U-Net
    model = UNet3D(in_channels=1, num_classes=6).to(device)
    
    # Training voxel counts for each class
    class_counts = torch.tensor(
        [87254835, 53391820, 4918408, 855653, 217516, 162408],
        dtype=torch.float32,
        device=device
    )

    # Give less common classes greater importance
    class_weights = torch.sqrt(class_counts.sum() / class_counts)
    class_weights = class_weights / class_weights.mean()

    # Weighted cross-entropy loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Keep track of the best validation loss
    best_val_loss = float("inf")

    # Training epochs
    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for images, masks, mask_one_hot in tqdm(
            train_loader,
            desc=f"3D Epoch {epoch + 1}/{epochs}"
        ):

            images = images.to(device)
            masks = masks.to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(
                outputs,
                masks
            )

            # Backpropagation
            loss.backward()

            # Update model weights
            optimizer.step()

            running_loss += loss.item()

        average_loss = (
            running_loss
            / len(train_loader)
        )

        print(
            f"3D Epoch {epoch + 1}/{epochs} "
            f"Training Loss: {average_loss:.4f}"
        )
        
        # Validation
        model.eval()
        val_loss = 0.0

        # Store Dice and IoU for each foreground class
        dice_scores = {class_id: [] for class_id in range(1, 6)}
        iou_scores = {class_id: [] for class_id in range(1, 6)}

        # Gradients are not needed during validation
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                # Forward pass
                outputs = model(images)

                # Calculate validation loss
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Get the predicted label for each voxel
                predictions = torch.argmax(outputs, dim=1)

                # Calculate Dice and IoU for labels 1 to 5
                results = calculate_dice_iou(
                    predictions, masks, num_classes=6
                )

                for class_id, scores in results.items():
                    dice_scores[class_id].append(scores["dice"])
                    iou_scores[class_id].append(scores["iou"])

        average_val_loss = val_loss / len(val_loader)

        print(f"Validation Loss: {average_val_loss:.4f}")

        # Display average Dice and IoU for each class
        for class_id in range(1, 6):
            if dice_scores[class_id]:
                average_dice = (
                    sum(dice_scores[class_id]) / len(dice_scores[class_id])
                )
                average_iou = (
                    sum(iou_scores[class_id]) / len(iou_scores[class_id])
                )

                print(
                    f"Class {class_id}: "
                    f"Dice = {average_dice:.4f}, "
                    f"IoU = {average_iou:.4f}"
                )

        # Save the model if validation loss improves
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss

            torch.save(
                model.state_dict(),
                "best_3d_unet_weighted.pth"
            )

            print("Best 3D model saved")
    
    return model

def plot_3d_training(log_file="3D_TRAIN_707312.out"):
    """
    Plot 3D U-Net training results from an existing SLURM log.
    
    Assisted by ChatGPT Model 6 to generate plotting
    """
    # Read the training log
    log = (PROJECT_DIR / log_file).read_text()

    # Extract training and validation losses
    train_matches = re.findall(
        r"3D Epoch (\d+)/\d+ Training Loss: ([\d.]+)",
        log
    )

    val_matches = re.findall(
        r"Validation Loss: ([\d.]+)",
        log
    )

    dice_matches = re.findall(
        r"Class (\d+): Dice = ([\d.]+), IoU = ([\d.]+)",
        log
    )

    epochs = [int(epoch) for epoch, _ in train_matches]
    train_loss = [float(loss) for _, loss in train_matches]
    val_loss = [float(loss) for loss in val_matches]

    # Check that each epoch has complete results
    if not epochs or len(epochs) != len(val_loss):
        raise ValueError("Incomplete training loss data")

    if len(dice_matches) != len(epochs) * 5:
        raise ValueError("Incomplete validation Dice data")

    # Calculate mean Dice across foreground classes 1-5
    mean_dice = []

    for i in range(len(epochs)):
        scores = dice_matches[i * 5:(i + 1) * 5]

        if [int(s[0]) for s in scores] != [1, 2, 3, 4, 5]:
            raise ValueError("Unexpected class order")

        mean_dice.append(
            sum(float(s[1]) for s in scores) / 5
        )

    # Plot losses and validation Dice
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, train_loss, label="Training")
    axes[0].plot(epochs, val_loss, label="Validation")
    axes[0].set_title("3D U-Net Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Weighted Cross-Entropy")
    axes[0].legend()

    axes[1].plot(epochs, mean_dice)
    axes[1].set_title("3D U-Net Validation Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Foreground Dice")
    axes[1].set_ylim(0, 1)

    fig.tight_layout()

    # Save the figure
    output_dir = PROJECT_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "3d_training_curves.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    best_index = val_loss.index(min(val_loss))

    print(f"Epochs found: {len(epochs)}")
    print(f"Lowest validation-loss epoch: {epochs[best_index]}")
    print(f"Lowest validation loss: {val_loss[best_index]:.4f}")
    print(f"Plot saved: {output_path}")
