import torch
import torch.nn as nn
import torch.optim as optim
import modules 
import dataset
from modules import MODEL_SAVE_LOCATION
history = {
    "train_loss": [],
    "val_loss": [],
    "val_pixel_accuracy": [],
    "val_mean_iou": [],
    "val_mean_dice": [],
    "val_dice_per_class": []
}

EARLY_STOP_LOADING =  False

"""The validation function, the history dictionary, the training loop and the testing loop were all created by ChatGPT with the prompts:
"Please create a testing loop for the model",
"Can you please create a validation loop for the model"
"""

#VALIDATION
def validate_model(model, val_loader, criterion, device, num_classes):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    correct_pixels = 0
    total_pixels = 0

    # Confusion matrix: rows = ground truth,
    # columns = predicted classes
    confusion_matrix = torch.zeros(
        num_classes, num_classes,
        dtype=torch.int64,
        device=device
    )

    with torch.no_grad():
        for images, masks in val_loader:

            # Ensure images use (B, C, H, W) format
            if images.ndim == 3:
                images = images.unsqueeze(1)

            elif images.ndim == 4 and images.shape[-1] == 1:
                images = images.permute(0, 3, 1, 2).contiguous()

            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long)

            # Forward pass
            logits = model(images)

            # Verify output and mask dimensions
            assert logits.shape[0] == masks.shape[0]
            assert logits.shape[2:] == masks.shape[1:], (
                f"Logits: {logits.shape}, masks: {masks.shape}"
            )

            # Validation loss
            loss = criterion(logits, masks)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Predicted class for each pixel
            predictions = logits.argmax(dim=1)

            # Pixel accuracy
            correct_pixels += (predictions == masks).sum().item()
            total_pixels += masks.numel()

            # Confusion matrix, excluding invalid target labels
            valid = (masks >= 0) & (masks < num_classes)

            indices = (
                masks[valid] * num_classes + predictions[valid]
            )

            confusion_matrix += torch.bincount(
                indices,
                minlength=num_classes ** 2
            ).reshape(num_classes, num_classes)

    # Calculate aggregate metrics
    val_loss = total_loss / max(total_samples, 1)
    pixel_accuracy = (
        correct_pixels / total_pixels if total_pixels else 0.0
    )

    confusion = confusion_matrix.float()
    true_positives = torch.diag(confusion)
    actual = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    union = actual + predicted - true_positives

    # Ignore classes absent from both predictions and targets
    valid_classes = union > 0
    iou_per_class = torch.full(
        (num_classes,), float("nan"), device=device
    )

    iou_per_class[valid_classes] = (
        true_positives[valid_classes] / union[valid_classes]
    )

    mean_iou = (
        iou_per_class[valid_classes].mean().item()
        if valid_classes.any() else 0.0
    )

    # Calculate Dice coefficient for each class
    dice_denominator = actual + predicted

    # Ignore classes absent from both ground truth and predictions
    valid_dice_classes = dice_denominator > 0

    dice_per_class = torch.full(
        (num_classes,), float("nan"), device=device
    )

    dice_per_class[valid_dice_classes] = (
        2.0 * true_positives[valid_dice_classes]
        / dice_denominator[valid_dice_classes]
    )

    # Mean Dice across classes that are present
    mean_dice = (
        dice_per_class[valid_dice_classes].mean().item()
        if valid_dice_classes.any() else 0.0
    )

    print(f"Mean Dice: {mean_dice:.4f}")

    print("\nDice coefficient per class:")
    for class_id in range(num_classes):
        dice = dice_per_class[class_id].item()
        if dice == dice:  # Check for NaN
            print(f"Class {class_id}: {dice:.4f}")
        else:
            print(f"Class {class_id}: N/A")


    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Pixel Accuracy: {pixel_accuracy:.4%} | "
        f"mIoU: {mean_iou:.4f}"
    )

    return {
    "val_loss": val_loss,
    "val_pixel_accuracy": pixel_accuracy,
    "val_mean_iou": mean_iou,
    "val_iou_per_class": iou_per_class.cpu(),
    "val_mean_dice": mean_dice,
    "val_dice_per_class": dice_per_class.cpu()
    }

#Training
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
NUM_CLASSES = modules.NUM_CLASSES
model = modules.Improved2DUNet(
    in_channels=1,
    num_classes=modules.NUM_CLASSES,
    base_filters=32,
    dropout_rate=0.2
).to(device)

batch_size = modules.batch_size
trainloader = dataset.loadTrainingData(batch_size=batch_size, earlyStop=EARLY_STOP_LOADING)
testloader = dataset.loadTestingData(batch_size=batch_size, earlyStop=EARLY_STOP_LOADING)
validationloader = dataset.loadValidationData(batch_size=batch_size,earlyStop=EARLY_STOP_LOADING)
EPOCHS = 50
LEARNING_RATE = 1e-4

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

#Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    total_samples = 0.0
    print("EPOCH: " + str(epoch) + "\n")

    for images, masks in trainloader:
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.long)

        # Expected:
        # images: (B, 1, 256, 128)
        # masks:  (B, 256, 128)

        optimizer.zero_grad(set_to_none=True)
        # Images: (batch_size, height, width)
        images = images.unsqueeze(1)

        #print(images.shape)
        # Expected: [64, 1, 256, 128]
        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()


        running_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = running_loss / len(trainloader)
    train_loss = running_loss / max(total_samples, 1)
    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {avg_loss:.4f}"
    )
    # -------------------------
    # Validation phase
    # -------------------------
    metrics = validate_model(
        model=model,
        val_loader=validationloader,
        criterion=criterion,
        device=device,
        num_classes=NUM_CLASSES
    )

    history["train_loss"].append(train_loss)
    history["val_loss"].append(metrics["val_loss"])
    history["val_pixel_accuracy"].append(
        metrics["val_pixel_accuracy"]
    )
    history["val_mean_iou"].append(metrics["val_mean_iou"])
    history["val_mean_dice"].append(metrics["val_mean_dice"])
    history["val_dice_per_class"].append(
      metrics["val_dice_per_class"].numpy()
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {metrics['val_loss']:.4f}"
    )

#TESTING

def test_model(model, test_loader, criterion, device, num_classes):
    model.eval()

    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0

    # Confusion matrix: rows = true classes,
    # columns = predicted classes
    confusion_matrix = torch.zeros(
        num_classes, num_classes,
        dtype=torch.int64,
        device=device
    )

    with torch.no_grad():
        for images, masks in test_loader:

            # Fix image dimensions if necessary
            if images.ndim == 3:
                images = images.unsqueeze(1)

            elif images.ndim == 4 and images.shape[-1] == 1:
                images = images.permute(0, 3, 1, 2).contiguous()

            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long)

            # Forward pass
            logits = model(images)

            # Verify target dimensions
            assert logits.shape[0] == masks.shape[0]
            assert logits.shape[2:] == masks.shape[1:], (
                f"Output shape {logits.shape}, "
                f"mask shape {masks.shape}"
            )

            # Calculate loss
            loss = criterion(logits, masks)

            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size

            # Predicted class for each pixel
            predictions = torch.argmax(logits, dim=1)

            # Pixel accuracy
            correct_pixels += (predictions == masks).sum().item()
            total_pixels += masks.numel()

            # Update confusion matrix
            valid = (
                (masks >= 0) &
                (masks < num_classes)
            )

            indices = (
                masks[valid] * num_classes +
                predictions[valid]
            )

            confusion_matrix += torch.bincount(
                indices,
                minlength=num_classes ** 2
            ).reshape(num_classes, num_classes)

    # Aggregate metrics
    average_loss = total_loss / len(test_loader.dataset)

    pixel_accuracy = (
        correct_pixels / total_pixels
        if total_pixels > 0 else 0.0
    )

    confusion = confusion_matrix.float()

    true_positives = torch.diag(confusion)
    actual_per_class = confusion.sum(dim=1)
    predicted_per_class = confusion.sum(dim=0)

    union = actual_per_class + predicted_per_class - true_positives

    # Ignore classes absent from both predictions and ground truth
    valid_classes = union > 0

    iou_per_class = torch.full(
        (num_classes,), float("nan"), device=device
    )
    iou_per_class[valid_classes] = (
        true_positives[valid_classes] /
        union[valid_classes]
    )

    mean_iou = (
        iou_per_class[valid_classes].mean().item()
        if valid_classes.any() else 0.0
    )

    # Calculate Dice coefficient for each class
    dice_denominator = actual_per_class + predicted_per_class
    valid_dice_classes = dice_denominator > 0

    dice_per_class = torch.full(
        (num_classes,), float("nan"), device=device
    )

    dice_per_class[valid_dice_classes] = (
        2.0 * true_positives[valid_dice_classes]
        / dice_denominator[valid_dice_classes]
    )

    mean_dice = (
        dice_per_class[valid_dice_classes].mean().item()
        if valid_dice_classes.any() else 0.0
    )



    print("\nTest Results")
    print("-" * 35)
    print(f"Test Loss:       {average_loss:.4f}")
    print(f"Pixel Accuracy:  {pixel_accuracy:.4%}")
    print(f"Mean IoU:        {mean_iou:.4f}")
    print(f"Mean Dice:       {mean_dice:.4f}")

    print("\nDice coefficient per class:")
    for class_id in range(num_classes):
        dice = dice_per_class[class_id].item()
        if dice == dice:
            print(f"Class {class_id}: {dice:.4f}")
        else:
            print(f"Class {class_id}: N/A")


    print("\nIoU per class:")
    for class_id in range(num_classes):
        iou = iou_per_class[class_id].item()
        if iou == iou:  # Check for NaN
            print(f"Class {class_id}: {iou:.4f}")
        else:
            print(f"Class {class_id}: N/A")

    return {
    "test_loss": average_loss,
    "pixel_accuracy": pixel_accuracy,
    "mean_iou": mean_iou,
    "iou_per_class": iou_per_class.cpu(),
    "mean_dice": mean_dice,
    "dice_per_class": dice_per_class.cpu(),
    "confusion_matrix": confusion_matrix.cpu()
    }

criterion = nn.CrossEntropyLoss()

results = test_model(
    model=model,
    test_loader=testloader,
    criterion=criterion,
    device=device,
    num_classes=NUM_CLASSES
)

torch.save(model.state_dict(), MODEL_SAVE_LOCATION)