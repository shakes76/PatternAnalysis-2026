import time

from torch.utils.data import DataLoader, Subset

import torch
import torch.nn.functional as F
import modules
import dataset

import numpy as np
from scipy import ndimage as ndi

import argparse

def calculate_dice_loss(predictions, labels, eps=1e-6):
    """Calculate the mean multi-class soft Dice loss.

    Args:
        predictions: Model logits with shape (B, C, H, W).
        labels: One-hot target masks with shape (B, C, H, W).
        eps: Small value added for numerical stability.

    Returns:
        Scalar Dice loss.
    """
    prediction_probs = torch.softmax(predictions, dim=1)

    numerator = (prediction_probs * labels).sum(dim=(0, 2, 3))
    denominator = prediction_probs.sum(dim=(0, 2, 3)) + labels.sum(dim=(0, 2, 3))

    sum_dice = (2*numerator + eps) / (denominator + eps) # Per class dice
    multi_class_dice_loss = 1 - sum_dice.mean() # mean() for the dividing by |K| classes

    return multi_class_dice_loss


def calculate_dice_components(predictions, labels, num_classes):
    """Calculate per-class overlap and combined pixel counts.

    Predictions are converted to class labels using argmax before computing
    the counts.

    Args:
        predictions: Model logits with shape (B, C, H, W).
        labels: One-hot target masks with shape (B, C, H, W).
        num_classes: Number of segmentation classes.

    Returns:
        Tuple of per-class overlap counts and combined prediction/target counts.
    """
    prediction_answers = predictions.argmax(dim=1) # Picks the class with the highest score (and stores the classID as the entries)
    predictions_onehot = F.one_hot(prediction_answers, num_classes).permute(0, 3, 1, 2).float() # Convert classIDs to one-hot entries, also (B, 256, 128, 6) -> (B, 6, 256, 128)
    
    overlap = (predictions_onehot * labels).sum(dim=(0, 2, 3)) # 1 if both are 1 (1*1), 0 otherwise
    total = predictions_onehot.sum(dim=(0, 2, 3)) + labels.sum(dim=(0, 2, 3))

    return overlap, total


def train_model(model, train_dataloader, validate_dataloader, optimizer, num_epochs, model_save_path, device, num_classes, eps=1e-6):
    """Train a segmentation model and evaluate it after each epoch.

    Tracks training and validation Dice loss, reports per-class validation Dice
    and IoU, and saves the model weights when validation loss improves.

    Args:
        model: Segmentation model to train.
        train_dataloader: Data loader for training samples.
        validate_dataloader: Data loader for validation samples.
        optimizer: Optimizer used to update model parameters.
        num_epochs: Number of training epochs.
        model_save_path: Destination for the best model's state dictionary.
        device: Device on which to run training and validation.
        num_classes: Number of segmentation classes.
        eps: Small value used for numerical stability in metric calculations.

    Returns:
        Dictionary containing per-epoch training and validation loss histories.
    """

    model = model.to(device)
    training_history = {"Train Loss": [], "Validate Loss": []}
    best_validation_loss = float("inf")

    for epoch in range(num_epochs):

        # Step 1: Train
        train_loss = 0.0
        model.train()
        for images, masks, _ in train_dataloader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = calculate_dice_loss(outputs, masks)

            # Backpropagation and optimization
            optimizer.zero_grad()   # Reset/clean
            loss.backward()         # Backpropagation
            optimizer.step()        # Update weights/gradient descent

            train_loss += loss.item()

        # Step 2: Validate set evaluation
        overlap = torch.zeros(num_classes, device=device)
        total = torch.zeros(num_classes, device=device)

        validate_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, masks, _ in validate_dataloader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                validate_loss += calculate_dice_loss(outputs, masks).item()

                new_overlap, new_total = calculate_dice_components(outputs, masks, num_classes)
                
                overlap += new_overlap
                total += new_total

        # Step 3: Record training and validation losses
        avg_training_loss = train_loss / len(train_dataloader)
        avg_validate_loss = validate_loss / len(validate_dataloader)
        training_history["Train Loss"].append(avg_training_loss)
        training_history["Validate Loss"].append(avg_validate_loss)

        print("=========================================")
        print(f"epoch: {epoch+1}")
        print(f"\t- train_loss: {train_loss / len(train_dataloader)}")
        print(f"\t- validate_loss: {validate_loss / len(validate_dataloader)}")

        # Calculate dice coefficient and IoU per class
        dice = (2*overlap + eps) / (total + eps)
        IoU = (overlap + eps) /(total - overlap + eps)
        print("\t-------- per-class -----------")
        for c in range(num_classes):
            if total[c] == 0:
                print(f"\t\t- {c}: dice coefficient: N/A | IoU: N/A (absent from validation labels and predictions)")
                continue

            print(f"\t\t- {c}: dice coefficient: {dice[c].item()} | IoU: {IoU[c].item()}") # Ideally want both to be 1 (perfect match). 0 means no overlap.
        print("\t-----------------------------")
        
        
        # Step 4: Store the model if it's the best seen so far
        if (best_validation_loss > avg_validate_loss):
            print(f"\t- Saving for {avg_validate_loss}")
            best_validation_loss = avg_validate_loss
            torch.save(model.state_dict(), model_save_path)

        print("=========================================")

    return training_history


def test_model(model, test_dataloader, num_classes, model_save_path, device, eps=1e-6):
    """Evaluate a saved segmentation model on the test data.

    Loads model weights, reports per-class Dice, IoU, and boundary F1 scores, and
    reports peak GPU memory use and average per-sample inference time.

    Args:
        model: Segmentation model to evaluate.
        test_dataloader: Data loader for test samples.
        num_classes: Number of segmentation classes.
        model_save_path: Path to the saved model state dictionary.
        device: Device on which to run evaluation.
        eps: Small value added for numerical stability in Dice and IoU calculations.
    """
    model.load_state_dict(torch.load(model_save_path))
    model = model.to(device)

    print("================ TESTING =========================")

    # Keep a tally/count of all overlap and total so that we can compute dice and IoU per-class at the end
    overlap = torch.zeros(num_classes, device=device)
    total = torch.zeros(num_classes, device=device)

    # For boundary F1 score tracking
    boundary_score_sums = np.zeros(num_classes, dtype=np.float64)
    boundary_score_count = np.zeros(num_classes, dtype=np.int64)

    ### For calculating per class dice coefficint and IoU
    model.eval()
    with torch.inference_mode():
        for images, masks, _ in test_dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            new_overlap, new_total = calculate_dice_components(outputs, masks, num_classes)

            overlap += new_overlap
            total += new_total

            # BF score calculations
            batch_sums, batch_counts = calculate_bf_score(outputs, masks, num_classes)

            boundary_score_sums += batch_sums
            boundary_score_count += batch_counts


    # Calculate dice coefficient + IoU + BF score per class
    dice = (2*overlap + eps) / (total + eps)
    IoU = (overlap + eps) /(total - overlap + eps)
    bf_score = np.divide(boundary_score_sums, boundary_score_count)

    for c in range(num_classes):
        print(f"{c}: dice coefficient: {dice[c].item()} | IoU: {IoU[c].item()} | BF score: {bf_score[c]}") # Ideally want both to be 1 (perfect match). 0 means no overlap.


    ### For Peak VRAM and per-case inference time
    profiling_dataloader = DataLoader(test_dataloader.dataset, batch_size=1, num_workers=test_dataloader.num_workers)
    inference_times = []

    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for images, _, _ in profiling_dataloader:
            images = images.to(device)
            
            torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            outputs = model(images) 
            torch.cuda.synchronize(device)

            time_elapsed = time.perf_counter() - start_time
            inference_times.append(time_elapsed)

    peak_GPU_VRAM = torch.cuda.max_memory_allocated(device) / (1024 ** 2) # in MB
    avg_inference_time = sum(inference_times) / len(inference_times) # in seconds

    print(f"Peak GPU Vram: {peak_GPU_VRAM} MB")
    print(f"Average inference time: {avg_inference_time}s")

    print("==================================================")

def calculate_bf_score(predictions, labels, num_classes, distance_threshold=2):
    """Calculate per-class boundary F1 scores for each image in a batch.

    A predicted boundary pixel is considered matched when it is within
    distance_threshold pixels of a target boundary, and vice versa.

    Args:
        predictions: Model logits with shape (B, C, H, W).
        labels: One-hot target masks with shape (B, C, H, W).
        num_classes: Number of segmentation classes.
        distance_threshold: Maximum boundary distance, in pixels, for a match.

    Returns:
        Tuple of per-class sums of boundary F1 scores and per-class counts of
        scored images. Images where a class is absent from both prediction and
        target are excluded.
    """
    
    prediction_IDs = predictions.argmax(dim=1).detach().cpu().numpy()
    label_IDs = labels.argmax(dim=1).detach().cpu().numpy()

    accumulative_bf_scores = np.zeros(num_classes, dtype=np.float64)
    score_counts = np.zeros(num_classes, dtype=np.int64)

    for class_id in range(0, num_classes):
        for batch_index in range(prediction_IDs.shape[0]):
            prediction_binary_mask = (prediction_IDs[batch_index] == class_id) # "is each pixel in this segmentation mask predicted to be class 'class_id'"
            label_binary_mask = (label_IDs[batch_index] == class_id) # "is each pixel in this segmentation mask actually class 'class_id'"

            prediction_edge = prediction_binary_mask & ~ndi.binary_erosion(prediction_binary_mask)
            label_edge = label_binary_mask & ~ndi.binary_erosion(label_binary_mask)


            score = None
            if prediction_edge.sum() == 0 and label_edge.sum() == 0:
                # Class not in either, so ignore
                continue
            elif prediction_edge.sum() == 0 or label_edge.sum() == 0:
                # Class in one but not the other, so score is 0
                score = 0.0
            else:
                dist_to_label = ndi.distance_transform_edt(~label_edge)
                dist_to_prediction = ndi.distance_transform_edt(~prediction_edge)

                precision = (dist_to_label[prediction_edge] <= distance_threshold).mean()
                recall = (dist_to_prediction[label_edge] <= distance_threshold).mean()

                if (precision + recall > 0):
                    score = (2 * precision * recall) / (precision + recall) 
                else:
                    score = 0.0

            accumulative_bf_scores[class_id] += score
            score_counts[class_id] += 1

    return accumulative_bf_scores, score_counts



############# HYPER-PARAMETERS #############
learning_rate = 1e-4
batch_size = 16
num_epochs = 50 # 50 for real
base_channels = 8 # 32 for real
############################################
ROOT = r"Dataset\keras_slices_data"
UNET_MODEL_PATH = "Basic-U-Net.pt"
CAN_MODEL_PATH = "Basic-2D-CAN.pt"

def parse_args():
    """Parse command-line options for training and model configuration.

    Returns:
        Parsed arguments, including model choice, data and save paths, and
        training hyperparameters.
    """
    parser = argparse.ArgumentParser()

    # Model choice
    parser.add_argument("--model", choices = ("2D CAN", "2D U-Net"), default = "2D CAN")

    # File IO
    parser.add_argument("--data_dir", default = ROOT)
    parser.add_argument("--save_path", default = CAN_MODEL_PATH)

    # Hyper parameters
    parser.add_argument("--lr", type=float, default=learning_rate)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--num_epochs", type=int, default=num_epochs)
    parser.add_argument("--base_channels", type=int, default=base_channels)

    return parser.parse_args()



if __name__ == "__main__":
    # Taking in arguments
    args = parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")

    train_dataloader, validate_dataloader, test_dataloader, num_classes = dataset.get_dataloaders(args.data_dir, args.batch_size)
    print("Finished data loading")

    ### For testing purposes ###
    train_small = DataLoader(
        Subset(train_dataloader.dataset, range(2000, 2064)),
                         batch_size=8, shuffle=True)
    validate_small = DataLoader(
        Subset(validate_dataloader.dataset, range(100, 132)),
                        batch_size=8, shuffle=False)
    ############################

    # Initialising model
    model = None
    if (args.model == "2D CAN"):
        model = modules.Basic2DCAN(1, args.base_channels, num_classes)
    elif (args.model == "2D U-Net"):
        model = modules.BasicUNet2D(1, args.base_channels, num_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"chosen model: {args.model}")
    print(f"params: lr={args.lr}, batch_size={args.batch_size}, num_epoch={args.num_epochs}, base_channels={args.base_channels}")

    # Training
    start_time = time.perf_counter()
    training_history = train_model(model, train_dataloader, validate_dataloader, optimizer, args.num_epochs, args.save_path, device, num_classes)
    end_time = time.perf_counter()

    time_elapsed = end_time - start_time
    print(f"Total training time: {time_elapsed/60} minutes")

    # Testing
    test_model(model, test_dataloader, num_classes, args.save_path, device)
