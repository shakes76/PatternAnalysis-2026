import dataset
import modules 
import data

import os
import torch 
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

def plot_training_loss_and_dice_coefficient(num_epochs, u_net_statistics, can_statistics, output_dir):
    """Plot and save training metrics for the U-Net and CAN models.

    Each model's training and validation losses and per-class Dice coefficients
    are plotted over the specified number of epochs.

    Args:
        num_epochs: Number of epochs represented in the statistics.
        u_net_statistics: U-Net metrics containing train_loss,
            validation_loss, and dice_coefficient entries.
        can_statistics: CAN metrics with the same entries as
            u_net_statistics.
        output_dir: Directory where the plot image is saved.
    """

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, num_epochs + 1)

    # Plot U-net's training and validation loss
    axes[0, 0].plot(epochs, u_net_statistics["train_loss"], label="Training Loss")
    axes[0, 0].plot(epochs, u_net_statistics["validation_loss"], label="Validation Loss")
    axes[0, 0].set_title(f"Loss for 2D U-Net overtime")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha = 0.3)
    axes[0, 0].legend()

    # Plot U-net's validation dice coefficients
    for class_ID, dice_values in enumerate(u_net_statistics["dice_coefficient"]):
        axes[0, 1].plot(epochs, dice_values, label=f"Class {class_ID}")

    axes[0, 1].set_title(f"Dice coefficient for 2D U-Net overtime")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Dice coefficient")
    axes[0, 1].grid(True, alpha = 0.3)
    axes[0, 1].legend()

    # Plot CAN's training and validation loss
    axes[1, 0].plot(epochs, can_statistics["train_loss"], label="Training Loss")
    axes[1, 0].plot(epochs, can_statistics["validation_loss"], label="Validation Loss")
    axes[1, 0].set_title(f"Loss for 2D CAN overtime")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True, alpha = 0.3)
    axes[1, 0].legend()

    # Plot CAN's validation dice coefficients
    for class_ID, dice_values in enumerate(can_statistics["dice_coefficient"]):
        axes[1, 1].plot(epochs, dice_values, label=f"Class {class_ID}")

    axes[1, 1].set_title(f"Dice coefficient for 2D CAN overtime")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Dice coefficient")
    axes[1, 1].grid(True, alpha = 0.3)
    axes[1, 1].legend()

    # Ensure consistent y-axis scaling
    axes[0, 0].set_ylim(0, 0.65)
    axes[1, 0].set_ylim(0, 0.65)

    axes[0, 1].set_ylim(0.3, 1)
    axes[1, 1].set_ylim(0.3, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Training statistics.png"))
    plt.close(figure)
    

def difference_overlay(image, ground_truth, prediction, class_id):
    """Create an RGB overlay showing per-pixel classification results.

    For the requested class, correct predictions are shown in green, false
    negatives in red, and false positives in blue, blended with the grayscale
    input image.

    Args:
        image: Two-dimensional input image.
        ground_truth: Two-dimensional ground-truth class map.
        prediction: Two-dimensional predicted class map.
        class_id: Class whose prediction results to display.

    Returns:
        RGB image as a NumPy array.
    """

    true_positive = (ground_truth == class_id) & (prediction == class_id)
    false_positive = (ground_truth != class_id) & (prediction == class_id)
    false_negative = (ground_truth == class_id) & (prediction != class_id)

    overlay = np.zeros(ground_truth.shape, dtype=np.int32)
    overlay[true_positive] = 1 # Correct classification = green
    overlay[false_negative] = 2 # False negative
    overlay[false_positive] = 3 # False positive

    colours = np.array([
        [0, 0, 0], # 0 = Black
        [0, 1, 0], # 1 = Green
        [1, 0, 0], # 2 = Red
        [0, 0, 1], # 3 = Blue
    ])

    overlay = colours[overlay]

    # Normalise image
    image = image.astype(np.float32)
    gray = (image - image.min()) / (image.max() - image.min() + 1e-8)
    rgb = gray[..., None]

    # Single alpha blend
    alpha = 0.6
    overlay = (1 - alpha)*rgb + alpha*overlay
    return (overlay*255).astype(np.uint8)

def save_test_slice_overlays(image, ground_truth_seg, u_net_prediction, can_prediction, key, output_dir, num_classes):
    """Save per-class difference overlays for both models for one test slice.

    Args:
        image: Two-dimensional input image.
        ground_truth_seg: Two-dimensional ground-truth class map.
        u_net_prediction: U-Net's predicted class segmentation mask.
        can_prediction: CAN's predicted class segmentation mask.
        key: Slice identifier containing patient, week, and slice index.
        output_dir: Directory where the overlay figure is saved.
        num_classes: Number of segmentation classes to display.
    """
    patient, week, slice_index = key

    figure, axes = plt.subplots(2, num_classes, figsize=(32, 8), squeeze = False)

    # Plot 
    for class_id in range(num_classes):
        axes[0, class_id].imshow(difference_overlay(image, ground_truth_seg, u_net_prediction, class_id))
        axes[0, class_id].set_title(f"2D U-net with classID = {class_id}")

        axes[1, class_id].imshow(difference_overlay(image, ground_truth_seg, can_prediction, class_id))
        axes[1, class_id].set_title(f"2D CAN with classID = {class_id}")

    figure.suptitle(f"2D U-Net and CAN differnce overlays for patient {patient}, week {week}, slice {slice_index}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"patient_{patient}_week_{week}_slice_{slice_index}_difference_overlays.png"), dpi=250)
    plt.close(figure)


def save_test_slice(image, ground_truth_seg, u_net_prediction, can_prediction, key, output_dir, num_classes):
    """Save a comparison figure for one test slice.

    The figure shows the input image, ground-truth segmentation, and predictions
    from the U-Net and CAN models.

    Args:
        image: Two-dimensional input image.
        ground_truth_seg: Two-dimensional ground-truth class map.
        u_net_prediction: U-Net's predicted class map.
        can_prediction: CAN's predicted class map.
        key: Slice identifier containing patient, week, and slice index.
        output_dir: Directory where the figure is saved.
        num_classes: Number of segmentation classes used for the color map.
    """
    patient, week, slice_index = key 

    class_colourmap = ListedColormap(plt.get_cmap("tab10").colors[:num_classes]) # For colouring segmentations
    class_norm = BoundaryNorm(np.arange(num_classes + 1) - 0.5, class_colourmap.N) # For ensuring no blending of colours

    figure, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Plot image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input HipMRI Image")

    # Plot ground truth
    axes[1].imshow(ground_truth_seg, cmap=class_colourmap, norm=class_norm)
    axes[1].set_title("Ground truth segmentation")

    # Plot U-Net prediction
    axes[2].imshow(u_net_prediction, cmap=class_colourmap, norm=class_norm)
    axes[2].set_title("2D U-Net segmentation prediction")

    # Plot CAN prediction
    axes[3].imshow(can_prediction, cmap=class_colourmap, norm=class_norm)
    axes[3].set_title("2D CAN segmentation prediction")

    figure.suptitle(f"Patient {patient}, week {week}, slice {slice_index}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"patient_{patient}_week_{week}_slice_{slice_index}.png"))
    plt.close(figure)

def get_predictions(u_net_model, can_model, dataset, output_directory, num_classes, device):
    """Run both models on a dataset and save visualizations for each slice.

    Args:
        u_net_model: U-Net model used to generate predictions.
        can_model: CAN model used to generate predictions.
        dataset: Iterable yielding image batches, one-hot masks, and slice keys.
        output_directory: Directory where slice figures and overlays are saved.
        num_classes: Number of segmentation classes.
        device: Device on which to run model inference.
    """
    u_net_model.eval()
    can_model.eval()

    with torch.inference_mode():
        for images, masks, keys in dataset:
            images = images.to(device)
            
            u_net_prediction = u_net_model(images).argmax(dim=1).cpu().numpy()
            can_prediction = can_model(images).argmax(dim=1).cpu().numpy()
            ground_truth = masks.argmax(dim=1).cpu().numpy()
            images = images.cpu().numpy()

            for i in range(images.shape[0]):
                save_test_slice(images[i, 0], 
                                ground_truth[i], 
                                u_net_prediction[i], 
                                can_prediction[i], 
                                keys[i], output_directory, num_classes)
                save_test_slice_overlays(images[i, 0],
                                         ground_truth[i],
                                         u_net_prediction[i],
                                         can_prediction[i],
                                         keys[i],
                                         output_directory,
                                         num_classes)




############# HYPER-PARAMETERS #############
learning_rate = 1e-4
batch_size = 16
num_epochs = 50 # 50 for real
base_channels = 32 # 32 for real
############################################
U_NET_MODEL_PATH = r"ExperimentData\Basic-U-Net.pt"
CAN_MODEL_PATH = r"ExperimentData\Basic-2D-CAN.pt"
ROOT = r"Dataset\keras_slices_data"
PREDICTION_PATH = r"ExperimentData\TestSlices"
TRAINING_PLOTS_PATH = r"ExperimentData"

def parse_args():
    """Parse command-line options for prediction data and model configuration.

    Returns:
        Parsed arguments, including the data directory and training hyperparameters.
    """
    parser = argparse.ArgumentParser()

    # File IO
    parser.add_argument("--data_dir", default = ROOT)

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

    _, _, test_dataloader, num_classes = dataset.get_dataloaders(args.data_dir, batch_size=1)
    print("Finished data loading")

    # Loading model
    u_net_model = modules.BasicUNet2D(1, args.base_channels, num_classes)
    u_net_model.load_state_dict(torch.load(U_NET_MODEL_PATH))
    u_net_model.to(device)

    can_model = modules.Basic2DCAN(1, args.base_channels, num_classes)
    can_model.load_state_dict(torch.load(CAN_MODEL_PATH))
    can_model.to(device)

    # Get predictions
    get_predictions(u_net_model, can_model, test_dataloader, PREDICTION_PATH, num_classes, device)

    # Get training statistics
    plot_training_loss_and_dice_coefficient(num_epochs, data.U_NET_STATISTICS, data.CAN_STATISTICS, TRAINING_PLOTS_PATH)