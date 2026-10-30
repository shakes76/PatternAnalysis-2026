import matplotlib.pyplot as plt
import pandas as pd
import os

# Load the training results
baseline = pd.read_csv("checkpoints/unet_baseline_history.csv")
improved = pd.read_csv("checkpoints/unet_improved_history.csv")

os.makedirs("results", exist_ok=True)


def plot_training_results(data, model_name, filename):
    epochs = data["epoch"]
    train_losses = data["training_loss"]
    val_losses = data["validation_loss"]
    val_dices = data["validation_dice"]

    # Find the best validation Dice and corresponding epoch
    best_index = val_dices.idxmax()
    best_epoch = int(data.loc[best_index, "epoch"])
    best_dice = data.loc[best_index, "validation_dice"]
    
    # Graph 1: Training and Validation Loss
    plt.figure(figsize=(8, 5))

    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name}: Training and Validation Loss")
    plt.legend()

    plt.savefig(f"results/{filename}_loss.png")
    plt.show()
    plt.close()

    # Graph 2: Validation Dice
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_dices, label="Validation Dice")

    # Highlight the best epoch
    plt.scatter(
        best_epoch,
        best_dice,
        color="red",
        label=f"Best Dice: {best_dice:.4f} (Epoch {best_epoch})",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title(f"{model_name}: Validation Dice")
    plt.legend()

    plt.savefig(f"results/{filename}_dice.png")
    plt.show()
    plt.close()

    print(f"\n{model_name}")
    print(f"Best Validation Dice: {best_dice:.4f}")
    print(f"Best Epoch: {best_epoch}")

# Standard U-Net
plot_training_results(
    baseline,
    "Standard U-Net",
    "baseline"
)

# Improved U-Net
plot_training_results(
    improved,
    "Improved U-Net",
    "improved"
)