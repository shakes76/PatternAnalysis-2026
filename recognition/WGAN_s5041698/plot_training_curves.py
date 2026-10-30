import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_curves(csv_path: str, output_path: str = None):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Metrics CSV file not found: {csv_path}")

    # Load metrics
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Critic Loss Subplot
    axes[0].plot(df["epoch"], df["train_critic_loss"], label="Train Critic Loss", color="#1f77b4", linewidth=2)
    if "val_critic_loss" in df.columns and not df["val_critic_loss"].isna().all():
        axes[0].plot(df["epoch"], df["val_critic_loss"], label="Val Critic Loss", color="#aec7e8", linestyle="--", linewidth=1.5)
    axes[0].set_title("Critic Loss Curve", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=11)

    # Generator Loss Subplot
    axes[1].plot(df["epoch"], df["train_gen_loss"], label="Train Generator Loss", color="#d62728", linewidth=2)
    if "val_gen_loss" in df.columns and not df["val_gen_loss"].isna().all():
        axes[1].plot(df["epoch"], df["val_gen_loss"], label="Val Generator Loss", color="#ff9896", linestyle="--", linewidth=1.5)
    axes[1].set_title("Generator Loss Curve", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Loss", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=11)

    plt.suptitle("WGAN Training Loss Curves per Epoch", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    # Determine save path if not provided
    if output_path is None:
        base_dir = os.path.dirname(os.path.abspath(csv_path))
        parent_dir = os.path.dirname(base_dir)
        plots_dir = os.path.join(parent_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        output_path = os.path.join(plots_dir, "training_curves.png")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Training curves successfully plotted and saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot WGAN training curves for critic and generator.")
    default_csv = os.path.join(os.path.dirname(__file__), "logs", "training_metrics.csv")
    parser.add_argument("--csv_path", type=str, default=default_csv, help="Path to training_metrics.csv")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the output plot PNG")
    args = parser.parse_args()

    plot_training_curves(args.csv_path, args.output_path)

