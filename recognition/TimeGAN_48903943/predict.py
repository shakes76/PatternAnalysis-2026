from dataset import get_datasets, LOBNormaliser
from modules import TimeVAE, TimeGAN
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from torch.utils.data import DataLoader

# Convert normalised LOB sequences into depth heatmaps
def to_depth_heatmap(Z, mid, norm, use_log=True):
    real = norm.inverse(Z, mid)

    # extract ask/bid volumes across all 10 levels
    ask_v = real[:, 1::4]
    bid_v = real[:, 3::4]
    img = np.vstack((ask_v.T[::-1], bid_v.T))
    return np.log1p(img) if use_log else img

# Compare original and reconstructed LOB sequences from the TimeVAE
def visualise_reconstructions(model, val_loader, norm, device, output_dir, n=8):
    x, mid = next(iter(val_loader)) # get a batch of real validation windows

    # Reconstruct LOB sequences
    with torch.no_grad():
        x_hat, mu, logvar = model(x.to(device))
    x, x_hat = x.numpy(), x_hat.cpu().numpy()
    mid = mid.numpy()

    # Display original sequences above their reconstructed counterparts
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))

    for i in range(n):
        # Convert original and reconstructed sequences into depth heatmaps
        orig = to_depth_heatmap(x[i], mid[i], norm)
        recon = to_depth_heatmap(x_hat[i], mid[i], norm)

        vmax = max(orig.max(), recon.max())      # shared colour scale per pair
        axes[0, i].imshow(orig, aspect="auto", vmin=0, vmax=vmax)
        axes[1, i].imshow(recon, aspect="auto", vmin=0, vmax=vmax)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Original", loc="left")
    axes[1, 0].set_title("Reconstructed", loc="left")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reconstructions.png"))
    plt.close()

# Compare real validation sequences against independently generated LOB samples
def visualise_samples(model, val_set, norm, device, output_dir, model_type, n=8):
    # generate synthetic samples
    with torch.no_grad():
        if model_type == "vae":
            samples = model.sample(n)
        else:
            z = torch.randn(n, 64, model.noise_dim, device=device)
            samples = model.generate(z)

    samples = samples.cpu().numpy() # Use reference midpoint prices for inverse normalisation

    mid = val_set[0][1].numpy()

    # Convert synthetic sequences into depth heatmaps
    imgs = [
        to_depth_heatmap(samples[i], mid, norm)
        for i in range(n)
    ]

    # Retrieve real validation sequences for comparison
    real_samples = [
        val_set[i][0].numpy()
        for i in range(n)
    ]

    # Convert real validation sequences into depth heatmaps
    real_imgs = [
        to_depth_heatmap(real_samples[i], mid, norm)
        for i in range(n)
    ]

    vmax = max(
        max(img.max() for img in real_imgs),
        max(img.max() for img in imgs)
    )

    # Display real sequences in the first row and generated sequences below
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 5))

    for i in range(n):
        axes[0, i].imshow(real_imgs[i], aspect="auto", vmin=0, vmax=vmax)
        axes[1, i].imshow(imgs[i], aspect="auto", vmin=0, vmax=vmax)

        axes[0, i].axis("off")
        axes[1, i].axis("off")

    axes[0, 0].set_title("Real", loc="left")
    axes[1, 0].set_title("TimeGAN", loc="left")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "real_vs_timegan.png"))
    plt.close()

# Visualise a generated LOB sequence as an order book depth heatmap
def plot_example_output(generated, norm, device, output_dir):
    # Volumes are independent of the absolute midpoint,
    # so an arbitrary midpoint is sufficient for this heatmap.
    dummy_mid = np.full(64, 1e6)

    # Convert generated LOB features into a 20-level volume heatmap
    heatmap = to_depth_heatmap(generated, dummy_mid, norm)

    plt.figure(figsize=(10, 5))
    plt.imshow(heatmap, aspect="auto", origin="upper", cmap="viridis")

    plt.xlabel("Event Number")
    plt.ylabel("Order Book Level")
    plt.title("TimeGAN Generated LOB Sequence (64 Events)")

    plt.xticks(np.arange(0, 64, 8), np.arange(1, 65, 8))
    plt.yticks(
        np.arange(20),
        [f"Ask {i}" for i in range(10, 0, -1)]
        + [f"Bid {i}" for i in range(1, 11)]
    )

    plt.colorbar(label="log(1 + Order Volume)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "example_output.png"),
        dpi=150
    )
    plt.close()

# Visualise the change in generated ask/bid prices over 64 events
def plot_example_prices(generated, norm, val_set, device, output_dir):
    # Use a real starting midpoint so absolute prices can be reconstructed
    mid0 = val_set[0][1][0].item()

    # Reconstruct midpoint path for this generated window
    mids = norm.mid_path(generated[None, :, :], np.array([mid0]))[0]

    # Inverse transform back to raw LOB values
    book = norm.inverse(generated, mids)

    # Extract all 10 ask/bid price levels
    ask_prices = book[:, 0::4] / 10000   # convert to dollar prices
    bid_prices = book[:, 2::4] / 10000

    events = np.arange(1, 65)

    plt.figure(figsize=(10, 5))

    # Plot all 10 ask and 10 bid price levels
    for level in range(10):
        plt.plot(events, ask_prices[:, level], alpha=0.7, label=f"Ask L{level+1}" if level == 0 else "")
        plt.plot(events, bid_prices[:, level], alpha=0.7, label=f"Bid L{level+1}" if level == 0 else "")

    plt.xlabel("Event Number")
    plt.ylabel("Price ($)")
    plt.title("TimeGAN Generated Bid and Ask Price Levels")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "example_prices.png"), dpi=150)
    plt.close()

def main():
    model_type = "timegan"

    DATA_PATH = os.path.expanduser("~/lobster/data/order_sample.csv")
    train_set, val_set, test_set, _ = get_datasets(DATA_PATH)

    val_loader = DataLoader(val_set, batch_size=16, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "vae":
        output_dir = os.path.expanduser("~/baselineOutputs")
        model = TimeVAE(latent_dim=128).to(device)
        checkpoint = "timevae_best.pt"

    elif model_type == "timegan":
        output_dir = os.path.expanduser("~/TimeGAN_gap_penalty")
        model = TimeGAN().to(device)
        checkpoint = "timegan_best.pt"

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(
        torch.load(
            os.path.join(output_dir, checkpoint),
            map_location=device,
            weights_only=True
        )
    )
    model.eval()

    norm = LOBNormaliser.load(
        os.path.join(output_dir, "normaliser.joblib")
    )

    if model_type == "vae":
        visualise_reconstructions(model, val_loader, norm, device, output_dir)

    visualise_samples(model, val_set, norm, device, output_dir, model_type)

    if model_type == "timegan":
        with torch.no_grad():
            z = torch.randn(1, 64, model.noise_dim, device=device)
            generated = model.generate(z)[0].cpu().numpy()

        mid0 = val_set[0][1][0].item()

        # Reconstruct synthetic midpoint prices
        mids = norm.mid_path(
            generated[None, :, :],
            np.array([mid0])
        )[0]

        # Inverse preprocessing
        book = norm.inverse(generated, mids)

        # print out example synthetic outputs
        print("Event | Ask Price | Ask Vol | Bid Price | Bid Vol")

        for i in range(10):
            ask_price = book[i, 0] / 10000
            ask_volume = book[i, 1]
            bid_price = book[i, 2] / 10000
            bid_volume = book[i, 3]

            print(
                f"{i + 1:5d} | "
                f"{ask_price:9.4f} | "
                f"{ask_volume:7.2f} | "
                f"{bid_price:9.4f} | "
                f"{bid_volume:7.2f}"
            )

        plot_example_output(generated, norm, device, output_dir)
        plot_example_prices(generated, norm, val_set, device, output_dir)


if __name__ == "__main__":
    main()