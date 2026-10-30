import numpy as np
from dataset import get_datasets, LOBNormaliser
from modules import TimeVAE, TimeGAN
from scipy.stats import kurtosis
from skimage.metrics import structural_similarity
import torch
import os
import matplotlib.pyplot as plt
import time

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Model Sampling
def generate_samples(model, model_type, n, device, batch_size=64):
    samples = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            b = min(batch_size, n - start)

            if model_type == "vae":
                x = model.sample(b)
            else:
                z = torch.randn(b, 64, model.noise_dim, device=device)  # sample noise
                x = model.generate(z)                                   # generate from noise

            samples.append(x.cpu().numpy())

    return np.concatenate(samples, axis=0)

# Financial validity
def rule_violations(Z, norm, eps=1e-6):
    Z = np.asarray(Z, dtype=np.float64).reshape(-1, 40)

    gap = np.exp(norm.p_scaler.inverse_transform(Z[:, :18]))        # ticks
    vol = np.expm1(norm.v_scaler.inverse_transform(Z[:, 18:38]))    # shares
    spr = np.exp(norm.s_scaler.inverse_transform(Z[:, 38:39]).ravel())

    # print violation percentages
    # we introduce tiny epsilon leniency since inverse_transform might cause small floating point
    # error
    return {
        "spread_below_one_tick":    np.mean(spr < 1.0 - eps),
        "ask_gap_below_one_tick":   np.mean((gap[:, :9] < 1.0 - eps).any(axis=1)),
        "bid_gap_below_one_tick":   np.mean((gap[:, 9:] < 1.0 - eps).any(axis=1)),
        "neg_volume":               np.mean((vol < -eps).any(axis=1)),
    }

# ------------------------------- KL spread and midprice returns ----------------------------------
# rRecover ask/bid spread values in ticks from normalised form
def spread_ticks(Z, norm):
    Z = np.asarray(Z).reshape(-1, 40) # flatten
    log_spread = norm.s_scaler.inverse_transform(Z[:, 38:39]) # undo standardise
    spread = np.exp(log_spread) # undo log
    return spread.ravel()

# Recover original midprice log returns from normalised form
def midprice_returns(Z, norm):
    Z = np.asarray(Z, dtype=np.float64)
    original_shape = Z.shape[:-1]

    # extract mid-price return feature from column 39 and undo standardise
    returns = norm.r_scaler.inverse_transform(
        Z[..., 39].reshape(-1, 1)
    )
    return returns.reshape(original_shape)

# Calculate KL divergence between real and generated feature distributions
def kl_hist(real, gen, edges, eps=1e-8):
    # flatten and clip outliers to the histogram boundaries
    real = np.clip(np.asarray(real).ravel(), edges[0], edges[-1])
    gen = np.clip(np.asarray(gen).ravel(), edges[0], edges[-1])

    # count real and generated samples in each bin
    real_counts, _ = np.histogram(real, bins=edges)
    gen_counts, _ = np.histogram(gen, bins=edges)

    # convert to probabilities
    real_probs = real_counts / real_counts.sum()
    gen_probs = gen_counts / gen_counts.sum()

    # add epsilon to prevent div by 0 and log(0)
    real_probs += eps
    gen_probs += eps

    real_probs /= real_probs.sum()
    gen_probs /= gen_probs.sum()

    # compute KL
    kl = np.sum(real_probs * np.log(real_probs/gen_probs))
    return float(kl)

# -------------------------------------------- VWAP -----------------------------------------------
def calculate_vwap(X):
    # extract ask/bid prices/volumes
    ask_prices = X[:, 0::4]
    ask_volumes = X[:, 1::4]
    bid_prices = X[:, 2::4]
    bid_volumes = X[:, 3::4]

    # get total volume for asks/bids
    ask_total = ask_volumes.sum(axis=1)
    bid_total = bid_volumes.sum(axis=1)

    # weight each price by corresponding volume and div by total
    ask_vwap = np.divide(
        (ask_prices * ask_volumes).sum(axis=1),
        ask_total,
        out=np.full(len(X), np.nan),
        where=ask_total > 0
    )

    bid_vwap = np.divide(
        (bid_prices * bid_volumes).sum(axis=1),
        bid_total,
        out=np.full(len(X), np.nan),
        where=bid_total > 0
    )

    return bid_vwap, ask_vwap

# Calculate statistics of VWAP offset from mid-price distribution
def vwap_stats(bid_vwap, ask_vwap, mid):
    # offset is difference from mid-price
    bid_offset = (mid - bid_vwap) / 100
    ask_offset = (ask_vwap - mid) / 100

    return {
        "bid_mean": np.mean(bid_offset),
        "bid_variance": np.var(bid_offset),
        "ask_mean": np.mean(ask_offset),
        "ask_variance": np.var(ask_offset)
    }

# Compare real and generated VWAP offset distributions using histograms
def plot_vwap_distributions(real_vwap, gen_vwap, real_mid, gen_mid, side, output_dir):
    # Calculate VWAP offsets from the corresponding midpoint for each side
    if side == "bid":
        real_offsets = (real_mid - real_vwap) / 100
        gen_offsets = (gen_mid - gen_vwap) / 100
    else:
        real_offsets = (real_vwap - real_mid) / 100
        gen_offsets = (gen_vwap - gen_mid) / 100

    # Shared bin edges for a fair comparison
    combined = np.concatenate([real_offsets, gen_offsets])
    bins = np.histogram_bin_edges(combined, bins=60)

    # plot generated vs real histograms
    plt.figure(figsize=(8, 5))
    plt.hist(real_offsets, bins=bins, density=True, alpha=0.5, label="Real")
    plt.hist(gen_offsets, bins=bins, density=True, alpha=0.5, label="Generated")

    plt.xlabel(f"{side.capitalize()} VWAP offset from midpoint (ticks)")
    plt.ylabel("Probability density")
    plt.title(f"{side.capitalize()} VWAP Distribution")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f"{side}_vwap_distribution.png"), dpi=150)
    plt.close()

# ----------------------------------- Temporal Metrics --------------------------------------------
# Calculate autocorrelation of absolute midprice returns to measure volatility clustering
def abs_return_acf(returns, max_lag=20):
    # Convert returns to absolute values to measure magnitude of price movements
    # Input shape: (n_windows, window_length).
    a = np.abs(np.asarray(returns, dtype=np.float64))

    # Ensure input has a window dimension
    if a.ndim == 1:
        a = a[None, :]

    assert max_lag < a.shape[1]

    a = a - a.mean() # centre absolute returns around their global mean
    var = np.mean(a * a) # calculate variance for normalising autocorrelation

    # Calculate autocorrelation at each lag without crossing window boundaries
    return np.array([
        np.mean(a[:, :-k] * a[:, k:]) / var
        for k in range(1, max_lag + 1)
    ])

# Calculate mid-price return distribution statistics to evaluate variability and heavy tails
def tail_stats(returns, qs=(0.01, 0.05, 0.95, 0.99)):
    r = np.asarray(returns, dtype=np.float64).ravel() # flatten

    # Calculate standard deviation and excess kurtosis
    # Higher excess kurtosis generally indicates heavier tails
    out = {"std": r.std(), "excess_kurtosis": kurtosis(r)}   # scipy default = excess

    # Calculate lower and upper return quantiles to examine extreme movements
    for q, v in zip(qs, np.quantile(r, qs)):
        out[f"q{q}"] = v
    return out

# ----------------------------------------------- SSIM --------------------------------------------
def heatmap(Z, norm, use_log=True):
    # One normalised (64, 40) window -> (20, 64) depth heatmap.
    dummy_mid = np.full(len(Z), 1e6)
    real = norm.inverse(Z, dummy_mid)

    img = np.vstack((
        real[:, 1::4].T[::-1],
        real[:, 3::4].T
    ))

    return np.log1p(img) if use_log else img

def mean_ssim(a_imgs, b_imgs, data_range):
    # Mean SSIM over index-paired images
    return float(np.mean([
        structural_similarity(a, b, data_range=data_range)
        for a, b in zip(a_imgs, b_imgs)
    ]))


def nn_ssim(gen_imgs, real_imgs, data_range):
    # For each generated image, find its highest SSIM against real images.
    return float(np.mean([
        max(structural_similarity(g, r, data_range=data_range)
            for r in real_imgs)
        for g in gen_imgs
    ]))

## ----------------------------------- Computational efficiency -----------------------------------
# Count total and trainable parameters to measure model complexity
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# Benchmark synthetic LOB generation latency and throughput
def benchmark_generation(model, model_type, device, batch_size=64, repeats=100):
    model.eval()

    # Generate a batch of synthetic LOB sequences using the selected model
    def generate():
        if model_type == "vae":
            return model.sample(batch_size)
        else:
            z = torch.randn(batch_size, 64, model.noise_dim, device=device)
            return model.generate(z)

    with torch.inference_mode():
        # Warm up GPU
        for _ in range(10):
            generate()

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(repeats):
            generate()

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    batch_latency = elapsed / repeats # average time required to generate one batch in seconds
    throughput = (batch_size * repeats) / elapsed # number of synthetic LOB windows generated/second

    return batch_latency, throughput


def main():
    model_type = "timegan"  # "vae" or "timegan"

    DATA_PATH = os.path.expanduser("~/lobster/data/order_sample.csv")
    train_set, val_set, test_set, _ = get_datasets(DATA_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "vae":
        output_dir = os.path.expanduser("~/baselineOutputs")
        model = TimeVAE(latent_dim=128).to(device)
        checkpoint = "timevae_best.pt"

    elif model_type == "timegan":
        # use ~/TimeGANoutputs for v1, ~/TimeGAN_gap_penalty for v2 and v3
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

    gen_Z = generate_samples(model, model_type, 2000, device)
    real_Z = test_set.Z.numpy()

    # Diagnostic: Generated price-gap distribution
    gaps = np.exp(
        norm.p_scaler.inverse_transform(
            gen_Z.reshape(-1, 40)[:, :18]
        )
    )

    print("Generated gap percentiles:",
          np.percentile(gaps, [50, 90, 95, 99, 99.9]))

    real_gaps = np.exp(
        norm.p_scaler.inverse_transform(real_Z[:, :18])
    )

    print("Real gap percentiles:",
          np.percentile(real_gaps, [50, 90, 95, 99, 99.9]))

    # Financial validity
    print("Real violations:", rule_violations(real_Z, norm))
    print("Generated violations:", rule_violations(gen_Z, norm))

    # KL spread
    real_spread = spread_ticks(real_Z, norm)
    gen_spread = spread_ticks(gen_Z, norm)

    print("Real spread range:", real_spread.min(), real_spread.max())
    print("Generated spread range:", gen_spread.min(), gen_spread.max())

    print("Real percentiles:", np.percentile(real_spread, [50, 90, 95, 99]))
    print("Generated percentiles:", np.percentile(gen_spread, [50, 90, 95, 99]))

    spread_edges = np.concatenate([
        np.arange(0.5, 25.5, 1),
        [np.inf]
    ])

    print("Spread KL:", kl_hist(real_spread, gen_spread, spread_edges))

    # KL midprice return
    real_returns = midprice_returns(real_Z, norm)
    gen_returns = midprice_returns(gen_Z, norm)

    max_return = np.quantile(np.abs(real_returns), 0.999)
    return_edges = np.linspace(-max_return, max_return, 42)

    print("Return KL:", kl_hist(real_returns, gen_returns, return_edges))

    # VWAP calculation
    real_mid = test_set.mid.numpy()
    real_books = norm.inverse(real_Z, real_mid)

    # Use a real midpoint as the starting price for synthetic windows
    mid0 = np.full(len(gen_Z), real_mid[0])
    gen_mids = norm.mid_path(gen_Z, mid0)

    gen_books = np.concatenate([
        norm.inverse(gen_Z[i], gen_mids[i])[None, :, :]
        for i in range(len(gen_Z))
    ], axis=0)

    real_bid_vwap, real_ask_vwap = calculate_vwap(real_books)

    gen_bid_vwap, gen_ask_vwap = calculate_vwap(
        gen_books.reshape(-1, 40)
    )

    print("Real bid VWAP finite:", np.isfinite(real_bid_vwap).all())
    print("Real ask VWAP finite:", np.isfinite(real_ask_vwap).all())

    print("Generated bid VWAP finite:", np.isfinite(gen_bid_vwap).all())
    print("Generated ask VWAP finite:", np.isfinite(gen_ask_vwap).all())

    print("Real bid < ask:", np.mean(real_bid_vwap < real_ask_vwap))
    print("Generated bid < ask:", np.mean(gen_bid_vwap < gen_ask_vwap))

    real_stats = vwap_stats(real_bid_vwap, real_ask_vwap, real_mid)
    gen_stats = vwap_stats(
        gen_bid_vwap,
        gen_ask_vwap,
        gen_mids.reshape(-1)
    )

    print("Real VWAP stats:", real_stats)
    print("Generated VWAP stats:", gen_stats)

    # VWAP distributions
    plot_vwap_distributions(
        real_bid_vwap, gen_bid_vwap,
        real_mid, gen_mids.reshape(-1),
        "bid", output_dir
    )

    plot_vwap_distributions(
        real_ask_vwap, gen_ask_vwap,
        real_mid, gen_mids.reshape(-1),
        "ask", output_dir
    )

    # Autocorrelation of absolute midprice returns
    window = gen_returns.shape[1]

    n_windows = len(real_returns) // window
    real_return_windows = real_returns[:n_windows * window].reshape(
        n_windows, window
    )

    real_acf = abs_return_acf(real_return_windows)
    gen_acf = abs_return_acf(gen_returns)

    print("Real ACF:", np.round(real_acf, 4))
    print("Generated ACF:", np.round(gen_acf, 4))

    # Plot ACF curves

    lags = np.arange(1, len(real_acf) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(lags, real_acf, marker="o", label="Real")
    plt.plot(lags, gen_acf, marker="o", label="Generated")

    plt.axhline(0, color="gray", linestyle="--")
    plt.xlabel("Lag (events)")
    plt.ylabel("Autocorrelation")
    plt.title("Absolute Midprice Return Autocorrelation")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "absolute_return_acf.png"), dpi=150)
    plt.close()

    # Heavy-tail statistics
    real_tails = tail_stats(real_returns)
    gen_tails = tail_stats(gen_returns)

    print("Real return statistics:", real_tails)
    print("Generated return statistics:", gen_tails)

    # SSIM
    rng = np.random.default_rng(SEED)

    # Select two independent sets of 300 real windows
    idx_a = rng.choice(len(test_set), 300, replace=False)
    idx_b = rng.choice(len(test_set), 300, replace=False)

    real_a = [heatmap(test_set[i][0].numpy(), norm) for i in idx_a]
    real_b = [heatmap(test_set[i][0].numpy(), norm) for i in idx_b]

    # Select 300 generated windows
    gen_imgs = [heatmap(z, norm) for z in gen_Z[:300]]

    # Shared intensity range
    real_imgs = real_a + real_b

    data_range = (
            max(img.max() for img in real_imgs)
            - min(img.min() for img in real_imgs)
    )

    # SSIM comparisons
    print("Real vs Real SSIM:", mean_ssim(real_a, real_b, data_range))
    print("Generated vs Real SSIM:", mean_ssim(gen_imgs, real_b, data_range))
    print("Generated nearest Real SSIM:", nn_ssim(gen_imgs[:100], real_a, data_range))

    # Computational efficiency
    total, trainable = count_parameters(model)

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Count parameters used specifically during synthetic data generation
    if model_type == "timegan":
        generation_params = sum(
            p.numel()
            for module in [model.generator, model.supervisor, model.recovery]
            for p in module.parameters()
        )
    else:
        generation_params = sum(
            p.numel() for p in model.decoder.parameters()
        )

    print(f"Generation parameters: {generation_params:,}")

    # Benchmark generation latency and throughout over different batch sizes
    for batch_size in [1, 64]:
        latency, throughput = benchmark_generation(
            model, model_type, device, batch_size=batch_size
        )

        print(f"Batch size {batch_size}:")
        print(f"  Latency: {latency * 1000:.3f} ms")
        print(f"  Throughput: {throughput:.2f} sequences/s")

if __name__ == "__main__":
    main()