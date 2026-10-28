"""
train.py: Train TimeGAN on LOBSTER order book windows.

Follows the three-phase schedule of Yoon et al. (2019):
  1. Embedding:  train embedder + recovery as an autoencoder on real data.
  2. Supervised: train the supervisor to predict the next latent step.
  3. Joint:      adversarial training of generator + supervisor against the
                 discriminator, while fine-tuning the embedder and recovery.

Each window step has len(FEATURE_COLS) + len(FLAG_GROUPS) channels: the
scaled feature values, then a 0/1 change flag per price group (did that part
of the book move since the previous step?). The flags let the model reproduce
the piecewise-constant behaviour of real 1-second books.

Every network is conditioned on a one-hot ticker vector, each ticker has its
own scaler, and every batch holds windows of a single ticker, so the batch
moment and distribution losses compare like with like.

Saves the trained model, its config, the scaler and the feature names to
checkpoints/timegan.pt, and a loss plot to images/timegan_losses.png.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, TensorDataset

from common.constants import DEVICE
from common.scalers import LogMinMaxScaler
from common.visualizations import plot_losses
from dataset import FEATURE_COLS, FLAG_GROUPS, SZ_IDX, encode, load_books, split_books
from modules import TimeGAN

# --- Data Config ---
TICKERS = ["AMZN", "AAPL", "GOOG", "INTC", "MSFT"]
RESAMPLE_FREQ = "1s"
WINDOW_LENGTH = 64
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.7, 0.15, 0.15   # split train, val, test, chronological per ticker

# --- Model Config ---
HIDDEN_DIM = 64
NOISE_DIM = 32
NUM_LAYERS = 3

# --- Training Config ---
BATCH_SIZE = 128
LR = 1e-3
EMBED_EPOCHS = 50
SUPERVISED_EPOCHS = 25
JOINT_EPOCHS = 100
GAMMA = 1.0          # weight of the adversarial loss on unsupervised latents
D_THRESHOLD = 0.5   # only update the discriminator while its loss is above this
W1_WEIGHT = 1.0    # weight of the distribution-matching losses on spread and mid changes
SEED = 0
NUM_VAL = 4096       # windows used for the per-epoch validation loss

CHECKPOINT_DIR = Path("models/checkpoints")
IMAGE_DIR = Path("figures")

# Channel groups for the reconstruction loss. Each group counts equally, so the
# lone mid and spread channels are not drowned out by 20 sizes and 40 flags
N_FEAT = len(FEATURE_COLS)
LOSS_GROUPS = [
    [FEATURE_COLS.index("mid_rel")],
    [FEATURE_COLS.index("spread")],
    [i for i, c in enumerate(FEATURE_COLS) if "_gap_" in c],
    SZ_IDX,
    list(range(N_FEAT, N_FEAT + len(FLAG_GROUPS))),
]


# ---------------------------------------------------------------------------
# --- Losses ---
# ---------------------------------------------------------------------------

def grouped_mse(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Mean of the per-group MSEs, so every channel group weighs the same."""
    return torch.stack([F.mse_loss(x_hat[..., g], x[..., g]) for g in LOSS_GROUPS]).mean()


def reconstruction_loss(model: TimeGAN, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Group-weighted MSE between real features and their embed -> recover round trip."""
    return grouped_mse(model.recovery(model.embedder(x, c), c), x)


def supervised_loss(model: TimeGAN, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """MSE between the supervisor's next-step prediction and the actual next step."""
    return F.mse_loss(model.supervisor(h, c)[:, :-1], h[:, 1:])


def moment_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Match per-(timestep, feature) mean and std across the batch.

    Uses sqrt(var + eps) rather than std directly, because the gradient of
    std is undefined when a generated feature has zero variance, which is
    exactly the collapsed state this loss should push the generator out of.
    """
    var_hat, mean_hat = torch.var_mean(x_hat, dim=0)
    var, mean = torch.var_mean(x, dim=0)
    std_gap = (torch.sqrt(var_hat + 1e-6) - torch.sqrt(var + 1e-6)).abs().mean()
    return std_gap + (mean_hat - mean).abs().mean()


def unscale(x: torch.Tensor, scaler: LogMinMaxScaler, col: int) -> torch.Tensor:
    """Torch version of scaler.inverse_transform for one feature, so gradients flow through."""
    logged = x[..., col] * float(scaler.span[col]) + float(scaler.lo[col])
    return torch.sign(logged) * torch.expm1(logged.abs())


def w1(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """1D Wasserstein distance between two equal-sized samples, in units of the real std.

    Sorting both samples pairs their quantiles, so the whole distribution is
    compared, tails included, which matching mean and std alone cannot do.
    """
    fake, real = torch.sort(fake.flatten())[0], torch.sort(real.flatten())[0]
    return (fake - real).abs().mean() / (real.std() + 1e-6)


def distribution_losses(x_hat: torch.Tensor, x: torch.Tensor, scaler: LogMinMaxScaler) -> tuple[torch.Tensor, torch.Tensor]:
    """W1 between generated and real batches for the spread and the step-to-step mid change, in ticks."""
    spread_col, mid_col = FEATURE_COLS.index("spread"), FEATURE_COLS.index("mid_rel")
    spread_w1 = w1(unscale(x_hat, scaler, spread_col), unscale(x, scaler, spread_col))
    mid_w1 = w1(torch.diff(unscale(x_hat, scaler, mid_col), dim=1),
                torch.diff(unscale(x, scaler, mid_col), dim=1))
    return spread_w1, mid_w1


def bce(logits: torch.Tensor, target: float) -> torch.Tensor:
    """Binary cross-entropy against a constant label."""
    return F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, target))


# ---------------------------------------------------------------------------
# --- Data ---
# ---------------------------------------------------------------------------

class TickerBatchSampler(Sampler):
    """Batches that each hold windows of a single ticker, in shuffled order.

    Every epoch, each ticker's windows are shuffled and cut into full batches
    (the remainder is dropped, keeping batch statistics stable), then the
    batches of all tickers are shuffled together.
    """
    def __init__(self, labels: np.ndarray, batch_size: int, seed: int = 0):
        self.groups = [np.flatnonzero(labels == k) for k in np.unique(labels)]
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        batches = []
        for idx in self.groups:
            idx = self.rng.permutation(idx)
            n_full = len(idx) // self.batch_size * self.batch_size
            batches.extend(np.split(idx[:n_full], n_full // self.batch_size))
        for i in self.rng.permutation(len(batches)):
            yield batches[i].tolist()

    def __len__(self) -> int:
        return sum(len(idx) // self.batch_size for idx in self.groups)


# --- Training Phase 1: Embedder ---
def train_embedding(model: TimeGAN, loader: DataLoader, val_x: torch.Tensor, val_c: torch.Tensor,
                    epochs: int) -> dict:
    """Phase 1: train embedder + recovery to reconstruct real windows."""
    params = list(model.embedder.parameters()) + list(model.recovery.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    losses = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        total = 0.0
        for x, y in loader:
            x, c = x.to(DEVICE), model.one_hot(y.to(DEVICE))
            loss = reconstruction_loss(model, x, c)
            opt.zero_grad()
            (10 * torch.sqrt(loss)).backward()
            opt.step()
            total += loss.item()

        with torch.no_grad():
            val = reconstruction_loss(model, val_x, val_c).item()
        losses["train"].append(total / len(loader))
        losses["val"].append(val)
        print(f"[Embedding  {epoch:3d}/{epochs}] recon MSE train {losses['train'][-1]:.5f}  val {val:.5f}")

    return losses

# --- Training Phase 2: Supervisor ---
def train_supervisor(model: TimeGAN, loader: DataLoader, val_x: torch.Tensor, val_c: torch.Tensor,
                     epochs: int) -> dict:
    """Phase 2: train the supervisor on real latents (embedder frozen)."""
    opt = torch.optim.Adam(model.supervisor.parameters(), lr=LR)
    losses = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        total = 0.0
        for x, y in loader:
            x, c = x.to(DEVICE), model.one_hot(y.to(DEVICE))
            with torch.no_grad():
                h = model.embedder(x, c)
            loss = supervised_loss(model, h, c)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        with torch.no_grad():
            val = supervised_loss(model, model.embedder(val_x, val_c), val_c).item()
        losses["train"].append(total / len(loader))
        losses["val"].append(val)
        print(f"[Supervised {epoch:3d}/{epochs}] next-step MSE train {losses['train'][-1]:.5f}  val {val:.5f}")

    return losses

# --- Training Phase 2: Joint Generator and Discriminator ---
def train_joint(model: TimeGAN, loader: DataLoader, epochs: int, scalers: list[LogMinMaxScaler]) -> dict:
    """Phase 3: adversarial training with embedder fine-tuning.

    Each batch performs two generator + embedder updates and then one
    discriminator update, skipped while the discriminator is already winning
    (loss below D_THRESHOLD) so it cannot overpower the generator.
    """
    opt_g = torch.optim.Adam(
        list(model.generator.parameters()) + list(model.supervisor.parameters()), lr=LR)
    opt_e = torch.optim.Adam(
        list(model.embedder.parameters()) + list(model.recovery.parameters()), lr=LR)
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=LR)
    losses = {name: [] for name in ("d_loss", "g_adv", "g_sup", "g_moment", "g_spread_w1", "g_mid_w1", "e_loss")}

    for epoch in range(1, epochs + 1):
        totals = dict.fromkeys(losses, 0.0)
        for x, y in loader:
            x, c = x.to(DEVICE), model.one_hot(y.to(DEVICE))
            n, seq_len = x.shape[:2]
            scaler = scalers[int(y[0])]   # batches hold a single ticker

            for _ in range(2):
                # Generator + supervisor
                h = model.embedder(x, c)
                e_hat = model.generator(model.noise(n, seq_len), c)
                h_hat = model.supervisor(e_hat, c)
                x_hat = model.recovery(h_hat, c)

                g_adv = bce(model.discriminator(h_hat, c), 1.0)
                g_adv_e = bce(model.discriminator(e_hat, c), 1.0)
                g_sup = supervised_loss(model, h, c)
                g_moment = moment_loss(x_hat, x)
                g_spread_w1, g_mid_w1 = distribution_losses(x_hat, x, scaler)
                g_loss = (g_adv + GAMMA * g_adv_e + 100 * torch.sqrt(g_sup) + 100 * g_moment
                          + W1_WEIGHT * (g_spread_w1 + g_mid_w1))

                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()

                # Embedder + recovery fine-tuning
                h = model.embedder(x, c)
                e_loss = (10 * torch.sqrt(grouped_mse(model.recovery(h, c), x))
                          + 0.1 * supervised_loss(model, h, c))
                opt_e.zero_grad()
                e_loss.backward()
                opt_e.step()

            # Discriminator
            with torch.no_grad():
                h = model.embedder(x, c)
                e_hat = model.generator(model.noise(n, seq_len), c)
                h_hat = model.supervisor(e_hat, c)
            d_loss = (bce(model.discriminator(h, c), 1.0)
                      + bce(model.discriminator(h_hat, c), 0.0)
                      + GAMMA * bce(model.discriminator(e_hat, c), 0.0))
            if d_loss.item() > D_THRESHOLD:
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

            totals["d_loss"] += d_loss.item()
            totals["g_adv"] += g_adv.item()
            totals["g_sup"] += g_sup.item()
            totals["g_moment"] += g_moment.item()
            totals["g_spread_w1"] += g_spread_w1.item()
            totals["g_mid_w1"] += g_mid_w1.item()
            totals["e_loss"] += e_loss.item()

        for name in losses:
            losses[name].append(totals[name] / len(loader))
        print(f"[Joint      {epoch:3d}/{epochs}] "
              + "  ".join(f"{k} {v[-1]:.4f}" for k, v in losses.items()))

    return losses


def main() -> None:
    Path("models/checkpoints").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    torch.manual_seed(SEED)

    # Data pipeline: one scaler per ticker, fitted on the training split only
    books = load_books(TICKERS, resample_freq=RESAMPLE_FREQ)
    splits = split_books(books, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, test_frac=TEST_FRAC)
    scalers = {ticker: LogMinMaxScaler() for ticker in TICKERS}
    train_windows, _, train_labels = encode(splits["train"], scalers, WINDOW_LENGTH, fit=True)
    val_windows, _, val_labels = encode(splits["val"], scalers, WINDOW_LENGTH)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_windows).float(), torch.from_numpy(train_labels)),
        batch_sampler=TickerBatchSampler(train_labels, BATCH_SIZE, seed=SEED),
    )
    # Validation subset (all windows at once would not fit in GPU memory)
    val_idx = np.random.default_rng(SEED).choice(len(train_windows), min(NUM_VAL, len(train_windows)), replace=False)
    # val_x = torch.from_numpy(val_windows[val_idx]).float().to(DEVICE)
    val_x = torch.from_numpy(train_windows[val_idx]).float().to(DEVICE)
    val_y = torch.from_numpy(train_labels[val_idx]).to(DEVICE)

    # Model and training
    config = {
        "feat_dim": train_windows.shape[2],
        "hidden_dim": HIDDEN_DIM,
        "noise_dim": NOISE_DIM,
        "num_layers": NUM_LAYERS,
        "n_tickers": len(TICKERS),
        "window_length": WINDOW_LENGTH,
        "tickers": TICKERS,
        "resample_freq": RESAMPLE_FREQ,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
    }
    model = TimeGAN(config["feat_dim"], HIDDEN_DIM, NOISE_DIM, NUM_LAYERS, len(TICKERS)).to(DEVICE)
    val_c = model.one_hot(val_y)

    embedding_losses = train_embedding(model, loader, val_x, val_c, EMBED_EPOCHS)
    supervisor_losses = train_supervisor(model, loader, val_x, val_c, SUPERVISED_EPOCHS)
    joint_losses = train_joint(model, loader, JOINT_EPOCHS, list(scalers.values()))

    # Save model and loss plots
    torch.save({
        "model": model.state_dict(),
        "config": config,
        "scaler_lo": {ticker: s.lo.tolist() for ticker, s in scalers.items()},
        "scaler_hi": {ticker: s.hi.tolist() for ticker, s in scalers.items()},
        "feature_names": FEATURE_COLS,
    }, CHECKPOINT_DIR / "timegan.pt")

    plot_losses(embedding_losses, supervisor_losses, joint_losses, Path("figures/timegan_losses.png"))


if __name__ == "__main__":
    main()