from dataclasses import dataclass
from dataclasses import asdict


@dataclass(frozen=True)
class Config:
    """Settings for data, model, training and evaluation, loaded from config.toml.

    Frozen so a run's settings cannot change after loading. The same values are
    saved in each checkpoint (see `to_dict`), so `predict.py` rebuilds the model
    and data pipeline exactly as they were trained.
    """
    # Data
    tickers: list[str]          # LOBSTER tickers to train on, e.g. ["AMZN", "AAPL"]
    window_length: int          # steps per training window (seconds, at 1 s resampling)
    resample_freq: str          # pandas frequency the book is resampled to, e.g. "1s"
    trim_regimes: str           # pandas timedelta to remove from the eod regimes
    train_frac: float           # chronological share of each ticker's day used for training
    val_frac: float             # following share used for validation (early stopping, model choice)
    test_frac: float            # final share held out for the reported results; the three must sum to 1

    # Model
    hidden_dim: int             # width of every GRU and of the latent space
    noise_dim: int              # size of the generator's per-step noise vector
    num_layers: int             # stacked GRU layers per network

    # Training
    batch_size: int             # windows per batch (each batch holds a single ticker)
    learning_rate: float        # Adam learning rate for all three phases
    embedding_epochs: int       # phase 1: embedder + recovery (autoencoder)
    supervised_epochs: int      # phase 2: supervisor on real latents
    joint_epochs: int           # phase 3: adversarial training of all networks
    d_threshold: float          # discriminator only updates while its loss is above this, avoids dominating generator
    train_seed: int             # seed for torch and batch shuffling during training

    # Evaluation
    sample_length: int          # steps per generated window in predict.py (may exceed window_length)
    num_samples: int            # real and synthetic windows compared per ticker
    burn_in: int                # generated steps discarded at the start while the GRUs warm up
    eval_seed: int              # seed for sampling windows and generating in predict.py

    def to_dict(self) -> dict:
        """Plain dict of all settings, for saving in the checkpoint."""
        return asdict(self)
