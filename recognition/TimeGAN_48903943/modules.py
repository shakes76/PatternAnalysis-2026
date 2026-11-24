import torch
import torch.nn as nn
import torch.nn.functional as F

#---------------------------------TIMEVAE----------------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # 5 conv blocks, each stride=2, doubling channels roughly:
        # each block: Conv1d + BatchNorm1d + activation
        # input: (B, 40, 64) -> output: (B, 512, 2) in format (Batch size, # features, # timesteps)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=40, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        flattened_dim = 512 * 2
        self.fc_mu = nn.Linear(flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(flattened_dim, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)     # (B, 512, 2)
        x = x.flatten(start_dim=1)  # flatten to (B, 1024)
        mu = self.fc_mu(x)          # transform 1024 features into 128 means
        logvar = self.fc_logvar(x)  # transform 1024 features into 128 log-variances
        return mu, logvar           # return 128 mean-logvar pairs to sample from

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        flattened_dim = 512 * 2
        self.fc = nn.Linear(latent_dim, flattened_dim)

        # mirror of encoder: 5 ConvTranspose1d blocks
        # (B, 512, 2) -> (B, 40, 64)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(in_channels=512, out_channels=512, kernel_size=3, stride=2,
                                padding=1, output_padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=2,
                                padding=1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, stride=2,
                                padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=40, kernel_size=3, stride=2,
                                padding=1, output_padding=1)
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 2)      # unflatten
        x = self.deconv_layers(x)   # pass through decoder layers
        return x


class TimeVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    # reparamaterisation allows us to sample from a distribution while maintaining differentiability
    def reparameterize(self, mu, logvar):
        # sample once from each of the 128 distributions learned by the encoder
        std = torch.exp(0.5 * logvar) # undo log-var to get standard deviation
        eps = torch.randn_like(std)   # introduce some noise for randomness
        return mu + std * eps

    def forward(self, x):                          # x: (B, 64, 40)
        mu, logvar = self.encoder(x.permute(0, 2, 1))
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z).permute(0, 2, 1)   # back to (B, 64, 40)
        return x_hat, mu, logvar

    @torch.no_grad()
    def sample(self, n):
        z = torch.randn(n, self.latent_dim, device=next(self.parameters()).device)
        return self.decoder(z).permute(0, 2, 1)

# Loss function combines reconstruction loss and KL loss
def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    b = x.size(0)

    # calculate reconstruction loss using mean squared error
    recon_loss = F.mse_loss(x_hat, x, reduction='sum') / b

    # KL loss using discrete KL loss formula
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / b
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ---------------------------------TIMEGAN--------------------------------

# Embeds real LOB events into latent representations
class Embedder(nn.Module):
    def __init__(self, feature_dim=40, latent_dim=128, num_layers=2):
        super().__init__()

        # GRU learns temporal relationships between consecutive events
        # Input: (B, 64, 40) -> Output: (B, 64, 128)
        self.gru = nn.GRU(
            feature_dim,
            latent_dim,
            num_layers,
            batch_first=True
        )

        # learnable projection of GRU hidden states into latent space
        self.projection = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        h, _ = self.gru(x) # extract hidden states for every timestep
        return self.projection(h)

# Recovers LOB events from their latent representations
class Recovery(nn.Module):
    def __init__(self, feature_dim=40, latent_dim=128, num_layers=2):
        super().__init__()

        # GRU learns to decode temporal dependencies in latent sequences
        # Input: (B, 64, 128) -> Output: (B, 64, 128)
        self.gru = nn.GRU(
            latent_dim,
            latent_dim,
            num_layers,
            batch_first=True
        )

        # projects latent representations back to the original LOB feature space
        self.projection = nn.Linear(latent_dim, feature_dim)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.projection(h)

# Learns to predict the next latent state from previous states
class Supervisor(nn.Module):
    def __init__(self, latent_dim=128, num_layers=2):
        super().__init__()

        # GRU models temporal transitions between consecutive latent states
        # Input: (B, 64, 128) -> Output: (B, 64, 128)
        self.gru = nn.GRU(
            latent_dim,
            latent_dim,
            num_layers,
            batch_first=True
        )

        # projects GRU hidden states to predicted latent states
        self.projection = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.projection(out)

# Generates synthetic latent sequences from random noise
class Generator(nn.Module):
    def __init__(self, noise_dim=40, latent_dim=128, num_layers=2):
        super().__init__()

        # GRU transforms random noise into time-dependent hidden states
        # Input: (B, 64, 40) -> Output: (B, 64, 128)
        self.gru = nn.GRU(
            noise_dim,
            latent_dim,
            num_layers,
            batch_first=True
        )

        # projects GRU hidden states into the learned latent space
        self.projection = nn.Linear(latent_dim, latent_dim)

    def forward(self, z):
        h, _ = self.gru(z)
        return self.projection(h)

# Distinguises between real embedded sequences and fake synthetic latent sequences
class Discriminator(nn.Module):
    def __init__(self, latent_dim=128, num_layers=2):
        super().__init__()

        # GRU analyses temporal patterns in latent sequences
        # Input: (B, 64, 128) -> Output: (B, 64, 128)
        self.gru = nn.GRU(
            latent_dim,
            latent_dim,
            num_layers,
            batch_first=True
        )

        # produces a real/fake logit for each timestep
        self.projection = nn.Linear(latent_dim, 1)

    def forward(self, h):
        out, _ = self.gru(h)
        return self.projection(out)


class TimeGAN(nn.Module):
    def __init__(self, feature_dim=40, latent_dim=128, noise_dim=40, num_layers=2):
        super().__init__()

        self.embedder = Embedder(feature_dim, latent_dim, num_layers)
        self.recovery = Recovery(feature_dim, latent_dim, num_layers)
        self.supervisor = Supervisor(latent_dim, num_layers)
        self.generator = Generator(noise_dim, latent_dim, num_layers)
        self.discriminator = Discriminator(latent_dim, num_layers)

        self.noise_dim = noise_dim
        self.latent_dim = latent_dim

    # reconstruction
    def forward(self, x):
        return self.recovery(self.embedder(x))

    # synthetic data generation
    def generate(self, z):
        return self.recovery(self.supervisor(self.generator(z)))