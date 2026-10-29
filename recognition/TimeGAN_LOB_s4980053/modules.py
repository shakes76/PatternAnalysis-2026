# modules.py - the networks

import torch
import torch.nn as nn
import torch.nn.functional as F

def with_label(x, labels, num_stocks=5):
    """Attach each window's one-hot stock label to every time step: (B, T, d) -> (B, T, d + num_stocks)."""
    onehot = F.one_hot(labels, num_stocks).float()                                              # (B,) labels -> (B, num_stocks) switches
    onehot = onehot[:, None, :].expand(-1, x.shape[1], -1)    # copy them to every time step -> (B, T, num_stocks)
    return torch.cat([x, onehot], dim=-1)                                                # glue onto x along the LAST axis

# ---------------------------------------------------------------------------
# RecurrentBlock - the shared building block
# shape: (batch, time, in_dim) -> (batch, time, out_dim), for any number of time steps
# # ---------------------------------------------------------------------------

class RecurrentBlock(nn.Module):
    """GRU over the sequence, then a linear layer at every step (optional sigmoid). Shared by all networks."""    
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, sigmoid=False):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.sigmoid = sigmoid

    def forward(self, x):
        """(batch, time, in_dim) -> (batch, time, out_dim)."""        
        output, _ = self.gru(x)
        out = self.linear(output)
        if self.sigmoid:
            out = torch.sigmoid(out)                      # squash into 0-1 
        return out

# ---------------------------------------------------------------------------
# BaselineGAN - a plain GAN for sequences
# ---------------------------------------------------------------------------

class BaselineGAN(nn.Module):
    """Plain GAN for sequences, conditioned on stock: GRU generator + GRU discriminator, no autoencoder or supervisor."""
    def __init__(self, feature_dim, hidden_dim, num_layers, num_stocks=5):
        super().__init__()
        self.noise_dim = feature_dim
        self.num_stocks = num_stocks
        self.generator = RecurrentBlock(self.noise_dim + num_stocks, hidden_dim, feature_dim, num_layers)
        self.discriminator = RecurrentBlock(feature_dim + num_stocks, hidden_dim, 1, num_layers)

    def sample_noise(self, batch_size, seq_len, device):
        """Gaussian noise, one vector per time step: (batch, seq_len, noise_dim)."""
        return torch.randn(batch_size, seq_len, self.noise_dim, device=device)

    def make_fake(self, noise, labels):
        """Noise + stock label -> fake windows."""
        return self.generator(with_label(noise, labels, self.num_stocks))
    
    def judge(self, x, labels):
        """Window + stock label -> real/fake score per step."""
        return self.discriminator(with_label(x, labels, self.num_stocks))
        
    @torch.no_grad()
    def generate(self, labels, seq_len, device):
        """One fake window per label, e.g. labels = [0, 0, 3] gives two AAPL windows and one INTC."""
        noise = self.sample_noise(len(labels), seq_len, device)
        return self.make_fake(noise, labels)

class TimeGAN(nn.Module):
    """TimeGAN (Yoon et al., 2019) conditioned on stock: autoencoder, supervisor, generator, discriminator."""

    def __init__(self, feature_dim, hidden_dim, num_layers, num_stocks=5):
        super().__init__()
        self.noise_dim = feature_dim
        self.num_stocks = num_stocks
        self.embedder = RecurrentBlock(feature_dim, hidden_dim, hidden_dim, num_layers, sigmoid=True)     # zip:   feature_dim in -> hidden_dim out, WITH sigmoid
        self.recovery = RecurrentBlock(hidden_dim, hidden_dim, feature_dim, num_layers)     # unzip: hidden_dim in -> feature_dim out, no sigmoid
        self.supervisor = RecurrentBlock(hidden_dim + num_stocks, hidden_dim, hidden_dim, num_layers, sigmoid=True)    # code + switches
        self.generator = RecurrentBlock(self.noise_dim + num_stocks, hidden_dim, hidden_dim, num_layers, sigmoid=True) # noise + switches
        self.discriminator = RecurrentBlock(hidden_dim + num_stocks, hidden_dim, 1, num_layers)                        # code + switches

    def reconstruct(self, x):
        """Zip then unzip a real window. Used to train phase 1."""
        code = self.embedder(x)
        return self.recovery(code)              # x -> embedder -> recovery

    def supervise(self, codes, labels):
        """Codes + stock -> guess of each next code."""
        return self.supervisor(with_label(codes, labels, self.num_stocks))

    def make_codes(self, noise, labels):
        """Noise + stock -> fake codes."""
        return self.generator(with_label(noise, labels, self.num_stocks))

    def judge(self, codes, labels):
        """Codes + stock -> real/fake score per step."""
        return self.discriminator(with_label(codes, labels, self.num_stocks))

    def sample_noise(self, batch_size, seq_len, device):
        """Gaussian noise, one vector per time step: (batch, seq_len, noise_dim)."""
        return torch.randn(batch_size, seq_len, self.noise_dim, device=device)

    @torch.no_grad()
    def generate(self, labels, seq_len, device):
        """Noise -> generator -> supervisor -> recovery: one fake window per label."""
        noise = self.sample_noise(len(labels), seq_len, device)
        codes = self.make_codes(noise, labels)                      # fake codes for these stocks
        codes = self.supervise(codes, labels)                      # supervisor makes them move realistically, per stock
        return self.recovery(codes)

def count_parameters(model):
    """Number of trainable weights in a model."""
    total = 0
    for p in model.parameters():    # p = one block of weights at a time
        total += p.numel()          # add how many numbers are in it
    return total

# ---------------------------------------------------------------------------
# quick self-test: python modules.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    labels = torch.tensor([0, 2, 4])
    tagged = with_label(torch.randn(3, 500, 41), labels)
    print("tagged:", tagged.shape)              # expect (3, 500, 46)
    print("window 1, step 0, switches:", tagged[1, 0, 41:])   # expect GOOG: [0, 0, 1, 0, 0]

    block = RecurrentBlock(in_dim=41, hidden_dim=24, out_dim=41, num_layers=3)
    x = torch.randn(128, 60, 41)
    print(block(x).shape)

    gan = BaselineGAN(feature_dim=41, hidden_dim=24, num_layers=3)
    stock = torch.tensor([0, 1, 2, 3, 4, 2, 2, 3])
    fake = gan.generate(stock, seq_len=60, device="cpu")
    print("fake windows:", fake.shape)                  # expect (8, 60, 41)
    print("scores:", gan.judge(fake, stock).shape)     # expect (8, 60, 1)

    print("parameters: ", count_parameters(gan))

    squashed = RecurrentBlock(41, 24, 24, 3, sigmoid=True)(x)
    print("sigmoid block:", squashed.shape, "min", squashed.min().item(), "max", squashed.max().item())

    tg = TimeGAN(feature_dim=41, hidden_dim=24, num_layers=3)
    print("reconstruction:", tg.reconstruct(x).shape)                   # expect (128, 60, 41)
    codes = tg.embedder(x)
    labels = torch.randint(0, 5, (128,))                           # a random stock for each window
    print("supervisor:", tg.supervise(codes, labels).shape)        # expect (128, 60, 24)
    print("judge:", tg.judge(codes, labels).shape)                 # expect (128, 60, 1)
    print("TimeGAN fake:", tg.generate(stock, 60, "cpu").shape)    # expect (8, 60, 41)
    print("TimeGAN parameters (all five):", count_parameters(tg))