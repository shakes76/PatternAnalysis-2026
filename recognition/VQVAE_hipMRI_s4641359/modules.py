# AI Declaration
#
# Tool: Claude (Anthropic), used conversationally over several requests.
#
# What the AI did in this file:
# - Wrote the convolutional VAE baseline and its loss, and later clamped
#   the log-variance after my first VAE run showed an exploding KL term.
# - Adapted the VectorQuantizer from the COMP3710 VQ-VAE MNIST demo and
#   extended it with perplexity tracking and dead-code resets.
# - Wrote the residual VQ-VAE encoder/decoder, following the VQ-VAE paper,
#   and made the downsampling depth configurable (n_down).
# - Wrote the gated PixelCNN prior (mask-A and mask-B convolutions) and
#   its sampling loop with temperature.
# - After I reported horizontal bands in the deep prior's samples,
#   suggested and added the optional learned row/column position embedding.
#
# What I did:
# - Chose the VQ-VAE + PixelCNN approach and the VAE baseline.
# - Trained every model, inspected the samples after each change, and
#   decided which configuration to try next (grid size, depth, width,
#   position embedding) based on what the samples showed.
# - Tested the position embedding on the smaller 32x16 prior first before
#   using it in the final 64x32 model.
#


import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
def down_block(in_ch, out_ch):
    """Conv layer that halves height and width (stride 2)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )
 
 
def up_block(in_ch, out_ch):
    """Transposed conv layer that doubles height and width."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )
 
 
class VAE(nn.Module):
    """Convolutional VAE baseline with a flat Gaussian latent.
 
    Encoder: 4 stride-2 convs, 256x128 -> 16x8 feature map, then linear
    layers predicting the latent mean and log-variance.
    Decoder: mirror of the encoder, sigmoid output in [0, 1].
    """
 
    def __init__(self, latent_dim=256, base_ch=32, in_shape=(256, 128)):
        super().__init__()
        self.latent_dim = latent_dim
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 4]
        self.feat_shape = (chs[-1], in_shape[0] // 16, in_shape[1] // 16)
        feat_size = chs[-1] * self.feat_shape[1] * self.feat_shape[2]
 
        self.encoder = nn.Sequential(
            down_block(1, chs[0]),
            down_block(chs[0], chs[1]),
            down_block(chs[1], chs[2]),
            down_block(chs[2], chs[3]),
        )
        self.fc_mu = nn.Linear(feat_size, latent_dim)
        self.fc_logvar = nn.Linear(feat_size, latent_dim)
 
        self.fc_dec = nn.Linear(latent_dim, feat_size)
        self.decoder = nn.Sequential(
            up_block(chs[3], chs[2]),
            up_block(chs[2], chs[1]),
            up_block(chs[1], chs[0]),
            nn.ConvTranspose2d(chs[0], 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )
 
    def encode(self, x):
        h = self.encoder(x).flatten(1)
        # Clamp log-variance so exp(logvar) can't overflow and blow up the KL term.
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10.0, 10.0)
 
    def reparameterise(self, mu, logvar):
        """Sample z = mu + sigma * eps so gradients flow through mu and sigma.
 
        At evaluation the mean is used instead, giving deterministic
        reconstructions (as in the course CNN-VAE demo).
        """
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
 
    def decode(self, z):
        h = self.fc_dec(z).view(-1, *self.feat_shape)
        return self.decoder(h)
 
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar
 
    @torch.no_grad()
    def sample(self, n, device):
        """Generate n new slices by decoding z ~ N(0, I)."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)
 
 
def vae_loss(recon, x, mu, logvar, beta=1.0):
    """VAE loss = reconstruction (summed MSE per image) + beta * KL divergence.
 
    Both terms are summed over pixels/latent dims and averaged over the batch.
    Returns (total, reconstruction, kl) so each can be logged.
    """
    batch = x.size(0)
    recon_loss = F.mse_loss(recon, x, reduction="sum") / batch
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch
    return recon_loss + beta * kl, recon_loss, kl
 
 
# ---------------------------------------------------------------------------
# VQ-VAE (van den Oord et al., 2017, "Neural Discrete Representation Learning")
#
# The VectorQuantizer distance / argmin / straight-through structure is adapted
# from the COMP3710 VQ-VAE MNIST demo. Additions for HipMRI: codebook usage
# tracking (perplexity) and resetting of unused ("dead") codes, plus a deeper
# residual encoder/decoder sized for 256x128 slices.
# ---------------------------------------------------------------------------
 
 
class ResidualBlock(nn.Module):
    """3x3 -> 1x1 conv residual block, as used in the VQ-VAE paper."""
 
    def __init__(self, channels, hidden):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
 
    def forward(self, x):
        return x + self.block(x)
 
 
class VectorQuantizer(nn.Module):
    """Maps each encoder vector to its nearest codebook entry.
 
    Returns the quantised features (with a straight-through gradient), the
    VQ loss (codebook + beta * commitment), the code indices and the codebook
    perplexity (exp of the entropy of code usage; K means every code is used
    equally, values near 1 mean the codebook has collapsed).
    """
 
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25,
                 dead_threshold=1e-3, reset_every=200):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.dead_threshold = dead_threshold
        self.reset_every = reset_every
 
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        # Running (EMA) fraction of vectors assigned to each code, for dead-code resets.
        self.register_buffer("usage", torch.full((num_embeddings,), 1.0 / num_embeddings))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))
 
    def forward(self, z_e):
        # (B, D, H, W) -> (B*H*W, D)
        z_perm = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z_perm.view(-1, self.embedding_dim)
 
        # Squared Euclidean distance to every code: ||z||^2 + ||e||^2 - 2 z.e
        distances = (flat.pow(2).sum(1, keepdim=True)
                     + self.embedding.weight.pow(2).sum(1)
                     - 2.0 * flat @ self.embedding.weight.t())
        indices = distances.argmin(dim=1)
        z_q = self.embedding(indices).view(z_perm.shape)
 
        # Codebook loss moves codes toward encoder outputs; commitment loss
        # keeps the encoder close to its chosen codes.
        loss = (F.mse_loss(z_q, z_perm.detach())
                + self.commitment_cost * F.mse_loss(z_perm, z_q.detach()))
 
        # Straight-through estimator: forward uses z_q, backward copies gradients to z_e.
        z_q = z_perm + (z_q - z_perm).detach()
 
        counts = torch.bincount(indices, minlength=self.num_embeddings).float()
        probs = counts / counts.sum()
        perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum())
 
        if self.training:
            self._track_and_reset(probs, flat.detach())
 
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        indices = indices.view(z_e.shape[0], z_e.shape[2], z_e.shape[3])
        return z_q, loss, indices, perplexity
 
    @torch.no_grad()
    def _track_and_reset(self, probs, flat):
        """Every `reset_every` steps, move rarely used codes onto random encoder outputs."""
        self.usage.mul_(0.99).add_(0.01 * probs)
        self.steps += 1
        if self.steps % self.reset_every == 0:
            dead = self.usage < self.dead_threshold
            n_dead = int(dead.sum())
            if n_dead > 0:
                choice = torch.randint(0, flat.size(0), (n_dead,), device=flat.device)
                self.embedding.weight.data[dead] = flat[choice]
                self.usage[dead] = 1.0 / self.num_embeddings
 
    def lookup(self, indices):
        """Code indices (B, H, W) -> quantised features (B, D, H, W)."""
        return self.embedding(indices).permute(0, 3, 1, 2).contiguous()
 
 
class VQEncoder(nn.Module):
    """Downsamples by 2**n_down with stride-2 convs, then residual blocks -> D channels."""
 
    def __init__(self, hidden=128, res_hidden=64, n_res=2, embedding_dim=64, n_down=2):
        super().__init__()
        layers, in_ch = [], 1
        for i in range(n_down):
            out_ch = hidden // 2 if i < n_down - 1 else hidden
            layers += [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1), nn.ReLU()]
            in_ch = out_ch
        layers.append(nn.Conv2d(hidden, hidden, kernel_size=3, padding=1))
        layers += [ResidualBlock(hidden, res_hidden) for _ in range(n_res)]
        layers += [nn.ReLU(), nn.Conv2d(hidden, embedding_dim, kernel_size=1)]
        self.net = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.net(x)
 
 
class VQDecoder(nn.Module):
    """Mirror of VQEncoder: residual blocks then 2**n_down upsampling, sigmoid output."""
 
    def __init__(self, hidden=128, res_hidden=64, n_res=2, embedding_dim=64, n_down=2):
        super().__init__()
        layers = [nn.Conv2d(embedding_dim, hidden, kernel_size=3, padding=1)]
        layers += [ResidualBlock(hidden, res_hidden) for _ in range(n_res)]
        layers.append(nn.ReLU())
        in_ch = hidden
        for i in range(n_down):
            last = i == n_down - 1
            out_ch = 1 if last else hidden // 2
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.Sigmoid() if last else nn.ReLU())
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
 
    def forward(self, z_q):
        return self.net(z_q)
 
 
class VQVAE(nn.Module):
    """Encoder -> VectorQuantizer -> Decoder.
 
    With n_down=2 a 256x128 slice becomes a 64x32 grid of code indices.
    """
 
    def __init__(self, num_embeddings=512, embedding_dim=64, hidden=128,
                 res_hidden=64, n_res=2, n_down=2, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.encoder = VQEncoder(hidden, res_hidden, n_res, embedding_dim, n_down)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = VQDecoder(hidden, res_hidden, n_res, embedding_dim, n_down)
 
    def forward(self, x):
        z_q, vq_loss, indices, perplexity = self.vq(self.encoder(x))
        return self.decoder(z_q), vq_loss, indices, perplexity
 
    @torch.no_grad()
    def encode_indices(self, x):
        """Image batch -> grid of code indices (B, H', W')."""
        return self.vq(self.encoder(x))[2]
 
    @torch.no_grad()
    def decode_indices(self, indices):
        """Grid of code indices -> images. Used for prior samples."""
        return self.decoder(self.vq.lookup(indices))
 
 
def vqvae_loss(recon, x, vq_loss):
    """VQ-VAE loss = mean-squared reconstruction error + VQ loss.
 
    Returns (total, reconstruction) so each can be logged.
    """
    recon_loss = F.mse_loss(recon, x)
    return recon_loss + vq_loss, recon_loss
 
 
# ---------------------------------------------------------------------------
# PixelCNN prior over VQ-VAE code grids
# (van den Oord et al., 2016, "Conditional Image Generation with PixelCNN
# Decoders"). Learns p(code grid) autoregressively in raster order so new
# code grids can be sampled and decoded into new slices.
# ---------------------------------------------------------------------------
 
 
class MaskedConv2d(nn.Conv2d):
    """Conv layer that only sees pixels above, and to the left in the same row.
 
    Mask type 'A' also hides the centre pixel (first layer, so a position
    never sees its own value); type 'B' allows the centre (later layers).
    """
 
    def __init__(self, mask_type, in_ch, out_ch, kernel_size):
        super().__init__(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        k = kernel_size
        mask = torch.ones(k, k)
        mask[k // 2, k // 2 + (mask_type == "B"):] = 0  # right of centre (and centre for 'A')
        mask[k // 2 + 1:] = 0                            # rows below centre
        self.register_buffer("mask", mask[None, None])
 
    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, padding=self.padding)
 
 
class GatedMaskedBlock(nn.Module):
    """Residual masked conv block with a gated tanh * sigmoid activation."""
 
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv = MaskedConv2d("B", channels, 2 * channels, kernel_size)
        self.out = nn.Conv2d(channels, channels, kernel_size=1)
 
    def forward(self, x):
        a, b = self.conv(x).chunk(2, dim=1)
        return x + self.out(torch.tanh(a) * torch.sigmoid(b))
 
 
class PixelCNNPrior(nn.Module):
    """Autoregressive model of VQ-VAE code grids (B, H, W) with K classes per position."""
 
    def __init__(self, num_embeddings=512, channels=128, n_blocks=10, use_pos=False,
                 max_grid=(64, 64)):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed = nn.Embedding(num_embeddings, channels)
        # Optional learned position embedding. Convolutions are translation
        # equivariant, so without it the prior cannot tell an edge row (where
        # background belongs) from a central row (where the bladder sits).
        # The embedding depends only on (row, col), never on code values, so
        # it does not break the autoregressive ordering.
        self.use_pos = use_pos
        if use_pos:
            self.row_pos = nn.Parameter(torch.randn(1, channels, max_grid[0], 1) * 0.02)
            self.col_pos = nn.Parameter(torch.randn(1, channels, 1, max_grid[1]) * 0.02)
        self.input_conv = MaskedConv2d("A", channels, channels, kernel_size=7)
        self.blocks = nn.Sequential(*[GatedMaskedBlock(channels) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.ReLU(), nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(), nn.Conv2d(channels, num_embeddings, kernel_size=1),
        )
 
    def forward(self, codes):
        """Code grid (B, H, W) -> logits over codes (B, K, H, W)."""
        x = self.embed(codes).permute(0, 3, 1, 2).contiguous()
        if self.use_pos:
            h, w = codes.shape[1:]
            x = x + self.row_pos[:, :, :h] + self.col_pos[:, :, :, :w]
        return self.head(self.blocks(self.input_conv(x)))
 
    @torch.no_grad()
    def sample(self, n, grid_shape, device, temperature=1.0):
        """Sample n code grids one position at a time in raster order."""
        h, w = grid_shape
        codes = torch.zeros(n, h, w, dtype=torch.long, device=device)
        for i in range(h):
            for j in range(w):
                # Masks only look up/left, so rows below i can't affect position (i, j):
                # run on rows 0..i only, which roughly halves sampling time.
                logits = self.forward(codes[:, :i + 1])[:, :, i, j] / temperature
                codes[:, i, j] = torch.multinomial(F.softmax(logits, dim=1), 1).squeeze(1)
        return codes
