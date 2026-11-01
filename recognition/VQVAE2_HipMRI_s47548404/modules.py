"""
modules.py - Model components for the HipMRI generation project (COMP3710, task 2.4).

v1: shared building blocks (residual block, encoder, decoder) and the baseline
autoencoder. The baseline uses the same encoder and decoder as the VQ models
but has NO codebook, so comparing them isolates what quantisation adds.

Architecture follows the VQ-VAE paper (van den Oord et al., 2017,
arXiv:1711.00937): strided convolutions to downsample, residual blocks
(ReLU, 3x3 conv, ReLU, 1x1 conv, skip connection), transposed convolutions
to upsample.

Shapes for a (B, 1, 256, 128) batch with downsample=4:
    encoder: (B, 1, 256, 128) -> (B, hidden, 64, 32)
    decoder: (B, hidden, 64, 32) -> (B, 1, 256, 128)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """x + conv1x1(relu(conv3x3(relu(x)))). Keeps the shape unchanged."""

    def __init__(self, channels, res_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, res_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(res_channels, channels, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    """Downsample by `downsample` (2 or 4) with stride-2 convs, then residual blocks."""

    def __init__(self, in_channels, hidden, res_channels=32, n_res=2, downsample=4):
        super().__init__()
        if downsample == 4:
            layers = [
                nn.Conv2d(in_channels, hidden // 2, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden // 2, hidden, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
            ]
        elif downsample == 2:
            layers = [
                nn.Conv2d(in_channels, hidden, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
            ]
        else:
            raise ValueError("downsample must be 2 or 4")
        layers.append(nn.Conv2d(hidden, hidden, kernel_size=3, padding=1))
        layers += [ResidualBlock(hidden, res_channels) for _ in range(n_res)]
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Residual blocks, then upsample by `upsample` (2 or 4) with transposed convs."""

    def __init__(self, in_channels, out_channels, hidden, res_channels=32, n_res=2,
                 upsample=4):
        super().__init__()
        layers = [nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)]
        layers += [ResidualBlock(hidden, res_channels) for _ in range(n_res)]
        layers.append(nn.ReLU())
        if upsample == 4:
            layers += [
                nn.ConvTranspose2d(hidden, hidden // 2, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(hidden // 2, out_channels, kernel_size=4, stride=2, padding=1),
            ]
        elif upsample == 2:
            layers.append(
                nn.ConvTranspose2d(hidden, out_channels, kernel_size=4, stride=2, padding=1))
        else:
            raise ValueError("upsample must be 2 or 4")
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class VectorQuantizer(nn.Module):
    """
    Quantizing the output continuous latent tensor Z_e(x) from the encoder 
    by performing KNN (K=1) to choose the nearest codebook entry.
    OUtputting discrete latent vector Z_q(x).
    """

    def __init__(self, num_codes=512, dim=64, beta=0.25, reset_every=100):
        super(VectorQuantizer, self).__init__()

        self.num_codes = num_codes
        self.dim = dim
        self.beta = beta
        self.reset_every = reset_every

        self.embedding = nn.Embedding(self.num_codes, self.dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codes, 1.0 / self.num_codes)

        # Buffers: saved with the model but not trained by the optimiser
        self.register_buffer("usage", torch.zeros(num_codes))           # times each code was chosen
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))  # training forward passes

    def forward(self, z):
        # reform input z (B, D, H, W) → (B, H, W, D) -> (N,D)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.dim) 

        # compute the distance from every vector to every codebook entry, giving an (N, K) matrix.
        # distance from a 2D ‖z − e‖² = ‖z‖² + ‖e‖² − 2·z·e
        dist = torch.sum(z_flattened**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # pick the nearest code for each vector (ids), then look up those code vectors.
        min_encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.num_codes, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute the loss
        loss_codebook = F.mse_loss(z_q, z.detach())
        loss_commitment = F.mse_loss(z_q.detach(), z)
        loss = loss_codebook + self.beta * loss_commitment

        # the straight-through estimator
        z_q = z + (z_q - z).detach()

        # reshape q -> (B, D, H, W)
        q = z_q.permute(0, 3, 1, 2).contiguous()
        # reshape ids -> (B,H,W)
        ids = min_encoding_indices.view(z.shape[:-1])

        if self.training:
            self._reset_dead_codes(z_flattened.detach(), min_encodings)

        return q, loss, ids 

    def embed_code(self, ids):
        """(B, H, W) integer code ids -> (B, D, H, W) code vectors, e.g. for decoding generated codes."""
        return self.embedding(ids).permute(0, 3, 1, 2).contiguous()
    
    @torch.no_grad()
    def _reset_dead_codes(self, z_flattened, min_encodings):
        """Every `reset_every` training steps, move unused codes onto random encoder outputs."""
        self.usage += min_encodings.sum(0)   # how many positions picked each code this batch
        self.steps += 1
        if self.steps % self.reset_every == 0:
            dead = self.usage == 0
            n_dead = int(dead.sum())
            if n_dead > 0:
                pick = torch.randint(0, z_flattened.shape[0], (n_dead,), device=z_flattened.device)
                self.embedding.weight.data[dead] = z_flattened[pick]
            self.usage.zero_()

class VQVAE2(nn.Module):
    def __init__(self, in_channel=1, channel=128, n_res_block=2, n_res_channel=32, 
                 embed_dim=64, n_embed=512):

        super(VQVAE2, self).__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_channel, n_res_block, downsample=4)
        self.enc_t = Encoder(channel, channel, n_res_channel, n_res_block, downsample=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = VectorQuantizer(n_embed, embed_dim)
        self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_channel, n_res_block, upsample=2)
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = VectorQuantizer(n_embed, embed_dim)
        self.upsample_t = nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_channel,
            n_res_block,
            upsample=4,
        )

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)

        return torch.sigmoid(self.dec(quant))

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_b = self.quantize_b.embed_code(code_b)

        dec = self.decode(quant_t, quant_b)

        return dec

class BaselineAE(nn.Module):
    """Plain autoencoder: encoder -> 1x1 conv to a latent of `embed_dim` channels -> decoder.

    Same encoder/decoder as the VQ models, but the latent is continuous
    (no codebook), so it can only reconstruct, not generate.
    """

    def __init__(self, hidden=128, embed_dim=64, res_channels=32, n_res=2):
        super().__init__()
        self.encoder = Encoder(1, hidden, res_channels, n_res, downsample=4)
        self.to_latent = nn.Conv2d(hidden, embed_dim, kernel_size=1)
        self.decoder = Decoder(embed_dim, 1, hidden, res_channels, n_res, upsample=4)

    def forward(self, x):
        z = self.to_latent(self.encoder(x))
        return torch.sigmoid(self.decoder(z))  # outputs in [0, 1], like the inputs


class MaskedConv2d(nn.Conv2d):
    """Convolution that only sees positions above, and to the left on the same row.

    Type 'A' also hides the centre: used in the first layer, so a position
    cannot see its own code. Type 'B' keeps the centre: used in later layers,
    where the centre already holds only "past" information.
    """

    def __init__(self, mask_type, in_channels, out_channels, kernel_size):
        super().__init__(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        if mask_type not in ("A", "B"):
            raise ValueError("mask_type must be 'A' or 'B'")
        mask = torch.ones_like(self.weight)
        c = kernel_size // 2
        mask[:, :, c, c + int(mask_type == "B"):] = 0  # centre row: centre (A only) and right
        mask[:, :, c + 1:, :] = 0                      # all rows below
        self.register_buffer("mask", mask)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, padding=self.padding)


class GatedResBlock(nn.Module):
    """Masked conv -> (+ condition) -> tanh(a) * sigmoid(b) -> 1x1 conv, with a skip connection."""

    def __init__(self, channels, kernel_size, cond_channels=0):
        super().__init__()
        self.conv = MaskedConv2d("B", channels, 2 * channels, kernel_size)
        self.cond = nn.Conv2d(cond_channels, 2 * channels, 1) if cond_channels else None
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, cond=None):
        h = self.conv(x)
        if self.cond is not None:
            h = h + self.cond(cond)
        a, b = h.chunk(2, dim=1)
        return x + self.out(torch.tanh(a) * torch.sigmoid(b))


class PixelCNNPrior(nn.Module):
    """Autoregressive prior over a grid of code ids.

    Unconditional (top prior):  PixelCNNPrior(num_codes=512)
    Conditional (bottom prior): PixelCNNPrior(num_codes=512, cond_num_codes=512);
        the top grid (H/2, W/2) is embedded and upsampled x2 to condition every block.

    forward(ids, cond_ids) -> logits (B, num_codes, H, W); train with
    F.cross_entropy(logits, ids).
    """

    def __init__(self, num_codes=512, channels=128, n_layers=10, kernel_size=5,
                 cond_num_codes=None):
        super().__init__()
        self.num_codes = num_codes
        self.embed = nn.Embedding(num_codes, channels)
        self.input_conv = MaskedConv2d("A", channels, channels, 7)
        cond_channels = channels if cond_num_codes else 0
        self.blocks = nn.ModuleList(
            [GatedResBlock(channels, kernel_size, cond_channels) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.ReLU(), nn.Conv2d(channels, channels, 1),
            nn.ReLU(), nn.Conv2d(channels, num_codes, 1))
        if cond_num_codes:
            self.cond_embed = nn.Embedding(cond_num_codes, channels)
            self.cond_net = nn.Sequential(  # (B, C, H/2, W/2) -> (B, C, H, W)
                nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
                nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1))
        else:
            self.cond_embed = None

    def forward(self, ids, cond_ids=None):
        x = self.embed(ids).permute(0, 3, 1, 2).contiguous()  # (B, H, W) -> (B, C, H, W)
        cond = None
        if self.cond_embed is not None:
            cond = self.cond_net(self.cond_embed(cond_ids).permute(0, 3, 1, 2)).contiguous()
        h = self.input_conv(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.head(h)

    @torch.no_grad()
    def sample(self, n, shape, device, cond_ids=None, temperature=1.0):
        """Generate n grids of `shape` (H, W), one position at a time in raster order.

        Slow by design: H * W full forward passes (the sampling-speed risk from the review).
        """
        H, W = shape
        ids = torch.zeros(n, H, W, dtype=torch.long, device=device)
        for i in range(H):
            for j in range(W):
                logits = self.forward(ids, cond_ids)[:, :, i, j] / temperature
                ids[:, i, j] = torch.multinomial(F.softmax(logits, dim=1), 1).squeeze(1)
        return ids

if __name__ == "__main__":
    # Shape test: a fake batch must come back with exactly the input shape
    x = torch.rand(8, 1, 256, 128)
    model = BaselineAE()
    z = model.to_latent(model.encoder(x))
    out = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"input {tuple(x.shape)} -> latent {tuple(z.shape)} -> output {tuple(out.shape)}")
    print(f"output range {out.min():.3f} to {out.max():.3f}, parameters: {n_params:,}")
    assert out.shape == x.shape, "output shape must match input shape"
    print("Shape test passed.")

    # VQ-VAE2 shape test: reconstruction path and generation path
    vq2 = VQVAE2()
    x = torch.rand(8, 1, 256, 128)
    quant_t, quant_b, diff, id_t, id_b = vq2.encode(x)
    out, loss = vq2(x)
    gen = vq2.decode_code(id_t, id_b)
    print(f"VQ-VAE2: input {tuple(x.shape)} -> id_t {tuple(id_t.shape)}, id_b {tuple(id_b.shape)}"
          f" -> output {tuple(out.shape)}, decode_code {tuple(gen.shape)}")
    assert out.shape == x.shape and gen.shape == x.shape, "output shapes must match input"
    assert id_t.shape == (8, 32, 16) and id_b.shape == (8, 64, 32), "unexpected code grid shapes"
    assert loss.dim() == 0, f"VQ loss should be a single number, got shape {tuple(loss.shape)}"
    (out.mean() + loss).backward()
    assert vq2.enc_b.net[0].weight.grad is not None, "no gradient reached the bottom encoder"
    print(f"VQ loss {loss.item():.4f}, parameters: {sum(p.numel() for p in vq2.parameters()):,}")
    print("VQ-VAE2 shape test passed.")