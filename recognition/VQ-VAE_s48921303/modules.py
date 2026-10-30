"""
==============================================================================
File:        modules.py
Author:      Jonathan Lu
Course:      COMP3710 / COMP7710 - Pattern Recognition
Date:        October 2026
Description: Model architecture components for Hip MRI generation:
             - Baseline Variational Autoencoder (VAE) with Gaussian latent sampling.
             - Vector Quantised Variational Autoencoder (VQ-VAE) with discrete codebook.
             - Gated PixelCNN autoregressive prior with vertical and horizontal stacks
               for sampling discrete codebook index configurations.
             Implemented purely in PyTorch without external NumPy dependencies.
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Downsampling convolutional block consisting of Conv2d, BatchNorm, and ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        """
        Initialise convolutional downsampling block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel. Defaults to 4.
            stride (int): Stride for spatial downsampling. Defaults to 2.
            padding (int): Padding applied to both sides. Defaults to 1.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward pass for downsampling block.

        Args:
            x (torch.Tensor): Input feature tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Downsampled feature tensor of shape (B, C_out, H/2, W/2).
        """
        return self.net(x)


class Decoder(nn.Module):
    """
    Upsampling transposed convolutional block consisting of ConvTranspose2d, BatchNorm, and ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        """
        Initialise transposed convolutional upsampling block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the transposed convolution kernel. Defaults to 4.
            stride (int): Stride for spatial upsampling. Defaults to 2.
            padding (int): Padding applied to features. Defaults to 1.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward pass for upsampling block.

        Args:
            x (torch.Tensor): Input feature tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Upsampled feature tensor of shape (B, C_out, 2*H, 2*W).
        """
        return self.net(x)


class VAE(nn.Module):
    """
    Baseline Variational Autoencoder (VAE) with continuous Gaussian latent space.
    Serves as the benchmark model for comparison against the discrete VQ-VAE pipeline.
    """
    def __init__(self, latent=64, embedding_dim=64):
        """
        Initialise baseline VAE architecture.

        Args:
            latent (int): Dimension of the continuous bottleneck latent space. Defaults to 64.
            embedding_dim (int): Channel capacity for bottleneck convolutions. Defaults to 64.
        """
        super().__init__()
        self.latent = latent
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            Encoder(1, 64),
            Encoder(64, 128),
            nn.Conv2d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.Flatten()
        )

        self.mean = nn.Linear(embedding_dim * 64 * 32, latent)
        self.log_var = nn.Linear(embedding_dim * 64 * 32, latent)
        self.fc = nn.Sequential(
            nn.Linear(latent, embedding_dim * 64 * 32),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, 128, kernel_size=3, stride=1, padding=1),
            Decoder(128, 64),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Encode input image, sample Gaussian latent code via reparameterisation, and decode.

        Args:
            x (torch.Tensor): Input batch of MRI slices of shape (B, 1, 256, 128).

        Returns:
            tuple: (reconstructed_image, latent_mean, latent_log_variance)
        """
        x = self.encoder(x)

        mean = self.mean(x)
        log_var = torch.clamp(self.log_var(x), min=-10, max=10)
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        
        # Reparameterisation trick: sample z = mean + std * epsilon for differentiable backpropagation
        x = mean + std * epsilon
        x = self.fc(x)
        x = x.view(-1, self.embedding_dim, 64, 32)
        
        x = self.decoder(x)

        return x, mean, log_var

    def sample(self, samples, device):
        """
        Sample synthetic slices from the standard Gaussian prior N(0, I).

        Args:
            samples (int): Number of synthetic slices to generate.
            device (torch.device): Target device for tensor allocation.

        Returns:
            torch.Tensor: Generated synthetic images of shape (samples, 1, 256, 128).
        """
        with torch.no_grad():
            z = torch.randn(samples, self.latent, device=device)
            gen = self.fc(z)
            gen = gen.view(-1, self.embedding_dim, 64, 32)
            gen = self.decoder(gen)

            return gen


class VectorQuantiser(nn.Module):
    """
    Discrete Vector Quantiser module (van den Oord et al., NeurIPS 2017).
    Maps continuous encoder outputs to the nearest discrete embedding vector in the codebook
    and backpropagates gradients using the Straight-Through Estimator (STE).
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        """
        Initialise codebook and commitment hyperparameter.

        Args:
            num_embeddings (int): Size of codebook dictionary K.
            embedding_dim (int): Dimensionality D of each embedding vector.
            commitment_cost (float): Weight beta for the commitment loss term.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook dictionary: K vectors of dimension 
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e):
        """
        Quantise continuous encoder activations into nearest codebook embeddings.

        Args:
            z_e (torch.Tensor): Continuous encoder features of shape (B, D, H, W).

        Returns:
            tuple: (quantised_tensor z_q, vq_loss, encoding_indices)
        """
        # Flatten spatial feature map from [B, D, H, W] to [B*H*W, D] for vector distance calculation
        z_e_perm = z_e.permute(0, 2, 3, 1).contiguous()
        flat_z_e = z_e_perm.view(-1, self.embedding_dim)

        # Compute pairwise squared Euclidean distance: ||z_e - e||^2 = ||z_e||^2 + ||e||^2 - 2(z_e . e^T)
        distances = (
            torch.sum(flat_z_e ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2.0 * torch.matmul(flat_z_e, self.embedding.weight.t())
        )

        # Quantisation: assign each latent vector to the nearest codebook entry
        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices).view(z_e_perm.shape)

        # Vector quantisation loss = codebook loss + commitment loss
        loss = F.mse_loss(z_q, z_e_perm.detach()) + self.commitment_cost * F.mse_loss(z_e_perm, z_q.detach())

        # Straight-Through Estimator (STE): pass gradients directly from z_q to z_e
        z_q = z_e_perm + (z_q - z_e_perm).detach()

        # Reshape quantised latent tensor back to [B, D, H, W]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        indices = encoding_indices.view(z_e.shape[0], z_e.shape[2], z_e.shape[3])

        return z_q, loss, indices


class VQVAE(nn.Module):
    """
    Vector Quantised Variational Autoencoder (VQ-VAE).
    Compacts 256x128 MRI slices into a 64x32 discrete latent token grid.
    """
    def __init__(self, num_embeddings=64, embedding_dim=64, commitment_cost=0.25):
        """
        Initialise VQ-VAE model architecture.

        Args:
            num_embeddings (int): Number of discrete codebook entries K. Defaults to 64.
            embedding_dim (int): Dimensionality D of embedding vectors. Defaults to 64.
            commitment_cost (float): Weight for commitment loss. Defaults to 0.25.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            Encoder(1, 64),
            Encoder(64, 128),
            nn.Conv2d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
        )

        self.vq = VectorQuantiser(num_embeddings, embedding_dim, commitment_cost)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, 128, kernel_size=3, stride=1, padding=1),
            Decoder(128, 64),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Forward pass through encoder, vector quantiser, and decoder.

        Args:
            x (torch.Tensor): Input batch of MRI slices of shape (B, 1, 256, 128).

        Returns:
            tuple: (x_recon, vq_loss, codebook_indices)
        """
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, indices

    def decode_indices(self, indices):
        """
        Reconstruct image slices directly from discrete codebook index tokens.

        Args:
            indices (torch.Tensor): Discrete token grid of shape (B, 64, 32).

        Returns:
            torch.Tensor: Reconstructed MRI slices of shape (B, 1, 256, 128).
        """
        z_q = self.vq.embedding(indices)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)

    def get_codebook_indices(self, x):
        """
        Extract discrete codebook tokens corresponding to an input image.

        Args:
            x (torch.Tensor): Input MRI slices of shape (B, 1, 256, 128).

        Returns:
            torch.Tensor: Codebook index tensor of shape (B, 64, 32).
        """
        z_e = self.encoder(x)
        _, _, indices = self.vq(z_e)
        return indices


class MaskedConv2d(nn.Module):
    """
    Causally masked 2D convolution for autoregressive image priors.
    Mask types:
    - 'V': Vertical stack convolution (conditions strictly on rows above).
    - 'A': Horizontal stack Type A mask (excludes current pixel; used at the first layer).
    - 'B': Horizontal stack Type B mask (includes current pixel; used in intermediate layers).
    """
    def __init__(self, type, in_channels, out_channels, kernel_size=5):
        """
        Initialise masked convolution with causality buffer.

        Args:
            type (str): Mask type: 'V' (vertical), 'A' (horizontal input), or 'B' (horizontal hidden).
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Convolutional kernel width/height. Defaults to 5.
        """
        super().__init__()
        centre = kernel_size // 2
        
        if type == 'V':
            mask = torch.ones(kernel_size, kernel_size)
            mask[centre + 1:, :] = 0
            k_size = (kernel_size, kernel_size)
            padding = (centre, centre)
        elif type in ('A', 'B'):
            mask = torch.ones(1, kernel_size)
            if type == 'A':
                mask[0, centre:] = 0
            else:
                mask[0, centre + 1:] = 0
            k_size = (1, kernel_size)
            padding = (0, centre)
        else:
            raise ValueError('type parameter for MaskedConv2d must either be "V", "A" or "B"')
            
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding=padding)

    def forward(self, x):
        """
        Apply masked convolution by zeroing out causal weights.

        Args:
            x (torch.Tensor): Input feature tensor.

        Returns:
            torch.Tensor: Causally masked convolved features.
        """
        return F.conv2d(x, self.conv2d.weight * self.mask, self.conv2d.bias, padding=self.conv2d.padding)


class GatedMaskedConv2d(nn.Module):
    """
    Gated activation block pairing vertical and horizontal masked stacks (van den Oord et al., 2016).
    Applies tanh(filter) * sigmoid(gate) activations and eliminates the blind spot.
    """
    def __init__(self, type, in_channels, out_channels, kernel_size=5):
        """
        Initialise vertical and horizontal gated masked convolutions.

        Args:
            type (str): Horizontal stack mask type ('A' or 'B').
            in_channels (int): Input feature channels.
            out_channels (int): Output feature channels.
            kernel_size (int): Filter kernel size. Defaults to 5.
        """
        super().__init__()
        self.v_conv = MaskedConv2d('V', in_channels, out_channels * 2, kernel_size=kernel_size)
        self.h_conv = MaskedConv2d(type, in_channels, out_channels * 2, kernel_size=kernel_size)
        self.v_to_h = nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=1)
        self.h_out = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, v, h, pos):
        """
        Forward pass combining vertical and horizontal feature streams.

        Args:
            v (torch.Tensor): Vertical stack feature map.
            h (torch.Tensor): Horizontal stack feature map.
            pos (torch.Tensor): Learnable 2D spatial positional embeddings.

        Returns:
            tuple: (updated_vertical_features, updated_horizontal_features)
        """
        v_feat = self.v_conv(v)
        # Shift vertical features down by 1 row to prevent future information leakage into horizontal stack
        v_shift = F.pad(v_feat[:, :, :-1, :], (0, 0, 1, 0))
        h_feat = self.h_conv(h) + self.v_to_h(v_shift) + pos

        # Gated activation unit: element-wise product of tanh (filter) and sigmoid (gate)
        v_f, v_g = torch.chunk(v_feat, 2, dim=1)
        v_out = torch.tanh(v_f) * torch.sigmoid(v_g)

        h_f, h_g = torch.chunk(h_feat, 2, dim=1)
        h_out = self.h_out(torch.tanh(h_f) * torch.sigmoid(h_g))

        return v_out, h_out


class MaskedConv2dLayer(nn.Module):
    """
    Wrapper layer combining GatedMaskedConv2d with residual skip connections.
    """
    def __init__(self, type, in_channel, out_channel, kernel_size=5):
        """
        Initialise gated layer with optional residual connection.

        Args:
            type (str): Mask type: 'A' (no residual) or 'B' (with residual).
            in_channel (int): Number of input channels.
            out_channel (int): Number of output channels.
            kernel_size (int): Kernel size for masked convolution. Defaults to 5.
        """
        super().__init__()
        self.type = type
        self.layer = GatedMaskedConv2d(type, in_channel, out_channel, kernel_size=kernel_size)

    def forward(self, x):
        """
        Process vertical and horizontal stacks with residual summation for type 'B'.

        Args:
            x (tuple): Tuple of (vertical_stream, horizontal_stream, pos_embedding).

        Returns:
            tuple: (updated_v, updated_h, pos_embedding)
        """
        v, h, pos = x
        v_out, h_out = self.layer(v, h, pos)
        if self.type == 'A':
            return v_out, h_out, pos
        else:
            return v + v_out, h + h_out, pos


class PixelCNN(nn.Module):
    """
    Gated PixelCNN prior for autoregressive discrete token modelling.
    Learns the joint distribution p(z) over the 64x32 codebook indices produced by VQ-VAE.
    """
    def __init__(self, num_embeddings=128, hidden_dim=64, B_layers=8, latent_h=64, latent_w=32):
        """
        Initialise Gated PixelCNN prior.

        Args:
            num_embeddings (int): Vocabulary size (number of codebook entries K). Defaults to 128.
            hidden_dim (int): Intermediate channel depth. Defaults to 64.
            B_layers (int): Number of intermediate Type B gated residual layers. Defaults to 8.
            latent_h (int): Height of the discrete latent grid. Defaults to 64.
            latent_w (int): Width of the discrete latent grid. Defaults to 32.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.hidden_dim = hidden_dim
        self.latent_h = latent_h
        self.latent_w = latent_w
        
        self.embedding = nn.Embedding(num_embeddings, hidden_dim)
        self.row_embed = nn.Parameter(torch.randn(1, hidden_dim * 2, latent_h, 1) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(1, hidden_dim * 2, 1, latent_w) * 0.02)
        
        self.net = nn.Sequential(
            MaskedConv2dLayer('A', hidden_dim, hidden_dim, kernel_size=5),
            *[MaskedConv2dLayer('B', hidden_dim, hidden_dim, kernel_size=5) for _ in range(B_layers)]
        )
        self.out_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, num_embeddings, kernel_size=1)
        )

    def forward(self, x):
        """
        Compute unnormalised token logits for each spatial position.

        Args:
            x (torch.Tensor): Integer token grid tensor of shape (B, H, W).

        Returns:
            torch.Tensor: Logits tensor of shape (B, num_embeddings, H, W).
        """
        b, h, w = x.shape
        embeddings = self.embedding(x.long()).permute(0, 3, 1, 2).contiguous()
        pos = self.row_embed[:, :, :h, :] + self.col_embed[:, :, :, :w]
        
        _, features, _ = self.net((embeddings, embeddings, pos))
        return self.out_conv(features)
        
    def sample(self, vqvae, samples, device, temperature=0.9, top_k=0):
        """
        Autoregressively sample new token grids and decode them into synthetic MRI slices.

        Args:
            vqvae (VQVAE): Trained VQ-VAE model with decode_indices() capability.
            samples (int): Number of synthetic slices to generate.
            device (torch.device): Target computation device.
            temperature (float): Softmax sampling temperature (>0; lower = more conservative). Defaults to 0.9.
            top_k (int): If > 0, restricts sampling to top-k highest probability tokens. Defaults to 0.

        Returns:
            torch.Tensor: Decoded synthetic MRI slices of shape (samples, 1, 256, 128).
        """
        self.eval()
        vqvae.eval()
        
        with torch.no_grad():
            tokens = torch.zeros(samples, self.latent_h, self.latent_w, device=device, dtype=torch.long)
            # Autoregressive generation in raster scan order (row by row, column by column)
            for r in range(self.latent_h):
                for c in range(self.latent_w):
                    # Condition on all previous rows and current row history
                    logits = self.forward(tokens[:, :r + 1, :])
                    logits_rc = logits[:, :, r, c] / temperature

                    # Optional top-k truncation to prevent sampling from low-probability tails
                    if top_k > 0:
                        v, _ = torch.topk(logits_rc, min(top_k, logits_rc.size(-1)), dim=-1)
                        logits_rc[logits_rc < v[:, [-1]]] = -float('inf')
                    
                    # Convert logits to categorical distribution and sample discrete token
                    probs = F.softmax(logits_rc, dim=1)
                    sampled_tokens = torch.multinomial(probs, 1).squeeze(-1)
                    
                    tokens[:, r, c] = sampled_tokens

        return vqvae.decode_indices(tokens)
