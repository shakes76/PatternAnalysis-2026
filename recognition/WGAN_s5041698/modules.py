import torch
import torch.nn as nn
import torch.nn.functional as F

class WGANGP(nn.Module):
    """
    Wasserstein GAN with Gradient Penalty (WGAN-GP) for Synthetic Medical Slices.
    Contains both the Generator and Critic (Discriminator) networks, along with 
    the gradient penalty loss formulation.
    """
    def __init__(self, latent_dim=128, img_channels=1, feature_map_size=64):
        super(WGANGP, self).__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        
        # Initialize Generator and Critic
        self.generator = Generator(latent_dim=latent_dim, img_channels=img_channels, base_filters=feature_map_size)
        self.critic = Critic(img_channels=img_channels, base_filters=feature_map_size)

    def forward(self, z):
        """Generates fake images from latent vectors z."""
        return self.generator(z)

    def compute_gradient_penalty(self, real_samples, fake_samples, device):
        """
        Computes the gradient penalty loss for WGAN-GP.
        Enforces the 1-Lipschitz constraint by penalizing critic gradient norms.
        """
        batch_size = real_samples.size(0)
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        alpha = alpha.expand_as(real_samples)

        # Get interpolated samples
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)

        # Calculate critic output on interpolated images
        critic_interpolated = self.critic(interpolated)

        # Calculate gradients of critic with respect to interpolated images
        gradients = torch.autograd.grad(
            outputs=critic_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty


class Generator(nn.Module):
    """
    Generator network using resize-convolutions to eliminate 
    checkerboard artifacts common in transposed convolutions.
    """
    def __init__(self, latent_dim=128, img_channels=1, base_filters=64):
        super(Generator, self).__init__()
        
        # Maps latent vector into starting feature map dimensions (e.g., 4x4 or 8x8)
        self.init_size = 8 # Assuming target image is 256x256 (8 -> 16 -> 32 -> 64 -> 128 -> 256)
        self.l1 = nn.Sequential(nn.Linear(latent_dim, base_filters * 16 * self.init_size * self.init_size))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(base_filters * 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Upsample to 16x16
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_filters * 16, base_filters * 8, kernel_size=3, stride=1, padding=1),
            ResNetBlock(base_filters * 8),

            # Upsample to 32x32
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_filters * 8, base_filters * 4, kernel_size=3, stride=1, padding=1),
            ResNetBlock(base_filters * 4),

            # Upsample to 64x64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_filters * 4, base_filters * 2, kernel_size=3, stride=1, padding=1),
            ResNetBlock(base_filters * 2),

            # Upsample to 128x128
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_filters * 2, base_filters, kernel_size=3, stride=1, padding=1),
            ResNetBlock(base_filters),

            # Upsample to 256x256 & Output Image Layer
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_filters, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output range [-1, 1] matching medical normalization
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Critic(nn.Module):
    """
    Critic (Discriminator) network using LayerNormalization instead of BatchNorm 
    to prevent inter-sample dependencies required by WGAN-GP gradient penalty.
    """
    def __init__(self, img_channels=1, base_filters=64):
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(img_channels, base_filters, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 -> 64
            nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 -> 32
            nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 -> 16
            nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 16 -> 8
            nn.Conv2d(base_filters * 8, base_filters * 16, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(base_filters * 16 * 8 * 8, 1) # Assumes 256x256 input spatial dimensions
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


class ResNetBlock(nn.Module):
    """Residual block for sharper medical image generation."""
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x) # Skip connection preserves sharp details


class ConvVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder Architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # -> [32, 128, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> [64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> [128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> [256, 16, 16]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # -> [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Mean and Log-Variance Linear Layers
        self.fc_mu = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(512 * 8 * 8, latent_dim)

        # Decoder Architecture
        self.decoder_input = nn.Linear(latent_dim, 512 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # -> [256, 16, 16]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> [128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> [64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> [32, 128, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),  # -> [in_channels, 256, 256]
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 512, 8, 8)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def vae_loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta * KLD, BCE, KLD