"""
modules.py
Authored by: Benigno Della Bosca

Implementations of VAE and UNet architectures for use in the latent diffusion
model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ENCODE = 0
DECODE = 1

# VAE
#-------------------------------------------------------------------------------
"""
VAEBlock
Block implementation for use in VAE architecture,
convolution + norm + activation
blocktype: ENCODE/DECODE differentiates the block type
in_c: in channels
out_c: out channels
kern: kernel size
groups: number of groups to be used in group norm
"""
class VAEBlock(nn.Module):
    def __init__(self, blocktype, in_c, out_c, kern, groups=32):
        super().__init__()
        if blocktype == ENCODE:
            self.do = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=kern, stride=2, padding=1),
                nn.GroupNorm(groups, out_c),
                nn.SiLU() # avoids neuron death
            )
        elif blocktype == DECODE:
            self.do = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.GroupNorm(groups, out_c),
                nn.SiLU()
            )

    def forward(self, x):
        return self.do(x)

"""
LDMVAE
VAE model architecture -> 3 encode blocks and latent space and 2 decode blocks
in_c: in channels, defaults to single channel (grayscale)
latent_c: latent channels, defaults to 4 channel latent space
"""
class LDMVAE(nn.Module):
    def __init__(self, in_c=1, latent_c=4):
        # Grayscale in to 4 channel latent space
        super().__init__()
        # Encoding layers doubling
        self.encoder = nn.Sequential(
            VAEBlock(ENCODE, in_c, 64, 4),
            VAEBlock(ENCODE, 64, 128, 4),
            VAEBlock(ENCODE, 128, 256, 4)
        )

        # mean and std var in latent space
        self.mu = nn.Conv2d(256, latent_c, kernel_size=1, padding=0)
        self.logvar = nn.Conv2d(256, latent_c, kernel_size=1, padding=0)

        # Decoder layers halving 
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_c, 256, kernel_size=3, padding=1),
            nn.SiLU(),
            VAEBlock(DECODE, 256, 128, 4),
            VAEBlock(DECODE, 128, 64, 4),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(64, in_c, kernel_size=3, padding=1)
            # no final activation 
        )

    # reparameterisation trick to avoid stochastic node
    def reparameterise(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(sigma)
        return (mu + sigma * epsilon)

    # latent space encoding
    def encode(self, x):
        hidden = self.encoder(x)
        return (self.mu(hidden), self.logvar(hidden))

    # latent space decoding
    def decode(self, z):
        return self.decoder(z)

    # get the latent space mean vector, for use in training UNet
    def get_latent(self, x):
        mu, _ = self.encode(x)
        return mu

    # return the reconstructed slice (encode -> reparam -> decode)
    def forward(self, x):
        (mu, logvar) = self.encode(x)
        x_ = self.decode(self.reparameterise(mu, logvar))
        return (x_, mu, logvar)


# UNet
#-------------------------------------------------------------------------------
"""
UNetBlock
Block implementation for use in UNet architecture, dual resblock sequence with
added conditioning.
in_c: in channels
out_c: out channels
condition: condition vector
"""
class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c, condition):
        super().__init__()
        self.condition = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition, out_c)
        )
        self.blk1 = nn.Sequential(
            nn.GroupNorm(32, in_c),
            nn.SiLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        )
        self.blk2 = nn.Sequential(
            nn.GroupNorm(32, out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )
        self.res = nn.Conv2d(in_c, out_c, kernel_size=1)

    # t :: entire conditioning vector (convention is just timestep)
    def forward(self, x, t):
        h = self.blk1(x)
        # add conditioning
        h = h + self.condition(t)[:, :, None, None]
        h = self.blk2(h)
        return (h + self.res(x)) # retain residual information

"""
LDMUNet
UNet architecture, conditioning mlp, 3 downsample, 2 bottleneck, 3 upsample
with skip connections
in_c: in channels default 4 (VAE latent space expectation)
out_c: out channels default 4 (VAE latent space expectation)
cond: conditioning vector size
timesteps: timesteps used for evaluating noisiness
"""
class LDMUNet(nn.Module):
    def __init__(self, in_c=4, out_c=4, f=96, cond=256, timesteps=1000):
        super().__init__()
        self.time = nn.Embedding(timesteps, cond)
        self.classes = nn.Embedding(2, cond) 

        # lightweight mlp for class conditioning 
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond, cond),
        )

        # from latent space to features
        self.from_latent = nn.Conv2d(in_c, f, kernel_size=3, padding=1)

        # down
        self.d1 = UNetBlock(f,f,condition=cond)
        self.down1 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1)
        self.d2 = UNetBlock(f, f * 2, condition=cond)
        self.down2 = nn.Conv2d(f * 2, f * 2, kernel_size=3, stride=2, padding=1)
        self.d3 = UNetBlock(f * 2 , f * 4, condition=cond)
        self.down3 = nn.Conv2d(f * 4, f * 4, kernel_size=3, stride=2, padding=1)

        # bottleneck
        self.b1 = UNetBlock(f*4, f*8, condition=cond)
        self.b2 = UNetBlock(f*8, f*8, condition=cond)

        # up
        # bilinear upsample ai recom. to eliminate checkerboard artifact
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(f*8, f*4, kernel_size=3, padding=1)
        )
        self.u1 = UNetBlock(f*8, f*4, condition=cond)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(f*4, f*2, kernel_size=3, padding=1)
        )
        self.u2 = UNetBlock(f*4, f*2, condition=cond)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(f*2, f, kernel_size=3, padding=1)
        )
        self.u3 = UNetBlock(f*2, f, condition=cond)

        # from features back to latent
        self.to_latent = nn.Sequential(
            nn.GroupNorm(32, f),
            nn.SiLU(),
            nn.Conv2d(f, out_c, kernel_size=3, padding=1)
        )

    def forward(self, x, tstep, classification):
        condition = self.time(tstep.view(-1).long()) + self.classes(
            classification.view(-1).long())
        condition = self.mlp(condition)

        # Downsampling layers
        x = self.from_latent(x)
        d1 = self.d1(x, condition)
        d2 = self.d2(self.down1(d1), condition)
        d3 = self.d3(self.down2(d2), condition)

        # Bottleneck
        b1 = self.b1(self.down3(d3), condition)
        b2 = self.b2(b1, condition)

        # Upsampling + skip connection
        # (Interpolation AI addition for dimension mismatch)
        u1 = self.up1(b2)
        if u1.shape[2:] != d3.shape[2:]:
            u1 = F.interpolate(u1, size=d3.shape[2:], mode="nearest")
        u1 = self.u1(torch.cat([u1, d3], dim=1), condition)

        u2 = self.up2(u1)
        if u2.shape[2:] != d2.shape[2:]:
            u2 = F.interpolate(u2, size=d2.shape[2:], mode="nearest")
        u2 = self.u2(torch.cat([u2, d2], dim=1), condition)
        
        u3 = self.up3(u2)
        if u3.shape[2:] != d1.shape[2:]:
            u3 = F.interpolate(u3, size=d1.shape[2:], mode="nearest")
        u3 = self.u3(torch.cat([u3, d1], dim=1), condition)

        return self.to_latent(u3)