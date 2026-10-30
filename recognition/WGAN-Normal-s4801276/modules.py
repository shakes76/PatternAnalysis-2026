"""
Model components including Generator (for DCGAN and WGAN), Discriminator (for DCGAN),
and Critic (for WGAN) with helper weights_init() function to be used for
synthetic HipMRI 2D slice generation.
"""

import torch.nn as nn
from torch.nn import Module

# GENERATOR
# Latent space vector (z) must be mapped to training images (1 x 128 x 64)
# Strided 2-dim convolutional transpose layers
# MUST batchnorm after conv to be DCGAN
class Generator(Module):
    """
    Generator for both the DCGAN and WGAN. Specify the desired num gpus, size of
    latent z (generator input), size of feature maps and number of channels during
    init.
    """
    def __init__(self, ngpu, nz, ngf, nc):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, (8, 4), 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 8 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 16 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 32 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 64 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 128 x 64``
        )

    # Forward run through the generator
    def forward(self, input):
        return self.main(input)

# DISCRIMINATOR
# Binary classification network that takes (1 x 128 x 64) input and outputs final prob
# Essentially opposite of Generator but uses Conv2D and LeakyReLU with final Sigmoid
class Discriminator(Module):
    """
    Discriminator for the DCGAN ONLY. It matches/inverses the Generator architecture
    and uses BatchNorm, LeakyReLU, with a final Sigmoid. Needs num gpus, size of
    feature map, and number of image channels at init.
    """
    def __init__(self, ngpu, ndf, nc):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 128 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 64 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 32 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 16 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 8 x 4``
            nn.Conv2d(ndf * 8, 1, (8, 4), 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# CRITIC
# Classification network that takes (1 x 128 x 64) input and outputs final score
# NO FINAL SIGMOID unlike DCGAN because we want raw score, not probability
# NO BATCH NORM as it creates correlations between samples in the same batch
class Critic(Module):
    """
    Critic for the WGAN ONLY. It matches/inverts the generator architecure and uses
    GroupNorm and LeakyReLU with no final Sigmoid. Needs num gpus, size of
    feature map, and number of image channels at init.
    """
    def __init__(self, ngpu, ndf, nc):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 128 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 64 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(1, ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 32 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.GroupNorm(1, ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 16 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.GroupNorm(1, ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 8 x 4``
            nn.Conv2d(ndf * 8, 1, (8, 4), 1, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input)

# DCGAN paper says model weights are randomly initialised from normal distribution
# with mean = 0 and stdev = 0.02
def weights_init(m):
    """
    Based on the DCGAN paper. Initialises model weights randomly from a normal
    distribution with mean = 0 and stdev = 0.02.
    """
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)