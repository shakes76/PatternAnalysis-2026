import torch.nn as nn

# Generator and Discriminator for the GAN. Partially Written by AI.

class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=1, image_size=(256, 128)):
        """
        Generator Class: transforms random noise vector into a full sized image through upsampling (Conv2d)
        The assumed input shape is (100, 1, 1) for the latent vector.
        Input:
            z (torch.Tensor): Random noise vector of shape (batch_size, latent_dim, 1, 1)
        Output (torch.Tensor): Generated image of shape (batch_size, channels, 256, 128)
        Architecture: (ConvTranspose2d + BatchNorm2d + ReLU) blocks for upsampling, followed by Tanh for output.
        Parameters: latent_dim - dimension of random noise vector
                    channels - number of output image channels in final layer (1 for grayscale)
                    image_size - size of the output image (height, width)
        """
        super().__init__()

        if image_size != (256, 128):
            raise ValueError("This generator outputs 256x128 images")

        self.model = nn.Sequential(
            # 1x1 -> 8x4
            nn.ConvTranspose2d(
                latent_dim, 512, kernel_size=(8, 4),
                stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 8x4 -> 16x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 16x8 -> 32x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 32x16 -> 64x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64x32 -> 128x64
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 128x64 -> 256x128
            nn.ConvTranspose2d(32, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)



class Discriminator(nn.Module):
    def __init__(self, channels=1, image_size=(256, 128)):
        """
        Critic/Discriminator Class: evaluates the authenticity of an image (real or generated)
        Parameters: img_channels - number of input image channels (1 for grayscale)
                    feature_maps - number of feature maps in each convolutional layer
        """
        super().__init__()

        if image_size != (256, 128):
            raise ValueError("This discriminator expects 256x128 images")

        self.features = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=(16, 8), stride=1, padding=0, bias=False),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.features(x)


if __name__ == "__main__":
    # Count the number of trainable parameters; This is the model capacity.
    G = Generator()
    D = Discriminator()
    sum_parameters = 0
    for name, model in [("Generator", G), ("Discriminator", D)]:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"{name}: {total:,} parameters ({trainable:,} trainable)")
        sum_parameters += total
    print(f"Total parameters: {sum_parameters:,} ({sum(p.numel() for p in G.parameters() if p.requires_grad) + sum(p.numel() for p in D.parameters() if p.requires_grad):,} trainable)")