import torch.nn as nn
# Encoder and Decoder for the VAE. Partially Written by AI.

class Encoder(nn.Module):
    def __init__(self, latent_dim, channels=1, image_size=(256, 128)):
        super().__init__()
        if image_size != (256, 128):
            raise ValueError("Only image size (256, 128) is supported.")
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1),  # 256x128 -> 128x64
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),        # 128x64 -> 64x32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),       # 64x32 -> 32x16
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),      # 32x16 -> 16x8
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 4, 2, 1),      # 16x8 -> 8x4
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 8 * 4, 512),
            nn.ReLU(inplace=True),
        )

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        features = self.features(x)
        return self.fc_mu(features), self.fc_logvar(features)


class Decoder(nn.Module):
    def __init__(self, latent_dim, channels=1, image_size=(256, 128)):
        super().__init__()
        if image_size != (256, 128):
            raise ValueError("Only image size (256, 128) is supported.")

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 4),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 8, 4)),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),  # 8x4 -> 16x8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16x8 -> 32x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32x16 -> 64x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 64x32 -> 128x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, channels, 4, 2, 1),  # 128x64 -> 256x128
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)



if __name__ == "__main__":
    # Count the number of trainable parameters; This is the model capacity.
    image_size = (256, 128)
    G = Encoder(latent_dim=128, channels=1, image_size=image_size)
    D = Decoder(latent_dim=128, channels=1, image_size=image_size)
    sum_parameters = 0
    for name, model in [("Encoder", G), ("Decoder", D)]:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"{name}: {total:,} parameters ({trainable:,} trainable)")
        sum_parameters += total
    print(f"Total parameters: {sum_parameters:,} ({sum(p.numel() for p in G.parameters() if p.requires_grad) + sum(p.numel() for p in D.parameters() if p.requires_grad):,} trainable)")