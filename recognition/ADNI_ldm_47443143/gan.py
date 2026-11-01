"""
gan.py
Authored by: Benigno Della Bosca

A DCGAN implementation as a baseline for comparing the overall effectiveness of
the ldm. Unlike the rest of the ldm this file will be monolithic and separate,
with the intention of being separated from  the rest of the ldm as needed.

Implementation mostly borrowed from previous course content
"""
import sys
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset import get_train_ldr

# Hyperparameters
#-------------------------------------------------------------------------------

z_dim = 256
batch_size = 64
epochs = 50
lr_g = 2e-4
lr_d = 5e-5 # Descriminator learns slower (avoid overpowering generator)

# Generator
#-------------------------------------------------------------------------------
"""
Generator
The generative architecture in the GAN, 6 layer transpose + norm + relu
z_dim: dimension of noise vector
out_c: out channel, single channel default (grayscale)
f: features
"""
class Generator(nn.Module):
    def __init__(self, z_dim, out_c=1, f=32):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, f * 8, 4, 1, 0),
            nn.BatchNorm2d(f * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(f * 8, f * 8, 4, 2, 1),
            nn.BatchNorm2d(f * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(f * 8, f * 8, 4, 2, 1),
            nn.BatchNorm2d(f * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(f * 8, f * 4, 4, 2, 1),
            nn.BatchNorm2d(f * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(f * 4, f * 2, 4, 2, 1),
            nn.BatchNorm2d(f * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(f * 2, f, 4, 2, 1),
            nn.BatchNorm2d(f),
            nn.ReLU(True),

            nn.ConvTranspose2d(f, out_c, 4,2,1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

# Discriminator
#-------------------------------------------------------------------------------
"""
Discriminator
discriminating architecture in GAN
in_c: in channels, single channel default (grayscale)
f: features
"""
class Discriminator(nn.Module):
    def __init__(self, in_c=1, f=32):
        super().__init__()
        self.discrim = nn.Sequential(
            nn.Conv2d(in_c, f, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(f, f * 2, 4, 2, 1,bias=False),
            nn.BatchNorm2d(f * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(f * 2, f * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(f * 4, f * 8, 4,2,1,bias=False),
            nn.BatchNorm2d(f * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(f * 8, f * 8, 4,2,1,bias=False),
            nn.BatchNorm2d(f * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(f * 8, f * 8, 4,2,1,bias=False),
            nn.BatchNorm2d(f * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(f * 8, 1,4,1,0, bias=False),
        )

    def forward(self, x):
        return self.discrim(x).view(-1)

#-------------------------------------------------------------------------------
# generate a random noise vector, generate a slice and save
def sample(gen, dev):
    with torch.no_grad():
        noise = torch.randn(1, z_dim, 1, 1, device=dev)
        sample = gen(noise)
        # map [-1,1] to [0,1] from tanh
        sample = ((sample + 1.0) / 2.0).clamp(0.0, 1.0).squeeze().cpu().numpy()
        plt.imsave("gan_sample.png", sample, cmap="gray")

"""
train_gan
training loop for gan model, c
gen: generator model
discrim: discriminator mode
ldr: training loader
loss: loss function
opt_g: generator optimiser
opt_d: discriminator optimiser
dev: device (CUDA)
epochs: num epochs
"""
def train_gan(gen, discrim, ldr, loss, opt_g, opt_d, dev, epochs):
    gen.train()
    discrim.train()
    start_time = time.time()
    
    with open("gan_loss.csv", "w", encoding="utf-8") as file:
        file.write("epoch,d_loss,g_loss\n")

    for epoch in range(epochs):
        total_d_loss, total_g_loss = 0.0, 0.0
        for slices, _ in ldr:
            # map [0,1] tp [-1,1] for tanh
            real_imgs = slices.to(dev) * 2.0 - 1.0 
            b_size = real_imgs.size(0)
            
            # ---- discriminator ----
            opt_d.zero_grad()
            # How good do we recognise real images?
            real  = discrim(real_imgs)
            # 0.9 to reduce certainty
            d_loss_real = loss(real, torch.full_like(real, 0.9))

            # How good do we recognise generated images?
            noise = torch.randn(b_size, z_dim, 1, 1, device=dev)
            fake_imgs = gen(noise)
            fake = discrim(fake_imgs.detach())
            d_loss_fake = loss(fake, torch.zeros_like(fake))

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_d.step()

            # ---- generator ----
            opt_g.zero_grad()
            fake = discrim(fake_imgs)
            # did I do a good job?
            g_loss = loss(fake, torch.ones_like(fake))

            g_loss.backward()
            opt_g.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

        n_batches = len(ldr)
        avg_d_loss = total_d_loss / n_batches
        avg_g_loss = total_g_loss / n_batches

        elapsed = time.time() - start_time
        timef = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        
        print(f"Epoch [{epoch+1}/{epochs}] {timef} | Discrim. Loss: {avg_d_loss:.4f} | Gen. Loss: {avg_g_loss:.4f}")

        with open("gan_loss.csv", "a", encoding="utf-8") as file:
            file.write(f"{epoch+1}, {avg_d_loss:.6f}, {avg_g_loss:.6f}\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device :: {device}")

    gen = Generator(z_dim=z_dim).to(device)
    discrim = Discriminator().to(device)

    optim_d = torch.optim.Adam(discrim.parameters(), lr=lr_d, betas=(0.5,0.999))
    optim_g = torch.optim.Adam(gen.parameters(), lr=lr_g, betas=(0.5,0.999))

    loss = nn.BCEWithLogitsLoss().to(device)

    # load MRI slices
    train_ldr = get_train_ldr(sys.argv[1], batch_size=batch_size)

    train_gan(
        gen=gen,
        discrim=discrim,
        loss=loss,
        opt_g=optim_g,
        opt_d=optim_d,
        ldr=train_ldr,
        epochs=epochs,
        dev=device
    )

    torch.save(gen.state_dict(), "generator_weights.pt")
    torch.save(discrim.state_dict(), "discriminator_weights.pt")

    sample(gen=gen, dev=device)

if __name__ == "__main__":
    main()