"""
Contains helper functions that are used in both train.py as well as
predict.py, mostly to assist with validating already trained models.
"""

import torch
from modules import Generator
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

NUM_SAMPLES = 256

BEST_CHECKPOINTS = {"dcgan": "dcgan_epoch_050.pt",
        "wgan_gp": "wgan_gp_epoch_240.pt",
        "wgan_gp_wide": "wgan_gp_epoch_250.pt",
}

def load_generator(path, device, nz, ngf):
    """
    Loads the generator data from a saved model.
    """
    # Get the saved model
    model = torch.load(path, map_location=device)
    print(path, "keys:", list(model.keys()), "epoch:", model["epoch"])

    netG = Generator(ngpu=1, nz=nz, ngf=ngf, nc=1).to(device)
    netG.load_state_dict(model["netG"])
    netG.eval()

    return netG

def invert_ssim(netG, ref, nz, steps=500, lr=0.05, z_reg=1e-3, batch=64, seed=0):
    """
    Compute a GAN inversion. For each slice, optimise a latent z so that G(z) approximates
    slice, then compute the paired SSIM between the slice and its reconstruction G(z). SSIM
    is done this way because MRI slices are too varied to get valuable SSIM score
    from finding maximum SSIM against test slices (and would reward memorisation).
    """

    netG.eval()
    for p in netG.parameters():
        p.requires_grad_(False) # only z is optimised, not the generator

    torch.manual_seed(seed)

    recons, scores = [], []
    for s in range(0, ref.size(0), batch):
        x = ref[s:s + batch]
        z = torch.randn(x.size(0), nz, 1, 1, device=x.device, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)

        for _ in range(steps):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(netG(z), x) + z_reg * z.pow(2).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            r = netG(z)
            recons.append(r)
            scores.append(ssim(r, x, data_range=2.0, reduction=None))

    for p in netG.parameters():
        p.requires_grad_(True)

    return torch.cat(recons), torch.cat(scores)

# Need to change images from [-1, 1] grayscale to [0, 1] RGB
# Repeat gray channel 3 times to have 3 channels for RGB
to_rgb_01 = lambda x: ((x + 1) / 2).clamp(0, 1).repeat(1, 3, 1, 1)

def compute_fid(fid, gen, batch=100):
    """
    Compute FID using the pre-trained FID from torchmetrics (this
    was approved on Ed).
    """
    fid.reset() # Clear generated features

    for s in range(0, gen.size(0), batch):
        fid.update(to_rgb_01(gen[s:s + batch]), real=False)

    return fid.compute().item()

def nearest_train(x, train_flat, chunk=256):
    """
    Nearest neighbours audit, finds the shortest distance between
    the provided image x and training images.
    """
    x_flat = x.flatten(1)
    dists, idxs = [], []
    for s in range(0, x_flat.size(0), chunk):
        # Find shortest distance between image x and training images
        d = torch.cdist(x_flat[s:s + chunk], train_flat)
        m, i = d.min(dim=1)
        dists.append(m)
        idxs.append(i)

    # Per-pixel RMSE
    return torch.cat(dists) / x_flat.size(1) ** 0.5, torch.cat(idxs)