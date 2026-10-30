"""
Source code for training, validating, testing and saving DCGAN or WGAN-GP models
for HipMRI 2D slices. Results were produced with pre-structure scripts (see commits)
and moved into the correct format.
"""

import os
import json
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from dataset import load_data, load_split
from modules import Generator, Discriminator, Critic, weights_init
from pathlib import Path
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from utils import load_generator, compute_fid, nearest_train, to_rgb_01, invert_ssim
from utils import NUM_SAMPLES, BEST_CHECKPOINTS
from torchmetrics.image.fid import FrechetInceptionDistance

BASE = Path(__file__).resolve().parent

NGPU, NC, NDF = 1, 1, 64
BATCH_SIZE = 128
N_CRITIC = 5
LAMBDA_GP = 10
SAVE_EVERY = 25

# Configurations of the different models that have been tested in this project
CONFIGS = {
    "dcgan": dict(model="dcgan", epochs=50, lr=0.0002, betas=(0.5, 0.999), nz=100,
                  ngf=64, downsample=False),
    "wgan_gp": dict(model="wgan_gp", epochs=250, lr=0.0001, betas=(0.0, 0.9), nz=100,
                  ngf=64, downsample=False),
    "wgan_gp_wide": dict(model="wgan_gp", epochs=250, lr=0.0001, betas=(0.0, 0.9), nz=128,
                      ngf=128, downsample=True)
}


def setup_out_dir(name):
    """
    Set up the output directory for results of this model training.
    """
    out_dir = BASE / "outputs" / name

    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    return out_dir

def save_sample_grid(netG, noise, out_dir, epoch):
    """
    Saves a sample grid of current generator output for some noise and
    saves the image result. For purpose of viewing training progression
    and creating generator training animations.
    """
    # Sample grid from the fixed noise (same z every time, so grids are comparable)
    with torch.no_grad():
        # Noise SHOULD BE SAME (seeded) for all epochs of training
        fake = netG(noise).cpu()

    grid = vutils.make_grid(fake, nrow=8, padding=2, normalize=True, value_range=(-1, 1))
    vutils.save_image(grid, os.path.join(out_dir, "samples", f"epoch_{epoch + 1:03d}.png"))

def save_checkpoint(out_dir, prefix, epoch, state):
    """
    Saves a checkpoint with the current progress of the model, in case of
    failures mid-run or for validating at which point in epochs the model
    performed best.
    """
    state = {"epoch": epoch + 1, **state}
    path = os.path.join(out_dir, "checkpoints", f"{prefix}_epoch_{epoch + 1:03d}.pt")
    torch.save(state, path)

    print(f"Saved checkpoint at epoch {epoch + 1}")

def profile_generation(netG, nz, device):
    """
    Will return the generation (inference) time for 64 images with
    the generator at its current epoch.
    """
    training = netG.training

    netG.eval()
    with torch.no_grad():
        # Generate noise
        z = torch.randn(64, nz, 1, 1, device=device)
        netG(z)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        netG(z)

        if device.type == "cuda":
            torch.cuda.synchronize()
        gen_ms = (time.time() - t0) * 1000

    netG.train(training)
    return gen_ms

def plot_dcgan_loss(out_dir, g_losses, d_losses):
    """
    Plot the losses of the generator and discriminator across all
    iterations of training
    """

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")

    plt.plot(g_losses,label="G")
    plt.plot(d_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")

    plt.legend()
    plt.savefig(os.path.join(out_dir, "dcgan_losses.png"), dpi=150)
    plt.close()

def plot_wgan_curves(out_dir, g_losses, c_losses, W, GP, n_critic):
    """
    Plot the generator/critic losses, Wasserstein estimate and gradient
    penalty across training on one figure. Assumes batches per epoch is
    divisible by n_critic.
    """

    c_iters = range(len(c_losses))
    # critic step at which each G update happened
    g_iters = [(k + 1) * n_critic - 1 for k in range(len(g_losses))]  

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Loss plot
    axes[0].plot(c_iters, c_losses, label="Critic loss", alpha=0.6)
    axes[0].plot(g_iters, g_losses, label="Generator loss", alpha=0.6)
    axes[0].legend(); axes[0].set_ylabel("Loss")

    # Wasserstein estimate plot
    axes[1].plot(c_iters, W); axes[1].set_ylabel("W estimate")

    # Gradient penalty plot
    axes[2].plot(c_iters, GP); axes[2].set_ylabel("Gradient penalty")
    axes[2].set_xlabel("Critic iterations")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "wgan_gp_curves.png"), dpi=150)
    plt.close()

def save_resources(out_dir, gpu_name, n_params_G, n_params_D, peak_vram, epoch_times,
                   num_epochs, gen_ms):
    """
    Saves a file containing information on the resources used by the run
    including GPU, peak VRAM, mean epoch time, total training time,
    and image generation speed.
    """

    summary = (f"Device: {gpu_name}\n"
                    f"G params: {n_params_G:,} | D params: {n_params_D:,}\n"
                    f"Peak VRAM: {peak_vram:.0f} MB\n"
                    f"Mean epoch time: {sum(epoch_times) / len(epoch_times):.1f} s "
                    f"(total {sum(epoch_times) / 60:.1f} min for {num_epochs} epochs)\n"
                    f"Generate 64 images: {gen_ms:.1f} ms")
    print(summary)

    # Save text summary for later
    with open(os.path.join(out_dir, "resources.txt"), "w") as f:
        f.write(summary)

def gradient_penalty(critic, real, fake):
    """Help function that computers WGAN-GP penalty:
    mean((||grad_x D(x_hat)||_2 - 1)^2)"""
    epsilon = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    x_hat = (epsilon * real + (1 - epsilon) * fake).requires_grad_(True)
    grad = torch.autograd.grad(critic(x_hat).sum(), x_hat, create_graph=True)[0]
    return ((grad.flatten(1).norm(2, dim=1) - 1) ** 2).mean()

def train_dcgan(cfg, name):
    """
    For training the baseline DCGAN model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = setup_out_dir(name)
    nz, ngf, num_epochs = cfg["nz"], cfg["ngf"], cfg["epochs"]

    train_loader = load_data("train", batch_size=BATCH_SIZE, downsample=cfg["downsample"])

    netG = Generator(NGPU, nz, ngf, NC).to(device)
    netG.apply(weights_init)
    netD = Discriminator(NGPU, NDF, NC).to(device)
    netD.apply(weights_init)

    optimiserG = optim.Adam(netG.parameters(), lr=cfg["lr"], betas=cfg["betas"])
    optimiserD = optim.Adam(netD.parameters(), lr=cfg["lr"], betas=cfg["betas"])

    criterion = nn.BCELoss()

    # Fixed noise for the per-epoch sample grids (seeded so it is reproducible)
    fixed_noise = torch.randn(16, nz, 1, 1, generator=torch.Generator().manual_seed(0)).to(device)

    G_losses, D_losses, epoch_times = [], [], []
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    n_params_G = sum(p.numel() for p in netG.parameters())
    n_params_D = sum(p.numel() for p in netD.parameters())

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
    
        # For each batch in the dataloader
        for i, data in enumerate(train_loader, 0):
            # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()  

            ## Train with all-real batch
            # Format batch
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)

            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            
            optimiserD.step() # Update D

            # Update G network: maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimiserG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(train_loader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_times.append(time.time() - t0)
        print(f"Epoch {epoch + 1} took {epoch_times[-1]:.1f} s")

        save_sample_grid(netG, fixed_noise, out_dir, epoch)
        
        if (epoch + 1) % SAVE_EVERY == 0 or epoch == num_epochs - 1:
            save_checkpoint(out_dir, "dcgan", epoch, {
                "netG": netG.state_dict(), "netD": netD.state_dict(),
                "optG": optimiserG.state_dict(), "optD": optimiserD.state_dict(),
                "G_losses": G_losses, "D_losses": D_losses,
            })

    peak_vram = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
    gen_ms = profile_generation(netG, nz, device)
    save_resources(out_dir, gpu_name, n_params_G, n_params_D, peak_vram, epoch_times, num_epochs,
                   gen_ms)
    plot_dcgan_loss(out_dir, G_losses, D_losses)

    return netG, netD

def train_wgan_gp(cfg, name):
    """
    For training the WGAN model. Runs can be done fresh or be resumed from
    a checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = setup_out_dir(name)
    nz, ngf, num_epochs = cfg["nz"], cfg["ngf"], cfg["epochs"]

    train_loader = load_data("train", batch_size=BATCH_SIZE, downsample=cfg["downsample"])

    netG = Generator(NGPU, nz, ngf, NC).to(device)
    netG.apply(weights_init)
    netD = Critic(NGPU, NDF, NC).to(device)
    netD.apply(weights_init)

    optimiserG = optim.Adam(netG.parameters(), lr=cfg["lr"], betas=cfg["betas"])
    optimiserD = optim.Adam(netD.parameters(), lr=cfg["lr"], betas=cfg["betas"])

    # Fixed noise for the per-epoch sample grids (seeded so it is reproducible)
    fixed_noise = torch.randn(16, nz, 1, 1, generator=torch.Generator().manual_seed(0)).to(device)

    G_losses, D_losses, W_estimates, GP_values, epoch_times = [], [], [], [], []
    start_epoch = 0

    # Optionally continue an earlier run (weights, optimiser state and loss histories)
    if cfg.get("resume"):
        model = torch.load(BASE / cfg["resume"], map_location=device)

        netG.load_state_dict(model["netG"])
        netD.load_state_dict(model["netD"])
        optimiserG.load_state_dict(model["optG"])
        optimiserD.load_state_dict(model["optD"])

        G_losses, D_losses = model["G_losses"], model["D_losses"]
        GP_values, W_estimates = model["GP_values"], model["W_estimates"]
        start_epoch = model["epoch"]

        print(f"Resumed from {cfg['resume']} at epoch {start_epoch}")

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    n_params_G = sum(p.numel() for p in netG.parameters())
    n_params_D = sum(p.numel() for p in netD.parameters())

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"Training {name}: epochs {start_epoch}->{num_epochs} on {gpu_name}")
    for epoch in range(start_epoch, num_epochs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        for i, data in enumerate(train_loader):
            # Critic update: D(G(z)) - D(x) + lambda * GP
            real = data.to(device)
            b_size = real.size(0)
            fake = netG(torch.randn(b_size, nz, 1, 1, device=device)).detach()

            gp = gradient_penalty(netD, real, fake)
            d_real = netD(real).mean()
            d_fake = netD(fake).mean()
            loss_D = d_fake - d_real + LAMBDA_GP * gp

            optimiserD.zero_grad()
            loss_D.backward()
            optimiserD.step()

            W_estimates.append((d_real - d_fake).item())
            GP_values.append(gp.item())
            D_losses.append(loss_D.item())

            # Generator update every N_CRITIC critic steps: minimise -D(G(z))
            if (i + 1) % N_CRITIC == 0:
                loss_G = -netD(netG(torch.randn(b_size, nz, 1, 1, device=device))).mean()
                optimiserG.zero_grad()
                loss_G.backward()
                optimiserG.step()
                G_losses.append(loss_G.item())

            if i % 50 == 0:
                print(f"[{epoch}/{num_epochs}][{i}/{len(train_loader)}]  "
                      f"Loss_D: {loss_D.item():.4f}  W: {W_estimates[-1]:.4f}  GP: {gp.item():.4f}")

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.time() - t0)
        print(f"Epoch {epoch + 1} took {epoch_times[-1]:.1f} s")

        save_sample_grid(netG, fixed_noise, out_dir, epoch)

        if (epoch + 1) % SAVE_EVERY == 0 or epoch == num_epochs - 1:
            save_checkpoint(out_dir, "wgan_gp", epoch, {
                "netG": netG.state_dict(), "netD": netD.state_dict(),
                "optG": optimiserG.state_dict(), "optD": optimiserD.state_dict(),
                "G_losses": G_losses, "D_losses": D_losses,
                "GP_values": GP_values, "W_estimates": W_estimates,
            })

    peak_vram = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
    gen_ms = profile_generation(netG, nz, device)
    save_resources(out_dir, gpu_name, n_params_G, n_params_D, peak_vram, epoch_times, num_epochs, 
                   gen_ms)
    plot_wgan_curves(out_dir, G_losses, D_losses, W_estimates, GP_values, N_CRITIC)
    return netG, netD

def validate(cfg, name):
    """
    For a given model, returns the "best" saved checkpoint by computing FID
    score at each checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = setup_out_dir(name)

    ref = load_split("validate", downsample=cfg["downsample"]).to(device)
    
    fid = FrechetInceptionDistance(feature=2048, normalize=True,
                                    reset_real_features=False).to(device)
    
    for s in range(0, ref.size(0), 100):
        fid.update(to_rgb_01(ref[s:s + 100]), real=True)

    fid_results = {}

    # For each checkpoint in the folder, validate
    checkpoint_path = out_dir / "checkpoints"
    # Go through them in order
    for file_path in sorted(checkpoint_path.glob("*.pt")):
        netG = load_generator(file_path, device, nz=cfg["nz"], ngf=cfg["ngf"])
        torch.manual_seed(0) # Same noise for all checkpoints
        
        with torch.no_grad():
            gen = netG(torch.randn(NUM_SAMPLES, cfg["nz"], 1, 1, device=device))

        fid_results[file_path.name] = compute_fid(fid, gen)
        print(f"{file_path.name}: FID {fid_results[file_path.name]:.4f}")

    best_model_fid = min(fid_results, key=fid_results.get) # best FID is small
    print(f"\nBest checkpoint: {best_model_fid} (val FID {fid_results[best_model_fid]:.4f})")

    with open(BASE / "outputs" / f"val_results_{name}.json", "w") as f:
        json.dump({"fid": fid_results}, f, indent=2)

    return best_model_fid, fid_results

def test(cfg, name):
    """
    Will test the best checkpoint for a model (BEST_CHECKPOINTS from utils.py,
    should be validated with validate()) for ssim, fid, nn audit. Prints results,
    saves results to json, and plots NN histogram metric.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = setup_out_dir(name)

    train = load_split("train", downsample=cfg["downsample"]).to(device)
    train_flat = train.flatten(1)
    ref = load_split("test", downsample=cfg["downsample"]).to(device)

    # Get FID for real test slices once
    fid = FrechetInceptionDistance(feature=2048, normalize=True,
                                            reset_real_features=False).to(device)
    for s in range(0, ref.size(0), 100):
        fid.update(to_rgb_01(ref[s:s + 100]), real=True)

    # Get baseline references for NN and FID from real data
    torch.manual_seed(0)
    idx = torch.randperm(train.size(0))[:ref.size(0)]
    fid_floor = compute_fid(fid, train[idx])
    test_d, _ = nearest_train(ref, train_flat)
    mem_threshold = torch.quantile(test_d, 0.01)

    test_results = {}      
    netG = load_generator(out_dir / "checkpoints" / BEST_CHECKPOINTS[name],
                          device=device, nz=cfg["nz"], ngf=cfg["ngf"])

    # Same noise as validity testing
    torch.manual_seed(0)
    with torch.no_grad():
        # Want to generate for size of test set
        gen = netG(torch.randn(ref.size(0), cfg["nz"], 1, 1, device=device))

    fid_score = compute_fid(fid, gen)
    print(f"{name}: FID {fid_score:.2f} (real-data floor {fid_floor:.2f})")

    # Save results in dict for log/ease of find later
    test_results[name] = {"checkpoint": BEST_CHECKPOINTS[name], "fid": fid_score,
                          "fid_real_floor": fid_floor,
                          "data": "downsampled" if cfg["downsample"] else "centre crop"}

    # GAN-inversion SSIM: paired reconstruction of each unseen test slice (target > 0.6)
    _, inv_scores = invert_ssim(netG, ref, nz=cfg["nz"])
    test_results[name].update({"inv_ssim_mean": inv_scores.mean().item(),
                                "inv_ssim_std": inv_scores.std().item(),
                                "inv_frac_above_0.6": (inv_scores > 0.6).float().mean().item()})
    
    print(f"{name}: inversion SSIM {inv_scores.mean():.4f} ± {inv_scores.std():.4f} "
            f"({(inv_scores > 0.6).float().mean():.1%} of slices > 0.6)")

    # Nearest-neighbour memorisation audit
    gen_d, _ = nearest_train(gen, train_flat)
    frac_close = (gen_d < mem_threshold).float().mean().item()
    test_results[name].update({"nn_gen_median": gen_d.median().item(),
                                "nn_test_median": test_d.median().item(),
                                "nn_frac_below_test_p1": frac_close})
    
    print(f"{name}: NN RMSE median gen {gen_d.median():.3f} vs test {test_d.median():.3f} | "
            f"{frac_close:.1%} closer than 99% of real test slices")

    plt.figure(figsize=(7, 4))
    plt.hist(test_d.cpu(), bins=40, alpha=0.6, density=True, label="Real test → train")
    plt.hist(gen_d.cpu(), bins=40, alpha=0.6, density=True, label=f"{name} → train")

    plt.axvline(mem_threshold.item(), color="k", ls="--", label="Test 1st percentile")
    plt.xlabel("RMSE to nearest training slice")
    plt.ylabel("Density")

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"nn_hist_{name}.png", dpi=150)
    plt.close()

    # Save results from model evaluation
    with open(out_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

if __name__ == "__main__":
    # For ease of running/reproduction add main that allows running from the command line
    parser = argparse.ArgumentParser(description="Train DCGAN or WGAN-GP on HipMRI 2D slices.")

    parser.add_argument("--run", required=True, choices=CONFIGS.keys(),
                     help="Which configuration from configs to train.")
    parser.add_argument("--mode", default="train", choices=["train", "validate", "test"])
    args = parser.parse_args()

    # Can run DCGAN or the different versions of WGAN
    if args.mode == "train":
        cfg = CONFIGS[args.run]
        if cfg["model"] == "wgan_gp":
            train_wgan_gp(cfg, args.run)

        elif cfg["model"] == "dcgan":
            train_dcgan(cfg, args.run)
    # Can choose to validate existing training
    elif args.mode == "validate":
        validate(CONFIGS[args.run], args.run)
        
    elif args.mode == "test":
        test(CONFIGS[args.run], args.run)