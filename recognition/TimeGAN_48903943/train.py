import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import get_datasets
from modules import TimeVAE, vae_loss, TimeGAN

# a single epoch for TimeVAE
def run_epoch(model, loader, device, beta, optimizer=None):
    training = optimizer is not None
    model.train(training)                 # BatchNorm: batch stats vs running stats
    sums, n = torch.zeros(3), 0
    with torch.set_grad_enabled(training):
        for x, _ in loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta) # loss function
            if training:
                optimizer.zero_grad()
                loss.backward()     # backprop
                optimizer.step()    # update weights
            b = x.size(0)
            sums += torch.tensor([loss.item(), recon.item(), kl.item()]) * b
            n += b
    return (sums / n).tolist()


def vae():
    num_epochs, beta, lr = 100, 1.0, 1e-3

    # local
    # train_set, val_set, test_set, norm = get_datasets("order_sample.csv")

    # rangpur
    DATA_PATH = os.path.expanduser("~/lobster/data/order_sample.csv")
    train_set, val_set, test_set, norm = get_datasets(DATA_PATH)

    # load 64 windows at a time
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeVAE(latent_dim=128).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    checkpoint_dir = os.path.expanduser("~/baselineOutputs")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # fitted on train only, predict.py reloads it
    norm.save(os.path.join(checkpoint_dir, "normaliser.joblib"))

    history = {"train": [], "val": []}
    best_val = float("inf")

    # training loop
    for epoch in range(num_epochs):
        tr = run_epoch(model, train_loader, device, beta, optimizer)
        va = run_epoch(model, val_loader, device, beta)
        history["train"].append(tr)
        history["val"].append(va)

        # save best validation result
        if va[0] < best_val:
            best_val = va[0]
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "timevae_best.pt"))
        print(f"Epoch {epoch+1}/{num_epochs} | train {tr[0]:.1f} "
              f"(recon {tr[1]:.1f}, kl {tr[2]:.1f}) | val {va[0]:.1f} "
              f"(recon {va[1]:.1f}, kl {va[2]:.1f})")

    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "timevae_best.pt"),
                                     map_location=device))
    te = run_epoch(model, test_loader, device, beta)
    print(f"Test (best-val checkpoint): total {te[0]:.1f}, recon {te[1]:.1f}, kl {te[2]:.1f}")

    # plot loss curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, k, name in zip(axes, (1, 2), ("Reconstruction", "KL")):
        ax.plot([h[k] for h in history["train"]], label="train")
        ax.plot([h[k] for h in history["val"]], label="val")
        ax.set_title(name); ax.set_xlabel("epoch"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(checkpoint_dir, "loss_curves.png"))


#------------------------------TimeGAN--------------------------------------

# train embedder and recovery network to transform from real LOB events into latent
# representations and back
def pretrain_embedder(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0

    for x, _ in loader:
        x = x.to(device)

        with torch.set_grad_enabled(training):
            # encode real LOB events into latent space and reconstruct them
            x_hat = model(x)

            # reconstruction loss measures difference between real and recovered events
            loss = torch.nn.functional.mse_loss(x_hat, x)

            if training:
                optimizer.zero_grad()
                loss.backward()     # backprop
                optimizer.step()    # update weights

        total_loss += loss.item()

    return total_loss / len(loader)

# Train supervisor to predict next-event latent representations
def pretrain_supervisor(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0

    for x, _ in loader:
        x = x.to(device)

        # extract real latent representations without updating the Embedder
        with torch.no_grad():
            h = model.embedder(x)

        with torch.set_grad_enabled(training):
            # predict latent states using the Supervisor
            h_hat = model.supervisor(h)

            # measure difference between predicted state vs actual next state
            loss = torch.nn.functional.mse_loss(h_hat[:, :-1, :], h[:, 1:, :])

            if training:
                optimizer.zero_grad()
                loss.backward()     # backprop
                optimizer.step()    # update weights

        total_loss += loss.item()

    return total_loss / len(loader)

# -------------------------------- TimeGAN ver 2 improvement --------------------------------------
# Penalise price gaps < 1 tick
def price_gap_penalty(x_fake, gap_mean, gap_scale):
    # Convert standardised generated gaps back to log values
    log_gaps = x_fake[:, :, :18] * gap_scale + gap_mean

    # Penalise gaps below one tick (log-gap < 0)
    return torch.relu(-log_gaps).mean()

# -------------------------------- TimeGAN ver 3 improvement --------------------------------------
# Penalise extremely large price gaps
def extreme_gap_penalty(x_fake, gap_mean, gap_scale, upper_threshold):
    # Convert standardised generated gaps back to log values
    log_gaps = x_fake[:, :, :18] * gap_scale + gap_mean

    # Upper threshold in log space
    log_upper = torch.log(
        torch.tensor(upper_threshold, device=x_fake.device, dtype=x_fake.dtype)
    )

    # Penalise gaps above the upper threshold
    return torch.relu(log_gaps - log_upper).mean()

# -------------------------------------------------------------------------------------------------

# Calculate Generator loss to encourage realistic synthetic LOB sequences
def generator_loss(model, x, z, gap_mean, gap_scale, upper_threshold, gamma=1.0):
    # Encode real sequences without updating the Embedder
    with torch.no_grad():
        h = model.embedder(x)

    # Generate synthetic latent sequences and recover synthetic LOB events
    h_fake_raw = model.generator(z)
    h_fake = model.supervisor(h_fake_raw)
    x_fake = model.recovery(h_fake)

    # Discriminator evaluates both supervised and raw generated latent sequences
    fake_logits = model.discriminator(h_fake)
    fake_raw_logits = model.discriminator(h_fake_raw)

    criterion = torch.nn.BCEWithLogitsLoss()

    # Adversarial losses encourage generated sequences to be classified as real
    loss_adv = criterion(fake_logits, torch.ones_like(fake_logits))
    loss_adv_raw = criterion(fake_raw_logits, torch.ones_like(fake_raw_logits))

    # Supervised loss encourages accurate next-timestep latent predictions
    loss_s = torch.nn.functional.mse_loss(
        model.supervisor(h)[:, :-1, :], h[:, 1:, :])

    # Calculate feature-wise statistics across batches and timesteps
    real_mean = x.mean(dim=(0, 1))
    fake_mean = x_fake.mean(dim=(0, 1))

    real_std = x.std(dim=(0, 1), unbiased=False)
    fake_std = x_fake.std(dim=(0, 1), unbiased=False)

    # Moment matching encourages similar means and standard deviations
    # between real and generated LOB features
    loss_m = (
            torch.mean(torch.abs(real_mean - fake_mean))
            + torch.mean(torch.abs(real_std - fake_std))
    )

    # Combine adversarial, supervised, moment matching and price gap losses
    loss_min_gap = price_gap_penalty(x_fake, gap_mean, gap_scale)
    loss_extreme_gap = extreme_gap_penalty(x_fake, gap_mean, gap_scale, upper_threshold)

    # calculate total generator loss
    loss_g = (
            loss_adv
            + gamma * loss_adv_raw
            + 100 * torch.sqrt(loss_s + 1e-8)
            + 100 * loss_m
            + 100 * loss_min_gap
            + 100 * loss_extreme_gap
    )

    # diagnostic print statements for loss contributions
    if torch.rand(()) < 0.001:
        print(f"Weighted lower gap penalty: "
              f"{(100 * loss_min_gap).item():.6f}")
        print(f"Weighted upper gap penalty: "
              f"{(100 * loss_extreme_gap).item():.6f}")
        print(f"Moment contribution: {(100 * loss_m).item():.6f}")

    return loss_g

# Calculate Discriminator loss to distinguish real and synthetic latent sequences
def discriminator_loss(model, x, z, gamma=1.0):

    # Prepare real and generated latent sequences without updating their networks
    with torch.no_grad():
        h = model.embedder(x)

        # raw genertor outputs
        h_fake_raw = model.generator(z)

        # pass them through supervisor to learn temporal relationships
        h_fake = model.supervisor(h_fake_raw)

    # Classify real latent sequences and both types of generated sequences
    real_logits = model.discriminator(h)
    fake_logits = model.discriminator(h_fake)
    fake_raw_logits = model.discriminator(h_fake_raw)

    criterion = torch.nn.BCEWithLogitsLoss()

    # Real sequences are labelled 1, generated sequences are labelled 0
    loss_real = criterion(real_logits, torch.ones_like(real_logits))
    loss_fake = criterion(fake_logits, torch.zeros_like(fake_logits))
    loss_fake_raw = criterion(fake_raw_logits, torch.zeros_like(fake_raw_logits))

    # Combined binary classification loss
    loss_d = loss_real + loss_fake + gamma * loss_fake_raw

    return loss_d

# Join adversarial training using alternating Generator, Embedder and Discriminator updates
def joint_train_epoch(model, loader, device,
                      generator_optimizer,
                      embedder_optimizer,
                      discriminator_optimizer,
                      gap_mean, gap_scale,
                      upper_threshold):
    model.train()

    total_g_loss = 0
    total_e_loss = 0
    total_d_loss = 0

    for x, _ in loader:
        x = x.to(device)

        # Update Generator/Supervisor and Embedder/Recovery twice per batch
        # for every Discriminator update
        for _ in range(2):
            z = torch.randn(x.size(0), x.size(1), model.noise_dim, device=device)

            # Generator/Supervisor update
            model.discriminator.requires_grad_(False)
            model.recovery.requires_grad_(False)
            model.supervisor.requires_grad_(True)

            generator_optimizer.zero_grad()

            # generator loss
            loss_g = generator_loss(model, x, z, gap_mean, gap_scale, upper_threshold)
            loss_g.backward()           # backprop
            generator_optimizer.step()  # update generator and supervisor weights

            total_g_loss += loss_g.item()

            # Embedder/Recovery update
            model.supervisor.requires_grad_(False)
            model.recovery.requires_grad_(True)

            embedder_optimizer.zero_grad()
            h = model.embedder(x)
            x_hat = model.recovery(h)

            # reconstruction loss
            loss_r = torch.nn.functional.mse_loss(x_hat, x)

            # supervisor loss
            loss_s = torch.nn.functional.mse_loss(
                model.supervisor(h)[:, :-1, :], h[:, 1:, :])

            # combined loss
            loss_e = 10 * torch.sqrt(loss_r + 1e-8) + 0.1 * loss_s

            loss_e.backward()           # backprop
            embedder_optimizer.step()   # update embedder and recovery network weights

            total_e_loss += loss_e.item()

        # Discriminator update
        model.supervisor.requires_grad_(True)
        model.discriminator.requires_grad_(True)
        z = torch.randn(x.size(0), x.size(1), model.noise_dim, device=device)
        discriminator_optimizer.zero_grad()

        # Train Discriminator to distinguish real and synthetic latent sequences
        loss_d = discriminator_loss(model, x, z)    # discriminator loss
        loss_d.backward()               # backprop
        discriminator_optimizer.step()  # update discriminator weights

        total_d_loss += loss_d.item()

    # Return average losses across all batches and update steps
    return (
        total_g_loss / (2 * len(loader)),
        total_e_loss / (2 * len(loader)),
        total_d_loss / len(loader)
    )

# Validation step
def validate_timegan(model, loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)

            # Generate random noise matching real sequence dimensions
            z = torch.randn(
                x.size(0), x.size(1), model.noise_dim,
                device=device
            )

            # Generate synthetic LOB sequences
            x_fake = model.generate(z)

            # compare real vs fake distributions of each feature (of 40)
            real_mean = x.mean(dim=(0, 1))
            fake_mean = x_fake.mean(dim=(0, 1))

            real_std = x.std(dim=(0, 1), unbiased=False)
            fake_std = x_fake.std(dim=(0, 1), unbiased=False)

            loss = (
                torch.mean(torch.abs(real_mean - fake_mean))
                + torch.mean(torch.abs(real_std - fake_std))
            )

            total_loss += loss.item()

    return total_loss / len(loader)

def timegan():
    lr = 1e-3
    embedder_epochs = 50
    supervisor_epochs = 50
    joint_epochs = 100

    DATA_PATH = os.path.expanduser("~/lobster/data/order_sample.csv")
    train_set, val_set, test_set, norm = get_datasets(DATA_PATH)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeGAN().to(device)
    checkpoint_dir = os.path.expanduser("~/TimeGAN_gap_penalty")
    os.makedirs(checkpoint_dir, exist_ok=True)
    norm.save(os.path.join(checkpoint_dir, "normaliser.joblib"))

    embedder_optimizer = Adam(
        list(model.embedder.parameters()) +
        list(model.recovery.parameters()),
        lr=lr
    )

    # embedder-recovery pretraining
    for epoch in range(embedder_epochs):
        train_loss = pretrain_embedder(model, train_loader, device, embedder_optimizer)
        val_loss = pretrain_embedder(model, val_loader, device)
        print(f"Embedder Epoch {epoch + 1}: Train={train_loss:.6f}, Val={val_loss:.6f}")

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "timegan_embedder.pt"))

    supervisor_optimizer = Adam(
        model.supervisor.parameters(),
        lr=lr
    )

    # supervisor pretraining
    for epoch in range(supervisor_epochs):
        train_loss = pretrain_supervisor(model, train_loader, device, supervisor_optimizer)
        val_loss = pretrain_supervisor(model, val_loader, device)

        print(f"Supervisor Epoch {epoch + 1}: Train={train_loss:.6f}, Val={val_loss:.6f}")

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "timegan_supervisor.pt"))

    generator_optimizer = Adam(
        list(model.generator.parameters()) +
        list(model.supervisor.parameters()),
        lr=1e-4
    )

    discriminator_optimizer = Adam(
        model.discriminator.parameters(),
        lr=1e-4
    )

    embedder_optimizer = Adam(
        list(model.embedder.parameters()) +
        list(model.recovery.parameters()),
        lr=1e-4
    )

    gap_mean = torch.tensor(
        norm.p_scaler.mean_,
        dtype=torch.float32,
        device=device
    )

    gap_scale = torch.tensor(
        norm.p_scaler.scale_,
        dtype=torch.float32,
        device=device
    )

    train_gaps = np.exp(
        norm.p_scaler.inverse_transform(train_set.Z.numpy()[:, :18])
    )

    # get upper price gap threshold for TimeGAN v3 penalty
    upper_threshold = np.quantile(train_gaps, 0.999)

    best_val_loss = float("inf")

    # joint adversarial training
    for epoch in range(joint_epochs):
        g_loss, e_loss, d_loss = joint_train_epoch(
            model,
            train_loader,
            device,
            generator_optimizer,
            embedder_optimizer,
            discriminator_optimizer,
            gap_mean, gap_scale,
            upper_threshold
        )

        val_loss = validate_timegan(model, val_loader, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "timegan_best.pt"))

        print(
            f"Joint Epoch {epoch + 1}: "
            f"G={g_loss:.6f}, "
            f"E={e_loss:.6f}, "
            f"D={d_loss:.6f}, "
            f"Val={val_loss:.6f}"
        )

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "timegan_final.pt"))

def main():
    model_type = "timegan"  # Change to "vae" for baseline

    if model_type == "vae":
        vae()
    elif model_type == "timegan":
        timegan()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    main()