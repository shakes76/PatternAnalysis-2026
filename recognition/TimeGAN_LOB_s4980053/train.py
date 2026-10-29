# train.py - training and saving the baseline GAN and TimeGAN
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import time

from dataset import make_loaders, TICKERS
from modules import BaselineGAN, TimeGAN, count_parameters
from utils import collect, invalid_rates, per_stock_report, windows_to_books
# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

BASELINE_EPOCHS = 50   # baseline GAN: settles by ~epoch 25
AE_EPOCHS = 100        # TimeGAN phase 1
SUP_EPOCHS = 100       # TimeGAN phase 2: already 0.0006 next-step error; 500 added an hour for little gain
JOINT_EPOCHS = 100      # TimeGAN phase 3
LR = 1e-3              # learning rate: how big each dial turn is (TimeGAN's paper uses 1e-3)
BATCH_SIZE = 128
SEQ_LEN = 500
HIDDEN_DIM = 24        # GRU memory size
NUM_LAYERS = 3
SEED = 42              # fixed randomness, so a run can be repeated exactly
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"    # GPU if there is one
SMOKE_TEST = os.environ.get("SMOKE_TEST", "1") == "1"   # default: smoke test. Real run: submit with SMOKE_TEST=0
MODEL = os.environ.get("MODEL", "timegan")              # "baseline" or "timegan", chosen when submitting
PENALTY = float(os.environ.get("PENALTY", "0"))          # weight of the rule penalty; 0 = off (submit with PENALTY=1 to turn on)
RESUME = os.environ.get("RESUME", "")                    # e.g. timegan_phase2.pt: skip phases 1-2 and start phase 3 from it

def bce(scores, real):
    """Binary cross-entropy against an all-'real' (1) or all-'fake' (0) target."""
    target = torch.ones_like(scores) if real else torch.zeros_like(scores)
    return F.binary_cross_entropy_with_logits(scores, target)

def moment_loss(fake, real):
    """How far the fake batch's average and spread are from the real batch's (per time step and feature)."""
    mean_gap = (fake.mean(dim=0) - real.mean(dim=0)).abs().mean()
    std_gap = (torch.sqrt(fake.var(dim=0) + 1e-6) - torch.sqrt(real.var(dim=0) + 1e-6)).abs().mean()
    return mean_gap + std_gap

def rule_penalty(fake, mean_t, std_t, max_log_size):
    """How badly generated windows break the book's rules, as a number the generator can learn from.
    0 when every rule holds; grows with how far (in ticks / log-shares) each rule is broken."""
    x = fake * std_t + mean_t                                          # undo the normalisation (torch tensors, same formula as decode)
    ask_off, bid_off, log_sizes = x[..., 1:11], x[..., 11:21], x[..., 21:41]

    spread = torch.relu(1 - (ask_off[..., 0] - bid_off[..., 0])).mean()     # best ask must sit >= 1 tick above best bid
    ask_ladder = torch.relu(1 - (ask_off[..., 1:] - ask_off[..., :-1])).mean()   # each ask level >= 1 tick above the last
    bid_ladder = torch.relu(1 - (bid_off[..., :-1] + bid_off[..., 1:])).mean()                                # mirror: each bid level >= 1 tick BELOW the last
    volume = torch.relu(log_sizes - max_log_size).mean()                    # no log size above the real maximum
    return spread + ask_ladder + bid_ladder + volume

# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------
def train_baseline(model, loader, epochs, lr, device, max_batches=None, penalty=None):
    """Plain GAN training: per batch, a discriminator turn (real -> 1, fake -> 0) then a generator turn (fool it)."""
    g_opt = torch.optim.Adam(model.generator.parameters(), lr=lr)
    d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=lr)
    history = {"d_loss": [], "g_loss": []}

    for epoch in range(1, epochs + 1):
        d_total, g_total, n = 0.0, 0.0, 0
        for real_window, stock in loader:
            real_window = real_window.to(device)
            stock = stock.to(device)
            noise = model.sample_noise(real_window.shape[0], real_window.shape[1], device)
            fake = model.make_fake(noise, stock)

            # discriminator catches fakes
            # generator dials → fake, so we fake.detach() → discriminator → fake_scores → d_loss
            real_scores = model.judge(real_window, stock)
            fake_scores = model.judge(fake.detach(), stock)
            d_loss = F.binary_cross_entropy_with_logits(real_scores, torch.ones_like(real_scores)) \
                   + F.binary_cross_entropy_with_logits(fake_scores, torch.zeros_like(fake_scores))
            d_opt.zero_grad() 
            d_loss.backward() 
            d_opt.step() 

            # generator's turn: fool the discriminator
            # generator dials → fake → discriminator → g_scores → g_loss -> loop backward() to generator dials
            g_scores = model.judge(fake, stock)  # NOT detached: the trace must reach the generator
            g_loss = F.binary_cross_entropy_with_logits(g_scores, torch.ones_like(g_scores))                          # g_scores vs target 1
            if penalty is not None:
                g_loss = g_loss + penalty(fake)                       # punish broken books
            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            d_total += d_loss.item()
            g_total += g_loss.item()
            n += 1
            if max_batches is not None and n == max_batches:
                break                             # smoke test: stop early

        history["d_loss"].append(d_total / n)
        history["g_loss"].append(g_total / n)
        print(f"epoch {epoch}/{epochs}  d_loss {d_total / n:.4f}  g_loss {g_total / n:.4f}")

    return history

def train_autoencoder(model, loader, epochs, lr, device, max_batches=None):
    """TimeGAN phase 1: embedder + recovery learn to zip and unzip real windows."""
    params = list(model.embedder.parameters()) + list(model.recovery.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    history = []

    for epoch in range(1, epochs + 1):
        total, n = 0.0, 0
        for real, stock in loader:
            real = real.to(device)
            rebuilt = model.reconstruct(real)                           # zip + unzip
            loss = F.mse_loss(rebuilt, real)        # mean squared error
            opt.zero_grad()
            loss.backward()
            opt.step()                                     

            total += loss.item()
            n += 1
            if max_batches is not None and n == max_batches:
                break

        history.append(total / n)
        print(f"[autoencoder] epoch {epoch}/{epochs}  reconstruction mse {total / n:.4f}")

    return history

def train_supervisor(model, loader, epochs, lr, device, max_batches=None):
    """TimeGAN phase 2: the supervisor learns to predict the next latent code. Embedder frozen."""
    opt = torch.optim.Adam(model.supervisor.parameters(), lr=lr)              # ONLY the supervisor's dials
    history = []

    for epoch in range(1, epochs + 1):
        total, n = 0.0, 0
        for real, stock in loader:
            real = real.to(device)
            stock = stock.to(device)
            with torch.no_grad():                      # frozen: no trace into the embedder
                codes = model.embedder(real)                            # zip the real window
            guesses = model.supervise(codes, stock)                              # supervisor on the codes
            loss = F.mse_loss(guesses[:, :-1], codes[:, 1:])   # guess for t+1 vs the real code at t+1
            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            n += 1
            if max_batches is not None and n == max_batches:
                break

        history.append(total / n)
        print(f"[supervisor] epoch {epoch}/{epochs}  next-step mse {total / n:.4f}")

    return history

def train_joint(model, loader, epochs, lr, device, max_batches=None, gamma=1.0, penalty=None, save_fn=None):
    """TimeGAN phase 3: generator, supervisor, zip tool and discriminator train together."""
    g_params = list(model.generator.parameters()) + list(model.supervisor.parameters())
    e_params = list(model.embedder.parameters()) + list(model.recovery.parameters())
    g_opt = torch.optim.Adam(g_params, lr=lr)
    e_opt = torch.optim.Adam(e_params, lr=lr)
    d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=lr)
    history = {"g_loss": [], "e_loss": [], "d_loss": []}

    for epoch in range(1, epochs + 1):
        g_total, e_total, d_total, n = 0.0, 0.0, 0.0, 0
        for real, stock in loader:
            real = real.to(device)
            stock = stock.to(device)

            for _ in range(2):                                     # turns 1 and 2 happen twice
                # --- turn 1: generator + supervisor ---
                noise = model.sample_noise(real.shape[0], real.shape[1], device)
                fake_raw = model.make_codes(noise, stock)           # generator's fake codes
                fake = model.supervise(fake_raw, stock)             # after the supervisor
                fake_windows = model.recovery(fake)                 # unzipped into features
                real_codes = model.embedder(real)
                adv = bce(model.judge(fake, stock), True) + gamma * bce(model.judge(fake_raw, stock), True)
                sup = F.mse_loss(model.supervise(real_codes, stock)[:, :-1], real_codes[:, 1:])
                g_loss = adv + 100 * torch.sqrt(sup + 1e-8) + 100 * moment_loss(fake_windows, real)
                if penalty is not None:
                    g_loss = g_loss + penalty(fake_windows)           # punish broken books
                g_opt.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(g_params, 1.0)                # cap the update size (NaN safety)
                g_opt.step()

                # --- turn 2: "zip tool" ---
                real_codes = model.embedder(real)
                recon = F.mse_loss(model.recovery(real_codes), real)                                         # mse between recovery(real_codes) and real
                sup = F.mse_loss(model.supervise(real_codes, stock)[:, :-1], real_codes[:, 1:])
                e_loss = 10 * torch.sqrt(recon + 1e-8) + 0.1 * sup
                e_opt.zero_grad()
                e_loss.backward()
                torch.nn.utils.clip_grad_norm_(e_params, 1.0)                # cap the update size (NaN safety)
                e_opt.step()

            # --- turn 3: discriminator (once per batch) ---
            with torch.no_grad():                                   # the judge's turn: nothing else learns
                real_codes = model.embedder(real)
                fake_raw = model.make_codes(noise, stock)
                fake = model.supervise(fake_raw, stock)
            d_loss = bce(model.judge(real_codes, stock), True) + bce(model.judge(fake, stock), False) \
                   + gamma * bce(model.judge(fake_raw, stock), False)
            if d_loss.item() > 0.15:                                # skip if already winning
                d_opt.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), 1.0)   # cap the update size (NaN safety)
                d_opt.step()

            g_total += g_loss.item()
            e_total += e_loss.item()
            d_total += d_loss.item()
            n += 1
            if not torch.isfinite(g_loss + e_loss + d_loss):
                raise RuntimeError(f"NaN/inf loss at joint epoch {epoch}, batch {n}")
            
            elif max_batches is not None and n == max_batches:
                break

        for key, total in (("g_loss", g_total), ("e_loss", e_total), ("d_loss", d_total)):
            history[key].append(total / n)
        print(f"[joint] epoch {epoch}/{epochs}  g {g_total / n:.4f}  e {e_total / n:.4f}  d {d_total / n:.4f}")
        if save_fn is not None and epoch % 10 == 0:
            save_fn(f"ep{epoch}")                    # keep a copy every 10 epochs, so the best one can be picked later

    return history

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(SEED)

    # data (shared by both models)
    train_loader, val_loader, test_loader, mean, std = make_loaders(seq_len=SEQ_LEN, batch_size=BATCH_SIZE)    
    feature_dim = mean.shape[0]                                # 41: one mean per feature
    max_batches = 5 if SMOKE_TEST else None

    # rule penalty (off unless PENALTY > 0): needs mean/std as tensors and the largest real log size
    mean_t = torch.tensor(mean, dtype=torch.float32, device=DEVICE)
    std_t = torch.tensor(std, dtype=torch.float32, device=DEVICE)
    train_rows = torch.cat([d.data for d in train_loader.dataset.datasets]).numpy()   # every normalised training row
    max_log_size = float((train_rows[:, 21:41] * std[21:41] + mean[21:41]).max())    # biggest real log size
    penalty = (lambda w: PENALTY * rule_penalty(w, mean_t, std_t, max_log_size)) if PENALTY > 0 else None
    suffix = "_penalty" if PENALTY > 0 else ""                                         # keep penalty runs' files separate

    if MODEL == "baseline":
        model = BaselineGAN(feature_dim, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
        print(f"device: {DEVICE} | parameters: {count_parameters(model)}")

        start = time.time()
        history = train_baseline(model, train_loader, 1 if SMOKE_TEST else BASELINE_EPOCHS, LR, DEVICE, max_batches, penalty=penalty)
        minutes = (time.time() - start) / 60

        torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                    "settings": {"hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "seq_len": SEQ_LEN}},
                   f"baseline{suffix}.pt")

        plt.plot(history["d_loss"], label="discriminator")
        plt.plot(history["g_loss"], label="generator")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(f"baseline_losses{suffix}.png", dpi=150)

        print(f"training time: {minutes:.1f} min")
        if DEVICE == "cuda":
            print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")

        real_w, real_l = collect(test_loader)                           # all 315 test windows
        fake_l = torch.arange(1000) % 5                                 # 200 fake windows per stock
        fake_w = model.generate(fake_l.to(DEVICE), SEQ_LEN, DEVICE).cpu().numpy()
        per_stock_report(real_w, real_l, fake_w, fake_l.numpy(), mean, std, TICKERS)

    else:
        model = TimeGAN(feature_dim, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
        print(f"device: {DEVICE} | parameters: {count_parameters(model)}")

        start = time.time()
        def save(tag):
            """Save the model + everything predict.py needs, as timegan{suffix}_{tag}.pt."""
            torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                        "settings": {"hidden_dim": HIDDEN_DIM, "num_layers": NUM_LAYERS, "seq_len": SEQ_LEN}},
                       f"timegan{suffix}_{tag}.pt")

        if RESUME:
            model.load_state_dict(torch.load(RESUME, map_location=DEVICE, weights_only=False)["model"])
            print(f"resumed from {RESUME}: skipping phases 1-2")
        else:
            ae_history = train_autoencoder(model, train_loader, 1 if SMOKE_TEST else AE_EPOCHS, LR, DEVICE, max_batches)
            supervisor_history = train_supervisor(model, train_loader, 1 if SMOKE_TEST else SUP_EPOCHS, LR, DEVICE, max_batches)
            save("phase2")
        joint_history = train_joint(model, train_loader, 1 if SMOKE_TEST else JOINT_EPOCHS, LR, DEVICE,
                                    max_batches, penalty=penalty, save_fn=save)
        print(f"training time: {(time.time() - start) / 60:.1f} min")
        if DEVICE == "cuda":                                                                                       # criterion 5
            print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")

        save("final")

        # phase 3 loss curves
        plt.figure()
        for key in ("g_loss", "e_loss", "d_loss"):
            plt.plot(joint_history[key], label=key)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(f"timegan_losses{suffix}.png", dpi=150)

        # check phase 1 on TEST windows: hours the autoencoder never trained on
        real, stock = next(iter(test_loader))
        real = real.to(DEVICE)
        stock = stock.to(DEVICE)
        with torch.no_grad():
            rebuilt = model.reconstruct(real)
        print(f"test reconstruction mse: {F.mse_loss(rebuilt, real).item():.4f}")
        books = windows_to_books(rebuilt.cpu().numpy(), mean, std)
        spread, ladder, volume = invalid_rates(books)
        print(f"rebuilt books broken - spread: {spread:.1%}  ladder: {ladder:.1%}  volume: {volume:.1%}")

        # check phase 2 on TEST windows
        with torch.no_grad():
            codes = model.embedder(real)
            guesses = model.supervise(codes, stock)
        print(f"test next-step mse: {F.mse_loss(guesses[:, :-1], codes[:, 1:]).item():.2e}")
        print(f"'no change' guess mse: {F.mse_loss(codes[:, :-1], codes[:, 1:]).item():.2e}")

        # the real test: generated books, per stock (same report as the baseline)
        real_w, real_l = collect(test_loader)
        fake_l = torch.arange(1000) % 5
        fake_w = model.generate(fake_l.to(DEVICE), SEQ_LEN, DEVICE).cpu().numpy()
        per_stock_report(real_w, real_l, fake_w, fake_l.numpy(), mean, std, TICKERS)