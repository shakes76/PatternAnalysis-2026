"""Train, validate, test and save TimeGAN or the recurrent-GAN baseline.

    python train.py --model timegan  --data-dir <dir> --out-dir outputs
    python train.py --model baseline --data-dir <dir> --out-dir outputs

TimeGAN training has three phases (Yoon et al., 2019):
    1. autoencoder  : embedder + recovery learn to reconstruct real windows
    2. supervised   : supervisor learns the next latent step on real latents
    3. joint        : generator/supervisor, embedder/recovery and discriminator
                      are trained together with the combined losses
The baseline trains a standard GAN (adversarial loss only) for the same number of
joint epochs.

Every `--eval-every` joint epochs the model generates windows that are compared
with the *validation* split; the checkpoint with the best validation
KL(spread) + KL(return) is kept and finally scored on the *test* split, which is
never used for any model selection. Loss curves, validation metrics, a summary
JSON and the checkpoint are written to --out-dir.
"""
import argparse
import copy
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import metrics
from dataset import N_FEATURES, denormalise, get_dataloaders, real_windows
from modules import build_model, count_parameters

GAMMA = 1.0          # weight of the discriminator term on raw generator output
D_THRESHOLD = 0.15   # only update the discriminator when it is not already winning


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def temporal_loss(h, h_sup):
    """Supervised loss: predict latent step t from steps before t."""
    return nn.functional.mse_loss(h[:, 1:], h_sup[:, :-1])


def diff_moment_loss(x_hat, x):
    """Moment matching on first differences over time, so generated sequences copy how
    much the real features change from one step to the next (hold-then-jump behaviour)."""
    return moment_loss(x_hat[:, 1:] - x_hat[:, :-1], x[:, 1:] - x[:, :-1])


def moment_loss(x_hat, x):
    """Match per-feature mean and std (over the batch) of generated and real data."""
    std_hat = torch.sqrt(x_hat.var(dim=0) + 1e-6)
    std_real = torch.sqrt(x.var(dim=0) + 1e-6)
    return (std_hat - std_real).abs().mean() + (x_hat.mean(0) - x.mean(0)).abs().mean()


def gate_loss(model, h, x):
    """Teach the hold gate what real data does: open it exactly when a book feature changed.

    Without this the autoencoder can reconstruct perfectly with the gate always open, so
    nothing pushes the decoder to hold values. Target is 1 on the first step and wherever
    a feature differs from the previous step (identical raw values stay identical after
    normalisation, so exact comparison is safe).
    """
    if not hasattr(model.recovery, "gate_logits"):
        return x.new_zeros(())
    changed = torch.ones_like(x[..., 1:])
    changed[:, 1:] = (x[:, 1:, 1:] != x[:, :-1, 1:]).float()
    return nn.functional.binary_cross_entropy_with_logits(model.recovery.gate_logits(h), changed)


def batches(loader, device, max_batches=None):
    for i, x in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            return
        yield x.to(device)


def train_timegan(model, loaders, args, device, history, evaluator):
    bce = nn.BCEWithLogitsLoss()
    lr = args.lr
    opt_ae = torch.optim.Adam(list(model.embedder.parameters())
                              + list(model.recovery.parameters()), lr=lr)
    opt_s = torch.optim.Adam(model.supervisor.parameters(), lr=lr)
    opt_g = torch.optim.Adam(list(model.generator.parameters())
                             + list(model.supervisor.parameters()), lr=lr)
    opt_e = torch.optim.Adam(list(model.embedder.parameters())
                             + list(model.recovery.parameters()), lr=lr)
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr)
    opt_dx = (torch.optim.Adam(model.feature_discriminator.parameters(), lr=lr)
              if model.feature_discriminator is not None else None)
    joint_opts = [opt_g, opt_e, opt_d] + ([opt_dx] if opt_dx else [])
    schedulers = ([torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=max(args.epochs, 1))
                   for o in joint_opts] if args.lr_schedule == "cosine" else [])
    # separate optimisers, so each phase only updates the networks it is meant to train
    ones = lambda t: torch.ones_like(t)
    zeros = lambda t: torch.zeros_like(t)

    # Phase 1: autoencoder
    for epoch in range(args.ae_epochs):
        losses = []
        for x in batches(loaders["train"], device, args.max_batches):
            h = model.embedder(x)
            x_tilde = model.recovery(h)
            loss = 10 * torch.sqrt(nn.functional.mse_loss(x_tilde, x)) + gate_loss(model, h, x)
            opt_ae.zero_grad(); loss.backward(); opt_ae.step()
            losses.append(loss.item())
        history["ae_loss"].append(float(np.mean(losses)))
        print(f"[AE {epoch + 1}/{args.ae_epochs}] loss {history['ae_loss'][-1]:.4f}", flush=True)

    # Phase 2: supervisor on real latents
    for epoch in range(args.sup_epochs):
        losses = []
        for x in batches(loaders["train"], device, args.max_batches):
            with torch.no_grad():
                h = model.embedder(x)
            loss = temporal_loss(h, model.supervisor(h))
            opt_s.zero_grad(); loss.backward(); opt_s.step()
            losses.append(loss.item())
        history["sup_loss"].append(float(np.mean(losses)))
        print(f"[SUP {epoch + 1}/{args.sup_epochs}] loss {history['sup_loss'][-1]:.5f}", flush=True)

    # Phase 3: joint training. The generator side takes two steps for each discriminator step,
    # and the discriminator is skipped when it is already winning (D_THRESHOLD), which keeps
    # the two sides balanced.
    for epoch in range(args.epochs):
        g_l, e_l, d_l = [], [], []
        for x in batches(loaders["train"], device, args.max_batches):
            for _ in range(2):  # two generator/embedder steps per discriminator step
                z = model.noise(len(x), x.shape[1], device)
                h = model.embedder(x)
                e_hat = model.generator(z)
                h_hat = model.supervisor(e_hat)
                x_hat = model.recovery(h_hat)
                y_fake, y_fake_e = model.discriminator(h_hat), model.discriminator(e_hat)
                g_loss = (bce(y_fake, ones(y_fake)) + GAMMA * bce(y_fake_e, ones(y_fake_e))
                          + args.w_sup * torch.sqrt(temporal_loss(h.detach(), model.supervisor(h.detach())))
                          + args.w_moment * moment_loss(x_hat, x))
                if args.w_diff:
                    g_loss = g_loss + args.w_diff * diff_moment_loss(x_hat, x)
                if opt_dx:  # also fool the feature-space discriminator
                    y_x = model.feature_discriminator(x_hat)
                    g_loss = g_loss + bce(y_x, ones(y_x))
                opt_g.zero_grad(); g_loss.backward(); opt_g.step()

                h = model.embedder(x)
                e_loss = (10 * torch.sqrt(nn.functional.mse_loss(model.recovery(h), x))
                          + gate_loss(model, h, x)
                          + 0.1 * temporal_loss(h, model.supervisor(h)))
                opt_e.zero_grad(); e_loss.backward(); opt_e.step()

            with torch.no_grad():
                h = model.embedder(x)
                e_hat = model.generator(model.noise(len(x), x.shape[1], device))
                h_hat = model.supervisor(e_hat)
            y_real, y_fake = model.discriminator(h), model.discriminator(h_hat)
            y_fake_e = model.discriminator(e_hat)
            # discriminator: real latents should score 1, generated latents 0
            d_loss = (bce(y_real, ones(y_real)) + bce(y_fake, zeros(y_fake))
                      + GAMMA * bce(y_fake_e, zeros(y_fake_e)))
            if d_loss.item() > D_THRESHOLD:
                opt_d.zero_grad(); d_loss.backward(); opt_d.step()
            if opt_dx:
                with torch.no_grad():
                    x_fake = model.recovery(h_hat)
                y_xr, y_xf = model.feature_discriminator(x), model.feature_discriminator(x_fake)
                dx_loss = bce(y_xr, ones(y_xr)) + bce(y_xf, zeros(y_xf))
                if dx_loss.item() > D_THRESHOLD:
                    opt_dx.zero_grad(); dx_loss.backward(); opt_dx.step()
            g_l.append(g_loss.item()); e_l.append(e_loss.item()); d_l.append(d_loss.item())
        history["g_loss"].append(float(np.mean(g_l)))
        history["e_loss"].append(float(np.mean(e_l)))
        history["d_loss"].append(float(np.mean(d_l)))
        for sch in schedulers:
            sch.step()
        print(f"[JOINT {epoch + 1}/{args.epochs}] G {history['g_loss'][-1]:.3f} "
              f"E {history['e_loss'][-1]:.3f} D {history['d_loss'][-1]:.3f}", flush=True)
        evaluator(epoch)


def train_baseline(model, loaders, args, device, history, evaluator):
    bce = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=args.lr)
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        g_l, d_l = [], []
        for x in batches(loaders["train"], device, args.max_batches):
            fake = model.generator(model.noise(len(x), x.shape[1], device))
            y_real, y_fake = model.discriminator(x), model.discriminator(fake.detach())
            d_loss = (bce(y_real, torch.ones_like(y_real))
                      + bce(y_fake, torch.zeros_like(y_fake)))
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            y_fake = model.discriminator(fake)
            g_loss = bce(y_fake, torch.ones_like(y_fake))
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()
            g_l.append(g_loss.item()); d_l.append(d_loss.item())
        history["g_loss"].append(float(np.mean(g_l)))
        history["d_loss"].append(float(np.mean(d_l)))
        print(f"[GAN {epoch + 1}/{args.epochs}] G {history['g_loss'][-1]:.3f} "
              f"D {history['d_loss'][-1]:.3f}", flush=True)
        evaluator(epoch)


def plot_history(history, name, out_dir):
    """Loss curves per phase plus validation metrics over joint epochs."""
    panels = [(k, v) for k, v in history.items()
              if k in ("ae_loss", "sup_loss", "g_loss", "e_loss", "d_loss") and v]
    n_val = 1 if history["val"] else 0
    fig, axes = plt.subplots(1, len(panels) + n_val, figsize=(3.6 * (len(panels) + n_val), 3.2))
    for ax, (k, v) in zip(np.atleast_1d(axes), panels):
        ax.plot(range(1, len(v) + 1), v); ax.set_title(k); ax.set_xlabel("epoch")
    if n_val:
        ax = np.atleast_1d(axes)[-1]
        ep = [r["epoch"] for r in history["val"]]
        for key in ("kl_spread", "kl_return", "audit_crossed"):
            ax.plot(ep, [r[key] for r in history["val"]], marker="o", label=key)
        ax.set_title("validation metrics"); ax.set_xlabel("joint epoch"); ax.legend(fontsize=7)
    fig.suptitle(f"{name} training"); fig.tight_layout()
    fig.savefig(f"{out_dir}/{name}_training.png", dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", choices=["timegan", "baseline"], required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--step", type=float, default=1.0, help="grid step in seconds")
    p.add_argument("--return-transform", choices=["zscore", "clip", "quantile"],
                   default="zscore", help="transform applied to the mid-return feature")
    p.add_argument("--ladder", choices=["levels", "increments"], default="levels",
                   help="price ladder features: distance per level, or gaps between levels")
    p.add_argument("--clip-q", type=float, default=0.5, help="clip percentile for --return-transform clip")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ae-epochs", type=int, default=10)
    p.add_argument("--sup-epochs", type=int, default=10)
    p.add_argument("--epochs", type=int, default=30, help="joint / GAN epochs")
    p.add_argument("--w-sup", type=float, default=100.0, help="TimeGAN supervised-loss weight")
    p.add_argument("--w-moment", type=float, default=100.0, help="TimeGAN mean/std matching weight")
    p.add_argument("--w-diff", type=float, default=0.0,
                   help="weight of moment matching on first differences over time (0 = off)")
    p.add_argument("--feature-adv", action="store_true",
                   help="TimeGAN: add a discriminator on decoded features")
    p.add_argument("--hold-gate", action="store_true",
                   help="TimeGAN: decoder may hold the previous step's book values")
    p.add_argument("--lr-schedule", choices=["none", "cosine"], default="none")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--n-eval", type=int, default=2000, help="synthetic windows per evaluation")
    p.add_argument("--max-batches", type=int, default=None, help="limit batches per epoch (smoke tests)")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = pick_device(args.device)

    loaders, info = get_dataloaders(args.data_dir, args.seq_len, args.step,
                                    batch_size=args.batch_size,
                                    return_transform=args.return_transform, clip_q=args.clip_q,
                                    ladder=args.ladder)
    raw = {k: real_windows(info, k) for k in loaders}   # untransformed real windows
    model = build_model(args.model, N_FEATURES, args.hidden, args.layers,
                        feature_adv=args.feature_adv,
                        hold_gate=args.hold_gate).to(device)
    print(f"{args.model}: {count_parameters(model):,} parameters on {device}", flush=True)

    history = {"ae_loss": [], "sup_loss": [], "g_loss": [], "e_loss": [], "d_loss": [], "val": []}
    best = {"score": float("inf"), "state": None, "epoch": -1}

    def synth_raw(n):
        return denormalise(model.sample(n, args.seq_len, device).cpu().numpy(), info)

    def evaluator(epoch):
        """Validation: score generated windows against the validation split."""
        if (epoch + 1) % args.eval_every and epoch + 1 != args.epochs:
            return
        res = metrics.evaluate(raw["val"], synth_raw(args.n_eval), reference_raw=raw["train"],
                               ladder=args.ladder)
        res["epoch"] = epoch + 1
        history["val"].append(res)
        score = res["kl_spread"] + res["kl_return"]
        if score < best["score"]:
            best.update(score=score, state=copy.deepcopy(model.state_dict()), epoch=epoch + 1)
        print(f"   val: KL spread {res['kl_spread']:.3f} return {res['kl_return']:.3f} "
              f"crossed {res['audit_crossed']:.3f} ladder {res['audit_ladder']:.3f} "
              f"SSIM {res['ssim']:.3f}", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer = train_timegan if args.model == "timegan" else train_baseline
    trainer(model, loaders, args, device, history, evaluator)
    train_seconds = time.time() - t0

    # Test with the best-validation checkpoint (test split used only here). Going back to the
    # best checkpoint matters because GAN quality can get worse late in training.
    model.load_state_dict(best["state"])
    test = metrics.evaluate(raw["test"], synth_raw(args.n_eval), reference_raw=raw["train"],
                          ladder=args.ladder)
    # Real-data reference at this grid step: how far real training windows are from the
    # real test windows (the floor any generator is measured against)
    reference = metrics.evaluate(raw["test"], raw["train"][:: max(1, len(raw["train"]) // args.n_eval)],
                                 reference_raw=raw["test"], ladder=args.ladder)
    t1 = time.time(); model.sample(1000, args.seq_len, device); latency = time.time() - t1
    profile = {
        "parameters": count_parameters(model), "train_seconds": train_seconds,
        "sample_1000_windows_seconds": latency, "device": str(device),
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 2 ** 20
                         if device.type == "cuda" else None),
    }
    print("TEST (best val epoch %d):" % best["epoch"],
          {k: round(v, 4) for k, v in test.items()}, flush=True)

    plot_history(history, args.model, args.out_dir)
    torch.save({"state_dict": model.state_dict(), "args": vars(args), "best_epoch": best["epoch"]},
               f"{args.out_dir}/{args.model}.pt")
    with open(f"{args.out_dir}/{args.model}_summary.json", "w") as f:
        json.dump({"args": vars(args), "best_epoch": best["epoch"], "test": test,
                   "reference": reference, "profile": profile, "history": history}, f, indent=2)


if __name__ == "__main__":
    main()
