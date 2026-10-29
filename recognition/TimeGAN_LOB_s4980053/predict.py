# predict.py - example usage of a trained model: load it, generate order books, score them, plot them

import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import TICKERS, make_loaders
from modules import BaselineGAN, TimeGAN
from utils import collect, per_stock_report, vwap_offset, windows_to_books
from matplotlib.animation import FuncAnimation, PillowWriter

def load_model(kind, path, feature_dim=41):
    """Rebuild a saved model. kind is "baseline" or "timegan". Returns the model, mean, std, settings."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    s = ckpt["settings"]
    model_class = BaselineGAN if kind == "baseline" else TimeGAN
    model = model_class(feature_dim, s["hidden_dim"], s["num_layers"])
    model.load_state_dict(ckpt["model"])                     # pour the saved dials into it
    model.eval()
    return model, ckpt["mean"], ckpt["std"], s

def plot_vwap_over_time(real_w, real_l, fake_w, fake_l, mean, std, path, n=3):
    """One panel per stock: VWAP offset over time for the first n real and first n fake windows."""
    fig, axes = plt.subplots(len(TICKERS), 1, figsize=(10, 12), sharex=True)
    for stock_id, (ticker, ax) in enumerate(zip(TICKERS, axes)):
        real_books = windows_to_books(real_w[real_l == stock_id][:n], mean, std)
        fake_books = windows_to_books(fake_w[fake_l == stock_id][:n], mean, std)                                       # same, for this stock's first n fake windows
        for i in range(n):
            ax.plot(vwap_offset(real_books[i]), color="tab:blue", alpha=0.7, label="real" if i == 0 else None)
            ax.plot(vwap_offset(fake_books[i]), color="tab:orange", alpha=0.7, label="fake" if i == 0 else None)
        ax.set_title(ticker)
        ax.set_ylabel("VWAP - mid (ticks)")
    axes[-1].set_xlabel("time step (seconds)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_vwap_per_stock(real_w, real_l, fake_w, fake_l, mean, std, path):
    """Mean +/- std of the VWAP offset per stock, real next to fake."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for stock_id in range(len(TICKERS)):
        for windows, labels, shift, color, name in ((real_w, real_l, -0.15, "tab:blue", "real"),
                                                    (fake_w, fake_l, 0.15, "tab:orange", "fake")):
            v = vwap_offset(windows_to_books(windows[labels == stock_id], mean, std))
            ax.errorbar(stock_id + shift, v.mean(), yerr=v.std(), fmt="o", color=color, capsize=4,
                        label=name if stock_id == 0 else None)
    ax.set_xticks(range(len(TICKERS)), TICKERS)
    ax.set_ylabel("VWAP - mid (ticks)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def make_gif(real_book, fake_book, ticker, path, every=5):
    """Animated order book, real next to generated: volume per level, in ticks from the mid."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    books = ((axes[0], real_book, "real"), (axes[1], fake_book, "generated"))
    # one shared scale, taken from the real window only
    y_top = np.percentile(real_book[:, 1::2], 99) * 1.2                          # real volumes (size columns)
    real_mid = (real_book[:, 0] + real_book[:, 2]) / 2
    x_reach = np.abs((real_book[:, 0::2] - real_mid[:, None]) / 100).max() * 1.2   # real prices (price columns)

    def draw(t):
        for ax, book, name in books:
            ax.clear()
            row = book[t]
            ask_p, ask_s, bid_p, bid_s = row[0::4], row[1::4], row[2::4], row[3::4]
            mid = (ask_p[0] + bid_p[0]) / 2
            ax.bar((bid_p - mid) / 100, bid_s, width=0.8, color="tab:blue", label="bids")
            ax.bar((ask_p - mid) / 100, ask_s, width=0.8, color="tab:red", label="asks")
            ax.axvline(0, color="black", linestyle="--", label="mid")
            ax.set_ylim(0, y_top)                                                # same height for both panels
            ax.set_xlim(-x_reach, x_reach)                                       # same width for both panels
            ax.set_xlabel("price (ticks from mid)")
            ax.set_ylabel("volume (shares)")
            ax.set_title(f"{ticker} {name}, t = {t} s")
            ax.legend(loc="upper right")

    frames = range(0, len(real_book), every)
    FuncAnimation(fig, draw, frames=frames).save(path, writer=PillowWriter(fps=10))
    plt.close(fig)

if __name__ == "__main__":
    kind, path = sys.argv[1], sys.argv[2]          # e.g. python predict.py baseline baseline_trained.pt
    tag = path.replace(".pt", "")                  # e.g. "timegan_ep40": names run's figures
    model, mean, std, settings = load_model(kind, path)

    train_loader, _, test_loader, _, _ = make_loaders(seq_len=settings["seq_len"])     # test windows to score; training windows for the reference
    real_w, real_l = collect(test_loader)

    ds = train_loader.dataset
    idx = range(0, len(ds), 250)                                  # ~314 training windows, spread over the morning
    ref_w = np.stack([ds[i][0].numpy() for i in idx])
    ref_l = np.array([ds[i][1] for i in idx])
    print("--- reference: real training hours vs real test hour (the best a perfect generator could do) ---")
    per_stock_report(real_w, real_l, ref_w, ref_l, mean, std, TICKERS)

    fake_l = torch.arange(1000) % 5
    fake_w = model.generate(fake_l, settings["seq_len"], "cpu").numpy()
    print(f"--- {tag} vs real test hour ---")
    per_stock_report(real_w, real_l, fake_w, fake_l.numpy(), mean, std, TICKERS)

    plot_vwap_over_time(real_w, real_l, fake_w, fake_l.numpy(), mean, std, f"{tag}_vwap_over_time.png")
    plot_vwap_per_stock(real_w, real_l, fake_w, fake_l.numpy(), mean, std, f"{tag}_vwap_per_stock.png")

    for stock_id, ticker in enumerate(TICKERS):
        real_book = windows_to_books(real_w[real_l == stock_id][:1], mean, std)[0]
        fake_book = windows_to_books(fake_w[fake_l.numpy() == stock_id][:1], mean, std)[0]
        make_gif(real_book, fake_book, ticker, f"{tag}_{ticker}.gif")