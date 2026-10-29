"""Generate synthetic limit order books using the V5.38 pipeline."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import prepare_dataset_v4
from modules import (
    TimeGANV42,
    MLPConstraintAwareLOBHead,
    MultiFactorVolumeHead,
    ProbabilisticDepthNet,
)

TICK = 0.01
REFERENCE_PRICE = 100.0
RETURN_THRESHOLD = 1e-7


def load_checkpoint(path):
    return torch.load(
        path, map_location="cpu", weights_only=True
    )


def extract_volumes(book):
    return np.maximum(
        np.expm1(book[..., 1:40:2]), 0.0
    )


def fit_return_model(train):
    returns = train[..., 40]
    recent = returns[3 * len(returns) // 4:]
    active = np.abs(recent) >= RETURN_THRESHOLD

    previous = active[:, :-1].ravel()
    following = active[:, 1:].ravel()

    transition = np.zeros(2)
    for state in (0, 1):
        selected = following[previous == bool(state)]
        transition[state] = (
            float(selected.mean())
            if len(selected)
            else float(active.mean())
        )

    jumps = recent[active]
    return float(active.mean()), transition, jumps


def generate_returns(model, n, t, seed):
    initial_prob, transition, jumps = model
    rng = np.random.default_rng(seed)

    result = np.zeros((n, t), dtype=np.float64)
    active = rng.random(n) < initial_prob

    for step in range(1, t):
        active = (
            rng.random(n)
            < transition[active.astype(int)]
        )
        count = int(active.sum())
        if count:
            if len(jumps) == 0:
                raise RuntimeError(
                    "No active return jumps in training data"
                )
            result[active, step] = rng.choice(
                jumps, count, replace=True
            )

    return result


def fit_spread_markov(train):
    spread = np.maximum(
        np.rint(
            (train[..., 0] - train[..., 2]) / TICK
        ).astype(np.int64),
        1,
    )

    recent_count = round(len(spread) * 2 / 3)
    recent = spread[-recent_count:]

    max_tick = int(spread.max())
    states = np.arange(1, max_tick + 1)

    initial_counts = np.bincount(
        recent[:, 0], minlength=max_tick + 1
    )[1:].astype(np.float64)

    if initial_counts.sum() == 0:
        raise ValueError("Empty spread initial distribution")

    initial_prob = initial_counts / initial_counts.sum()

    counts = np.zeros(
        (max_tick + 1, max_tick + 1),
        dtype=np.float64,
    )
    np.add.at(
        counts,
        (recent[:, :-1].ravel(),
         recent[:, 1:].ravel()),
        1,
    )

    transitions = np.zeros_like(counts)
    for state in states:
        total = counts[state].sum()
        if total:
            transitions[state] = counts[state] / total
        else:
            transitions[state, state] = 1.0

    return states, initial_prob, transitions


def generate_spread_markov(model, n, t, seed):
    states, initial_prob, transitions = model
    rng = np.random.default_rng(seed)

    result = np.empty((n, t), dtype=np.int64)
    result[:, 0] = rng.choice(
        states, size=n, p=initial_prob
    )

    for step in range(1, t):
        current = result[:, step - 1]
        for state in np.unique(current):
            indices = np.flatnonzero(current == state)
            result[indices, step] = rng.choice(
                len(transitions),
                size=len(indices),
                p=transitions[state],
            )

    return result


def generate_depth(initial, train, checkpoint, seed):
    """V5.31 MDN rollout with train P99.9 per-level caps."""
    n, t, levels = initial.shape
    assert levels == 20

    ck = load_checkpoint(checkpoint)
    model = ProbabilisticDepthNet(
        hidden=ck["hidden"],
        components=ck["components"],
    )
    model.load_state_dict(
        ck["model_state_dict"], strict=True
    )
    model.eval()

    mu = np.asarray(
        torch.as_tensor(ck["mu"]), dtype=np.float64
    )
    sd = np.asarray(
        torch.as_tensor(ck["sd"]), dtype=np.float64
    )

    train_volume = extract_volumes(train)
    cap = np.quantile(
        train_volume.reshape(-1, 20),
        0.999,
        axis=0,
    )
    upper = np.log1p(cap)[None, :]

    rng = np.random.default_rng(seed)
    mask_u = rng.random((t - 1, n, 20))
    component_u = rng.random((t - 1, n, 20, 1))
    gaussian_z = rng.standard_normal((t - 1, n, 20))

    volumes = np.empty_like(initial)
    volumes[:, 0] = initial[:, 0]

    with torch.no_grad():
        for step in range(1, t):
            current = np.log1p(volumes[:, step - 1])
            inputs = torch.tensor(
                ((current - mu) / sd).astype(np.float32)
            )
            logits, weights, means, scales_log = model(
                inputs
            )

            probabilities = torch.sigmoid(
                logits
            ).numpy()

            weights = torch.softmax(
                weights, dim=-1
            ).numpy()
            means = means.numpy()
            scales = torch.exp(scales_log).numpy()

            cumulative = np.cumsum(weights, axis=-1)
            component = (
                component_u[step - 1]
                > cumulative[..., :-1]
            ).sum(axis=-1)

            chosen_means = np.take_along_axis(
                means, component[..., None], axis=-1
            )[..., 0]
            chosen_scales = np.take_along_axis(
                scales, component[..., None], axis=-1
            )[..., 0]

            normalized = (
                chosen_means
                + chosen_scales * gaussian_z[step - 1]
            )
            raw_log = normalized * sd + mu
            update = (
                mask_u[step - 1] < probabilities
            )
            bounded_log = np.clip(
                raw_log, 0, upper
            )
            proposed = np.expm1(bounded_log)

            volumes[:, step] = np.where(
                update,
                proposed,
                volumes[:, step - 1],
            )

    return volumes


def check_structure(book, returns, reference_price):
    asks = book[..., 0:40:4]
    bids = book[..., 2:40:4]
    volumes = extract_volumes(book)

    midpoint = (
        asks[..., 0] + bids[..., 0]
    ) / 2 + reference_price

    implied_returns = np.zeros_like(returns)
    implied_returns[:, 1:] = np.diff(
        np.log(midpoint), axis=1
    )

    checks = {
        "positive_spread": bool(
            np.all(asks[..., 0] > bids[..., 0])
        ),
        "ascending_asks": bool(
            np.all(np.diff(asks, axis=-1) > 0)
        ),
        "descending_bids": bool(
            np.all(np.diff(bids, axis=-1) < 0)
        ),
        "nonnegative_volumes": bool(
            np.all(volumes >= 0)
        ),
        "finite": bool(
            np.all(np.isfinite(book))
        ),
        "returns_consistent": bool(
            np.allclose(
                implied_returns,
                returns,
                atol=1e-10,
                rtol=1e-6,
            )
        ),
    }

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs_formal_predict")
    parser.add_argument("--num-sequences", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=371021)
    parser.add_argument("--depth-seed", type=int, default=371031)
    parser.add_argument("--spread-seed", type=int, default=371038)

    parser.add_argument(
        "--timegan-checkpoint",
        default="outputs_v5_10_latent_matching/timegan_finetuned.pt",
    )
    parser.add_argument(
        "--price-checkpoint",
        default="outputs_v5_9_variance_aware/variance_aware_best.pt",
    )
    parser.add_argument(
        "--volume-checkpoint",
        default="outputs_v5_14_multifactor/best_multifactor_head.pt",
    )
    parser.add_argument(
        "--depth-checkpoint",
        default="outputs_v5_29_mdn/best_mdn_safe.pt",
    )

    args = parser.parse_args()
    if args.num_sequences < 1 or args.seq_len < 2:
        parser.error(
            "num-sequences must be >= 1 and seq-len >= 2"
        )

    torch.set_num_threads(4)
    n, t = args.num_sequences, args.seq_len
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    data = prepare_dataset_v4(args.data_dir)
    norm = data["normalizer"]
    train = norm.inverse_transform(
        np.asarray(data["train"], dtype=np.float64)
    )
    bn = norm.book_normalizer

    print("Loading TimeGAN...")
    ck = load_checkpoint(args.timegan_checkpoint)
    gan = TimeGANV42(
        feature_dim=ck["feature_dim"],
        hidden_dim=ck["hidden_dim"],
        noise_dim=ck["noise_dim"],
        num_layers=ck["num_layers"],
    )
    gan.load_state_dict(
        ck["model_state_dict"], strict=True
    )
    gan.eval()

    print("Loading Price Head...")
    pc = load_checkpoint(args.price_checkpoint)
    price_head = MLPConstraintAwareLOBHead(
        hidden_dim=pc["hidden_dim"],
        minimum=bn.minimum,
        scale=bn.scale,
        min_spread=TICK,
        min_gap=TICK,
        intermediate=pc["intermediate"],
    )
    price_head.load_state_dict(
        pc["head_state_dict"], strict=True
    )
    price_head.eval()

    print("Loading MultiFactor Volume Head...")
    vc = load_checkpoint(args.volume_checkpoint)
    volume_head = MultiFactorVolumeHead(
        hidden_dim=vc["hidden_dim"],
        mean_log=torch.ones(20),
        std_log=torch.ones(20),
    )
    volume_head.load_state_dict(
        vc["head_state_dict"], strict=True
    )
    volume_head.eval()

    print("Generating TimeGAN latent states...")
    with torch.no_grad():
        generator = torch.Generator().manual_seed(
            args.seed
        )
        z = torch.rand(
            n, t, ck["noise_dim"],
            generator=generator,
        )
        hidden = gan.supervisor(gan.generator(z))

        base_book = price_head.forward_raw(
            hidden
        ).detach().numpy().astype(np.float64)

        innovations = torch.randn(
            n, t, 20,
            generator=generator,
        )
        initial_log_volumes = volume_head(
            hidden, innovations
        ).detach().numpy()

    initial_volumes = np.maximum(
        np.expm1(initial_log_volumes), 0.0
    )

    print("Generating Markov returns...")
    return_model = fit_return_model(train)
    returns = generate_returns(
        return_model, n, t, args.seed
    )

    print("Generating MDN depths...")
    volumes = generate_depth(
        initial_volumes,
        train,
        args.depth_checkpoint,
        args.depth_seed,
    )

    print("Generating Markov spreads...")
    spread_model = fit_spread_markov(train)
    spread_ticks = generate_spread_markov(
        spread_model, n, t, args.spread_seed
    )
    spread = spread_ticks * TICK

    # Build the V5.21 price ladder and integrate
    # the V5.37 spread while preserving midpoint.
    midprice = REFERENCE_PRICE * np.exp(
        np.cumsum(returns, axis=1)
    )
    mid_relative = midprice - REFERENCE_PRICE

    book = base_book.copy()
    book[..., 0] = mid_relative + spread / 2
    book[..., 2] = mid_relative - spread / 2

    for level in range(1, 10):
        prev_ask = 4 * (level - 1)
        prev_bid = prev_ask + 2
        ask_idx = 4 * level
        bid_idx = ask_idx + 2

        ask_gap = np.maximum(
            base_book[..., ask_idx]
            - base_book[..., prev_ask],
            TICK,
        )
        bid_gap = np.maximum(
            base_book[..., prev_bid]
            - base_book[..., bid_idx],
            TICK,
        )

        book[..., ask_idx] = (
            book[..., prev_ask] + ask_gap
        )
        book[..., bid_idx] = (
            book[..., prev_bid] - bid_gap
        )

    book[..., 1:40:2] = np.log1p(volumes)

    reconstructed_midprice = (
        book[..., 0] + book[..., 2]
    ) / 2 + REFERENCE_PRICE
    reconstructed_returns = np.zeros(
        (n, t), dtype=np.float64
    )
    reconstructed_returns[:, 1:] = np.diff(
        np.log(reconstructed_midprice), axis=1
    )

    features = np.concatenate(
        [book, reconstructed_returns[..., None]],
        axis=-1,
    )

    checks = check_structure(
        book,
        reconstructed_returns,
        REFERENCE_PRICE,
    )

    assert features.shape == (n, t, 41)
    assert all(checks.values()), checks

    result_path = output / "generated_lob.npz"
    np.savez_compressed(
        result_path,
        features=features.astype(np.float32),
        relative_orderbook=book.astype(np.float32),
        log_returns=reconstructed_returns.astype(
            np.float32
        ),
        spread_ticks=spread_ticks.astype(np.int16),
        volumes=volumes.astype(np.float32),
        midprice=midprice.astype(np.float64),
    )

    metrics = {
        "shape": list(features.shape),
        "market_constraints": checks,
        "spread_mean_ticks": float(
            spread_ticks.mean()
        ),
        "spread_unchanged_fraction": float(
            (np.diff(spread_ticks, axis=1) == 0).mean()
        ),
        "volume_mean": float(volumes.mean()),
        "return_zero_fraction": float(
            (np.abs(reconstructed_returns[:, 1:])
             < RETURN_THRESHOLD).mean()
        ),
    }
    (output / "results.json").write_text(
        json.dumps(metrics, indent=2)
    )

    print(json.dumps(metrics, indent=2))
    print("Saved:", result_path)
    print("PREDICTION COMPLETE")


if __name__ == "__main__":
    main()
