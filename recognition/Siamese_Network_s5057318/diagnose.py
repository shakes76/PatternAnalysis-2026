"""Diagnostics for a trained baseline / Siamese pair (read-only: nothing is saved).

1. Reference noise: Siamese val/test AUROC and PR-AUC for reference-sample seeds 1..10.
2. Frozen-embedding linear classifier on the Siamese embeddings and on the baseline's
   pooled features (weighted BCE, stopped on validation loss).
3. Pair distances on val for the Siamese embeddings.

Selection uses the validation split only; test is printed for reporting.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

import dataset as d
from modules import BaselineCNN, SiameseEncoder
from predict import HERE, N_REF, N_REF_DEBUG, auroc, average_precision, load_checkpoint, run_scores
from train import subsample

REF_SEEDS = range(1, 11)
N_PAIRS = 5000
LINEAR_STEPS, LINEAR_LR = 500, 0.05


def embed(fn, df, image_dir, transform, device):
    """Features for every row of `df` (no augmentation); returns (features, labels) tensors."""
    loader = torch.utils.data.DataLoader(d.ISICDataset(df, image_dir, transform), batch_size=64)
    feats, labels, _ = run_scores(fn, loader, device)
    return torch.from_numpy(feats).float(), torch.from_numpy(labels)


def draw_references(train_labels, n_ref, seed):
    """Reference rows per class, drawn exactly as predict.py does for its --seed."""
    gen = torch.Generator().manual_seed(seed)
    rows = {}
    for cls in (0, 1):
        pool = torch.from_numpy(np.flatnonzero(train_labels.numpy() == cls))
        rows[cls] = pool[torch.randperm(len(pool), generator=gen)[:n_ref or None]]  # 0 = every image
    return rows


# Both scores below are 'distance to benign minus distance to malignant', so a higher
# score means an image sits closer to the malignant references (same as predict.py).
def prototype_score(emb, ref_emb):
    """Distance to the mean benign reference minus distance to the mean malignant reference."""
    return (torch.cdist(emb, ref_emb[0].mean(0, keepdim=True))
            - torch.cdist(emb, ref_emb[1].mean(0, keepdim=True))).squeeze(1)


def knn_score(emb, ref_emb, k):
    """Mean distance to the k nearest benign refs minus to the k nearest malignant refs."""
    def mean_nearest(refs):
        """Mean distance from each embedding to its k nearest references in refs."""
        return torch.cdist(emb, refs).topk(k, dim=1, largest=False).values.mean(dim=1)
    return mean_nearest(ref_emb[0]) - mean_nearest(ref_emb[1])


def metrics(scores, labels):
    """(AUROC, PR-AUC) computed with predict.py's metric functions."""
    scores, labels = np.asarray(scores), np.asarray(labels).astype(int)
    return auroc(scores, labels), average_precision(scores, labels)


def linear_probe(train_x, train_y, val_x, val_y, test_x, test_y, seed):
    """One linear layer, weighted BCE (pos_weight = benign/malignant of train), full batch.

    Features are standardised with train statistics. The step with the lowest
    validation loss is kept. Returns (best_step, val metrics, test metrics).
    """
    # standardise every split with the train mean/std (the clamp avoids dividing by ~0)
    mu, sd = train_x.mean(0), train_x.std(0).clamp(min=1e-6)
    train_x, val_x, test_x = [(x - mu) / sd for x in (train_x, val_x, test_x)]
    # weight malignant examples by the benign/malignant count in train, as in the baseline
    pos_weight = torch.tensor([float((train_y == 0).sum()) / float((train_y == 1).sum())])
    torch.manual_seed(seed)
    layer = torch.nn.Linear(train_x.shape[1], 1)
    optimizer = torch.optim.Adam(layer.parameters(), lr=LINEAR_LR)
    # best = (lowest val loss so far, its step, a copy of the layer weights at that step)
    best = (float("inf"), 0, None)
    for step in range(1, LINEAR_STEPS + 1):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(layer(train_x).squeeze(1), train_y.float(),
                                                  pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        # val loss after this step, used to pick the best step
        with torch.no_grad():
            val_loss = F.binary_cross_entropy_with_logits(
                layer(val_x).squeeze(1), val_y.float(), pos_weight=pos_weight).item()
        if val_loss < best[0]:
            best = (val_loss, step, {k: v.clone() for k, v in layer.state_dict().items()})
    # restore the weights from the step with the lowest validation loss
    layer.load_state_dict(best[2])
    with torch.no_grad():
        val_m = metrics(layer(val_x).squeeze(1).numpy(), val_y.numpy())
        test_m = metrics(layer(test_x).squeeze(1).numpy(), test_y.numpy())
    return best[1], best[0], val_m, test_m


def main():
    """Embed every split once with both models and print the three diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="folder with cleaned_metadata.csv and combined_images/")
    parser.add_argument("--out_dir", default="outputs",
                        help="relative paths are taken from this script's folder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=None,
                        help="seed of the train/val/test split (default: same as --seed)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n_refs", type=int, default=None,
                        help=f"reference photos per class (default {N_REF}, or {N_REF_DEBUG} with --debug; "
                             "0 = every training image)")
    parser.add_argument("--score", choices=["knn", "prototype"], default="knn")
    parser.add_argument("--preprocess", choices=["none", "grayworld"], default="none")
    parser.add_argument("--no_cudnn", action="store_true", help="disable cuDNN")
    parser.add_argument("--debug", action="store_true",
                        help="same 256 / 128 / 128 image subsets as train.py --debug")
    args = parser.parse_args()
    split_seed = args.seed if args.split_seed is None else args.split_seed
    if args.no_cudnn:
        torch.backends.cudnn.enabled = False
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(HERE, args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_ref = args.n_refs if args.n_refs is not None else (N_REF_DEBUG if args.debug else N_REF)
    print(f"device: {device} | out_dir: {out_dir} | seed: {args.seed} | split_seed: {split_seed}"
          f" | k: {args.k} | n_refs: {n_ref}{' | DEBUG' if args.debug else ''}")
    print(f"siamese scoring: --score {args.score} | --n_refs {n_ref}"
          f"{' (every training image)' if n_ref == 0 else ' per class'}"
          f"{f' | --k {args.k}' if args.score == 'knn' else ''}")

    image_dir = os.path.join(args.data_dir, "combined_images")
    splits = d.split_by_id(d.load_clean_metadata(args.data_dir), seed=split_seed)
    d.check_no_id_overlap(splits)
    if args.debug:
        splits = subsample(splits, args.seed)
    # Eval transform (no augmentation) with mean/std from the train split only.
    _, val_loader, _ = d.get_dataloaders(splits, image_dir, batch_size=64, seed=args.seed,
                                         preprocess=args.preprocess)
    transform = val_loader.dataset.transform
    baseline, _ = load_checkpoint(BaselineCNN(), os.path.join(out_dir, "baseline_best.pt"), device)
    encoder, _ = load_checkpoint(SiameseEncoder(), os.path.join(out_dir, "siamese_best.pt"), device)

    def baseline_features(x):  # the pooled 256-d vector the final Linear layer sees
        """Return the 256-d pooled feature vector the baseline's final linear layer sees."""
        x = baseline.block4(baseline.block3(baseline.block2(baseline.block1(x))))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    frames = {name: splits[name].reset_index(drop=True) for name in ("train", "val", "test")}
    siam, base, labels = {}, {}, {}
    for name, df in frames.items():
        siam[name], labels[name] = embed(encoder, df, image_dir, transform, device)
        base[name], _ = embed(baseline_features, df, image_dir, transform, device)
    print("images: " + " | ".join(f"{n} {len(labels[n])} ({int(labels[n].sum())} malignant)"
                                  for n in frames))

    # ------------------------------------------------ 1. reference noise
    score_name = f"kNN score, k={args.k}" if args.score == "knn" else "prototype score"
    refs_txt = "all train images" if n_ref == 0 else f"{n_ref} refs per class"
    print(f"\n== 1. Reference noise: Siamese {score_name}, {refs_txt} ==")
    print("ref seed   val AUROC  val PR-AUC  test AUROC  test PR-AUC")
    rows = []
    for ref_seed in REF_SEEDS:
        ref_rows = draw_references(labels["train"], n_ref, ref_seed)
        ref_emb = {cls: siam["train"][ref_rows[cls]] for cls in (0, 1)}
        def score(emb):
            """Score embeddings against this reference draw (kNN or prototype, as set by --score)."""
            return (knn_score(emb, ref_emb, args.k) if args.score == "knn"
                    else prototype_score(emb, ref_emb)).numpy()
        values = (*metrics(score(siam["val"]), labels["val"].numpy()),
                  *metrics(score(siam["test"]), labels["test"].numpy()))
        rows.append(values)
        print(f"{ref_seed:8d}   " + "  ".join(f"{v:>10.4f}" for v in values))
    rows = np.array(rows)
    print("mean       " + "  ".join(f"{v:>10.4f}" for v in rows.mean(0)))
    print("std (n-1)  " + "  ".join(f"{v:>10.4f}" for v in rows.std(0, ddof=1)))

    # ------------------------------------------------ 2. frozen-embedding linear classifier
    print(f"\n== 2. Frozen features + one linear layer (weighted BCE, {LINEAR_STEPS} full-batch Adam "
          f"steps, lr {LINEAR_LR}, best step chosen on val loss) ==")
    for title, feats in (("siamese embeddings (128-d)", siam), ("baseline pooled features (256-d)", base)):
        step, val_loss, val_m, test_m = linear_probe(
            feats["train"], labels["train"], feats["val"], labels["val"],
            feats["test"], labels["test"], args.seed)
        print(f"{title}: best step {step}, val loss {val_loss:.4f}")
        print(f"   val   AUROC={val_m[0]:.4f}  PR-AUC={val_m[1]:.4f}")
        print(f"   test  AUROC={test_m[0]:.4f}  PR-AUC={test_m[1]:.4f}")

    # ------------------------------------------------ 3. pair distances on val
    print(f"\n== 3. Siamese embedding distances on val ({N_PAIRS} random pairs per type) ==")
    gen = torch.Generator().manual_seed(args.seed)
    by_class = {cls: (labels["val"] == cls).nonzero().squeeze(1) for cls in (0, 1)}
    print("pair type            mean   median      p10      p90")
    for title, (c1, c2) in (("benign-benign", (0, 0)), ("malignant-malignant", (1, 1)),
                            ("benign-malignant", (0, 1))):
        a = by_class[c1][torch.randint(len(by_class[c1]), (N_PAIRS,), generator=gen)]
        b = by_class[c2][torch.randint(len(by_class[c2]), (N_PAIRS,), generator=gen)]
        keep = a != b  # never pair an image with itself
        dist = (siam["val"][a[keep]] - siam["val"][b[keep]]).norm(dim=1)
        q = torch.quantile(dist, torch.tensor([0.5, 0.1, 0.9]))
        print(f"{title:19s} {dist.mean():8.4f} {q[0]:8.4f} {q[1]:8.4f} {q[2]:8.4f}")


if __name__ == "__main__":
    main()
