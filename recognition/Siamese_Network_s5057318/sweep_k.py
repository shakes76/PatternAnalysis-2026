"""Sweep the Siamese scoring settings (n_refs, knn k / prototype) over reference-sample seeds.

Selection uses the validation split only; test is printed for reporting.
"""
import argparse
import os

import numpy as np
import torch

import dataset as d
from diagnose import REF_SEEDS, draw_references, embed, knn_score, metrics, prototype_score
from modules import BaselineCNN, SiameseEncoder
from predict import HERE, load_checkpoint
from train import subsample

N_REFS = [200, 400, 1000, 0]  # 0 = every training image
KS = [5, 10, 20, 30, 50, 100, 200, 500]


def main():
    """Embed every split once, then print val/test metrics for every scoring setting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="folder with cleaned_metadata.csv and combined_images/")
    parser.add_argument("--out_dir", default="outputs",
                        help="relative paths are taken from this script's folder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=None,
                        help="seed of the train/val/test split (default: same as --seed)")
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
    print(f"device: {device} | out_dir: {out_dir} | seed: {args.seed} | split_seed: {split_seed}"
          f"{' | DEBUG' if args.debug else ''}")

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

    frames = {name: splits[name].reset_index(drop=True) for name in ("train", "val", "test")}
    siam, labels = {}, {}
    for name, df in frames.items():
        siam[name], labels[name] = embed(encoder, df, image_dir, transform, device)
    n_class = {cls: int((labels["train"] == cls).sum()) for cls in (0, 1)}
    print("images: " + " | ".join(f"{n} {len(labels[n])} ({int(labels[n].sum())} malignant)"
                                  for n in frames)
          + f" | train references available: {n_class[0]} benign, {n_class[1]} malignant")

    print("\n== Baseline (for comparison) ==")
    for name in ("val", "test"):
        logits, _ = embed(lambda x: baseline(x).squeeze(1), frames[name], image_dir, transform, device)
        au, ap = metrics(logits.numpy(), labels[name].numpy())
        print(f"baseline {name:4s} AUROC={au:.4f}  PR-AUC={ap:.4f}")

    print(f"\n== Siamese scoring sweep (mean +- std over reference seeds "
          f"{REF_SEEDS.start}-{REF_SEEDS.stop - 1}; n_refs 0 = all, run once) ==")
    print(f"{'n_refs':>6s}  {'refs b/m':>9s}  {'score':10s} {'seeds':>5s}  "
          f"{'val AUROC':>15s}  {'val PR-AUC':>15s}  {'test AUROC':>15s}  {'test PR-AUC':>15s}")
    results = []  # (label, mean values)
    for n_ref in N_REFS:
        # References per class actually available; k values above the smaller class are skipped.
        per_class = {cls: n_class[cls] if n_ref == 0 else min(n_ref, n_class[cls]) for cls in (0, 1)}
        scores = [("knn", k) for k in KS if k <= min(per_class.values())] + [("prototype", None)]
        seeds = [REF_SEEDS.start] if n_ref == 0 else list(REF_SEEDS)
        for kind, k in scores:
            rows = []
            for ref_seed in seeds:
                ref_rows = draw_references(labels["train"], n_ref, ref_seed)
                ref_emb = {cls: siam["train"][ref_rows[cls]] for cls in (0, 1)}

                def score(emb):
                    """Score embeddings against this reference draw (kNN with k, or prototype)."""
                    return (knn_score(emb, ref_emb, k) if kind == "knn"
                            else prototype_score(emb, ref_emb)).numpy()
                rows.append((*metrics(score(siam["val"]), labels["val"].numpy()),
                             *metrics(score(siam["test"]), labels["test"].numpy())))
            rows = np.array(rows)
            mean = rows.mean(0)
            std = rows.std(0, ddof=1) if len(rows) > 1 else np.zeros(4)
            label = f"n_refs {'all' if n_ref == 0 else n_ref}, " + (f"knn k={k}" if kind == "knn" else "prototype")
            results.append((label, mean))
            score_txt = f"knn k={k}" if kind == "knn" else "prototype"
            print(f"{'all' if n_ref == 0 else n_ref:>6}  {per_class[0]:>4d}/{per_class[1]:<4d}  {score_txt:10s} "
                  f"{len(seeds):>5d}  " + "  ".join(f"{m:.4f}+-{s:.4f}" for m, s in zip(mean, std)))

    print("\n== Selected on VALIDATION only (first setting wins ties) ==")
    for title, col in (("highest mean val AUROC", 0), ("highest mean val PR-AUC", 1)):
        label, mean = max(results, key=lambda r: r[1][col])
        print(f"{title}: {label} | val AUROC={mean[0]:.4f} PR-AUC={mean[1]:.4f} | "
              f"test AUROC={mean[2]:.4f} PR-AUC={mean[3]:.4f}")


if __name__ == "__main__":
    main()
