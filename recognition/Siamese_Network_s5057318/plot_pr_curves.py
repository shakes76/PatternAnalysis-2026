"""Plot test-set precision-recall curves for the saved baseline and Siamese checkpoints.

Scoring is reused from predict.py / diagnose.py (the Siamese score is the same kNN or
prototype score, with references drawn exactly as predict.py draws them). Nothing is
chosen on the test split. The figure goes to figures/pr_curves.png.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import dataset as d
from diagnose import draw_references, embed, knn_score, prototype_score
from modules import BaselineCNN, SiameseEncoder
from predict import (HERE, N_REF, N_REF_DEBUG, _precision_recall_points, average_precision,
                     load_checkpoint)
from train import subsample


def main():
    """Score the test split with both models, print PR-AUC and save the PR-curve figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", help="folder with cleaned_metadata.csv and combined_images/")
    parser.add_argument("--out_dir", default="outputs",
                        help="folder with the checkpoints; relative paths are taken from this script's folder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=None,
                        help="seed of the train/val/test split (default: same as --seed)")
    parser.add_argument("--k", type=int, default=5,
                        help="nearest references averaged per class in the knn score")
    parser.add_argument("--n_refs", type=int, default=None,
                        help=f"reference photos per class (default {N_REF}, or {N_REF_DEBUG} with --debug; "
                             "0 = every training image)")
    parser.add_argument("--score", choices=["knn", "prototype"], default="knn",
                        help="knn: k nearest refs per class (default); prototype: distance to each class's mean ref")
    parser.add_argument("--preprocess", choices=["none", "grayworld"], default="none",
                        help="colour-constancy step for all images (must match training)")
    parser.add_argument("--no_cudnn", action="store_true", help="disable cuDNN")
    parser.add_argument("--debug", action="store_true",
                        help="same small image subsets as predict.py --debug")
    args = parser.parse_args()
    split_seed = args.seed if args.split_seed is None else args.split_seed
    if args.no_cudnn:
        torch.backends.cudnn.enabled = False
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(HERE, args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    n_ref = args.n_refs if args.n_refs is not None else (N_REF_DEBUG if args.debug else N_REF)
    print(f"device: {device} | seed: {args.seed} | split_seed: {split_seed} | "
          f"--score {args.score} | --n_refs {n_ref}"
          f"{f' | --k {args.k}' if args.score == 'knn' else ''}{' | DEBUG' if args.debug else ''}")

    image_dir = os.path.join(args.data_dir, "combined_images")
    splits = d.split_by_id(d.load_clean_metadata(args.data_dir), seed=split_seed)
    d.check_no_id_overlap(splits)
    if args.debug:
        splits = subsample(splits, args.seed)
    # Eval transform (no augmentation) with mean/std from the train split only.
    _, _, test_loader = d.get_dataloaders(splits, image_dir, batch_size=64, seed=args.seed,
                                          preprocess=args.preprocess)
    transform = test_loader.dataset.transform
    train_df = splits["train"].reset_index(drop=True)
    test_df = splits["test"].reset_index(drop=True)

    baseline, _ = load_checkpoint(BaselineCNN(), os.path.join(out_dir, "baseline_best.pt"), device)
    encoder, _ = load_checkpoint(SiameseEncoder(), os.path.join(out_dir, "siamese_best.pt"), device)

    # References come from the train split only, drawn as predict.py draws them.
    train_labels = torch.from_numpy(train_df["target"].to_numpy().astype(int))
    ref_rows = draw_references(train_labels, n_ref, args.seed)
    ref_emb = {cls: embed(encoder, train_df.iloc[rows.tolist()], image_dir, transform, device)[0]
               for cls, rows in ref_rows.items()}

    def siamese_score(emb):
        """Siamese score for embeddings: kNN or prototype, as chosen by --score."""
        return knn_score(emb, ref_emb, args.k) if args.score == "knn" else prototype_score(emb, ref_emb)

    scores = {}
    base_logits, labels = embed(lambda x: baseline(x).squeeze(1), test_df, image_dir, transform, device)
    scores["baseline"] = base_logits.numpy()
    test_emb, _ = embed(encoder, test_df, image_dir, transform, device)
    scores["siamese"] = siamese_score(test_emb).numpy()
    labels = labels.numpy().astype(int)
    rate = labels.mean()
    print(f"test: {len(labels)} images, {int(labels.sum())} malignant ({100 * rate:.2f}%)")

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, colour in (("baseline", "tab:blue"), ("siamese", "tab:orange")):
        ap = average_precision(scores[name], labels)
        print(f"{name:9s} PR-AUC={ap:.4f}")
        precision, recall, _ = _precision_recall_points(scores[name], labels)
        ax.plot(np.r_[0.0, recall], np.r_[precision[0], precision], drawstyle="steps-post",
                color=colour, label=f"{name} (PR-AUC {ap:.4f})")
    ax.axhline(rate, color="gray", ls="--", lw=1, label=f"malignant rate ({100 * rate:.2f}%)")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Test precision-recall curves" + (f" (siamese {args.score})"))
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    path = os.path.join(HERE, "figures", "pr_curves.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
