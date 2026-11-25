"""Train Standard or v3 with the recorded settings, or evaluate saved weights.

Use the train command for a new run or to resume a run created by this script.
Recovery restores the optimizer, random states and next batch. Historical
research checkpoints can be evaluated but are not accepted for training resume.
The evaluate command can read Test; the train command only reads Train/Validation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from dataset import (discover, load_case, downsample, zscore_nonzero, epoch_cases,
                     crop_origin, crop_for_3d_model, ExposureRecorder)
from modules import StandardUNet3D, ImprovedUNet3DV3
from utils import (load_model, ce_dice_loss, evaluate, atomic_json,
                   atomic_checkpoint, sha256_file, confusion, overlap, CLASS_NAMES)

ROOT = Path(__file__).resolve().parent


def build_model(name):
    """Construct the model classes imported directly from modules.py."""
    if name == "standard":
        return StandardUNet3D()
    if name == "v3":
        return ImprovedUNet3DV3()
    raise ValueError("Model must be standard or v3")


def plot_history(history, path):
    """Save epoch losses and measured Validation scores; neither curve is a Test score."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([r["epoch"] for r in history], [r["loss"] for r in history])
    axes[0].set(xlabel="Epoch", ylabel="Weighted CE + foreground Dice loss")
    selected = [r for r in history if r["validation"] is not None]
    axes[1].plot([r["epoch"] for r in selected],
                 [r["validation"]["selection_score"] for r in selected], "o-")
    axes[1].set(xlabel="Epoch", ylabel="Patient-equal small-organ Validation Dice", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def train(args):
    """Check the frozen recipe/output identity, acquire one writer lock, then train.

    New runs create a directory; --resume requires its manifest and latest.pt.
    The lock covers checkpoint and history writes for the entire invocation.
    """
    config = json.loads(args.config.read_text())
    frozen = json.loads((ROOT / "config/training.json").read_text())
    if config != frozen:
        raise ValueError("This entry point preserves the submitted frozen recipe")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; allocate a GPU or explicitly use --device cpu")
    groups = discover(args.dataset_root)
    if args.run_dir.exists() != args.resume:
        raise ValueError("New output must be absent; --resume requires an existing run")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents two writers even on platforms without fcntl.
    lock = args.run_dir / ".writer.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    try:
        _train_locked(args, config, groups)
    finally:
        lock.unlink()


def _train_locked(args, config, groups):
    """Optimise paired XYZ crops and evaluate full Validation volumes every five epochs.

    epoch_index is zero-based; next_batch is the first unfinished batch. A saved
    next_batch=130 means optimisation ended but epoch finalisation may still be
    pending. Restored optimizer/RNG/exposure/count states continue that cursor.
    """
    seed = config["seed"]
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    model = build_model(args.model).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0)
    weights = torch.tensor(config["class_weights"], dtype=torch.float32, device=args.device)
    sources = {p.name: sha256_file(p) for p in
               [ROOT / n for n in ("modules.py", "dataset.py", "utils.py", "train.py")]
               + [ROOT / "config/patient_split.json", args.config]}
    identity = {"model": args.model, "config": config, "sources": sources,
                "dataset_root": str(args.dataset_root.resolve()), "torch": str(torch.__version__),
                "numpy": np.__version__, "device": args.device}
    epoch_index, next_batch, history, batch_losses = 0, 0, [], []
    best_score, best_epoch = -1., None
    exposure = ExposureRecorder()
    batch_matrix = np.zeros((6, 6), dtype=np.int64)
    if args.resume:
        prior = json.loads((args.run_dir / "manifest.json").read_text())
        if prior != identity: raise ValueError("Resume source/config/runtime identity changed")
        state = torch.load(args.run_dir / "latest.pt", map_location="cpu", weights_only=False)
        if state.get("format") != "submission_v1": raise ValueError("Not a submission-format recovery file")
        model.load_state_dict(state["model_state"]); optimizer.load_state_dict(state["optimizer_state"])
        epoch_index, next_batch = state["epoch_index"], state["next_batch"]
        history, batch_losses = state["history"], state["batch_losses"]
        batch_matrix = state["batch_matrix"]
        exposure = state["exposure"]
        best_score, best_epoch = state["best_score"], state["best_epoch"]
        # Restore RNG after model construction/loading so dropout continues its stream.
        torch.set_rng_state(state["torch_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        if args.device.startswith("cuda"): torch.cuda.set_rng_state_all(state["cuda_rng_states"])
    else:
        atomic_json(args.run_dir / "manifest.json", identity)
    cache = {}
    def checkpoint():
        """Snapshot weights and the exact next operation, including partial-epoch counts."""
        return {"format": "submission_v1", "model": args.model, "epoch_index": epoch_index,
                "next_batch": next_batch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "history": history,
                "batch_losses": batch_losses, "batch_matrix": batch_matrix, "exposure": exposure,
                "best_score": best_score, "best_epoch": best_epoch,
                "torch_rng_state": torch.get_rng_state(), "numpy_rng_state": np.random.get_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else []}
    if not args.resume: atomic_checkpoint(args.run_dir / "latest.pt", checkpoint())
    for epoch in range(epoch_index, config["epochs"]):
        epoch_index = epoch
        started = perf_counter(); model.train()
        chosen = epoch_cases(groups["train"], epoch=epoch, seed=seed)
        items = [(case, draw) for case in chosen for draw in range(10)]
        # A separate hash-derived batch order does not consume the dropout RNG.
        order_seed = int.from_bytes(hashlib.sha256(f"{seed}|{epoch}|batches".encode()).digest()[:8], "little")
        items = [items[int(i)] for i in np.random.default_rng(order_seed).permutation(len(items))]
        for batch in range(next_batch, len(items) // 2):
            xs, ys = [], []
            for case, draw in items[2*batch:2*batch+2]:
                if case.identifier not in cache:
                    cache[case.identifier] = zscore_nonzero(downsample(load_case(case)))
                volume = cache[case.identifier]
                origin = crop_origin(volume.mask, (64, 64, 32), "uniform", seed=seed,
                                     epoch=epoch, case_id=case.identifier, draw=draw)
                x, y = crop_for_3d_model(volume, origin)
                exposure.add(y, case.patient); xs.append(x); ys.append(y)
            x = torch.from_numpy(np.ascontiguousarray(np.stack(xs))).to(args.device)
            y = torch.from_numpy(np.ascontiguousarray(np.stack(ys))).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x); loss = ce_dice_loss(logits, y, ce_weight=.5, class_weights=weights)
            if not torch.isfinite(loss): raise ValueError("Nonfinite training loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Missing/nonfinite gradient")
            optimizer.step()
            prediction = logits.argmax(1).detach().cpu().numpy()
            truth = y.cpu().numpy()
            for a, b in zip(truth, prediction): batch_matrix += confusion(a, b)
            batch_losses.append(float(loss.detach())); next_batch = batch + 1
            # Advance the cursor only after optimizer.step; this batch is not repeated on resume.
            atomic_checkpoint(args.run_dir / "latest.pt", checkpoint())
            if next_batch % 10 == 0: print(f"epoch {epoch+1}: {next_batch}/130 loss={batch_losses[-1]:.5f}", flush=True)
        validation = evaluate(model, groups["validation"], args.device) if (epoch+1) % 5 == 0 else None
        improved = validation is not None and validation["selection_score"] > best_score
        if improved: best_score, best_epoch = validation["selection_score"], epoch + 1
        row = {"epoch": epoch+1, "loss": float(np.mean(batch_losses)),
               "train_overlap": overlap(batch_matrix), "train_exposure": exposure.summary(),
               "validation": validation, "learning_rate": .001,
               "invocation_epoch_seconds": perf_counter()-started,
               "timing_note": "Partial-epoch resume timing excludes prior invocation; not historical training benchmark"}
        history.append(row)
        atomic_json(args.run_dir / f"epoch_{epoch+1:03d}.json", row)
        epoch_index, next_batch, batch_losses = epoch+1, 0, []
        batch_matrix = np.zeros((6, 6), dtype=np.int64); exposure = ExposureRecorder()
        if improved: atomic_checkpoint(args.run_dir / "best.pt", checkpoint())
        atomic_checkpoint(args.run_dir / "latest.pt", checkpoint())
        if args.keep_epoch_weights: atomic_checkpoint(args.run_dir / f"epoch_{epoch+1:03d}.pt", checkpoint())
        plot_history(history, args.run_dir / "training.png")
        print(f"Completed epoch {epoch+1}; best Validation {best_score:.6f} at {best_epoch}", flush=True)



def inspect_record(args):
    """Recalculate one saved confusion matrix without reading MRI or model weights.

    This is an evidence/usage example, not a new inference run. Source and
    checkpoint identities accompany the exported six-class metrics so a reader
    can match the result to the locked experiment record.
    """
    bundle = json.loads(args.records.read_text(encoding="utf-8"))
    if bundle.get("class_order") != list(CLASS_NAMES):
        raise ValueError("Saved record has an unexpected class order")
    key = "Standard" if args.model == "standard" else "v3"
    model = bundle["models"][key]
    rows = [row for row in model["splits"][args.split] if row["case"] == args.case]
    if len(rows) != 1:
        raise ValueError("Expected exactly one saved model/split/case record")
    row = rows[0]
    matrix = np.asarray(row["confusion"], dtype=np.int64)
    metrics = overlap(matrix)
    result = {
        "kind": "saved_confusion_recalculation_not_new_inference",
        "case": args.case, "split": args.split, "model": key,
        "selected_epoch": model["best_epoch"],
        "checkpoint_sha256": model["selected"]["checkpoint_sha256"],
        "evidence_sha256": sha256_file(args.records),
        "input_sha256": {role: row[role + "_sha256"] for role in ("mri", "mask")},
        "shape_xyz": row["shape_xyz"], "spacing_mm": row["spacing_mm"],
        "confusion": matrix.tolist(),
        "per_class": {label: {name: value[index] for name, value in metrics.items()}
                      for index, label in enumerate(CLASS_NAMES)},
        "recorded_surface_distances": row["surfaces"],
    }
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError("Use a new record-export path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    """Expose training, fixed-weight evaluation and MRI-free saved-record inspection."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="Train/resume the frozen Standard or v3 recipe")
    fit.add_argument("--model", choices=("standard", "v3"), required=True)
    fit.add_argument("--dataset-root", type=Path, required=True)
    fit.add_argument("--run-dir", type=Path, required=True, help="New output directory, or an existing directory with --resume")
    fit.add_argument("--config", type=Path, default=ROOT / "config/training.json")
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--resume", action="store_true", help="Restore latest.pt, optimizer, RNG and next batch from the same run")
    fit.add_argument("--keep-epoch-weights", action="store_true", help="Also keep all epoch weights; increases disk use")
    test = commands.add_parser("evaluate", help="Evaluate trusted fixed weights on a frozen split")
    test.add_argument("--model", choices=("standard", "v3"), required=True)
    test.add_argument("--dataset-root", type=Path, required=True)
    test.add_argument("--checkpoint", type=Path, required=True)
    test.add_argument("--split", choices=("validation", "test"), required=True)
    test.add_argument("--output", type=Path, required=True)
    test.add_argument("--device", default="cuda")
    record = commands.add_parser("inspect-record", help="Recalculate saved metrics without MRI/weights")
    record.add_argument("--model", choices=("standard", "v3"), required=True)
    record.add_argument("--split", choices=("validation", "test"), default="validation")
    record.add_argument("--case", required=True, help="Recorded patient/visit, e.g. K008_Week0")
    record.add_argument("--records", type=Path, default=ROOT / "evidence/experiments.json")
    record.add_argument("--output", type=Path, help="Optional new JSON path; existing files are refused")
    args = parser.parse_args()
    if args.command == "train": train(args)
    elif args.command == "inspect-record": inspect_record(args)
    else:
        if args.output.exists(): raise ValueError("Refusing to overwrite evaluation output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        model = load_model(args.model, args.checkpoint, args.device)
        result = evaluate(model, discover(args.dataset_root)[args.split], args.device, detailed=True)
        result.update(model=args.model, split=args.split, checkpoint_sha256=sha256_file(args.checkpoint),
                      note="Fixed weights. Previously inspected Test is descriptive, not fresh blind validation.")
        atomic_json(args.output, result)


if __name__ == "__main__": main()
