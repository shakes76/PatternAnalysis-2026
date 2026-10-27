"""Two bounded B-weight warm-start recovery arms; no TEST pixels or selection.

Each arm gets a fresh optimizer/scaler and eight full TRAIN epochs. The frozen
B checkpoint, A1 sources and earlier results are never resumed or overwritten.
"""
from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path
import platform
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from experiments.evaluate_appearance import MANIFEST_SHA, REFERENCES, SOURCES_SHA, require
from experiments.recovery import POLARITY_PROBABILITIES
from tools.cache_identity import safe_relative
from utils import file_sha256, write_json

EXPERIMENT = "appearance_recovery_v1"
PARENT_SHA = "b13ba58fe8837fb9522b6108412281c51210a3be486011d59f902347ade2a82b"
APPEARANCE_SHA = "5b773d1369d8e7c772172530f909bdf6c720f522786aa2ed72cd045067224632"


def config_for_plan(plan, appearance):
    expected = dict(appearance["training_conditions"], lr=1e-4)
    require(plan["training_conditions"] == expected and expected["seed"] == 42
            and expected["workers"] == 0 and expected["amp"] is True, "Recovery config may differ only in the fixed lower learning rate.")
    return dict(expected)


def verified_plan(path):
    path = Path(path).expanduser().resolve(strict=True)
    plan = json.loads(path.read_text())
    require(plan["experiment_id"] == EXPERIMENT and plan["training_seed"] == 42
            and plan["training_epochs_per_model"] == 8 and plan["allowed_splits"] == ["train", "validation"], "Wrong recovery experiment scope/budget.")
    appearance_path = safe_relative(PROJECT, plan["appearance_protocol"])
    require(file_sha256(appearance_path) == plan["appearance_protocol_sha256"] == APPEARANCE_SHA, "Frozen A1 protocol differs.")
    appearance = json.loads(appearance_path.read_text())
    require(plan["manifest_sha256"] == MANIFEST_SHA and plan["training_sources_sha256"] == SOURCES_SHA,
            "Original data/source identities differ.")
    require(plan["manifest"] == appearance["manifest"] and plan["cache_identity"] == appearance["cache_identity"], "Canonical frozen cache paths differ.")
    require(plan["warm_start"] == {"checkpoint": appearance["models"]["gamma_polarity"]["checkpoint"],
                                  "checkpoint_sha256": PARENT_SHA}, "Warm-start must use exactly the frozen B best checkpoint.")
    require(set(plan["models"]) == set(POLARITY_PROBABILITIES), "Exactly two recovery arms are required.")
    for name, probability in POLARITY_PROBABILITIES.items():
        require(plan["models"][name]["kind"] == "recovery_arm"
                and plan["models"][name]["polarity_probability"] == probability
                and Path(plan["models"][name]["checkpoint"]).name == "best.pt", "Recovery polarity probability/checkpoint differs.")
    require(len({safe_relative(PROJECT, m["checkpoint"]).parent for m in plan["models"].values()}) == 2, "Recovery arms must have distinct output directories.")
    config_for_plan(plan, appearance)
    required = plan["required_file_sha256"]
    require({"experiments/recovery.py", "experiments/train_recovery.py", "experiments/train_appearance.py",
             "experiments/appearance.py", "modules.py", "dataset.py", "train.py", "utils.py"} <= set(required), "Recovery source guards are incomplete.")
    for book in (appearance["required_file_sha256"], required):
        require(all(file_sha256(safe_relative(PROJECT, p)) == digest for p, digest in book.items()), "A protected source/input changed.")
    return plan, file_sha256(path), appearance


def warm_start_guard(checkpoint, plan, appearance):
    expected = {"experiment_id": appearance["experiment_id"], "arm_tag": "gamma_polarity", "protocol_sha256": APPEARANCE_SHA}
    require(checkpoint["config"] == appearance["training_conditions"] and checkpoint["experiment"] == expected
            and checkpoint["epoch"] == checkpoint["best_epoch"] == 29, "Warm-start checkpoint/config is not the frozen B best model.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(POLARITY_PROBABILITIES), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke-output", type=Path, help="Development only: fresh output, five TRAIN batches/two VAL cases, one epoch.")
    args = parser.parse_args(argv)
    plan, protocol_sha, appearance = verified_plan(args.protocol)
    import torch
    from torch.utils.data import DataLoader
    from dataset import HipMRIVolumeDataset, load_manifest
    from experiments.evaluate_appearance import finite_main_logits, validate_arm_files
    from experiments.recovery import RecoveryTrainingDataset
    from experiments.train_appearance import verify_development_cache
    from modules import ImprovedUNet3D, segmentation_loss
    from tools.diagnose_model_mechanism import state_sha256
    from train import optimizer_step, resolve_training_class_weights, rng_state, save_checkpoint, seed_everything, seed_worker, restore_rng
    from utils import evaluate_volumes, plot_training, synchronize

    manifest_path, identity_path = (safe_relative(PROJECT, plan[k]) for k in ("manifest", "cache_identity"))
    manifest = load_manifest(manifest_path)
    require(file_sha256(manifest_path) == plan["manifest_sha256"], "Frozen manifest changed.")
    device, smoke = torch.device(args.device), args.smoke_output is not None
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable.")
    require(smoke or (device.type == "cuda" and torch.cuda.device_count() == 1), "Formal recovery requires exactly one visible CUDA GPU.")
    parent_path = safe_relative(PROJECT, plan["warm_start"]["checkpoint"])
    require(file_sha256(parent_path) == PARENT_SHA, "Parent checkpoint SHA differs.")
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    warm_start_guard(parent, plan, appearance)
    parent_receipt = validate_arm_files(parent, parent["config"], "gamma_polarity", parent_path, manifest,
                                       APPEARANCE_SHA, appearance["required_file_sha256"])
    require(json.loads((parent_path.parent / "provenance.json").read_text())["history_sha256"] == parent_receipt["history_sha256"], "Parent history SHA differs.")
    output = args.smoke_output.expanduser().absolute() if smoke else safe_relative(PROJECT, plan["models"][args.variant]["checkpoint"]).parent
    old_dirs = [manifest_path.parent, *[safe_relative(PROJECT, m["checkpoint"]).parent for m in appearance["models"].values()]]
    require(not output.exists() and not output.is_symlink(), "Recovery output must be fresh; resume/restart into an old directory is forbidden.")
    require(all(not output.resolve().is_relative_to(p) for p in old_dirs), "Output enters a protected old checkpoint/cache directory.")
    if smoke:
        require(all(not output.resolve().is_relative_to(safe_relative(PROJECT, m["checkpoint"]).parent) for m in plan["models"].values()), "Smoke output enters a formal arm directory.")
    cache_evidence = verify_development_cache(manifest_path, identity_path, manifest)
    config = config_for_plan(plan, appearance)
    if smoke:
        config.update(max_batches=5, development_only=True, amp=device.type == "cuda")
    experiment = {"experiment_id": EXPERIMENT, "arm_tag": args.variant, "protocol_sha256": protocol_sha}
    seed_everything(config["seed"])
    training = RecoveryTrainingDataset(manifest_path, args.variant, config["seed"])
    validation = HipMRIVolumeDataset(manifest_path, "validation", augment=False)
    class_weights, counts, weights, _ = resolve_training_class_weights(manifest_path, manifest, config["class_weighting"], device)
    require(counts == config["training_class_voxel_counts"] and weights == config["resolved_class_weights"], "TRAIN-only class weights differ.")
    model = ImprovedUNet3D(6, config["base_channels"]).to(device)
    model.load_state_dict(parent["state_dict"], strict=True)
    model.register_forward_hook(finite_main_logits)
    initial_state = state_sha256(model)
    del parent
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])
    require(not optimizer.state, "Warm-start optimizer must be fresh.")
    output.mkdir(parents=True, exist_ok=False)
    cache_report = output / "cache_verification.json"
    write_json(cache_report, cache_evidence)
    history, best_score, best_epoch = [], None, None
    run = {"status": "running", "experiment": experiment, "config": config, "parent_checkpoint_sha256": PARENT_SHA,
           "parent_epoch": 29, "initial_parameter_state_sha256": initial_state, "warm_start_weights_only": True,
           "optimizer_reset": True, "scaler_reset": True, "requested_total_epochs": 1 if smoke else 8,
           "development_only": smoke, "source_file_sha256": plan["required_file_sha256"],
           "cache_verification_sha256": file_sha256(cache_report),
           "selection_metric": "Original whole-volume patient-balanced clean foreground Dice; earliest epoch wins ties; no TEST selection.",
           "runtime": {"gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                       "python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda},
           "determinism": "Same starting B weights and seeded streams; unsupported CUDA CE/maxpool kernels remain warn-only nondeterministic."}
    write_json(output / "run.json", run)
    try:
        for epoch in range(1, (1 if smoke else 8) + 1):
            training.set_epoch(epoch); training.reset_counts()
            loader = DataLoader(training, batch_size=1, shuffle=True, num_workers=0, pin_memory=device.type == "cuda",
                                worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(config["seed"] + epoch))
            model.train(); synchronize(device)
            if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter(); loss_sum = 0.; batches = examples = skipped = 0
            for batch in islice(loader, 5) if smoke else loader:
                images, targets = batch["image"].to(device), batch["mask"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=config["amp"]):
                    outputs = model(images)
                    loss = segmentation_loss(outputs, targets, class_weights=class_weights)
                require(bool(torch.isfinite(loss).item()), "Nonfinite recovery training loss.")
                scaler.scale(loss).backward()
                gradient_norm, was_skipped = optimizer_step(model, optimizer, scaler, config["clip_grad"], config["amp"])
                skipped += int(was_skipped); batches += 1; examples += int(images.shape[0]); loss_sum += float(loss.item()) * int(images.shape[0])
            require(batches == (5 if smoke else 143) and examples == batches and skipped < batches, "Incomplete training coverage or no successful updates.")
            policy_counts = dict(training.policy_counts)
            require(policy_counts["noninverted"] + policy_counts["inverted"] == examples, "Training exposure counters differ.")
            synchronize(device); seconds = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            optimizer.zero_grad(set_to_none=True)
            del outputs, loss, images, targets
            evaluation = evaluate_volumes(model, validation, "improved3d", 6, device, config["eval_batch_size"], config["amp"], max_cases=2 if smoke else None)
            score = evaluation["aggregate"]["foreground_mean_dice"]
            require(score is not None, "Undefined clean validation selector.")
            improved = best_score is None or score > best_score
            if improved: best_score, best_epoch = score, epoch
            history.append({"epoch": epoch, "train_loss": loss_sum / examples, "training_batches": batches, "training_examples": examples,
                            "policy_counts": policy_counts, "train_seconds": seconds, "peak_allocated_cuda_bytes_training": peak,
                            "last_gradient_norm_before_clipping": gradient_norm, "optimizer_steps": batches - skipped,
                            "amp_overflow_skipped_optimizer_steps": skipped, "validation_foreground_dice": score,
                            "validation_foreground_iou": evaluation["aggregate"]["foreground_mean_iou"], "validation": evaluation, "development_only": smoke})
            payload = {"schema_version": 3, "experiment": experiment, "config": config, "parent_checkpoint_sha256": PARENT_SHA,
                       "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                       "epoch": epoch, "best_score": best_score, "best_epoch": best_epoch, "history": history, "rng_state": rng_state()}
            save_checkpoint(output / "latest.pt", payload)
            if improved: save_checkpoint(output / "best.pt", payload)
            write_json(output / "history.json", history)
            try: plot_training(history, output / "curves.png")
            finally: restore_rng(payload["rng_state"])
            run.update(completed_epochs=epoch, best_epoch=best_epoch, best_validation_foreground_dice=best_score,
                       training_policy_counts={k: sum(r["policy_counts"][k] for r in history) for k in policy_counts})
            write_json(output / "run.json", run)
            print(f"{args.variant} recovery {epoch}/{1 if smoke else 8}: loss={history[-1]['train_loss']:.5f}; clean VAL FG={score:.5f}", flush=True)
        verified_plan(args.protocol)
        require(file_sha256(parent_path) == PARENT_SHA, "Warm-start file changed during recovery.")
        selected = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
        require(selected["epoch"] == best_epoch and selected["best_epoch"] == best_epoch and selected["config"] == config
                and selected["experiment"] == experiment and selected["parent_checkpoint_sha256"] == PARENT_SHA
                and len(history) == (1 if smoke else 8), "Selected recovery checkpoint/history identity differs.")
        run["status"] = "completed_development_smoke" if smoke else "completed"
        write_json(output / "run.json", run)
        if not smoke:
            write_json(output / "provenance.json", {"experiment": experiment, "parent_checkpoint_sha256": PARENT_SHA,
                "parent_epoch": 29, "warm_start_weights_only": True, "optimizer_reset": True, "scaler_reset": True,
                "checkpoint_sha256": file_sha256(output / "best.pt"), "run_sha256": file_sha256(output / "run.json"),
                "history_sha256": file_sha256(output / "history.json"), "protocol_sha256": protocol_sha,
                "manifest_sha256": file_sha256(manifest_path), "source_file_sha256": plan["required_file_sha256"],
                "cache_verification_sha256": file_sha256(cache_report), "cache_verification_path": str(cache_report.relative_to(PROJECT))})
    except BaseException as error:
        run.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "run.json", run)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
