"""Diagnose and tune reference TimeGAN; keep every experiment and the test set untouched."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from dataset import PROJECT_DIR, sha256_file
from train import fit_timegan, timegan_code_hashes, write_json, add_training_state_arguments
from timegan_report import diagnose, analyse_conditioning


def run_reconstruction(data_dir, output, hidden=24, steps=5000, seed=2026, balance=False, learning_rate=.001):
    """Fit only E/R, preserving the reference embedding update and held-out split."""
    import numpy as np
    import torch
    from dataset import prepare_datasets, FEATURE_NAMES
    from evaluation import validation_windows, metric_specification
    from modules import TimeGAN
    from protocol import comparison_protocol, comparison_signature
    from train import TimeGANTrainer, reconstruction_price_weight
    from timegan_report import reconstruction_scores

    if min(hidden, steps) < 1 or not 0 <= seed < 2**63 - 1000 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Use positive dimensions/budgets/rate and a valid seed.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Reconstruction output already exists; preserve it and use a new name.")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    sources = timegan_code_hashes()
    sources.update({name: hashlib.sha256((PROJECT_DIR / name).read_text(encoding="utf-8").encode()).hexdigest()
                    for name in ("tune_timegan.py", "timegan_report.py")})
    datasets, scaler, audit = prepare_datasets(data_dir, 32, 32)
    training, validation = datasets["train"], datasets["validation"]
    del datasets
    values = torch.stack([training[i]["features"] for i in range(len(training))])
    training_mean = values.double().mean((0, 1)).float()
    price_weight = reconstruction_price_weight(values, .5) if balance else 1.
    config = dict(architecture=dict(features=40, hidden=hidden, layers=3, noise_dim=40, cell="reference"),
                  training_seed=seed, embedding_steps=steps, batch_size=128, learning_rate=learning_rate,
                  balance_reconstruction=balance, balance_power=.5 if balance else 1., price_reconstruction_weight=price_weight,
                  structure_weight=0., threads=1, objective="reference E/R pretraining" + (" + partial price weighting" if balance else ""))
    evaluation = {}
    for name, dataset in (("train", training), ("validation", validation)):
        starts, anchors, real = validation_windows(dataset, 4)
        if name == "validation":
            signature = comparison_signature(comparison_protocol(), metric_specification(training), starts + dataset.raw_offset,
                                             audit["files"], scaler.to_dict())
        else:
            take = np.linspace(0, len(starts) - 1, min(512, len(starts)), dtype=int)
            starts, anchors, real = starts[take], anchors[take], real[take]
        features = torch.from_numpy(np.stack([dataset.features[s:s + 32] for s in starts]))
        evaluation[name] = (features, anchors, real, (starts + dataset.raw_offset).tolist())
    torch.manual_seed(seed)
    model = TimeGAN(**config["architecture"])
    def fingerprint(networks):
        digest = hashlib.sha256()
        for network in networks:
            for key, value in getattr(model, network).state_dict().items():
                digest.update((network + key).encode())
                digest.update(value.detach().numpy().tobytes())
        return digest.hexdigest()
    er_names, other_names = ("embedder", "recovery"), ("generator", "supervisor", "discriminator")
    initial, untouched = fingerprint(er_names), fingerprint(other_names)
    trainer = TimeGANTrainer(model, learning_rate=learning_rate, price_reconstruction_weight=price_weight)
    sampler, stream = torch.Generator().manual_seed(seed + 100), hashlib.sha256()
    history, evaluations = [], []
    def evaluate(step):
        model.eval()
        result = dict(step=step)
        with torch.no_grad():
            for split, (features, anchors, real, _) in evaluation.items():
                recovered = torch.cat([model.reconstruct(batch) for batch in features.split(128)])
                result[split] = reconstruction_scores(features, recovered, training_mean, scaler, anchors, real)
        evaluations.append(result)
        return result
    evaluate(0)
    training_seconds = 0.
    for step in range(1, steps + 1):
        tick = time.perf_counter()
        indices = torch.randperm(len(values), generator=sampler)[:128]
        stream.update(indices.numpy().tobytes())
        collect = step == 1 or step % 100 == 0 or step == steps
        losses = trainer.embedding_step(values[indices], collect=collect)
        training_seconds += time.perf_counter() - tick
        if collect:
            row = dict(step=step, training_seconds=training_seconds, **{key: float(value) for key, value in losses.items()})
            history.append(row)
            write_json(output / "progress.json", dict(config=config, history=history, elapsed_seconds=time.perf_counter() - started))
            print(f"embedding {step}/{steps}: mse={row['reconstruction_mse']:.7f}, seconds={training_seconds:.1f}", flush=True)
        if step in {1000, 2500, steps}:
            evaluate(step)
    if fingerprint(other_names) != untouched:
        raise RuntimeError("A reconstruction-only fit changed a non-E/R network.")
    checkpoint = dict(format_version=1, model_type="timegan_reconstruction_only", architecture=config["architecture"],
        networks={name: getattr(model, name).state_dict() for name in er_names}, config=config, feature_names=FEATURE_NAMES,
        scaler={"minimum": torch.from_numpy(scaler.minimum), "maximum": torch.from_numpy(scaler.maximum)},
        training_feature_mean=training_mean, source_files=sources, comparison_signature=signature)
    torch.save(checkpoint, output / "reconstruction.pt")
    saved = torch.load(output / "reconstruction.pt", weights_only=True, map_location="cpu")
    replay = TimeGAN(**saved["architecture"]).eval()
    for name in er_names:
        getattr(replay, name).load_state_dict(saved["networks"][name])
    difference = 0.
    model.eval()
    with torch.no_grad():
        for features in evaluation["validation"][0].split(128):
            difference = max(difference, float((model.reconstruct(features) - replay.reconstruct(features)).abs().max()))
    if difference != 0:
        raise RuntimeError("Reconstruction checkpoint replay was not exact.")
    report = dict(status="complete", experiment_type="reconstruction_only", config=config, source_files=sources,
        scope="Observed-input reconstruction; G/S/D untrained. Not a simulator or generation score.",
        fit_split="train", selection_split="validation", test_evaluated=False, input_files=audit["files"], splits=audit["splits"],
        comparison_signature=signature, raw_start_indices={k: v[3] for k, v in evaluation.items()},
        training_feature_mean=training_mean.tolist(), initial_er_sha256=initial, minibatch_stream_sha256=stream.hexdigest(),
        untrained_networks_unchanged=True, checkpoint_sha256=sha256_file(output / "reconstruction.pt"),
        reload_max_abs_difference=difference, history=history, evaluations=evaluations, final=evaluations[-1],
        training_seconds=training_seconds, total_seconds=time.perf_counter() - started,
        er_parameter_count=sum(p.numel() for name in er_names for p in getattr(model, name).parameters()))
    write_json(output / "results.json", report)
    return report


def run_candidate(data_dir, name, hidden=24, learning_rate=.001, embedding_steps=1000, supervised_steps=1000,
                  joint_steps=1000, seed=2026, eval_every=250, structure_weight=0., balance_reconstruction=False, balance_power=1., no_supervision=False, anchor_conditioning="none", price_decoder="none", volume_conditioning="none",
                  training_state_every=0, resume_state=None, stop_request=None, training_deadline=None, replay_checkpoint=None, replay_step=None):
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
        raise ValueError("Use a simple lowercase experiment name.")
    output = PROJECT_DIR / "outputs/timegan/experiments" / name
    report_path = output / "results.json"
    requested = dict(hidden=hidden, learning_rate=learning_rate, embedding_steps=embedding_steps,
                     supervised_steps=supervised_steps, joint_steps=joint_steps, seed=seed, eval_every=eval_every, structure_weight=structure_weight,
                     balance_reconstruction=balance_reconstruction, balance_power=balance_power, no_supervision=no_supervision, anchor_conditioning=anchor_conditioning, price_decoder=price_decoder, volume_conditioning=volume_conditioning)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        config = report["config"]
        actual = {key: config["architecture"]["hidden"] if key == "hidden" else config["training_seed"] if key == "seed" else config.get(key, "none" if key in ("anchor_conditioning", "price_decoder", "volume_conditioning") else 1. if key == "balance_power" else 0.)
                  for key in requested}
        if actual != requested or report["source_files"] != timegan_code_hashes():
            raise ValueError("Existing experiment differs; choose a new experiment name.")
        if sha256_file(output / "selected.pt") != report["selected_checkpoint_sha256"]:
            raise ValueError("Existing experiment checkpoint differs from its report.")
    else:
        report = fit_timegan(data_dir, output_dir=output, report_path=report_path, hidden=hidden, learning_rate=learning_rate,
                            embedding_steps=embedding_steps, supervised_steps=supervised_steps, joint_steps=joint_steps,
                            seed=seed, eval_every=eval_every, device="cpu", threads=1, structure_weight=structure_weight,
                            balance_reconstruction=balance_reconstruction, balance_power=balance_power, no_supervision=no_supervision, anchor_conditioning=anchor_conditioning, price_decoder=price_decoder, volume_conditioning=volume_conditioning,
                            training_state_every=training_state_every, resume_state=resume_state, stop_request=stop_request,
                            training_deadline=training_deadline, replay_checkpoint=replay_checkpoint, replay_step=replay_step)
        if report.get("status") == "incomplete":
            return report
    collect_summary()
    return report


def collect_summary():
    """Rebuild from completed reports, retaining historical diagnostics and all attempts."""
    summary_path = PROJECT_DIR / "reports/timegan/tuning.json"
    for attempt in range(5):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else dict(
                selection="Validation only: invalid rate, mean KL, ACF error, earlier checkpoint; no structural repair.",
                test_evaluated=False, candidates={})
            break
        except json.JSONDecodeError:
            if attempt == 4:
                raise
            time.sleep(.1)  # Older, already-running pilots predate atomic summary writes.
    for saved in sorted((PROJECT_DIR / "outputs/timegan/experiments").glob("*/results.json")):
        try:
            run = json.loads(saved.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue  # Another process may be finishing its report; final collection retries it.
        group_name = "volume_conditioned_candidates" if run["config"].get("volume_conditioning", "none") != "none" else "conditional_candidates" if run["config"].get("anchor_conditioning", "none") != "none" else "candidates"
        group = summary.setdefault(group_name, {})
        group[saved.parent.name] = {key: run[key] for key in ("config", "selected", "validation_history", "history",
            "total_seconds", "stage_training_seconds", "source_files", "comparison_signature", "selected_checkpoint_sha256", "reload_max_abs_difference")}
        group[saved.parent.name]["checkpoint"] = (saved.parent / "selected.pt").relative_to(PROJECT_DIR).as_posix()
    summary["runner_sha256"] = sha256_file(Path(__file__))
    write_json(summary_path, summary)
    return summary


def refresh_notebook():
    """Refresh the existing report; no additional results notebook."""
    import asyncio
    import nbformat
    from nbclient import NotebookClient
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    path = PROJECT_DIR / "timegan_results.ipynb"
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=180, startup_timeout=30, kernel_name="python3", resources={"metadata": {"path": str(PROJECT_DIR)}}).execute()
    nbformat.write(notebook, path)
    print("TimeGAN results notebook refreshed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--reconstruction-only", action="store_true", help="Fit E/R only; --balance-reconstruction uses partial (power 0.5) weighting.")
    parser.add_argument("--probe-output", type=Path)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--conditioning-report", action="store_true", help="Regenerate diagnostics from saved matched/zero fits; no training.")
    parser.add_argument("--matched-run", type=Path, default=PROJECT_DIR / "outputs/timegan/experiments/context_matched_s2026")
    parser.add_argument("--zero-run", type=Path, default=PROJECT_DIR / "outputs/timegan/experiments/context_zero_s2026")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_DIR / "outputs/timegan/conditioning_reproduced")
    parser.add_argument("--refresh-notebook", action="store_true", help="Refresh cached report outputs only.")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_DIR / "outputs/timegan/selected.pt")
    parser.add_argument("--diagnostic-output", type=Path, default=PROJECT_DIR / "reports/timegan/diagnostics.json")
    parser.add_argument("--name")
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--embedding-steps", type=int, default=1000)
    parser.add_argument("--supervised-steps", type=int, default=1000)
    parser.add_argument("--joint-steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--structure-weight", type=float, default=0.)
    parser.add_argument("--balance-reconstruction", action="store_true")
    parser.add_argument("--balance-power", type=float, default=1.)
    parser.add_argument("--no-supervision", action="store_true", help="Remove temporal supervision, retaining the five-network architecture.")
    parser.add_argument("--anchor-conditioning", choices=("none","matched","zero"), default="none")
    parser.add_argument("--price-decoder", choices=("none", "ladder"), default="none")
    parser.add_argument("--volume-conditioning", choices=("none", "zero", "matched"), default="none", help="Labelled extension: starting volumes or same-dimension zero-volume control; requires matched price context.")
    add_training_state_arguments(parser)
    args = parser.parse_args()
    if args.reconstruction_only:
        if args.data_dir is None or args.probe_output is None or args.name or args.diagnose or args.refresh_notebook or args.conditioning_report or args.structure_weight or args.no_supervision or args.anchor_conditioning != "none" or args.price_decoder != "none" or args.volume_conditioning != "none":
            parser.error("Reconstruction-only needs --data-dir and a fresh --probe-output, separately from other modes.")
        if args.balance_power not in (1., .5) or (not args.balance_reconstruction and args.balance_power != 1.):
            parser.error("The reconstruction pilot compares ordinary weights with partial (power 0.5) weighting only.")
        run_reconstruction(args.data_dir, args.probe_output, args.hidden, args.embedding_steps, args.seed,
                           args.balance_reconstruction, args.learning_rate)
        return
    if args.probe_output is not None:
        parser.error("--probe-output requires --reconstruction-only.")
    if args.refresh_notebook or args.conditioning_report:
        if args.name or args.diagnose or (args.refresh_notebook and args.conditioning_report):
            parser.error("Run report commands separately from training or other diagnostics.")
    if args.refresh_notebook:
        refresh_notebook()
        return
    if args.data_dir is None:
        parser.error("--data-dir is required for training and diagnostics.")
    if args.conditioning_report:
        report = analyse_conditioning(args.data_dir, args.matched_run, args.zero_run, args.report_dir)
        print(report["conclusion"])
        print(f"Saved report: {args.report_dir / 'conditioning.json'}")
        return
    if args.diagnose:
        result = diagnose(args.data_dir, args.checkpoint, args.diagnostic_output)
        for split, row in result["splits"].items():
            print(split, {"reconstruction": row["reconstruction"], "feature_groups": row["feature_groups"],
                          "shuffled_anchor_mean": row["shuffled_anchor_mean"]}, flush=True)
    if args.name:
        report = run_candidate(args.data_dir, args.name, args.hidden, args.learning_rate, args.embedding_steps,
                               args.supervised_steps, args.joint_steps, args.seed, args.eval_every, args.structure_weight, args.balance_reconstruction, args.balance_power, args.no_supervision, args.anchor_conditioning, args.price_decoder, args.volume_conditioning,
                               args.training_state_every, args.resume_state, args.stop_request, args.training_deadline, args.replay_checkpoint, args.replay_step)
        if report.get("status") == "incomplete":
            print(args.name, "incomplete", report["stage"], report["step"], "full state saved", flush=True)
        else:
            print(args.name, "selected", report["selected"]["step"], "invalid", report["selected"]["invalid_rate_mean"], flush=True)
    if not args.diagnose and not args.name:
        parser.error("Choose --diagnose and/or --name.")


if __name__ == "__main__":
    main()
