"""Diagnose and tune reference TimeGAN; keep every experiment and the test set untouched."""

import argparse
import json
import sys
import time
from pathlib import Path

from dataset import PROJECT_DIR, sha256_file
from train import fit_timegan, timegan_code_hashes, write_json
from timegan_report import diagnose, analyse_conditioning


def run_candidate(data_dir, name, hidden=24, learning_rate=.001, embedding_steps=1000, supervised_steps=1000,
                  joint_steps=1000, seed=2026, eval_every=250, structure_weight=0., balance_reconstruction=False, balance_power=1., no_supervision=False, anchor_conditioning="none"):
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
        raise ValueError("Use a simple lowercase experiment name.")
    output = PROJECT_DIR / "outputs/timegan/experiments" / name
    report_path = output / "results.json"
    requested = dict(hidden=hidden, learning_rate=learning_rate, embedding_steps=embedding_steps,
                     supervised_steps=supervised_steps, joint_steps=joint_steps, seed=seed, eval_every=eval_every, structure_weight=structure_weight,
                     balance_reconstruction=balance_reconstruction, balance_power=balance_power, no_supervision=no_supervision, anchor_conditioning=anchor_conditioning)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        config = report["config"]
        actual = {key: config["architecture"]["hidden"] if key == "hidden" else config["training_seed"] if key == "seed" else config.get(key, "none" if key == "anchor_conditioning" else 1. if key == "balance_power" else 0.)
                  for key in requested}
        if actual != requested or report["source_files"] != timegan_code_hashes():
            raise ValueError("Existing experiment differs; choose a new experiment name.")
        if sha256_file(output / "selected.pt") != report["selected_checkpoint_sha256"]:
            raise ValueError("Existing experiment checkpoint differs from its report.")
    else:
        report = fit_timegan(data_dir, output_dir=output, report_path=report_path, hidden=hidden, learning_rate=learning_rate,
                            embedding_steps=embedding_steps, supervised_steps=supervised_steps, joint_steps=joint_steps,
                            seed=seed, eval_every=eval_every, device="cpu", threads=1, structure_weight=structure_weight,
                            balance_reconstruction=balance_reconstruction, balance_power=balance_power, no_supervision=no_supervision, anchor_conditioning=anchor_conditioning)
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
        group = summary.setdefault("conditional_candidates", {}) if run["config"].get("anchor_conditioning", "none") != "none" else summary["candidates"]
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
    args = parser.parse_args()
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
                               args.supervised_steps, args.joint_steps, args.seed, args.eval_every, args.structure_weight, args.balance_reconstruction, args.balance_power, args.no_supervision, args.anchor_conditioning)
        print(args.name, "selected", report["selected"]["step"], "invalid", report["selected"]["invalid_rate_mean"], flush=True)
    if not args.diagnose and not args.name:
        parser.error("Choose --diagnose and/or --name.")


if __name__ == "__main__":
    main()
