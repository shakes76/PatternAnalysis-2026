"""Run the fixed 20-epoch Improved U-Net experiment and validation without chat polling.

Launch this with the project Python. Logs, status and provenance stay under runs/.
An existing output directory is refused; failures are recorded, never auto-restarted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parent


def run_experiment(output: Path, manifest: Path) -> None:
    output, manifest = Path(output).resolve(), Path(manifest).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest}")
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "starting", "wrapper_pid": os.getpid(),
             "started_utc": datetime.now(timezone.utc).isoformat(),
             "test_evaluated": False}

    def record(**updates):
        state.update(updates)
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "status.json")

    record()
    try:
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        provenance = {"manifest_sha256": sha(manifest), "python": sys.executable,
                      "source_sha256": {name: sha(PROJECT / name) for name in
                          ("train.py", "evaluate_cases.py", "dataset.py", "modules.py",
                           "metrics.py", "labels.json", "run_improved_experiment.py")}}
        (output / "launch_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        training = [sys.executable, "-u", str(PROJECT / "train.py"),
                    "--model", "improved_unet", "--manifest", str(manifest),
                    "--label-values", "0", "1", "2", "3", "4", "5", "--prostate-label", "5",
                    "--image-size", "256", "--resize-mode", "pad", "--base-channels", "16",
                    "--batch-size", "4", "--epochs", "20", "--lr", "0.001", "--seed", "3710",
                    "--device", "cuda", "--output", str(output)]
        evaluation = [sys.executable, "-u", str(PROJECT / "evaluate_cases.py"),
                      "--checkpoint", str(output / "best.pt"), "--manifest", str(manifest),
                      "--output", str(output / "case_validation"), "--device", "cuda", "--batch-size", "4"]
        for phase, command in (("training", training), ("evaluating", evaluation)):
            record(status=phase)
            with (output / f"{phase}.log").open("w", encoding="utf-8") as stdout, \
                    (output / f"{phase}_stderr.log").open("w", encoding="utf-8") as stderr:
                subprocess.run(command, cwd=PROJECT, stdout=stdout, stderr=stderr, check=True)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        cases = json.loads((output / "case_validation/summary.json").read_text(encoding="utf-8"))
        if (summary["status"] != "completed" or not summary["checkpoint_reload_exact"]
                or len(summary["history"]) != 20 or cases["split"] != "val" or cases["test_evaluated"]):
            raise RuntimeError("Output did not pass completion checks")
        record(status="completed", best_epoch=summary["best_epoch"],
               case_mean_prostate_dice=cases["prostate_case_dice"]["mean"])
    except BaseException as exc:
        record(status="failed", failed_phase=state["status"], error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "runs/real_improved_unet_20ep_s3710")
    parser.add_argument("--manifest", type=Path, default=PROJECT / "private_data/manifests/course_v1.csv")
    args = parser.parse_args()
    run_experiment(args.output, args.manifest)
