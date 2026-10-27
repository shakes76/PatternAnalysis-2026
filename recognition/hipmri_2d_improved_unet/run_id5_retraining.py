"""Run the two approved ID5-selected experiments in order, preserving logs."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    protocol = plan["shared_protocol"]
    if (plan["status"] != "approved_for_execution" or not plan["user_confirmation_received"]
            or plan["metrics_and_model_selection"]["checkpoint_selection"]["selection_metric"] != "id5_dice"
            or plan["epochs_per_model"] != 20):
        raise ValueError("Expected the approved two-model 20-epoch ID5 selection plan.")
    if [run["model"] for run in plan["runs_in_order"]] != ["baseline_unet", "improved_unet"]:
        raise ValueError("Unexpected model inventory or run order.")
    for name, digest in plan["source_sha256"].items():
        if sha(PROJECT / name) != digest:
            raise ValueError(f"Training source changed: {name}")
    for name, digest in plan["protected_original_files_sha256"].items():
        if sha(REPO / name) != digest:
            raise ValueError(f"Original record changed: {name}")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for run in plan["runs_in_order"]:
        if (REPO / run["output_dir"]).exists() or (REPO / run["console_log"]).exists():
            raise FileExistsError(f"Prior output exists for {run['model']}")
    args.output_dir.mkdir(parents=True)
    state = {"created_at": datetime.now(timezone(timedelta(hours=10))).isoformat(),
             "status": "running", "formal_plan_sha256": sha(args.plan),
             "test_split_read": False, "runs": []}
    state_path = args.output_dir / "execution.json"

    def persist():
        state_path.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    persist()
    try:
        for run in plan["runs_in_order"]:
            command = [sys.executable, "-B", "-u", str(PROJECT / "train.py"),
                       "--data-root", str(args.data_root.resolve()), "--model", run["model"],
                       "--output-dir", str(REPO / run["output_dir"]),
                       "--epochs", "20", "--batch-size", str(protocol["batch_size"]),
                       "--base-channels", str(protocol["model_config"]["base_channels"]),
                       "--num-classes", str(protocol["model_config"]["num_classes"]),
                       "--lr", str(protocol["optimizer"]["learning_rate"]),
                       "--seed", str(protocol["seed"]), "--device", "cuda",
                       "--formal-plan", str(args.plan.resolve()),
                       "--selection-metric", "id5_dice", "--progress-every", "1000"]
            entry = {"model": run["model"], "command": command, "status": "running"}
            state["runs"].append(entry)
            persist()
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
            environment["MPLCONFIGDIR"] = str(args.output_dir.resolve() / ".matplotlib")
            with (REPO / run["console_log"]).open("x", encoding="utf-8") as log:
                with subprocess.Popen(command, cwd=REPO, env=environment, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding="utf-8") as process:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    entry["exit_code"] = process.wait()
            entry["status"] = "passed" if entry["exit_code"] == 0 else "incomplete"
            persist()
            if entry["exit_code"]:
                raise RuntimeError(f"{run['model']} failed; preserving evidence and stopping.")
        state["status"] = "passed"
    except BaseException as error:
        state["status"] = "incomplete"
        state["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
