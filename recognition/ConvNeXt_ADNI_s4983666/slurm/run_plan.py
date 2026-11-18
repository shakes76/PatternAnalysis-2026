"""Run the training cases of a JSON experiment plan one after another.

Plan format (see config/experiments_*.json):

    {"name": "...", "common": {"fold": 1, ...},
     "cases": [{"id": "S1_lite_seed3710", "purpose": "...", "args": {"model": "convnext_lite", ...}}]}

Each case runs ``train.py`` with ``common`` and its own ``args`` (keys are
train.py options without the leading dashes; ``true`` means a flag) and writes
to ``<runs-root>/<plan name>/<case id>``. Training is skipped for a case whose
``metrics.json`` already exists, so a plan can be resumed after a time-out.
``--smoke`` runs every case for one epoch to check the commands.

After each case, ``predict.py`` reloads ``final.pt`` and re-scores the val
patients; ``replay_check.json`` records whether accuracy and AUROC at the
slice, scan and patient level match the training run (a mismatch counts as a
failed case).
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[1]


def case_command(case: dict, common: dict, output: Path, base_args: list[str], smoke: bool) -> list[str]:
    """Build the train.py command line of one case."""
    options = {**common, **case["args"]}
    if smoke:
        options.update(epochs=1, warmup_epochs=0, skip_inference_profile=True)
    command = [sys.executable, "-u", str(SOURCE / "train.py"), *base_args, "--output", str(output)]
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            command.append(flag)
        elif value is not False and value is not None:
            # String values may use ~ or $VARS (e.g. the location of downloaded pretrained weights).
            command.extend([flag, os.path.expandvars(os.path.expanduser(value)) if isinstance(value, str) else str(value)])
    return command


def replay_check(output: Path, options: dict, base_args: list[str]) -> bool:
    """Reload the saved checkpoint, re-score the val patients and compare with metrics.json."""
    if (output / "replay_check.json").exists():
        return json.loads((output / "replay_check.json").read_text())["match"]
    command = [sys.executable, "-u", str(SOURCE / "predict.py"), *base_args,
               "--checkpoint", str(output / "final.pt"), "--output", str(output / "replay"), "--role", "val",
               "--batch-size", str(options.get("batch_size", 32)), "--skip-inference-profile"]
    if options.get("device"):
        command.extend(["--device", str(options["device"])])
    result = subprocess.run(command, cwd=SOURCE, check=False, stdout=subprocess.DEVNULL)
    if result.returncode != 0:
        return False
    trained = json.loads((output / "metrics.json").read_text())["metrics"]
    replayed = json.loads((output / "replay" / "metrics.json").read_text())["metrics"]
    differences = {f"{unit}_{name}": abs(trained[unit][name] - replayed[unit][name])
                   for unit in ("slice", "scan", "patient") for name in ("accuracy", "auroc")}
    match = all(value <= 1e-6 for value in differences.values())
    (output / "replay_check.json").write_text(json.dumps(
        {"role": "val", "match": match, "max_abs_difference": differences}, indent=2) + "\n")
    return match


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("/home/groups/comp3710/ADNI"))
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--cases", nargs="*", default=None, help="Run only these case ids.")
    parser.add_argument("--smoke", action="store_true", help="One epoch per case.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them.")
    args = parser.parse_args(argv)

    plan = json.loads(args.plan.read_text())
    cases = [case for case in plan["cases"] if args.cases is None or case["id"] in args.cases]
    if args.cases and len(cases) != len(set(args.cases)):
        parser.error("Unknown case id in --cases.")
    run_dir = args.runs_root / (plan["name"] + ("_smoke" if args.smoke else ""))
    base_args = ["--data-root", str(args.data_root), "--splits-dir", str(args.splits_dir)]
    failures = 0
    for number, case in enumerate(cases, start=1):
        output = run_dir / case["id"]
        command = case_command(case, plan.get("common", {}), output, base_args, args.smoke)
        print(f"[{number}/{len(cases)}] {case['id']}: {case.get('purpose', '')}", flush=True)
        if args.dry_run:
            print(" ".join(command), flush=True)
            continue
        if (output / "metrics.json").exists():
            print("  training already complete", flush=True)
        elif output.exists():
            print(f"  incomplete output exists; remove {output} to rerun", flush=True)
            failures += 1
            continue
        else:
            result = subprocess.run(command, cwd=SOURCE, check=False)
            if result.returncode != 0:
                print(f"  FAILED with exit code {result.returncode}", flush=True)
                failures += 1
                continue
        options = {**plan.get("common", {}), **case["args"]}
        matched = replay_check(output, options, base_args)
        print(f"  checkpoint replay {'matches' if matched else 'MISMATCH'}", flush=True)
        failures += not matched
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
