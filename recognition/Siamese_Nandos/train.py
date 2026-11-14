"""Choose which COMP3710 model training script to run.

This entry point keeps the two training implementations separate while giving
the project one predictable command named `train.py`. Choose one model for each
run; use a different output directory if you train both models.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run either the ResNet-18 baseline trainer or the Siamese auxiliary-head trainer. "
            "Options after --model are passed to the selected script."
        )
    )
    parser.add_argument(
        "--model",
        choices=("baseline", "siamese_aux"),
        required=True,
        help="Select which model to train.",
    )

    # Keep each trainer's own command-line options in its original script. The
    # dispatcher only reads --model and forwards everything else unchanged.
    args, training_args = parser.parse_known_args()

    # Allow an optional separator when forwarding a trainer's own --help flag,
    # for example: python train.py --model baseline -- --help.
    if training_args and training_args[0] == "--":
        training_args = training_args[1:]

    script_name = {
        "baseline": "train_baseline.py",
        "siamese_aux": "train_aux.py",
    }[args.model]
    script_path = Path(__file__).resolve().with_name(script_name)
    if not script_path.is_file():
        raise FileNotFoundError(f"Training script was not found: {script_path}")

    # Replace the dispatcher so cancellation targets the trainer itself.
    # Unbuffered output also applies when stdout is piped by the notebook.
    command = [sys.executable, "-u", str(script_path), *training_args]
    print(f"Starting {args.model} training with {script_name}", flush=True)
    if os.name == "nt":
        # Windows exec does not preserve the caller's wait/exit semantics.
        # Keep an explicitly waited child for local tests and CLI use.
        subprocess.run(command, check=True)
    else:
        os.execv(sys.executable, command)



if __name__ == "__main__":
    main()
