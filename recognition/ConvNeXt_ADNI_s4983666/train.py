"""Train a selected ADNI model on one development fold.

Use ``python3 train.py --help`` for command-line options.
"""

from engine.training import main


if __name__ == "__main__":
    raise SystemExit(main())
