"""Run inference with a trained checkpoint or fold ensemble on test, calibration, or validation cases.

Supports ``--role test --calibration-file .../calibration.json`` (scoring held-out
test cases, printing per-patient probabilities and decisions, and saving example,
failure, and calibration plots), ``--role calibration`` (fitting Platt scaling
and the referral threshold on calibration patients), and ``--role val``
(reproducing a single fold's development validation result).
Use ``python3 predict.py --help`` for command-line options.
"""

from engine.prediction import main


if __name__ == "__main__":
    raise SystemExit(main())
