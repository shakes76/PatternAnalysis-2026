"""Audit ADNI sources or prepare/verify frozen patient-level splits.

Use ``python3 adni_splits.py --help`` for command-line options.
"""

from adni_data.splits import main


if __name__ == "__main__":
    raise SystemExit(main())
