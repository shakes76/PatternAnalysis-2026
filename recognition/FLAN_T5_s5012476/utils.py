"""
Shared utilities for FLAN-T5 experiments.

Provides logging configuration, JSON serialization,
and output directory management.
"""

import json
import logging
from pathlib import Path

def setup_logging(verbose=False):
    """Configure logging for training and evaluation."""

    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(message)s",
        force=True,
    )

def ensure_directory(path):
    """Create a directory if it does not exist."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)

    return directory

def save_json(data, path):
    """Save experiment data to a JSON file."""

    file_path = Path(path)

    ensure_directory(file_path.parent)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    return file_path