"""
Shared utilities: config loading, logging, small helpers used
across the pipeline scripts.
"""

import logging
import yaml
from pathlib import Path

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path = None) -> dict:
    """Load config.yaml from project root (or a custom path)."""
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Resolve a path from config.yaml relative to the project root."""
    return PROJECT_ROOT / relative_path


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger