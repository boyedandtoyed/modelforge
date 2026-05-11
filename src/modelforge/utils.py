"""Shared utilities: logging, reproducibility, and formatting helpers."""
from __future__ import annotations

import logging
import os
import random


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a clean, leveled format. Uses LOG_LEVEL env var (default INFO)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    """Set Python and NumPy seeds; also sets PyTorch seed when available."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def fmt_loss(loss: float) -> str:
    """Format a loss value to 4 decimal places."""
    return f"{loss:.4f}"


def fmt_duration(seconds: float) -> str:
    """Convert seconds to a human-readable string like '1m 23s'."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.0f}s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s" if s else f"{m}m"


def human_size(num_params: int) -> str:
    """Format a parameter count as a human-readable string."""
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.1f} B"
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.0f} M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.0f} K"
    return str(num_params)
