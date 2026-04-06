"""
utils.py — Utility Functions for Reproducibility and Logging

Handles:
  - Global random seed setting for numpy, TensorFlow, and Python random
  - Timer decorator for profiling pipeline stages
  - Console logging helpers
"""

import os
import time
import random
import functools
import numpy as np


def set_global_seeds(seed=42):
    """
    Set all random seeds for full reproducibility.

    Sets seeds for: Python random, NumPy, TensorFlow, and OS-level hash seed.
    Must be called BEFORE any TF or sklearn operations.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.get_logger().setLevel('ERROR')
        print(f"[SEED] All seeds set to {seed} (Python, NumPy, TensorFlow)")
    except ImportError:
        print(f"[SEED] Seeds set to {seed} (Python, NumPy only — TF not available)")


def timer(func):
    """Decorator that prints the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        print(f"[TIMER] {func.__name__} completed in {minutes}m {seconds:.1f}s")
        return result
    return wrapper


def print_banner(step_num, title):
    """Print a formatted step banner for pipeline progress."""
    print(f"\n{'=' * 60}")
    print(f"  STEP {step_num}: {title}")
    print(f"{'=' * 60}")


def print_class_distribution(y, label=""):
    """Print class distribution for a label array."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    ratio = n_neg / max(n_pos, 1)
    print(f"  {label}: {len(y)} total | Exoplanets: {n_pos} | Non-planets: {n_neg} | Ratio: 1:{ratio:.0f}")
