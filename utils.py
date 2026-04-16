"""
utils.py — Utility Functions for Reproducibility, Timing, and Logging

Provides:
  - set_global_seeds: Sets seeds across Python, NumPy, and TensorFlow for reproducibility
  - timer: Decorator that measures and prints execution time
  - print_banner: Prints a formatted step banner for pipeline progress
"""

import os
import time
import random
import functools
import numpy as np


def set_global_seeds(seed=42):
    """
    Set random seeds for reproducibility across all libraries.

    Sets seeds for:
      - Python's random module
      - NumPy
      - TensorFlow (including PYTHONHASHSEED for graph-level determinism)

    Parameters
    ----------
    seed : int
        The seed value (default: 42)
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        print(f"[INFO] Global seeds set to {seed} (Python, NumPy, TensorFlow)")
    except ImportError:
        print(f"[INFO] Global seeds set to {seed} (Python, NumPy — TensorFlow not available)")


def timer(func):
    """
    Decorator that measures and prints the execution time of a function.

    Usage:
        @timer
        def my_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start

        # Format elapsed time nicely
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        elif elapsed < 3600:
            mins = int(elapsed // 60)
            secs = elapsed % 60
            time_str = f"{mins}m {secs:.1f}s"
        else:
            hrs = int(elapsed // 3600)
            mins = int((elapsed % 3600) // 60)
            time_str = f"{hrs}h {mins}m"

        print(f"\n[TIMER] {func.__name__} completed in {time_str}")
        return result
    return wrapper


def print_banner(step, title):
    """
    Print a formatted banner for pipeline progress tracking.

    Parameters
    ----------
    step : int or str
        Step number or identifier
    title : str
        Description of the step
    """
    print()
    print("=" * 60)
    print(f"  STEP {step}: {title.upper()}")
    print("=" * 60)
