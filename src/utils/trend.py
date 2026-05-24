"""Trend score helpers for GRUT trend-aware inference."""
import math
import numpy as np


def log_max_trend_score(counts: np.ndarray) -> np.ndarray:
    """Log-max trend score: s_trend(i) = log(r_i / max_j r_j + 1).

    Args:
        counts: 1D array of item appearance counts r_i within the recent N-day window.
                Assumed nonnegative.
    Returns:
        1D array of trend scores, same shape as counts.
    """
    max_r = counts.max() if counts.size > 0 else 0
    if max_r <= 0:
        return np.zeros_like(counts, dtype=np.float64)
    return np.log(counts / max_r + 1.0)
