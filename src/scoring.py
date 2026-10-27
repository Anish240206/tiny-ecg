"""Turn per-sample squared error into one anomaly score per window.
  mean: average over the whole window (dilutes a single bad beat)
  peak: highest moving average of PEAK_POOL samples (catches a local anomaly)"""
import numpy as np
from . import config as C


def scores_from_sq(sq, mode=None, pool=None):
    """sq: (n, W) or (W,) squared errors -> (n,) scores (or scalar for 1-D input)."""
    mode = mode or C.SCORE_MODE
    pool = pool or C.PEAK_POOL
    one = sq.ndim == 1
    sq = np.atleast_2d(sq)
    if mode == "mean":
        s = sq.mean(axis=1)
    elif mode == "peak":
        c = np.cumsum(np.pad(sq, ((0, 0), (1, 0))), axis=1)
        s = ((c[:, pool:] - c[:, :-pool]) / pool).max(axis=1)
    else:
        raise ValueError(mode)
    return float(s[0]) if one else s
