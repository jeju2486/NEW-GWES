#!/usr/bin/env python3
from __future__ import annotations
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def minor_count_from_mean(mean: np.ndarray, n: int) -> np.ndarray:
    """
    mean should be the empirical allele frequency (from counts).
    Uses rounding; better long-term is to store counts in locus_fit.npz.
    """
    cnt1 = np.rint(mean * n).astype(np.int64)
    cnt0 = n - cnt1
    return np.minimum(cnt1, cnt0)
