#!/usr/bin/env python3
from __future__ import annotations

from typing import Tuple
import numpy as np


def log_or_from_counts(n00: np.ndarray, n01: np.ndarray, n10: np.ndarray, n11: np.ndarray, pc: float) -> np.ndarray:
    a = n11 + pc
    b = n10 + pc
    c = n01 + pc
    d = n00 + pc
    return np.log((a * d) / (b * c))


def smooth_probs_from_counts(n00: np.ndarray, n01: np.ndarray, n10: np.ndarray, n11: np.ndarray, pc: float):
    c00 = n00 + pc
    c01 = n01 + pc
    c10 = n10 + pc
    c11 = n11 + pc
    tot = c00 + c01 + c10 + c11
    return c00 / tot, c01 / tot, c10 / tot, c11 / tot


def mutual_information_from_probs(p00, p01, p10, p11, base: str = "e"):
    p00 = np.asarray(p00, dtype=np.float64)
    p01 = np.asarray(p01, dtype=np.float64)
    p10 = np.asarray(p10, dtype=np.float64)
    p11 = np.asarray(p11, dtype=np.float64)

    # Ensure nonnegative (tiny negatives can happen from roundoff)
    p00 = np.clip(p00, 0.0, 1.0)
    p01 = np.clip(p01, 0.0, 1.0)
    p10 = np.clip(p10, 0.0, 1.0)
    p11 = np.clip(p11, 0.0, 1.0)

    # Marginals
    p0i = p00 + p01
    p1i = p10 + p11
    p0j = p00 + p10
    p1j = p01 + p11

    if base == "2":
        lg = np.log2
    elif base == "10":
        lg = np.log10
    else:
        lg = np.log

    # Avoid divide-by-zero and log(0); also use the fact that p*log(p/q) -> 0 when p==0
    eps = 1e-300
    d00 = np.maximum(p0i * p0j, eps)
    d01 = np.maximum(p0i * p1j, eps)
    d10 = np.maximum(p1i * p0j, eps)
    d11 = np.maximum(p1i * p1j, eps)

    with np.errstate(divide="ignore", invalid="ignore"):
        t00 = np.maximum(p00 / d00, eps)
        t01 = np.maximum(p01 / d01, eps)
        t10 = np.maximum(p10 / d10, eps)
        t11 = np.maximum(p11 / d11, eps)

        out = (
            np.where(p00 > 0, p00 * lg(t00), 0.0)
            + np.where(p01 > 0, p01 * lg(t01), 0.0)
            + np.where(p10 > 0, p10 * lg(t10), 0.0)
            + np.where(p11 > 0, p11 * lg(t11), 0.0)
        )

    return out


def mi_max_given_marginals(p1i: np.ndarray, p1j: np.ndarray, base: str) -> np.ndarray:
    p1i = np.asarray(p1i, dtype=np.float64)
    p1j = np.asarray(p1j, dtype=np.float64)

    L = np.maximum(0.0, p1i + p1j - 1.0)
    U = np.minimum(p1i, p1j)

    def mi_at(p11):
        p11 = np.asarray(p11, dtype=np.float64)
        p10 = p1i - p11
        p01 = p1j - p11
        p00 = 1.0 - p11 - p10 - p01
        return mutual_information_from_probs(p00, p01, p10, p11, base=base)

    return np.maximum(mi_at(L), mi_at(U))


def enforce_valid_joint(p1i: np.ndarray, p1j: np.ndarray, p11: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p1i = np.clip(np.asarray(p1i, dtype=np.float64), 0.0, 1.0)
    p1j = np.clip(np.asarray(p1j, dtype=np.float64), 0.0, 1.0)
    p11 = np.asarray(p11, dtype=np.float64)

    lo = np.maximum(0.0, p1i + p1j - 1.0)
    hi = np.minimum(p1i, p1j)
    p11 = np.clip(p11, lo, hi)

    p10 = p1i - p11
    p01 = p1j - p11
    p00 = 1.0 - p10 - p01 - p11

    p00 = np.clip(p00, 0.0, 1.0)
    p01 = np.clip(p01, 0.0, 1.0)
    p10 = np.clip(p10, 0.0, 1.0)
    p11 = np.clip(p11, 0.0, 1.0)
    return p00, p01, p10, p11


def compute_nulls_for_chunk(P: np.ndarray, ci: np.ndarray, cj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    P: (n_tips, n_cols) float32
    ci,cj: (B,) column indices into P
    """
    n_tips = P.shape[0]
    Xi = P[:, ci].astype(np.float64, copy=False)
    Xj = P[:, cj].astype(np.float64, copy=False)

    p1i = np.sum(Xi, axis=0, dtype=np.float64) / n_tips
    p1j = np.sum(Xj, axis=0, dtype=np.float64) / n_tips
    p11 = np.sum(Xi * Xj, axis=0, dtype=np.float64) / n_tips
    return enforce_valid_joint(p1i, p1j, p11)
