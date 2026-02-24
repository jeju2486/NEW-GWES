from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: float) -> float:
    p = min(max(p, 1e-8), 1.0 - 1e-8)
    return math.log(p / (1.0 - p))


def neg_logpost_and_derivs(
    y: np.ndarray,
    Z: np.ndarray,
    alpha: float,
    b: np.ndarray,
    sigma: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Negative log posterior (up to additive const):
      f = -loglik(y|alpha,b) + 0.5/sigma^2 * ||b||^2
    Returns f, grad, Hessian for params [alpha, b...]
    """
    if sigma <= 0.0:
        # fit_locus_map() should never call this with sigma<=0 (it uses intercept-only path),
        # but guard anyway to avoid division-by-zero if used elsewhere.
        raise ValueError("neg_logpost_and_derivs requires sigma > 0. Use fit_locus_map intercept-only branch for sigma<=0.")

    eta = alpha + Z @ b
    p = sigmoid(eta)

    eps = 1e-12
    ll = np.sum(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
    f = -ll + 0.5 * (np.dot(b, b) / (sigma * sigma))

    r = (p - y)
    g_alpha = float(np.sum(r))
    g_b = Z.T @ r + (b / (sigma * sigma))
    g = np.concatenate(([g_alpha], g_b))

    w = p * (1.0 - p)
    s = float(np.sum(w))
    t = Z.T @ w
    Zw = Z * np.sqrt(w)[:, None]
    H_bb = (Zw.T @ Zw) + np.eye(Z.shape[1]) / (sigma * sigma)

    H = np.zeros((Z.shape[1] + 1, Z.shape[1] + 1), dtype=np.float64)
    H[0, 0] = s
    H[0, 1:] = t
    H[1:, 0] = t
    H[1:, 1:] = H_bb

    return float(f), g, H


def fit_locus_map(
    y: np.ndarray,
    Z: np.ndarray,
    sigma: float,
    max_iter: int = 50,
    tol: float = 1e-6,
    init_alpha: Optional[float] = None,
    init_b: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, bool, int, float, np.ndarray]:
    """
    Fit MAP for (alpha,b) with ridge prior on b.
    Returns: alpha, b, converged, iters, f_at_end, Hessian_at_end
    """
    n, K = Z.shape
    mean = float(np.mean(y))

    if sigma <= 0.0:
        a = logit(mean)
        b = np.zeros(K, dtype=np.float64)
        eta = a + Z @ b
        p = sigmoid(eta)
        w = p * (1.0 - p)
        H = np.eye(K + 1, dtype=np.float64) * 1e6
        H[0, 0] = max(float(np.sum(w)), 1e-9)
        eps = 1e-12
        ll = float(np.sum(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)))
        f = -ll
        return float(a), b, True, 0, float(f), H

    alpha = float(init_alpha) if init_alpha is not None else logit(mean)
    b = init_b.astype(np.float64, copy=True) if init_b is not None else np.zeros(K, dtype=np.float64)

    f_prev = None
    H = None
    for it in range(1, max_iter + 1):
        f, g, H = neg_logpost_and_derivs(y, Z, alpha, b, sigma)
        if float(np.linalg.norm(g, ord=2)) < tol:
            return float(alpha), b, True, it, float(f), H

        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.solve(H + np.eye(H.shape[0]) * 1e-6, g)

        t = 1.0
        gd = float(np.dot(g, step))
        for _ in range(20):
            a2 = alpha - t * step[0]
            b2 = b - t * step[1:]
            f2, _, _ = neg_logpost_and_derivs(y, Z, a2, b2, sigma)
            if f2 <= f - 1e-4 * t * gd:
                alpha, b, f = a2, b2, f2
                break
            t *= 0.5

        if f_prev is not None and abs(f_prev - f) < tol:
            return float(alpha), b, True, it, float(f), H
        f_prev = f

    return float(alpha), b, False, max_iter, float(f_prev if f_prev is not None else np.nan), H


def laplace_log_marginal(
    y: np.ndarray,
    Z: np.ndarray,
    alpha: float,
    b: np.ndarray,
    sigma: float,
    H: np.ndarray,
) -> float:
    """Laplace approx of log p(y | sigma), flat prior on alpha."""
    K = Z.shape[1]
    eta = alpha + Z @ b
    p = sigmoid(eta)
    eps = 1e-12
    loglik = float(np.sum(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)))

    if sigma <= 0.0:
        h00 = float(H[0, 0])
        logdet = math.log(max(h00, 1e-12))
        return loglik + 0.5 * math.log(2 * math.pi) - 0.5 * logdet

    logprior = (
        -0.5 * float(np.dot(b, b) / (sigma * sigma))
        -0.5 * K * math.log(2 * math.pi * sigma * sigma)
    )

    d = 1 + K
    try:
        R = np.linalg.cholesky(H)
        logdet = 2.0 * float(np.sum(np.log(np.diag(R) + 1e-30)))
    except np.linalg.LinAlgError:
        sign, ld = np.linalg.slogdet(H + np.eye(H.shape[0]) * 1e-9)
        logdet = float(ld) if sign > 0 else float(ld)

    return loglik + logprior + 0.5 * d * math.log(2 * math.pi) - 0.5 * logdet


# ---- Stage 6 compatibility ----
def laplace_log_marginal_raw(
    y: np.ndarray,
    Z: np.ndarray,
    alpha: float,
    b: np.ndarray,
    sigma: float,
    H: np.ndarray,
) -> float:
    """
    Stage 6 expects laplace_log_marginal_raw(). In this codebase it's identical
    to laplace_log_marginal().
    """
    return laplace_log_marginal(y=y, Z=Z, alpha=alpha, b=b, sigma=sigma, H=H)