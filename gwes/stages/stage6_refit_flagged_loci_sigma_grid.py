#!/usr/bin/env python3
"""
stage6_refit_flagged_loci_sigma_grid.py  (Stage 6)

Stage 6 — Refit flagged loci with per-locus σ selection (grid OR continuous 1D optimise)

Update:
- Implements Option 1: bounded 1D optimisation over log(sigma) using Laplace evidence + optional lognormal prior penalty.
- Default: --sigma-method opt
- Backward compatible: --sigma-method grid reproduces old grid-search behaviour.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import minimize_scalar
except Exception:
    print("[error] Stage 6 (opt mode) requires scipy.optimize.", file=sys.stderr)
    raise

from gwes.manifest import write_stage_meta
from gwes.model_ecobias import sigmoid, logit, fit_locus_map, laplace_log_marginal


# -------------------------
# Run-dir path resolution
# -------------------------

def _must_exist(p: Path, what: str) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {p}")
    return p

def _resolve_first_existing(cands: List[Path], what: str) -> Path:
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{what} not found. Tried:\n  " + "\n  ".join(str(x) for x in cands)
    )

def _resolve_stage0_fasta(run_dir: Path, fasta_arg: Optional[str]) -> Path:
    if fasta_arg is not None:
        p = Path(fasta_arg)
        return _must_exist(p if p.is_absolute() else (run_dir / p), "fake FASTA")
    cands = [
        run_dir / "work" / "stage0" / "fake.fasta",
        run_dir / "work" / "stage0" / "fake.fa",
        run_dir / "work" / "stage0" / "matrix.fake.fasta",
    ]
    return _resolve_first_existing(cands, "Stage0 fake FASTA")

def _resolve_stage3_basis(run_dir: Path, basis_arg: Optional[str]) -> Path:
    if basis_arg is not None:
        p = Path(basis_arg)
        return _must_exist(p if p.is_absolute() else (run_dir / p), "basis npz")
    cands = [
        run_dir / "work" / "stage3" / "phylo_basis.npz",
        run_dir / "work" / "stage1" / "phylo_basis.npz",
    ]
    return _resolve_first_existing(cands, "phylo_basis.npz")

def _resolve_stage5_loci(run_dir: Path, loci_arg: Optional[str]) -> Path:
    if loci_arg is not None:
        p = Path(loci_arg)
        return _must_exist(p if p.is_absolute() else (run_dir / p), "flagged loci list")
    cands = [
        run_dir / "work" / "stage5" / "loci_flagged_refit.txt",
        run_dir / "work" / "stage5" / "loci_flagged.txt",
    ]
    return _resolve_first_existing(cands, "Stage5 flagged loci list")

def _resolve_stage3_dir(run_dir: Path, stage3_arg: Optional[str]) -> Path:
    if stage3_arg is not None:
        p = Path(stage3_arg)
        return _must_exist(p if p.is_absolute() else (run_dir / p), "Stage3 dir")
    return run_dir / "work" / "stage3"

def _resolve_out_dir(run_dir: Path, out_arg: Optional[str]) -> Path:
    if out_arg is not None:
        p = Path(out_arg)
        return p if p.is_absolute() else (run_dir / p)
    return run_dir / "work" / "stage6"


# -------------------------
# FASTA -> packed bits
# -------------------------

def read_fasta_dict(path: str) -> Dict[str, str]:
    seqs: Dict[str, List[str]] = {}
    name: Optional[str] = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                name = s[1:].split()[0]
                if name in seqs:
                    raise ValueError(f"Duplicate FASTA header: {name}")
                seqs[name] = []
            else:
                if name is None:
                    raise ValueError("FASTA format error: sequence before header.")
                seqs[name].append(s)
    return {k: "".join(v) for k, v in seqs.items()}

def pack_bits_from_sequences(
    tips_order: List[str],
    seqs: Dict[str, str],
    presence_char: str = "a",
) -> Tuple[np.ndarray, int]:
    n_tips = len(tips_order)
    if n_tips == 0:
        raise ValueError("No tips provided.")

    first_tip = tips_order[0]
    if first_tip not in seqs:
        raise KeyError(f"Tip '{first_tip}' not found in FASTA.")
    n_loci = len(seqs[first_tip])
    if n_loci == 0:
        raise ValueError("FASTA sequences appear empty.")

    n_bytes = (n_loci + 7) // 8
    Y_bits = np.zeros((n_tips, n_bytes), dtype=np.uint8)

    pres = ord(presence_char)
    for ti, tip in enumerate(tips_order):
        s = seqs.get(tip)
        if s is None:
            raise KeyError(f"Tip '{tip}' not found in FASTA.")
        if len(s) != n_loci:
            raise ValueError(f"Sequence length mismatch at tip '{tip}': {len(s)} vs {n_loci}")
        arr = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
        mask = (arr == pres)
        packed = np.packbits(mask, bitorder="little")
        if packed.size != n_bytes:
            packed = packed[:n_bytes]
        Y_bits[ti, :] = packed

    return Y_bits, n_loci

def get_locus_column(Y_bits: np.ndarray, locus: int) -> np.ndarray:
    by = locus >> 3
    bi = locus & 7
    return ((Y_bits[:, by] >> bi) & 1).astype(np.uint8)


# -------------------------
# Basis loading + sigma loading
# -------------------------

def load_basis_npz(path: str) -> Tuple[List[str], np.ndarray, int]:
    z = np.load(path, allow_pickle=True)
    if "tips" not in z.files:
        raise ValueError("basis npz must contain 'tips' (tip order).")
    tips = [str(x) for x in z["tips"].tolist()]

    if "Z" in z.files:
        Z = z["Z"]
        if Z.ndim != 2:
            raise ValueError("'Z' in basis must be 2D")
        K = int(z["K"]) if "K" in z.files else int(Z.shape[1])
        return tips, Z.astype(np.float64, copy=False), K

    n_tips = len(tips)
    for key in z.files:
        arr = z[key]
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] == n_tips:
            K = int(z["K"]) if "K" in z.files else int(arr.shape[1])
            return tips, arr.astype(np.float64, copy=False), K

    raise ValueError("Could not find a 2D basis matrix in basis npz (expected key 'Z' or any (n_tips,K) array).")

def load_sigma_from_global_sigma_tsv(path: str) -> float:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    rows: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            if parts[0].lower() == "sigma" and parts[1].lower().startswith("score"):
                continue
            rows.append((parts[0], parts[1]))

    for a, b in rows:
        if a.lower() == "chosen":
            return float(b)

    best_sig = None
    best_sc = -float("inf")
    for a, b in rows:
        try:
            sig = float(a)
            sc = float(b)
        except Exception:
            continue
        if sc > best_sc:
            best_sc = sc
            best_sig = sig

    if best_sig is None:
        raise ValueError(f"Could not parse sigma from: {path}")
    return float(best_sig)

def load_sigma_global(stage3_dir: Path, sigma_global_arg: Optional[float], sigma_global_tsv_arg: Optional[str]) -> float:
    if sigma_global_arg is not None:
        sg = float(sigma_global_arg)
        if sg < 0.0:
            raise ValueError("--sigma-global must be >=0")
        return sg

    lf = stage3_dir / "locus_fit.npz"
    if lf.exists():
        z = np.load(lf, allow_pickle=True)
        if "sigma_hat" in z.files:
            sg = float(z["sigma_hat"])
            if sg < 0.0:
                raise ValueError("sigma_hat in locus_fit.npz is <0 (unexpected)")
            return sg

    if sigma_global_tsv_arg is not None:
        return float(load_sigma_from_global_sigma_tsv(sigma_global_tsv_arg))

    gtsv = stage3_dir / "global_sigma.tsv"
    if gtsv.exists():
        return float(load_sigma_from_global_sigma_tsv(str(gtsv)))

    raise ValueError(
        "Could not resolve sigma_global. Provide --sigma-global, or ensure Stage3 wrote "
        "work/stage3/locus_fit.npz with sigma_hat or work/stage3/global_sigma.tsv."
    )


# -------------------------
# Diagnostics helpers
# -------------------------

def fit_diagnostics_from_params(y: np.ndarray, Z: np.ndarray, alpha: float, b: np.ndarray, sat_eps: float) -> Tuple[float, float]:
    eta = float(alpha) + Z @ b
    p = sigmoid(eta)
    sat = float(np.mean((p < sat_eps) | (p > 1.0 - sat_eps)))
    w = p * (1.0 - p)
    info = float(np.sum(w))
    return sat, info

def median_abs_logit_change(Z: np.ndarray, a_g: float, b_g: np.ndarray, a_b: float, b_b: np.ndarray) -> float:
    eta_g = float(a_g) + Z @ b_g
    eta_b = float(a_b) + Z @ b_b
    return float(np.median(np.abs(eta_b - eta_g)))

def sigma_lognormal_prior_penalty(sigma: float, sigma_global: float, tau_logsigma: float) -> float:
    if tau_logsigma <= 0.0:
        return 0.0
    if sigma <= 0.0 or sigma_global <= 0.0:
        return 0.0
    x = math.log(sigma / sigma_global)
    return -0.5 * (x * x) / (tau_logsigma * tau_logsigma)

def bic_tau(n_eff: int, tau_mult: float = 1.0) -> float:
    return float(tau_mult) * 0.5 * math.log(max(int(n_eff), 2))


# -------------------------
# Sigma selection (grid or opt)
# -------------------------

def _parse_sigma_grid(arg: str) -> np.ndarray:
    raw = np.array([float(x) for x in arg.split(",") if x.strip() != ""], dtype=np.float64)
    raw = raw[raw >= 0.0]
    if raw.size == 0:
        raise ValueError("sigma grid is empty")
    return np.unique(np.sort(raw))

def _make_sigma_candidates_and_bounds(
    raw_grid: np.ndarray,
    sigma_global: float,
    relative: bool,
    cap_mult: float,
) -> Tuple[np.ndarray, bool, float, float, float, float]:
    """
    Returns:
      grid_used (includes sigma_global),
      used_cap,
      cap_min, cap_max,
      sigma_lo, sigma_hi  (positive bounds for opt mode)
    """
    if relative:
        grid = raw_grid.copy()
        grid[grid > 0.0] = grid[grid > 0.0] * float(sigma_global)
    else:
        grid = raw_grid.copy()

    grid = np.unique(np.sort(np.append(grid, float(sigma_global))))

    used_cap = False
    cap_min = 0.0
    cap_max = float("inf")
    if float(cap_mult) > 0.0 and float(sigma_global) > 0.0:
        used_cap = True
        cap_min = float(sigma_global) / float(cap_mult)
        cap_max = float(sigma_global) * float(cap_mult)
        grid = grid[(grid >= cap_min) & (grid <= cap_max)]
        if not np.any(np.isclose(grid, float(sigma_global), rtol=0, atol=0)):
            grid = np.unique(np.sort(np.append(grid, float(sigma_global))))

    if grid.size == 0:
        raise ValueError("sigma grid became empty after applying cap")

    # bounds for opt mode from positive values in grid (or fallback)
    pos = grid[grid > 0.0]
    if pos.size == 0:
        # sigma_global may be 0; opt bounds must still be positive
        # fallback: use [1e-6, 1.0]
        sigma_lo = 1e-6
        sigma_hi = 1.0
    else:
        sigma_lo = max(1e-12, float(np.min(pos)) * 0.25)
        sigma_hi = float(np.max(pos)) * 2.0
        if sigma_hi <= sigma_lo:
            sigma_hi = sigma_lo * 10.0

    # clamp bounds to cap if enabled
    if used_cap:
        sigma_lo = max(sigma_lo, cap_min)
        sigma_hi = min(sigma_hi, cap_max)
        if sigma_hi <= sigma_lo:
            sigma_hi = min(cap_max, sigma_lo * 1.5)

    return grid, used_cap, cap_min, cap_max, sigma_lo, sigma_hi


def best_vs_global_for_y_grid(
    y: np.ndarray,
    Z: np.ndarray,
    sigma_global: float,
    grid: np.ndarray,
    tau_logsigma: float,
    max_iter: int,
    tol: float,
) -> Tuple[
    float,
    Tuple[float, np.ndarray, bool, float],
    Tuple[float, float, np.ndarray, bool, float, float],
]:
    """
    Old behaviour: scan grid.
    """
    n, K = Z.shape
    a0 = logit(float(np.mean(y)))
    b0 = np.zeros(K, dtype=np.float64)

    a_g, b_g, conv_g, _, _, H_g = fit_locus_map(
        y=y, Z=Z, sigma=float(sigma_global),
        init_alpha=a0, init_b=b0,
        max_iter=max_iter, tol=tol,
    )
    sc_g_raw = laplace_log_marginal(y, Z, a_g, b_g, float(sigma_global), H_g)

    sc_best_raw = float(sc_g_raw)
    sc_best_pen = float(sc_g_raw) + sigma_lognormal_prior_penalty(float(sigma_global), float(sigma_global), tau_logsigma)
    best_sig = float(sigma_global)
    best_a = float(a_g)
    best_b = b_g.copy()
    best_conv = bool(conv_g)

    a_ws = float(a_g)
    b_ws = b_g.astype(np.float64, copy=True)

    for sig in grid:
        sigf = float(sig)
        if np.isclose(sigf, float(sigma_global), rtol=0, atol=0):
            continue

        a, bb, conv, _, _, H = fit_locus_map(
            y=y, Z=Z, sigma=sigf,
            init_alpha=a_ws, init_b=b_ws,
            max_iter=max_iter, tol=tol,
        )
        sc_raw = laplace_log_marginal(y, Z, a, bb, sigf, H)
        sc_pen = float(sc_raw) + sigma_lognormal_prior_penalty(sigf, float(sigma_global), tau_logsigma)

        a_ws, b_ws = float(a), bb

        if sc_pen > sc_best_pen:
            sc_best_pen = float(sc_pen)
            sc_best_raw = float(sc_raw)
            best_sig = sigf
            best_a = float(a)
            best_b = bb.copy()
            best_conv = bool(conv)

    delta_raw = float(sc_best_raw - sc_g_raw)
    return (
        delta_raw,
        (float(a_g), b_g, bool(conv_g), float(sc_g_raw)),
        (best_sig, float(best_a), best_b, bool(best_conv), float(sc_best_raw), float(sc_best_pen)),
    )


def best_vs_global_for_y_opt(
    y: np.ndarray,
    Z: np.ndarray,
    sigma_global: float,
    sigma_lo: float,
    sigma_hi: float,
    tau_logsigma: float,
    max_iter: int,
    tol: float,
    xatol_log: float,
    maxiter: int,
    sigma_floor: float = 1e-12,
) -> Tuple[
    float,
    Tuple[float, np.ndarray, bool, float],
    Tuple[float, float, np.ndarray, bool, float, float],
]:
    """
    Option 1: bounded 1D optimise on u=log(sigma), sigma in [sigma_lo, sigma_hi].
    Returns same tuple structure as grid version.
    """
    n, K = Z.shape
    a0 = logit(float(np.mean(y)))
    b0 = np.zeros(K, dtype=np.float64)

    # global
    a_g, b_g, conv_g, _, _, H_g = fit_locus_map(
        y=y, Z=Z, sigma=float(sigma_global),
        init_alpha=a0, init_b=b0,
        max_iter=max_iter, tol=tol,
    )
    sc_g_raw = laplace_log_marginal(y, Z, a_g, b_g, float(sigma_global), H_g)

    # track best across evaluations
    best_sig = float(sigma_global)
    best_a = float(a_g)
    best_b = b_g.copy()
    best_conv = bool(conv_g)
    best_raw = float(sc_g_raw)
    best_pen = float(sc_g_raw) + sigma_lognormal_prior_penalty(float(sigma_global), float(sigma_global), tau_logsigma)

    # warm start state for objective calls
    ws_a = float(a_g)
    ws_b = b_g.astype(np.float64, copy=True)

    def eval_sigma(sig: float) -> float:
        nonlocal ws_a, ws_b, best_sig, best_a, best_b, best_conv, best_raw, best_pen
        s = float(max(sigma_floor, sig))
        a, bb, conv, _, _, H = fit_locus_map(
            y=y, Z=Z, sigma=s,
            init_alpha=ws_a, init_b=ws_b,
            max_iter=max_iter, tol=tol,
        )
        # update warm start regardless (helps next call)
        ws_a, ws_b = float(a), bb

        sc_raw = float(laplace_log_marginal(y, Z, a, bb, s, H))
        sc_pen = sc_raw + sigma_lognormal_prior_penalty(s, float(sigma_global), tau_logsigma)

        # treat non-convergence as very poor objective for optimisation,
        # but still record if it's the only thing we get.
        if not conv:
            sc_pen_eff = sc_pen - 1e6
        else:
            sc_pen_eff = sc_pen

        if sc_pen_eff > best_pen:
            best_pen = float(sc_pen_eff)
            best_raw = float(sc_raw)
            best_sig = float(s)
            best_a = float(a)
            best_b = bb.copy()
            best_conv = bool(conv)

        return sc_pen_eff

    # objective in log-space; minimise negative
    u_lo = float(np.log(max(sigma_floor, sigma_lo)))
    u_hi = float(np.log(max(sigma_floor, sigma_hi)))
    if not (u_hi > u_lo):
        u_hi = u_lo + 1.0

    def neg_obj(u: float) -> float:
        s = float(np.exp(u))
        return -eval_sigma(s)

    # Evaluate bounds explicitly (also improves robustness)
    eval_sigma(float(sigma_lo))
    eval_sigma(float(sigma_hi))

    # If bounds are degenerate, skip optimise
    if sigma_hi > sigma_lo * 1.0000001:
        _ = minimize_scalar(
            neg_obj,
            bounds=(u_lo, u_hi),
            method="bounded",
            options={"xatol": float(xatol_log), "maxiter": int(maxiter)},
        )
        # evaluate at optimizer's sigma too (it’s already evaluated many times, but safe)
        try:
            s_star = float(np.exp(_.x))
            eval_sigma(s_star)
        except Exception:
            pass

    delta_raw = float(best_raw - sc_g_raw)
    return (
        delta_raw,
        (float(a_g), b_g, bool(conv_g), float(sc_g_raw)),
        (float(best_sig), float(best_a), best_b, bool(best_conv), float(best_raw), float(best_pen)),
    )


# -------------------------
# Bootstrap gate
# -------------------------

def bootstrap_p_value_D(
    Z: np.ndarray,
    a_g: float,
    b_g: np.ndarray,
    sigma_global: float,
    sigma_method: str,
    grid: np.ndarray,
    sigma_lo: float,
    sigma_hi: float,
    tau_logsigma: float,
    max_iter: int,
    tol: float,
    xatol_log: float,
    sigma_maxiter: int,
    D_obs: float,
    B: int,
    rng: np.random.Generator,
) -> float:
    p_g = sigmoid(float(a_g) + Z @ b_g)
    ge = 0
    n = p_g.shape[0]
    for _ in range(int(B)):
        yb = (rng.random(n) < p_g).astype(np.float64)
        if sigma_method == "grid":
            delta_b, _, _ = best_vs_global_for_y_grid(
                y=yb, Z=Z,
                sigma_global=float(sigma_global),
                grid=grid,
                tau_logsigma=tau_logsigma,
                max_iter=max_iter, tol=tol
            )
        else:
            delta_b, _, _ = best_vs_global_for_y_opt(
                y=yb, Z=Z,
                sigma_global=float(sigma_global),
                sigma_lo=float(sigma_lo),
                sigma_hi=float(sigma_hi),
                tau_logsigma=tau_logsigma,
                max_iter=max_iter, tol=tol,
                xatol_log=float(xatol_log),
                maxiter=int(sigma_maxiter),
            )
        D_b = 2.0 * float(delta_b)
        if D_b >= D_obs:
            ge += 1
    return float(1.0 + ge) / float(B + 1.0)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 6: refit flagged loci with per-locus sigma (grid or opt).")

    ap.add_argument("--run-dir", default=None, help="Run directory (preferred). Uses work/stage* layout.")

    ap.add_argument("--fasta", default=None, help="Fake FASTA presence/absence (override).")
    ap.add_argument("--basis", default=None, help="phylo_basis.npz (override).")
    ap.add_argument("--loci", default=None, help="Flagged loci list file (override).")
    ap.add_argument("--out", default=None, help="Output directory (default RUN_DIR/work/stage6).")

    ap.add_argument("--presence", default="a", help="Presence character in fake FASTA (default: a).")

    ap.add_argument("--sigma-global", type=float, default=None, help="Global sigma from Stage 3 (optional).")
    ap.add_argument("--sigma-global-tsv", default=None, help="Path to Stage3 global_sigma.tsv (optional).")
    ap.add_argument("--stage3-dir", default=None, help="Stage3 output dir (default RUN_DIR/work/stage3).")

    ap.add_argument("--sigma-grid", default="0,0.25,0.5,1,2,4,8", help="Comma-separated sigma grid (grid mode) or bounds source (opt mode).")
    ap.add_argument("--sigma-grid-relative", action="store_true", help="Interpret sigma grid as multipliers of sigma_global.")

    ap.add_argument("--sigma-method", choices=["opt", "grid"], default="opt",
                    help="Per-locus sigma selection: opt (bounded 1D) or grid (scan). Default: opt.")
    ap.add_argument("--sigma-xatol-log", type=float, default=0.05, help="Opt tolerance in log(sigma) space (opt mode).")
    ap.add_argument("--sigma-maxiter", type=int, default=30, help="Opt max iterations per locus (opt mode).")

    ap.add_argument("--tau-logsigma", type=float, default=0.0, help="If >0, apply lognormal prior penalty on sigma.")

    ap.add_argument("--gate", choices=["bic", "lrt_boot", "none"], default="bic")
    ap.add_argument("--no-gate", action="store_true", help="DEPRECATED: same as --gate none.")
    ap.add_argument("--n-eff", type=int, default=None, help="Effective n for BIC tau (default n_tips).")
    ap.add_argument("--tau-mult", type=float, default=1.0)

    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--boot-B", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--boot-min-D", type=float, default=0.0)

    ap.add_argument("--cap-mult", type=float, default=4.0)
    ap.add_argument("--reject-boundary", action="store_true")

    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--write-probs", action="store_true", help="Write P_refit.npz for accepted loci.")
    ap.add_argument("--progress-every", type=int, default=50)

    ap.add_argument("--min-deta-med", type=float, default=0.0)
    ap.add_argument("--sat-eps", type=float, default=1e-6)
    ap.add_argument("--sat-drop-min", type=float, default=0.0)
    ap.add_argument("--info-mult-min", type=float, default=1.0)

    ap.add_argument("--minimal", action="store_true", help="Only write refit_patch.npz (no diagnostics/probs).")

    args = ap.parse_args()
    if args.no_gate:
        args.gate = "none"

    run_dir = Path(args.run_dir) if args.run_dir is not None else None
    if run_dir is not None:
        fasta_path = _resolve_stage0_fasta(run_dir, args.fasta)
        basis_path = _resolve_stage3_basis(run_dir, args.basis)
        loci_path = _resolve_stage5_loci(run_dir, args.loci)
        stage3_dir = _resolve_stage3_dir(run_dir, args.stage3_dir)
        out_dir = _resolve_out_dir(run_dir, args.out)
        meta_path = run_dir / "meta" / "stage6.json"
        (run_dir / "meta").mkdir(parents=True, exist_ok=True)
    else:
        if args.fasta is None or args.basis is None or args.loci is None or args.out is None:
            raise ValueError("Without --run-dir, you must provide --fasta, --basis, --loci, and --out.")
        fasta_path = Path(args.fasta)
        basis_path = Path(args.basis)
        loci_path = Path(args.loci)
        stage3_dir = Path(args.stage3_dir) if args.stage3_dir is not None else Path(".")
        out_dir = Path(args.out)
        meta_path = out_dir / "stage6.json"

    out_dir.mkdir(parents=True, exist_ok=True)

    sigma_global = float(load_sigma_global(stage3_dir, args.sigma_global, args.sigma_global_tsv))
    if sigma_global < 0.0:
        raise ValueError("sigma_global must be >=0")

    tips, Z, K = load_basis_npz(str(basis_path))
    n_tips = int(Z.shape[0])
    if len(tips) != n_tips:
        raise ValueError("basis tips length != Z rows")

    loci_list: List[int] = []
    if loci_path.exists():
        with open(loci_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    loci_list.append(int(s))
    loci_flagged = np.array(sorted(set(loci_list)), dtype=np.int64)

    raw_grid = _parse_sigma_grid(args.sigma_grid)
    grid, used_cap, cap_min, cap_max, sigma_lo, sigma_hi = _make_sigma_candidates_and_bounds(
        raw_grid=raw_grid,
        sigma_global=sigma_global,
        relative=bool(args.sigma_grid_relative),
        cap_mult=float(args.cap_mult),
    )

    n_eff = int(args.n_eff) if args.n_eff is not None else int(n_tips)
    tau0 = bic_tau(n_eff, tau_mult=float(args.tau_mult))

    patch_path = out_dir / "refit_patch.npz"
    diag_path = out_dir / "refit_diagnostics.tsv"

    if loci_flagged.size == 0:
        np.savez_compressed(
            patch_path,
            tips=np.array(tips, dtype=object),
            loci_refit=np.array([], dtype=np.int64),
            sigma_refit=np.array([], dtype=np.float64),
            alpha_refit=np.array([], dtype=np.float32),
            b_refit=np.zeros((0, K), dtype=np.float32),
            converged_refit=np.array([], dtype=np.bool_),
            score_refit=np.array([], dtype=np.float64),
            score_global=np.array([], dtype=np.float64),
            delta_score=np.array([], dtype=np.float64),
            tau=np.array([], dtype=np.float64),
            sigma_global=np.float64(sigma_global),
            sigma_grid_used=grid.astype(np.float64),
            sigma_bounds=np.array([sigma_lo, sigma_hi], dtype=np.float64),
            sigma_method=np.array([args.sigma_method], dtype=object),
            K=np.int32(K),
            tau_logsigma=np.float64(args.tau_logsigma),
            gate=np.array([args.gate], dtype=object),
            alpha_gate=np.float64(args.alpha),
            boot_B=np.int32(args.boot_B),
            min_deta_med=np.float64(args.min_deta_med),
            sat_eps=np.float64(args.sat_eps),
            sat_drop_min=np.float64(args.sat_drop_min),
            info_mult_min=np.float64(args.info_mult_min),
        )
        if not args.minimal:
            with open(diag_path, "w", encoding="utf-8") as f:
                f.write(
                    "locus\taccepted\tgate\tsigma_best\tsigma_global\tscore_best_raw\tscore_global_raw\t"
                    "delta_raw\tD_obs\tpD\ttau\tconv_best\tconv_global\tboundary_hit\tused_cap\t"
                    "deta_med\tsat_best\tsat_global\tinfo_best\tinfo_global\ttau_logsigma\tsigma_method\n"
                )
        meta = {
            "stage": "stage6",
            "inputs": {"fasta": str(fasta_path), "basis": str(basis_path), "loci_flagged": str(loci_path), "stage3_dir": str(stage3_dir)},
            "params": {
                "sigma_global": sigma_global,
                "sigma_method": args.sigma_method,
                "grid": grid.tolist(),
                "sigma_bounds": [sigma_lo, sigma_hi],
                "gate": args.gate,
                "tau0": tau0 if args.gate == "bic" else None,
                "tau_logsigma": float(args.tau_logsigma),
                "cap_mult": float(args.cap_mult),
                "reject_boundary": bool(args.reject_boundary),
                "threads": int(args.threads),
                "write_probs": bool(args.write_probs) and (not args.minimal),
                "minimal": bool(args.minimal),
            },
            "outputs": {"refit_patch": str(patch_path), "diagnostics": str(diag_path) if not args.minimal else None},
        }
        write_stage_meta(meta_path, meta)
        print(f"[info] No loci to refit. Wrote empty patch: {patch_path}", file=sys.stderr)
        return

    seqs = read_fasta_dict(str(fasta_path))
    t_pack0 = time.perf_counter()
    Y_bits, n_loci_total = pack_bits_from_sequences(tips, seqs, presence_char=args.presence)
    t_pack1 = time.perf_counter()
    if int(loci_flagged.max()) >= n_loci_total:
        raise ValueError("Flagged loci contain index >= FASTA sequence length")

    def is_boundary(sig: float) -> bool:
        if not used_cap:
            return False
        tolb = 1e-12 + 1e-6 * max(1.0, abs(sig))
        return (abs(sig - cap_min) <= tolb) or (abs(sig - cap_max) <= tolb)

    print(
        f"[info] Packed bits: tips={n_tips}, loci_total={n_loci_total}, bytes={Y_bits.shape[1]}, time={t_pack1-t_pack0:.2f}s",
        file=sys.stderr,
    )
    print(
        f"[info] Flagged loci: {loci_flagged.size} | sigma_global={sigma_global:.6g} | "
        f"method={args.sigma_method} | bounds=[{sigma_lo:.3g},{sigma_hi:.3g}] | grid_size={grid.size} | "
        f"K={K} | gate={args.gate} | tau_logsigma={args.tau_logsigma:.4g}",
        file=sys.stderr,
    )
    if used_cap:
        print(f"[info] Cap enabled: [{cap_min:.6g}, {cap_max:.6g}] (mult={args.cap_mult})", file=sys.stderr)

    m = loci_flagged.size
    sigma_best_all = np.zeros(m, dtype=np.float64)
    score_best_raw_all = np.full(m, -np.inf, dtype=np.float64)
    score_best_pen_all = np.full(m, -np.inf, dtype=np.float64)
    conv_best_all = np.zeros(m, dtype=np.bool_)

    score_global_raw_all = np.full(m, -np.inf, dtype=np.float64)
    conv_global_all = np.zeros(m, dtype=np.bool_)

    delta_raw_all = np.full(m, np.nan, dtype=np.float64)
    D_obs_all = np.full(m, np.nan, dtype=np.float64)
    pD_all = np.full(m, np.nan, dtype=np.float64)

    accepted_all = np.zeros(m, dtype=np.bool_)
    boundary_hit_all = np.zeros(m, dtype=np.bool_)

    alpha_best_all = np.zeros(m, dtype=np.float32)
    b_best_all = np.zeros((m, K), dtype=np.float32)

    deta_med_all = np.full(m, np.nan, dtype=np.float64)
    sat_best_all = np.full(m, np.nan, dtype=np.float64)
    sat_global_all = np.full(m, np.nan, dtype=np.float64)
    info_best_all = np.full(m, np.nan, dtype=np.float64)
    info_global_all = np.full(m, np.nan, dtype=np.float64)

    probs_best: Dict[int, np.ndarray] = {}

    def _rng_for_locus(loc: int) -> np.random.Generator:
        seed = (int(args.seed) ^ (loc * 1000003)) & 0xFFFFFFFF
        return np.random.default_rng(seed)

    def refit_one(idx: int):
        loc = int(loci_flagged[idx])
        y = get_locus_column(Y_bits, loc).astype(np.float64)

        if args.sigma_method == "grid":
            delta_raw, (a_g, b_g, conv_g, sc_g_raw), (s_best, a_best, b_best, conv_best, sc_best_raw, sc_best_pen) = best_vs_global_for_y_grid(
                y=y, Z=Z,
                sigma_global=float(sigma_global),
                grid=grid,
                tau_logsigma=float(args.tau_logsigma),
                max_iter=args.max_iter, tol=args.tol
            )
        else:
            delta_raw, (a_g, b_g, conv_g, sc_g_raw), (s_best, a_best, b_best, conv_best, sc_best_raw, sc_best_pen) = best_vs_global_for_y_opt(
                y=y, Z=Z,
                sigma_global=float(sigma_global),
                sigma_lo=float(sigma_lo),
                sigma_hi=float(sigma_hi),
                tau_logsigma=float(args.tau_logsigma),
                max_iter=args.max_iter, tol=args.tol,
                xatol_log=float(args.sigma_xatol_log),
                maxiter=int(args.sigma_maxiter),
            )

        D_obs = 2.0 * float(delta_raw)
        boundary_hit = is_boundary(float(s_best))

        sat_g, info_g = fit_diagnostics_from_params(y, Z, a_g, b_g, float(args.sat_eps))
        sat_b, info_b = fit_diagnostics_from_params(y, Z, a_best, b_best, float(args.sat_eps))
        deta_med = median_abs_logit_change(Z, a_g, b_g, a_best, b_best)

        accepted = False
        pD = float("nan")

        if np.isclose(float(s_best), float(sigma_global), rtol=0, atol=0):
            accepted = False
        else:
            if args.gate == "none":
                accepted = True
            elif args.gate == "bic":
                if (conv_best and conv_g):
                    accepted = (float(delta_raw) >= float(tau0))
                elif conv_best and (not conv_g):
                    accepted = True
                else:
                    accepted = False
            elif args.gate == "lrt_boot":
                if (conv_best and conv_g) and (float(D_obs) >= float(args.boot_min_D)) and int(args.boot_B) > 0:
                    rng = _rng_for_locus(loc)
                    pD = bootstrap_p_value_D(
                        Z=Z,
                        a_g=float(a_g), b_g=b_g,
                        sigma_global=float(sigma_global),
                        sigma_method=str(args.sigma_method),
                        grid=grid,
                        sigma_lo=float(sigma_lo),
                        sigma_hi=float(sigma_hi),
                        tau_logsigma=float(args.tau_logsigma),
                        max_iter=args.max_iter, tol=args.tol,
                        xatol_log=float(args.sigma_xatol_log),
                        sigma_maxiter=int(args.sigma_maxiter),
                        D_obs=float(D_obs),
                        B=int(args.boot_B),
                        rng=rng,
                    )
                    accepted = (float(pD) <= float(args.alpha))
                else:
                    accepted = False

        if args.reject_boundary and boundary_hit:
            accepted = False

        if float(args.min_deta_med) > 0.0:
            accepted = accepted and (float(deta_med) >= float(args.min_deta_med))
        if float(args.sat_drop_min) > 0.0:
            accepted = accepted and (float(sat_b) <= float(sat_g) - float(args.sat_drop_min))
        if float(args.info_mult_min) > 1.0:
            accepted = accepted and (float(info_b) >= float(info_g) * float(args.info_mult_min))

        return (
            idx, loc,
            float(s_best), float(a_best), b_best, bool(conv_best),
            float(sc_best_raw), float(sc_best_pen),
            bool(conv_g), float(sc_g_raw),
            float(delta_raw), float(D_obs), float(pD),
            bool(accepted), bool(boundary_hit),
            float(deta_med), float(sat_b), float(sat_g), float(info_b), float(info_g),
            float(a_g), b_g,
        )

    t0 = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.threads)) as ex:
        futs = [ex.submit(refit_one, i) for i in range(m)]
        for fut in as_completed(futs):
            (
                idx, loc,
                s_best, a_best, b_best, conv_best,
                sc_best_raw, sc_best_pen,
                conv_g, sc_g_raw,
                delta_raw, D_obs, pD,
                accepted, boundary_hit,
                deta_med, sat_b, sat_g, info_b, info_g,
                a_g, b_g,
            ) = fut.result()

            sigma_best_all[idx] = float(s_best)
            alpha_best_all[idx] = np.float32(a_best)
            b_best_all[idx, :] = b_best.astype(np.float32, copy=False)
            conv_best_all[idx] = bool(conv_best)
            score_best_raw_all[idx] = float(sc_best_raw)
            score_best_pen_all[idx] = float(sc_best_pen)

            score_global_raw_all[idx] = float(sc_g_raw)
            conv_global_all[idx] = bool(conv_g)

            delta_raw_all[idx] = float(delta_raw)
            D_obs_all[idx] = float(D_obs)
            pD_all[idx] = float(pD) if not (pD != pD) else np.nan

            boundary_hit_all[idx] = bool(boundary_hit)
            accepted_all[idx] = bool(accepted)

            deta_med_all[idx] = float(deta_med)
            sat_best_all[idx] = float(sat_b)
            sat_global_all[idx] = float(sat_g)
            info_best_all[idx] = float(info_b)
            info_global_all[idx] = float(info_g)

            if (not args.minimal) and args.write_probs and accepted:
                probs_best[idx] = sigmoid(float(a_best) + Z @ b_best).astype(np.float32)

            done += 1
            if done % max(1, args.progress_every) == 0 or done == m:
                dt = time.perf_counter() - t0
                rate = done / max(dt, 1e-9)
                print(f"[info] processed {done}/{m}  rate={rate:.2f} loci/s", file=sys.stderr)

    if not args.minimal:
        with open(diag_path, "w", encoding="utf-8") as f:
            f.write(
                "locus\taccepted\tgate\tsigma_best\tsigma_global\tscore_best_raw\tscore_global_raw\t"
                "delta_raw\tD_obs\tpD\ttau\tconv_best\tconv_global\tboundary_hit\tused_cap\t"
                "deta_med\tsat_best\tsat_global\tinfo_best\tinfo_global\ttau_logsigma\tsigma_method\n"
            )
            for i in range(m):
                tau_out = tau0 if args.gate == "bic" else float("nan")
                f.write(
                    f"{int(loci_flagged[i])}\t{int(accepted_all[i])}\t{args.gate}\t"
                    f"{sigma_best_all[i]:.10g}\t{sigma_global:.10g}\t"
                    f"{score_best_raw_all[i]:.10g}\t{score_global_raw_all[i]:.10g}\t"
                    f"{delta_raw_all[i]:.10g}\t{D_obs_all[i]:.10g}\t{pD_all[i]:.10g}\t{tau_out:.10g}\t"
                    f"{int(conv_best_all[i])}\t{int(conv_global_all[i])}\t{int(boundary_hit_all[i])}\t{int(used_cap)}\t"
                    f"{deta_med_all[i]:.10g}\t{sat_best_all[i]:.10g}\t{sat_global_all[i]:.10g}\t"
                    f"{info_best_all[i]:.10g}\t{info_global_all[i]:.10g}\t"
                    f"{float(args.tau_logsigma):.10g}\t{args.sigma_method}\n"
                )
        print(f"[done] Wrote diagnostics: {diag_path}", file=sys.stderr)

    acc_idx = np.where(accepted_all)[0]
    loci_refit = loci_flagged[acc_idx].astype(np.int64, copy=True)

    np.savez_compressed(
        patch_path,
        tips=np.array(tips, dtype=object),
        loci_refit=loci_refit,
        sigma_refit=sigma_best_all[acc_idx].astype(np.float64, copy=True),
        alpha_refit=alpha_best_all[acc_idx].astype(np.float32, copy=True),
        b_refit=b_best_all[acc_idx, :].astype(np.float32, copy=True),
        converged_refit=conv_best_all[acc_idx].astype(np.bool_, copy=True),
        score_refit=score_best_raw_all[acc_idx].astype(np.float64, copy=True),
        score_global=score_global_raw_all[acc_idx].astype(np.float64, copy=True),
        delta_score=delta_raw_all[acc_idx].astype(np.float64, copy=True),
        tau=np.full(acc_idx.size, float(tau0), dtype=np.float64) if args.gate == "bic" else np.full(acc_idx.size, np.nan, dtype=np.float64),
        sigma_global=np.float64(sigma_global),
        sigma_grid_used=grid.astype(np.float64),
        sigma_bounds=np.array([sigma_lo, sigma_hi], dtype=np.float64),
        sigma_method=np.array([args.sigma_method], dtype=object),
        K=np.int32(K),
        tau_logsigma=np.float64(args.tau_logsigma),
        gate=np.array([args.gate], dtype=object),
        alpha_gate=np.float64(args.alpha),
        boot_B=np.int32(args.boot_B),
        min_deta_med=np.float64(args.min_deta_med),
        sat_eps=np.float64(args.sat_eps),
        sat_drop_min=np.float64(args.sat_drop_min),
        info_mult_min=np.float64(args.info_mult_min),
    )
    print(f"[done] Wrote patch: {patch_path} (accepted {loci_refit.size}/{m})", file=sys.stderr)

    if (not args.minimal) and args.write_probs:
        p_path = out_dir / "P_refit.npz"
        if loci_refit.size == 0:
            np.savez_compressed(
                p_path,
                tips=np.array(tips, dtype=object),
                loci=np.array([], dtype=np.int64),
                P=np.zeros((n_tips, 0), dtype=np.float32),
            )
        else:
            P_ref = np.zeros((n_tips, loci_refit.size), dtype=np.float32)
            for j, original_idx in enumerate(acc_idx.tolist()):
                if original_idx in probs_best:
                    P_ref[:, j] = probs_best[original_idx]
                else:
                    P_ref[:, j] = sigmoid(float(alpha_best_all[original_idx]) + Z @ b_best_all[original_idx, :].astype(np.float64)).astype(np.float32)
            np.savez_compressed(
                p_path,
                tips=np.array(tips, dtype=object),
                loci=loci_refit,
                P=P_ref,
            )
        print(f"[done] Wrote probabilities: {p_path}", file=sys.stderr)

    meta = {
        "stage": "stage6",
        "inputs": {
            "run_dir": str(run_dir) if run_dir is not None else None,
            "fasta": str(fasta_path),
            "basis": str(basis_path),
            "loci_flagged": str(loci_path),
            "stage3_dir": str(stage3_dir),
        },
        "params": {
            "sigma_global": sigma_global,
            "sigma_method": args.sigma_method,
            "sigma_bounds": [sigma_lo, sigma_hi],
            "sigma_xatol_log": float(args.sigma_xatol_log),
            "sigma_maxiter": int(args.sigma_maxiter),
            "sigma_grid_used": grid.tolist(),
            "sigma_grid_relative": bool(args.sigma_grid_relative),
            "tau_logsigma": float(args.tau_logsigma),
            "gate": args.gate,
            "tau0": float(tau0) if args.gate == "bic" else None,
            "alpha_gate": float(args.alpha),
            "boot_B": int(args.boot_B),
            "cap_mult": float(args.cap_mult),
            "reject_boundary": bool(args.reject_boundary),
            "threads": int(args.threads),
            "presence": args.presence,
            "minimal": bool(args.minimal),
            "write_probs": bool(args.write_probs) and (not args.minimal),
        },
        "outputs": {
            "refit_patch": str(patch_path),
            "diagnostics": str(diag_path) if not args.minimal else None,
            "P_refit": str(out_dir / "P_refit.npz") if ((not args.minimal) and args.write_probs) else None,
        },
    }
    write_stage_meta(meta_path, meta)


if __name__ == "__main__":
    main()