#!/usr/bin/env python3
"""
stage6_refit_flagged_loci_sigma_grid.py  (Stage 6)

Stage 6 — Refit flagged loci with per-locus σ grid (likelihood-gated / LRT-boot)

Refactor notes:
- Supports --run-dir (preferred). Defaults:
    fasta:  RUN_DIR/work/stage0/fake.fasta (fallbacks tried)
    basis:  RUN_DIR/work/stage1/phylo_basis.npz (fallbacks tried)
    loci:   RUN_DIR/work/stage5/loci_flagged_refit.txt
    stage3: RUN_DIR/work/stage3/
    out:    RUN_DIR/work/stage6/
- Backward compatible with the older "manual API" flags if --run-dir is not given.
- Optional --minimal: only writes refit_patch.npz (no diagnostics, no P_refit even if requested).
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

def _resolve_stage3_basis(run_dir: Path, basis_arg: Optional[str]) -> Path:
    if basis_arg is not None:
        p = Path(basis_arg)
        return _must_exist(p if p.is_absolute() else (run_dir / p), "basis npz")

    cands = [
        run_dir / "work" / "stage3" / "phylo_basis.npz"
    ]
    return _resolve_first_existing(cands, "Stage3 phylo_basis.npz")

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

    # Try locus_fit.npz first (sigma_hat)
    lf = stage3_dir / "locus_fit.npz"
    if lf.exists():
        z = np.load(lf, allow_pickle=True)
        if "sigma_hat" in z.files:
            sg = float(z["sigma_hat"])
            if sg < 0.0:
                raise ValueError("sigma_hat in locus_fit.npz is <0 (unexpected)")
            return sg

    # Else use global_sigma.tsv
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
# Core search + bootstrap LRT gate
# -------------------------

def best_vs_global_for_y(
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
    Tuple[float, float, float, float, np.ndarray, bool],
]:
    """
    Returns:
      delta_raw = best_raw - global_raw
      global: (a_g, b_g, conv_g, score_g_raw)
      best:   (sigma_best, a_best, b_best, conv_best, score_best_raw, score_best_pen)
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

    # initialize best at global
    sc_best_raw = float(sc_g_raw)
    sc_best_pen = float(sc_g_raw) + sigma_lognormal_prior_penalty(float(sigma_global), float(sigma_global), tau_logsigma)
    best_sig = float(sigma_global)
    best_a = float(a_g)
    best_b = b_g.copy()
    best_conv = bool(conv_g)

    # warm start
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

def bootstrap_p_value_D(
    Z: np.ndarray,
    a_g: float,
    b_g: np.ndarray,
    sigma_global: float,
    grid: np.ndarray,
    tau_logsigma: float,
    max_iter: int,
    tol: float,
    D_obs: float,
    B: int,
    rng: np.random.Generator,
) -> float:
    p_g = sigmoid(float(a_g) + Z @ b_g)
    ge = 0
    n = p_g.shape[0]
    for _ in range(int(B)):
        yb = (rng.random(n) < p_g).astype(np.float64)
        delta_b, _, _ = best_vs_global_for_y(
            y=yb, Z=Z,
            sigma_global=float(sigma_global),
            grid=grid,
            tau_logsigma=tau_logsigma,
            max_iter=max_iter, tol=tol
        )
        D_b = 2.0 * float(delta_b)
        if D_b >= D_obs:
            ge += 1
    return float(1.0 + ge) / float(B + 1.0)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 6: refit flagged loci with per-locus sigma grid (run-dir native).")

    # Preferred
    ap.add_argument("--run-dir", default=None, help="Run directory (preferred). Uses work/stage* layout.")

    # Manual / overrides (still allowed)
    ap.add_argument("--fasta", default=None, help="Fake FASTA presence/absence (if not using run-dir defaults).")
    ap.add_argument("--basis", default=None, help="phylo_basis.npz (if not using run-dir defaults).")
    ap.add_argument("--loci", default=None, help="Flagged loci list file (if not using run-dir defaults).")
    ap.add_argument("--out", default=None, help="Output directory (default RUN_DIR/work/stage6).")

    ap.add_argument("--presence", default="a", help="Presence character in fake FASTA (default: a).")

    # sigma_global inputs
    ap.add_argument("--sigma-global", type=float, default=None, help="Global sigma from Stage 3 (optional).")
    ap.add_argument("--sigma-global-tsv", default=None, help="Path to Stage3 global_sigma.tsv (optional).")
    ap.add_argument("--stage3-dir", default=None, help="Stage3 output dir (default RUN_DIR/work/stage3).")

    # sigma grid
    ap.add_argument("--sigma-grid", default="0,0.25,0.5,1,2,4,8", help="Comma-separated sigma grid.")
    ap.add_argument("--sigma-grid-relative", action="store_true", help="Interpret sigma grid as multipliers of sigma_global.")

    # sigma selection shrinkage
    ap.add_argument("--tau-logsigma", type=float, default=0.0, help="If >0, apply lognormal prior penalty on sigma.")

    # gating
    ap.add_argument("--gate", choices=["bic", "lrt_boot", "none"], default="bic")
    ap.add_argument("--no-gate", action="store_true", help="DEPRECATED: same as --gate none.")
    ap.add_argument("--n-eff", type=int, default=None, help="Effective n for BIC tau (default n_tips).")
    ap.add_argument("--tau-mult", type=float, default=1.0)

    # LRT bootstrap
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--boot-B", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--boot-min-D", type=float, default=0.0)

    # cap
    ap.add_argument("--cap-mult", type=float, default=4.0)
    ap.add_argument("--reject-boundary", action="store_true")

    # optimizer
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--write-probs", action="store_true", help="Write P_refit.npz for accepted loci.")
    ap.add_argument("--progress-every", type=int, default=50)

    # extra acceptance constraints
    ap.add_argument("--min-deta-med", type=float, default=0.0)
    ap.add_argument("--sat-eps", type=float, default=1e-6)
    ap.add_argument("--sat-drop-min", type=float, default=0.0)
    ap.add_argument("--info-mult-min", type=float, default=1.0)

    # minimal output
    ap.add_argument("--minimal", action="store_true", help="Only write refit_patch.npz (no diagnostics/probs).")

    args = ap.parse_args()
    if args.no_gate:
        args.gate = "none"

    run_dir = Path(args.run_dir) if args.run_dir is not None else None
    if run_dir is not None:
        fasta_path = args.fasta
        basis_path = _resolve_stage3_basis(run_dir, args.basis)
        loci_path = _resolve_stage5_loci(run_dir, args.loci)
        stage3_dir = _resolve_stage3_dir(run_dir, args.stage3_dir)
        out_dir = _resolve_out_dir(run_dir, args.out)
        meta_path = run_dir / "meta" / "stage6.json"
        (run_dir / "meta").mkdir(parents=True, exist_ok=True)
    else:
        # manual mode: require key args
        if args.fasta is None or args.basis is None or args.loci is None:
            raise ValueError("Without --run-dir, you must provide --fasta, --basis, and --loci.")
        if args.out is None:
            raise ValueError("Without --run-dir, you must provide --out.")
        fasta_path = Path(args.fasta)
        basis_path = Path(args.basis)
        loci_path = Path(args.loci)
        stage3_dir = Path(args.stage3_dir) if args.stage3_dir is not None else Path(".")
        out_dir = Path(args.out)
        meta_path = out_dir / "stage6.json"

    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve sigma_global
    sigma_global = load_sigma_global(stage3_dir, args.sigma_global, args.sigma_global_tsv)
    sigma_global = float(sigma_global)
    if sigma_global < 0.0:
        raise ValueError("sigma_global must be >=0")

    # Load basis
    tips, Z, K = load_basis_npz(str(basis_path))
    n_tips = int(Z.shape[0])
    if len(tips) != n_tips:
        raise ValueError("basis tips length != Z rows")

    # Load loci list
    loci_list: List[int] = []
    if loci_path.exists():
        with open(loci_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    loci_list.append(int(s))
    loci_flagged = np.array(sorted(set(loci_list)), dtype=np.int64)

    # Parse sigma grid
    raw = np.array([float(x) for x in args.sigma_grid.split(",") if x.strip() != ""], dtype=np.float64)
    raw = raw[raw >= 0.0]
    if raw.size == 0:
        raise ValueError("sigma grid is empty")

    if args.sigma_grid_relative:
        grid = raw.copy()
        grid[grid > 0.0] = grid[grid > 0.0] * sigma_global
    else:
        grid = raw

    grid = np.unique(np.sort(np.append(grid, sigma_global)))

    # Apply cap
    used_cap = False
    cap_min = 0.0
    cap_max = float("inf")
    if float(args.cap_mult) > 0.0 and sigma_global > 0.0:
        used_cap = True
        cap_min = sigma_global / float(args.cap_mult)
        cap_max = sigma_global * float(args.cap_mult)
        grid = grid[(grid >= cap_min) & (grid <= cap_max)]
        if not np.any(np.isclose(grid, sigma_global, rtol=0, atol=0)):
            grid = np.unique(np.sort(np.append(grid, sigma_global)))

    if grid.size == 0:
        raise ValueError("sigma grid became empty after applying cap")

    # BIC gate tau
    n_eff = int(args.n_eff) if args.n_eff is not None else int(n_tips)
    tau0 = bic_tau(n_eff, tau_mult=float(args.tau_mult))

    # If no loci: write empty patch and stop
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
                    "deta_med\tsat_best\tsat_global\tinfo_best\tinfo_global\ttau_logsigma\n"
                )
        meta = {
            "stage": "stage6",
            "inputs": {
                "fasta": str(fasta_path),
                "basis": str(basis_path),
                "loci_flagged": str(loci_path),
                "stage3_dir": str(stage3_dir),
            },
            "params": {
                "sigma_global": sigma_global,
                "grid": grid.tolist(),
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

    # Read FASTA and pack bits
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
        f"[info] Flagged loci: {loci_flagged.size} | sigma_global={sigma_global:.6g} | grid_size={grid.size} | "
        f"K={K} | gate={args.gate} | tau_logsigma={args.tau_logsigma:.4g}",
        file=sys.stderr,
    )
    if used_cap:
        print(f"[info] Cap enabled: [{cap_min:.6g}, {cap_max:.6g}] (mult={args.cap_mult})", file=sys.stderr)

    # Diagnostics arrays (all flagged)
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

        delta_raw, (a_g, b_g, conv_g, sc_g_raw), (s_best, a_best, b_best, conv_best, sc_best_raw, sc_best_pen) = best_vs_global_for_y(
            y=y, Z=Z,
            sigma_global=float(sigma_global),
            grid=grid,
            tau_logsigma=float(args.tau_logsigma),
            max_iter=args.max_iter, tol=args.tol
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
                        grid=grid,
                        tau_logsigma=float(args.tau_logsigma),
                        max_iter=args.max_iter, tol=args.tol,
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

    # Run refits
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

    # Diagnostics TSV
    if not args.minimal:
        with open(diag_path, "w", encoding="utf-8") as f:
            f.write(
                "locus\taccepted\tgate\tsigma_best\tsigma_global\tscore_best_raw\tscore_global_raw\t"
                "delta_raw\tD_obs\tpD\ttau\tconv_best\tconv_global\tboundary_hit\tused_cap\t"
                "deta_med\tsat_best\tsat_global\tinfo_best\tinfo_global\ttau_logsigma\n"
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
                    f"{float(args.tau_logsigma):.10g}\n"
                )
        print(f"[done] Wrote diagnostics: {diag_path}", file=sys.stderr)

    # Patch NPZ (accepted only)
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

    # Optional P_refit for accepted loci
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

    # Meta
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