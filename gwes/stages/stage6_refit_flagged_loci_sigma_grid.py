#!/usr/bin/env python3
from __future__ import annotations

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

from fasta_matrix import read_fasta_dict, pack_bits_from_sequences
from bits import get_locus_column_bits
from phylo import load_basis_npz
from prob_store import load_sigma_from_global_sigma_tsv

# You should already have these in model_ecobias.py from Stage 3.
# If names differ, keep wrappers in model_ecobias.py rather than re-defining here.
from model_ecobias import sigmoid, logit, fit_locus_map, laplace_log_marginal_raw


def sigma_lognormal_prior_penalty(sigma: float, sigma_global: float, tau_logsigma: float) -> float:
    if tau_logsigma <= 0.0 or sigma <= 0.0 or sigma_global <= 0.0:
        return 0.0
    x = math.log(sigma / sigma_global)
    return -0.5 * (x * x) / (tau_logsigma * tau_logsigma)


def fit_diagnostics_from_params(y: np.ndarray, Z: np.ndarray, alpha: float, b: np.ndarray, sat_eps: float) -> Tuple[float, float]:
    eta = float(alpha) + Z @ b
    p = sigmoid(eta)
    sat = float(np.mean((p < sat_eps) | (p > 1.0 - sat_eps)))
    info = float(np.sum(p * (1.0 - p)))
    return sat, info


def median_abs_logit_change(Z: np.ndarray, a_g: float, b_g: np.ndarray, a_b: float, b_b: np.ndarray) -> float:
    eta_g = float(a_g) + Z @ b_g
    eta_b = float(a_b) + Z @ b_b
    return float(np.median(np.abs(eta_b - eta_g)))


def load_loci_list(path: str) -> np.ndarray:
    if not os.path.exists(path):
        return np.array([], dtype=np.int64)
    loci: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                loci.append(int(s))
    return np.array(sorted(set(loci)), dtype=np.int64)


def bic_tau(n_eff: int, tau_mult: float = 1.0) -> float:
    return float(tau_mult) * 0.5 * math.log(max(int(n_eff), 2))


def best_vs_global_for_y(
    y: np.ndarray,
    Z: np.ndarray,
    sigma_global: float,
    grid: np.ndarray,
    tau_logsigma: float,
    max_iter: int,
    tol: float,
) -> Tuple[
    float,  # delta_raw
    Tuple[float, np.ndarray, bool, float],  # global (alpha, b, conv, score_raw)
    Tuple[float, float, float, float, np.ndarray, bool],  # best (sigma_best, a_best, score_raw, score_pen, b_best, conv)
]:
    n, K = Z.shape
    a0 = logit(float(np.mean(y)))
    b0 = np.zeros(K, dtype=np.float64)

    a_g, b_g, conv_g, _, _, H_g = fit_locus_map(
        y=y, Z=Z, sigma=float(sigma_global),
        init_alpha=a0, init_b=b0,
        max_iter=max_iter, tol=tol,
    )
    sc_g_raw = laplace_log_marginal_raw(y, Z, a_g, b_g, float(sigma_global), H_g)

    best_sig = float(sigma_global)
    best_a = float(a_g)
    best_b = b_g.copy()
    best_conv = bool(conv_g)
    best_raw = float(sc_g_raw)
    best_pen = float(sc_g_raw) + sigma_lognormal_prior_penalty(best_sig, float(sigma_global), tau_logsigma)

    a_ws = float(a_g)
    b_ws = b_g.astype(np.float64, copy=True)

    for sig in grid:
        sigf = float(sig)
        if sigf == float(sigma_global):
            continue

        a, bb, conv, _, _, H = fit_locus_map(
            y=y, Z=Z, sigma=sigf,
            init_alpha=a_ws, init_b=b_ws,
            max_iter=max_iter, tol=tol,
        )
        sc_raw = laplace_log_marginal_raw(y, Z, a, bb, sigf, H)
        sc_pen = float(sc_raw) + sigma_lognormal_prior_penalty(sigf, float(sigma_global), tau_logsigma)

        a_ws, b_ws = float(a), bb

        if sc_pen > best_pen:
            best_pen = float(sc_pen)
            best_raw = float(sc_raw)
            best_sig = sigf
            best_a = float(a)
            best_b = bb.copy()
            best_conv = bool(conv)

    delta_raw = float(best_raw - sc_g_raw)
    return (
        delta_raw,
        (float(a_g), b_g, bool(conv_g), float(sc_g_raw)),
        (float(best_sig), float(best_a), float(best_raw), float(best_pen), best_b, bool(best_conv)),
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
        if 2.0 * float(delta_b) >= D_obs:
            ge += 1
    return float(1.0 + ge) / float(B + 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--basis", required=True)
    ap.add_argument("--loci", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--presence", default="a")

    ap.add_argument("--sigma-global", type=float, default=None)
    ap.add_argument("--sigma-global-tsv", default=None)
    ap.add_argument("--stage3-dir", default=None)

    ap.add_argument("--sigma-grid", default="0,0.25,0.5,1,2,4,8")
    ap.add_argument("--sigma-grid-relative", action="store_true")
    ap.add_argument("--tau-logsigma", type=float, default=0.0)

    ap.add_argument("--gate", choices=["bic", "lrt_boot", "none"], default="bic")
    ap.add_argument("--n-eff", type=int, default=None)
    ap.add_argument("--tau-mult", type=float, default=1.0)

    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--boot-B", type=int, default=200)
    ap.add_argument("--boot-min-D", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--cap-mult", type=float, default=4.0)
    ap.add_argument("--reject-boundary", action="store_true")

    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--write-probs", action="store_true")
    ap.add_argument("--progress-every", type=int, default=50)

    ap.add_argument("--min-deta-med", type=float, default=0.0)
    ap.add_argument("--sat-eps", type=float, default=1e-6)
    ap.add_argument("--sat-drop-min", type=float, default=0.0)
    ap.add_argument("--info-mult-min", type=float, default=1.0)

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # sigma_global resolution
    sigma_global: Optional[float] = args.sigma_global
    if sigma_global is None:
        sg_tsv = args.sigma_global_tsv
        if sg_tsv is None and args.stage3_dir is not None:
            cand = os.path.join(args.stage3_dir, "global_sigma.tsv")
            sg_tsv = cand if os.path.exists(cand) else None
        if sg_tsv is None:
            raise ValueError("Provide --sigma-global, --sigma-global-tsv, or --stage3-dir (with global_sigma.tsv).")
        sigma_global = load_sigma_from_global_sigma_tsv(sg_tsv)
    sigma_global = float(sigma_global)
    if sigma_global < 0.0:
        raise ValueError("--sigma-global must be >= 0")

    tips, Z, K = load_basis_npz(args.basis, dtype=np.float64)
    n_tips = Z.shape[0]

    loci_flagged = load_loci_list(args.loci)

    # parse grid
    raw = np.array([float(x) for x in args.sigma_grid.split(",") if x.strip() != ""], dtype=np.float64)
    raw = raw[raw >= 0.0]
    if raw.size == 0:
        raise ValueError("sigma grid is empty")
    if args.sigma_grid_relative:
        grid = raw.copy()
        grid[grid > 0.0] *= sigma_global
    else:
        grid = raw
    grid = np.unique(np.sort(np.append(grid, sigma_global)))

    used_cap = False
    cap_min = 0.0
    cap_max = float("inf")
    if float(args.cap_mult) > 0.0 and sigma_global > 0.0:
        used_cap = True
        cap_min = sigma_global / float(args.cap_mult)
        cap_max = sigma_global * float(args.cap_mult)
        grid = grid[(grid >= cap_min) & (grid <= cap_max)]
        if not np.any(grid == sigma_global):
            grid = np.unique(np.sort(np.append(grid, sigma_global)))
    if grid.size == 0:
        raise ValueError("sigma grid empty after cap")

    def is_boundary(sig: float) -> bool:
        if not used_cap:
            return False
        tol = 1e-12 + 1e-6 * max(1.0, abs(sig))
        return (abs(sig - cap_min) <= tol) or (abs(sig - cap_max) <= tol)

    n_eff = int(args.n_eff) if args.n_eff is not None else int(n_tips)
    tau0 = bic_tau(n_eff, tau_mult=float(args.tau_mult))

    diag_path = os.path.join(args.out, "refit_diagnostics.tsv")
    patch_path = os.path.join(args.out, "refit_patch.npz")

    # empty loci: write empty outputs
    if loci_flagged.size == 0:
        with open(diag_path, "w", encoding="utf-8") as f:
            f.write(
                "locus\taccepted\tgate\tsigma_best\tsigma_global\tscore_best_raw\tscore_global_raw\t"
                "delta_raw\tD_obs\tpD\ttau\tconv_best\tconv_global\tboundary_hit\tused_cap\t"
                "deta_med\tsat_best\tsat_global\tinfo_best\tinfo_global\ttau_logsigma\n"
            )
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
        )
        print(f"[info] No loci to refit. Wrote empty outputs to {args.out}", file=sys.stderr)
        return

    # FASTA → bits aligned to basis tip order
    seqs = read_fasta_dict(args.fasta)
    Y_bits, n_loci_total = pack_bits_from_sequences(tips, seqs, presence_char=args.presence)
    if int(loci_flagged.max()) >= n_loci_total:
        raise ValueError("Flagged loci contain index >= FASTA sequence length")

    m = loci_flagged.size

    # outputs (per flagged locus)
    sigma_best_all = np.zeros(m, dtype=np.float64)
    score_best_raw_all = np.full(m, -np.inf, dtype=np.float64)
    score_global_raw_all = np.full(m, -np.inf, dtype=np.float64)
    conv_best_all = np.zeros(m, dtype=np.bool_)
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

    def rng_for_locus(loc: int) -> np.random.Generator:
        seed = (int(args.seed) ^ (loc * 1000003)) & 0xFFFFFFFF
        return np.random.default_rng(seed)

    def refit_one(idx: int):
        loc = int(loci_flagged[idx])
        y = get_locus_column_bits(Y_bits, loc).astype(np.float64)

        delta_raw, (a_g, b_g, conv_g, sc_g_raw), (s_best, a_best, sc_best_raw, sc_best_pen, b_best, conv_best) = best_vs_global_for_y(
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

        # base gate
        accepted = False
        pD = float("nan")
        if s_best == float(sigma_global):
            accepted = False
        else:
            if args.gate == "none":
                accepted = True
            elif args.gate == "bic":
                if conv_best and conv_g:
                    accepted = (float(delta_raw) >= float(tau0))
                elif conv_best and (not conv_g):
                    accepted = True
            elif args.gate == "lrt_boot":
                if (conv_best and conv_g) and (float(D_obs) >= float(args.boot_min_D)) and int(args.boot_B) > 0:
                    rng = rng_for_locus(loc)
                    pD = bootstrap_p_value_D(
                        Z=Z, a_g=float(a_g), b_g=b_g,
                        sigma_global=float(sigma_global),
                        grid=grid,
                        tau_logsigma=float(args.tau_logsigma),
                        max_iter=args.max_iter, tol=args.tol,
                        D_obs=float(D_obs),
                        B=int(args.boot_B),
                        rng=rng,
                    )
                    accepted = (float(pD) <= float(args.alpha))

        if args.reject_boundary and boundary_hit:
            accepted = False

        # extra constraints
        if float(args.min_deta_med) > 0.0:
            accepted = accepted and (float(deta_med) >= float(args.min_deta_med))
        if float(args.sat_drop_min) > 0.0:
            accepted = accepted and (float(sat_b) <= float(sat_g) - float(args.sat_drop_min))
        if float(args.info_mult_min) > 1.0:
            accepted = accepted and (float(info_b) >= float(info_g) * float(args.info_mult_min))

        return (idx, loc, s_best, a_best, b_best, conv_best, sc_best_raw,
                a_g, b_g, conv_g, sc_g_raw,
                delta_raw, D_obs, pD, accepted, boundary_hit,
                deta_med, sat_b, sat_g, info_b, info_g)

    t0 = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.threads)) as ex:
        futs = [ex.submit(refit_one, i) for i in range(m)]
        for fut in as_completed(futs):
            (idx, loc, s_best, a_best, b_best, conv_best, sc_best_raw,
             a_g, b_g, conv_g, sc_g_raw,
             delta_raw, D_obs, pD, accepted, boundary_hit,
             deta_med, sat_b, sat_g, info_b, info_g) = fut.result()

            sigma_best_all[idx] = float(s_best)
            alpha_best_all[idx] = np.float32(a_best)
            b_best_all[idx, :] = b_best.astype(np.float32, copy=False)
            conv_best_all[idx] = bool(conv_best)
            score_best_raw_all[idx] = float(sc_best_raw)

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

            if args.write_probs and accepted:
                probs_best[idx] = sigmoid(float(a_best) + Z @ b_best).astype(np.float32)

            done += 1
            if done % max(1, args.progress_every) == 0 or done == m:
                dt = time.perf_counter() - t0
                print(f"[info] processed {done}/{m}  rate={done/max(dt,1e-9):.2f} loci/s", file=sys.stderr)

    # write diagnostics
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

    # write patch
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

    if args.write_probs:
        p_path = os.path.join(args.out, "P_refit.npz")
        if loci_refit.size == 0:
            P_ref = np.zeros((n_tips, 0), dtype=np.float32)
        else:
            P_ref = np.zeros((n_tips, loci_refit.size), dtype=np.float32)
            for j, original_idx in enumerate(acc_idx.tolist()):
                P_ref[:, j] = probs_best.get(
                    original_idx,
                    sigmoid(float(alpha_best_all[original_idx]) + Z @ b_best_all[original_idx, :].astype(np.float64)).astype(np.float32),
                )
        np.savez_compressed(p_path, tips=np.array(tips, dtype=object), loci=loci_refit, P=P_ref)

    print(f"[done] wrote: {diag_path}", file=sys.stderr)
    print(f"[done] wrote: {patch_path} (accepted {loci_refit.size}/{m})", file=sys.stderr)


if __name__ == "__main__":
    main()
