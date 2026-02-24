#!/usr/bin/env python3
from __future__ import annotations

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception:
    print("[error] Stage 3 requires scipy (sparse + eigsh).", file=sys.stderr)
    raise

from gwes.fasta_matrix import load_locusmajor_npz
from gwes.manifest import write_stage_meta
from gwes.matrix_ops import locus_to_u8, loci_to_matrix_u8
from gwes.model_ecobias import logit, sigmoid, fit_locus_map, laplace_log_marginal


def parse_sigma_grid(s: str) -> List[float]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    out = [v for v in out if v >= 0.0]
    if not out:
        raise ValueError("sigma-grid is empty after parsing.")
    return sorted(set(out))


def objective_ridge_sigma(score_med: float, sigma: float, lam: float) -> float:
    return float(score_med - 0.5 * lam * sigma * sigma)


def load_covariance_stage1(npz_path: str) -> Tuple[List[str], object, np.ndarray, int]:
    """
    Accepts Stage 1 output npz containing L (required) and optionally A.
    Returns: tips, Aop_or_dense, diag_A, n
    """
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.files)

    if "tip_names" not in keys and "tips" not in keys:
        raise ValueError("Covariance npz must contain tip_names (Stage 1 output).")
    tips_arr = d["tip_names"] if "tip_names" in keys else d["tips"]
    tips = [str(x) for x in tips_arr.tolist()]

    if "A" in keys:
        A = np.asarray(d["A"])
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A in npz is not a square matrix.")
        return tips, A, np.diag(A), A.shape[0]

    if "L" not in keys:
        raise ValueError("Covariance npz must contain L or A.")
    L = np.asarray(d["L"])
    n = L.shape[0]

    def matvec(v: np.ndarray) -> np.ndarray:
        return L @ (L.T @ v)

    Aop = spla.LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    diag_A = np.sum(L * L, axis=1)
    return tips, Aop, diag_A, n


def read_loci_used(path: Path) -> List[int]:
    loci: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                loci.append(int(s))
    # unique preserve order
    seen = set()
    out = []
    for l in loci:
        if l not in seen:
            out.append(l)
            seen.add(l)
    return out


def stage3_fit_global_sigma(
    run_dir: str,
    k: int = 50,
    eig_tol: float = 1e-6,
    eig_maxiter: int = 5000,
    sigma_grid: str = "0,0.25,0.5,1,2,4,8,16,32,64",
    sigma_loci: int = 1000,
    sigma_seed: int = 1,
    sigma_min_maf: float = 0.05,
    sigma_sat_eps: float = 0.01,
    sigma_ridge_lam: float = 0.0,
    max_iter: int = 50,
    tol: float = 1e-6,
    min_minor: int = 3,
    min_maf: float = 0.0,
    threads: int = 1,
    probs_format: str = "auto",  # auto|npz|memmap
    max_npz_floats: float = 2e7,
) -> dict:
    run_dir = Path(run_dir)
    work0 = run_dir / "work" / "stage0"
    work1 = run_dir / "work" / "stage1"
    work3 = run_dir / "work" / "stage3"
    meta_dir = run_dir / "meta"
    work3.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Load Stage 0 matrix (canonical tips + n_loci)
    mat = load_locusmajor_npz(work0 / "matrix.locusmajor.npz")
    tips_order = mat.tips
    Y_locus_bits = mat.Y_locus_bits
    n_tips = len(tips_order)
    n_loci_total = int(mat.n_loci)

    # Load Stage 1 covariance (must match tips_order exactly)
    cov_path = work1 / "phylo_cov.npz"
    cov_tips, Aop, diag_A, n_cov = load_covariance_stage1(str(cov_path))
    if n_cov != n_tips:
        raise ValueError(f"Covariance n={n_cov} != matrix n_tips={n_tips}")
    if cov_tips != tips_order:
        raise ValueError("Tip order mismatch: Stage 1 covariance tips != Stage 0 matrix tips")

    # Loci used
    loci_used_path = work0 / "loci_used.txt"
    loci_used = read_loci_used(loci_used_path)
    if not loci_used:
        raise ValueError("loci_used is empty.")
    if max(loci_used) >= n_loci_total:
        raise ValueError("loci_used contains an index >= n_loci in matrix.")
    print(f"[info] loci_used: {len(loci_used)}", file=sys.stderr)

    # Build low-rank basis Z from top-K eigenpairs of A
    K = int(k)
    if K <= 0 or K >= n_tips:
        raise ValueError(f"k must be in [1, n_tips-1]. Got k={K}, n_tips={n_tips}.")

    traceA = float(np.sum(diag_A))
    print(f"[info] trace(A) ~ {traceA:.6g}", file=sys.stderr)
    print(f"[info] Computing top-{K} eigenpairs ...", file=sys.stderr)

    if isinstance(Aop, np.ndarray):
        A_lin = spla.aslinearoperator(Aop)
    elif sp.issparse(Aop):
        A_lin = spla.aslinearoperator(Aop)
    else:
        A_lin = Aop

    eigvals, eigvecs = spla.eigsh(A_lin, k=K, which="LA", tol=eig_tol, maxiter=eig_maxiter)
    idx = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[idx], 0.0)
    eigvecs = eigvecs[:, idx]

    var_frac = float(np.sum(eigvals) / max(traceA, 1e-30))
    print(f"[info] Top-{K} eigenvalue mass fraction: {var_frac:.4f}", file=sys.stderr)

    sqrt_lambda = np.sqrt(eigvals)
    Z64 = (eigvecs * sqrt_lambda[None, :]).astype(np.float64, copy=False)

    np.savez(
        work3 / "phylo_basis.npz",
        tips=np.array(tips_order, dtype=object),
        K=np.int32(K),
        eigvals=eigvals.astype(np.float64),
        sqrt_lambda=sqrt_lambda.astype(np.float64),
        Z=Z64.astype(np.float32),
    )

    # -------- sigma selection --------
    grid = parse_sigma_grid(sigma_grid)
    lam_sigma = float(sigma_ridge_lam)
    n = int(n_tips)

    print(f"[info] Building sigma-scoring pool: MAF>={sigma_min_maf}, minor>={min_minor} ...", file=sys.stderr)
    eligible: List[int] = []
    for ii, l in enumerate(loci_used, start=1):
        y = locus_to_u8(Y_locus_bits, l, n_tips)
        c1 = int(y.sum())
        mean = c1 / n
        maf = min(mean, 1.0 - mean)
        minor = min(c1, n - c1)
        if (maf >= sigma_min_maf) and (minor >= min_minor):
            eligible.append(l)
        if ii % 5000 == 0 or ii == len(loci_used):
            print(f"[info]  scanned {ii}/{len(loci_used)} eligible={len(eligible)}", file=sys.stderr)

    if not eligible:
        raise ValueError("No loci passed sigma-scoring filter.")

    m = min(int(sigma_loci), len(eligible))
    rng = np.random.default_rng(int(sigma_seed))
    loci_sigma = rng.choice(np.array(eligible, dtype=np.int64), size=m, replace=False).astype(np.int64)
    print(f"[info] Sigma scoring loci: {m} sampled from {len(eligible)} eligible", file=sys.stderr)

    Y_sigma = loci_to_matrix_u8(Y_locus_bits, loci_sigma.tolist(), n_tips)  # (n_tips, m)
    mean_sigma = Y_sigma.mean(axis=0).astype(np.float64, copy=False)

    alpha_ws = np.array([logit(float(mu)) for mu in mean_sigma], dtype=np.float64)
    b_ws = np.zeros((m, K), dtype=np.float64)

    sat_eps = float(sigma_sat_eps)
    best_obj = -float("inf")
    best_sigma = None
    rows = []

    print(f"[info] Scoring sigma grid on {m} loci (median Laplace) ...", file=sys.stderr)
    for s in grid:
        scores = np.zeros(m, dtype=np.float64)
        sat = np.zeros(m, dtype=np.float64)
        conv = np.zeros(m, dtype=np.bool_)

        for j in range(m):
            y = Y_sigma[:, j].astype(np.float64)

            a, b, ok, _, _, H = fit_locus_map(
                y=y, Z=Z64, sigma=float(s),
                max_iter=max_iter, tol=tol,
                init_alpha=float(alpha_ws[j]), init_b=b_ws[j, :]
            )
            alpha_ws[j] = float(a)
            b_ws[j, :] = b
            conv[j] = bool(ok)

            scores[j] = laplace_log_marginal(y, Z64, a, b, float(s), H)

            p = sigmoid(a + (Z64 @ b))
            sat[j] = float(np.mean((p < sat_eps) | (p > (1.0 - sat_eps))))

        score_med = float(np.median(scores))
        score_mean = float(np.mean(scores))
        q10 = float(np.quantile(scores, 0.10))
        q90 = float(np.quantile(scores, 0.90))
        sat_med = float(np.median(sat))
        sat_q90 = float(np.quantile(sat, 0.90))
        conv_rate = float(np.mean(conv.astype(np.float64)))
        obj = objective_ridge_sigma(score_med, float(s), lam_sigma)

        print(
            f"[info] sigma={s:g} median={score_med:.6g} mean={score_mean:.6g} "
            f"q10={q10:.6g} q90={q90:.6g} sat_med={sat_med:.3f} sat_q90={sat_q90:.3f} "
            f"conv={conv_rate:.3f} obj={obj:.6g} (lam={lam_sigma:g})",
            file=sys.stderr,
        )
        rows.append((float(s), score_med, score_mean, q10, q90, sat_med, sat_q90, conv_rate, obj))

        if (obj > best_obj + 1e-12) or (abs(obj - best_obj) <= 1e-12 and (best_sigma is None or float(s) < best_sigma)):
            best_obj = obj
            best_sigma = float(s)

    assert best_sigma is not None
    sigma_hat = float(best_sigma)
    print(f"[info] Chosen global sigma_hat={sigma_hat:g}", file=sys.stderr)

    # global_sigma.tsv
    with (work3 / "global_sigma.tsv").open("w", encoding="utf-8") as f:
        f.write("sigma\tscore_med\tscore_mean\tscore_q10\tscore_q90\tsat_med\tsat_q90\tconv_rate\tobjective\tlam_sigma\n")
        for s, med, mean_sc, q10, q90, sm, sq90, cr, obj in rows:
            f.write(f"{s}\t{med}\t{mean_sc}\t{q10}\t{q90}\t{sm}\t{sq90}\t{cr}\t{obj}\t{lam_sigma}\n")
        f.write(f"chosen\t{sigma_hat}\n")

    # -------- fit all loci_used with sigma_hat --------
    n_used = len(loci_used)
    n_prob = n_tips * n_used
    if probs_format == "auto":
        probs_format_eff = "npz" if n_prob <= max_npz_floats else "memmap"
    else:
        probs_format_eff = probs_format

    alpha_out = np.zeros(n_used, dtype=np.float32)
    b_out = np.zeros((n_used, K), dtype=np.float32)
    maf_out = np.zeros(n_used, dtype=np.float32)
    mean_out = np.zeros(n_used, dtype=np.float32)
    converged_out = np.zeros(n_used, dtype=np.bool_)
    flags_out = np.zeros(n_used, dtype=np.uint8)

    FLAG_LOWMAF = 1
    FLAG_NOCONV = 2

    P_mem = None
    P_npz_buf = None

    if probs_format_eff == "memmap":
        p_path = work3 / "P_hat.mmap"
        P_mem = np.memmap(str(p_path), dtype=np.float32, mode="w+", shape=(n_tips, n_used))
        meta = {
            "path": "P_hat.mmap",
            "dtype": "float32",
            "shape": [n_tips, n_used],
            "order": "C",
            "tips": "stored_in_locus_fit.npz",
            "loci": "stored_in_locus_fit.npz",
        }
        with (work3 / "P_hat_meta.json").open("w", encoding="utf-8") as jf:
            json.dump(meta, jf, indent=2)
    else:
        P_npz_buf = np.zeros((n_tips, n_used), dtype=np.float32)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def fit_one(idx_locus: int):
        locus = int(loci_used[idx_locus])
        y_u8 = locus_to_u8(Y_locus_bits, locus, n_tips)
        c1 = int(y_u8.sum())
        mean = c1 / n
        maf = min(mean, 1.0 - mean)
        minor = min(c1, n - c1)

        flag = 0
        if minor < int(min_minor) or maf < float(min_maf):
            flag |= FLAG_LOWMAF
            a = logit(float(mean))
            b = np.zeros(K, dtype=np.float64)
            p = np.full(n_tips, sigmoid(a), dtype=np.float32)
            return idx_locus, float(a), b, float(maf), float(mean), True, flag, p

        y = y_u8.astype(np.float64)
        a, b, conv, _, _, _ = fit_locus_map(
            y=y, Z=Z64, sigma=float(sigma_hat),
            max_iter=max_iter, tol=tol
        )
        if not conv:
            flag |= FLAG_NOCONV
        p = sigmoid(a + (Z64 @ b)).astype(np.float32, copy=False)
        return idx_locus, float(a), b.astype(np.float64), float(maf), float(mean), bool(conv), flag, p

    flagged_list: List[int] = []
    done = 0
    print(f"[info] Fitting {n_used} loci with sigma_hat={sigma_hat:g} ...", file=sys.stderr)

    if int(threads) <= 1:
        for i in range(n_used):
            idx_locus, a, b, maf, mean, conv, flag, p = fit_one(i)
            alpha_out[idx_locus] = np.float32(a)
            b_out[idx_locus, :] = b.astype(np.float32, copy=False)
            maf_out[idx_locus] = np.float32(maf)
            mean_out[idx_locus] = np.float32(mean)
            converged_out[idx_locus] = bool(conv)
            flags_out[idx_locus] = np.uint8(flag)
            if flag != 0:
                flagged_list.append(loci_used[idx_locus])
            if P_mem is not None:
                P_mem[:, idx_locus] = p
            else:
                P_npz_buf[:, idx_locus] = p
            done += 1
            if done % 200 == 0 or done == n_used:
                print(f"[info]  fitted {done}/{n_used}", file=sys.stderr)
    else:
        with ThreadPoolExecutor(max_workers=int(threads)) as ex:
            futs = [ex.submit(fit_one, i) for i in range(n_used)]
            for fut in as_completed(futs):
                idx_locus, a, b, maf, mean, conv, flag, p = fut.result()
                alpha_out[idx_locus] = np.float32(a)
                b_out[idx_locus, :] = b.astype(np.float32, copy=False)
                maf_out[idx_locus] = np.float32(maf)
                mean_out[idx_locus] = np.float32(mean)
                converged_out[idx_locus] = bool(conv)
                flags_out[idx_locus] = np.uint8(flag)
                if flag != 0:
                    flagged_list.append(loci_used[idx_locus])
                if P_mem is not None:
                    P_mem[:, idx_locus] = p
                else:
                    P_npz_buf[:, idx_locus] = p
                done += 1
                if done % 200 == 0 or done == n_used:
                    print(f"[info]  fitted {done}/{n_used}", file=sys.stderr)

    if P_mem is not None:
        P_mem.flush()
    else:
        np.savez_compressed(
            work3 / "P_hat.npz",
            tips=np.array(tips_order, dtype=object),
            loci=np.array(loci_used, dtype=np.int64),
            P=P_npz_buf,
        )

    np.savez_compressed(
        work3 / "locus_fit.npz",
        tips=np.array(tips_order, dtype=object),
        loci=np.array(loci_used, dtype=np.int64),
        K=np.int32(K),
        sigma_hat=np.float64(sigma_hat),
        alpha=alpha_out,
        b=b_out,
        maf=maf_out,
        mean=mean_out,
        converged=converged_out,
        flags=flags_out,
    )

    with (work3 / "loci_flagged.txt").open("w", encoding="utf-8") as f:
        for l in flagged_list:
            f.write(f"{int(l)}\n")

    meta = {
        "stage": "stage3",
        "inputs": {
            "matrix_locusmajor": str(work0 / "matrix.locusmajor.npz"),
            "loci_used": str(loci_used_path),
            "phylo_cov": str(cov_path),
        },
        "params": {
            "k": K,
            "eig_tol": eig_tol,
            "eig_maxiter": eig_maxiter,
            "sigma_grid": grid,
            "sigma_loci": int(sigma_loci),
            "sigma_seed": int(sigma_seed),
            "sigma_min_maf": float(sigma_min_maf),
            "sigma_ridge_lam": float(sigma_ridge_lam),
            "min_minor": int(min_minor),
            "min_maf": float(min_maf),
            "threads": int(threads),
            "probs_format": probs_format_eff,
        },
        "counts": {
            "n_tips": n_tips,
            "n_loci_total": n_loci_total,
            "n_loci_used": n_used,
        },
        "outputs": {
            "out_dir": str(work3),
            "phylo_basis": str(work3 / "phylo_basis.npz"),
            "global_sigma": str(work3 / "global_sigma.tsv"),
            "locus_fit": str(work3 / "locus_fit.npz"),
            "P_hat": str(work3 / ("P_hat.mmap" if probs_format_eff == "memmap" else "P_hat.npz")),
            "loci_flagged": str(work3 / "loci_flagged.txt"),
        },
        "summary": {"sigma_hat": sigma_hat, "eig_mass_frac": var_frac},
    }
    write_stage_meta(meta_dir / "stage3.json", meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser("Stage 3: global sigma ecobias fit (reads Stage0+Stage1 from --run-dir)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--eig-tol", type=float, default=1e-6)
    ap.add_argument("--eig-maxiter", type=int, default=5000)
    ap.add_argument("--sigma-grid", default="0,0.25,0.5,1,2,4,8,16,32,64")
    ap.add_argument("--sigma-loci", type=int, default=1000)
    ap.add_argument("--sigma-seed", type=int, default=1)
    ap.add_argument("--sigma-min-maf", type=float, default=0.05)
    ap.add_argument("--sigma-sat-eps", type=float, default=0.01)
    ap.add_argument("--sigma-ridge-lam", type=float, default=0.0)
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--min-minor", type=int, default=3)
    ap.add_argument("--min-maf", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--probs-format", choices=["auto", "npz", "memmap"], default="auto")
    ap.add_argument("--max-npz-floats", type=float, default=2e7)
    args = ap.parse_args()

    meta = stage3_fit_global_sigma(
        run_dir=args.run_dir,
        k=args.k,
        eig_tol=args.eig_tol,
        eig_maxiter=args.eig_maxiter,
        sigma_grid=args.sigma_grid,
        sigma_loci=args.sigma_loci,
        sigma_seed=args.sigma_seed,
        sigma_min_maf=args.sigma_min_maf,
        sigma_sat_eps=args.sigma_sat_eps,
        sigma_ridge_lam=args.sigma_ridge_lam,
        max_iter=args.max_iter,
        tol=args.tol,
        min_minor=args.min_minor,
        min_maf=args.min_maf,
        threads=args.threads,
        probs_format=args.probs_format,
        max_npz_floats=args.max_npz_floats,
    )
    print(f"[ok] stage3 wrote: {meta['outputs']['out_dir']}")


if __name__ == "__main__":
    main()
