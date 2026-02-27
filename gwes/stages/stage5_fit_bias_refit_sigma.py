#!/usr/bin/env python3
from __future__ import annotations

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from gwes.manifest import write_stage_meta
from gwes.prob_store import load_p_hat_from_stage3_dir
from gwes.mathutil import sigmoid, minor_count_from_mean


def try_load_basis_Z(basis_npz: Path, K: int) -> np.ndarray:
    """
    Load tip-by-K basis matrix from phylo_basis.npz.
    Your Stage 3 writes key 'Z' with shape (n_tips, K).
    """
    z = np.load(str(basis_npz), allow_pickle=True)
    if "Z" not in z.files:
        raise ValueError(f"{basis_npz} missing 'Z'. Keys={list(z.files)}")
    Z = z["Z"]
    if not (isinstance(Z, np.ndarray) and Z.ndim == 2 and Z.shape[1] == K):
        raise ValueError(f"{basis_npz} has Z with shape {getattr(Z, 'shape', None)}, expected (*,{K})")
    return Z.astype(np.float32, copy=False)


def build_locus_to_col(p_loci: np.ndarray, loci: np.ndarray) -> Optional[Dict[int, int]]:
    """
    If P_hat columns are already aligned to locus_fit.loci, return None (fast path).
    Else return a mapping locus_id -> column index into P_hat.
    """
    if p_loci.shape == loci.shape and np.array_equal(p_loci, loci):
        return None
    return {int(x): i for i, x in enumerate(p_loci.tolist())}


def stage5_flag_loci(
    run_dir: str,
    out_dirname: str = "stage5",
    diag_source: str = "auto",         # auto|phat|basis|none
    diag_tips: int = 200,
    seed: int = 12345,
    n_tips_fallback: int = 0,

    min_minor: int = 3,
    min_maf: float = 0.005,

    b_test: str = "z",                 # z|mult
    b_z: float = 8.0,
    b_norm_mult: float = 4.0,

    sat_eps: float = 1e-4,
    sat_frac: float = 0.50,

    alpha_abs: float = 12.0,

    qc_name: str = "locus_qc.tsv",
    flagged_name: str = "loci_flagged_refit.txt",
) -> dict:
    run_dir = Path(run_dir)
    stage3_dir = run_dir / "work" / "stage3"
    out_dir = run_dir / "work" / out_dirname
    meta_path = run_dir / "meta" / "stage5.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    locus_fit_path = stage3_dir / "locus_fit.npz"
    if not locus_fit_path.exists():
        raise FileNotFoundError(f"Stage 3 locus_fit not found: {locus_fit_path}")

    lf = np.load(str(locus_fit_path), allow_pickle=True)
    required = ["loci", "alpha", "b", "maf", "mean", "converged", "flags"]
    for k in required:
        if k not in lf.files:
            raise ValueError(f"locus_fit.npz missing '{k}'")

    loci = lf["loci"].astype(np.int64, copy=False)
    alpha = lf["alpha"].astype(np.float64, copy=False)
    b = lf["b"].astype(np.float64, copy=False)          # (n_loci, K)
    maf = lf["maf"].astype(np.float64, copy=False)
    mean = lf["mean"].astype(np.float64, copy=False)
    converged = lf["converged"].astype(bool, copy=False)
    flags_stage3 = lf["flags"].astype(np.uint8, copy=False)

    K = int(lf["K"]) if "K" in lf.files else int(b.shape[1])
    sigma_hat = float(lf["sigma_hat"]) if "sigma_hat" in lf.files else 1.0
    n_loci = int(loci.shape[0])

    # ---- choose diagnostics source ----
    basis_npz = stage3_dir / "phylo_basis.npz"
    if diag_source == "auto":
        if (stage3_dir / "P_hat.npz").exists() or (stage3_dir / "P_hat_meta.json").exists():
            diag_source = "phat"
        elif basis_npz.exists():
            diag_source = "basis"
        else:
            diag_source = "none"

    rng = np.random.default_rng(seed)
    n_tips: Optional[int] = None
    tip_idx: Optional[np.ndarray] = None

    # Optional diagnostics buffers
    Zs: Optional[np.ndarray] = None
    P: Optional[np.ndarray] = None
    locus_to_col: Optional[Dict[int, int]] = None

    if diag_source == "phat":
        P, p_loci, _tips = load_p_hat_from_stage3_dir(stage3_dir, locus_fit_npz=locus_fit_path)
        n_tips = int(P.shape[0])
        locus_to_col = build_locus_to_col(p_loci, loci)

        if diag_tips > 0 and diag_tips < n_tips:
            tip_idx = rng.choice(n_tips, size=diag_tips, replace=False)
        else:
            tip_idx = np.arange(n_tips, dtype=np.int64)

        print(f"[info] Diagnostics via P_hat: n_tips={n_tips}, diag_tips={tip_idx.size}", file=sys.stderr)

    elif diag_source == "basis":
        if not basis_npz.exists():
            raise FileNotFoundError(f"diag_source=basis but missing {basis_npz}")
        Z = try_load_basis_Z(basis_npz, K=K)
        n_tips = int(Z.shape[0])

        if diag_tips > 0 and diag_tips < n_tips:
            tip_idx = rng.choice(n_tips, size=diag_tips, replace=False)
        else:
            tip_idx = np.arange(n_tips, dtype=np.int64)

        Zs = Z[tip_idx, :].astype(np.float64, copy=False)
        print(f"[info] Diagnostics via basis: n_tips={n_tips}, diag_tips={tip_idx.size}", file=sys.stderr)

    else:
        # none
        if "tips" in lf.files:
            n_tips = int(len(lf["tips"]))
        elif n_tips_fallback > 0:
            n_tips = int(n_tips_fallback)
        else:
            n_tips = None
        print("[info] Diagnostics disabled (no saturation).", file=sys.stderr)

    # ---- QC metrics ----
    b_norm2 = np.sum(b * b, axis=1)
    b_norm = np.sqrt(b_norm2)

    # LARGE_B criterion
    if b_test == "mult":
        b_thresh = b_norm_mult * sigma_hat * np.sqrt(max(K, 1))
        b_z_vec = np.full(n_loci, np.nan, dtype=np.float64)
        largeb = (b_norm > b_thresh)
    else:
        denom = np.sqrt(2.0 * max(K, 1))
        b_z_vec = (b_norm2 / (sigma_hat * sigma_hat) - float(K)) / denom
        b_thresh = float("nan")
        largeb = (b_z_vec > b_z)

    # minor count (needs n_tips)
    minor = None
    if n_tips is not None:
        minor = minor_count_from_mean(mean, n_tips)

    # ---- Stage 5 flag bits ----
    LOWMAF = 1
    NOCONV = 2
    LARGE_B = 4
    SATURATED = 8
    EXTREME_ALPHA = 16

    flags5 = np.zeros(n_loci, dtype=np.uint8)

    lowmaf = (maf < min_maf)
    if minor is not None:
        lowmaf = lowmaf | (minor < min_minor)
    flags5[lowmaf] |= LOWMAF

    noc = (~converged) | ((flags_stage3 & 2) != 0)
    flags5[noc] |= NOCONV

    flags5[largeb] |= LARGE_B

    ext_a = (np.abs(alpha) > alpha_abs)
    flags5[ext_a] |= EXTREME_ALPHA

    # SATURATED
    sat_frac_vec = np.full(n_loci, np.nan, dtype=np.float64)
    if diag_source == "phat":
        assert P is not None and tip_idx is not None
        eps = float(sat_eps)

        if locus_to_col is None:
            Ps = P[tip_idx, :]  # (diag_tips, n_loci)
            sat = (Ps < eps) | (Ps > (1.0 - eps))
            sat_frac_vec = np.mean(sat, axis=0).astype(np.float64)
        else:
            for i, loc in enumerate(loci.tolist()):
                col = locus_to_col.get(int(loc), None)
                if col is None:
                    continue
                p = P[tip_idx, col]
                sat_frac_vec[i] = float(np.mean((p < eps) | (p > (1.0 - eps))))

        flags5[sat_frac_vec > sat_frac] |= SATURATED

    elif diag_source == "basis":
        assert Zs is not None
        eps = float(sat_eps)
        block = 256
        for start in range(0, n_loci, block):
            end = min(n_loci, start + block)
            bb = b[start:end, :]      # (blk, K)
            aa = alpha[start:end]     # (blk,)
            eta = Zs @ bb.T
            eta += aa[None, :]
            p = sigmoid(eta)
            sat = (p < eps) | (p > (1.0 - eps))
            sat_frac_vec[start:end] = np.mean(sat, axis=0)

        flags5[sat_frac_vec > sat_frac] |= SATURATED

    flagged_mask = flags5 != 0
    flagged_loci = loci[flagged_mask]

    # ---- write outputs ----
    flagged_path = out_dir / flagged_name
    with flagged_path.open("w", encoding="utf-8") as f:
        for x in flagged_loci.tolist():
            f.write(f"{int(x)}\n")

    qc_path = out_dir / qc_name
    with qc_path.open("w", encoding="utf-8") as f:
        f.write("locus\tmaf\tmean\tminor\tconverged\tflags_stage3\tb_norm\tb_z\tb_thresh\tsat_frac\talpha\tflags5\n")
        for i in range(n_loci):
            loc = int(loci[i])
            mf = float(maf[i])
            mn = float(mean[i])
            mi = int(minor[i]) if minor is not None else -1
            cv = int(converged[i])
            fs3 = int(flags_stage3[i])
            bn = float(b_norm[i])
            bz = float(b_z_vec[i]) if np.isfinite(b_z_vec[i]) else np.nan
            bt = float(b_thresh) if np.isfinite(b_thresh) else np.nan
            sf = float(sat_frac_vec[i]) if np.isfinite(sat_frac_vec[i]) else np.nan
            a = float(alpha[i])
            fl = int(flags5[i])
            f.write(f"{loc}\t{mf:.6g}\t{mn:.6g}\t{mi}\t{cv}\t{fs3}\t{bn:.6g}\t{bz:.6g}\t{bt:.6g}\t{sf:.6g}\t{a:.6g}\t{fl}\n")

    def _count(bit: int) -> int:
        return int(np.sum((flags5 & bit) != 0))

    print(f"[info] sigma_hat={sigma_hat:g}, K={K}", file=sys.stderr)
    print(f"[info] Flagged loci: {flagged_loci.size}/{n_loci}", file=sys.stderr)
    print(f"[info]  LOWMAF       : {_count(LOWMAF)}", file=sys.stderr)
    print(f"[info]  NOCONV       : {_count(NOCONV)}", file=sys.stderr)
    print(f"[info]  LARGE_B      : {_count(LARGE_B)}", file=sys.stderr)
    if diag_source in ("phat", "basis"):
        print(f"[info]  SATURATED    : {_count(SATURATED)}  (source={diag_source})", file=sys.stderr)
    print(f"[info]  EXTREME_ALPHA: {_count(EXTREME_ALPHA)}", file=sys.stderr)
    print(f"[done] Wrote {flagged_path} and {qc_path}", file=sys.stderr)

    meta = {
        "stage": "stage5",
        "inputs": {
            "stage3_dir": str(stage3_dir),
            "locus_fit": str(locus_fit_path),
            "p_hat_used": (diag_source == "phat"),
            "basis_used": (diag_source == "basis"),
        },
        "params": {
            "diag_source": diag_source,
            "diag_tips": int(diag_tips),
            "seed": int(seed),
            "n_tips_fallback": int(n_tips_fallback),
            "min_minor": int(min_minor),
            "min_maf": float(min_maf),
            "b_test": b_test,
            "b_z": float(b_z),
            "b_norm_mult": float(b_norm_mult),
            "sat_eps": float(sat_eps),
            "sat_frac": float(sat_frac),
            "alpha_abs": float(alpha_abs),
        },
        "outputs": {
            "flagged_loci": str(flagged_path),
            "qc_table": str(qc_path),
        },
        "counts": {
            "n_loci": int(n_loci),
            "n_flagged": int(flagged_loci.size),
        }
    }
    write_stage_meta(meta_path, meta)
    return meta


def main():
    ap = argparse.ArgumentParser(description="Stage 5 (run-dir): flag loci needing per-locus sigma refit.")
    ap.add_argument("--run-dir", required=True, help="RUN_DIR containing work/stage3 outputs.")

    ap.add_argument("--diag-source", choices=["auto", "phat", "basis", "none"], default="auto")
    ap.add_argument("--diag-tips", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--n-tips", type=int, default=0)

    ap.add_argument("--min-minor", type=int, default=3)
    ap.add_argument("--min-maf", type=float, default=0.005)

    ap.add_argument("--b-z", type=float, default=4.0)
    ap.add_argument("--b-norm-mult", type=float, default=4.0)
    ap.add_argument("--b-test", choices=["z", "mult"], default="z")

    ap.add_argument("--sat-eps", type=float, default=1e-4)
    ap.add_argument("--sat-frac", type=float, default=0.50)

    ap.add_argument("--alpha-abs", type=float, default=12.0)

    ap.add_argument("--qc-name", default="locus_qc.tsv")
    ap.add_argument("--flagged-name", default="loci_flagged_refit.txt")
    args = ap.parse_args()

    stage5_flag_loci(
        run_dir=args.run_dir,
        diag_source=args.diag_source,
        diag_tips=args.diag_tips,
        seed=args.seed,
        n_tips_fallback=args.n_tips,
        min_minor=args.min_minor,
        min_maf=args.min_maf,
        b_test=args.b_test,
        b_z=args.b_z,
        b_norm_mult=args.b_norm_mult,
        sat_eps=args.sat_eps,
        sat_frac=args.sat_frac,
        alpha_abs=args.alpha_abs,
        qc_name=args.qc_name,
        flagged_name=args.flagged_name,
    )


if __name__ == "__main__":
    main()
