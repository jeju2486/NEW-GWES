#!/usr/bin/env python3
from __future__ import annotations

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
    
import argparse
import heapq
import json
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np


# -------------------------
# Small helpers
# -------------------------

def _detect_delim(header_line: str) -> Optional[str]:
    return "\t" if "\t" in header_line else None

def _split(line: str, delim: Optional[str]) -> List[str]:
    line = line.rstrip("\n")
    return line.split(delim) if delim is not None else line.split()

def _find_col(cols: List[str], candidates: List[str]) -> Optional[int]:
    lower = [c.lower() for c in cols]
    for name in candidates:
        nl = name.lower()
        if nl in lower:
            return lower.index(nl)
    return None

def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


# -------------------------
# Stats: logOR and MI (robust, consistent with Stage 4)
# -------------------------

def log_or_from_counts(n00, n01, n10, n11, pc: float) -> np.ndarray:
    n00 = np.asarray(n00, dtype=np.float64)
    n01 = np.asarray(n01, dtype=np.float64)
    n10 = np.asarray(n10, dtype=np.float64)
    n11 = np.asarray(n11, dtype=np.float64)
    a = n11 + pc
    b = n10 + pc
    c = n01 + pc
    d = n00 + pc
    return np.log((a * d) / (b * c))

def smooth_probs_from_counts(n00, n01, n10, n11, pc: float):
    n00 = np.asarray(n00, dtype=np.float64)
    n01 = np.asarray(n01, dtype=np.float64)
    n10 = np.asarray(n10, dtype=np.float64)
    n11 = np.asarray(n11, dtype=np.float64)
    c00 = n00 + pc
    c01 = n01 + pc
    c10 = n10 + pc
    c11 = n11 + pc
    tot = c00 + c01 + c10 + c11
    return c00 / tot, c01 / tot, c10 / tot, c11 / tot

def mutual_information_from_probs(p00, p01, p10, p11, base: str) -> np.ndarray:
    """
    MI = sum p_ab log_base( p_ab / (p_a. p_.b) ), with eps guards.
    """
    p00 = np.asarray(p00, dtype=np.float64)
    p01 = np.asarray(p01, dtype=np.float64)
    p10 = np.asarray(p10, dtype=np.float64)
    p11 = np.asarray(p11, dtype=np.float64)

    p0i = p00 + p01
    p1i = p10 + p11
    p0j = p00 + p10
    p1j = p01 + p11

    eps = 1e-300
    t00 = (p00 + eps) / (p0i * p0j + eps)
    t01 = (p01 + eps) / (p0i * p1j + eps)
    t10 = (p10 + eps) / (p1i * p0j + eps)
    t11 = (p11 + eps) / (p1i * p1j + eps)

    if base == "2":
        lg = np.log2
    elif base == "10":
        lg = np.log10
    else:
        lg = np.log

    return p00 * lg(t00) + p01 * lg(t01) + p10 * lg(t10) + p11 * lg(t11)

def _enforce_valid_joint(p1i: float, p1j: float, p11: float) -> Tuple[float, float, float, float]:
    """
    Enforce a valid 2x2 joint consistent with marginals:
      p11 in [max(0,p1i+p1j-1), min(p1i,p1j)]
      p10=p1i-p11, p01=p1j-p11, p00=1-p10-p01-p11
    """
    p1i = float(np.clip(p1i, 0.0, 1.0))
    p1j = float(np.clip(p1j, 0.0, 1.0))
    lo = max(0.0, p1i + p1j - 1.0)
    hi = min(p1i, p1j)
    p11 = float(np.clip(p11, lo, hi))
    p10 = p1i - p11
    p01 = p1j - p11
    p00 = 1.0 - p10 - p01 - p11
    # numeric cleanup
    p00 = float(np.clip(p00, 0.0, 1.0))
    p01 = float(np.clip(p01, 0.0, 1.0))
    p10 = float(np.clip(p10, 0.0, 1.0))
    p11 = float(np.clip(p11, 0.0, 1.0))
    return p00, p01, p10, p11


# -------------------------
# Load Stage 3 P_hat (+ optional override)
# -------------------------

def load_stage3_probs(path: str, locus_fit: Optional[str] = None):
    """
    Returns:
      P: ndarray or memmap, shape (n_tips, n_loci_used), float32
      loci: ndarray int64, shape (n_loci_used,)
      tips: optional list[str] if present in npz, else None
    """
    if path.endswith(".json") and (not os.path.exists(path)):
        alt_npz = os.path.join(os.path.dirname(path), "P_hat.npz")
        if os.path.exists(alt_npz):
            path = alt_npz
        else:
            d = os.path.dirname(path)
            if os.path.isdir(d):
                path = d

    if os.path.isdir(path):
        d = path
        npz = os.path.join(d, "P_hat.npz")
        meta = os.path.join(d, "P_hat_meta.json")
        if os.path.exists(npz):
            path = npz
        elif os.path.exists(meta):
            path = meta
        else:
            raise FileNotFoundError(f"Could not find P_hat.npz or P_hat_meta.json in: {d}")

    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True)
        if "P" not in z.files or "loci" not in z.files:
            raise ValueError("P_hat.npz must contain arrays 'P' and 'loci'.")
        P = z["P"].astype(np.float32, copy=False)
        loci = z["loci"].astype(np.int64, copy=False)
        tips = [str(x) for x in z["tips"].tolist()] if "tips" in z.files else None
        return P, loci, tips

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        mmap_rel = meta["path"]
        mmap_path = mmap_rel if os.path.isabs(mmap_rel) else os.path.join(os.path.dirname(path), mmap_rel)
        shape = tuple(meta["shape"])
        dtype = np.dtype(meta["dtype"])
        P = np.memmap(mmap_path, dtype=dtype, mode="r", shape=shape, order="C")

        if locus_fit is None:
            guess = os.path.join(os.path.dirname(path), "locus_fit.npz")
            locus_fit = guess if os.path.exists(guess) else None
        if locus_fit is None or not os.path.exists(locus_fit):
            raise FileNotFoundError("For P_hat memmap, provide --locus-fit (or keep locus_fit.npz alongside).")

        lf = np.load(locus_fit, allow_pickle=True)
        if "loci" not in lf.files:
            raise ValueError("locus_fit.npz must contain 'loci'.")
        loci = lf["loci"].astype(np.int64, copy=False)
        return P, loci, None

    raise ValueError("Unrecognized --p-hat input.")

def load_override_probs(path: Optional[str], n_tips: int, tips_base: Optional[List[str]]):
    """
    Returns:
      P_ov: (n_tips, n_refit) float32 or None
      ov_map: locus->col dict
      tips_ov: list[str] or None
    """
    if path is None:
        return None, {}, None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    if not all(k in z.files for k in ["tips", "loci", "P"]):
        raise ValueError("P_refit.npz must contain tips,loci,P")
    tips_ov = [str(x) for x in z["tips"].tolist()]
    if len(tips_ov) != n_tips:
        raise ValueError("P_refit tips length != n_tips")
    if tips_base is not None and tips_ov != tips_base:
        raise ValueError("Tip order mismatch between P_hat.npz and P_refit.npz")
    P_ov = z["P"].astype(np.float32, copy=False)
    loci_ov = z["loci"].astype(np.int64, copy=False)
    if P_ov.shape[0] != n_tips or P_ov.shape[1] != loci_ov.shape[0]:
        raise ValueError("P_refit shape mismatch")
    ov_map = {int(l): i for i, l in enumerate(loci_ov.tolist())}
    return P_ov, ov_map, tips_ov


# -------------------------
# Pair record
# -------------------------

@dataclass
class PairRec:
    i: int
    j: int
    distance: int
    n00: int
    n01: int
    n10: int
    n11: int
    rank_val: float


# -------------------------
# Bootstrap for one pair
# -------------------------

def _get_prob_vector(
    locus: int,
    P: np.ndarray,
    locus_to_col: Dict[int, int],
    P_ov: Optional[np.ndarray],
    ov_locus_to_col: Dict[int, int],
) -> Optional[np.ndarray]:
    oc = ov_locus_to_col.get(int(locus), None) if P_ov is not None else None
    if oc is not None:
        return P_ov[:, oc].astype(np.float64, copy=False)
    bc = locus_to_col.get(int(locus), None)
    if bc is None:
        return None
    return P[:, bc].astype(np.float64, copy=False)

def bootstrap_pair(
    rec: PairRec,
    P: np.ndarray,
    locus_to_col: Dict[int, int],
    P_ov: Optional[np.ndarray],
    ov_locus_to_col: Dict[int, int],
    B: int,
    boot_block: int,
    pc: float,
    mi_base: str,
    two_sided: bool,
    seed: int,
) -> Dict[str, float]:
    pi = _get_prob_vector(rec.i, P, locus_to_col, P_ov, ov_locus_to_col)
    pj = _get_prob_vector(rec.j, P, locus_to_col, P_ov, ov_locus_to_col)
    if pi is None or pj is None:
        return {
            "missing": 1.0,
            "delta11_obs": np.nan, "rlogOR_obs": np.nan, "rMI_obs": np.nan,
            "p_delta11": np.nan, "p_rlogOR": np.nan, "p_rMI": np.nan,
            "z_delta11": np.nan, "z_rlogOR": np.nan, "z_rMI": np.nan,
        }

    n_tips = int(pi.shape[0])
    pi = np.clip(pi, 0.0, 1.0)
    pj = np.clip(pj, 0.0, 1.0)

    # Deterministic null (mixture over tips)
    p1i = float(np.mean(pi))
    p1j = float(np.mean(pj))
    p11_null = float(np.mean(pi * pj))
    p00_null, p01_null, p10_null, p11_null = _enforce_valid_joint(p1i, p1j, p11_null)

    # Observed stats from counts
    n00 = float(rec.n00); n01 = float(rec.n01); n10 = float(rec.n10); n11 = float(rec.n11)
    n_obs = n00 + n01 + n10 + n11
    if n_obs <= 0:
        return {
            "missing": 0.0,
            "delta11_obs": np.nan, "rlogOR_obs": np.nan, "rMI_obs": np.nan,
            "p_delta11": np.nan, "p_rlogOR": np.nan, "p_rMI": np.nan,
            "z_delta11": np.nan, "z_rlogOR": np.nan, "z_rMI": np.nan,
        }

    # Use observed sample size for expected counts (matches Stage 4 definition)
    n = float(n_obs)

    p11_obs = n11 / n
    delta11_obs = p11_obs - p11_null

    logOR_obs = float(log_or_from_counts(n00, n01, n10, n11, pc=pc))
    e00 = n * p00_null
    e01 = n * p01_null
    e10 = n * p10_null
    e11 = n * p11_null
    logOR_null = float(log_or_from_counts(e00, e01, e10, e11, pc=pc))
    rlogOR_obs = logOR_obs - logOR_null

    o00, o01, o10, o11 = smooth_probs_from_counts(n00, n01, n10, n11, pc=pc)
    MI_obs = float(mutual_information_from_probs(o00, o01, o10, o11, base=mi_base))
    q00, q01, q10, q11 = smooth_probs_from_counts(e00, e01, e10, e11, pc=pc)
    MI_null = float(mutual_information_from_probs(q00, q01, q10, q11, base=mi_base))
    rMI_obs = MI_obs - MI_null

    # Per-tip joint probs under null for sampling (vector length n_tips)
    p00t = (1.0 - pi) * (1.0 - pj)
    p01t = (1.0 - pi) * pj
    p10t = pi * (1.0 - pj)
    # p11t = pi * pj  # implied by 1 - t2

    # cumulative thresholds for categorical sampling (per tip)
    t0 = p00t.astype(np.float32, copy=False)
    t1 = (p00t + p01t).astype(np.float32, copy=False)
    t2 = (p00t + p01t + p10t).astype(np.float32, copy=False)

    # numeric monotonicity + clipping
    t0 = np.clip(t0, 0.0, 1.0)
    t1 = np.clip(np.maximum(t1, t0), 0.0, 1.0)
    t2 = np.clip(np.maximum(t2, t1), 0.0, 1.0)

    rng = np.random.default_rng(seed)

    delta11_b = np.empty(B, dtype=np.float64)
    rlogOR_b = np.empty(B, dtype=np.float64)
    rMI_b = np.empty(B, dtype=np.float64)

    done = 0
    n_int = int(n_obs)

    # If n differs from n_tips, fall back to sampling tips with replacement to match n.
    mismatch = (n_int != n_tips)

    while done < B:
        bb = min(int(boot_block), B - done)

        if not mismatch:
            # fixed design across tips: bb x n_tips
            u = rng.random((bb, n_tips)).astype(np.float32, copy=False)

            m0 = (u < t0).sum(axis=1).astype(np.float64)
            m1 = (u < t1).sum(axis=1).astype(np.float64)
            m2 = (u < t2).sum(axis=1).astype(np.float64)

            n00b = m0
            n01b = m1 - m0
            n10b = m2 - m1
            n11b = n_tips - m2

            denom = float(n_tips)

        else:
            # sample n_int draws by first sampling a tip index (with replacement), then sampling its joint state
            tip_idx = rng.integers(0, n_tips, size=(bb, n_int), dtype=np.int32)
            u = rng.random((bb, n_int)).astype(np.float32, copy=False)

            tt0 = t0[tip_idx]
            tt1 = t1[tip_idx]
            tt2 = t2[tip_idx]

            m0 = (u < tt0).sum(axis=1).astype(np.float64)
            m1 = (u < tt1).sum(axis=1).astype(np.float64)
            m2 = (u < tt2).sum(axis=1).astype(np.float64)

            n00b = m0
            n01b = m1 - m0
            n10b = m2 - m1
            n11b = n_int - m2

            denom = float(n_int)

        # delta11
        p11b = n11b / denom
        delta11_b[done:done + bb] = p11b - p11_null

        # rlogOR: compare against same expected logOR_null (based on observed n)
        logORb = log_or_from_counts(n00b, n01b, n10b, n11b, pc=pc)
        rlogOR_b[done:done + bb] = logORb - logOR_null

        # rMI: compare against same MI_null (based on expected counts with observed n)
        ob00, ob01, ob10, ob11 = smooth_probs_from_counts(n00b, n01b, n10b, n11b, pc=pc)
        MIb = mutual_information_from_probs(ob00, ob01, ob10, ob11, base=mi_base)
        rMI_b[done:done + bb] = MIb - MI_null

        done += bb

    def pval(boot: np.ndarray, obs: float) -> float:
        if np.isnan(obs):
            return np.nan
        if two_sided:
            return float((1.0 + np.sum(np.abs(boot) >= abs(obs))) / (boot.size + 1.0))
        return float((1.0 + np.sum(boot >= obs)) / (boot.size + 1.0))

    def zscore(boot: np.ndarray, obs: float) -> float:
        if np.isnan(obs):
            return np.nan
        mu = float(np.mean(boot))
        sd = float(np.std(boot, ddof=1))
        return float((obs - mu) / sd) if sd > 0 else np.nan

    return {
        "missing": 0.0,
        "delta11_obs": float(delta11_obs),
        "rlogOR_obs": float(rlogOR_obs),
        "rMI_obs": float(rMI_obs),
        "p_delta11": pval(delta11_b, delta11_obs),
        "p_rlogOR": pval(rlogOR_b, rlogOR_obs),
        "p_rMI": pval(rMI_b, rMI_obs),
        "z_delta11": zscore(delta11_b, delta11_obs),
        "z_rlogOR": zscore(rlogOR_b, rlogOR_obs),
        "z_rMI": zscore(rMI_b, rMI_obs),
    }


# -------------------------
# Top-N selection
# -------------------------

def _auto_rank_col(cols: List[str]) -> str:
    rmi_cols = [c for c in cols if c.startswith("rMI_")]
    if len(rmi_cols) == 1:
        return rmi_cols[0]
    if "rlogOR" in cols:
        return "rlogOR"
    if "delta11" in cols:
        return "delta11"
    raise ValueError("Could not auto-select rank column; pass --rank-by explicitly.")

def _infer_mi_base(cols: List[str], user_base: Optional[str]) -> str:
    if user_base is not None:
        return user_base
    mi_cols = [c for c in cols if c.startswith("MI_obs_")]
    if len(mi_cols) == 1:
        suf = mi_cols[0].split("MI_obs_")[-1]
        if suf in {"e", "2", "10"}:
            return suf
    return "e"

def select_top_pairs(
    pairs_path: str,
    top_n: int,
    rank_by: Optional[str],
    descending: bool,
    abs_rank: bool,
    min_distance: int,
    progress_every: int,
) -> Tuple[List[PairRec], str, str]:
    """
    Returns selected PairRec list, rank_by used, inferred mi_base (may be used later).
    """
    t0 = time.perf_counter()
    heap: List[Tuple[float, int, PairRec]] = []
    tie = 0
    scanned = 0

    with open(pairs_path, "r", encoding="utf-8") as fin:
        header = fin.readline()
        if not header:
            raise ValueError("pairs file is empty")
        delim = _detect_delim(header)
        cols = _split(header, delim)
        ncols = len(cols)

        col_i = _find_col(cols, ["v", "unitig_i", "locus_i", "site_i", "idx_i", "i"])
        col_j = _find_col(cols, ["w", "unitig_j", "locus_j", "site_j", "idx_j", "j"])
        if col_i is None or col_j is None:
            raise ValueError("Could not find locus columns (v/w or unitig_i/unitig_j etc.).")

        col_n00 = _find_col(cols, ["n00", "c00"])
        col_n01 = _find_col(cols, ["n01", "c01"])
        col_n10 = _find_col(cols, ["n10", "c10"])
        col_n11 = _find_col(cols, ["n11", "c11"])
        if None in (col_n00, col_n01, col_n10, col_n11):
            raise ValueError("Need n00,n01,n10,n11 (or c00..c11).")

        col_dist = _find_col(cols, ["distance", "dist"])
        col_miss = _find_col(cols, ["missing_loci"])  # Stage 4/7 usually provides this

        mi_base = _infer_mi_base(cols, user_base=None)

        used_rank = rank_by or _auto_rank_col(cols)
        if used_rank not in cols:
            raise ValueError(f"--rank-by not found in header: {used_rank}")
        col_rank = cols.index(used_rank)

        for line in fin:
            if not line.strip():
                continue
            parts = _split(line, delim)
            if len(parts) < ncols:
                continue

            scanned += 1

            # skip missing_loci==1 if present
            if col_miss is not None:
                try:
                    if int(float(parts[col_miss])) != 0:
                        continue
                except Exception:
                    pass

            try:
                i = int(parts[col_i]); j = int(parts[col_j])
                n00 = int(float(parts[col_n00])); n01 = int(float(parts[col_n01]))
                n10 = int(float(parts[col_n10])); n11 = int(float(parts[col_n11]))
                rv = float(parts[col_rank])
                if not np.isfinite(rv):
                    continue
            except Exception:
                continue

            dist = -1
            if col_dist is not None:
                try:
                    dist = int(float(parts[col_dist]))
                except Exception:
                    dist = -1

            # distance filter only if distance known (>=0) and filter enabled
            if min_distance > 0 and dist >= 0 and dist < min_distance:
                continue

            metric = abs(rv) if abs_rank else rv
            # objective to maximize
            key = metric if descending else -metric

            rec = PairRec(i=i, j=j, distance=dist, n00=n00, n01=n01, n10=n10, n11=n11, rank_val=rv)

            tie += 1
            if len(heap) < top_n:
                heapq.heappush(heap, (key, tie, rec))
            else:
                # replace worst if better
                if key > heap[0][0]:
                    heapq.heapreplace(heap, (key, tie, rec))

            if progress_every > 0 and (scanned % progress_every) == 0:
                dt = time.perf_counter() - t0
                print(f"[info] scanned={scanned:,} kept={len(heap)} rate={scanned/max(dt,1e-9):,.1f} lines/s",
                      file=sys.stderr)

    selected = [r for _, __, r in heap]
    selected.sort(
        key=lambda r: (abs(r.rank_val) if abs_rank else r.rank_val),
        reverse=descending
    )
    return selected, used_rank, mi_base


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 8: bootstrap p-values for top pairs under structured null.")
    ap.add_argument("--pairs", required=True, help="Stage 4/7 TSV (pairs_resid*.tsv).")
    ap.add_argument("--p-hat", required=True, help="Stage 3 P_hat.npz or P_hat_meta.json (or directory).")
    ap.add_argument("--locus-fit", default=None, help="Stage 3 locus_fit.npz (needed for memmap meta).")
    ap.add_argument("--p-override", default=None, help="Optional Stage 6 P_refit.npz (tips,loci,P).")
    ap.add_argument("--out", default="stage8_bootstrap.tsv", help="Output TSV.")

    ap.add_argument("--top", type=int, default=1000, help="Number of top pairs to test.")
    ap.add_argument("--rank-by", default=None,
                    help="Column to rank by (e.g. rMI_e, rlogOR, delta11). Default: rMI_* if exists else rlogOR else delta11.")
    ap.add_argument("--ascending", dest="descending", action="store_false",
                    help="Rank smallest first (default: largest first).")
    ap.add_argument("--descending", dest="descending", action="store_true", default=True,
                    help="Rank largest first (default).")
    ap.add_argument("--abs-rank", action="store_true", help="Rank by absolute value (useful for two-sided).")

    ap.add_argument("--B", type=int, default=2000, help="Bootstrap replicates per pair.")
    ap.add_argument("--boot-block", type=int, default=256, help="Bootstrap block size (replicates per chunk).")
    ap.add_argument("--threads", type=int, default=8, help="Threads over pairs.")
    ap.add_argument("--seed", type=int, default=12345, help="Base RNG seed.")
    ap.add_argument("--two-sided", action="store_true", help="Two-sided p-values (default: upper-tail).")

    ap.add_argument("--pc", type=float, default=0.5, help="Pseudocount (must match Stage 4/7).")
    ap.add_argument("--mi-base", choices=["e", "2", "10"], default=None,
                    help="MI log base (e/2/10). If not set, infer from header (MI_obs_*), else default e.")
    ap.add_argument("--min-distance", type=int, default=10000,
                    help="Ignore pairs with distance < this when selecting top pairs (default: 10000; set 0 to disable).")

    ap.add_argument("--progress-every", type=int, default=2000000, help="During scan, print every N lines.")
    args = ap.parse_args()

    t0 = time.perf_counter()

    # Load probabilities
    P, loci_used, tips_base = load_stage3_probs(args.p_hat, locus_fit=args.locus_fit)
    n_tips = int(P.shape[0])
    locus_to_col: Dict[int, int] = {int(l): i for i, l in enumerate(loci_used.tolist())}
    print(f"[info] Loaded P_hat: tips={n_tips}, loci_cols={P.shape[1]}", file=sys.stderr)

    # Optional overrides (Stage 6)
    P_ov, ov_locus_to_col, _ = load_override_probs(args.p_override, n_tips=n_tips, tips_base=tips_base)
    if args.p_override is not None:
        print(f"[info] Loaded P_override: loci={len(ov_locus_to_col)}", file=sys.stderr)

    # Select top pairs
    selected, used_rank, inferred_base = select_top_pairs(
        pairs_path=args.pairs,
        top_n=int(args.top),
        rank_by=args.rank_by,
        descending=bool(args.descending),
        abs_rank=bool(args.abs_rank),
        min_distance=int(args.min_distance),
        progress_every=int(args.progress_every),
    )

    # MI base (user override > inferred)
    mi_base = args.mi_base or inferred_base
    if mi_base not in {"e", "2", "10"}:
        mi_base = "e"

    print(f"[info] Selected {len(selected)} pairs for bootstrap (rank_by={used_rank}, mi_base={mi_base})", file=sys.stderr)

    if len(selected) == 0:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(
                "v\tw\tdistance\tn00\tn01\tn10\tn11"
                "\trank_by\trank_val"
                "\tdelta11_obs\tp_delta11\tz_delta11"
                "\trlogOR_obs\tp_rlogOR\tz_rlogOR"
                f"\trMI_obs_{mi_base}\tp_rMI\tz_rMI"
                "\tp_primary\tq_primary\tmissing\n"
            )
        print(f"[done] No pairs selected. Wrote header only: {args.out}", file=sys.stderr)
        return

    # Bootstrap in parallel
    base_ss = np.random.SeedSequence(int(args.seed))
    child_seeds = base_ss.spawn(len(selected))

    t1 = time.perf_counter()
    results: List[Tuple[int, Dict[str, float]]] = []

    def one(idx: int):
        rec = selected[idx]
        seed_int = int(child_seeds[idx].generate_state(1, dtype=np.uint64)[0])
        out = bootstrap_pair(
            rec=rec,
            P=P,
            locus_to_col=locus_to_col,
            P_ov=P_ov,
            ov_locus_to_col=ov_locus_to_col,
            B=int(args.B),
            boot_block=int(args.boot_block),
            pc=float(args.pc),
            mi_base=mi_base,
            two_sided=bool(args.two_sided),
            seed=seed_int,
        )
        return idx, out

    with ThreadPoolExecutor(max_workers=max(1, int(args.threads))) as ex:
        futs = [ex.submit(one, i) for i in range(len(selected))]
        done = 0
        for fut in as_completed(futs):
            idx, out = fut.result()
            results.append((idx, out))
            done += 1
            if done % max(1, min(100, len(selected))) == 0 or done == len(selected):
                dt = time.perf_counter() - t1
                print(f"[info] bootstrapped={done}/{len(selected)} rate={done/max(dt,1e-9):,.2f} pairs/s",
                      file=sys.stderr)

    results.sort(key=lambda x: x[0])
    out_rows = [r for _, r in results]

    # Primary p-value for BH
    if used_rank.startswith("rMI_"):
        p_primary = np.array([row["p_rMI"] for row in out_rows], dtype=np.float64)
    elif used_rank == "rlogOR":
        p_primary = np.array([row["p_rlogOR"] for row in out_rows], dtype=np.float64)
    elif used_rank == "delta11":
        p_primary = np.array([row["p_delta11"] for row in out_rows], dtype=np.float64)
    else:
        # fallback: rMI
        p_primary = np.array([row["p_rMI"] for row in out_rows], dtype=np.float64)

    q_primary = bh_qvalues(np.nan_to_num(p_primary, nan=1.0))

    # Write output
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(
            "v\tw\tdistance\tn00\tn01\tn10\tn11"
            "\trank_by\trank_val"
            "\tdelta11_obs\tp_delta11\tz_delta11"
            "\trlogOR_obs\tp_rlogOR\tz_rlogOR"
            f"\trMI_obs_{mi_base}\tp_rMI\tz_rMI"
            "\tp_primary\tq_primary\tmissing\n"
        )
        for rec, row, pp, qq in zip(selected, out_rows, p_primary.tolist(), q_primary.tolist()):
            f.write(
                f"{rec.i}\t{rec.j}\t{rec.distance}\t{rec.n00}\t{rec.n01}\t{rec.n10}\t{rec.n11}"
                f"\t{used_rank}\t{rec.rank_val:.10g}"
                f"\t{row['delta11_obs']:.10g}\t{row['p_delta11']:.10g}\t{row['z_delta11']:.10g}"
                f"\t{row['rlogOR_obs']:.10g}\t{row['p_rlogOR']:.10g}\t{row['z_rlogOR']:.10g}"
                f"\t{row['rMI_obs']:.10g}\t{row['p_rMI']:.10g}\t{row['z_rMI']:.10g}"
                f"\t{pp:.10g}\t{qq:.10g}\t{int(row['missing'])}\n"
            )

    dt = time.perf_counter() - t0
    print(f"[done] Wrote: {args.out}", file=sys.stderr)
    print(f"[done] elapsed={dt:.2f}s  tested={len(selected)}  B={args.B}  mi_base={mi_base}", file=sys.stderr)


if __name__ == "__main__":
    main()
