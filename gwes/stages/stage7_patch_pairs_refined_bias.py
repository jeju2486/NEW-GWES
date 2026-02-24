#!/usr/bin/env python3
from __future__ import annotations

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------
# Small helpers
# --------------------------

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

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------
# Stats: logOR and MI (robust, consistent with Stage 4)
# --------------------------

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

def mutual_information_from_probs(p00: np.ndarray, p01: np.ndarray, p10: np.ndarray, p11: np.ndarray, base: str) -> np.ndarray:
    """
    MI = sum p_ab log( p_ab / (p_a. p_.b) ), with eps guards.
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

def mi_max_given_marginals(p1i: np.ndarray, p1j: np.ndarray, base: str) -> np.ndarray:
    """
    Maximum MI achievable for two Bernoulli variables with fixed marginals.
    Achieved at p11 boundary L or U.
    """
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


# --------------------------
# Load Stage 3 P_hat (npz or memmap meta)  — same behavior as Stage 4/7 earlier
# --------------------------

def load_stage3_probs(path: str, locus_fit: Optional[str] = None):
    """
    Returns:
      P: ndarray or memmap, shape (n_tips, n_loci_used)
      loci: ndarray int64, shape (n_loci_used,)
      tips: optional list[str] if present (npz only), else None
    """
    # Robust fallback: user passed missing P_hat_meta.json but Stage3 wrote P_hat.npz
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


# --------------------------
# Vectorized locus->column mapping (base + override) + consistent joint enforcement
# --------------------------

def _make_sorted_mapper(loci: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a mapper for vectorized searchsorted mapping.
    Returns:
      loci_sorted: sorted loci values
      col_of_sorted_pos: for each position in loci_sorted, the ORIGINAL column index
    """
    loci = np.asarray(loci, dtype=np.int64)
    order = np.argsort(loci)
    return loci[order], order.astype(np.int64)

def _map_loci_to_cols(query: np.ndarray, loci_sorted: np.ndarray, col_of_sorted_pos: np.ndarray) -> np.ndarray:
    """
    Vectorized mapping; returns cols (int64) with -1 for missing.
    """
    query = np.asarray(query, dtype=np.int64)
    pos = np.searchsorted(loci_sorted, query)
    cols = np.full(query.shape[0], -1, dtype=np.int64)
    ok = (pos >= 0) & (pos < loci_sorted.shape[0])
    pos_ok = pos[ok]
    ok2 = ok.copy()
    ok2[ok] = (loci_sorted[pos_ok] == query[ok])
    cols[ok2] = col_of_sorted_pos[pos[ok2]]
    return cols

def _enforce_valid_joint(p1i: np.ndarray, p1j: np.ndarray, p11: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Enforce a valid joint consistent with marginals:
      p11 in [max(0,p1i+p1j-1), min(p1i,p1j)]
      p10=p1i-p11, p01=p1j-p11, p00=1-p10-p01-p11
    """
    p1i = np.clip(np.asarray(p1i, dtype=np.float64), 0.0, 1.0)
    p1j = np.clip(np.asarray(p1j, dtype=np.float64), 0.0, 1.0)
    p11 = np.asarray(p11, dtype=np.float64)

    lo = np.maximum(0.0, p1i + p1j - 1.0)
    hi = np.minimum(p1i, p1j)
    p11 = np.clip(p11, lo, hi)

    p10 = p1i - p11
    p01 = p1j - p11
    p00 = 1.0 - p10 - p01 - p11

    # minor numeric cleanup
    p00 = np.clip(p00, 0.0, 1.0)
    p01 = np.clip(p01, 0.0, 1.0)
    p10 = np.clip(p10, 0.0, 1.0)
    p11 = np.clip(p11, 0.0, 1.0)
    return p00, p01, p10, p11

def compute_nulls_mixed_batch(
    P_base: np.ndarray,
    base_loci_sorted: np.ndarray,
    base_col_of_sorted: np.ndarray,
    li: np.ndarray,
    lj: np.ndarray,
    P_refit: np.ndarray,
    ov_loci_sorted: np.ndarray,
    ov_col_of_sorted: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For arrays li, lj (length B), compute p00,p01,p10,p11 plus missing mask.
    Mixed means: for each locus, use override column if present; else base.
    """
    li = np.asarray(li, dtype=np.int64)
    lj = np.asarray(lj, dtype=np.int64)
    B = li.size

    # map to base / override cols
    bi = _map_loci_to_cols(li, base_loci_sorted, base_col_of_sorted)
    bj = _map_loci_to_cols(lj, base_loci_sorted, base_col_of_sorted)
    oi = _map_loci_to_cols(li, ov_loci_sorted, ov_col_of_sorted)
    oj = _map_loci_to_cols(lj, ov_loci_sorted, ov_col_of_sorted)

    ci = np.where(oi >= 0, oi, bi)
    cj = np.where(oj >= 0, oj, bj)
    missing = (ci < 0) | (cj < 0)

    p00 = np.full(B, np.nan, dtype=np.float64)
    p01 = np.full(B, np.nan, dtype=np.float64)
    p10 = np.full(B, np.nan, dtype=np.float64)
    p11 = np.full(B, np.nan, dtype=np.float64)

    idx = np.where(~missing)[0]
    if idx.size == 0:
        return p00, p01, p10, p11, missing

    # Unique loci for this batch (nonmissing only), build U once
    li_nm = li[idx]
    lj_nm = lj[idx]
    loci_u, inv = np.unique(np.concatenate([li_nm, lj_nm]), return_inverse=True)
    inv_i = inv[:li_nm.size]
    inv_j = inv[li_nm.size:]

    # Map unique loci to base/ov cols, prefer override
    bu = _map_loci_to_cols(loci_u, base_loci_sorted, base_col_of_sorted)
    ou = _map_loci_to_cols(loci_u, ov_loci_sorted, ov_col_of_sorted)
    cu = np.where(ou >= 0, ou, bu)

    # Fill U
    n_tips = P_base.shape[0]
    U = np.empty((n_tips, loci_u.size), dtype=np.float64)

    use_ov = (ou >= 0)
    if np.any(~use_ov):
        cols_base = cu[~use_ov]
        U[:, ~use_ov] = P_base[:, cols_base].astype(np.float64, copy=False)
    if np.any(use_ov):
        cols_ov = cu[use_ov]
        U[:, use_ov] = P_refit[:, cols_ov].astype(np.float64, copy=False)

    Xi = U[:, inv_i]
    Xj = U[:, inv_j]

    p1i = np.sum(Xi, axis=0, dtype=np.float64) / n_tips
    p1j = np.sum(Xj, axis=0, dtype=np.float64) / n_tips
    p11_nm = np.sum(Xi * Xj, axis=0, dtype=np.float64) / n_tips

    p00_nm, p01_nm, p10_nm, p11_nm = _enforce_valid_joint(p1i, p1j, p11_nm)

    p00[idx] = p00_nm
    p01[idx] = p01_nm
    p10[idx] = p10_nm
    p11[idx] = p11_nm
    return p00, p01, p10, p11, missing


# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 7: patch Stage 4 output only for pairs involving refit loci.")
    ap.add_argument("--in", dest="inp", required=True, help="Stage 4 output TSV (to patch).")
    ap.add_argument("--out", required=True, help="Patched output TSV.")

    ap.add_argument("--p-hat", required=True, help="Stage 3 P_hat.npz or P_hat_meta.json (or directory).")
    ap.add_argument("--locus-fit", default=None, help="Stage 3 locus_fit.npz (required if using memmap meta).")
    ap.add_argument("--basis", required=True, help="Stage 3 phylo_basis.npz (contains Z and tips).")
    ap.add_argument("--patch", required=True, help="Stage 6 refit_patch.npz (alpha_refit,b_refit,loci_refit,tips,K).")

    ap.add_argument("--pc", type=float, default=0.5, help="Pseudocount (MUST match Stage 4).")
    ap.add_argument("--block", type=int, default=200000, help="Lines per streaming block.")
    ap.add_argument("--progress-every", type=int, default=2000000, help="Progress print every N lines read.")
    ap.add_argument("--prefit-block", type=int, default=512, help="Compute P_refit in blocks of this many loci.")
    args = ap.parse_args()

    # Load basis (Z, tips)
    bz = np.load(args.basis, allow_pickle=True)
    if "Z" not in bz.files or "tips" not in bz.files:
        raise ValueError("basis must contain Z and tips")
    Z = bz["Z"].astype(np.float64, copy=False)
    tips_basis = [str(x) for x in bz["tips"].tolist()]
    n_tips = Z.shape[0]
    if len(tips_basis) != n_tips:
        raise ValueError("basis tips length != Z rows")

    # Load Stage 6 patch params
    pz = np.load(args.patch, allow_pickle=True)
    for k in ["loci_refit", "alpha_refit", "b_refit", "tips", "K"]:
        if k not in pz.files:
            raise ValueError(f"patch missing '{k}'")

    tips_patch = [str(x) for x in pz["tips"].tolist()]
    if tips_patch != tips_basis:
        raise ValueError("Tip order mismatch between patch and basis.")

    loci_refit = pz["loci_refit"].astype(np.int64, copy=False)
    alpha_refit = pz["alpha_refit"].astype(np.float64, copy=False)
    b_refit = pz["b_refit"].astype(np.float64, copy=False)
    K_patch = int(pz["K"])
    if Z.shape[1] != K_patch or b_refit.shape[1] != K_patch:
        raise ValueError("K mismatch between patch and basis.")

    # If no refit loci: just copy
    if loci_refit.size == 0:
        with open(args.inp, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
            for line in fin:
                fout.write(line)
        print(f"[info] patch has 0 loci. Copied {args.inp} -> {args.out}", file=sys.stderr)
        return

    # Build P_refit (n_tips x n_refit) from patch params, in blocks
    m_refit = int(loci_refit.size)
    P_refit = np.empty((n_tips, m_refit), dtype=np.float32)
    blk = max(1, int(args.prefit_block))
    for s in range(0, m_refit, blk):
        e = min(m_refit, s + blk)
        E = Z @ b_refit[s:e, :].T
        E += alpha_refit[None, s:e]
        P_refit[:, s:e] = sigmoid(E).astype(np.float32, copy=False)

    # Sorted mapper for overrides (IMPORTANT: keep columns consistent with P_refit)
    ov_loci_sorted, ov_col_of_sorted = _make_sorted_mapper(loci_refit)

    refit_set = set(int(x) for x in loci_refit.tolist())
    print(f"[info] Refitted loci: {m_refit} (patching pairs involving these loci)", file=sys.stderr)

    # Load Stage 3 P_hat
    P_base, loci_used, tips_base = load_stage3_probs(args.p_hat, locus_fit=args.locus_fit)
    if P_base.shape[0] != n_tips:
        raise ValueError("n_tips mismatch between P_hat and basis.")
    if tips_base is not None and tips_base != tips_basis:
        raise ValueError("Tip order mismatch between P_hat.npz and basis.")

    base_loci_sorted, base_col_of_sorted = _make_sorted_mapper(loci_used)

    t0 = time.perf_counter()
    n_read = 0
    n_patched = 0

    with open(args.inp, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        header = fin.readline()
        if not header:
            raise ValueError("Empty input file.")
        delim = _detect_delim(header)
        cols = _split(header, delim)
        fout.write(header)

        # Required indices
        col_i = _find_col(cols, ["v", "unitig_i", "locus_i", "site_i", "idx_i", "i"])
        col_j = _find_col(cols, ["w", "unitig_j", "locus_j", "site_j", "idx_j", "j"])
        if col_i is None or col_j is None:
            raise ValueError("Could not find locus columns (v/w or unitig_i/unitig_j etc.).")

        col_n00 = _find_col(cols, ["n00", "c00"])
        col_n01 = _find_col(cols, ["n01", "c01"])
        col_n10 = _find_col(cols, ["n10", "c10"])
        col_n11 = _find_col(cols, ["n11", "c11"])
        if None in (col_n00, col_n01, col_n10, col_n11):
            raise ValueError("Need n00,n01,n10,n11 (or c00,c01,c10,c11).")

        # Columns to patch (must exist)
        c_p00 = _find_col(cols, ["p00_null"])
        c_p01 = _find_col(cols, ["p01_null"])
        c_p10 = _find_col(cols, ["p10_null"])
        c_p11 = _find_col(cols, ["p11_null"])
        c_d11 = _find_col(cols, ["delta11"])
        c_lorn = _find_col(cols, ["logor_null"])
        c_rlo = _find_col(cols, ["rlogor"])
        c_miss = _find_col(cols, ["missing_loci"])

        mi_null_cols = [k for k, name in enumerate(cols) if name.startswith("MI_null_")]
        rmi_cols = [k for k, name in enumerate(cols) if name.startswith("rMI_")]
        if len(mi_null_cols) != 1 or len(rmi_cols) != 1:
            raise ValueError("Expected exactly one MI_null_* and one rMI_* column in Stage 4 output.")
        c_mi_null = mi_null_cols[0]
        c_rmi = rmi_cols[0]
        mi_base = cols[c_mi_null].split("MI_null_")[-1]
        if mi_base not in {"e", "2", "10"}:
            raise ValueError(f"Unrecognized MI base suffix in header: {mi_base}")

        # Optional srMI_* column (patched using Stage 4 headroom definition)
        srmi_cols = [k for k, name in enumerate(cols) if name.startswith("srMI_")]
        c_srmi = srmi_cols[0] if len(srmi_cols) == 1 else None

        needed = [c_p00, c_p01, c_p10, c_p11, c_d11, c_lorn, c_rlo, c_miss, c_mi_null, c_rmi]
        if any(x is None for x in needed):
            raise ValueError("Stage 4 output missing required patchable columns.")

        # Stream in blocks
        while True:
            lines: List[str] = []
            for _ in range(args.block):
                line = fin.readline()
                if not line:
                    break
                lines.append(line)
            if not lines:
                break

            n_read += len(lines)

            # Only store parsed fields for affected lines
            affected_idx: List[int] = []
            parts_store: List[List[str]] = []  # aligned with affected_idx

            li: List[int] = []
            lj: List[int] = []
            n00: List[float] = []
            n01: List[float] = []
            n10: List[float] = []
            n11: List[float] = []

            for idx, line in enumerate(lines):
                if not line.strip():
                    continue
                parts = _split(line, delim)
                if len(parts) < len(cols):
                    continue
                try:
                    i = int(parts[col_i]); j = int(parts[col_j])
                except ValueError:
                    continue
                if (i in refit_set) or (j in refit_set):
                    affected_idx.append(idx)
                    parts_store.append(parts)
                    li.append(i); lj.append(j)
                    n00.append(float(parts[col_n00]))
                    n01.append(float(parts[col_n01]))
                    n10.append(float(parts[col_n10]))
                    n11.append(float(parts[col_n11]))

            if affected_idx:
                li_a = np.array(li, dtype=np.int64)
                lj_a = np.array(lj, dtype=np.int64)
                n00_a = np.array(n00, dtype=np.float64)
                n01_a = np.array(n01, dtype=np.float64)
                n10_a = np.array(n10, dtype=np.float64)
                n11_a = np.array(n11, dtype=np.float64)

                # Recompute nulls for affected only
                p00_null, p01_null, p10_null, p11_null, missing = compute_nulls_mixed_batch(
                    P_base=P_base,
                    base_loci_sorted=base_loci_sorted,
                    base_col_of_sorted=base_col_of_sorted,
                    li=li_a,
                    lj=lj_a,
                    P_refit=P_refit,
                    ov_loci_sorted=ov_loci_sorted,
                    ov_col_of_sorted=ov_col_of_sorted,
                )

                # Recompute null-dependent stats
                n_tot = (n00_a + n01_a + n10_a + n11_a).astype(np.float64)
                p11_obs = n11_a / n_tot
                delta11 = p11_obs - p11_null

                e00 = n_tot * p00_null
                e01 = n_tot * p01_null
                e10 = n_tot * p10_null
                e11 = n_tot * p11_null

                logOR_null = log_or_from_counts(e00, e01, e10, e11, pc=args.pc)
                logOR_obs = log_or_from_counts(n00_a, n01_a, n10_a, n11_a, pc=args.pc)
                rlogOR = logOR_obs - logOR_null

                # MI with same smoothing scheme as Stage 4
                q00, q01, q10, q11 = smooth_probs_from_counts(e00, e01, e10, e11, pc=args.pc)
                MI_null = mutual_information_from_probs(q00, q01, q10, q11, base=mi_base)

                o00, o01, o10, o11 = smooth_probs_from_counts(n00_a, n01_a, n10_a, n11_a, pc=args.pc)
                MI_obs = mutual_information_from_probs(o00, o01, o10, o11, base=mi_base)
                rMI = MI_obs - MI_null

                # srMI (Stage 4 headroom definition), only if column exists
                srMI = None
                if c_srmi is not None:
                    p1i_null = q10 + q11
                    p1j_null = q01 + q11
                    MI_max = mi_max_given_marginals(p1i_null, p1j_null, base=mi_base)
                    headroom = np.maximum(1e-12, MI_max - MI_null)
                    rMI_room = (MI_obs - MI_null) / headroom
                    srMI = rMI_room * np.sign(rlogOR)

                # Patch lines back
                for t, line_idx in enumerate(affected_idx):
                    parts = parts_store[t]

                    if missing[t]:
                        parts[c_miss] = "1"
                        lines[line_idx] = ("\t".join(parts) + "\n") if delim == "\t" else (" ".join(parts) + "\n")
                        continue

                    parts[c_p00] = f"{p00_null[t]:.10g}"
                    parts[c_p01] = f"{p01_null[t]:.10g}"
                    parts[c_p10] = f"{p10_null[t]:.10g}"
                    parts[c_p11] = f"{p11_null[t]:.10g}"
                    parts[c_d11] = f"{delta11[t]:.10g}"
                    parts[c_lorn] = f"{logOR_null[t]:.10g}"
                    parts[c_rlo] = f"{rlogOR[t]:.10g}"
                    parts[c_mi_null] = f"{MI_null[t]:.10g}"
                    parts[c_rmi] = f"{rMI[t]:.10g}"
                    if c_srmi is not None and srMI is not None:
                        parts[c_srmi] = f"{srMI[t]:.10g}"
                    parts[c_miss] = "0"

                    lines[line_idx] = ("\t".join(parts) + "\n") if delim == "\t" else (" ".join(parts) + "\n")

                n_patched += len(affected_idx)

            for line in lines:
                fout.write(line)

            if args.progress_every > 0 and (n_read % args.progress_every) < args.block:
                dt = time.perf_counter() - t0
                rate = n_read / max(dt, 1e-9)
                print(f"[info] read={n_read:,} patched={n_patched:,} rate={rate:,.1f} lines/s", file=sys.stderr)

    dt = time.perf_counter() - t0
    print(f"[done] Patched pairs: {n_patched:,} / lines read: {n_read:,}  elapsed={dt:.2f}s", file=sys.stderr)
    print(f"[done] Wrote: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
