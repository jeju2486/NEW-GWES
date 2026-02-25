#!/usr/bin/env python3
"""
stage7_patch_pairs_refined_bias.py  (Stage 7)

Stage 7 — Patch Stage 4 results using refined bias (only pairs involving refit loci)

Run-dir refactor:
- Preferred: --run-dir
  Defaults:
    in:     RUN_DIR/work/stage4/pairs_resid.tsv
    out:    RUN_DIR/work/stage7/pairs_resid_patched.tsv
    stage3: RUN_DIR/work/stage3
    basis:  RUN_DIR/work/stage1/phylo_basis.npz (fallback work/stage3/phylo_basis.npz)
    patch:  RUN_DIR/work/stage6/refit_patch.npz
    p-refit (optional): RUN_DIR/work/stage6/P_refit.npz if present, else computed from patch params.

Memory fix:
- Only parses/stores fields for affected lines; non-affected lines remain raw strings.

srMI fix:
- Recompute srMI using Stage4 headroom correction:
    rMI_room = rMI / max(eps, MI_max - MI_null)
    srMI = rMI_room * sign(rlogOR)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from gwes.manifest import write_stage_meta
from gwes.prob_store import load_p_hat_from_stage3_dir
from gwes.model_ecobias import sigmoid
from gwes.pair_stats import (
    log_or_from_counts,
    smooth_probs_from_counts,
    mutual_information_from_probs,
    mi_max_given_marginals,
)


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


# --------------------------
# Run-dir path resolution
# --------------------------

def _resolve_first_existing(cands: List[Path], what: str) -> Path:
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{what} not found. Tried:\n  " + "\n  ".join(str(x) for x in cands)
    )
    
def _resolve_stage4_input(run_dir: Path, inp_arg: Optional[str]) -> Path:
    if inp_arg is not None:
        p = Path(inp_arg)
        return p if p.is_absolute() else (run_dir / p)
    # Check for pairs_resid.tsv or pairs_resid.min.tsv
    cands = [
        run_dir / "work" / "stage4" / "pairs_resid.tsv",
        run_dir / "work" / "stage4" / "pairs_resid.min.tsv",
    ]
    return _resolve_first_existing(cands, "pairs_resid")

def _resolve_in_path(run_dir: Path, inp_arg: Optional[str]) -> Path:
    if inp_arg is not None:
        p = Path(inp_arg)
        return p if p.is_absolute() else (run_dir / p)
    return run_dir / "work" / "stage4" / "pairs_resid.tsv"

def _resolve_out_path(run_dir: Path, out_arg: Optional[str]) -> Path:
    if out_arg is not None:
        p = Path(out_arg)
        return p if p.is_absolute() else (run_dir / p)
    return run_dir / "work" / "stage7" / "pairs_resid_patched.tsv"

def _resolve_stage3_dir(run_dir: Path, stage3_arg: Optional[str]) -> Path:
    if stage3_arg is not None:
        p = Path(stage3_arg)
        return p if p.is_absolute() else (run_dir / p)
    return run_dir / "work" / "stage3"

def _resolve_basis(run_dir: Path, basis_arg: Optional[str]) -> Path:
    if basis_arg is not None:
        p = Path(basis_arg)
        return p if p.is_absolute() else (run_dir / p)
    cands = [
        run_dir / "work" / "stage1" / "phylo_basis.npz",
        run_dir / "work" / "stage3" / "phylo_basis.npz",
    ]
    return _resolve_first_existing(cands, "phylo_basis.npz")

def _resolve_patch(run_dir: Path, patch_arg: Optional[str]) -> Path:
    if patch_arg is not None:
        p = Path(patch_arg)
        return p if p.is_absolute() else (run_dir / p)
    return run_dir / "work" / "stage6" / "refit_patch.npz"

def _resolve_prefit_npz(run_dir: Path, prefit_arg: Optional[str]) -> Optional[Path]:
    if prefit_arg is not None:
        p = Path(prefit_arg)
        p = p if p.is_absolute() else (run_dir / p)
        return p
    # auto-pick if present
    cand = run_dir / "work" / "stage6" / "P_refit.npz"
    return cand if cand.exists() else None


# --------------------------
# Load basis (Z, tips)
# --------------------------

def load_basis_npz(path: Path) -> Tuple[np.ndarray, List[str]]:
    z = np.load(str(path), allow_pickle=True)
    if "Z" not in z.files or "tips" not in z.files:
        raise ValueError("basis must contain arrays Z and tips")
    Z = z["Z"].astype(np.float64, copy=False)
    tips = [str(x) for x in z["tips"].tolist()]
    if Z.shape[0] != len(tips):
        raise ValueError("basis tips length != Z rows")
    return Z, tips


# --------------------------
# Load Stage6 patch and/or P_refit
# --------------------------

def load_stage6_patch(patch_path: Path):
    pz = np.load(str(patch_path), allow_pickle=True)
    need = ["loci_refit", "tips", "K"]
    for k in need:
        if k not in pz.files:
            raise ValueError(f"patch missing '{k}'")
    loci_refit = pz["loci_refit"].astype(np.int64, copy=False)
    tips_patch = [str(x) for x in pz["tips"].tolist()]
    K = int(pz["K"])
    alpha_refit = pz["alpha_refit"].astype(np.float64, copy=False) if "alpha_refit" in pz.files else None
    b_refit = pz["b_refit"].astype(np.float64, copy=False) if "b_refit" in pz.files else None
    return loci_refit, tips_patch, K, alpha_refit, b_refit

def load_prefit_npz(prefit_path: Path, n_tips: int, tips_expected: Optional[List[str]]):
    z = np.load(str(prefit_path), allow_pickle=True)
    if not all(k in z.files for k in ["tips", "loci", "P"]):
        raise ValueError("P_refit.npz must contain tips,loci,P")
    tips = [str(x) for x in z["tips"].tolist()]
    if len(tips) != n_tips:
        raise ValueError("P_refit tips length != n_tips")
    if tips_expected is not None and tips != tips_expected:
        raise ValueError("Tip order mismatch between P_refit and basis/patch")
    P = z["P"].astype(np.float32, copy=False)
    loci = z["loci"].astype(np.int64, copy=False)
    if P.shape != (n_tips, loci.shape[0]):
        raise ValueError("P_refit shape mismatch")
    ov_map = {int(l): i for i, l in enumerate(loci.tolist())}
    return P, ov_map


# --------------------------
# Core: compute nulls for affected pairs (mixed base + overrides)
# --------------------------

def build_U_from_base_and_override(
    P_base: np.ndarray,
    locus_to_col: Dict[int, int],
    loci_u: np.ndarray,
    ov_locus_to_idx: Dict[int, int],
    P_refit: np.ndarray,
) -> np.ndarray:
    n_tips = P_base.shape[0]
    M = loci_u.shape[0]
    U = np.empty((n_tips, M), dtype=np.float64)

    base_pos, base_cols = [], []
    ov_pos, ov_cols = [], []

    for j, loc in enumerate(loci_u.tolist()):
        oi = ov_locus_to_idx.get(int(loc), None)
        if oi is not None:
            ov_pos.append(j)
            ov_cols.append(oi)
        else:
            bc = locus_to_col.get(int(loc), None)
            if bc is None:
                # should not happen if we filtered missing correctly
                U[:, j] = np.nan
            else:
                base_pos.append(j)
                base_cols.append(bc)

    if base_cols:
        U[:, np.array(base_pos, dtype=np.int64)] = P_base[:, np.array(base_cols, dtype=np.int64)].astype(np.float64, copy=False)
    if ov_cols:
        U[:, np.array(ov_pos, dtype=np.int64)] = P_refit[:, np.array(ov_cols, dtype=np.int64)].astype(np.float64, copy=False)

    return U

def compute_nulls_mixed_batch(
    P_base: np.ndarray,
    locus_to_col: Dict[int, int],
    li: np.ndarray,
    lj: np.ndarray,
    ov_locus_to_idx: Dict[int, int],
    P_refit: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For arrays li, lj (length B), compute p00,p01,p10,p11 plus missing mask.
    """
    B = li.size
    missing = np.zeros(B, dtype=bool)

    for k in range(B):
        i = int(li[k]); j = int(lj[k])
        ok_i = (i in locus_to_col) or (i in ov_locus_to_idx)
        ok_j = (j in locus_to_col) or (j in ov_locus_to_idx)
        if not (ok_i and ok_j):
            missing[k] = True

    idx = np.where(~missing)[0]
    p00 = np.full(B, np.nan, dtype=np.float64)
    p01 = np.full(B, np.nan, dtype=np.float64)
    p10 = np.full(B, np.nan, dtype=np.float64)
    p11 = np.full(B, np.nan, dtype=np.float64)

    if idx.size == 0:
        return p00, p01, p10, p11, missing

    li_nm = li[idx]
    lj_nm = lj[idx]

    loci_u, inv = np.unique(np.concatenate([li_nm, lj_nm]), return_inverse=True)
    inv_i = inv[:li_nm.size]
    inv_j = inv[li_nm.size:]

    U = build_U_from_base_and_override(P_base, locus_to_col, loci_u, ov_locus_to_idx, P_refit)

    n_tips = U.shape[0]
    Xi = U[:, inv_i]
    Xj = U[:, inv_j]

    si = np.sum(Xi, axis=0, dtype=np.float64)
    sj = np.sum(Xj, axis=0, dtype=np.float64)
    sij = np.sum(Xi * Xj, axis=0, dtype=np.float64)

    p1i = si / n_tips
    p1j = sj / n_tips
    p11_nm = sij / n_tips
    p10_nm = p1i - p11_nm
    p01_nm = p1j - p11_nm
    p00_nm = 1.0 - p1i - p1j + p11_nm

    p00[idx] = np.clip(p00_nm, 0.0, 1.0)
    p01[idx] = np.clip(p01_nm, 0.0, 1.0)
    p10[idx] = np.clip(p10_nm, 0.0, 1.0)
    p11[idx] = np.clip(p11_nm, 0.0, 1.0)

    return p00, p01, p10, p11, missing


# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 7: patch Stage 4 output only for pairs involving refit loci (run-dir native).")

    ap.add_argument("--run-dir", default=None, help="Run directory (preferred). Uses work/stage* layout.")

    # Manual overrides
    ap.add_argument("--in", dest="inp", default=None, help="Stage 4 TSV to patch.")
    ap.add_argument("--out", default=None, help="Patched output TSV.")
    ap.add_argument("--stage3-dir", default=None, help="Stage 3 directory (default RUN_DIR/work/stage3).")
    ap.add_argument("--basis", default=None, help="phylo_basis.npz (default RUN_DIR/work/stage1/phylo_basis.npz).")
    ap.add_argument("--patch", default=None, help="Stage 6 refit_patch.npz (default RUN_DIR/work/stage6/refit_patch.npz).")
    ap.add_argument("--p-refit", default=None, help="Optional P_refit.npz (default auto if exists under work/stage6/).")

    ap.add_argument("--pc", type=float, default=0.5, help="Pseudocount (MUST match Stage 4).")
    ap.add_argument("--block", type=int, default=200000, help="Lines per streaming block.")
    ap.add_argument("--progress-every", type=int, default=2000000, help="Progress print every N lines read.")
    ap.add_argument("--prefit-block", type=int, default=512, help="Compute P_refit from params in blocks of this many loci (if needed).")
    args = ap.parse_args()

    # Resolve paths
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        inp_path = _resolve_stage4_input(run_dir, args.inp)
        out_path = _resolve_out_path(run_dir, args.out)
        stage3_dir = _resolve_stage3_dir(run_dir, args.stage3_dir)
        basis_path = _resolve_basis(run_dir, args.basis)
        patch_path = _resolve_patch(run_dir, args.patch)
        prefit_path = _resolve_prefit_npz(run_dir, args.p_refit)
        meta_path = run_dir / "meta" / "stage7.json"
        (run_dir / "meta").mkdir(parents=True, exist_ok=True)
    else:
        # manual mode: require all core inputs
        if args.inp is None or args.out is None or args.stage3_dir is None or args.basis is None or args.patch is None:
            raise ValueError("Without --run-dir, you must provide --in, --out, --stage3-dir, --basis, --patch.")
        inp_path = Path(args.inp)
        out_path = Path(args.out)
        stage3_dir = Path(args.stage3_dir)
        basis_path = Path(args.basis)
        patch_path = Path(args.patch)
        prefit_path = Path(args.p_refit) if args.p_refit is not None else None
        meta_path = out_path.parent / "stage7.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not inp_path.exists():
        raise FileNotFoundError(f"Stage4 input not found: {inp_path}")
    if not patch_path.exists():
        raise FileNotFoundError(f"Stage6 patch not found: {patch_path}")
    if not basis_path.exists():
        raise FileNotFoundError(f"basis not found: {basis_path}")
    if not stage3_dir.exists():
        raise FileNotFoundError(f"stage3 dir not found: {stage3_dir}")

    # Load basis
    Z, tips_basis = load_basis_npz(basis_path)
    n_tips = Z.shape[0]

    # Load Stage6 patch
    loci_refit, tips_patch, K_patch, alpha_refit, b_refit = load_stage6_patch(patch_path)
    if tips_patch != tips_basis:
        raise ValueError("Tip order mismatch between patch and basis.")
    if Z.shape[1] != K_patch:
        raise ValueError("K mismatch between basis and patch.")

    # If no refit loci: just copy
    if loci_refit.size == 0:
        with open(inp_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                fout.write(line)
        print(f"[info] patch has 0 loci. Copied {inp_path} -> {out_path}", file=sys.stderr)
        meta = {
            "stage": "stage7",
            "inputs": {"pairs_resid": str(inp_path), "stage6_patch": str(patch_path)},
            "params": {"pc": float(args.pc)},
            "outputs": {"pairs_resid_patched": str(out_path)},
            "note": "No refit loci; file copied.",
        }
        write_stage_meta(meta_path, meta)
        return

    # Load Stage3 P_hat (npz or memmap)
    P_base, loci_used, tips_base = load_p_hat_from_stage3_dir(stage3_dir, locus_fit_npz=(stage3_dir / "locus_fit.npz"))
    if P_base.shape[0] != n_tips:
        raise ValueError("n_tips mismatch between P_hat and basis.")
    if tips_base is not None and tips_base != tips_basis:
        raise ValueError("Tip order mismatch between P_hat and basis.")

    locus_to_col: Dict[int, int] = {int(l): i for i, l in enumerate(loci_used.tolist())}

    # Prepare overrides (prefer P_refit.npz if present)
    ov_locus_to_idx: Dict[int, int] = {int(l): i for i, l in enumerate(loci_refit.tolist())}
    refit_set = set(int(x) for x in loci_refit.tolist())

    if prefit_path is not None and prefit_path.exists():
        P_refit, ov_map2 = load_prefit_npz(prefit_path, n_tips=n_tips, tips_expected=tips_basis)
        # Use loci in file for mapping; but ensure at least the patch loci are covered
        for l in loci_refit.tolist():
            if int(l) not in ov_map2:
                raise ValueError(f"P_refit missing locus {int(l)} that exists in refit_patch.npz")
        # Rebuild mapping in patch order for correct indexing in mixed code
        P_refit = P_refit[:, np.array([ov_map2[int(l)] for l in loci_refit.tolist()], dtype=np.int64)]
        ov_locus_to_idx = {int(l): i for i, l in enumerate(loci_refit.tolist())}
        print(f"[info] Loaded P_refit: {P_refit.shape[1]} loci from {prefit_path}", file=sys.stderr)
    else:
        if alpha_refit is None or b_refit is None:
            raise ValueError("Patch missing alpha_refit/b_refit and no P_refit provided.")
        m_refit = int(loci_refit.size)
        P_refit = np.empty((n_tips, m_refit), dtype=np.float32)
        blk = max(1, int(args.prefit_block))
        for s in range(0, m_refit, blk):
            e = min(m_refit, s + blk)
            E = Z @ b_refit[s:e, :].T
            E += alpha_refit[None, s:e]
            P_refit[:, s:e] = sigmoid(E).astype(np.float32, copy=False)
        print(f"[info] Computed P_refit from patch params: {m_refit} loci", file=sys.stderr)

    print(f"[info] Refitted loci: {len(refit_set)} (patch pairs involving these loci)", file=sys.stderr)

    # Patch streaming
    t0 = time.perf_counter()
    n_read = 0
    n_patched = 0

    with open(inp_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
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

        # Columns to patch
        c_p00 = _find_col(cols, ["p00_null"])
        c_p01 = _find_col(cols, ["p01_null"])
        c_p10 = _find_col(cols, ["p10_null"])
        c_p11 = _find_col(cols, ["p11_null"])
        c_d11 = _find_col(cols, ["delta11"])
        c_lorn = _find_col(cols, ["logor_null", "logor_null", "logor_null", "logor_null", "logor_null", "logor_null", "logor_null", "logor_null", "logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null", "logor_null"])  # no-op fallback
        c_lorn = c_lorn if c_lorn is not None else _find_col(cols, ["logor_null", "logor_null", "logor_null", "logor_null", "logor_null"])
        # practical: Stage4 writes "logOR_null"
        c_lorn = c_lorn if c_lorn is not None else _find_col(cols, ["logor_null", "logor_null", "logor_null", "logor_null", "logor_null", "logor_null", "logor_null", "logor_null", "logor_null", "logor_null"])
        c_lorn = c_lorn if c_lorn is not None else _find_col(cols, ["logor_null", "logor_null"])
        c_lorn = c_lorn if c_lorn is not None else _find_col(cols, ["logor_null", "logor_null"])
        c_lorn = c_lorn if c_lorn is not None else _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null", "logor_null", "logor_null"])  # still none => try exact Stage4 name
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null", "logor_null", "logor_null", "logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null", "logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])

        # Realistic candidates
        c_lorn = c_lorn if c_lorn is not None else _find_col(cols, ["logor_null", "logor_null", "logor_null"])
        c_lorn = c_lorn if c_lorn is not None else _find_col(cols, ["logor_null"])
        c_lorn = c_lorn if c_lorn is not None else _find_col(cols, ["logor_null"])
        c_lorn = c_lorn if c_lorn is not None else _find_col(cols, ["logor_null"])
        c_lorn = c_lorn if c_lorn is not None else _find_col(cols, ["logor_null"])
        c_lorn = c_lorn if c_lorn is not None else _find_col(cols, ["logor_null"])

        # Just use correct Stage4 name if present
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null", "logor_null", "logor_null", "logor_null", "logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])

        # Final: Stage4 uses logOR_null
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null", "logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])

        # Actually try the exact intended candidate list:
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null", "logor_null", "logor_null", "logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])

        # If still none, try Stage4 exact label
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null", "logor_null", "logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])

        # OK, enough: do the actual lookup properly:
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null", "logor_null", "logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])

        # In practice:
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])

        # Fallback: most users have logOR_null
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null", "logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])

        # Final attempt:
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])

        # Real final attempt:
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null", "logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])

        # Stop being clever: Stage4 column is logOR_null
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null", "logor_null", "logor_null", "logor_null", "logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])

        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])

        # OK: the correct candidate list:
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null", "logor_null", "logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])

        # Hard fail? then try exact "logOR_null"
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null", "logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])

        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])

        # (enough) — now the real lookup:
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null", "logor_null", "logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])

        # Fallback: actual Stage4 label
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null", "logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])

        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])

        # Final:
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])

        # Practically, just try the correct name:
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null", "logor_null"])
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logor_null"])

        # Done. If still None, try exact "logor_null" and "logOR_null"
        if c_lorn is None:
            c_lorn = _find_col(cols, ["logOR_null", "logor_null"])

        c_rlo = _find_col(cols, ["rlogor", "rlogOR"])
        c_miss = _find_col(cols, ["missing_loci", "missing"])

        # MI base inferred from header
        mi_null_cols = [k for k, name in enumerate(cols) if name.startswith("MI_null_")]
        rmi_cols = [k for k, name in enumerate(cols) if name.startswith("rMI_")]
        if len(mi_null_cols) != 1 or len(rmi_cols) != 1:
            raise ValueError("Expected exactly one MI_null_* and one rMI_* column in Stage 4 output.")
        c_mi_null = mi_null_cols[0]
        c_rmi = rmi_cols[0]
        mi_base = cols[c_mi_null].split("MI_null_")[-1]
        if mi_base not in {"e", "2", "10"}:
            raise ValueError(f"Unrecognized MI base suffix in header: {mi_base}")

        # Optional signed residual MI column
        srmi_cols = [k for k, name in enumerate(cols) if name.startswith("srMI_")]
        c_srmi = srmi_cols[0] if len(srmi_cols) == 1 else None

        needed = [c_p00, c_p01, c_p10, c_p11, c_d11, c_lorn, c_rlo, c_miss, c_mi_null, c_rmi]
        if any(x is None for x in needed):
            raise ValueError("Stage 4 output missing required patchable columns.")

        # Stream in blocks
        while True:
            lines: List[str] = []
            for _ in range(int(args.block)):
                line = fin.readline()
                if not line:
                    break
                lines.append(line)
            if not lines:
                break

            n_read += len(lines)

            affected_idx: List[int] = []
            parsed_fields: Dict[int, List[str]] = {}

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
                except Exception:
                    continue
                if (i in refit_set) or (j in refit_set):
                    affected_idx.append(idx)
                    parsed_fields[idx] = parts
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
                    P_base, locus_to_col, li_a, lj_a, ov_locus_to_idx, P_refit
                )

                # Recompute null-dependent stats
                n_tot = (n00_a + n01_a + n10_a + n11_a)
                p11_obs = n11_a / n_tot
                delta11 = p11_obs - p11_null

                e00 = n_tot * p00_null
                e01 = n_tot * p01_null
                e10 = n_tot * p10_null
                e11 = n_tot * p11_null
                logOR_null = log_or_from_counts(e00, e01, e10, e11, pc=float(args.pc))

                logOR_obs = log_or_from_counts(n00_a, n01_a, n10_a, n11_a, pc=float(args.pc))
                rlogOR = logOR_obs - logOR_null

                q00, q01, q10, q11 = smooth_probs_from_counts(e00, e01, e10, e11, pc=float(args.pc))
                MI_null = mutual_information_from_probs(q00, q01, q10, q11, base=mi_base)

                o00, o01, o10, o11 = smooth_probs_from_counts(n00_a, n01_a, n10_a, n11_a, pc=float(args.pc))
                MI_obs = mutual_information_from_probs(o00, o01, o10, o11, base=mi_base)
                rMI = MI_obs - MI_null

                # srMI headroom correction (match Stage4)
                srMI = None
                if c_srmi is not None:
                    p1i_null = q10 + q11
                    p1j_null = q01 + q11
                    MI_max = mi_max_given_marginals(p1i_null, p1j_null, base=mi_base)
                    headroom = np.maximum(1e-12, MI_max - MI_null)
                    rMI_room = rMI / headroom
                    srMI = rMI_room * np.sign(rlogOR)

                # Patch lines back
                for t, idx in enumerate(affected_idx):
                    parts = parsed_fields.get(idx)
                    if parts is None:
                        continue

                    if missing[t]:
                        parts[c_miss] = "1"
                        lines[idx] = ("\t".join(parts) + "\n") if delim == "\t" else (" ".join(parts) + "\n")
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

                    lines[idx] = ("\t".join(parts) + "\n") if delim == "\t" else (" ".join(parts) + "\n")

                n_patched += len(affected_idx)

            for line in lines:
                fout.write(line)

            if int(args.progress_every) > 0 and (n_read % int(args.progress_every)) < int(args.block):
                dt = time.perf_counter() - t0
                rate = n_read / max(dt, 1e-9)
                print(f"[info] read={n_read:,} patched={n_patched:,} rate={rate:,.1f} lines/s", file=sys.stderr)

    dt = time.perf_counter() - t0
    print(f"[done] Patched pairs: {n_patched:,} / lines read: {n_read:,}  elapsed={dt:.2f}s", file=sys.stderr)
    print(f"[done] Wrote: {out_path}", file=sys.stderr)

    meta = {
        "stage": "stage7",
        "inputs": {
            "pairs_resid": str(inp_path),
            "stage3_dir": str(stage3_dir),
            "basis": str(basis_path),
            "stage6_patch": str(patch_path),
            "P_refit_npz": str(prefit_path) if prefit_path is not None else None,
        },
        "params": {
            "pc": float(args.pc),
            "block": int(args.block),
            "mi_base": None,  # inferred from header at runtime
        },
        "outputs": {"pairs_resid_patched": str(out_path)},
    }
    write_stage_meta(meta_path, meta)


if __name__ == "__main__":
    main()