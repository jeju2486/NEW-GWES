#!/usr/bin/env python3
"""
stage8_bootstrap_top_pairs.py  (Stage 8)

Stage 8 — Significance for top pairs (parametric bootstrap under structured null)

Run-dir refactor:
- Preferred: --run-dir
  Defaults:
    pairs:   RUN_DIR/work/stage7/pairs_resid_patched.tsv if exists else RUN_DIR/work/stage4/pairs_resid.tsv
    stage3:  RUN_DIR/work/stage3
    override (optional): RUN_DIR/work/stage6/P_refit.npz if exists
    out:     RUN_DIR/work/stage8/stage8_bootstrap.tsv
    meta:    RUN_DIR/meta/stage8.json

Minimal output:
- --minimal writes a small TSV with only:
    v, w, distance, rank_by, rank_val, stat_obs, p_primary, q_primary, missing

Notes:
- Requires pairs file to contain observed counts n00,n01,n10,n11 and ranking column.
- Bootstrap is under per-tip conditional independence using p_i(t), p_j(t).
"""

from __future__ import annotations

import argparse
import heapq
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from gwes.manifest import write_stage_meta
from gwes.prob_store import load_p_hat_from_stage3_dir
from gwes.pair_stats import (
    log_or_from_counts,
    smooth_probs_from_counts,
    mutual_information_from_probs,
    mi_max_given_marginals,
)


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
# Run-dir path resolution
# -------------------------

def _resolve_stage3_dir(run_dir: Path, stage3_arg: Optional[str]) -> Path:
    if stage3_arg is not None:
        p = Path(stage3_arg)
        return p if p.is_absolute() else (run_dir / p)
    return run_dir / "work" / "stage3"

def _resolve_pairs(run_dir: Path, pairs_arg: Optional[str]) -> Path:
    if pairs_arg is not None:
        p = Path(pairs_arg)
        return p if p.is_absolute() else (run_dir / p)
    cand7 = run_dir / "work" / "stage7" / "pairs_resid_patched.tsv"
    if cand7.exists():
        return cand7
    cand4 = run_dir / "work" / "stage4" / "pairs_resid.tsv"
    return cand4

def _resolve_override(run_dir: Path, ov_arg: Optional[str]) -> Optional[Path]:
    if ov_arg is not None:
        p = Path(ov_arg)
        p = p if p.is_absolute() else (run_dir / p)
        return p
    cand = run_dir / "work" / "stage6" / "P_refit.npz"
    return cand if cand.exists() else None

def _resolve_out(run_dir: Path, out_arg: Optional[str]) -> Path:
    if out_arg is not None:
        p = Path(out_arg)
        return p if p.is_absolute() else (run_dir / p)
    return run_dir / "work" / "stage8" / "stage8_bootstrap.tsv"


# -------------------------
# Optional override probs
# -------------------------

def load_override_probs(path: Path, n_tips: int, tips_base: Optional[List[str]]):
    """
    Returns:
      P_ov: (n_tips, n_refit) float32
      ov_map: locus->col dict
    """
    z = np.load(str(path), allow_pickle=True)
    if not all(k in z.files for k in ["tips", "loci", "P"]):
        raise ValueError("P_refit.npz must contain tips,loci,P")
    tips_ov = [str(x) for x in z["tips"].tolist()]
    if len(tips_ov) != n_tips:
        raise ValueError("P_refit tips length != n_tips")
    if tips_base is not None and tips_ov != tips_base:
        raise ValueError("Tip order mismatch between P_hat (npz) and P_refit.npz")
    P_ov = z["P"].astype(np.float32, copy=False)
    loci_ov = z["loci"].astype(np.int64, copy=False)
    if P_ov.shape != (n_tips, loci_ov.shape[0]):
        raise ValueError("P_refit shape mismatch")
    ov_map = {int(l): i for i, l in enumerate(loci_ov.tolist())}
    return P_ov, ov_map


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

def _get_prob_vec(
    locus: int,
    P: np.ndarray,
    locus_to_col: Dict[int, int],
    P_ov: Optional[np.ndarray],
    ov_locus_to_col: Dict[int, int],
) -> Optional[np.ndarray]:
    if P_ov is not None:
        oc = ov_locus_to_col.get(int(locus), None)
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
    compute_srmi: bool,
) -> Dict[str, float]:
    pi = _get_prob_vec(rec.i, P, locus_to_col, P_ov, ov_locus_to_col)
    pj = _get_prob_vec(rec.j, P, locus_to_col, P_ov, ov_locus_to_col)
    if pi is None or pj is None:
        return {"missing": 1.0}

    pi = np.clip(pi, 0.0, 1.0)
    pj = np.clip(pj, 0.0, 1.0)
    n_tips = int(pi.shape[0])

    # deterministic null mixture
    p1i = float(np.mean(pi))
    p1j = float(np.mean(pj))
    p11_null = float(np.mean(pi * pj))
    p11_null = float(np.clip(p11_null, 0.0, 1.0))
    p10_null = float(np.clip(p1i - p11_null, 0.0, 1.0))
    p01_null = float(np.clip(p1j - p11_null, 0.0, 1.0))
    p00_null = float(np.clip(1.0 - p1i - p1j + p11_null, 0.0, 1.0))

    # observed stats from counts
    n00 = float(rec.n00); n01 = float(rec.n01); n10 = float(rec.n10); n11 = float(rec.n11)
    n = n00 + n01 + n10 + n11
    if n <= 0:
        return {"missing": 0.0}

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

    # srMI headroom correction (optional)
    srMI_obs = np.nan
    headroom = np.nan
    if compute_srmi:
        p1i_null = q10 + q11
        p1j_null = q01 + q11
        MI_max = mi_max_given_marginals(p1i_null, p1j_null, base=mi_base)
        headroom = float(np.maximum(1e-12, MI_max - MI_null))
        srMI_obs = float((rMI_obs / headroom) * np.sign(rlogOR_obs))

    # per-tip joint probs for sampling
    p11t = pi * pj
    p10t = pi * (1.0 - pj)
    p01t = (1.0 - pi) * pj
    p00t = (1.0 - pi) * (1.0 - pj)

    t0 = np.clip(p00t.astype(np.float32, copy=False), 0.0, 1.0)
    t1 = np.clip((p00t + p01t).astype(np.float32, copy=False), 0.0, 1.0)
    t2 = np.clip((p00t + p01t + p10t).astype(np.float32, copy=False), 0.0, 1.0)

    rng = np.random.default_rng(seed)

    delta11_b = np.empty(B, dtype=np.float64)
    rlogOR_b = np.empty(B, dtype=np.float64)
    rMI_b = np.empty(B, dtype=np.float64)
    srMI_b = np.empty(B, dtype=np.float64) if compute_srmi else None

    done = 0
    while done < B:
        bb = min(int(boot_block), B - done)
        u = rng.random((bb, n_tips), dtype=np.float32)

        m0 = (u < t0).sum(axis=1).astype(np.float64)
        m1 = (u < t1).sum(axis=1).astype(np.float64)
        m2 = (u < t2).sum(axis=1).astype(np.float64)

        n00b = m0
        n01b = m1 - m0
        n10b = m2 - m1
        n11b = n_tips - m2

        p11b = n11b / n_tips
        d11 = p11b - p11_null
        delta11_b[done:done + bb] = d11

        logORb = log_or_from_counts(n00b, n01b, n10b, n11b, pc=pc)
        rlo = logORb - logOR_null
        rlogOR_b[done:done + bb] = rlo

        ob00, ob01, ob10, ob11 = smooth_probs_from_counts(n00b, n01b, n10b, n11b, pc=pc)
        MIb = mutual_information_from_probs(ob00, ob01, ob10, ob11, base=mi_base)
        rmi = MIb - MI_null
        rMI_b[done:done + bb] = rmi

        if compute_srmi and srMI_b is not None:
            srMI_b[done:done + bb] = (rmi / headroom) * np.sign(rlo)

        done += bb

    def pval(boot: np.ndarray, obs: float) -> float:
        if not np.isfinite(obs):
            return np.nan
        if two_sided:
            return float((1 + np.sum(np.abs(boot) >= abs(obs))) / (boot.size + 1))
        return float((1 + np.sum(boot >= obs)) / (boot.size + 1))

    def zscore(boot: np.ndarray, obs: float) -> float:
        if not np.isfinite(obs):
            return np.nan
        mu = float(np.mean(boot))
        sd = float(np.std(boot, ddof=1))
        return float((obs - mu) / sd) if sd > 0 else np.nan

    out = {
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
    if compute_srmi and srMI_b is not None:
        out.update({
            "srMI_obs": float(srMI_obs),
            "p_srMI": pval(srMI_b, srMI_obs),
            "z_srMI": zscore(srMI_b, srMI_obs),
        })
    return out


# -------------------------
# Stage 8 driver
# -------------------------

def stage8_bootstrap_top_pairs(
    run_dir: str,
    pairs_path: Optional[str] = None,
    stage3_dir: Optional[str] = None,
    p_override: Optional[str] = None,
    out_path: Optional[str] = None,
    top: int = 1000,
    rank_by: Optional[str] = None,
    descending: bool = True,
    abs_rank: bool = False,
    B: int = 2000,
    boot_block: int = 256,
    threads: int = 8,
    seed: int = 12345,
    two_sided: bool = False,
    pc: float = 0.5,
    mi_base: Optional[str] = None,
    min_distance: int = 10000,
    progress_every: int = 2000000,
    minimal: bool = False,
) -> dict:
    run_dir_p = Path(run_dir)
    stage3_p = _resolve_stage3_dir(run_dir_p, stage3_dir)
    pairs_p = _resolve_pairs(run_dir_p, pairs_path)
    ov_p = _resolve_override(run_dir_p, p_override)
    out_p = _resolve_out(run_dir_p, out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    (run_dir_p / "meta").mkdir(parents=True, exist_ok=True)
    meta_path = run_dir_p / "meta" / "stage8.json"

    if not pairs_p.exists():
        raise FileNotFoundError(f"pairs TSV not found: {pairs_p}")
    if not stage3_p.exists():
        raise FileNotFoundError(f"stage3 dir not found: {stage3_p}")

    # Load P_hat from Stage3
    P, loci_used, tips_base = load_p_hat_from_stage3_dir(stage3_p, locus_fit_npz=(stage3_p / "locus_fit.npz"))
    n_tips = int(P.shape[0])
    locus_to_col: Dict[int, int] = {int(l): i for i, l in enumerate(loci_used.tolist())}
    print(f"[info] Loaded P_hat: tips={n_tips}, loci_cols={P.shape[1]}", file=sys.stderr)

    # Optional overrides
    P_ov = None
    ov_locus_to_col: Dict[int, int] = {}
    if ov_p is not None and ov_p.exists():
        P_ov, ov_locus_to_col = load_override_probs(ov_p, n_tips=n_tips, tips_base=tips_base)
        print(f"[info] Loaded P_override: loci={len(ov_locus_to_col)} from {ov_p}", file=sys.stderr)

    # Scan file and keep top-N
    t0 = time.perf_counter()
    heap: List[Tuple[float, int, PairRec]] = []
    scanned = 0
    tie = 0

    with open(pairs_p, "r", encoding="utf-8") as fin:
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

        # MI base inference
        if mi_base is None:
            mi_cols = [c for c in cols if c.startswith("MI_obs_")]
            if len(mi_cols) == 1:
                mi_base = mi_cols[0].split("MI_obs_")[-1]
        if mi_base is None:
            mi_base = "e"
        if mi_base not in {"e", "2", "10"}:
            raise ValueError(f"Invalid mi_base: {mi_base}")

        # rank column selection (prefer srMI, then rMI, then rlogOR, then delta11)
        if rank_by is None:
            srmi_cols = [c for c in cols if c.startswith("srMI_")]
            if len(srmi_cols) == 1:
                rank_by = srmi_cols[0]
            else:
                rmi_cols = [c for c in cols if c.startswith("rMI_")]
                if len(rmi_cols) == 1:
                    rank_by = rmi_cols[0]
                elif "rlogOR" in cols:
                    rank_by = "rlogOR"
                elif "delta11" in cols:
                    rank_by = "delta11"
                else:
                    raise ValueError("Could not auto-select rank column; pass --rank-by explicitly.")
        if rank_by not in cols:
            raise ValueError(f"--rank-by not found in header: {rank_by}")
        col_rank = cols.index(rank_by)

        for line in fin:
            if not line.strip():
                continue
            parts = _split(line, delim)
            if len(parts) < ncols:
                continue

            scanned += 1

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

            # apply min-distance only if distance is known (>=0)
            if min_distance > 0 and dist >= 0 and dist < min_distance:
                continue

            rank_metric = abs(rv) if abs_rank else rv
            key = rank_metric if descending else -rank_metric  # larger key = better

            rec = PairRec(i=i, j=j, distance=dist, n00=n00, n01=n01, n10=n10, n11=n11, rank_val=rv)

            tie += 1
            if len(heap) < int(top):
                heapq.heappush(heap, (key, tie, rec))
            else:
                if key > heap[0][0]:
                    heapq.heapreplace(heap, (key, tie, rec))

            if progress_every > 0 and (scanned % int(progress_every)) == 0:
                dt = time.perf_counter() - t0
                print(f"[info] scanned={scanned:,} kept={len(heap)} rate={scanned/max(dt,1e-9):,.1f} lines/s", file=sys.stderr)

    selected = [r for _, __, r in heap]
    selected.sort(
        key=lambda r: (abs(r.rank_val) if abs_rank else r.rank_val),
        reverse=descending
    )

    print(f"[info] Selected top {len(selected)} pairs for bootstrap (rank_by={rank_by})", file=sys.stderr)

    # Bootstrap in parallel
    compute_srmi = bool(rank_by.startswith("srMI_")) or (not minimal)  # minimal still needs srMI if ranked by srMI
    base_ss = np.random.SeedSequence(int(seed))
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
            B=int(B),
            boot_block=int(boot_block),
            pc=float(pc),
            mi_base=str(mi_base),
            two_sided=bool(two_sided),
            seed=seed_int,
            compute_srmi=compute_srmi,
        )
        return idx, out

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as ex:
        futs = [ex.submit(one, i) for i in range(len(selected))]
        done = 0
        for fut in as_completed(futs):
            idx, out = fut.result()
            results.append((idx, out))
            done += 1
            if done % max(1, min(100, len(selected))) == 0 or done == len(selected):
                dt = time.perf_counter() - t1
                print(f"[info] bootstrapped={done}/{len(selected)} rate={done/max(dt,1e-9):,.2f} pairs/s", file=sys.stderr)

    results.sort(key=lambda x: x[0])
    out_rows = [r for _, r in results]

    # Primary p-value for BH
    if rank_by.startswith("srMI_"):
        p_primary = np.array([row.get("p_srMI", np.nan) for row in out_rows], dtype=np.float64)
        stat_obs = np.array([row.get("srMI_obs", np.nan) for row in out_rows], dtype=np.float64)
    elif rank_by.startswith("rMI_"):
        p_primary = np.array([row.get("p_rMI", np.nan) for row in out_rows], dtype=np.float64)
        stat_obs = np.array([row.get("rMI_obs", np.nan) for row in out_rows], dtype=np.float64)
    elif rank_by == "rlogOR":
        p_primary = np.array([row.get("p_rlogOR", np.nan) for row in out_rows], dtype=np.float64)
        stat_obs = np.array([row.get("rlogOR_obs", np.nan) for row in out_rows], dtype=np.float64)
    elif rank_by == "delta11":
        p_primary = np.array([row.get("p_delta11", np.nan) for row in out_rows], dtype=np.float64)
        stat_obs = np.array([row.get("delta11_obs", np.nan) for row in out_rows], dtype=np.float64)
    else:
        p_primary = np.array([row.get("p_rMI", np.nan) for row in out_rows], dtype=np.float64)
        stat_obs = np.array([row.get("rMI_obs", np.nan) for row in out_rows], dtype=np.float64)

    q_primary = bh_qvalues(np.nan_to_num(p_primary, nan=1.0))

    # Write output
    with open(out_p, "w", encoding="utf-8") as f:
        if minimal:
            f.write("v\tw\tdistance\trank_by\trank_val\tstat_obs\tp_primary\tq_primary\tmissing\n")
            for rec, row, pp, qq, so in zip(selected, out_rows, p_primary.tolist(), q_primary.tolist(), stat_obs.tolist()):
                f.write(
                    f"{rec.i}\t{rec.j}\t{rec.distance}\t{rank_by}\t{rec.rank_val:.10g}\t"
                    f"{so:.10g}\t{pp:.10g}\t{qq:.10g}\t{int(row.get('missing', 1.0))}\n"
                )
        else:
            # Full output
            # include srMI columns if computed
            has_srmi = compute_srmi
            f.write(
                "v\tw\tdistance\tn00\tn01\tn10\tn11"
                "\trank_by\trank_val"
                "\tdelta11_obs\tp_delta11\tz_delta11"
                "\trlogOR_obs\tp_rlogOR\tz_rlogOR"
                f"\trMI_obs_{mi_base}\tp_rMI\tz_rMI"
            )
            if has_srmi:
                f.write(f"\tsrMI_obs_{mi_base}\tp_srMI\tz_srMI")
            f.write("\tp_primary\tq_primary\tmissing\n")

            for rec, row, pp, qq in zip(selected, out_rows, p_primary.tolist(), q_primary.tolist()):
                f.write(
                    f"{rec.i}\t{rec.j}\t{rec.distance}\t{rec.n00}\t{rec.n01}\t{rec.n10}\t{rec.n11}"
                    f"\t{rank_by}\t{rec.rank_val:.10g}"
                    f"\t{row.get('delta11_obs', np.nan):.10g}\t{row.get('p_delta11', np.nan):.10g}\t{row.get('z_delta11', np.nan):.10g}"
                    f"\t{row.get('rlogOR_obs', np.nan):.10g}\t{row.get('p_rlogOR', np.nan):.10g}\t{row.get('z_rlogOR', np.nan):.10g}"
                    f"\t{row.get('rMI_obs', np.nan):.10g}\t{row.get('p_rMI', np.nan):.10g}\t{row.get('z_rMI', np.nan):.10g}"
                )
                if has_srmi:
                    f.write(
                        f"\t{row.get('srMI_obs', np.nan):.10g}\t{row.get('p_srMI', np.nan):.10g}\t{row.get('z_srMI', np.nan):.10g}"
                    )
                f.write(f"\t{pp:.10g}\t{qq:.10g}\t{int(row.get('missing', 1.0))}\n")

    dt = time.perf_counter() - t0
    print(f"[done] Wrote: {out_p}", file=sys.stderr)
    print(f"[done] elapsed={dt:.2f}s  scanned={scanned:,}  tested={len(selected)}  B={B}  mi_base={mi_base}  rank_by={rank_by}", file=sys.stderr)

    meta = {
        "stage": "stage8",
        "inputs": {
            "pairs": str(pairs_p),
            "stage3_dir": str(stage3_p),
            "p_override": str(ov_p) if ov_p is not None else None,
        },
        "params": {
            "top": int(top),
            "rank_by": str(rank_by),
            "descending": bool(descending),
            "abs_rank": bool(abs_rank),
            "B": int(B),
            "boot_block": int(boot_block),
            "threads": int(threads),
            "seed": int(seed),
            "two_sided": bool(two_sided),
            "pc": float(pc),
            "mi_base": str(mi_base),
            "min_distance": int(min_distance),
            "minimal": bool(minimal),
        },
        "outputs": {"bootstrap_tsv": str(out_p)},
    }
    write_stage_meta(meta_path, meta)
    return meta


def main():
    ap = argparse.ArgumentParser(description="Stage 8: bootstrap p-values for top pairs under structured null (run-dir native).")
    ap.add_argument("--run-dir", required=True)

    ap.add_argument("--pairs", default=None, help="Pairs TSV. Default: work/stage7/pairs_resid_patched.tsv else work/stage4/pairs_resid.tsv")
    ap.add_argument("--stage3-dir", default=None, help="Stage3 dir. Default: work/stage3")
    ap.add_argument("--p-override", default=None, help="Optional P_refit.npz. Default: auto if work/stage6/P_refit.npz exists")
    ap.add_argument("--out", default=None, help="Output TSV. Default: work/stage8/stage8_bootstrap.tsv")
    ap.add_argument("--minimal", action="store_true", help="Write minimal output only (simple TSV).")

    ap.add_argument("--top", type=int, default=1000)
    ap.add_argument("--rank-by", default=None)
    ap.add_argument("--ascending", dest="descending", action="store_false")
    ap.add_argument("--descending", dest="descending", action="store_true", default=True)
    ap.add_argument("--abs-rank", action="store_true")

    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--boot-block", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--two-sided", action="store_true")

    ap.add_argument("--pc", type=float, default=0.5)
    ap.add_argument("--mi-base", choices=["e", "2", "10"], default=None)
    ap.add_argument("--min-distance", type=int, default=10000)
    ap.add_argument("--progress-every", type=int, default=2000000)

    args = ap.parse_args()

    meta = stage8_bootstrap_top_pairs(
        run_dir=args.run_dir,
        pairs_path=args.pairs,
        stage3_dir=args.stage3_dir,
        p_override=args.p_override,
        out_path=args.out,
        top=args.top,
        rank_by=args.rank_by,
        descending=args.descending,
        abs_rank=args.abs_rank,
        B=args.B,
        boot_block=args.boot_block,
        threads=args.threads,
        seed=args.seed,
        two_sided=args.two_sided,
        pc=args.pc,
        mi_base=args.mi_base,
        min_distance=args.min_distance,
        progress_every=args.progress_every,
        minimal=args.minimal,
    )
    print(f"[ok] stage8 wrote: {meta['outputs']['bootstrap_tsv']}")


if __name__ == "__main__":
    main()