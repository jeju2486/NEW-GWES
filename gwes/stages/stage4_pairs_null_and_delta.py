#!/usr/bin/env python3
from __future__ import annotations

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
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
    compute_nulls_for_chunk,
)


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


# -------- override helpers (optional) --------

def _enforce_valid_joint(p1i: np.ndarray, p1j: np.ndarray, p11: np.ndarray):
    # minimal inline (keeps stage4 self-contained for overrides)
    p1i = np.clip(np.asarray(p1i, dtype=np.float64), 0.0, 1.0)
    p1j = np.clip(np.asarray(p1j, dtype=np.float64), 0.0, 1.0)
    p11 = np.asarray(p11, dtype=np.float64)
    lo = np.maximum(0.0, p1i + p1j - 1.0)
    hi = np.minimum(p1i, p1j)
    p11 = np.clip(p11, lo, hi)
    p10 = p1i - p11
    p01 = p1j - p11
    p00 = 1.0 - p10 - p01 - p11
    return (np.clip(p00, 0.0, 1.0),
            np.clip(p01, 0.0, 1.0),
            np.clip(p10, 0.0, 1.0),
            np.clip(p11, 0.0, 1.0))


def _compute_nulls_mixed(
    P_base: np.ndarray,
    locus_to_col: Dict[int, int],
    P_ov: Optional[np.ndarray],
    ov_locus_to_col: Dict[int, int],
    li: np.ndarray,
    lj: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loci_u, inv = np.unique(np.concatenate([li, lj]), return_inverse=True)
    inv_i = inv[:li.size]
    inv_j = inv[li.size:]

    n_tips = P_base.shape[0]
    M = loci_u.shape[0]
    U = np.empty((n_tips, M), dtype=np.float64)

    base_pos, base_cols = [], []
    ov_pos, ov_cols = [], []

    for j, loc in enumerate(loci_u.tolist()):
        oc = ov_locus_to_col.get(int(loc), None) if P_ov is not None else None
        if oc is not None:
            ov_pos.append(j); ov_cols.append(oc)
        else:
            bc = locus_to_col.get(int(loc), None)
            if bc is None:
                U[:, j] = np.nan
            else:
                base_pos.append(j); base_cols.append(bc)

    if base_cols:
        U[:, np.array(base_pos, dtype=np.int64)] = P_base[:, np.array(base_cols, dtype=np.int64)].astype(np.float64, copy=False)
    if ov_cols:
        U[:, np.array(ov_pos, dtype=np.int64)] = P_ov[:, np.array(ov_cols, dtype=np.int64)].astype(np.float64, copy=False)

    Xi = U[:, inv_i]
    Xj = U[:, inv_j]

    p1i = np.sum(Xi, axis=0, dtype=np.float64) / n_tips
    p1j = np.sum(Xj, axis=0, dtype=np.float64) / n_tips
    p11 = np.sum(Xi * Xj, axis=0, dtype=np.float64) / n_tips
    return _enforce_valid_joint(p1i, p1j, p11)


# -------- chunk worker --------

def process_chunk(
    chunk_id: int,
    lines: List[str],
    dist: List[str],
    mind: List[str],
    maxd: List[str],
    loci_i: np.ndarray,
    loci_j: np.ndarray,
    n00: np.ndarray,
    n01: np.ndarray,
    n10: np.ndarray,
    n11: np.ndarray,
    P: np.ndarray,
    locus_to_col: Dict[int, int],
    P_ov: Optional[np.ndarray],
    ov_locus_to_col: Dict[int, int],
    drop_missing: bool,
    pc: float,
    mi_base: str,
    want_minimal: bool,
) -> Tuple[int, str, Optional[str], int, int]:
    B = len(lines)

    missing = np.zeros(B, dtype=bool)
    for k in range(B):
        li = int(loci_i[k]); lj = int(loci_j[k])
        ok_i = (li in locus_to_col) or (li in ov_locus_to_col)
        ok_j = (lj in locus_to_col) or (lj in ov_locus_to_col)
        if not (ok_i and ok_j):
            missing[k] = True

    n = (n00 + n01 + n10 + n11).astype(np.float64)
    p11_obs = n11 / n

    logOR_obs = log_or_from_counts(n00, n01, n10, n11, pc=pc)
    o00, o01, o10, o11 = smooth_probs_from_counts(n00, n01, n10, n11, pc=pc)
    MI_obs = mutual_information_from_probs(o00, o01, o10, o11, base=mi_base)

    p00_null = np.full(B, np.nan, dtype=np.float64)
    p01_null = np.full(B, np.nan, dtype=np.float64)
    p10_null = np.full(B, np.nan, dtype=np.float64)
    p11_null = np.full(B, np.nan, dtype=np.float64)

    logOR_null = np.full(B, np.nan, dtype=np.float64)
    MI_null = np.full(B, np.nan, dtype=np.float64)
    srMI = np.full(B, np.nan, dtype=np.float64)

    nonmiss = ~missing
    if np.any(nonmiss):
        li_nm = loci_i[nonmiss].astype(np.int64)
        lj_nm = loci_j[nonmiss].astype(np.int64)

        if P_ov is None or len(ov_locus_to_col) == 0:
            ci = np.array([locus_to_col[int(x)] for x in li_nm.tolist()], dtype=np.int64)
            cj = np.array([locus_to_col[int(x)] for x in lj_nm.tolist()], dtype=np.int64)
            p00, p01, p10, p11 = compute_nulls_for_chunk(P, ci, cj)
        else:
            p00, p01, p10, p11 = _compute_nulls_mixed(P, locus_to_col, P_ov, ov_locus_to_col, li_nm, lj_nm)

        idx = np.where(nonmiss)[0]
        p00_null[idx] = p00
        p01_null[idx] = p01
        p10_null[idx] = p10
        p11_null[idx] = p11

        nn = n[idx]
        e00 = nn * p00
        e01 = nn * p01
        e10 = nn * p10
        e11 = nn * p11

        logOR_null[idx] = log_or_from_counts(e00, e01, e10, e11, pc=pc)

        q00, q01, q10, q11 = smooth_probs_from_counts(e00, e01, e10, e11, pc=pc)
        MI_null[idx] = mutual_information_from_probs(q00, q01, q10, q11, base=mi_base)

        p1i_null = q10 + q11
        p1j_null = q01 + q11
        MI_max = mi_max_given_marginals(p1i_null, p1j_null, base=mi_base)

        rMI_nm = (MI_obs[idx] - MI_null[idx])
        headroom = np.maximum(1e-12, MI_max - MI_null[idx])
        rMI_room = rMI_nm / headroom

        rlogOR_nm = (logOR_obs[idx] - logOR_null[idx])
        srMI[idx] = rMI_room * np.sign(rlogOR_nm)

    delta11 = p11_obs - p11_null
    rlogOR = logOR_obs - logOR_null
    rMI = MI_obs - MI_null

    out_lines: List[str] = []
    out_min: List[str] = [] if want_minimal else None
    written = 0

    for k in range(B):
        if missing[k] and drop_missing:
            continue

        out_lines.append(
            lines[k].rstrip("\n")
            + f"\t{p00_null[k]:.10g}\t{p01_null[k]:.10g}\t{p10_null[k]:.10g}\t{p11_null[k]:.10g}"
            + f"\t{delta11[k]:.10g}"
            + f"\t{logOR_obs[k]:.10g}\t{logOR_null[k]:.10g}\t{rlogOR[k]:.10g}"
            + f"\t{MI_obs[k]:.10g}\t{MI_null[k]:.10g}\t{rMI[k]:.10g}\t{srMI[k]:.10g}"
            + f"\t{int(missing[k])}\n"
        )

        if want_minimal:
            out_min.append(
                f"{int(loci_i[k])}\t{int(loci_j[k])}\t{dist[k]}\t{mind[k]}\t{maxd[k]}\t"
                f"{MI_obs[k]:.10g}\t{MI_null[k]:.10g}\t{srMI[k]:.10g}\n"
            )

        written += 1

    return chunk_id, "".join(out_lines), ("".join(out_min) if want_minimal else None), B, written


def stage4_pairs_null_and_delta(
    run_dir: str,
    out_name: str = "pairs_resid.tsv",
    minimal_out_name: Optional[str] = None,
    p_override: Optional[str] = None,
    chunk: int = 8192,
    threads: int = 16,
    inflight: int = 8,
    pc: float = 0.5,
    mi_base: str = "e",
    drop_missing: bool = False,
    progress_every: int = 200000,
) -> dict:
    run_dir = Path(run_dir)
    results = run_dir / "results"
    work3 = run_dir / "work" / "stage3"
    meta_dir = run_dir / "meta"
    results.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = results / "pairs_obs.tsv"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Stage 2 output not found: {pairs_path}")

    out_path = results / out_name
    out_min_path = (results / minimal_out_name) if minimal_out_name else None

    # Load P_hat (npz or memmap)
    P, loci_used, tips_base = load_p_hat_from_stage3_dir(work3, locus_fit_npz=(work3 / "locus_fit.npz"))
    locus_to_col: Dict[int, int] = {int(l): i for i, l in enumerate(loci_used.tolist())}
    print(f"[info] Loaded P_hat: tips={P.shape[0]} loci_cols={P.shape[1]}", file=sys.stderr)

    # Optional override
    P_ov = None
    ov_locus_to_col: Dict[int, int] = {}
    if p_override is not None:
        p_override = str(p_override)
        if not os.path.exists(p_override):
            print(f"[warn] --p-override not found: {p_override} (ignoring)", file=sys.stderr)
        else:
            oz = np.load(p_override, allow_pickle=True)
            if not all(k in oz.files for k in ["tips", "loci", "P"]):
                raise ValueError("P_refit.npz must contain tips, loci, P")
            tips_ov = [str(x) for x in oz["tips"].tolist()]
            if len(tips_ov) != P.shape[0]:
                raise ValueError("P_refit tips length != P_hat tips length")
            if tips_base is not None and tips_ov != tips_base:
                raise ValueError("Tip order mismatch between P_hat (npz) and P_refit. Do not mix runs.")
            P_ov = oz["P"].astype(np.float32, copy=False)
            loci_ov = oz["loci"].astype(np.int64, copy=False)
            ov_locus_to_col = {int(l): i for i, l in enumerate(loci_ov.tolist())}
            print(f"[info] Loaded override probs: {len(ov_locus_to_col)} loci", file=sys.stderr)

    # Progress
    t0 = time.perf_counter()
    processed = 0
    written_total = 0
    next_report = progress_every if progress_every > 0 else None

    pending: Dict[int, "object"] = {}
    next_write = 0

    def report():
        dt = time.perf_counter() - t0
        rate = processed / max(dt, 1e-9)
        print(f"[info] processed={processed:,} written={written_total:,} inflight={len(pending)} rate={rate:,.1f} pairs/s", file=sys.stderr)

    def write_ready(fout_full, fout_min):
        nonlocal next_write, written_total
        while next_write in pending and pending[next_write].done():
            _cid, out_txt, out_min_txt, _B, w = pending[next_write].result()
            fout_full.write(out_txt)
            if fout_min is not None and out_min_txt is not None:
                fout_min.write(out_min_txt)
            written_total += w
            del pending[next_write]
            next_write += 1

    with pairs_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        fout_min = out_min_path.open("w", encoding="utf-8") if out_min_path is not None else None

        header = fin.readline()
        if not header:
            raise ValueError("pairs_obs.tsv is empty.")

        delim = _detect_delim(header)
        cols = _split(header, delim)

        col_i = _find_col(cols, ["v", "unitig_i", "locus_i", "site_i", "idx_i", "i"])
        col_j = _find_col(cols, ["w", "unitig_j", "locus_j", "site_j", "idx_j", "j"])
        if col_i is None or col_j is None:
            raise ValueError("Could not find locus columns (need v/w).")

        col_dist = _find_col(cols, ["distance"])
        col_mind = _find_col(cols, ["min_distance", "mindistance"])
        col_maxd = _find_col(cols, ["max_distance", "maxdistance"])

        col_n00 = _find_col(cols, ["n00", "c00"])
        col_n01 = _find_col(cols, ["n01", "c01"])
        col_n10 = _find_col(cols, ["n10", "c10"])
        col_n11 = _find_col(cols, ["n11", "c11"])
        if None in (col_n00, col_n01, col_n10, col_n11):
            raise ValueError("Need n00,n01,n10,n11 in pairs_obs.tsv.")

        # Full header
        fout.write(header.rstrip("\n")
            + "\tp00_null\tp01_null\tp10_null\tp11_null"
            + "\tdelta11"
            + "\tlogOR_obs\tlogOR_null\trlogOR"
            + f"\tMI_obs_{mi_base}\tMI_null_{mi_base}\trMI_{mi_base}\tsrMI_{mi_base}"
            + "\tmissing_loci\n"
        )

        # Minimal header (optional)
        if fout_min is not None:
            fout_min.write(f"v\tw\tdistance\tmin_distance\tmax_distance\tMI_obs_{mi_base}\tMI_null_{mi_base}\tsrMI_{mi_base}\n")

        # Buffers
        chunk_id = 0
        lines_buf: List[str] = []
        dist_buf: List[str] = []
        mind_buf: List[str] = []
        maxd_buf: List[str] = []

        li_buf: List[int] = []
        lj_buf: List[int] = []
        n00_buf: List[float] = []
        n01_buf: List[float] = []
        n10_buf: List[float] = []
        n11_buf: List[float] = []

        def submit_current(ex):
            nonlocal chunk_id, processed
            if not lines_buf:
                return

            pending[chunk_id] = ex.submit(
                process_chunk,
                chunk_id,
                list(lines_buf),
                list(dist_buf),
                list(mind_buf),
                list(maxd_buf),
                np.array(li_buf, dtype=np.int64),
                np.array(lj_buf, dtype=np.int64),
                np.array(n00_buf, dtype=np.float64),
                np.array(n01_buf, dtype=np.float64),
                np.array(n10_buf, dtype=np.float64),
                np.array(n11_buf, dtype=np.float64),
                P,
                locus_to_col,
                P_ov,
                ov_locus_to_col,
                drop_missing,
                pc,
                mi_base,
                fout_min is not None,
            )

            processed += len(lines_buf)
            chunk_id += 1

            lines_buf.clear()
            dist_buf.clear()
            mind_buf.clear()
            maxd_buf.clear()
            li_buf.clear()
            lj_buf.clear()
            n00_buf.clear()
            n01_buf.clear()
            n10_buf.clear()
            n11_buf.clear()

        with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
            for line in fin:
                if not line.strip():
                    continue
                parts = _split(line, delim)
                if len(parts) < len(cols):
                    continue

                lines_buf.append(line)
                li_buf.append(int(parts[col_i]))
                lj_buf.append(int(parts[col_j]))

                # keep for minimal export (fall back to NA if absent)
                dist_buf.append(parts[col_dist] if col_dist is not None else "NA")
                mind_buf.append(parts[col_mind] if col_mind is not None else "NA")
                maxd_buf.append(parts[col_maxd] if col_maxd is not None else "NA")

                n00_buf.append(float(parts[col_n00]))
                n01_buf.append(float(parts[col_n01]))
                n10_buf.append(float(parts[col_n10]))
                n11_buf.append(float(parts[col_n11]))

                if len(lines_buf) >= chunk:
                    submit_current(ex)

                    while len(pending) >= max(1, inflight):
                        pending[next_write].result()
                        write_ready(fout, fout_min)

                    write_ready(fout, fout_min)

                    if next_report is not None and processed >= next_report:
                        report()
                        while processed >= next_report:
                            next_report += progress_every

            submit_current(ex)

            while pending:
                pending[next_write].result()
                write_ready(fout, fout_min)

        if fout_min is not None:
            fout_min.close()

    report()
    print(f"[done] Wrote: {out_path}", file=sys.stderr)
    if out_min_path is not None:
        print(f"[done] Wrote: {out_min_path}", file=sys.stderr)

    meta = {
        "stage": "stage4",
        "inputs": {
            "pairs_obs": str(pairs_path),
            "p_hat_dir": str(work3),
            "p_override": str(p_override) if p_override else None,
        },
        "params": {
            "chunk": int(chunk),
            "threads": int(threads),
            "inflight": int(inflight),
            "pc": float(pc),
            "mi_base": mi_base,
            "drop_missing": bool(drop_missing),
            "progress_every": int(progress_every),
        },
        "outputs": {
            "pairs_resid": str(out_path),
            "pairs_resid_min": str(out_min_path) if out_min_path else None,
        },
    }
    write_stage_meta(run_dir / "meta" / "stage4.json", meta)
    return meta


def main():
    ap = argparse.ArgumentParser("Stage 4: structured-mixture null + residual scores (run-dir)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="pairs_resid.tsv")
    ap.add_argument("--minimal-out", default=None, help="Optional minimal export file name (e.g. pairs_resid.min.tsv)")
    ap.add_argument("--p-override", default=None, help="Optional P_refit.npz (tips,loci,P) overriding P_hat for those loci.")

    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--inflight", type=int, default=8)

    ap.add_argument("--pc", type=float, default=0.5)
    ap.add_argument("--mi-base", choices=["e", "2", "10"], default="e")
    ap.add_argument("--drop-missing", action="store_true")
    ap.add_argument("--progress-every", type=int, default=200000)
    args = ap.parse_args()

    meta = stage4_pairs_null_and_delta(
        run_dir=args.run_dir,
        out_name=args.out,
        minimal_out_name=args.minimal_out,
        p_override=args.p_override,
        chunk=args.chunk,
        threads=args.threads,
        inflight=args.inflight,
        pc=args.pc,
        mi_base=args.mi_base,
        drop_missing=args.drop_missing,
        progress_every=args.progress_every,
    )
    print(f"[ok] stage4 wrote: {meta['outputs']['pairs_resid']}")


if __name__ == "__main__":
    main()
