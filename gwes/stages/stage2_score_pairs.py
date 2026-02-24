#!/usr/bin/env python3
from __future__ import annotations

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from tqdm import tqdm

from gwes.fasta_matrix import load_locusmajor_npz
from gwes.bits import popcount_lut_u8, valid_mask_bytes
from gwes.manifest import write_stage_meta


def _count_pairs_tsv(pairs_path: Path, max_pairs: Optional[int] = None) -> int:
    n = 0
    with pairs_path.open() as fh:
        header = True
        for ln in fh:
            if header:
                header = False
                continue
            ln = ln.strip()
            if not ln:
                continue
            n += 1
            if max_pairs is not None and n >= max_pairs:
                break
    return n


def _existing_chunks_info(chunk_dir: Path) -> Tuple[int, int]:
    """
    Returns:
      pairs_done: total data lines already present across chunks
      next_chunk_idx: next chunk index to write
    """
    chunk_files = sorted(chunk_dir.glob("pairs_obs.chunk_*.tsv"))
    if not chunk_files:
        return 0, 1

    pairs_done = 0
    max_idx = 0
    for f in chunk_files:
        try:
            idx = int(f.stem.split("_")[-1])
            max_idx = max(max_idx, idx)
        except Exception:
            continue

        with f.open() as fh:
            header = True
            for _ln in fh:
                if header:
                    header = False
                    continue
                pairs_done += 1

    return pairs_done, max_idx + 1


def _chunk_generator(
    pairs_path: Path,
    chunk_size: int,
    max_pairs: Optional[int],
    start_chunk_idx: int,
    skip_pairs: int,
) -> Tuple[int, List[List[str]]]:
    """
    Yields (chunk_idx, rows) where rows are split TSV fields (strings).
    Expects canonical header in the first line.
    """
    chunk_idx = start_chunk_idx
    rows: List[List[str]] = []
    seen = 0
    skipped = 0

    with pairs_path.open() as fh:
        header = True
        for ln in fh:
            if header:
                header = False
                continue

            ln = ln.strip()
            if not ln:
                continue

            if skipped < skip_pairs:
                skipped += 1
                continue

            parts = ln.split("\t")
            if len(parts) < 2:
                continue
            rows.append(parts)
            seen += 1

            if len(rows) >= chunk_size:
                yield chunk_idx, rows
                chunk_idx += 1
                rows = []

            if max_pairs is not None and seen >= max_pairs:
                break

    if rows:
        yield chunk_idx, rows


def _worker_chunk(args):
    """
    Compute observed contingency counts for one chunk and write chunk TSV.

    args:
      (chunk_idx, rows, Y_locus_bits, mask, n_tips, out_dir)
    """
    chunk_idx, rows, Y, mask, n_tips, out_dir = args
    out_dir = Path(out_dir)
    outpath = out_dir / f"pairs_obs.chunk_{chunk_idx:06d}.tsv"

    # local LUT (fast; avoids pickling cached function state)
    lut = popcount_lut_u8()

    def popcount(arr_u8: np.ndarray) -> int:
        return int(lut[arr_u8].sum(dtype=np.uint32))

    with outpath.open("w") as fh:
        fh.write(
            "v\tw\tdistance\tflag\tscore\tcount\tM2\tmin_distance\tmax_distance\t"
            "n11\tn10\tn01\tn00\tp11_obs\n"
        )

        n_written = 0
        for parts in rows:
            # canonical cols: v w distance flag score count M2 min_distance max_distance
            # ensure indices exist (Stage 0 guarantees 9 columns)
            try:
                v = int(parts[0])
                w = int(parts[1])
            except Exception:
                continue

            if v < 0 or w < 0 or v >= Y.shape[0] or w >= Y.shape[0]:
                continue

            A = Y[v]  # uint8 vector over tips bytes
            B = Y[w]

            B0 = (~B) & mask
            A0 = (~A) & mask

            n11 = popcount(A & B)
            n10 = popcount(A & B0)
            n01 = popcount(A0 & B)
            n00 = popcount(A0 & B0)
            p11 = n11 / float(n_tips)

            fh.write(
                f"{v}\t{w}\t{parts[2]}\t{parts[3]}\t{parts[4]}\t{parts[5]}\t{parts[6]}\t{parts[7]}\t{parts[8]}\t"
                f"{n11}\t{n10}\t{n01}\t{n00}\t{p11:.8g}\n"
            )
            n_written += 1

    return chunk_idx, n_written


def _combine_chunks(chunk_dir: Path, final_out: Path, delete_chunks: bool) -> int:
    chunk_files = sorted(chunk_dir.glob("pairs_obs.chunk_*.tsv"))
    if not chunk_files:
        raise RuntimeError(f"No chunk files found in {chunk_dir}")

    final_out.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with final_out.open("w") as fout:
        fout.write(
            "v\tw\tdistance\tflag\tscore\tcount\tM2\tmin_distance\tmax_distance\t"
            "n11\tn10\tn01\tn00\tp11_obs\n"
        )
        for cf in chunk_files:
            with cf.open() as fh:
                header = True
                for ln in fh:
                    if header:
                        header = False
                        continue
                    fout.write(ln)
                    total += 1

    if delete_chunks:
        for cf in chunk_files:
            try:
                cf.unlink()
            except Exception:
                pass

    return total


def stage2_score_pairs(
    run_dir: str,
    out_name: str = "pairs_obs.tsv",
    chunk_size: int = 50000,
    threads: int = 16,
    keep_chunks: bool = False,
    no_resume: bool = False,
    max_pairs: Optional[int] = None,
) -> dict:
    run_dir = Path(run_dir)
    work0 = run_dir / "work" / "stage0"
    work2 = run_dir / "work" / "stage2"
    chunk_dir = work2 / "chunks"
    results = run_dir / "results"
    meta_dir = run_dir / "meta"

    chunk_dir.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Load Stage 0 matrix (canonical tip order)
    mat = load_locusmajor_npz(work0 / "matrix.locusmajor.npz")
    Y = mat.Y_locus_bits  # (n_loci, ceil(n_tips/8))
    n_tips = len(mat.tips)
    mask = valid_mask_bytes(n_tips)

    pairs_path = work0 / "pairs.candidates.tsv"
    loci_used_src = work0 / "loci_used.txt"

    total_pairs = _count_pairs_tsv(pairs_path, max_pairs=max_pairs)

    # multiprocessing start method
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass

    # Resume bookkeeping
    if no_resume:
        for cf in chunk_dir.glob("pairs_obs.chunk_*.tsv"):
            try:
                cf.unlink()
            except Exception:
                pass
        pairs_done, next_chunk_idx = 0, 1
    else:
        pairs_done, next_chunk_idx = _existing_chunks_info(chunk_dir)

    if pairs_done > 0:
        print(f"[resume] pairs_done={pairs_done} next_chunk_idx={next_chunk_idx:06d}")

    if pairs_done >= total_pairs and total_pairs > 0:
        print("[resume] all requested pairs already scored; combining chunks only.")
        out_path = results / out_name
        n_written = _combine_chunks(chunk_dir, out_path, delete_chunks=not keep_chunks)
        # copy loci_used for downstream convenience
        shutil.copyfile(loci_used_src, results / "loci_used.txt")
        meta = {
            "stage": "stage2",
            "inputs": {"pairs": str(pairs_path), "matrix": str(work0 / "matrix.locusmajor.npz")},
            "params": {"chunk_size": chunk_size, "threads": threads, "keep_chunks": keep_chunks, "no_resume": no_resume, "max_pairs": max_pairs},
            "counts": {"n_pairs_total": total_pairs, "n_pairs_written": n_written, "n_tips": n_tips, "n_loci": int(mat.n_loci)},
            "outputs": {"pairs_obs": str(out_path), "loci_used": str(results / "loci_used.txt")},
        }
        write_stage_meta(meta_dir / "stage2.json", meta)
        return meta

    # Build jobs
    jobs = []
    max_new = (max_pairs - pairs_done) if (max_pairs is not None) else None
    gen = _chunk_generator(
        pairs_path=pairs_path,
        chunk_size=chunk_size,
        max_pairs=max_new,
        start_chunk_idx=next_chunk_idx,
        skip_pairs=pairs_done,
    )
    for chunk_idx, rows in gen:
        jobs.append((chunk_idx, rows, Y, mask, n_tips, str(chunk_dir)))

    print(f"[info] tips={n_tips} loci={mat.n_loci} total_pairs={total_pairs} new_chunks={len(jobs)} threads={threads}")

    processed_now = 0
    with ProcessPoolExecutor(max_workers=threads) as pool, tqdm(
        total=total_pairs, initial=pairs_done, unit="pair", desc="[stage2_score_pairs]"
    ) as bar:
        for _chunk_idx, n_written in pool.map(_worker_chunk, jobs):
            processed_now += n_written
            bar.update(n_written)

    # Combine and cleanup
    out_path = results / out_name
    n_written = _combine_chunks(chunk_dir, out_path, delete_chunks=not keep_chunks)

    # Copy loci_used for Stage 3 compatibility
    shutil.copyfile(loci_used_src, results / "loci_used.txt")

    meta = {
        "stage": "stage2",
        "inputs": {
            "pairs": str(pairs_path),
            "matrix": str(work0 / "matrix.locusmajor.npz"),
            "loci_used_src": str(loci_used_src),
        },
        "params": {
            "chunk_size": chunk_size,
            "threads": threads,
            "keep_chunks": keep_chunks,
            "no_resume": no_resume,
            "max_pairs": max_pairs,
        },
        "counts": {
            "n_pairs_total": total_pairs,
            "n_pairs_written": n_written,
            "n_tips": n_tips,
            "n_loci": int(mat.n_loci),
        },
        "outputs": {
            "pairs_obs": str(out_path),
            "loci_used": str(results / "loci_used.txt"),
            "chunk_dir": str(chunk_dir),
        },
    }
    write_stage_meta(meta_dir / "stage2.json", meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser("Stage 2: score observed pairs from Stage 0 artifacts (resumable)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="pairs_obs.tsv")
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--keep-chunks", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=None)
    args = ap.parse_args()

    meta = stage2_score_pairs(
        run_dir=args.run_dir,
        out_name=args.out,
        chunk_size=args.chunk,
        threads=args.threads,
        keep_chunks=args.keep_chunks,
        no_resume=args.no_resume,
        max_pairs=args.max_pairs,
    )
    print(f"[ok] stage2 wrote: {meta['outputs']['pairs_obs']}")


if __name__ == "__main__":
    main()