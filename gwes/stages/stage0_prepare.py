from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np

from gwes.io import read_newick_tip_names, compute_tip_permutation
from gwes.fasta_matrix import (
    read_fake_fasta_tipmajor,
    align_tipmajor_to_tree,
    tipmajor_to_locusmajor,
    save_tipmajor_npz,
    save_locusmajor_npz,
)
from gwes.pairs import (
    read_spydrpick_pairs_whitespace_to_arrays,
    write_pairs_canonical_tsv,
    write_loci_used,
)
from gwes.manifest import write_stage_meta


def stage0_prepare(
    tree_path: str,
    fasta_path: str,
    pairs_path: str,
    run_dir: str,
    presence: str = "a",
    absence: str = "c",
    strict: bool = True,
    block_loci: int = 8192,
    max_pairs: int | None = None,
    max_tips: int | None = None,
) -> dict:
    run_dir = Path(run_dir)
    work = run_dir / "work" / "stage0"
    meta_dir = run_dir / "meta"
    work.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    tree = read_newick_tip_names(tree_path)

    tipmaj_raw = read_fake_fasta_tipmajor(
        fasta_path=fasta_path,
        presence_char=presence,
        absence_char=absence,
        max_tips=max_tips,
    )

    perm, aligned_tips = compute_tip_permutation(
        tree_tips=tree.tip_names,
        fasta_tips=tipmaj_raw.tips,
        strict=strict,
    )

    tipmaj = align_tipmajor_to_tree(tipmaj_raw, perm=perm, aligned_tips=aligned_tips)

    # Convert to locus-major for downstream speed
    locmaj = tipmajor_to_locusmajor(tipmaj, block_loci=block_loci)

    # Read and canonicalize pairs
    v, w, opt = read_spydrpick_pairs_whitespace_to_arrays(pairs_path, max_pairs=max_pairs)

    # Range checks against n_loci
    if v.size:
        vmax = int(max(v.max(), w.max()))
        vmin = int(min(v.min(), w.min()))
        if vmin < 0:
            raise ValueError(f"Pairs contain negative indices (min={vmin}).")
        if vmax >= tipmaj.n_loci:
            raise ValueError(f"Pairs refer to locus {vmax} but FASTA has n_loci={tipmaj.n_loci}.")

    # Write artifacts
    (work / "tips.tree.txt").write_text("\n".join(aligned_tips) + "\n")
    (work / "tips.fasta.txt").write_text("\n".join(tipmaj_raw.tips) + "\n")
    np.save(work / "tip_perm.npy", perm)

    save_tipmajor_npz(work / "matrix.tipmajor.npz", tipmaj)
    save_locusmajor_npz(work / "matrix.locusmajor.npz", locmaj)

    write_pairs_canonical_tsv(work / "pairs.candidates.tsv", v=v, w=w, opt=opt)
    loci_used = write_loci_used(work / "loci_used.txt", v=v, w=w)

    meta = {
        "stage": "stage0",
        "inputs": {"tree": str(tree_path), "fasta": str(fasta_path), "pairs": str(pairs_path)},
        "params": {
            "presence": presence,
            "absence": absence,
            "strict": strict,
            "block_loci": block_loci,
            "max_pairs": max_pairs,
            "max_tips": max_tips,
        },
        "counts": {
            "n_tree_tips": len(tree.tip_names),
            "n_aligned_tips": len(aligned_tips),
            "n_loci": int(tipmaj.n_loci),
            "n_pairs": int(v.size),
            "n_loci_used": int(loci_used.size),
        },
        "outputs": {
            "work_dir": str(work),
            "matrix_locusmajor": str(work / "matrix.locusmajor.npz"),
            "pairs_canon": str(work / "pairs.candidates.tsv"),
        },
    }
    write_stage_meta(meta_dir / "stage0.json", meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser("Stage 0: prepare and write aligned artifacts into --run-dir")
    ap.add_argument("--tree", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--presence", default="a")
    ap.add_argument("--absence", default="c")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--non-strict", action="store_true")
    ap.add_argument("--block-loci", type=int, default=8192)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--max-tips", type=int, default=None)
    args = ap.parse_args()

    if args.strict and args.non_strict:
        raise ValueError("Choose only one of --strict or --non-strict.")
    strict = True if args.strict else (False if args.non_strict else True)

    meta = stage0_prepare(
        tree_path=args.tree,
        fasta_path=args.fasta,
        pairs_path=args.pairs,
        run_dir=args.run_dir,
        presence=args.presence,
        absence=args.absence,
        strict=strict,
        block_loci=args.block_loci,
        max_pairs=args.max_pairs,
        max_tips=args.max_tips,
    )
    print(f"[ok] stage0 wrote: {meta['outputs']['work_dir']}")


if __name__ == "__main__":
    main()
