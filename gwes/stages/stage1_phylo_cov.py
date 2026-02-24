from __future__ import annotations

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
from pathlib import Path
import numpy as np

from gwes.fasta_matrix import load_locusmajor_npz
from gwes.phylo import build_phylo_cov_aligned
from gwes.manifest import write_stage_meta


def stage1_phylo_cov(
    run_dir: str,
    tree_path: str,
    out_name: str = "phylo_cov.npz",
    jitter: float = 1e-6,
    save_A: bool = False,
    default_branch_length: float | None = None,
    no_sparse: bool = False,
    dtype: str = "float64",
) -> dict:
    run_dir = Path(run_dir)
    work0 = run_dir / "work" / "stage0"
    work1 = run_dir / "work" / "stage1"
    meta_dir = run_dir / "meta"
    work1.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Canonical tip order comes from Stage 0 matrix artifact
    mat = load_locusmajor_npz(work0 / "matrix.locusmajor.npz")
    tips_order = mat.tips

    A, L = build_phylo_cov_aligned(
        tree_path=tree_path,
        tips_order=tips_order,
        jitter=jitter,
        default_branch_length=default_branch_length,
        use_sparse=not no_sparse,
        dtype=dtype,
    )

    out_path = work1 / out_name
    save = {
        "tip_names": np.asarray(tips_order, dtype=object),
        "L": L.astype(np.float64 if dtype == "float64" else np.float32, copy=False),
        "jitter": np.asarray([jitter], dtype=np.float64),
        "dtype": np.asarray([dtype], dtype=object),
        "method": np.asarray(["brownian_shared_root_path"], dtype=object),
        "tree_path": np.asarray([str(tree_path)], dtype=object),
    }
    if save_A:
        save["A"] = A.astype(np.float64 if dtype == "float64" else np.float32, copy=False)

    np.savez_compressed(out_path, **save)

    logdet = 2.0 * float(np.log(np.diag(L)).sum())
    meta = {
        "stage": "stage1",
        "inputs": {
            "tree": str(tree_path),
            "stage0_matrix": str(work0 / "matrix.locusmajor.npz"),
        },
        "params": {
            "jitter": float(jitter),
            "save_A": bool(save_A),
            "default_branch_length": default_branch_length,
            "use_sparse": bool(not no_sparse),
            "dtype": dtype,
        },
        "counts": {"n_tips": len(tips_order)},
        "outputs": {"phylo_cov_npz": str(out_path)},
        "summary": {"logdet_A": logdet},
    }
    write_stage_meta(meta_dir / "stage1.json", meta)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser("Stage 1: build phylogenetic covariance aligned to Stage 0 tips")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", default="phylo_cov.npz")
    ap.add_argument("--jitter", type=float, default=1e-6)
    ap.add_argument("--save-A", action="store_true")
    ap.add_argument("--default-branch-length", type=float, default=None)
    ap.add_argument("--no-sparse", action="store_true")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    args = ap.parse_args()

    meta = stage1_phylo_cov(
        run_dir=args.run_dir,
        tree_path=args.tree,
        out_name=args.out,
        jitter=args.jitter,
        save_A=args.save_A,
        default_branch_length=args.default_branch_length,
        no_sparse=args.no_sparse,
        dtype=args.dtype,
    )
    print(f"[ok] stage1 wrote: {meta['outputs']['phylo_cov_npz']}")


if __name__ == "__main__":
    main()
