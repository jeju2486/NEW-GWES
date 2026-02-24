#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import os
import numpy as np


def load_p_hat_from_stage3_dir(stage3_dir: Path, locus_fit_npz: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    """
    Load P_hat from Stage 3 output directory.

    Accepts:
      - stage3_dir/P_hat.npz
      - stage3_dir/P_hat_meta.json (+ stage3_dir/P_hat.mmap)

    Returns:
      P: ndarray or memmap, shape (n_tips, n_loci_used), float32
      loci: ndarray int64, shape (n_loci_used,)
      tips: list[str] if present (npz only), else None
    """
    stage3_dir = Path(stage3_dir)
    npz_path = stage3_dir / "P_hat.npz"
    meta_path = stage3_dir / "P_hat_meta.json"

    if npz_path.exists():
        z = np.load(npz_path, allow_pickle=True)
        if "P" not in z.files or "loci" not in z.files:
            raise ValueError("P_hat.npz must contain arrays 'P' and 'loci'.")
        P = z["P"].astype(np.float32, copy=False)
        loci = z["loci"].astype(np.int64, copy=False)
        tips = [str(x) for x in z["tips"].tolist()] if "tips" in z.files else None
        return P, loci, tips

    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        mmap_rel = meta["path"]
        mmap_path = Path(mmap_rel) if Path(mmap_rel).is_absolute() else (meta_path.parent / mmap_rel)
        shape = tuple(meta["shape"])
        dtype = np.dtype(meta["dtype"])
        P = np.memmap(str(mmap_path), dtype=dtype, mode="r", shape=shape, order="C")

        if locus_fit_npz is None:
            guess = stage3_dir / "locus_fit.npz"
            locus_fit_npz = guess if guess.exists() else None
        if locus_fit_npz is None or (not Path(locus_fit_npz).exists()):
            raise FileNotFoundError("For P_hat memmap, locus_fit.npz must exist (or provide its path).")

        lf = np.load(str(locus_fit_npz), allow_pickle=True)
        if "loci" not in lf.files:
            raise ValueError("locus_fit.npz must contain 'loci'.")
        loci = lf["loci"].astype(np.int64, copy=False)
        return P, loci, None

    raise FileNotFoundError(f"Neither P_hat.npz nor P_hat_meta.json found in {stage3_dir}")

def load_sigma_from_global_sigma_tsv(path: str) -> float:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            if parts[0].lower() == "sigma" and parts[1].lower().startswith("score"):
                continue
            rows.append((parts[0], parts[1]))
    for a, b in rows:
        if a.lower() == "chosen":
            return float(b)
    # fallback: max score (2nd col)
    best_sig, best_sc = None, -1e300
    for a, b in rows:
        try:
            sig = float(a); sc = float(b)
        except Exception:
            continue
        if sc > best_sc:
            best_sc = sc
            best_sig = sig
    if best_sig is None:
        raise ValueError(f"Could not parse sigma from: {path}")
    return float(best_sig)