from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class PairsChunk:
    v: np.ndarray
    w: np.ndarray
    distance: Optional[np.ndarray] = None
    flag: Optional[np.ndarray] = None
    score: Optional[np.ndarray] = None
    count: Optional[np.ndarray] = None
    M2: Optional[np.ndarray] = None
    min_distance: Optional[np.ndarray] = None
    max_distance: Optional[np.ndarray] = None


PAIR_COLS_CANON = [
    "v", "w", "distance", "flag", "score", "count", "M2", "min_distance", "max_distance"
]


def read_spydrpick_pairs_whitespace_to_arrays(
    pairs_path: Union[str, Path],
    max_pairs: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Reads whitespace pairs file. Returns v, w, and an optional-columns dict.
    """
    pairs_path = str(pairs_path)
    p = Path(pairs_path)
    if not p.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

    v_list: List[int] = []
    w_list: List[int] = []
    dist_list: List[int] = []
    flag_list: List[int] = []
    score_list: List[float] = []
    count_list: List[int] = []
    M2_list: List[int] = []
    mind_list: List[int] = []
    maxd_list: List[int] = []

    with open(pairs_path, "r") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if max_pairs is not None and len(v_list) >= max_pairs:
                break
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Bad pairs line {line_idx}: expected ≥2 columns")

            v_list.append(int(parts[0]))
            w_list.append(int(parts[1]))

            if len(parts) >= 3:
                dist_list.append(int(parts[2]))
            if len(parts) >= 4:
                flag_list.append(int(parts[3]))
            if len(parts) >= 5:
                score_list.append(float(parts[4]))
            if len(parts) >= 6:
                count_list.append(int(parts[5]))
            if len(parts) >= 7:
                M2_list.append(int(parts[6]))
            if len(parts) >= 8:
                mind_list.append(int(parts[7]))
            if len(parts) >= 9:
                maxd_list.append(int(parts[8]))

    v = np.asarray(v_list, dtype=np.int64)
    w = np.asarray(w_list, dtype=np.int64)

    opt = {}
    if len(dist_list) == len(v_list): opt["distance"] = np.asarray(dist_list, dtype=np.int64)
    if len(flag_list) == len(v_list): opt["flag"] = np.asarray(flag_list, dtype=np.int8)
    if len(score_list) == len(v_list): opt["score"] = np.asarray(score_list, dtype=np.float32)
    if len(count_list) == len(v_list): opt["count"] = np.asarray(count_list, dtype=np.int64)
    if len(M2_list) == len(v_list): opt["M2"] = np.asarray(M2_list, dtype=np.int64)
    if len(mind_list) == len(v_list): opt["min_distance"] = np.asarray(mind_list, dtype=np.int64)
    if len(maxd_list) == len(v_list): opt["max_distance"] = np.asarray(maxd_list, dtype=np.int64)
    return v, w, opt


def write_pairs_canonical_tsv(
    out_path: Union[str, Path],
    v: np.ndarray,
    w: np.ndarray,
    opt: dict,
) -> None:
    """
    Writes a canonical TSV with header (always same columns).
    Missing optional columns are written as '.'.
    """
    out_path = str(out_path)
    n = v.size
    cols = {k: opt.get(k, None) for k in PAIR_COLS_CANON if k not in ("v", "w")}
    with open(out_path, "w") as f:
        f.write("\t".join(PAIR_COLS_CANON) + "\n")
        for i in range(n):
            row = [str(int(v[i])), str(int(w[i]))]
            for k in PAIR_COLS_CANON[2:]:
                arr = cols.get(k, None)
                if arr is None:
                    row.append(".")
                else:
                    row.append(str(arr[i]))
            f.write("\t".join(row) + "\n")


def write_loci_used(out_path: Union[str, Path], v: np.ndarray, w: np.ndarray) -> np.ndarray:
    loci = np.unique(np.concatenate([v, w]).astype(np.int64, copy=False))
    loci.sort()
    out_path = str(out_path)
    with open(out_path, "w") as f:
        for x in loci:
            f.write(f"{int(x)}\n")
    return loci
