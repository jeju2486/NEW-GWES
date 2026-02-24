from __future__ import annotations

from typing import Sequence
import numpy as np


def locus_to_u8(Y_locus_bits: np.ndarray, locus: int, n_tips: int) -> np.ndarray:
    """
    Y_locus_bits: (n_loci, ceil(n_tips/8)) packed over tips, bitorder='little'
    Returns y: (n_tips,) uint8 in {0,1}
    """
    row = Y_locus_bits[locus]  # (n_bytes_tips,)
    bits = np.unpackbits(row, bitorder="little")
    return bits[:n_tips].astype(np.uint8, copy=False)


def loci_to_matrix_u8(Y_locus_bits: np.ndarray, loci: Sequence[int], n_tips: int) -> np.ndarray:
    """
    Returns Y: (n_tips, m) uint8 for loci list.
    Efficient for sigma-scoring preloads.
    """
    loci = np.asarray(list(loci), dtype=np.int64)
    sub = Y_locus_bits[loci, :]  # (m, n_bytes_tips)
    bits = np.unpackbits(sub, bitorder="little", axis=1)[:, :n_tips]  # (m, n_tips)
    return bits.T.astype(np.uint8, copy=False)
