from __future__ import annotations

from functools import lru_cache
import numpy as np


@lru_cache(maxsize=1)
def popcount_lut_u8() -> np.ndarray:
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = bin(i).count("1")
    return lut


def popcount_u8(arr_u8: np.ndarray) -> int:
    """
    Sum popcount over a uint8 array.
    """
    if arr_u8.dtype != np.uint8:
        arr_u8 = arr_u8.astype(np.uint8, copy=False)
    lut = popcount_lut_u8()
    return int(lut[arr_u8].sum())


def valid_mask_bytes(n_bits: int) -> np.ndarray:
    """
    Mask to ensure bitwise NOT (~) doesn’t introduce 1s beyond n_bits in the last byte.
    Returns a uint8 array of length ceil(n_bits/8).
    """
    n_bytes = (n_bits + 7) // 8
    mask = np.full(n_bytes, 0xFF, dtype=np.uint8)
    r = n_bits % 8
    if r != 0:
        mask[-1] = (1 << r) - 1
    return mask

def get_locus_column_bits(Y_bits: np.ndarray, locus: int) -> np.ndarray:
    """
    Extract y (n_tips,) as uint8 {0,1} for locus index from packed bits.
    Assumes loci are packed along columns with bitorder='little'.
    """
    by = locus >> 3
    bi = locus & 7
    return ((Y_bits[:, by] >> bi) & 1).astype(np.uint8, copy=False)