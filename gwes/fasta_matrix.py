from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

@dataclass(frozen=True)
class TipMajorMatrix:
    tips: List[str]           # order corresponds to rows
    n_loci: int
    Y_tip_bits: np.ndarray    # (n_tips, ceil(n_loci/8)) uint8, bitorder="little"
    presence_char: str
    absence_char: str
    bitorder: str = "little"


@dataclass(frozen=True)
class LocusMajorMatrix:
    tips: List[str]            # tree-aligned tip order
    n_loci: int
    Y_locus_bits: np.ndarray   # (n_loci, ceil(n_tips/8)) uint8, bitorder="little"
    presence_char: str
    absence_char: str
    bitorder: str = "little"


def read_fake_fasta_tipmajor(
    fasta_path: Union[str, Path],
    presence_char: str = "a",
    absence_char: str = "c",
    max_tips: Optional[int] = None,
) -> TipMajorMatrix:
    """
    Reads fake FASTA into tip-major packed bits: rows=tips, bits=loci.
    Strictly validates characters (only presence/absence allowed).
    """
    fasta_path = str(fasta_path)
    p = Path(fasta_path)
    if not p.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    pres = presence_char.lower()
    absn = absence_char.lower()
    if len(pres) != 1 or len(absn) != 1 or pres == absn:
        raise ValueError("presence_char and absence_char must be distinct single characters.")

    tips: List[str] = []
    packed_rows: List[np.ndarray] = []

    cur_name: Optional[str] = None
    cur_seq_chunks: List[str] = []
    n_loci: Optional[int] = None

    def flush() -> None:
        nonlocal cur_name, cur_seq_chunks, n_loci
        if cur_name is None:
            return
        seq = "".join(cur_seq_chunks).strip()
        if not seq:
            raise ValueError(f"Empty sequence for tip '{cur_name}' in {fasta_path}")

        if n_loci is None:
            n_loci = len(seq)
        elif len(seq) != n_loci:
            raise ValueError(
                f"Inconsistent sequence length: expected {n_loci}, got {len(seq)} for '{cur_name}'."
            )

        s = seq.lower()
        bad = set(s) - {pres, absn}
        if bad:
            raise ValueError(
                f"Unexpected characters {sorted(bad)} in '{cur_name}'. "
                f"Expected only '{presence_char}' and '{absence_char}'."
            )

        y_bool = (np.frombuffer(s.encode("ascii"), dtype=np.uint8) == ord(pres))
        y_bits = np.packbits(y_bool, bitorder="little")
        tips.append(cur_name)
        packed_rows.append(y_bits)

        cur_name = None
        cur_seq_chunks = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                cur_name = line[1:].strip()
                if not cur_name:
                    raise ValueError("Found FASTA header with empty tip name.")
                if max_tips is not None and len(tips) >= max_tips:
                    break
            else:
                if cur_name is None:
                    raise ValueError("Sequence encountered before first FASTA header.")
                cur_seq_chunks.append(line)

        if max_tips is None or len(tips) < max_tips:
            flush()

    if n_loci is None or not tips:
        raise ValueError(f"No records read from FASTA: {fasta_path}")

    Y_tip_bits = np.vstack(packed_rows).astype(np.uint8, copy=False)
    return TipMajorMatrix(
        tips=tips,
        n_loci=int(n_loci),
        Y_tip_bits=Y_tip_bits,
        presence_char=presence_char,
        absence_char=absence_char,
    )


def align_tipmajor_to_tree(
    tipmajor: TipMajorMatrix,
    perm: np.ndarray,
    aligned_tips: List[str],
) -> TipMajorMatrix:
    Y = tipmajor.Y_tip_bits[perm, :]
    return TipMajorMatrix(
        tips=list(aligned_tips),
        n_loci=tipmajor.n_loci,
        Y_tip_bits=Y,
        presence_char=tipmajor.presence_char,
        absence_char=tipmajor.absence_char,
        bitorder=tipmajor.bitorder,
    )


def tipmajor_to_locusmajor(
    tipmajor_aligned: TipMajorMatrix,
    block_loci: int = 8192,
) -> LocusMajorMatrix:
    """
    Converts tip-major packed bits (tips × bytes_loci) to locus-major packed bits
    (loci × bytes_tips) in blocks to avoid large intermediate memory.
    """
    Y_tip_bits = tipmajor_aligned.Y_tip_bits
    n_tips = Y_tip_bits.shape[0]
    n_loci = tipmajor_aligned.n_loci

    n_bytes_tips = (n_tips + 7) // 8
    Y_locus_bits = np.empty((n_loci, n_bytes_tips), dtype=np.uint8)

    # Helper: unpack a locus-block for all tips
    # np.unpackbits yields bits per byte; we slice to exact loci range.
    for start in range(0, n_loci, block_loci):
        end = min(n_loci, start + block_loci)
        byte_start = start // 8
        byte_end = (end + 7) // 8  # inclusive -> exclusive

        chunk = Y_tip_bits[:, byte_start:byte_end]  # (n_tips, n_bytes_chunk)
        bits = np.unpackbits(chunk, bitorder="little", axis=1)  # (n_tips, n_bits_chunk)

        # Slice the block bits precisely
        bit_offset = start - byte_start * 8
        bits_block = bits[:, bit_offset:bit_offset + (end - start)]  # (n_tips, block_len)

        # Transpose to loci × tips, then pack across tips
        bits_T = bits_block.T  # (block_len, n_tips)
        packed = np.packbits(bits_T, bitorder="little", axis=1)  # (block_len, n_bytes_tips)
        Y_locus_bits[start:end, :] = packed

    return LocusMajorMatrix(
        tips=tipmajor_aligned.tips,
        n_loci=n_loci,
        Y_locus_bits=Y_locus_bits,
        presence_char=tipmajor_aligned.presence_char,
        absence_char=tipmajor_aligned.absence_char,
        bitorder=tipmajor_aligned.bitorder,
    )


def save_tipmajor_npz(path: Union[str, Path], mat: TipMajorMatrix) -> None:
    path = str(path)
    np.savez_compressed(
        path,
        tips=np.array(mat.tips, dtype=object),
        n_loci=np.int64(mat.n_loci),
        Y_tip_bits=mat.Y_tip_bits,
        presence_char=np.array(mat.presence_char),
        absence_char=np.array(mat.absence_char),
        bitorder=np.array(mat.bitorder),
    )


def save_locusmajor_npz(path: Union[str, Path], mat: LocusMajorMatrix) -> None:
    path = str(path)
    np.savez_compressed(
        path,
        tips=np.array(mat.tips, dtype=object),
        n_loci=np.int64(mat.n_loci),
        Y_locus_bits=mat.Y_locus_bits,
        presence_char=np.array(mat.presence_char),
        absence_char=np.array(mat.absence_char),
        bitorder=np.array(mat.bitorder),
    )


def load_locusmajor_npz(path: Union[str, Path]) -> LocusMajorMatrix:
    z = np.load(str(path), allow_pickle=True)
    tips = list(z["tips"].tolist())
    return LocusMajorMatrix(
        tips=tips,
        n_loci=int(z["n_loci"]),
        Y_locus_bits=z["Y_locus_bits"].astype(np.uint8, copy=False),
        presence_char=str(z["presence_char"]),
        absence_char=str(z["absence_char"]),
        bitorder=str(z["bitorder"]),
    )
    
def read_fasta_dict(path: str) -> Dict[str, str]:
    seqs: Dict[str, List[str]] = {}
    name: Optional[str] = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].split()[0]
                if name in seqs:
                    raise ValueError(f"Duplicate FASTA header: {name}")
                seqs[name] = []
            else:
                if name is None:
                    raise ValueError("FASTA format error: sequence before header.")
                seqs[name].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


def pack_bits_from_sequences(
    tips_order: List[str],
    seqs: Dict[str, str],
    presence_char: str = "a",
) -> Tuple[np.ndarray, int]:
    n_tips = len(tips_order)
    if n_tips == 0:
        raise ValueError("No tips provided.")
    s0 = seqs.get(tips_order[0])
    if s0 is None:
        raise KeyError(f"Tip '{tips_order[0]}' not found in FASTA.")
    n_loci = len(s0)
    if n_loci == 0:
        raise ValueError("FASTA sequences appear empty.")

    n_bytes = (n_loci + 7) // 8
    Y_bits = np.zeros((n_tips, n_bytes), dtype=np.uint8)
    pres = ord(presence_char)

    for ti, tip in enumerate(tips_order):
        s = seqs.get(tip)
        if s is None:
            raise KeyError(f"Tip '{tip}' not found in FASTA.")
        if len(s) != n_loci:
            raise ValueError(f"Sequence length mismatch at tip '{tip}': {len(s)} vs {n_loci}")
        arr = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
        mask = (arr == pres)
        packed = np.packbits(mask, bitorder="little")
        Y_bits[ti, :] = packed[:n_bytes]

    return Y_bits, n_loci

