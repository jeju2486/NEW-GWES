from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class TreeData:
    newick_path: str
    tip_names: List[str]


def read_newick_tip_names(newick_path: Union[str, Path]) -> TreeData:
    newick_path = str(newick_path)
    p = Path(newick_path)
    if not p.exists():
        raise FileNotFoundError(f"Tree file not found: {newick_path}")

    # Biopython preferred
    try:
        from Bio import Phylo  # type: ignore
        tree = Phylo.read(newick_path, "newick")
        tips = [t.name for t in tree.get_terminals() if t.name]
        if not tips:
            raise ValueError("No tip names found in Newick tree.")
        return TreeData(newick_path=newick_path, tip_names=tips)
    except ImportError:
        pass
    except Exception as e:
        raise RuntimeError(f"Failed to parse Newick with Biopython: {e}") from e

    # ete3 fallback
    try:
        from ete3 import Tree  # type: ignore
        t = Tree(newick_path, format=1)
        tips = [leaf.name for leaf in t.get_leaves() if leaf.name]
        if not tips:
            raise ValueError("No tip names found in Newick tree.")
        return TreeData(newick_path=newick_path, tip_names=tips)
    except ImportError as e:
        raise ImportError(
            "No supported Newick parser found. Install one of:\n"
            "  - biopython (recommended)\n"
            "  - ete3\n"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to parse Newick with ete3: {e}") from e


def compute_tip_permutation(
    tree_tips: Sequence[str],
    fasta_tips: Sequence[str],
    strict: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Returns:
      perm: array of length len(tree_tips), where perm[i] is the FASTA row index
            corresponding to tree tip tree_tips[i].
      aligned_tips: list of tips in the final aligned order (tree order, or intersection order)
    """
    tree_tips = list(tree_tips)
    fasta_tips = list(fasta_tips)
    idx: Dict[str, int] = {t: i for i, t in enumerate(fasta_tips)}

    if strict:
        missing_in_fasta = [t for t in tree_tips if t not in idx]
        if missing_in_fasta:
            raise ValueError(
                f"{len(missing_in_fasta)} tree tips missing in FASTA: "
                f"{missing_in_fasta[:10]}{' ...' if len(missing_in_fasta) > 10 else ''}"
            )
        missing_in_tree = [t for t in fasta_tips if t not in set(tree_tips)]
        if missing_in_tree:
            raise ValueError(
                f"{len(missing_in_tree)} FASTA tips missing in tree: "
                f"{missing_in_tree[:10]}{' ...' if len(missing_in_tree) > 10 else ''}"
            )
        perm = np.array([idx[t] for t in tree_tips], dtype=np.int64)
        return perm, tree_tips

    shared = [t for t in tree_tips if t in idx]
    if not shared:
        raise ValueError("No shared tips between tree and FASTA.")
    perm = np.array([idx[t] for t in shared], dtype=np.int64)
    return perm, shared
