from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Sequence

import numpy as np

@dataclass
class ParsedTree:
    tip_names: List[str]                 # tips as discovered by backend
    parent: Dict[int, Optional[int]]     # node_id -> parent node_id
    edge_len: Dict[int, float]           # node_id -> length(parent->node)
    children: Dict[int, List[int]]       # node_id -> children
    node_name: Dict[int, Optional[str]]  # node_id -> name (tips have names)
    root_id: int


def _parse_newick_biophylo(newick_path: str, default_branch_length: Optional[float]) -> ParsedTree:
    from Bio import Phylo  # type: ignore

    tree = Phylo.read(newick_path, "newick")
    root = tree.root
    root_id = id(root)

    parent: Dict[int, Optional[int]] = {root_id: None}
    edge_len: Dict[int, float] = {}
    children: Dict[int, List[int]] = {}
    node_name: Dict[int, Optional[str]] = {root_id: getattr(root, "name", None)}

    stack = [root]
    while stack:
        node = stack.pop()
        nid = id(node)
        children.setdefault(nid, [])
        node_name.setdefault(nid, getattr(node, "name", None))

        for ch in node.clades:
            cid = id(ch)
            parent[cid] = nid
            children.setdefault(nid, []).append(cid)
            children.setdefault(cid, [])
            node_name[cid] = getattr(ch, "name", None)

            bl = getattr(ch, "branch_length", None)
            if bl is None:
                if default_branch_length is None:
                    raise ValueError(
                        f"Missing branch_length at node '{node_name[cid]}' (id={cid}). "
                        f"Provide --default-branch-length."
                    )
                bl = float(default_branch_length)
            edge_len[cid] = float(bl)

            stack.append(ch)

    tips = [t.name for t in tree.get_terminals() if t.name]
    if not tips:
        raise ValueError("No tip names found in Newick tree.")

    return ParsedTree(
        tip_names=tips,
        parent=parent,
        edge_len=edge_len,
        children=children,
        node_name=node_name,
        root_id=root_id,
    )


def _parse_newick_ete3(newick_path: str, default_branch_length: Optional[float]) -> ParsedTree:
    from ete3 import Tree  # type: ignore

    t = Tree(newick_path, format=1)
    root_id = id(t)

    parent: Dict[int, Optional[int]] = {root_id: None}
    edge_len: Dict[int, float] = {}
    children: Dict[int, List[int]] = {}
    node_name: Dict[int, Optional[str]] = {root_id: t.name if t.name else None}

    stack = [t]
    while stack:
        node = stack.pop()
        nid = id(node)
        children.setdefault(nid, [])
        node_name.setdefault(nid, node.name if node.name else None)

        for ch in node.children:
            cid = id(ch)
            parent[cid] = nid
            children.setdefault(nid, []).append(cid)
            children.setdefault(cid, [])
            node_name[cid] = ch.name if ch.name else None

            bl = getattr(ch, "dist", None)
            if bl is None:
                if default_branch_length is None:
                    raise ValueError(
                        f"Missing branch length at node '{node_name[cid]}' (id={cid}). "
                        f"Provide --default-branch-length."
                    )
                bl = float(default_branch_length)
            edge_len[cid] = float(bl)

            stack.append(ch)

    tips = [leaf.name for leaf in t.get_leaves() if leaf.name]
    if not tips:
        raise ValueError("No tip names found in Newick tree.")

    return ParsedTree(
        tip_names=tips,
        parent=parent,
        edge_len=edge_len,
        children=children,
        node_name=node_name,
        root_id=root_id,
    )


def parse_newick(newick_path: Union[str, Path], default_branch_length: Optional[float]) -> ParsedTree:
    newick_path = str(newick_path)
    if not Path(newick_path).exists():
        raise FileNotFoundError(f"Tree file not found: {newick_path}")

    try:
        return _parse_newick_biophylo(newick_path, default_branch_length)
    except ImportError:
        pass

    try:
        return _parse_newick_ete3(newick_path, default_branch_length)
    except ImportError as e:
        raise ImportError(
            "No supported Newick parser found. Install one of:\n"
            "  - biopython (recommended)\n"
            "  - ete3\n"
        ) from e


def build_tip_edge_incidence_for_tips(
    pt: ParsedTree,
    tips_order: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build COO incidence for a *specified* tip order (tips_order).
    This is the key to aligning covariance with Stage 0’s tip order.
    """
    name_to_node: Dict[str, int] = {}
    for nid, nm in pt.node_name.items():
        if nm is not None:
            name_to_node[nm] = nid

    # Edge indices: one per non-root node
    edge_idx: Dict[int, int] = {}
    w_list: List[float] = []
    edge_counter = 0
    for nid, par in pt.parent.items():
        if par is None:
            continue
        edge_idx[nid] = edge_counter
        w_list.append(pt.edge_len[nid])
        edge_counter += 1

    w = np.asarray(w_list, dtype=np.float64)
    n_tips = len(tips_order)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for r, tip in enumerate(tips_order):
        nid = name_to_node.get(tip, None)
        if nid is None:
            raise ValueError(f"Tip '{tip}' not found as a named node in the tree.")

        while True:
            par = pt.parent.get(nid, None)
            if par is None:
                break
            c = edge_idx[nid]
            rows.append(r)
            cols.append(c)
            data.append(1.0)
            nid = par

    return (
        np.asarray(rows, dtype=np.int32),
        np.asarray(cols, dtype=np.int32),
        np.asarray(data, dtype=np.float64),
        w,
    )


def covariance_from_incidence(
    n_tips: int,
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    w: np.ndarray,
    use_sparse: bool = True,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    if use_sparse:
        try:
            import scipy.sparse as sp  # type: ignore
            M = sp.coo_matrix((data, (rows, cols)), shape=(n_tips, w.shape[0]), dtype=dtype).tocsr()
            Mw = M.multiply(w)  # scales columns by w at nonzeros
            A = (Mw @ M.T).toarray()
            return np.asarray(A, dtype=dtype)
        except ImportError:
            pass

    # Dense fallback
    n_edges = w.shape[0]
    M = np.zeros((n_tips, n_edges), dtype=dtype)
    M[rows, cols] = data.astype(dtype, copy=False)
    A = (M * w) @ M.T
    return np.asarray(A, dtype=dtype)


def build_phylo_cov_aligned(
    tree_path: Union[str, Path],
    tips_order: Sequence[str],
    jitter: float = 1e-6,
    default_branch_length: Optional[float] = None,
    use_sparse: bool = True,
    dtype: str = "float64",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (A, L) in the *tips_order*.
    """
    dtype_np = np.float64 if dtype == "float64" else np.float32
    pt = parse_newick(tree_path, default_branch_length=default_branch_length)

    rows, cols, data, w = build_tip_edge_incidence_for_tips(pt, tips_order=tips_order)
    A = covariance_from_incidence(
        n_tips=len(tips_order),
        rows=rows,
        cols=cols,
        data=data,
        w=w.astype(dtype_np, copy=False),
        use_sparse=use_sparse,
        dtype=dtype_np,
    )

    if jitter and jitter > 0:
        A = A + (dtype_np(jitter) * np.eye(len(tips_order), dtype=dtype_np))

    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError as e:
        evals = np.linalg.eigvalsh(A.astype(np.float64, copy=False))
        raise np.linalg.LinAlgError(
            f"Cholesky failed. min_eig={evals.min():.6g}. "
            f"Increase jitter or check branch lengths/rooting."
        ) from e

    return A, L

def load_basis_npz(path: str, *, dtype=np.float64) -> Tuple[List[str], np.ndarray, int]:
    """
    Load Stage3 phylo_basis.npz: returns (tips, Z, K).
    Requires 'tips'. Prefers 'Z' if present.
    """
    z = np.load(path, allow_pickle=True)
    if "tips" not in z.files:
        raise ValueError("basis npz must contain 'tips'.")
    tips = [str(x) for x in z["tips"].tolist()]

    if "Z" in z.files:
        Z = np.asarray(z["Z"])
        if Z.ndim != 2:
            raise ValueError("'Z' must be 2D.")
        K = int(z["K"]) if "K" in z.files else int(Z.shape[1])
        return tips, Z.astype(dtype, copy=False), K

    n_tips = len(tips)
    for k in z.files:
        arr = z[k]
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] == n_tips:
            K = int(z["K"]) if "K" in z.files else int(arr.shape[1])
            return tips, arr.astype(dtype, copy=False), K

    raise ValueError("Could not find a 2D basis matrix in basis npz.")
