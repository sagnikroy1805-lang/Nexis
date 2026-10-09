"""Graph diagnostics to run before trusting (or blaming) a GNN.

Implements Concept Mastery §8.7 (heterophily) and the checks listed in the
Getting Started guide §4.1 "My GNN is worse than XGBoost".
"""

from __future__ import annotations

import numpy as np


def edge_homophily(edge_index: np.ndarray, y: np.ndarray) -> float:
    """Fraction of edges joining two nodes with the same label.

    Below ~0.3 the graph is heterophilous: plain GCN, which averages neighbours,
    is expected to underperform. That is a finding to report, not a failure.
    """
    ei = np.asarray(edge_index)
    if ei.shape[1] == 0:
        return float("nan")
    y = np.asarray(y)
    return float((y[ei[0]] == y[ei[1]]).mean())


def positive_edge_adjacency(
    src: np.ndarray, dst: np.ndarray, y_edge: np.ndarray
) -> dict[str, float]:
    """For edge classification: do positive edges touch other positive edges?

    Message passing helps an edge classifier only if positives cluster. Reports
    the share of positive edges whose sender or receiver is an endpoint of at
    least one other positive edge, against the same share for negatives.
    """
    src, dst, y = np.asarray(src), np.asarray(dst), np.asarray(y_edge).astype(bool)
    pos_nodes, counts = np.unique(np.concatenate([src[y], dst[y]]), return_counts=True)
    pos_count = dict(zip(pos_nodes.tolist(), counts.tolist(), strict=True))

    def touches(i: int, own: int) -> bool:
        return pos_count.get(int(src[i]), 0) - own > 0 or pos_count.get(int(dst[i]), 0) - own > 0

    pos_idx = np.flatnonzero(y)
    neg_idx = np.flatnonzero(~y)
    rng = np.random.default_rng(0)
    neg_sample = rng.choice(neg_idx, size=min(len(neg_idx), 200_000), replace=False)
    return {
        "positive_touching_positive": float(np.mean([touches(int(i), 1) for i in pos_idx]))
        if len(pos_idx)
        else float("nan"),
        "negative_touching_positive": float(np.mean([touches(int(i), 0) for i in neg_sample]))
        if len(neg_sample)
        else float("nan"),
    }


def degree_summary(edge_index: np.ndarray, n_nodes: int) -> dict[str, float]:
    """Degree distribution and isolated-node fraction: is there anything to pass?"""
    ei = np.asarray(edge_index)
    deg = np.bincount(np.concatenate([ei[0], ei[1]]), minlength=n_nodes)
    return {
        "isolated_fraction": float((deg == 0).mean()),
        "median_degree": float(np.median(deg)),
        "p99_degree": float(np.percentile(deg, 99)),
        "max_degree": float(deg.max()) if len(deg) else 0.0,
    }
