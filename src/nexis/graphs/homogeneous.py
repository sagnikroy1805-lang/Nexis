"""Homogeneous account graph and leak-free structural features (rung 5).

Implements Concept Mastery §6.2-6.7 (directed weighted graphs, degree,
reachability, components, centrality), §7.1 (homogeneous account graph:
accounts are nodes, transactions are edges) and §9.1 (time snapshots).

The t_cutoff contract (CLAUDE.md rule 1): every graph here is built from
transactions with timestamp < t_cutoff, passed explicitly, and the constructor
asserts it. Structural features for a transaction at time t come from the
snapshot whose cutoff is the start of t's snapshot period, so they never contain
the transaction itself or anything after it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

GRAPH_FEATURES_NODE = (
    "in_degree",
    "out_degree",
    "in_tx",
    "out_tx",
    "log_in_weight",
    "log_out_weight",
    "degree_asymmetry",
    "flow_through",
    "reciprocal_partners",
    "pagerank",
    "log_scc_size",
    "nbr_out_degree_mean",
    "nbr_in_degree_mean",
)
GRAPH_FEATURES_EDGE = ("closes_cycle", "reverse_edge_exists", "prior_pair_tx")


@dataclass
class AccountGraph:
    """Directed weighted account graph, aggregated over transactions < t_cutoff.

    adj_count[i, j] = number of payments i -> j; adj_weight[i, j] = their USD sum.
    Node indices are global account codes, shared across snapshots.
    """

    t_cutoff: pd.Timestamp
    n_nodes: int
    adj_count: sp.csr_matrix
    adj_weight: sp.csr_matrix
    max_edge_time: pd.Timestamp | None


def build_account_graph(
    src: np.ndarray,
    dst: np.ndarray,
    amount: np.ndarray,
    timestamp: pd.Series,
    n_nodes: int,
    t_cutoff: pd.Timestamp,
) -> AccountGraph:
    """Aggregate every transaction strictly before t_cutoff into an account graph.

    Args are aligned arrays over all transactions; src/dst are integer account
    codes in [0, n_nodes). Rows at or after t_cutoff are excluded here, and the
    assertion below proves it -- a graph without that check is how future edges
    end up in message passing (§9.4 form 4).
    """
    if t_cutoff is None:
        raise ValueError("t_cutoff is required: graphs are always built as of a time")
    keep = (timestamp < t_cutoff).to_numpy()
    s, d, w = src[keep], dst[keep], amount[keep].astype(np.float64)
    max_t = timestamp[keep].max() if keep.any() else None
    assert max_t is None or max_t < t_cutoff, "LEAK: graph contains an edge at/after t_cutoff"

    shape = (n_nodes, n_nodes)
    count = sp.coo_matrix((np.ones(len(s)), (s, d)), shape=shape).tocsr()
    weight = sp.coo_matrix((w, (s, d)), shape=shape).tocsr()
    count.sum_duplicates()
    weight.sum_duplicates()
    return AccountGraph(t_cutoff, n_nodes, count, weight, max_t)


def _pagerank(adj: sp.csr_matrix, alpha: float = 0.85, iters: int = 50) -> np.ndarray:
    """Power-iteration PageRank on the unweighted pattern of adj (§6.7)."""
    n = adj.shape[0]
    a = (adj > 0).astype(np.float64).tocsr()
    out_deg = np.asarray(a.sum(axis=1)).ravel()
    inv = np.divide(1.0, out_deg, out=np.zeros(n), where=out_deg > 0)
    pt = (sp.diags(inv) @ a).T.tocsr()
    pr = np.full(n, 1.0 / n)
    dangling = out_deg == 0
    for _ in range(iters):
        pr = alpha * (pt @ pr + pr[dangling].sum() / n) + (1 - alpha) / n
    return pr


def _active_pagerank(adj: sp.csr_matrix, active: np.ndarray) -> np.ndarray:
    """PageRank over accounts that have transacted before the cutoff only.

    LEAKAGE GUARD: node codes cover every account in the dataset, including ones
    that first appear later. Running PageRank over all of them would let the
    number of FUTURE accounts change today's values through the teleport term.
    Scaled so 1.0 is the average active account; inactive accounts get 0.
    """
    out = np.zeros(adj.shape[0])
    idx = np.flatnonzero(active)
    if len(idx):
        out[idx] = _pagerank(adj[idx][:, idx]) * len(idx)
    return out


def node_features(g: AccountGraph) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-account structural features of one snapshot, plus SCC labels.

    flow_through is min(in, out) / max(in, out) of USD volume: 1 means the
    account passes on exactly what it receives (the mule signature), 0 means
    money only enters or only leaves. An isolated account has no flow at all,
    so its value is 0, not undefined.
    """
    a = (g.adj_count > 0).astype(np.float64).tocsr()
    in_deg = np.asarray(a.sum(axis=0)).ravel()
    out_deg = np.asarray(a.sum(axis=1)).ravel()
    in_w = np.asarray(g.adj_weight.sum(axis=0)).ravel()
    out_w = np.asarray(g.adj_weight.sum(axis=1)).ravel()
    hi = np.maximum(in_w, out_w)
    flow = np.divide(np.minimum(in_w, out_w), hi, out=np.zeros_like(hi), where=hi > 0)
    recip = np.asarray(a.multiply(a.T).sum(axis=1)).ravel()

    _, labels = connected_components(a, directed=True, connection="strong")
    scc_size = np.bincount(labels)[labels]

    # Mean degree of the accounts this account pays / is paid by: a cheap 2-hop
    # signal that stays bounded even next to the 169k-degree hub.
    nbr_out = np.divide(a @ out_deg, out_deg, out=np.zeros_like(out_deg), where=out_deg > 0)
    nbr_in = np.divide(a.T @ in_deg, in_deg, out=np.zeros_like(in_deg), where=in_deg > 0)

    return pd.DataFrame(
        {
            "in_degree": in_deg,
            "out_degree": out_deg,
            "in_tx": np.asarray(g.adj_count.sum(axis=0)).ravel(),
            "out_tx": np.asarray(g.adj_count.sum(axis=1)).ravel(),
            "log_in_weight": np.log1p(in_w),
            "log_out_weight": np.log1p(out_w),
            "degree_asymmetry": (out_deg - in_deg) / (out_deg + in_deg + 1.0),
            "flow_through": flow,
            "reciprocal_partners": recip,
            "pagerank": _active_pagerank(g.adj_count, in_deg + out_deg > 0),
            "log_scc_size": np.log(scc_size),
            "nbr_out_degree_mean": nbr_out,
            "nbr_in_degree_mean": nbr_in,
        }
    ).astype(np.float32), labels


def snapshot_starts(timestamp: pd.Series, delta: str) -> pd.DatetimeIndex:
    """Snapshot period boundaries covering the data: t0, t0 + delta, ..."""
    t0 = timestamp.min().floor(delta)
    return pd.date_range(t0, timestamp.max() + pd.Timedelta(delta), freq=delta)


def graph_features(
    df: pd.DataFrame, delta: str = "24h", codes: Sequence[int] | None = None
) -> pd.DataFrame:
    """Structural features for every transaction from its period's snapshot.

    For a transaction in [p, p + delta) the graph is built with t_cutoff = p, so
    it holds strictly earlier transactions only. Within a period features are up
    to delta stale; the behavioural features cover that gap at minute resolution.

    Returns src_* and dst_* copies of the node features plus edge features:
      closes_cycle         src and dst share a strongly connected component, so
                           dst can already reach src: this payment closes a loop
      reverse_edge_exists  dst has paid src before
      prior_pair_tx        earlier payments src -> dst
    """
    acct, _ = pd.factorize(pd.concat([df["src"], df["dst"]], ignore_index=True))
    src = acct[: len(df)]
    dst = acct[len(df) :]
    n_nodes = int(acct.max()) + 1
    amount = df["amount"].to_numpy()
    ts = df["timestamp"].reset_index(drop=True)

    out = np.full((len(df), 2 * len(GRAPH_FEATURES_NODE) + len(GRAPH_FEATURES_EDGE)), np.nan, np.float32)
    starts = snapshot_starts(ts, delta)
    for p, q in zip(starts[:-1], starts[1:], strict=True):
        rows = np.flatnonzero(((ts >= p) & (ts < q)).to_numpy())
        if len(rows) == 0:
            continue
        g = build_account_graph(src, dst, amount, ts, n_nodes, t_cutoff=p)
        nf, scc = node_features(g)
        nf_arr = nf.to_numpy()
        s, d = src[rows], dst[rows]
        k = len(GRAPH_FEATURES_NODE)
        out[rows, :k] = nf_arr[s]
        out[rows, k : 2 * k] = nf_arr[d]
        has_edges = g.adj_count.nnz > 0
        same_scc = (scc[s] == scc[d]) & (np.bincount(scc)[scc[s]] > 1) if has_edges else np.zeros(len(rows), bool)
        out[rows, 2 * k] = same_scc
        out[rows, 2 * k + 1] = np.asarray(g.adj_count[d, s]).ravel() > 0
        out[rows, 2 * k + 2] = np.asarray(g.adj_count[s, d]).ravel()

    cols = (
        [f"src_{c}" for c in GRAPH_FEATURES_NODE]
        + [f"dst_{c}" for c in GRAPH_FEATURES_NODE]
        + list(GRAPH_FEATURES_EDGE)
    )
    return pd.DataFrame(out, columns=cols, index=df.index)
