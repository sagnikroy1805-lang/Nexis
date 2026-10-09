"""Leak-free snapshot graphs for GNN edge classification (rungs 6-8).

Implements Concept Mastery §7.1-7.2 (homogeneous and heterogeneous graph
modelling), §7.6 (building the heterogeneous graph), §9.1 (time snapshots) and
the guard of §9.4 form 4 (message passing over future edges).

For every snapshot period [p, p + delta):
  - the GRAPH holds transactions with timestamp < p only (t_cutoff = p);
  - the TARGETS are the transactions inside [p, p + delta).
A target is therefore never in the graph that scores it, and neither is anything
later. `assert_snapshot_is_causal` checks both, for every snapshot built.

This module is numpy-only; models/gnn turns snapshots into PyG objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nexis.graphs.homogeneous import build_account_graph, node_features, snapshot_starts

EPOCH = pd.Timestamp("1970-01-01")


def _epoch_seconds(ts: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    return ((pd.Series(ts) - EPOCH) // pd.Timedelta("1s")).to_numpy(dtype=np.int64)


@dataclass
class Snapshot:
    """One period: the graph as of p, and the transactions to score in [p, p+Δ)."""

    period_start: pd.Timestamp
    period_end: pd.Timestamp
    node_x: np.ndarray  # (n_nodes, F) structural features of graph < p
    edge_index: np.ndarray  # (2, E) unique account pairs with a payment < p
    edge_weight: np.ndarray  # (E,) log1p(number of payments on the pair)
    edge_time: np.ndarray  # (E,) epoch seconds of the pair's LATEST payment < p
    # Accounts with at least one payment before p. Anything that pools over
    # accounts (bank nodes, normalisers) must use only these: the rest exist
    # only in the future, and even their COUNT would leak.
    active: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    # Heterogeneous view: one edge list per payment format, same cutoff.
    typed_edges: dict[str, np.ndarray] = field(default_factory=dict)
    typed_weight: dict[str, np.ndarray] = field(default_factory=dict)
    typed_time: dict[str, np.ndarray] = field(default_factory=dict)
    recent_edges: np.ndarray | None = None  # pairs active in [p - Δ, p), for temporal models
    recent_weight: np.ndarray | None = None
    recent_typed_edges: dict[str, np.ndarray] = field(default_factory=dict)
    recent_typed_weight: dict[str, np.ndarray] = field(default_factory=dict)
    recent_typed_time: dict[str, np.ndarray] = field(default_factory=dict)
    target_rows: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))


@dataclass
class SnapshotSet:
    """All snapshots over a dataset, plus the shared account/bank coding."""

    snapshots: list[Snapshot]
    n_accounts: int
    n_banks: int
    account_bank: np.ndarray  # (n_accounts,) bank code of each account
    src: np.ndarray  # (n_rows,) account code of each transaction's sender
    dst: np.ndarray
    formats: list[str]
    delta: str


def assert_snapshot_is_causal(snap: Snapshot, ts_epoch: np.ndarray) -> None:
    """Fail loudly if any edge or target violates the snapshot's time contract."""
    cutoff = int((snap.period_start - EPOCH) // pd.Timedelta("1s"))
    end = int((snap.period_end - EPOCH) // pd.Timedelta("1s"))
    times = [snap.edge_time, *snap.typed_time.values(), *snap.recent_typed_time.values()]
    for t in times:
        if len(t):
            assert int(t.max()) < cutoff, "LEAK: snapshot graph has an edge at/after t_cutoff"
    if len(snap.target_rows):
        tt = ts_epoch[snap.target_rows]
        assert tt.min() >= cutoff and tt.max() < end, "target outside its period"


def _pairs(
    s: np.ndarray, d: np.ndarray, t: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique (s, d) pairs with payment count and latest time."""
    if len(s) == 0:
        return np.zeros((2, 0), np.int64), np.zeros(0, np.float32), np.zeros(0, np.int64)
    key = s.astype(np.int64) * n + d
    order = np.argsort(key, kind="stable")
    key, t = key[order], t[order]
    uniq, start, counts = np.unique(key, return_index=True, return_counts=True)
    latest = np.maximum.reduceat(t, start)
    ei = np.vstack([uniq // n, uniq % n]).astype(np.int64)
    return ei, np.log1p(counts).astype(np.float32), latest.astype(np.int64)


def _node_matrix(features: pd.DataFrame) -> np.ndarray:
    """Compress heavy-tailed counts and sums; keep ratios as they are."""
    x = features.to_numpy(dtype=np.float64)
    return (np.sign(x) * np.log1p(np.abs(x))).astype(np.float32)


def build_snapshots(
    df: pd.DataFrame,
    delta: str = "24h",
    with_typed: bool = True,
    with_recent: bool = True,
) -> SnapshotSet:
    """Build every snapshot over df (standard schema, any order).

    Node codes are shared across snapshots so a node's embedding refers to the
    same account at every time. That coding is a lookup table, not a statistic:
    knowing that an account EXISTS later gives a snapshot nothing, because an
    account with no edges before p is an isolated node with all-zero features.
    """
    acct, _ = pd.factorize(pd.concat([df["src"], df["dst"]], ignore_index=True))
    src = acct[: len(df)].astype(np.int64)
    dst = acct[len(df) :].astype(np.int64)
    n = int(acct.max()) + 1

    bank_of = pd.concat(
        [
            pd.Series(df["src_bank"].astype(str).to_numpy(), index=src),
            pd.Series(df["dst_bank"].astype(str).to_numpy(), index=dst),
        ]
    )
    bank_of = bank_of[~bank_of.index.duplicated()].sort_index()
    bank_codes, _ = pd.factorize(bank_of)
    account_bank = np.zeros(n, np.int64)
    account_bank[bank_of.index.to_numpy()] = bank_codes

    ts = df["timestamp"].reset_index(drop=True)
    t_epoch = _epoch_seconds(ts)
    amount = df["amount"].to_numpy()
    fmt = df["payment_format"].astype(str).to_numpy()
    formats = sorted(set(fmt))

    snaps: list[Snapshot] = []
    starts = snapshot_starts(ts, delta)
    step = pd.Timedelta(delta)
    for p, q in zip(starts[:-1], starts[1:], strict=True):
        cutoff = int((p - EPOCH) // pd.Timedelta("1s"))
        past = t_epoch < cutoff  # LEAKAGE GUARD: strictly before the period start
        targets = np.flatnonzero((t_epoch >= cutoff) & (t_epoch < int((q - EPOCH) // pd.Timedelta("1s"))))
        if len(targets) == 0:
            continue
        g = build_account_graph(src, dst, amount, ts, n, t_cutoff=p)
        nf, _ = node_features(g)
        ei, w, et = _pairs(src[past], dst[past], t_epoch[past], n)
        snap = Snapshot(p, q, _node_matrix(nf), ei, w, et, target_rows=targets)
        snap.active = (
            np.asarray(g.adj_count.sum(axis=0)).ravel() + np.asarray(g.adj_count.sum(axis=1)).ravel()
        ) > 0
        if with_typed:
            for f in formats:
                m = past & (fmt == f)
                snap.typed_edges[f], snap.typed_weight[f], snap.typed_time[f] = _pairs(
                    src[m], dst[m], t_epoch[m], n
                )
        if with_recent:
            lo = int((p - step - EPOCH) // pd.Timedelta("1s"))
            m = past & (t_epoch >= lo)
            snap.recent_edges, snap.recent_weight, _ = _pairs(src[m], dst[m], t_epoch[m], n)
            for f in formats:
                mf = m & (fmt == f)
                (
                    snap.recent_typed_edges[f],
                    snap.recent_typed_weight[f],
                    snap.recent_typed_time[f],
                ) = _pairs(src[mf], dst[mf], t_epoch[mf], n)
        assert_snapshot_is_causal(snap, t_epoch)
        snaps.append(snap)

    return SnapshotSet(
        snapshots=snaps,
        n_accounts=n,
        n_banks=int(bank_codes.max()) + 1 if len(bank_codes) else 0,
        account_bank=account_bank,
        src=src,
        dst=dst,
        formats=formats,
        delta=delta,
    )
