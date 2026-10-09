"""Graph construction and structural features (rung 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nexis.graphs.diagnostics import degree_summary, edge_homophily
from nexis.graphs.homogeneous import build_account_graph, graph_features, node_features


def _tx(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "src": [r[0] for r in rows],
            "dst": [r[1] for r in rows],
            "timestamp": pd.to_datetime([r[2] for r in rows]),
            "amount": [r[3] for r in rows],
        }
    )


def test_build_requires_t_cutoff():
    with pytest.raises(ValueError, match="t_cutoff"):
        build_account_graph(
            np.array([0]), np.array([1]), np.array([1.0]),
            pd.Series(pd.to_datetime(["2026-01-01"])), 2, t_cutoff=None,  # type: ignore[arg-type]
        )


def test_graph_excludes_edges_at_or_after_cutoff():
    ts = pd.Series(pd.to_datetime(["2026-01-01 09:00", "2026-01-01 10:00", "2026-01-01 11:00"]))
    g = build_account_graph(
        np.array([0, 1, 2]), np.array([1, 2, 0]), np.array([5.0, 6.0, 7.0]),
        ts, 3, t_cutoff=pd.Timestamp("2026-01-01 10:00"),
    )
    assert g.adj_count.nnz == 1, "only the 09:00 edge is strictly before 10:00"
    assert g.max_edge_time < g.t_cutoff


def test_flow_through_and_isolated_nodes():
    """A receives 100 and sends 90: flow 0.9. An isolated node has no flow: 0."""
    ts = pd.Series(pd.to_datetime(["2026-01-01 09:00", "2026-01-01 09:30"]))
    g = build_account_graph(
        np.array([1, 0]), np.array([0, 2]), np.array([100.0, 90.0]),
        ts, 4, t_cutoff=pd.Timestamp("2026-01-02"),
    )
    nf, _ = node_features(g)
    assert nf.loc[0, "flow_through"] == pytest.approx(0.9)
    assert nf.loc[3, "flow_through"] == 0.0
    assert nf.loc[0, "in_degree"] == 1 and nf.loc[0, "out_degree"] == 1


def test_closes_cycle_uses_only_earlier_snapshot():
    """A->B, B->C on day 1; C->A on day 2 closes the loop A->B->C->A.

    On day 2 the day-1 graph has A, B, C in different SCCs (no loop yet), so
    closes_cycle is 0. A second C->A on day 3 sees the loop and gets 1.
    """
    df = _tx(
        [
            ("A", "B", "2026-01-01 09:00", 10.0),
            ("B", "C", "2026-01-01 10:00", 10.0),
            ("C", "A", "2026-01-02 09:00", 10.0),
            ("C", "A", "2026-01-03 09:00", 10.0),
        ]
    )
    f = graph_features(df, delta="24h")
    assert f["closes_cycle"].tolist() == [0, 0, 0, 1]
    assert f["reverse_edge_exists"].tolist() == [0, 0, 0, 0]
    assert f["prior_pair_tx"].tolist() == [0, 0, 0, 1]


def test_same_period_transactions_do_not_see_each_other():
    """Two payments on the same day: neither appears in the other's graph."""
    df = _tx(
        [
            ("A", "B", "2026-01-01 09:00", 10.0),
            ("A", "C", "2026-01-01 23:00", 10.0),
        ]
    )
    f = graph_features(df, delta="24h")
    assert f["src_out_degree"].tolist() == [0, 0]


def test_diagnostics():
    ei = np.array([[0, 1, 2], [1, 2, 3]])
    assert edge_homophily(ei, np.array([0, 0, 1, 1])) == pytest.approx(2 / 3)
    d = degree_summary(ei, 5)
    assert d["isolated_fraction"] == pytest.approx(0.2)
