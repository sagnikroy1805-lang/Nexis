"""Fraud-ring detection: candidate generation + candidate scoring (rung 9).

Implements Concept Mastery §11.1 (transaction risk vs network risk), §11.2-11.3
(cycles, fan-in, fan-out as candidate features), §11.4 (the two-stage
architecture) and §11.5 (early warning at the ring level).

Stage 1 (unsupervised, high recall): keep transactions whose risk score is in
the top `candidate_quantile` of the window, connect their accounts, and take
connected components; components larger than `max_ring_size` are split by
Louvain community detection so one hub cannot swallow everything.

Stage 2 (supervised): featurise each candidate (size, density, risk, timing,
amount consistency, cycle present, fan-in/out shape, cross-bank share) and score
it with a classifier trained on candidates from an EARLIER window (the
validation fold). Candidate labels come from the documented patterns by
Jaccard overlap; they are never inputs to stage 1 or to any feature.

The output is a ranked list of account sets with scores ("network risk"), which
is what the evaluation (§11.4.3: ring-level precision/recall at Jaccard 0.5,
member-level precision/recall) and the API consume.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nexis.evaluation.metrics import jaccard

CANDIDATE_FEATURES = (
    "n_members",
    "n_edges",
    "density",
    "mean_score",
    "max_score",
    "min_score",
    "log_total_amount",
    "amount_cv",
    "span_hours",
    "has_cycle",
    "max_in_degree_share",
    "max_out_degree_share",
    "n_formats",
    "cross_bank_share",
)


@dataclass
class Ring:
    ring_id: str
    members: frozenset[str]
    tx_ids: list[str]
    first_seen: pd.Timestamp
    last_seen: pd.Timestamp
    features: dict[str, float]
    score: float = float("nan")


def generate_candidates(
    scored: pd.DataFrame,
    candidate_quantile: float = 0.99,
    max_ring_size: int = 50,
    min_ring_size: int = 3,
    seed: int = 0,
    prefix: str = "R",
) -> list[Ring]:
    """Stage 1. `scored` has tx_id, src, dst, timestamp, amount, score,
    payment_format, src_bank, dst_bank -- transactions of ONE evaluation window.
    """
    tau = float(scored["score"].quantile(candidate_quantile))
    hot = scored.loc[scored["score"] >= tau]
    g = nx.Graph()
    g.add_edges_from(zip(hot["src"], hot["dst"], strict=True))
    groups: list[set[str]] = []
    for comp in nx.connected_components(g):
        if len(comp) <= max_ring_size:
            groups.append(set(comp))
            continue
        sub = g.subgraph(comp)
        for c in nx.community.louvain_communities(sub, seed=seed):
            groups.append(set(c))

    by_member = hot.assign(_i=np.arange(len(hot)))
    rings: list[Ring] = []
    for k, members in enumerate(sorted(groups, key=lambda s: (-len(s), sorted(s)[0]))):
        if len(members) < min_ring_size:
            continue
        edges = by_member.loc[by_member["src"].isin(members) & by_member["dst"].isin(members)]
        if len(edges) == 0:
            continue
        rings.append(
            Ring(
                ring_id=f"{prefix}-{k:04d}",
                members=frozenset(members),
                tx_ids=edges["tx_id"].tolist(),
                first_seen=edges["timestamp"].min(),
                last_seen=edges["timestamp"].max(),
                features=candidate_features(edges, members),
            )
        )
    return rings


def candidate_features(edges: pd.DataFrame, members: set[str]) -> dict[str, float]:
    """Shape, risk and timing of one candidate subgraph (§11.4.2)."""
    n = len(members)
    dg = nx.DiGraph()
    dg.add_edges_from(zip(edges["src"], edges["dst"], strict=True))
    amounts = edges["amount"].to_numpy(dtype=float)
    indeg = np.array([d for _, d in dg.in_degree()], dtype=float)
    outdeg = np.array([d for _, d in dg.out_degree()], dtype=float)
    m = max(dg.number_of_edges(), 1)
    try:
        nx.find_cycle(dg)
        has_cycle = 1.0
    except nx.NetworkXNoCycle:
        has_cycle = 0.0
    span = (edges["timestamp"].max() - edges["timestamp"].min()).total_seconds() / 3600
    return {
        "n_members": float(n),
        "n_edges": float(len(edges)),
        "density": dg.number_of_edges() / max(n * (n - 1), 1),
        "mean_score": float(edges["score"].mean()),
        "max_score": float(edges["score"].max()),
        "min_score": float(edges["score"].min()),
        "log_total_amount": float(np.log1p(amounts.sum())),
        "amount_cv": float(amounts.std() / max(amounts.mean(), 1e-9)),
        "span_hours": float(span),
        "has_cycle": has_cycle,
        "max_in_degree_share": float(indeg.max() / m) if len(indeg) else 0.0,
        "max_out_degree_share": float(outdeg.max() / m) if len(outdeg) else 0.0,
        "n_formats": float(edges["payment_format"].nunique()),
        "cross_bank_share": float((edges["src_bank"].astype(str) != edges["dst_bank"].astype(str)).mean()),
    }


def truth_rings(patterns: pd.DataFrame, transactions: pd.DataFrame, window_tx: set[str]) -> dict[str, dict]:
    """Ground-truth account sets for patterns active in an evaluation window.

    Members are the accounts of the pattern's transactions that fall inside the
    window (what the window can possibly reveal). t_start / t_end come from the
    FULL pattern, including rows in the dropped tail, so early-warning time is
    measured against the pattern's true completion.
    """
    tx = transactions.set_index("tx_id")[["src", "dst"]]
    full = patterns.groupby("pattern_id").agg(
        t_start=("timestamp", "min"), t_end=("timestamp", "max"), typology=("typology", "first")
    )
    inside = patterns.loc[patterns["tx_id"].isin(window_tx)]
    out: dict[str, dict] = {}
    for pid, grp in inside.groupby("pattern_id"):
        rows = tx.loc[grp["tx_id"]]
        out[f"P-{int(pid):04d}"] = {
            "members": set(rows["src"]) | set(rows["dst"]),
            "t_start": full.loc[pid, "t_start"],
            "t_end": full.loc[pid, "t_end"],
            "typology": str(full.loc[pid, "typology"]),
            "n_tx_in_window": len(grp),
        }
    return out


def label_candidates(rings: Sequence[Ring], truth: dict[str, dict], theta: float = 0.5) -> np.ndarray:
    """1 if a candidate overlaps some true pattern with Jaccard >= theta."""
    truth_sets = [t["members"] for t in truth.values()]
    return np.array(
        [float(any(jaccard(set(r.members), t) >= theta for t in truth_sets)) for r in rings]
    )


@dataclass
class RingScorer:
    """Stage 2: logistic regression over candidate features.

    Deliberately simple: with a few hundred candidates per window, a linear model
    is what the data can support, and its weights are explainable to an analyst.
    Falls back to mean member risk when the training window has one class only.
    """

    theta: float = 0.5
    model: object | None = None
    fallback: bool = False
    weights: dict[str, float] = field(default_factory=dict)

    def fit(self, rings: Sequence[Ring], truth: dict[str, dict]) -> RingScorer:
        y = label_candidates(rings, truth, self.theta)
        if len(rings) < 5 or y.min() == y.max():
            self.fallback = True
            return self
        x = np.array([[r.features[f] for f in CANDIDATE_FEATURES] for r in rings])
        self.model = make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)
        ).fit(x, y)
        coef = self.model[-1].coef_[0]  # type: ignore[index]
        self.weights = dict(zip(CANDIDATE_FEATURES, map(float, coef), strict=True))
        return self

    def score(self, rings: Sequence[Ring]) -> np.ndarray:
        if not rings:
            return np.zeros(0)
        if self.fallback or self.model is None:
            return np.array([r.features["mean_score"] for r in rings])
        x = np.array([[r.features[f] for f in CANDIDATE_FEATURES] for r in rings])
        return self.model.predict_proba(x)[:, 1]  # type: ignore[attr-defined]


def window_frame(df: pd.DataFrame, scores: pd.DataFrame, fold: str) -> pd.DataFrame:
    """Join a fold's scores to transaction fields needed for ring detection."""
    s = scores.loc[scores["fold"] == fold, ["tx_id", "score"]]
    cols = ["tx_id", "src", "dst", "timestamp", "amount", "payment_format", "src_bank", "dst_bank"]
    return df[cols].merge(s, on="tx_id", how="inner")
