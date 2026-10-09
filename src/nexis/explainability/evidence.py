"""Evidence packets: everything an alert rests on, and nothing else.

Implements Concept Mastery §12.1-12.2 (SHAP, local explanation), §12.4 (the
evidence graph) and §12.5 (explanation is not causation), plus PROJECT_RULES.md
rules 5 and 6.

Rule 6: an alert and its packet are written in the same database transaction,
and `source_record_ids` is never empty -- `build_packet` raises if it would be.
Rule 5: the packet reports what the model weighted and what the system observed.
It never contains a typology label (that is label-derived) or a conclusion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nexis.explainability.labels import feature_label

LANGUAGE_NOTE = (
    "Risk scores reflect model output on recorded data. They are not findings of wrongdoing."
)


@dataclass
class FeatureContribution:
    feature: str
    label: str
    value: float | None
    contribution: float


@dataclass
class EvidencePacket:
    alert_id: int | None
    tx_id: str
    model_version: str
    risk_score: float
    threshold: float
    generated_at: str
    source_record_ids: list[str]
    transaction: dict[str, Any]
    feature_contributions: list[FeatureContribution]
    rule_hits: list[str]
    account_context: dict[str, Any]
    graph_evidence: dict[str, Any]
    ring_candidate: dict[str, Any] | None = None
    language_note: str = LANGUAGE_NOTE
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceError(ValueError):
    """Raised when an alert would be created without evidence (rule 6)."""


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else v


def top_contributions(
    shap_row: np.ndarray, values: pd.Series, columns: Sequence[str], k: int = 8
) -> list[FeatureContribution]:
    """The k features the model weighted most for this transaction, by |SHAP|."""
    order = np.argsort(-np.abs(shap_row))[:k]
    return [
        FeatureContribution(
            feature=columns[i],
            label=feature_label(columns[i]),
            value=_num(values.iloc[i]),
            contribution=float(shap_row[i]),
        )
        for i in order
    ]


def graph_evidence(
    tx: pd.Series,
    history: pd.DataFrame,
    scores: pd.Series | None,
    max_edges: int = 40,
) -> dict[str, Any]:
    """Earlier transactions touching the two accounts, strictly before this one.

    LEAKAGE GUARD: only rows with timestamp < the alert's transaction time are
    eligible, so the evidence an analyst sees is exactly what was knowable when
    the alert fired. Edges are ranked by their own risk score, then recency.
    """
    accounts = {tx["src"], tx["dst"]}
    past = history.loc[
        (history["timestamp"] < tx["timestamp"])
        & (history["src"].isin(accounts) | history["dst"].isin(accounts))
    ].copy()
    past["risk_score"] = scores.reindex(past["tx_id"]).to_numpy() if scores is not None else np.nan
    past = past.sort_values(["risk_score", "timestamp"], ascending=[False, False], na_position="last")
    past = past.head(max_edges)
    edges = [
        {
            "tx_id": r.tx_id,
            "src": r.src,
            "dst": r.dst,
            "amount_usd": float(r.amount),
            "occurred_at": pd.Timestamp(r.timestamp).isoformat(),
            "risk_score": _num(r.risk_score),
            "payment_format": str(r.payment_format),
            "important": bool(_num(r.risk_score) is not None and r.risk_score >= 0.5),
        }
        for r in past.itertuples()
    ]
    node_ids = {tx["src"], tx["dst"], *past["src"], *past["dst"]}
    node_risk: dict[str, float] = {}
    for e in edges:
        for n in (e["src"], e["dst"]):
            if e["risk_score"] is not None:
                node_risk[n] = max(node_risk.get(n, 0.0), e["risk_score"])
    nodes = [
        {"id": n, "kind": "account", "risk": node_risk.get(n), "is_focus": n in accounts}
        for n in sorted(node_ids)
    ]
    return {"nodes": nodes, "edges": edges}


def build_packet(
    *,
    tx: pd.Series,
    features: pd.Series,
    shap_row: np.ndarray,
    columns: Sequence[str],
    risk_score: float,
    threshold: float,
    model_version: str,
    rule_hits: Sequence[str],
    history: pd.DataFrame,
    history_scores: pd.Series | None,
    ring: dict[str, Any] | None = None,
    alert_id: int | None = None,
) -> EvidencePacket:
    """Assemble the packet for one alert. Raises EvidenceError if it would be empty."""
    graph = graph_evidence(tx, history, history_scores)
    source_ids = [str(tx["tx_id"]), *(e["tx_id"] for e in graph["edges"])]
    if not source_ids or not source_ids[0]:
        raise EvidenceError("an alert needs at least its own transaction as evidence")

    def ctx(prefix: str) -> dict[str, Any]:
        keys = {
            "src": {
                "prior_sends": "src_out_n_hist",
                "prior_receipts": "src_in_n_hist",
                "distinct_counterparties": "src_n_counterparties",
                "secs_since_last_receipt": "src_secs_since_in",
                "secs_since_last_send": "src_secs_since_out",
            },
            "dst": {
                "prior_receipts": "dst_in_n_hist",
                "prior_sends": "dst_out_n_hist",
                "distinct_counterparties": "dst_n_counterparties",
            },
        }[prefix]
        out: dict[str, Any] = {"account": str(tx[prefix])}
        out |= {k: _num(features.get(v)) for k, v in keys.items()}
        return out

    return EvidencePacket(
        alert_id=alert_id,
        tx_id=str(tx["tx_id"]),
        model_version=model_version,
        risk_score=float(risk_score),
        threshold=float(threshold),
        generated_at=pd.Timestamp(tx["timestamp"]).isoformat(),
        source_record_ids=source_ids,
        transaction={
            "tx_id": str(tx["tx_id"]),
            "occurred_at": pd.Timestamp(tx["timestamp"]).isoformat(),
            "src_account": str(tx["src"]),
            "dst_account": str(tx["dst"]),
            "amount_usd": float(tx["amount"]),
            "amount_paid": float(tx["amount_paid"]),
            "pay_currency": str(tx["pay_currency"]),
            "payment_format": str(tx["payment_format"]),
            "is_cross_bank": bool(tx["is_cross_bank"]),
            "is_cross_currency": bool(tx["is_cross_currency"]),
        },
        feature_contributions=top_contributions(shap_row, features[list(columns)], columns),
        rule_hits=list(rule_hits),
        account_context={"src": ctx("src"), "dst": ctx("dst")},
        graph_evidence=graph,
        ring_candidate=ring,
    )
