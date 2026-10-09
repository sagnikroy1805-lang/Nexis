"""Alert creation: the only place alerts are written, always with evidence.

PROJECT_RULES.md rule 6: alerts and their evidence packets are written in the same
database transaction, and an alert with no source_record_ids is a bug.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from nexis.db.models import Alert, Evidence
from nexis.explainability.evidence import EvidenceError, EvidencePacket


def create_alert_with_evidence(
    session: Session, packet: EvidencePacket, created_at: datetime
) -> Alert:
    """Add an alert and its packet to the session's open transaction.

    The caller commits (session_scope). If anything fails before the commit,
    neither row exists. A packet without source ids raises before any write.
    """
    if not packet.source_record_ids or not all(packet.source_record_ids):
        raise EvidenceError(f"alert for {packet.tx_id} has no source_record_ids")

    top: list[dict[str, Any]] = [
        {"feature": c.feature, "label": c.label, "contribution": c.contribution}
        for c in packet.feature_contributions
        if c.contribution > 0
    ][:3]
    alert = Alert(
        tx_id=packet.tx_id,
        model_version=packet.model_version,
        risk_score=packet.risk_score,
        threshold=packet.threshold,
        created_at=created_at,
        status="open",
        top_reasons=top,
    )
    session.add(alert)
    session.flush()  # assigns alert_id inside the same transaction
    packet.alert_id = alert.alert_id
    session.add(
        Evidence(
            alert_id=alert.alert_id,
            packet=packet.to_dict(),
            source_ids=list(packet.source_record_ids),
        )
    )
    return alert
