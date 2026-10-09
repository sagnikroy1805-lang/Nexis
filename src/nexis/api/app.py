"""NEXIS analyst API (contract: docs/api_contract.md).

Implements Concept Mastery §19.1 (REST design), §19.2 (validation) and §19.3
(the model service), with CLAUDE.md rules 5 and 6 as hard constraints:
  - every alert returned carries an evidence packet written with it;
  - responses describe risk scores and observations, never conclusions;
  - there is no endpoint that acts on an account. Dispositions are analyst
    feedback only.

Run:  uvicorn nexis.api.app:app --port 8000
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from nexis.db.models import Alert, DriftEvent, DriftWindow, ModelRun, Ring, Transaction
from nexis.db.session import database_url, get_engine, init_db, load_dotenv
from nexis.explainability.evidence import LANGUAGE_NOTE
from nexis.investigation.investigator import Investigator

load_dotenv()  # before anything reads NEXIS_* settings

RESULTS_DIR = Path("results")
FRONTEND_DIST = Path("frontend") / "dist"

app = FastAPI(
    title="NEXIS analyst API",
    version="1.0",
    description="Risk scores and evidence for analyst review. Scores are model output, "
    "not findings of wrongdoing. No endpoint acts on an account.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_session():  # FastAPI dependency
    init_db()
    session = sessionmaker(bind=get_engine(), expire_on_commit=False)()
    try:
        yield session
    finally:
        session.close()


_investigator = Investigator()


# ---------------------------------------------------------------- helpers ----
def _iso(d: datetime | None) -> str | None:
    return d.isoformat() if d else None


def _tx_dict(t: Transaction) -> dict[str, Any]:
    return {
        "tx_id": t.tx_id,
        "occurred_at": _iso(t.occurred_at),
        "src_account": t.src_account,
        "dst_account": t.dst_account,
        "amount_usd": t.amount_usd,
        "amount_paid": t.amount_paid,
        "pay_currency": t.pay_currency,
        "payment_format": t.payment_format,
        "is_cross_bank": t.is_cross_bank,
        "is_cross_currency": t.is_cross_currency,
        "risk_score": t.risk_score,
    }


def _serving(session: Session) -> ModelRun | None:
    return session.scalars(select(ModelRun).where(ModelRun.is_serving.is_(True))).first()


def _alert_item(a: Alert, rank: int | None) -> dict[str, Any]:
    t = a.transaction
    return {
        "alert_id": a.alert_id,
        "tx_id": a.tx_id,
        "occurred_at": _iso(t.occurred_at),
        "created_at": _iso(a.created_at),
        "src_account": t.src_account,
        "dst_account": t.dst_account,
        "amount_usd": t.amount_usd,
        "payment_format": t.payment_format,
        "risk_score": a.risk_score,
        "rank": rank,
        "status": a.status,
        "top_reasons": a.top_reasons or [],
    }


def _rank_of(session: Session, a: Alert) -> int:
    return int(session.scalar(select(func.count()).where(Alert.risk_score > a.risk_score)) or 0) + 1


def _get_alert(session: Session, alert_id: int) -> Alert:
    a = session.get(Alert, alert_id)
    if a is None:
        raise HTTPException(404, f"alert {alert_id} not found")
    if a.evidence is None:  # rule 6: should be impossible; refuse to serve it
        raise HTTPException(500, f"alert {alert_id} has no evidence packet")
    return a


# ---------------------------------------------------------------- routes -----
@app.get("/api/health")
def health(session: Session = Depends(get_session)) -> dict[str, Any]:
    run = _serving(session)
    return {
        "status": "ok",
        "model_version": run.model_version if run else None,
        "db": get_engine().dialect.name if database_url() else "unknown",
    }


@app.get("/api/summary")
def summary(session: Session = Depends(get_session)) -> dict[str, Any]:
    run = _serving(session)
    scored = select(Transaction).where(Transaction.risk_score.is_not(None)).subquery()
    start, end, n = session.execute(
        select(func.min(scored.c.occurred_at), func.max(scored.c.occurred_at), func.count())
    ).one()
    total = session.scalar(select(func.count()).select_from(Alert)) or 0
    open_ = session.scalar(select(func.count()).where(Alert.status == "open")) or 0
    metrics = (run.metrics or {}) if run else {}
    return {
        "model_version": run.model_version if run else None,
        "dataset": metrics.get("dataset", "IBM AML HI-Small"),
        "replay_period": {"start": _iso(start), "end": _iso(end)},
        "transactions_scored": int(n or 0),
        "alerts_total": int(total),
        "alerts_open": int(open_),
        "alert_budget": metrics.get("alert_budget"),
        "threshold": metrics.get("threshold"),
        "metrics": {
            k: metrics.get(k)
            for k in ("pr_auc", "pr_auc_std", "prevalence", "recall_at_budget", "precision_at_budget")
        },
    }


@app.get("/api/alerts")
def list_alerts(
    status: Literal["open", "escalated", "dismissed", "needs_info", "all"] = "open",
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    sort: Literal["risk_score", "occurred_at"] = "risk_score",
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    q = select(Alert).join(Transaction)
    if status != "all":
        q = q.where(Alert.status == status)
    total = session.scalar(select(func.count()).select_from(q.subquery())) or 0
    order = Alert.risk_score.desc() if sort == "risk_score" else Transaction.occurred_at.desc()
    rows = session.scalars(q.order_by(order).offset(offset).limit(limit)).all()
    return {"total": int(total), "items": [_alert_item(a, _rank_of(session, a)) for a in rows]}


@app.get("/api/alerts/{alert_id}")
def get_alert(alert_id: int, session: Session = Depends(get_session)) -> dict[str, Any]:
    a = _get_alert(session, alert_id)
    packet = dict(a.evidence.packet)
    packet.setdefault("language_note", LANGUAGE_NOTE)
    return {"alert": _alert_item(a, _rank_of(session, a)), "evidence": packet}


class DispositionIn(BaseModel):
    disposition: Literal["escalated", "dismissed", "needs_info"]
    note: str = Field("", max_length=4000)


@app.post("/api/alerts/{alert_id}/disposition")
def set_disposition(
    alert_id: int, body: DispositionIn, session: Session = Depends(get_session)
) -> dict[str, Any]:
    """Record analyst feedback. This changes the alert's status and nothing else."""
    a = _get_alert(session, alert_id)
    a.status = body.disposition
    a.disposition_note = body.note
    a.disposed_at = datetime.utcnow()
    session.commit()
    return _alert_item(a, _rank_of(session, a))


@app.post("/api/alerts/{alert_id}/investigate")
def investigate(alert_id: int, session: Session = Depends(get_session)) -> dict[str, Any]:
    a = _get_alert(session, alert_id)
    result = _investigator.investigate(dict(a.evidence.packet), alert_id=alert_id)
    return result.to_dict()


@app.get("/api/accounts/{account_id}/network")
def network(
    account_id: str,
    until: datetime | None = None,
    hops: int = Query(2, ge=1, le=2),
    limit: int = Query(300, ge=1, le=2000),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Transactions around an account strictly before `until` (leakage guard).

    The LIMIT is the hub-explosion guard of §17.2.1: two hops from an account
    that paid a large hub would otherwise return hundreds of thousands of rows.
    """
    if until is None:
        until = session.scalar(select(func.max(Transaction.occurred_at))) or datetime.utcnow()
    first = session.scalars(
        select(Transaction)
        .where(
            or_(Transaction.src_account == account_id, Transaction.dst_account == account_id),
            Transaction.occurred_at < until,
        )
        .order_by(Transaction.occurred_at.desc())
        .limit(limit)
    ).all()
    edges = {t.tx_id: t for t in first}
    if hops == 2 and len(edges) < limit:
        nbrs = ({t.src_account for t in first} | {t.dst_account for t in first}) - {account_id}
        budget = limit - len(edges)
        if nbrs and budget > 0:
            second = session.scalars(
                select(Transaction)
                .where(
                    or_(Transaction.src_account.in_(nbrs), Transaction.dst_account.in_(nbrs)),
                    Transaction.occurred_at < until,
                )
                .order_by(Transaction.risk_score.desc().nulls_last(), Transaction.occurred_at.desc())
                .limit(budget)
            ).all()
            for t in second:
                edges.setdefault(t.tx_id, t)
    truncated = len(edges) >= limit
    risk: dict[str, float] = {}
    for t in edges.values():
        if t.risk_score is not None:
            for n in (t.src_account, t.dst_account):
                risk[n] = max(risk.get(n, 0.0), t.risk_score)
    node_ids = {account_id} | {n for t in edges.values() for n in (t.src_account, t.dst_account)}
    return {
        "center": account_id,
        "until": until.isoformat(),
        "truncated": truncated,
        "nodes": [
            {"id": n, "kind": "account", "risk": risk.get(n), "is_center": n == account_id}
            for n in sorted(node_ids)
        ],
        "edges": [
            {
                "tx_id": t.tx_id,
                "src": t.src_account,
                "dst": t.dst_account,
                "amount_usd": t.amount_usd,
                "occurred_at": _iso(t.occurred_at),
                "risk_score": t.risk_score,
                "payment_format": t.payment_format,
            }
            for t in edges.values()
        ],
    }


@app.get("/api/accounts/{account_id}/transactions")
def account_transactions(
    account_id: str,
    until: datetime | None = None,
    limit: int = Query(100, ge=1, le=1000),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    q = select(Transaction).where(
        or_(Transaction.src_account == account_id, Transaction.dst_account == account_id)
    )
    if until is not None:
        q = q.where(Transaction.occurred_at < until)
    rows = session.scalars(q.order_by(Transaction.occurred_at.desc()).limit(limit)).all()
    return {"items": [_tx_dict(t) for t in rows]}


@app.get("/api/rings")
def rings(limit: int = Query(50, ge=1, le=500), session: Session = Depends(get_session)) -> dict[str, Any]:
    rows = session.scalars(select(Ring).order_by(Ring.score.desc()).limit(limit)).all()
    return {
        "items": [
            {
                "ring_id": r.ring_id,
                "members": r.members,
                "n_transactions": len(r.tx_ids),
                "score": r.score,
                "first_seen": _iso(r.first_seen),
                "last_seen": _iso(r.last_seen),
                "tx_ids": r.tx_ids,
            }
            for r in rows
        ]
    }


@app.get("/api/drift")
def drift(session: Session = Depends(get_session)) -> dict[str, Any]:
    windows = session.scalars(select(DriftWindow).order_by(DriftWindow.start)).all()
    events = session.scalars(select(DriftEvent).order_by(DriftEvent.detected_at)).all()
    run = _serving(session)
    ref = (run.metrics or {}).get("drift_reference", {}) if run else {}
    return {
        "reference": {"start": ref.get("start"), "end": ref.get("end")},
        "windows": [w.report for w in windows],
        "events": [
            {
                "detected_at": _iso(e.detected_at),
                "metric": e.metric,
                "value": e.value,
                "level": e.level,
                "action": e.action,
            }
            for e in events
        ],
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    """The modelling ladder from results/: the latest run of each model name."""
    latest: dict[str, tuple[float, dict[str, Any]]] = {}
    prevalence = None
    for manifest_path in RESULTS_DIR.glob("*/manifest.json"):
        summary_path = manifest_path.with_name("summary.json")
        if not summary_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rung = manifest.get("config", {}).get("rung")
        for model, _ in summary.get("pr_auc", {}).items():
            row = {
                "name": model,
                "rung": rung,
                "pr_auc": summary["pr_auc"][model],
                "pr_auc_std": summary.get("pr_auc_std", {}).get(model),
                "roc_auc": summary.get("roc_auc", {}).get(model),
                "roc_auc_std": summary.get("roc_auc_std", {}).get(model),
                "recall_at_fpr_1e3": summary.get("recall_at_fpr_1e3", {}).get(model),
                "recall_at_fpr_1e3_std": summary.get("recall_at_fpr_1e3_std", {}).get(model),
                "precision_at_budget": summary.get("precision_at_budget", {}).get(model),
                "precision_at_budget_std": summary.get("precision_at_budget_std", {}).get(model),
                "n_seeds": len(manifest.get("seeds", [])),
                "working_tree_dirty": manifest.get("working_tree_dirty"),
            }
            prevalence = summary.get("prevalence", {}).get(model, prevalence)
            ts = float(manifest.get("timestamp", 0))
            if model not in latest or ts > latest[model][0]:
                latest[model] = (ts, row)
    items = sorted((r for _, r in latest.values()), key=lambda r: (r["rung"] or 0, -(r["pr_auc"] or 0)))
    return {"items": items, "prevalence": prevalence}


if FRONTEND_DIST.exists():  # serve the built dashboard from the same origin
    from fastapi.responses import FileResponse

    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        """Client-side routes (/alerts, /graph/...) all load the app shell."""
        if path.startswith("api/"):
            raise HTTPException(404, "not found")
        file = FRONTEND_DIST / path
        if path and file.is_file():
            return FileResponse(file)
        return FileResponse(FRONTEND_DIST / "index.html")
