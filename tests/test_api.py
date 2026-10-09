"""Integration: the API serves scores, every alert carries evidence (priority 4)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nexis.db import session as db_session
from nexis.db.alerts import create_alert_with_evidence
from nexis.db.models import Alert, Evidence, ModelRun, Transaction
from nexis.explainability.evidence import EvidenceError, build_packet
from nexis.investigation.investigator import Investigator, template_summary, verify

T0 = datetime(2022, 9, 10, 4, 0)


def _history() -> pd.DataFrame:
    rows = [
        ("HI:0000001", T0 - timedelta(minutes=30), "B_1", "A_1", 1000.0, "ACH"),
        ("HI:0000002", T0 - timedelta(minutes=10), "C_1", "A_1", 500.0, "ACH"),
        ("HI:0000003", T0, "A_1", "D_1", 1400.0, "ACH"),
        ("HI:0000004", T0 + timedelta(minutes=5), "A_1", "E_1", 9.0, "Cheque"),  # future
    ]
    return pd.DataFrame(rows, columns=["tx_id", "timestamp", "src", "dst", "amount", "payment_format"])


def _packet(tx_row: pd.Series, hist: pd.DataFrame):
    feats = pd.Series({"log_amount": 7.2, "counterparty_is_new": 1.0, "src_secs_since_in": 600.0,
                       "src_out_n_hist": 0.0, "src_in_n_hist": 2.0, "src_n_counterparties": 0.0,
                       "dst_in_n_hist": 0.0, "dst_out_n_hist": 0.0, "dst_n_counterparties": 0.0,
                       "src_secs_since_out": np.nan})
    tx = tx_row.copy()
    tx["amount_paid"], tx["pay_currency"] = tx["amount"], "US Dollar"
    tx["is_cross_bank"], tx["is_cross_currency"] = 1, 0
    cols = ["log_amount", "counterparty_is_new", "src_secs_since_in"]
    return build_packet(
        tx=tx, features=feats, shap_row=np.array([0.2, 1.8, 1.1]), columns=cols,
        risk_score=0.97, threshold=0.9, model_version="test@abc", rule_hits=["pass_through_1h"],
        history=hist, history_scores=pd.Series({"HI:0000001": 0.8, "HI:0000002": 0.2}),
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'api.db'}"
    monkeypatch.setenv("NEXIS_DATABASE_URL", url)
    monkeypatch.setenv("NEXIS_INVESTIGATOR", "template")
    db_session.get_engine.cache_clear()
    db_session.init_db(url)
    hist = _history()
    with db_session.session_scope(url) as s:
        for r in hist.itertuples():
            s.add(Transaction(
                tx_id=r.tx_id, occurred_at=r.timestamp, ingested_at=r.timestamp,
                src_account=r.src, dst_account=r.dst, amount_usd=r.amount, amount_paid=r.amount,
                pay_currency="US Dollar", payment_format=r.payment_format, is_cross_bank=True,
                is_cross_currency=False, risk_score=0.5, model_version="test@abc",
            ))
        s.add(ModelRun(model_version="test@abc", code_commit="abc", data_version="x",
                       train_start=T0 - timedelta(days=9), train_end=T0 - timedelta(days=3),
                       hyperparams={}, seed=0, metrics={"pr_auc": 0.5, "threshold": 0.9},
                       trained_at=T0, is_serving=True))
        s.flush()
        create_alert_with_evidence(s, _packet(hist.iloc[2], hist), created_at=T0)
    from nexis.api import app as api_module

    monkeypatch.setattr(api_module, "_investigator", Investigator(mode="template"))
    yield TestClient(api_module.app)
    db_session.get_engine.cache_clear()


def test_alert_list_and_detail_carry_evidence(client):
    r = client.get("/api/alerts").json()
    assert r["total"] == 1
    item = r["items"][0]
    assert item["risk_score"] == pytest.approx(0.97)
    assert item["top_reasons"][0]["feature"] == "counterparty_is_new"
    d = client.get(f"/api/alerts/{item['alert_id']}").json()
    ev = d["evidence"]
    assert ev["source_record_ids"][0] == "HI:0000003"
    assert ev["language_note"]
    # Evidence graph holds only transactions strictly before the alert.
    assert {e["tx_id"] for e in ev["graph_evidence"]["edges"]} == {"HI:0000001", "HI:0000002"}


def test_network_respects_until(client):
    net = client.get("/api/accounts/A_1/network", params={"until": T0.isoformat()}).json()
    assert {e["tx_id"] for e in net["edges"]} == {"HI:0000001", "HI:0000002"}
    later = client.get("/api/accounts/A_1/network", params={"until": (T0 + timedelta(hours=1)).isoformat()}).json()
    assert "HI:0000004" in {e["tx_id"] for e in later["edges"]}


def test_disposition_is_feedback_only(client):
    aid = client.get("/api/alerts").json()["items"][0]["alert_id"]
    r = client.post(f"/api/alerts/{aid}/disposition", json={"disposition": "escalated", "note": "x"})
    assert r.status_code == 200 and r.json()["status"] == "escalated"
    assert client.post(f"/api/alerts/{aid}/disposition", json={"disposition": "freeze"}).status_code == 422
    routes = {getattr(rt, "path", "") for rt in client.app.routes}
    assert not any(w in p for p in routes for w in ("block", "freeze", "report", "suspend"))


def test_investigate_template(client):
    aid = client.get("/api/alerts").json()["items"][0]["alert_id"]
    res = client.post(f"/api/alerts/{aid}/investigate").json()
    assert res["mode"] == "template"
    assert all(c["verified"] for c in res["claims"]), res["warnings"]


def test_summary_and_health(client):
    assert client.get("/api/health").json()["model_version"] == "test@abc"
    s = client.get("/api/summary").json()
    assert s["alerts_total"] == 1 and s["transactions_scored"] == 4


def test_alert_without_evidence_is_never_written(tmp_path):
    url = f"sqlite:///{tmp_path / 'r6.db'}"
    db_session.get_engine.cache_clear()
    db_session.init_db(url)
    hist = _history()
    packet = _packet(hist.iloc[2], hist)
    packet.source_record_ids = []
    with pytest.raises(EvidenceError), db_session.session_scope(url) as s:
        create_alert_with_evidence(s, packet, created_at=T0)
    with db_session.session_scope(url) as s:
        assert s.query(Alert).count() == 0 and s.query(Evidence).count() == 0
    db_session.get_engine.cache_clear()


def _fake_llm(payload: dict) -> object:
    resp = SimpleNamespace(
        stop_reason="end_turn", model="claude-opus-5-5",
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
    )
    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: resp)))


def test_investigator_flags_unsupported_and_accusatory_claims():
    hist = _history()
    packet = _packet(hist.iloc[2], hist).to_dict()
    llm = _fake_llm({
        "summary": "The system observed a payment soon after receipts.",
        "claims": [
            {"text": "The sender received 1,000 USD 30 minutes earlier.", "source_record_ids": ["HI:0000001"]},
            {"text": "This account is laundering money.", "source_record_ids": ["HI:0000003"]},
            {"text": "It matches HI:9999999.", "source_record_ids": ["HI:9999999"]},
        ],
    })
    res = Investigator(mode="llm", client=llm).investigate(packet, alert_id=1)
    assert res.mode == "llm"
    assert [c.verified for c in res.claims] == [True, False, False]
    assert len(res.warnings) == 2


def test_investigator_falls_back_to_template_on_failure():
    class Boom:
        class beta:  # noqa: N801
            class messages:  # noqa: N801
                @staticmethod
                def create(**kw):
                    raise RuntimeError("no credentials")

    hist = _history()
    packet = _packet(hist.iloc[2], hist).to_dict()
    res = Investigator(mode="auto", client=Boom()).investigate(packet)
    assert res.mode == "template" and "LLM unavailable" in res.warnings[0]


def test_template_summary_has_no_accusatory_language():
    hist = _history()
    res = verify(template_summary(_packet(hist.iloc[2], hist).to_dict()), _packet(hist.iloc[2], hist).to_dict())
    assert not [w for w in res.warnings if "rule 5" in w]
