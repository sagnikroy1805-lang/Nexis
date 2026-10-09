"""Streaming replay: score the test period in time order and populate the database.

CLAUDE.md: "simulate streaming by replaying a sorted file". This script is that
simulation, and the bridge between the research pipeline and the analyst app.

Causality, step by step:
  1. The serving model (rung-5 XGBoost) is fitted on the training fold and
     early-stopped on the validation fold.
  2. The alert threshold is calibrated on VALIDATION scores so the expected
     alert count over the test period matches the budget. Picking the top-N of
     the whole test period would use scores from the future.
  3. Test transactions are replayed in hourly batches. Features are strict-past
     by construction; ring candidates attached to an alert are detected from
     scored transactions BEFORE that batch only.
  4. Each alert and its evidence packet commit in one database transaction
     (rule 6).

Usage:
    python scripts/replay.py --config configs/ibm_aml.yaml [--db sqlite:///data/nexis.db]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nexis.data.ibm_aml import load_patterns
from nexis.db.alerts import create_alert_with_evidence
from nexis.db.models import DriftEvent, DriftWindow, ModelRun, Ring
from nexis.db.session import get_engine, init_db, session_scope
from nexis.evaluation.metrics import evaluate
from nexis.experiments.common import LABEL, prepare
from nexis.explainability.evidence import build_packet
from nexis.explainability.shap_explain import tree_shap
from nexis.models.baselines.scorers import RulesScorer, XGBScorer
from nexis.models.rings import (
    RingScorer,
    generate_candidates,
    truth_rings,
    window_frame,
)

RESULTS = Path("results")
TX_COLUMNS = {
    "tx_id": "tx_id",
    "timestamp": "occurred_at",
    "src": "src_account",
    "dst": "dst_account",
    "amount": "amount_usd",
    "amount_paid": "amount_paid",
    "pay_currency": "pay_currency",
    "payment_format": "payment_format",
    "is_cross_bank": "is_cross_bank",
    "is_cross_currency": "is_cross_currency",
}


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def load_transactions(df: pd.DataFrame, scores: pd.Series, model_version: str, url: str | None) -> None:
    """Bulk-load every transaction (history for the graph explorer) with its score."""
    out = df[list(TX_COLUMNS)].rename(columns=TX_COLUMNS).copy()
    out["ingested_at"] = out["occurred_at"]  # §17.1.1: dataset records occurrence only
    for c in ("pay_currency", "payment_format"):
        out[c] = out[c].astype(str)
    for c in ("is_cross_bank", "is_cross_currency"):
        out[c] = out[c].astype(bool)
    out["risk_score"] = out["tx_id"].map(scores)
    out["model_version"] = np.where(out["risk_score"].notna(), model_version, None)
    engine = get_engine(url)
    t0 = time.time()
    if engine.dialect.name == "postgresql":
        # COPY streams CSV straight into the table: minutes faster than INSERTs
        # for 5M rows. Empty unquoted fields load as NULL (unscored rows).
        import io

        cols = list(out.columns)
        raw = engine.raw_connection()
        try:
            with raw.cursor() as cur, cur.copy(
                f"COPY transactions ({', '.join(cols)}) FROM STDIN WITH (FORMAT csv)"
            ) as cp:
                for start in range(0, len(out), 500_000):
                    buf = io.StringIO()
                    out.iloc[start : start + 500_000].to_csv(buf, header=False, index=False, na_rep="")
                    cp.write(buf.getvalue())
            raw.commit()
        finally:
            raw.close()
    else:
        out.to_sql("transactions", engine, if_exists="append", index=False, chunksize=100_000)
    print(f"  loaded {len(out):,} transactions in {time.time() - t0:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/ibm_aml.yaml"))
    parser.add_argument("--db", default=None, help="SQLAlchemy URL; default NEXIS_DATABASE_URL")
    parser.add_argument("--batch", default="1h")
    parser.add_argument("--ring-window", default="96h")
    args = parser.parse_args()

    t0 = time.time()
    prep = prepare(args.config)
    df = prep.df
    train, val, test = prep.split.train, prep.split.val, prep.split.test
    cols = prep.with_graph_cols
    budget = int(prep.evaluation["alert_budget"])

    tuning = RESULTS / "tuning_ibm_aml_xgb_graph.json"
    params = json.loads(tuning.read_text())["best"] if tuning.exists() else {}
    model = XGBScorer(cols, seed=0, params=params).fit(train, val)
    version = f"xgb_graph@{_commit()}"
    raw_val, raw_test = model.score(val), model.score(test)
    # Calibration (Concept Mastery §2.5): scale_pos_weight pushes raw scores of
    # every strong alert to ~1.0, which tells an analyst nothing. Platt scaling
    # fitted on the VALIDATION fold maps scores to P(label | score) at the
    # validation base rate. It is monotone, so ranking, alerts and every
    # ranking metric are unchanged; only the displayed number gains meaning.
    from sklearn.linear_model import LogisticRegression

    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p)).reshape(-1, 1)

    platt = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw_val), val[LABEL].to_numpy())
    s_val = platt.predict_proba(_logit(raw_val))[:, 1]
    s_test = platt.predict_proba(_logit(raw_test))[:, 1]
    print(f"serving model {version} trained and calibrated ({time.time() - t0:.0f}s)")

    # Label-free threshold: the val-score quantile giving `budget` alerts over a
    # test-sized stream.
    rate = budget / len(test)
    tau = float(np.quantile(s_val, 1.0 - rate))
    res = evaluate(test[LABEL].to_numpy(), s_test, model="xgb_graph", seed=0, split="test", alert_budget=budget)
    realised = s_test >= tau
    print(f"threshold {tau:.4f} -> {int(realised.sum())} alerts (budget {budget}); test PR-AUC {res.pr_auc:.4f}")

    scores = pd.concat([pd.Series(s_val, index=val["tx_id"].to_numpy()), pd.Series(s_test, index=test["tx_id"].to_numpy())])

    url = args.db
    init_db(url, drop=True)
    load_transactions(df, scores, version, url)

    summary_path = sorted(RESULTS.glob("ibm_aml_r5_xgb_graph_*/summary.json"))
    pr_std = json.loads(summary_path[-1].read_text())["pr_auc_std"].get("xgb_graph") if summary_path else None
    with session_scope(url) as s:
        s.add(ModelRun(
            model_version=version, code_commit=_commit(),
            data_version=json.loads(prep.cfg.manifest_path.read_text())["processed_sha256"][:16],
            train_start=train["timestamp"].min().to_pydatetime(), train_end=train["timestamp"].max().to_pydatetime(),
            hyperparams=params, seed=0, trained_at=pd.Timestamp.now().to_pydatetime(), is_serving=True,
            metrics={"dataset": f"IBM AML {prep.cfg.variant}", "pr_auc": res.pr_auc, "pr_auc_std": pr_std,
                     "prevalence": res.prevalence, "recall_at_budget": res.recall_at_budget,
                     "precision_at_budget": res.precision_at_budget, "roc_auc": res.roc_auc,
                     "threshold": tau, "alert_budget": budget,
                     "replay_period": {"start": test["timestamp"].min().isoformat(),
                                       "end": test["timestamp"].max().isoformat()},
                     "transactions_replayed": len(test), "note": "seed-0 serving model; "
                     "the paper table reports mean ± std over 5 seeds"},
        ))

    # Explanation inputs.
    rules = RulesScorer().fit(train, val)
    flagged = test.loc[realised].copy()
    flagged_scores = s_test[realised]
    flagged_tx = df.set_index("tx_id").loc[flagged["tx_id"]].reset_index()
    shap_vals = tree_shap(model.model, flagged[cols])
    hits = rules.rule_hits(flagged)
    accounts = set(flagged_tx["src"]) | set(flagged_tx["dst"])
    history = df.loc[df["src"].isin(accounts) | df["dst"].isin(accounts),
                     ["tx_id", "timestamp", "src", "dst", "amount", "payment_format"]]

    # Ring stage-2 scorer: trained on validation-window candidates only.
    patterns = load_patterns(prep.cfg)
    fused = pd.DataFrame({"tx_id": scores.index, "score": scores.to_numpy(),
                          "fold": np.where(scores.index.isin(val["tx_id"]), "val", "test")})
    val_win = window_frame(df, fused, "val")
    ring_scorer = RingScorer().fit(generate_candidates(val_win, prefix="V"), truth_rings(patterns, df, set(val["tx_id"])))
    scored_win = window_frame(df, fused.assign(fold="all"), "all")

    # Replay.
    flagged_tx = flagged_tx.assign(row_i=np.arange(len(flagged_tx))).sort_values("timestamp", kind="stable")
    start = test["timestamp"].min().floor(args.batch)
    n_alerts = 0
    for b0 in pd.date_range(start, test["timestamp"].max() + pd.Timedelta(args.batch), freq=args.batch):
        b1 = b0 + pd.Timedelta(args.batch)
        batch = flagged_tx.loc[(flagged_tx["timestamp"] >= b0) & (flagged_tx["timestamp"] < b1)]
        if batch.empty:
            continue
        past = scored_win.loc[(scored_win["timestamp"] < b0) & (scored_win["timestamp"] >= b0 - pd.Timedelta(args.ring_window))]
        rings = generate_candidates(past, prefix=f"T{b0:%m%d%H}") if len(past) > 100 else []
        for r, sc in zip(rings, ring_scorer.score(rings), strict=True):
            r.score = float(sc)
        member_of = {m: r for r in sorted(rings, key=lambda r: r.score) for m in r.members}
        with session_scope(url) as s:
            for _, tx in batch.iterrows():
                i = int(tx["row_i"])
                ring = member_of.get(tx["src"]) or member_of.get(tx["dst"])
                packet = build_packet(
                    tx=tx, features=flagged.iloc[i], shap_row=shap_vals[i], columns=cols,
                    risk_score=float(flagged_scores[i]),
                    threshold=tau, model_version=version,
                    rule_hits=[k for k, v in hits.iloc[i].items() if v],
                    history=history, history_scores=scores,
                    ring={"ring_id": ring.ring_id, "members": sorted(ring.members), "score": ring.score} if ring else None,
                )
                create_alert_with_evidence(s, packet, created_at=pd.Timestamp(tx["timestamp"]).to_pydatetime())
                n_alerts += 1
    print(f"  wrote {n_alerts} alerts with evidence")

    # Final ring list over the test window, for the rings view.
    test_win = window_frame(df, fused, "test")
    final = generate_candidates(test_win, prefix="R")
    for r, sc in zip(final, ring_scorer.score(final), strict=True):
        r.score = float(sc)
    with session_scope(url) as s:
        for r in sorted(final, key=lambda r: -r.score)[:200]:
            s.add(Ring(ring_id=r.ring_id, members=sorted(r.members), tx_ids=r.tx_ids, score=r.score,
                       first_seen=r.first_seen.to_pydatetime(), last_seen=r.last_seen.to_pydatetime()))
    print(f"  wrote {min(len(final), 200)} rings")

    write_drift(prep, cols, s_val, s_test, tau, version, url)
    print(f"replay done in {time.time() - t0:.0f}s")


def write_drift(prep: Any, cols: list[str], s_val: np.ndarray, s_test: np.ndarray,
                tau: float, version: str, url: str | None) -> None:
    """Label-free monitoring of the replayed stream (Concept Mastery §13.3)."""
    from dataclasses import asdict

    from nexis.drift.monitors import PSIMonitor

    # Reference = the VALIDATION fold: out-of-sample scores, like the stream's.
    # Training-fold scores are inflated by fit, so comparing against them would
    # flag "drift" that is only overfitting.
    val, test = prep.split.val, prep.split.test
    # Monitor only features that should be stationary. Cumulative ones (an
    # account's earlier-payment count, graph degree, PageRank...) grow with time
    # by construction and would read as "drift" in every window (PSI ~8 in a
    # first replay), drowning the signal.
    cumulative = ("_n_hist", "n_counterparties", "_degree", "_tx", "_weight", "pagerank",
                  "scc_size", "reciprocal", "prior_pair", "secs_since")
    monitored = [c for c in cols if not any(k in c for k in cumulative)]
    monitor = PSIMonitor.fit(val, s_val, monitored, alert_threshold=tau)
    test = test.assign(_score=s_test)
    with session_scope(url) as s:
        run = s.get(ModelRun, version)
        if run is not None:
            run.metrics = {**run.metrics, "drift_reference": {
                "start": val["timestamp"].min().isoformat(), "end": val["timestamp"].max().isoformat()}}
        prev = "ok"
        for w0 in pd.date_range(test["timestamp"].min().floor("6h"), test["timestamp"].max(), freq="6h"):
            win = test.loc[(test["timestamp"] >= w0) & (test["timestamp"] < w0 + pd.Timedelta("6h"))]
            if len(win) < 1000:
                continue
            d = asdict(monitor.window_report(win, win["_score"].to_numpy()))
            s.add(DriftWindow(start=w0.to_pydatetime(), end=(w0 + pd.Timedelta("6h")).to_pydatetime(), report=d))
            if d["status"] != "ok" and d["status"] != prev:
                # Status is driven by score PSI (monitors.PSIMonitor), so the
                # event reports that value; feature PSIs stay in the window report.
                s.add(DriftEvent(model_version=version, detected_at=(w0 + pd.Timedelta("6h")).to_pydatetime(),
                                 metric="score_psi", value=float(d["score_psi"]),
                                 level=d["status"], action="recalibrate_threshold" if d["status"] == "drift" else "monitor"))
            prev = d["status"]
    print("  wrote drift windows")


if __name__ == "__main__":
    main()
