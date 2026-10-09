# NEXIS API contract (v1)

The FastAPI backend (`src/nexis/api/`) serves this contract; the React dashboard
(`frontend/`) consumes it. All paths are under `/api`. Timestamps are ISO-8601
strings. Amounts are USD floats.

**Language (PROJECT_RULES.md rule 5).** No field name, value or UI string may assert
that a person committed fraud, laundered money or acted with intent. Use "risk
score", "the model weighted", "the system observed". Every evidence payload
carries `language_note`.

**No automated action.** The API has no endpoint that blocks, freezes or reports an
account. Analysts record a *disposition*; that is the only write.

---

## GET /api/health

```json
{"status": "ok", "model_version": "xgb_behavioural@a1b2c3d", "db": "postgresql"}
```

## GET /api/summary

```json
{
  "model_version": "xgb_behavioural@a1b2c3d",
  "dataset": "IBM AML HI-Small",
  "replay_period": {"start": "2022-09-09T16:10:00", "end": "2022-09-10T23:59:00"},
  "transactions_scored": 416866,
  "alerts_total": 500,
  "alerts_open": 487,
  "alert_budget": 500,
  "threshold": 0.9731,
  "metrics": {"pr_auc": 0.61, "pr_auc_std": 0.01, "prevalence": 0.00147,
              "recall_at_budget": 0.52, "precision_at_budget": 0.64}
}
```

## GET /api/alerts

Query: `status` (`open|escalated|dismissed|needs_info|all`, default `open`),
`limit` (default 50, max 500), `offset` (default 0),
`sort` (`risk_score` default | `occurred_at`).

```json
{
  "total": 487,
  "items": [
    {
      "alert_id": 17,
      "tx_id": "HI-Small:0412337",
      "occurred_at": "2022-09-10T04:12:00",
      "created_at": "2022-09-10T04:12:00",
      "src_account": "011_8000ABC10",
      "dst_account": "0220_8001DD320",
      "amount_usd": 18234.55,
      "payment_format": "ACH",
      "risk_score": 0.9934,
      "rank": 1,
      "status": "open",
      "top_reasons": [
        {"feature": "counterparty_is_new", "label": "First payment to this counterparty", "contribution": 1.84},
        {"feature": "src_secs_since_in", "label": "Seconds since the sender last received funds", "contribution": 1.12}
      ]
    }
  ]
}
```

## GET /api/alerts/{alert_id}

The alert (fields as above) plus `evidence`, the evidence packet written in the
same database transaction as the alert (rule 6). `source_record_ids` is never
empty.

```json
{
  "alert": { "...": "as in the list item" },
  "evidence": {
    "alert_id": 17,
    "tx_id": "HI-Small:0412337",
    "model_version": "xgb_behavioural@a1b2c3d",
    "risk_score": 0.9934,
    "threshold": 0.9731,
    "generated_at": "2022-09-10T04:12:00",
    "source_record_ids": ["HI-Small:0412337", "HI-Small:0411002", "HI-Small:0409876"],
    "transaction": {
      "tx_id": "HI-Small:0412337", "occurred_at": "2022-09-10T04:12:00",
      "src_account": "011_8000ABC10", "dst_account": "0220_8001DD320",
      "amount_usd": 18234.55, "amount_paid": 18234.55, "pay_currency": "US Dollar",
      "payment_format": "ACH", "is_cross_bank": true, "is_cross_currency": false
    },
    "feature_contributions": [
      {"feature": "counterparty_is_new", "label": "First payment to this counterparty",
       "value": 1.0, "contribution": 1.84}
    ],
    "rule_hits": ["pass_through_1h", "new_cross_bank_counterparty"],
    "account_context": {
      "src": {"account": "011_8000ABC10", "prior_sends": 3, "prior_receipts": 5,
              "distinct_counterparties": 2, "secs_since_last_receipt": 840},
      "dst": {"account": "0220_8001DD320", "prior_receipts": 9,
              "distinct_counterparties": 7}
    },
    "graph_evidence": {
      "nodes": [{"id": "011_8000ABC10", "kind": "account", "risk": 0.99, "is_focus": true}],
      "edges": [{"tx_id": "HI-Small:0411002", "src": "0070_8000F0001", "dst": "011_8000ABC10",
                 "amount_usd": 18500.0, "occurred_at": "2022-09-10T03:58:00",
                 "risk_score": 0.97, "payment_format": "ACH", "important": true}]
    },
    "ring_candidate": {"ring_id": "R-0031", "members": ["011_8000ABC10", "0220_8001DD320"], "score": 0.91},
    "language_note": "Risk scores reflect model output on recorded data. They are not findings of wrongdoing."
  }
}
```

`ring_candidate` may be `null`.

## POST /api/alerts/{alert_id}/disposition

Body: `{"disposition": "escalated" | "dismissed" | "needs_info", "note": "free text"}`.
Returns the updated alert. This is analyst feedback only.

## POST /api/alerts/{alert_id}/investigate

Runs the investigator over the alert's evidence packet only (retrieval-first).
Every claim carries the `source_record_ids` it rests on; claims citing ids that are
not in the packet are marked `verified: false` and listed in `warnings`.

```json
{
  "alert_id": 17,
  "mode": "llm",
  "model": "claude-sonnet-5-5",
  "summary": "The system observed a first payment to a new counterparty ...",
  "claims": [
    {"text": "The sender received 18,500 USD 14 minutes before this payment.",
     "source_record_ids": ["HI-Small:0411002"], "verified": true}
  ],
  "warnings": []
}
```

`mode` is `template` when no LLM is configured; the shape is identical.

## GET /api/accounts/{account_id}/network

Query: `until` (ISO timestamp, default = the alert time or replay end), `hops`
(1–2, default 2), `limit` (max edges, default 300).
Only edges with `occurred_at < until` are returned (leakage guard).

```json
{
  "center": "011_8000ABC10",
  "until": "2022-09-10T04:12:00",
  "truncated": false,
  "nodes": [{"id": "011_8000ABC10", "kind": "account", "risk": 0.99, "is_center": true}],
  "edges": [{"tx_id": "HI-Small:0411002", "src": "0070_8000F0001", "dst": "011_8000ABC10",
             "amount_usd": 18500.0, "occurred_at": "2022-09-10T03:58:00",
             "risk_score": 0.97, "payment_format": "ACH"}]
}
```

## GET /api/accounts/{account_id}/transactions

Query: `until`, `limit` (default 100). Returns `{"items": [transaction, ...]}` with
the transaction shape above plus `risk_score` (null if not scored).

## GET /api/rings

Query: `limit` (default 50).

```json
{
  "items": [
    {"ring_id": "R-0031", "members": ["011_8000ABC10", "0220_8001DD320", "0070_8000F0001"],
     "n_transactions": 6, "score": 0.91,
     "first_seen": "2022-09-09T22:10:00", "last_seen": "2022-09-10T04:12:00",
     "tx_ids": ["HI-Small:0411002", "HI-Small:0412337"]}
  ]
}
```

## GET /api/drift

```json
{
  "reference": {"start": "2022-09-01T00:00:00", "end": "2022-09-06T13:34:00"},
  "windows": [
    {"start": "2022-09-09T16:10:00", "end": "2022-09-09T22:10:00",
     "psi_max": 0.08, "psi_top": {"src_out_cnt_24h": 0.08, "log_amount": 0.02},
     "score_psi": 0.03, "alert_rate": 0.0011, "status": "ok"}
  ],
  "events": [{"detected_at": "2022-09-10T06:10:00", "metric": "score_psi",
              "value": 0.27, "level": "drift", "action": "recalibrate_threshold"}]
}
```

`status` is `ok | warn | drift`.

## GET /api/models

The modelling ladder, from `results/`: mean ± std over seeds.

```json
{
  "items": [
    {"name": "xgb_behavioural", "rung": 4, "pr_auc": 0.61, "pr_auc_std": 0.01,
     "roc_auc": 0.99, "roc_auc_std": 0.002, "recall_at_fpr_1e3": 0.5,
     "recall_at_fpr_1e3_std": 0.01, "precision_at_budget": 0.64,
     "precision_at_budget_std": 0.007, "n_seeds": 5}
  ],
  "prevalence": 0.00147
}
```
