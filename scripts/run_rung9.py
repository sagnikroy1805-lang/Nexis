"""Rung 9: score fusion + fraud-ring detection + early-warning time.

Fusion (§22.3): a logistic stacker over the saved scores of the base models,
fitted on the VALIDATION fold's scores only and applied to the test fold.

Rings (§11.4): two-stage detection on the fused scores. The stage-2 scorer is
trained on validation-window candidates against validation-window patterns, then
applied to test-window candidates. Ring-level precision/recall is reported at
Jaccard theta in {0.3, 0.5, 0.7} (defined before looking at results, §11.4.3),
with member-level precision/recall alongside.

Early warning (§4.7, §9.5): minutes between the first alert on any member
account and the pattern's TRUE completion (including tail rows dropped from the
data). Undetected patterns are right-censored, not dropped.

Usage:
    python scripts/run_rung9.py --base xgb_graph gnn_temporal [--top-k 0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from nexis.data.ibm_aml import load_patterns
from nexis.evaluation.harness import aggregate_over_seeds, format_table, run_experiment
from nexis.evaluation.metrics import (
    EvalResult,
    early_warning_times,
    evaluate,
    ring_level_metrics,
    summarise_ewt,
    threshold_for_alert_budget,
)
from nexis.evaluation.seeds import SEEDS
from nexis.experiments.common import LABEL, load_scores, prepare, save_scores
from nexis.models.rings import (
    RingScorer,
    generate_candidates,
    truth_rings,
    window_frame,
)

RESULTS = Path("results")


def fused_scores(base: list[str], seed: int, labels: pd.Series) -> pd.DataFrame:
    """Stack base-model scores; the stacker sees validation labels only."""
    frames = [load_scores(m, seed).rename(columns={"score": m}) for m in base]
    s = frames[0]
    for f in frames[1:]:
        s = s.merge(f, on=["tx_id", "fold"], how="inner")
    logit = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))  # noqa: E731
    x = np.column_stack([logit(s[m].to_numpy()) for m in base])
    y = s["tx_id"].map(labels).to_numpy()
    is_val = (s["fold"] == "val").to_numpy()
    stacker = LogisticRegression(class_weight="balanced", max_iter=2000).fit(x[is_val], y[is_val])
    s["score"] = stacker.predict_proba(x)[:, 1]
    s.attrs["weights"] = dict(zip(base, map(float, stacker.coef_[0]), strict=True))
    return s[["tx_id", "fold", "score"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/ibm_aml.yaml"))
    parser.add_argument("--base", nargs="+", required=True)
    parser.add_argument("--candidate-quantile", type=float, default=0.99)
    args = parser.parse_args()

    prep = prepare(args.config, with_graph=False)
    df, val, test = prep.df, prep.split.val, prep.split.test
    labels = prep.frame.set_index("tx_id")[LABEL]
    patterns = load_patterns(prep.cfg)
    budget = prep.evaluation["alert_budget"]
    y_test = test[LABEL].to_numpy()

    val_truth = truth_rings(patterns, df, set(val["tx_id"]))
    test_truth = truth_rings(patterns, df, set(test["tx_id"]))
    print(f"patterns active: val {len(val_truth)}, test {len(test_truth)}")

    ring_log: list[dict] = []

    def evaluate_fn(_model: object, seed: int) -> EvalResult:
        fused = fused_scores(args.base, seed, labels)
        save_scores("fusion", seed, {
            "val": (val, fused.set_index("tx_id").loc[val["tx_id"], "score"].to_numpy()),
            "test": (test, fused.set_index("tx_id").loc[test["tx_id"], "score"].to_numpy()),
        })
        test_scores = fused.set_index("tx_id").loc[test["tx_id"], "score"].to_numpy()

        # Ring detection: stage-2 trained on the validation window only.
        val_win = window_frame(df, fused, "val")
        test_win = window_frame(df, fused, "test")
        val_c = generate_candidates(val_win, args.candidate_quantile, seed=seed, prefix="V")
        test_c = generate_candidates(test_win, args.candidate_quantile, seed=seed, prefix="R")
        scorer = RingScorer(theta=0.5).fit(val_c, val_truth)
        ring_scores = scorer.score(test_c)
        for r, sc in zip(test_c, ring_scores, strict=True):
            r.score = float(sc)
        ranked = sorted(test_c, key=lambda r: -r.score)
        truth_sets = [t["members"] for t in test_truth.values()]
        per_theta = {}
        for k_name, preds in (("all_candidates", ranked), ("top_k_truth", ranked[: len(truth_sets)])):
            for theta in (0.3, 0.5, 0.7):
                per_theta[f"{k_name}@{theta}"] = ring_level_metrics(
                    [set(r.members) for r in preds], truth_sets, theta
                )

        # Early-warning time at the alert-budget operating point.
        tau, _, _ = threshold_for_alert_budget(y_test, test_scores, budget)
        events = pd.concat([
            pd.DataFrame({"account": test_win["src"], "timestamp": test_win["timestamp"], "score": test_win["score"]}),
            pd.DataFrame({"account": test_win["dst"], "timestamp": test_win["timestamp"], "score": test_win["score"]}),
        ])
        ewt = summarise_ewt(early_warning_times(events, test_truth, tau))
        ring_log.append({
            "seed": seed,
            "n_candidates": len(test_c),
            "n_truth": len(truth_sets),
            "ring_metrics": per_theta,
            "ewt": ewt,
            "stage2_weights": scorer.weights,
            "stage2_fallback": scorer.fallback,
            "fusion_weights": fused.attrs.get("weights", {}),
            "top_rings": [
                {"ring_id": r.ring_id, "members": sorted(r.members), "score": r.score,
                 "n_transactions": len(r.tx_ids), "first_seen": str(r.first_seen),
                 "last_seen": str(r.last_seen), "tx_ids": r.tx_ids[:200]}
                for r in ranked[:50]
            ],
        })
        res = evaluate(y_test, test_scores, model="fusion", seed=seed, split="test",
                       alert_budget=budget, extra={"rung": 9, "base": args.base})
        res.median_ewt_minutes = ewt["median_ewt_minutes"]
        res.ring_detection_rate = ewt["detection_rate"]
        return res

    summary = run_experiment(
        name="ibm_aml_r9_fusion",
        config={"dataset": "ibm_aml", "rung": 9, "base": args.base,
                "candidate_quantile": args.candidate_quantile, "alert_budget": budget},
        model_factory=lambda _c, _s: None,
        evaluate_fn=evaluate_fn,
        seeds=SEEDS,
        results_dir=RESULTS,
    )
    (RESULTS / "rings_ibm_aml.json").write_text(json.dumps(ring_log, indent=2, default=str))
    print(format_table(summary))
    r = pd.DataFrame([
        {"seed": x["seed"], **{f"{k}_{m}": v for k, d in x["ring_metrics"].items() for m, v in d.items()},
         "ewt_detection_rate": x["ewt"]["detection_rate"], "ewt_median_min": x["ewt"]["median_ewt_minutes"]}
        for x in ring_log
    ])
    keep = [c for c in r.columns if c.endswith(("ring_precision", "ring_recall", "member_recall")) or c.startswith("ewt")]
    print(r[keep].agg(["mean", "std"]).T.to_string())
    _ = aggregate_over_seeds  # summary already aggregated by the harness


if __name__ == "__main__":
    main()
