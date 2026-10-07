"""Metric correctness against hand-computed values.

A metric bug is silent: it produces a plausible number that is simply wrong, and
it corrupts every comparison in the results table.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexis.evaluation.metrics import (
    evaluate,
    jaccard,
    ring_level_metrics,
    summarise_ewt,
    threshold_for_alert_budget,
)


def test_alert_budget_produces_requested_volume():
    y = np.array([0] * 990 + [1] * 10)
    scores = np.linspace(0, 1, 1000)
    tau, precision, recall = threshold_for_alert_budget(y, scores, n_alerts=50)
    n_flagged = int((scores >= tau).sum())
    assert 45 <= n_flagged <= 55, f"expected ~50 alerts, got {n_flagged}"
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0


def test_evaluate_rejects_all_negative_labels():
    with pytest.raises(ValueError):
        evaluate(
            np.zeros(100, dtype=int),
            np.random.default_rng(0).random(100),
            model="m",
            seed=0,
            split="test",
        )


def test_pr_auc_lift_is_relative_to_prevalence():
    """PR-AUC is uninterpretable without prevalence; lift makes it comparable."""
    rng = np.random.default_rng(0)
    y = (rng.random(10_000) < 0.01).astype(int)
    scores = rng.random(10_000)
    result = evaluate(y, scores, model="random", seed=0, split="test")
    # A random scorer should sit near chance: PR-AUC ~ prevalence, lift ~ 1.
    assert 0.4 < result.pr_auc_lift < 2.5
    assert abs(result.roc_auc - 0.5) < 0.05


def test_perfect_scorer_reaches_ceiling():
    y = np.array([0] * 900 + [1] * 100)
    scores = y.astype(float)
    result = evaluate(y, scores, model="oracle", seed=0, split="test")
    assert result.pr_auc > 0.99
    assert result.roc_auc > 0.99


def test_jaccard_basic():
    assert jaccard({1, 2, 3}, {1, 2, 3}) == 1.0
    assert jaccard({1, 2, 3}, {4, 5, 6}) == 0.0
    assert jaccard({1, 2, 3, 4}, {3, 4, 5, 6}) == pytest.approx(2 / 6)


def test_ring_metrics_one_to_one_matching():
    truth = [{1, 2, 3, 4}, {5, 6, 7, 8}]
    predicted = [{1, 2, 3, 9}, {5, 6, 7, 8}, {20, 21, 22}]
    m = ring_level_metrics(predicted, truth, theta=0.5)
    assert m["ring_recall"] == 1.0, "both true rings should match"
    assert m["ring_precision"] == pytest.approx(2 / 3), "one spurious prediction"


def test_ewt_treats_undetected_rings_as_censored():
    """Never-detected rings must be counted, not dropped."""
    ewt = {"R1": 30.0, "R2": 10.0, "R3": None, "R4": None}
    s = summarise_ewt(ewt)
    assert s["detection_rate"] == 0.5
    assert s["median_ewt_minutes"] == 20.0
    assert s["n_censored"] == 2.0


def test_ewt_all_censored_is_not_a_crash():
    s = summarise_ewt({"R1": None, "R2": None})
    assert s["detection_rate"] == 0.0
    assert np.isnan(s["median_ewt_minutes"])
