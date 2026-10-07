"""Metrics for rare-event alert ranking.

Implements Concept Mastery Module 4.

The framing throughout: NEXIS is a ranking function under a hard alert budget,
not a classifier. Metrics are chosen accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvalResult:
    """One model, one seed, one split. The unit of every results table."""

    model: str
    seed: int
    split: str
    prevalence: float
    n_samples: int
    n_positives: int

    pr_auc: float
    roc_auc: float
    pr_auc_lift: float  # pr_auc / prevalence; PR-AUC is uninterpretable without it

    threshold_budget: float
    precision_at_budget: float
    recall_at_budget: float

    recall_at_fpr_1e2: float
    recall_at_fpr_1e3: float

    median_ewt_minutes: float = float("nan")
    ring_detection_rate: float = float("nan")

    extra: dict[str, Any] = field(default_factory=dict)


def threshold_for_alert_budget(
    y_true: np.ndarray, scores: np.ndarray, n_alerts: int
) -> tuple[float, float, float]:
    """Choose the threshold that produces exactly `n_alerts` alerts.

    This is the operationally meaningful operating point (§1.3): analyst capacity
    is fixed, so the real question is how much fraud surfaces within that budget.
    Compare models here, not at max-F1.

    Returns:
        (threshold, precision, recall) at that operating point.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    k = int(min(max(n_alerts, 1), len(scores) - 1))
    tau = float(np.sort(scores)[::-1][k])

    flagged = scores >= tau
    tp = float(y_true[flagged].sum())
    precision = tp / max(flagged.sum(), 1)
    recall = tp / max(y_true.sum(), 1)
    return tau, float(precision), float(recall)


def threshold_for_max_f1(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[float, float, float]:
    """The academic-convention operating point. Report alongside, never instead."""
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    k = int(np.nanargmax(f1[:-1]))
    return float(t[k]), float(p[k]), float(r[k])


def recall_at_fpr(
    y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 1e-3
) -> tuple[float, float]:
    """Recall at a fixed false-positive rate.

    The fair comparison between two models: at identical analyst workload, which
    finds more fraud?

    Returns:
        (recall, threshold)
    """
    fpr, tpr, thr = roc_curve(y_true, scores)
    idx = max(int(np.searchsorted(fpr, target_fpr, side="right")) - 1, 0)
    return float(tpr[idx]), float(thr[idx])


def evaluate(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    model: str,
    seed: int,
    split: str,
    alert_budget: int = 500,
    rings: Mapping[str, dict] | None = None,
    scored_events: pd.DataFrame | None = None,
    extra: dict | None = None,
) -> EvalResult:
    """Compute the full metric suite for one model/seed/split.

    Every model in the project goes through this function. Model code must never
    compute its own metrics -- that is how incomparable numbers get into a table.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    if y_true.sum() == 0:
        raise ValueError("no positives in y_true; metrics would be undefined")

    prevalence = float(y_true.mean())
    pr_auc = float(average_precision_score(y_true, scores))
    tau_b, p_b, r_b = threshold_for_alert_budget(y_true, scores, alert_budget)

    result = EvalResult(
        model=model,
        seed=seed,
        split=split,
        prevalence=prevalence,
        n_samples=int(len(y_true)),
        n_positives=int(y_true.sum()),
        pr_auc=pr_auc,
        roc_auc=float(roc_auc_score(y_true, scores)),
        pr_auc_lift=pr_auc / max(prevalence, 1e-12),
        threshold_budget=tau_b,
        precision_at_budget=p_b,
        recall_at_budget=r_b,
        recall_at_fpr_1e2=recall_at_fpr(y_true, scores, 1e-2)[0],
        recall_at_fpr_1e3=recall_at_fpr(y_true, scores, 1e-3)[0],
        extra=extra or {},
    )

    if rings is not None and scored_events is not None:
        ewt = early_warning_times(scored_events, rings, tau_b)
        summary = summarise_ewt(ewt)
        result.median_ewt_minutes = summary["median_ewt_minutes"]
        result.ring_detection_rate = summary["detection_rate"]

    return result


def early_warning_times(
    scored_events: pd.DataFrame,
    rings: Mapping[str, dict],
    tau: float,
) -> dict[str, float | None]:
    """Minutes before ring completion that the first alert fired.

    Implements §4.7. Rings never detected are recorded as None -- right-censored,
    not dropped. Dropping them biases the median in your favour and a reviewer
    will notice.

    Args:
        scored_events: columns [account, timestamp, score].
        rings: ring_id -> {"members": set, "t_start": ts, "t_end": ts}.
        tau: operating threshold. EWT is meaningless without one.
    """
    out: dict[str, float | None] = {}
    for ring_id, ring in rings.items():
        hits = scored_events[
            scored_events["account"].isin(ring["members"])
            & (scored_events["score"] >= tau)
            & (scored_events["timestamp"] <= ring["t_end"])
        ]
        if len(hits) == 0:
            out[ring_id] = None
        else:
            delta = ring["t_end"] - hits["timestamp"].min()
            out[ring_id] = delta.total_seconds() / 60.0
    return out


def summarise_ewt(ewt: Mapping[str, float | None]) -> dict[str, float]:
    """Median and IQR over detected rings, plus the detection rate.

    Always report detection_rate alongside the median. A model detecting 20% of
    rings very early is not obviously better than one detecting 80% slightly late.
    """
    vals = [v for v in ewt.values() if v is not None]
    n_total = max(len(ewt), 1)
    if not vals:
        return {
            "detection_rate": 0.0,
            "median_ewt_minutes": float("nan"),
            "iqr_low": float("nan"),
            "iqr_high": float("nan"),
            "n_censored": float(n_total),
        }
    return {
        "detection_rate": len(vals) / n_total,
        "median_ewt_minutes": float(np.median(vals)),
        "iqr_low": float(np.percentile(vals, 25)),
        "iqr_high": float(np.percentile(vals, 75)),
        "n_censored": float(n_total - len(vals)),
    }


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def ring_level_metrics(
    predicted: Sequence[set], truth: Sequence[set], theta: float = 0.5
) -> dict[str, float]:
    """Ring-level precision/recall under greedy one-to-one Jaccard matching.

    Implements §11.4.3. Define theta before looking at results, and report the
    curve over theta in {0.3, 0.5, 0.7} -- partial rings are still useful to an
    analyst, and hiding that would be dishonest.
    """
    pairs = sorted(
        (
            (jaccard(p, t), i, j)
            for i, p in enumerate(predicted)
            for j, t in enumerate(truth)
        ),
        reverse=True,
    )
    used_p: set[int] = set()
    used_t: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, i, j in pairs:
        if score < theta:
            break
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        matches.append((i, j, score))

    pred_members: set = set().union(*predicted) if predicted else set()
    true_members: set = set().union(*truth) if truth else set()
    hit = pred_members & true_members

    return {
        "ring_precision": len(matches) / max(len(predicted), 1),
        "ring_recall": len(matches) / max(len(truth), 1),
        "mean_jaccard": float(np.mean([s for _, _, s in matches])) if matches else 0.0,
        "member_precision": len(hit) / max(len(pred_members), 1),
        "member_recall": len(hit) / max(len(true_members), 1),
    }
