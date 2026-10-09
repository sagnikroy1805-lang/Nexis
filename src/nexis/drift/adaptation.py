"""Adaptation policies: what the system does when a drift detector fires.

Implements Concept Mastery §13.6 (triggered retraining on a recent window,
retraining on everything, and a label-free recalibration of the alert
threshold) for the single experiment of §13.6.1 / feasibility §2.3.

Policies never choose their own training data. The experiment hands them an
`AdaptationContext` whose `labelled` frame has already been cut to the rows
whose labels are usable at that moment, and every fit goes through
`fit_chronological`, which refuses a frame reaching into the window about to be
scored. A policy therefore cannot leak a label even by mistake.

Every policy returns a `PolicyOutcome`; the experiment wraps it in a
`DriftAction` record, which is the audit trail of why the model in production
changed (§13.7: drift_events_since_deploy).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from nexis.evaluation.harness import Scorer


class LabelLeakError(RuntimeError):
    """A model was about to be fitted on rows it would later be scored on."""


def alert_threshold_for_budget(scores: Any, alert_budget_rate: float) -> float:
    """Score threshold that flags `alert_budget_rate` of these scores (§1.3).

    Uses only scores, never labels: analyst capacity is a share of volume, and
    a bank sets the cut-off on volume, not on outcomes.
    """
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        raise ValueError("cannot set a threshold on zero scores")
    if not 0.0 < alert_budget_rate < 1.0:
        raise ValueError("alert_budget_rate must be in (0, 1)")
    return float(np.quantile(s, 1.0 - alert_budget_rate))


@dataclass
class FittedModel:
    """A freshly fitted model and the extent of the labels it saw.

    val: the newest slice of the labelled rows, held out from parameter fitting
    (scorers use it for early stopping at most). Policies set the new alert
    threshold on its scores, so no model is ever scored on its own training rows.
    """

    model: Scorer
    train_start: str
    train_end: str
    n_train: int
    n_train_pos: int
    val: pd.DataFrame


def fit_chronological(
    model_factory: Callable[[], Scorer],
    labelled: pd.DataFrame,
    *,
    time_col: str,
    label_col: str,
    not_after: pd.Timestamp,
    val_frac: float = 0.2,
) -> FittedModel:
    """Fit a fresh model: oldest (1 - val_frac) of rows to train, newest to val.

    LEAKAGE GUARD: raises LabelLeakError if any row is at or after `not_after`,
    the start of the next window to be scored. A window's labels become usable
    only after that window has been scored; this check is what makes that a
    property of the code rather than of the caller's care.

    The split is by row order in time (never shuffled, rule 1), so validation
    is always the most recent slice -- the closest proxy for what comes next.
    """
    if labelled.empty:
        raise ValueError("no labelled rows to fit on")
    t = pd.to_datetime(labelled[time_col])
    if t.max() >= not_after:
        raise LabelLeakError(
            f"training rows reach {t.max()}, at or after the next scored window "
            f"starting {not_after}"
        )
    ordered = labelled.iloc[np.argsort(t.to_numpy(), kind="stable")]
    n_val = int(round(len(ordered) * val_frac))
    n_val = min(max(n_val, 1), len(ordered) - 1) if len(ordered) > 1 else 0
    train, val = ordered.iloc[: len(ordered) - n_val], ordered.iloc[len(ordered) - n_val :]
    return FittedModel(
        model=model_factory().fit(train, val),
        train_start=t.min().isoformat(),
        train_end=t.max().isoformat(),
        n_train=len(ordered),
        n_train_pos=int(ordered[label_col].sum()),
        val=val,
    )


@dataclass
class AdaptationContext:
    """Everything a policy may use at decision time `now` (end of a scored window).

    labelled: rows whose labels are usable now (timestamp < usable_until), sorted
        by time. Never contains a row of the next window.
    recent_frame / recent_scores: the window just scored, label column removed,
        and the current model's scores on it.
    """

    now: pd.Timestamp
    next_window_start: pd.Timestamp
    labelled: pd.DataFrame
    recent_frame: pd.DataFrame
    recent_scores: np.ndarray
    model: Scorer
    threshold: float
    model_factory: Callable[[], Scorer]
    alert_budget_rate: float
    label_col: str
    time_col: str
    val_frac: float = 0.2


@dataclass
class PolicyOutcome:
    """What a policy decided. None means 'keep the current one'."""

    action: str
    model: Scorer | None = None
    threshold: float | None = None
    train_start: str | None = None
    train_end: str | None = None
    n_train: int = 0
    n_train_pos: int = 0


@dataclass
class DriftAction:
    """One alarm and the response to it. JSON-serialisable via asdict."""

    detected_at: str
    window_index: int
    metric: str
    value: float
    level: str
    action: str
    threshold_before: float
    threshold_after: float
    model_version: int
    train_start: str | None = None
    train_end: str | None = None
    n_train: int = 0
    n_train_pos: int = 0


class AdaptationPolicy(Protocol):
    name: str

    def adapt(self, ctx: AdaptationContext) -> PolicyOutcome: ...

    def describe(self) -> dict[str, Any]: ...


@dataclass
class StaticPolicy:
    """Never changes anything: the baseline every adaptive policy must beat."""

    name: str = "static"

    def adapt(self, ctx: AdaptationContext) -> PolicyOutcome:
        return PolicyOutcome(action="none")

    def describe(self) -> dict[str, Any]:
        return {"policy": self.name}


def _retrain(ctx: AdaptationContext, rows: pd.DataFrame, label: str, min_pos: int) -> PolicyOutcome:
    """Shared retrain path: fit, then re-set the threshold for the new score scale.

    The new model's scores live on a different scale, so the old threshold is
    meaningless. It is re-derived label-free from the new model's scores on its
    validation slice: the newest usable rows, which the model was not fitted on.
    Scoring the training rows instead would give in-sample (too confident)
    scores and a threshold that is off from the first window it is used on.
    """
    n_pos = int(rows[ctx.label_col].sum()) if not rows.empty else 0
    if n_pos < min_pos:
        return PolicyOutcome(
            action=f"{label}: skipped, {n_pos} positives < min_positives={min_pos}",
            n_train=len(rows),
            n_train_pos=n_pos,
        )
    fitted = fit_chronological(
        ctx.model_factory,
        rows,
        time_col=ctx.time_col,
        label_col=ctx.label_col,
        not_after=ctx.next_window_start,
        val_frac=ctx.val_frac,
    )
    holdout = fitted.val if not fitted.val.empty else rows
    scores = fitted.model.score(holdout.drop(columns=[ctx.label_col]))
    return PolicyOutcome(
        action=f"{label} [{fitted.train_start} .. {fitted.train_end}]",
        model=fitted.model,
        threshold=alert_threshold_for_budget(scores, ctx.alert_budget_rate),
        train_start=fitted.train_start,
        train_end=fitted.train_end,
        n_train=fitted.n_train,
        n_train_pos=fitted.n_train_pos,
    )


@dataclass
class RetrainRecentWindow:
    """On alarm, refit on the most recent `window` of usable labels (§13.6).

    Window length trades adaptability against sample size: with rare positives
    a short window may hold too few to learn from, so a retrain on fewer than
    `min_positives` is skipped (and recorded as skipped) rather than run.
    """

    window: str = "7D"
    min_positives: int = 5
    name: str = "retrain_recent_window"

    def adapt(self, ctx: AdaptationContext) -> PolicyOutcome:
        t = pd.to_datetime(ctx.labelled[ctx.time_col])
        rows = ctx.labelled.loc[t >= t.max() - pd.Timedelta(self.window)] if len(t) else ctx.labelled
        return _retrain(ctx, rows, f"retrain_recent_window({self.window})", self.min_positives)

    def describe(self) -> dict[str, Any]:
        return {"policy": self.name, "window": self.window, "min_positives": self.min_positives}


@dataclass
class RetrainAll:
    """On alarm, refit on every usable label so far. Keeps rare old positives."""

    min_positives: int = 5
    name: str = "retrain_all"

    def adapt(self, ctx: AdaptationContext) -> PolicyOutcome:
        return _retrain(ctx, ctx.labelled, "retrain_all", self.min_positives)

    def describe(self) -> dict[str, Any]:
        return {"policy": self.name, "min_positives": self.min_positives}


@dataclass
class RecalibrateThreshold:
    """Label-free: re-set the alert threshold so the budget holds on recent scores.

    Fixes the operational symptom of drift (alert volume off budget) without
    labels, so it can act weeks before any retrain could. It cannot fix the
    ranking: PR-AUC is unchanged by construction.
    """

    name: str = "recalibrate_threshold"

    def adapt(self, ctx: AdaptationContext) -> PolicyOutcome:
        thr = alert_threshold_for_budget(ctx.recent_scores, ctx.alert_budget_rate)
        return PolicyOutcome(action=f"recalibrate_threshold {ctx.threshold:.6g} -> {thr:.6g}", threshold=thr)

    def describe(self) -> dict[str, Any]:
        return {"policy": self.name}
