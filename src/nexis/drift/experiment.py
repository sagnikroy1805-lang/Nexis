"""The drift experiment: one known shift, one detector, one policy, versus static.

Implements Concept Mastery §13.6.1 and the protocol of feasibility §2.3: a
controlled shift at a known time (from nexis.data.synthetic), a detector with a
calibrated threshold (detectors.py), an adaptation policy (adaptation.py), a
static model run over the identical stream, and latency accounting -- how long
after onset the alarm fires, and how much PR-AUC is lost in that gap. Uses the
monitoring of §13.3-13.4 and the change-point detectors of §13.5.

Timeline (all boundaries half-open, [start, end)):

    | initial fit | gap | reference window | w0 | w1 | w2 | ...
                        ^ref_start          ^reference_end

- The initial model is fitted on rows before ref_start - gap.
- The reference window is scored out-of-sample by that model; it fixes the
  monitor's reference distributions and the initial alert threshold.
- Each window is then scored by both arms (static, adaptive) BEFORE anything
  uses its labels. Labels only become usable for retraining after the window
  is scored, and only up to end - gap, where gap = max(embargo, label_delay).
  The label column is dropped from every frame a model scores or a monitor
  reads, so neither can see a label even by accident.

Rule 3 note: one call is one seed. Run it over SEEDS (different generator and
model seeds) and report mean +/- std of latency and PR-AUC lost; a single run
is an illustration, not a result.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nexis.drift.adaptation import (
    AdaptationContext,
    AdaptationPolicy,
    DriftAction,
    LabelLeakError,
    alert_threshold_for_budget,
    fit_chronological,
)
from nexis.drift.detectors import Detector
from nexis.drift.monitors import PSI_DRIFT, PSI_WARN, PSIMonitor, WindowReport
from nexis.evaluation.harness import Scorer
from nexis.evaluation.metrics import evaluate


@dataclass
class ArmMetrics:
    """One arm (static or adaptive) on one window. None = undefined (no positives)."""

    pr_auc: float | None
    roc_auc: float | None
    precision_at_budget: float | None
    recall_at_budget: float | None
    alert_rate: float
    recall_at_threshold: float | None
    threshold: float
    model_version: int


@dataclass
class WindowRecord:
    """Everything observed on one window, in the order it was observed."""

    index: int
    start: str
    end: str
    n: int
    n_pos: int
    prevalence: float
    static: ArmMetrics
    adaptive: ArmMetrics
    monitor: WindowReport
    monitored_value: float
    detector_updated: bool
    alarm: bool
    action: str | None
    # Newest labelled row seen by the adaptive model that SCORED this window
    # (not the one an alarm here installs for the next window). Always earlier
    # than start - gap; that is the audit trail for the label discipline.
    adaptive_train_end: str


@dataclass
class DriftExperimentResult:
    """Full record of one run. `dataclasses.asdict(result)` is JSON-serialisable.

    Loss metrics are sums over windows of (baseline PR-AUC - window PR-AUC), in
    PR-AUC x windows; windows without positives are skipped. They are signed: a
    window better than the baseline reduces the loss.
    """

    name: str
    detector: dict[str, Any]
    policy: dict[str, Any]
    monitor_metric: str
    window: str
    gap: str
    alert_budget_rate: float
    initial_train_start: str
    initial_train_end: str
    n_initial_train: int
    reference_start: str
    reference_end: str
    reference_pr_auc: float | None
    drift_onset: str | None
    windows: list[WindowRecord] = field(default_factory=list)
    actions: list[DriftAction] = field(default_factory=list)
    alarm_windows: list[int] = field(default_factory=list)
    first_alarm_at: str | None = None
    detected_at: str | None = None
    detection_window: int | None = None
    detection_latency_hours: float | None = None
    false_alarms_before_onset: int = 0
    baseline_pr_auc: float | None = None
    pr_auc_lost_onset_to_detection: float | None = None
    pr_auc_lost_static: float | None = None
    pr_auc_lost_adaptive: float | None = None
    mean_pr_auc_post_onset_static: float | None = None
    mean_pr_auc_post_onset_adaptive: float | None = None


def _arm_metrics(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    alert_budget_rate: float,
    model_version: int,
    split: str,
) -> ArmMetrics:
    """Ranking metrics via the shared metrics module, plus the operating point.

    Ranking metrics (PR-AUC, recall in the top budget-share) ignore the
    threshold; alert_rate and recall_at_threshold measure the threshold the arm
    was actually running, which is what a label-free recalibration changes.
    """
    n, n_pos = int(y.size), int(y.sum())
    alerts = scores >= threshold
    out = ArmMetrics(
        pr_auc=None,
        roc_auc=None,
        precision_at_budget=None,
        recall_at_budget=None,
        alert_rate=float(alerts.mean()) if n else 0.0,
        recall_at_threshold=float(alerts[y == 1].mean()) if n_pos else None,
        threshold=float(threshold),
        model_version=model_version,
    )
    if n >= 2 and 0 < n_pos < n:
        r = evaluate(
            y,
            scores,
            model="window",
            seed=0,
            split=split,
            alert_budget=max(1, round(alert_budget_rate * n)),
        )
        out.pr_auc, out.roc_auc = r.pr_auc, r.roc_auc
        out.precision_at_budget, out.recall_at_budget = r.precision_at_budget, r.recall_at_budget
    return out


def _mean(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def _loss(values: Sequence[float | None], baseline: float | None) -> float | None:
    vals = [v for v in values if v is not None]
    if baseline is None or not vals:
        return None
    return float(sum(baseline - v for v in vals))


def run_drift_experiment(
    stream: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str,
    time_col: str,
    model_factory: Callable[[], Scorer],
    reference_end: str | pd.Timestamp,
    window: str,
    detector: Detector,
    policy: AdaptationPolicy,
    alert_budget_rate: float,
    *,
    drift_onset: str | pd.Timestamp | None = None,
    reference_window: str | None = None,
    embargo: str = "0min",
    label_delay: str = "0min",
    val_frac: float = 0.2,
    monitor_metric: str = "score_psi",
    psi_bins: int = 10,
    psi_warn: float = PSI_WARN,
    psi_drift: float = PSI_DRIFT,
    rereference_after_action: bool = True,
    name: str = "drift_experiment",
) -> DriftExperimentResult:
    """Walk forward window by window; run a static and an adaptive arm side by side.

    Args:
        stream: rows with feature columns, label and time; leak-free features
            already computed (this function does not compute features).
        feature_cols: columns the monitor tracks (Tier 1). Models choose their
            own columns from the label-free frame they are given.
        model_factory: zero-argument factory of a fresh Scorer; seed it inside.
        reference_end: first scored window starts here.
        window: walk-forward step, e.g. "1D".
        detector: consumes `monitor_metric` once per window; deep-copied, so the
            caller's object is not mutated.
        policy: what the adaptive arm does on alarm.
        alert_budget_rate: share of rows analysts can review; sets thresholds.
        drift_onset: known onset (synthetic data) for latency accounting.
        reference_window: monitor reference length; defaults to `window`.
        embargo / label_delay: rows usable for any fit end this long before the
            first window the fitted model scores (rule 1 embargo; label lag).
        rereference_after_action: after a model or threshold change, the next
            window becomes the monitor's new reference (it is still scored and
            evaluated) and the detector is reset. Without this, the new model's
            score distribution is compared with the old model's reference and
            the detector re-fires every window -- a retraining loop (§13.6).
    """
    missing = [c for c in (*feature_cols, label_col, time_col) if c not in stream.columns]
    if missing:
        raise KeyError(f"stream is missing columns {missing}")
    times = pd.to_datetime(stream[time_col])
    order = np.argsort(times.to_numpy(), kind="stable")
    data = stream.iloc[order].reset_index(drop=True)
    t = pd.Series(times.to_numpy()[order])
    y_all = data[label_col].to_numpy().astype(int)
    unlabelled = data.drop(columns=[label_col])

    step = pd.Timedelta(window)
    ref_len = pd.Timedelta(reference_window) if reference_window else step
    gap = max(pd.Timedelta(embargo), pd.Timedelta(label_delay))
    ref_end = pd.Timestamp(reference_end)
    ref_start = ref_end - ref_len
    onset = pd.Timestamp(drift_onset) if drift_onset is not None else None
    if step <= pd.Timedelta(0):
        raise ValueError("window must be positive")

    initial = (t < ref_start - gap).to_numpy()
    in_ref = ((t >= ref_start) & (t < ref_end)).to_numpy()
    if not in_ref.any():
        raise ValueError("reference window holds no rows")
    fitted = fit_chronological(
        model_factory,
        data.loc[initial],
        time_col=time_col,
        label_col=label_col,
        not_after=ref_start,
        val_frac=val_frac,
    )
    model = fitted.model
    ref_scores = np.asarray(model.score(unlabelled.loc[in_ref]), dtype=float)
    threshold = alert_threshold_for_budget(ref_scores, alert_budget_rate)
    ref_metrics = _arm_metrics(y_all[in_ref], ref_scores, threshold, alert_budget_rate, 0, "reference")

    def fit_monitor(frame: pd.DataFrame, scores: np.ndarray, thr: float) -> PSIMonitor:
        return PSIMonitor.fit(
            frame, scores, feature_cols, bins=psi_bins, alert_threshold=thr,
            warn=psi_warn, drift=psi_drift, time_col=time_col,
        )

    monitor = fit_monitor(unlabelled.loc[in_ref], ref_scores, threshold)
    detector = copy.deepcopy(detector)

    static_model, static_thr = model, threshold
    adapt_model, adapt_thr, version = model, threshold, 0
    static_train_end = adapt_train_end = pd.Timestamp(fitted.train_end)
    pending_rereference = False

    result = DriftExperimentResult(
        name=name,
        detector=detector.describe(),
        policy=policy.describe(),
        monitor_metric=monitor_metric,
        window=str(step),
        gap=str(gap),
        alert_budget_rate=alert_budget_rate,
        initial_train_start=fitted.train_start,
        initial_train_end=fitted.train_end,
        n_initial_train=fitted.n_train,
        reference_start=ref_start.isoformat(),
        reference_end=ref_end.isoformat(),
        reference_pr_auc=ref_metrics.pr_auc,
        drift_onset=onset.isoformat() if onset is not None else None,
    )

    t_max = t.iloc[-1]
    k = 0
    while (start := ref_end + k * step) <= t_max:
        end = start + step
        mask = ((t >= start) & (t < end)).to_numpy()
        if not mask.any():
            k += 1
            continue
        # LEAKAGE GUARD: both arms must have been fitted on rows ending more than
        # `gap` before this window. fit_chronological already enforces "before
        # the window" at fit time; this re-checks the invariant, embargo
        # included, at use time -- where a violation would actually matter.
        for arm, fitted_to in (("static", static_train_end), ("adaptive", adapt_train_end)):
            if fitted_to >= start - gap:
                raise LabelLeakError(
                    f"{arm} model fitted on rows up to {fitted_to}, scoring window "
                    f"from {start} with gap {gap}"
                )
        scored_by_train_end = adapt_train_end

        frame = unlabelled.loc[mask]
        y = y_all[mask]
        s_static = np.asarray(static_model.score(frame), dtype=float)
        s_adapt = s_static if adapt_model is static_model else np.asarray(adapt_model.score(frame), dtype=float)
        split = f"window_{k}"
        m_static = _arm_metrics(y, s_static, static_thr, alert_budget_rate, 0, split)
        m_adapt = _arm_metrics(y, s_adapt, adapt_thr, alert_budget_rate, version, split)

        report = monitor.window_report(frame, s_adapt, alert_threshold=adapt_thr)
        value = float(getattr(report, monitor_metric))
        alarm, updated, action_text = False, False, None
        if pending_rereference:
            monitor = fit_monitor(frame, s_adapt, adapt_thr)
            detector.reset()
            pending_rereference = False
        else:
            alarm, updated = bool(detector.update(value)), True

        if alarm:
            usable = (t < end - gap).to_numpy()
            ctx = AdaptationContext(
                now=end,
                next_window_start=end,
                labelled=data.loc[usable],
                recent_frame=frame,
                recent_scores=s_adapt,
                model=adapt_model,
                threshold=adapt_thr,
                model_factory=model_factory,
                alert_budget_rate=alert_budget_rate,
                label_col=label_col,
                time_col=time_col,
                val_frac=val_frac,
            )
            outcome = policy.adapt(ctx)
            thr_before = adapt_thr
            changed = False
            if outcome.model is not None:
                adapt_model, version, changed = outcome.model, version + 1, True
                adapt_train_end = pd.Timestamp(outcome.train_end)
            if outcome.threshold is not None:
                adapt_thr, changed = outcome.threshold, True
            pending_rereference = changed and rereference_after_action
            action_text = outcome.action
            result.actions.append(
                DriftAction(
                    detected_at=end.isoformat(),
                    window_index=k,
                    metric=monitor_metric,
                    value=value,
                    level=report.status,
                    action=outcome.action,
                    threshold_before=float(thr_before),
                    threshold_after=float(adapt_thr),
                    model_version=version,
                    train_start=outcome.train_start,
                    train_end=outcome.train_end,
                    n_train=outcome.n_train,
                    n_train_pos=outcome.n_train_pos,
                )
            )

        result.windows.append(
            WindowRecord(
                index=k,
                start=start.isoformat(),
                end=end.isoformat(),
                n=int(mask.sum()),
                n_pos=int(y.sum()),
                prevalence=float(y.mean()),
                static=m_static,
                adaptive=m_adapt,
                monitor=report,
                monitored_value=value,
                detector_updated=updated,
                alarm=alarm,
                action=action_text,
                adaptive_train_end=scored_by_train_end.isoformat(),
            )
        )
        k += 1

    _summarise(result, onset)
    return result


def _summarise(result: DriftExperimentResult, onset: pd.Timestamp | None) -> None:
    """Detection time, latency, false alarms and PR-AUC lost (§13.5, §13.6.1).

    A window counts as post-onset if it ends after the onset (a window that
    straddles the onset contains drifted rows). Alarms are raised at the end of
    a window, so latency is measured from onset to that window's end.
    """
    wins = result.windows
    result.alarm_windows = [w.index for w in wins if w.alarm]
    alarms = [w for w in wins if w.alarm]
    result.first_alarm_at = alarms[0].end if alarms else None

    def ends(w: WindowRecord) -> pd.Timestamp:
        return pd.Timestamp(w.end)

    if onset is None:
        pre, post = wins, []
        result.false_alarms_before_onset = len(alarms)
    else:
        pre = [w for w in wins if ends(w) <= onset]
        post = [w for w in wins if ends(w) > onset]
        result.false_alarms_before_onset = sum(w.alarm for w in pre)

    pre_mean = _mean([w.static.pr_auc for w in pre])
    result.baseline_pr_auc = pre_mean if pre_mean is not None else result.reference_pr_auc
    if onset is None:
        return

    hit = next((w for w in post if w.alarm), None)
    if hit is not None:
        result.detected_at = hit.end
        result.detection_window = hit.index
        result.detection_latency_hours = (ends(hit) - onset) / pd.Timedelta(hours=1)
        gap_windows = [w for w in post if ends(w) <= ends(hit)]
    else:
        gap_windows = post  # never detected: the whole post-onset period is the gap
    base = result.baseline_pr_auc
    result.pr_auc_lost_onset_to_detection = _loss([w.adaptive.pr_auc for w in gap_windows], base)
    result.pr_auc_lost_static = _loss([w.static.pr_auc for w in post], base)
    result.pr_auc_lost_adaptive = _loss([w.adaptive.pr_auc for w in post], base)
    result.mean_pr_auc_post_onset_static = _mean([w.static.pr_auc for w in post])
    result.mean_pr_auc_post_onset_adaptive = _mean([w.adaptive.pr_auc for w in post])


# Categorical slots 1-2 of the project's reference chart palette, and neutral ink
# for annotations (text and reference lines never wear a series colour).
_ADAPTIVE, _STATIC = "#2a78d6", "#eb6834"
_INK, _MUTED, _GRID = "#0b0b0b", "#52514e", "#e4e3df"


def plot_drift_experiment(result: DriftExperimentResult, path: str | Path) -> Path:
    """The paper's drift figure (§13.6.1): performance over time, static vs adaptive.

    Three stacked panels on a shared time axis (never a dual y-axis): PR-AUC per
    window, the monitored drift statistic with its warn/drift levels, and the
    alert rate against the budget. Onset (dashed) and detection (solid) are
    marked on every panel.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    wins = result.windows
    if not wins:
        raise ValueError("result has no windows to plot")
    mid = [pd.Timestamp(w.start) + (pd.Timestamp(w.end) - pd.Timestamp(w.start)) / 2 for w in wins]

    def series(get: Callable[[WindowRecord], float | None]) -> np.ndarray:
        return np.array([np.nan if (v := get(w)) is None else v for w in wins], dtype=float)

    fig = Figure(figsize=(9, 7.5), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 1, sharex=True, height_ratios=[2.2, 1.2, 1.2])
    line = {"linewidth": 2, "marker": "o", "markersize": 4}

    ax = axes[0]
    ax.plot(mid, series(lambda w: w.static.pr_auc), color=_STATIC, label="static", **line)
    ax.plot(mid, series(lambda w: w.adaptive.pr_auc), color=_ADAPTIVE, label=f"adaptive ({result.policy.get('policy', '')})", **line)
    if result.baseline_pr_auc is not None:
        ax.axhline(result.baseline_pr_auc, color=_MUTED, linewidth=1, linestyle=":", label="pre-onset baseline")
    ax.set_ylabel("PR-AUC per window")
    ax.legend(loc="lower left", frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(mid, series(lambda w: w.monitored_value), color=_INK, **line)
    if "psi" in result.monitor_metric:
        for level, text in ((PSI_WARN, "warn"), (PSI_DRIFT, "drift")):
            ax.axhline(level, color=_MUTED, linewidth=1, linestyle=":")
            ax.annotate(text, (1.0, level), xycoords=("axes fraction", "data"), xytext=(-4, 2),
                        textcoords="offset points", ha="right", fontsize=8, color=_MUTED)
    ax.set_ylabel(result.monitor_metric.replace("_", " "))

    ax = axes[2]
    ax.plot(mid, series(lambda w: w.static.alert_rate), color=_STATIC, label="static", **line)
    ax.plot(mid, series(lambda w: w.adaptive.alert_rate), color=_ADAPTIVE, label="adaptive", **line)
    ax.axhline(result.alert_budget_rate, color=_MUTED, linewidth=1, linestyle=":", label="budget")
    ax.set_ylabel("alert rate")
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    marks: list[tuple[str, str, str]] = []
    if result.drift_onset is not None:
        marks.append((result.drift_onset, "--", "drift onset"))
    if result.detected_at is not None:
        lat = result.detection_latency_hours
        marks.append((result.detected_at, "-", f"detected (+{lat:.0f}h)" if lat is not None else "detected"))
    for a in result.actions:
        if a.detected_at != result.detected_at:
            marks.append((a.detected_at, ":", ""))
    for ax in axes:
        ax.grid(axis="y", color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for when, style, _ in marks:
            ax.axvline(pd.Timestamp(when), color=_MUTED, linestyle=style, linewidth=1)
    for when, _, text in marks:
        if text:
            axes[0].annotate(text, (pd.Timestamp(when), 1.0), xycoords=("data", "axes fraction"),
                             xytext=(3, -10), textcoords="offset points", fontsize=8, color=_INK)

    fig.suptitle(result.name, color=_INK, fontsize=11)
    fig.autofmt_xdate()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    return out
