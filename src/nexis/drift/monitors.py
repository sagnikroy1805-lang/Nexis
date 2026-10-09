"""Label-free drift monitoring: PSI and KS on input features and risk scores.

Implements Concept Mastery §13.1 (data drift: P(x) moves), §13.3 (monitoring when
labels are absent -- Tier 1 input distributions, Tier 2 the score distribution,
Tier 3 the alert rate) and §13.4 (PSI with reference-decile bins and Laplace
smoothing; KS reported as an effect size, not a p-value).

Nothing here sees a label. That is the point: labels lag by weeks (§13.2), so
these are the only signals available when drift begins. Note what that means for
concept drift: if P(x) is unchanged and only P(y | x) moves (for example a new
typology at 0.2% prevalence), none of these monitors is expected to fire, and
saying so is part of an honest drift result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Conventional credit-risk reading of PSI (§13.4.1): < 0.1 no significant shift,
# 0.1-0.25 moderate (investigate), >= 0.25 significant (act).
PSI_WARN = 0.1
PSI_DRIFT = 0.25


def _clean(values: Any) -> np.ndarray:
    """Float array of a sample; bool and integer inputs are cast, NaN is kept."""
    arr = np.asarray(values)
    if arr.dtype == object or arr.dtype.kind in "OSUM":
        raise TypeError(
            "PSI needs numeric values; encode categoricals and timestamps first"
        )
    return arr.astype(float).ravel()


def psi_bin_edges(reference: Any, bins: int = 10) -> np.ndarray:
    """Interior bin edges: the reference sample's quantiles, de-duplicated.

    LEAKAGE GUARD: edges come from the reference only. If they were taken from
    the pooled reference + current sample, the window being judged would shape
    the yardstick it is judged by, and a shift would partly hide itself.
    Duplicate quantiles (mass points, binary flags) collapse into one edge, and
    bins are right-closed, so a 0/1 feature still gets one bin per value.
    """
    ref = _clean(reference)
    ref = ref[~np.isnan(ref)]
    if ref.size == 0:
        return np.zeros(0)
    qs = np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    return np.unique(qs)


@dataclass(frozen=True)
class PSIReference:
    """A frozen reference distribution: bin edges and smoothed bin proportions.

    Fitted once on the reference window, then reused unchanged for every later
    window, so every PSI in a time series is measured against the same bins.
    The last bin counts NaN, so a change in missing rate is a shift too (§13.3).
    """

    edges: np.ndarray
    ref_props: np.ndarray
    n_ref: int
    alpha: float = 1.0

    @classmethod
    def fit(cls, reference: Any, bins: int = 10, alpha: float = 1.0) -> PSIReference:
        ref = _clean(reference)
        edges = psi_bin_edges(ref, bins)
        return cls(
            edges=edges,
            ref_props=_proportions(ref, edges, alpha),
            n_ref=int(ref.size),
            alpha=alpha,
        )

    def proportions(self, current: Any) -> np.ndarray:
        return _proportions(_clean(current), self.edges, self.alpha)

    def psi(self, current: Any) -> float:
        """PSI = sum (p_cur - p_ref) ln(p_cur / p_ref): the Jeffreys divergence."""
        p_cur = self.proportions(current)
        return float(np.sum((p_cur - self.ref_props) * np.log(p_cur / self.ref_props)))


def _proportions(values: np.ndarray, edges: np.ndarray, alpha: float) -> np.ndarray:
    """Laplace-smoothed bin shares, so an empty bin never gives an infinite PSI."""
    nan = np.isnan(values)
    idx = np.searchsorted(edges, values[~nan], side="left")
    counts = np.bincount(idx, minlength=edges.size + 1).astype(float)
    counts = np.append(counts, float(nan.sum()))
    return (counts + alpha) / (counts.sum() + alpha * counts.size)


def psi(reference: Any, current: Any, bins: int = 10) -> float:
    """Population Stability Index of `current` against `reference` (§13.4.1)."""
    return PSIReference.fit(reference, bins).psi(current)


def ks_drift(reference: Any, current: Any) -> tuple[float, float]:
    """Two-sample KS (statistic, p-value), NaNs dropped.

    Report the statistic: with thousands of rows per window, p < 0.001 is
    uninformative, while the statistic is an effect size (§13.4.2).
    """
    ref, cur = _clean(reference), _clean(current)
    res = ks_2samp(ref[~np.isnan(ref)], cur[~np.isnan(cur)])
    return float(res.statistic), float(res.pvalue)


def psi_null_distribution(
    values: Any, sample_size: int, n_draws: int = 200, bins: int = 10, seed: int = 0
) -> np.ndarray:
    """PSI values under no drift, for calibrating an alarm threshold.

    Each draw splits `values` (a stationary sample, e.g. reference-period scores)
    at random into a disjoint reference part and a current part of
    `sample_size`, and computes PSI between them. Disjoint halves matter: a
    subsample compared with the sample it came from understates the noise.
    Feed the result to nexis.drift.detectors.calibrate_threshold.
    """
    arr = _clean(values)
    if arr.size < sample_size + 10 * bins:
        raise ValueError(
            f"need at least sample_size + 10*bins = {sample_size + 10 * bins} values"
        )
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws)
    for i in range(n_draws):
        perm = rng.permutation(arr)
        out[i] = psi(perm[sample_size:], perm[:sample_size], bins)
    return out


@dataclass
class WindowReport:
    """One monitoring window. JSON-serialisable via dataclasses.asdict."""

    start: str | None
    end: str | None
    n: int
    psi_max: float
    psi_top: dict[str, float]
    score_psi: float
    score_ks: float
    alert_rate: float | None
    status: str


@dataclass
class PSIMonitor:
    """Tier 1 (features) + Tier 2 (scores) + Tier 3 (alert rate) monitor (§13.3).

    `status` is driven by the score PSI, because the score aggregates every
    input change through the model: a shift the model ignores does not move it
    (§13.3, "build the dashboard around Tier 2"). Feature PSIs are diagnosis --
    they say *which* input moved once the score PSI fires.
    """

    feature_cols: tuple[str, ...]
    feature_refs: dict[str, PSIReference]
    score_ref: PSIReference
    reference_scores: np.ndarray
    alert_threshold: float | None = None
    warn: float = PSI_WARN
    drift: float = PSI_DRIFT
    time_col: str = "timestamp"
    top_k: int = 5
    reference_span: tuple[str | None, str | None] = field(default=(None, None))

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        scores: Any,
        feature_cols: Sequence[str],
        *,
        bins: int = 10,
        alert_threshold: float | None = None,
        warn: float = PSI_WARN,
        drift: float = PSI_DRIFT,
        time_col: str = "timestamp",
    ) -> PSIMonitor:
        """Freeze the reference: one PSIReference per feature, one for the scores."""
        scores = _clean(scores)
        if len(scores) != len(frame):
            raise ValueError("scores and frame have different lengths")
        return cls(
            feature_cols=tuple(feature_cols),
            feature_refs={c: PSIReference.fit(frame[c], bins) for c in feature_cols},
            score_ref=PSIReference.fit(scores, bins),
            reference_scores=scores.copy(),
            alert_threshold=alert_threshold,
            warn=warn,
            drift=drift,
            time_col=time_col,
            reference_span=_span(frame, time_col),
        )

    def status_for(self, value: float) -> str:
        if value < self.warn:
            return "ok"
        if value < self.drift:
            return "warn"
        return "drift"

    def window_report(
        self,
        frame: pd.DataFrame,
        scores: Any,
        *,
        alert_threshold: float | None = None,
    ) -> WindowReport:
        """Compare one window with the frozen reference. Uses no labels."""
        scores = _clean(scores)
        if len(scores) != len(frame):
            raise ValueError("scores and frame have different lengths")
        feat = {c: self.feature_refs[c].psi(frame[c]) for c in self.feature_cols}
        top = dict(sorted(feat.items(), key=lambda kv: -kv[1])[: self.top_k])
        score_psi = self.score_ref.psi(scores)
        threshold = self.alert_threshold if alert_threshold is None else alert_threshold
        alert_rate = (
            float(np.mean(scores >= threshold))
            if threshold is not None and scores.size
            else None
        )
        score_ks = ks_drift(self.reference_scores, scores)[0] if scores.size else 0.0
        start, end = _span(frame, self.time_col)
        return WindowReport(
            start=start,
            end=end,
            n=len(frame),
            psi_max=max(feat.values(), default=0.0),
            psi_top={k: float(v) for k, v in top.items()},
            score_psi=score_psi,
            score_ks=score_ks,
            alert_rate=alert_rate,
            status=self.status_for(score_psi),
        )


def _span(frame: pd.DataFrame, time_col: str) -> tuple[str | None, str | None]:
    if time_col not in frame.columns or frame.empty:
        return None, None
    t = pd.to_datetime(frame[time_col])
    return t.min().isoformat(), t.max().isoformat()
