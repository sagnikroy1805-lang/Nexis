"""Drift monitoring, detection and adaptation tests (Concept Mastery Module 13).

The failures that matter here are silent: PSI bins that peek at the window being
judged, a detector that fires on noise, a recalibration that misses its budget,
and -- the important one -- an experiment that lets a model learn from the
labels of a window before scoring it, which would make "adaptive beats static"
an artefact of leakage.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from nexis.data.synthetic import GeneratorConfig, generate
from nexis.drift.adaptation import (
    AdaptationContext,
    LabelLeakError,
    RecalibrateThreshold,
    RetrainAll,
    RetrainRecentWindow,
    StaticPolicy,
    alert_threshold_for_budget,
    fit_chronological,
)
from nexis.drift.detectors import (
    ADWIN,
    PageHinkley,
    ThresholdDetector,
    calibrate_threshold,
)
from nexis.drift.experiment import plot_drift_experiment, run_drift_experiment
from nexis.drift.monitors import (
    PSIMonitor,
    PSIReference,
    ks_drift,
    psi,
    psi_bin_edges,
    psi_null_distribution,
)

# --------------------------------------------------------------------------- #
# monitors
# --------------------------------------------------------------------------- #


def test_psi_is_zero_on_identical_samples():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-12)


def test_psi_is_small_on_same_distribution_and_large_on_shifted():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=20_000)
    assert psi(ref, rng.normal(size=20_000)) < 0.01
    assert psi(ref, rng.normal(1.0, 1.0, 20_000)) > 0.5


def test_psi_bins_depend_only_on_the_reference():
    """LEAKAGE GUARD: the window being judged must not shape the bins it is judged by."""
    rng = np.random.default_rng(2)
    ref = rng.normal(size=10_000)
    expected = np.unique(np.quantile(ref, np.linspace(0, 1, 11)[1:-1]))
    np.testing.assert_array_equal(psi_bin_edges(ref), expected)

    frozen = PSIReference.fit(ref)
    edges, props = frozen.edges.copy(), frozen.ref_props.copy()
    for current in (rng.normal(size=3000), rng.normal(3.0, 0.1, 500), rng.uniform(-9, 9, 50)):
        assert psi(ref, current) == pytest.approx(frozen.psi(current))
        np.testing.assert_array_equal(frozen.edges, edges)
        np.testing.assert_array_equal(frozen.ref_props, props)


def test_psi_monitor_bins_unchanged_by_reports():
    rng = np.random.default_rng(3)
    ref = pd.DataFrame({"a": rng.normal(size=4000)})
    mon = PSIMonitor.fit(ref, rng.random(4000), ["a"])
    before = (mon.feature_refs["a"].edges.copy(), mon.score_ref.edges.copy())
    mon.window_report(pd.DataFrame({"a": rng.normal(5.0, 1.0, 999)}), rng.random(999) + 3)
    np.testing.assert_array_equal(mon.feature_refs["a"].edges, before[0])
    np.testing.assert_array_equal(mon.score_ref.edges, before[1])


def test_psi_handles_binary_features_and_missing_rate():
    ref = np.r_[np.zeros(900), np.ones(100)]
    assert psi(ref, ref) == pytest.approx(0.0, abs=1e-12)
    assert psi(ref, np.r_[np.zeros(500), np.ones(500)]) > 0.25
    x = np.random.default_rng(4).normal(size=2000)
    with_nans = x.copy()
    with_nans[:600] = np.nan
    assert psi(x, with_nans) > 0.1


def test_psi_rejects_non_numeric_input():
    with pytest.raises(TypeError):
        psi(np.array(["a", "b"] * 50), np.array(["a"] * 100))


def test_ks_drift_reports_effect_size_and_pvalue():
    rng = np.random.default_rng(5)
    stat, p = ks_drift(rng.normal(size=3000), rng.normal(0.5, 1.0, 3000))
    assert 0.1 < stat < 0.3 and p < 1e-6
    stat_same, _ = ks_drift(rng.normal(size=3000), rng.normal(size=3000))
    assert stat_same < 0.05


def test_window_report_flags_the_shifted_feature():
    rng = np.random.default_rng(6)
    n = 5000
    ref = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    ref_scores = rng.random(n)
    thr = alert_threshold_for_budget(ref_scores, 0.02)
    mon = PSIMonitor.fit(ref, ref_scores, ["a", "b"], alert_threshold=thr)

    same = mon.window_report(
        pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}), rng.random(n)
    )
    assert same.status == "ok" and same.score_psi < 0.1
    assert same.alert_rate == pytest.approx(0.02, abs=0.01)

    moved = mon.window_report(
        pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(1.5, 1.0, n)}),
        rng.random(n) ** 0.3,
    )
    assert moved.status == "drift"
    assert next(iter(moved.psi_top)) == "b"
    assert moved.psi_max == pytest.approx(moved.psi_top["b"])
    assert moved.alert_rate > 0.05
    json.dumps(dataclasses.asdict(moved), allow_nan=False)


# --------------------------------------------------------------------------- #
# detectors
# --------------------------------------------------------------------------- #


def _stream(seed: int, n: int = 1000, shift_at: int | None = None, shift: float = 2.0):
    x = np.random.default_rng(seed).normal(size=n)
    if shift_at is not None:
        x[shift_at:] += shift
    return x


def _run(detector, values) -> list[int]:
    return [i for i, v in enumerate(values) if detector.update(v)]


def test_page_hinkley_is_silent_on_a_stationary_stream():
    for seed in range(5):
        assert _run(PageHinkley(delta=0.5, threshold=10.0), _stream(seed)) == []


def test_page_hinkley_detects_a_mean_shift_within_bounded_delay():
    for seed in range(5):
        ph = PageHinkley(delta=0.5, threshold=10.0)
        alarms = _run(ph, _stream(seed, shift_at=500))
        assert alarms, seed
        assert ph.detected_at == alarms[0]
        assert 500 <= ph.detected_at <= 520, (seed, ph.detected_at)


def test_page_hinkley_downward_direction():
    ph = PageHinkley(delta=0.5, threshold=10.0, direction="down")
    alarms = _run(ph, _stream(0, shift_at=500, shift=-2.0))
    assert alarms and 500 <= alarms[0] <= 520


def test_adwin_detects_shift_and_is_silent_without_one():
    for seed in range(3):
        assert _run(ADWIN(delta=0.002), _stream(seed, n=600)) == []
        ad = ADWIN(delta=0.002)
        alarms = _run(ad, _stream(seed, n=600, shift_at=300, shift=1.5))
        assert alarms and 300 <= alarms[0] <= 340, (seed, alarms)
        assert ad.width < 340  # the pre-shift regime was dropped


def test_threshold_detector_levels_and_patience():
    det = ThresholdDetector(drift_level=0.25, warn_level=0.1, patience=2)
    out = [det.update(v) for v in (0.05, 0.15, 0.3, 0.05, 0.3, 0.4, 0.5)]
    assert out == [False, False, False, False, False, True, False]
    assert det.detections == [5]
    assert det.last_level == "drift"


def test_calibrate_threshold_achieves_target_false_alarm_rate():
    """Calibrated on one stationary sample, checked on a fresh one (§13.5)."""
    rng = np.random.default_rng(7)
    scores = rng.beta(2, 5, 8000)
    null = psi_null_distribution(scores, sample_size=1000, n_draws=800, seed=1)
    target = 0.05
    level = calibrate_threshold(null[:400], target)
    observed = float(np.mean(null[400:] >= level))
    assert 0.5 * target <= observed <= 1.8 * target, observed

    det = ThresholdDetector(drift_level=level, warn_level=0.0)
    rate = len(_run(det, null[400:])) / 400
    assert rate == pytest.approx(observed)


def test_calibrate_threshold_refuses_too_few_values():
    with pytest.raises(ValueError):
        calibrate_threshold(np.arange(10.0), 0.01)


# --------------------------------------------------------------------------- #
# adaptation
# --------------------------------------------------------------------------- #


def _ctx(recent_scores: np.ndarray, threshold: float, budget: float) -> AdaptationContext:
    t = pd.Timestamp("2026-01-10")
    return AdaptationContext(
        now=t,
        next_window_start=t,
        labelled=pd.DataFrame(),
        recent_frame=pd.DataFrame(index=range(len(recent_scores))),
        recent_scores=recent_scores,
        model=None,  # type: ignore[arg-type]  # unused by a label-free policy
        threshold=threshold,
        model_factory=lambda: None,  # type: ignore[arg-type,return-value]
        alert_budget_rate=budget,
        label_col="is_fraud",
        time_col="timestamp",
    )


def test_recalibrate_threshold_restores_the_alert_budget_after_a_score_shift():
    rng = np.random.default_rng(8)
    budget = 0.01
    before = rng.normal(0.0, 1.0, 20_000)
    thr = alert_threshold_for_budget(before, budget)
    shifted_recent, shifted_next = rng.normal(0.8, 1.0, 20_000), rng.normal(0.8, 1.0, 20_000)
    assert np.mean(shifted_next >= thr) > 4 * budget  # static threshold blows the budget

    outcome = RecalibrateThreshold().adapt(_ctx(shifted_recent, thr, budget))
    assert outcome.model is None and outcome.threshold is not None
    assert np.mean(shifted_next >= outcome.threshold) == pytest.approx(budget, abs=0.003)


def test_static_policy_changes_nothing():
    outcome = StaticPolicy().adapt(_ctx(np.zeros(10), 0.5, 0.1))
    assert outcome.model is None and outcome.threshold is None and outcome.action == "none"


# --------------------------------------------------------------------------- #
# experiment: label discipline
# --------------------------------------------------------------------------- #


class _Recorder:
    """A Scorer that logs what it was fitted on and everything it scores.

    score = feature x (the label is a noisy threshold of x), so it has real
    signal; the point is the log, not the model.
    """

    def __init__(self, log: list[dict]):
        self.log = log

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> _Recorder:
        assert "is_fraud" in train.columns
        seen = pd.concat([train, val])
        self.fit_until = seen["timestamp"].max()
        self.train_ids = set(train["tx_id"])
        self.seen_ids = set(seen["tx_id"])
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        assert "is_fraud" not in df.columns, "a label column reached score()"
        self.log.append(
            {
                "fit_until": self.fit_until,
                "ids": set(df["tx_id"]),
                "unseen_min": df.loc[~df["tx_id"].isin(self.seen_ids), "timestamp"].min(),
                "scored_train_rows": bool(self.train_ids & set(df["tx_id"])),
            }
        )
        return df["x"].to_numpy(dtype=float)


def _toy_stream(days: int = 12, per_day: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * per_day
    t = pd.Timestamp("2026-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, days * 1440, n)), unit="min")
    drift = (np.arange(n) >= n // 2) * 1.0
    x = rng.normal(size=n) + drift
    y = (x + rng.normal(0, 0.5, n) > 2.2 + drift).astype("int8")
    return pd.DataFrame(
        {"tx_id": [f"T{i:06d}" for i in range(n)], "timestamp": t, "x": x, "is_fraud": y}
    )


@pytest.mark.parametrize("embargo", ["0min", "1D"])
@pytest.mark.parametrize("policy", [RetrainRecentWindow("3D", min_positives=1), RetrainAll(min_positives=1)])
def test_experiment_never_uses_window_labels_before_scoring_it(embargo, policy):
    """The central leakage guard of the drift experiment.

    The detector alarms on every window it sees, so the policy retrains as
    often as the protocol allows. Every score call must then be on rows that
    are strictly later than every label the model saw, except its own held-out
    validation rows (used, label-free, to set its threshold) -- and never on
    its own training rows.
    """
    stream = _toy_stream()
    log: list[dict] = []
    result = run_drift_experiment(
        stream,
        ["x"],
        "is_fraud",
        "timestamp",
        lambda: _Recorder(log),
        reference_end="2026-01-04",
        window="1D",
        detector=ThresholdDetector(drift_level=0.0, warn_level=0.0),
        policy=policy,
        alert_budget_rate=0.02,
        embargo=embargo,
    )
    retrains = [a for a in result.actions if a.train_end is not None]
    assert len(retrains) >= 3
    assert log
    for call in log:
        assert not call["scored_train_rows"]
        if pd.notna(call["unseen_min"]):
            assert call["fit_until"] < call["unseen_min"]
    gap = pd.Timedelta(embargo)
    for w in result.windows:
        assert pd.Timestamp(w.adaptive_train_end) < pd.Timestamp(w.start) - gap
    for a in retrains:
        assert pd.Timestamp(a.train_end) < pd.Timestamp(a.detected_at) - gap


def test_fit_chronological_refuses_rows_from_the_window_to_be_scored():
    stream = _toy_stream(days=3)
    with pytest.raises(LabelLeakError):
        fit_chronological(
            lambda: _Recorder([]),
            stream,
            time_col="timestamp",
            label_col="is_fraud",
            not_after=stream["timestamp"].iloc[-1],
        )


def test_static_policy_arms_are_identical():
    result = run_drift_experiment(
        _toy_stream(), ["x"], "is_fraud", "timestamp", lambda: _Recorder([]),
        reference_end="2026-01-04", window="1D",
        detector=ThresholdDetector(), policy=StaticPolicy(), alert_budget_rate=0.02,
    )
    for w in result.windows:
        assert w.static == w.adaptive


# --------------------------------------------------------------------------- #
# experiment: end to end on the synthetic generator
# --------------------------------------------------------------------------- #

FEATS = ["log_amount", "hour", "is_ach", "is_cross_bank"]


class _LogReg:
    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> _LogReg:
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=0)
        self.model.fit(train[FEATS], train["is_fraud"])
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[FEATS])[:, 1]


@pytest.fixture(scope="module")
def synthetic_stream():
    cfg = GeneratorConfig(
        seed=1, n_accounts=1500, n_days=14, prevalence=0.01, camouflage=0.3,
        drift_onset_day=8, drift_kind="amount_shift", drift_magnitude=4.0,
    )
    tx = generate(cfg).transactions
    # Row-local features only, so they carry no temporal leakage of their own.
    tx = tx.assign(
        log_amount=np.log1p(tx["amount"]),
        hour=tx["timestamp"].dt.hour.astype(float),
        is_ach=(tx["payment_format"] == "ACH").astype(float),
        is_cross_bank=tx["is_cross_bank"].astype(float),
    )
    return cfg, tx


@pytest.mark.parametrize(
    "detector", [ThresholdDetector(), PageHinkley(delta=0.01, threshold=0.2)], ids=["psi_threshold", "page_hinkley"]
)
def test_end_to_end_detects_injected_drift_after_onset(synthetic_stream, detector, tmp_path):
    cfg, tx = synthetic_stream
    result = run_drift_experiment(
        tx, FEATS, "is_fraud", "timestamp", _LogReg,
        reference_end=cfg.start + pd.Timedelta(days=5), window="1D",
        detector=detector, policy=RetrainRecentWindow("2D"), alert_budget_rate=0.01,
        drift_onset=cfg.drift_onset, name="test",
    )
    assert result.false_alarms_before_onset == 0
    assert result.detected_at is not None
    assert pd.Timestamp(result.detected_at) > cfg.drift_onset
    assert 0 < result.detection_latency_hours <= 48
    assert result.actions and result.actions[0].train_end is not None
    assert result.pr_auc_lost_static is not None
    # Static threshold overshoots the budget once normal amounts move; the
    # retrained arm is back near it.
    last = result.windows[-1]
    assert last.static.alert_rate > 2 * result.alert_budget_rate
    assert last.adaptive.alert_rate < 2 * result.alert_budget_rate

    json.dumps(dataclasses.asdict(result), allow_nan=False)
    out = plot_drift_experiment(result, tmp_path / "drift.png")
    assert out.exists() and out.stat().st_size > 10_000


def test_end_to_end_without_drift_raises_no_alarm(synthetic_stream):
    cfg, tx = synthetic_stream
    result = run_drift_experiment(
        tx[tx["timestamp"] < cfg.drift_onset], FEATS, "is_fraud", "timestamp", _LogReg,
        reference_end=cfg.start + pd.Timedelta(days=5), window="1D",
        detector=ThresholdDetector(), policy=RetrainRecentWindow("2D"), alert_budget_rate=0.01,
    )
    assert result.alarm_windows == []
    assert result.detected_at is None and result.drift_onset is None
