"""Leakage regression tests.

Every test here corresponds to a specific bug that produces inflated results with
no visible symptom. These are the most important tests in the repository.

When you add feature-engineering or graph-construction code, add a test here in
the same change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nexis.evaluation.splits import (
    assert_no_id_overlap,
    assert_temporal_integrity,
    temporal_split,
)
from nexis.features.velocity import (
    personal_baseline_features,
    rolling_velocity,
)


@pytest.fixture
def transactions() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 4000
    return pd.DataFrame(
        {
            "tx_id": [f"TX{i:06d}" for i in range(n)],
            "src": rng.choice([f"ACC_{i:03d}" for i in range(50)], n),
            "dst": rng.choice([f"ACC_{i:03d}" for i in range(50)], n),
            "amount": rng.lognormal(7.0, 1.0, n).round(2),
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="15min"),
            "is_fraud": rng.random(n) < 0.02,
        }
    )


def test_temporal_split_is_ordered_and_embargoed(transactions):
    split = temporal_split(transactions, embargo="1D")
    assert_temporal_integrity(split)
    assert_no_id_overlap(split.train, split.test)
    assert len(split.train) > len(split.test)


def test_temporal_split_rejects_impossible_fractions(transactions):
    with pytest.raises(ValueError):
        temporal_split(transactions, train_frac=0.8, val_frac=0.3)


def test_rolling_velocity_excludes_current_row():
    """closed='left' guard. Without it the feature includes the event it scores."""
    df = pd.DataFrame(
        {
            "src": ["A"] * 3,
            "dst": ["B", "C", "D"],
            "timestamp": pd.to_datetime(
                ["2026-01-01 10:00", "2026-01-01 10:30", "2026-01-01 10:45"]
            ),
            "amount": [100.0, 200.0, 300.0],
        }
    )
    out = rolling_velocity(df, windows=("1h",))
    assert out["cnt_1h"].iloc[0] == 0, "first event must see no history"
    assert out["cnt_1h"].iloc[1] == 1
    assert out["cnt_1h"].iloc[2] == 2, "must exclude the current row"
    assert out["sum_1h"].iloc[2] == 300.0, "100 + 200, not 600"


def test_personal_baseline_is_shifted():
    """shift(1) guard. The scored event must not be inside its own baseline."""
    df = pd.DataFrame(
        {
            "src": ["A"] * 6,
            "dst": list("BCDEFG"),
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="h"),
            "amount": [100.0] * 5 + [10_000.0],
        }
    )
    out = personal_baseline_features(df, min_history=3)
    # The huge final amount must appear extreme relative to a baseline of 100s.
    # If it leaked into its own mean/std, the z-score would be small.
    assert out["amt_z_personal"].iloc[-1] > 10, "large event was in its own baseline"
    assert np.isnan(out["amt_z_personal"].iloc[0]), "no history should be NaN"


def test_personal_baseline_flags_new_counterparties():
    df = pd.DataFrame(
        {
            "src": ["A", "A", "A"],
            "dst": ["B", "B", "C"],
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
            "amount": [100.0, 100.0, 100.0],
        }
    )
    out = personal_baseline_features(df, min_history=1)
    assert out["counterparty_is_new"].tolist() == [1.0, 0.0, 1.0]


def test_embargo_actually_removes_rows(transactions):
    """A larger embargo must drop more rows. If not, it is not being applied."""
    small = temporal_split(transactions, embargo="1h")
    large = temporal_split(transactions, embargo="3D")
    total_small = len(small.train) + len(small.val) + len(small.test)
    total_large = len(large.train) + len(large.val) + len(large.test)
    assert total_large < total_small


@pytest.mark.slow
def test_leakage_canary_scores_at_chance(transactions):
    """Shuffled training labels must not predict the future.

    This is the strongest test in the suite. Wire your real pipeline into
    fit_predict once it exists; the stub below documents the interface.
    """
    from nexis.evaluation.leakage import leakage_canary

    def fit_predict(df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        # Replace with the real pipeline. A constant predictor is leak-free
        # by construction and establishes the expected chance-level behaviour.
        return np.full(len(df), 0.5)

    result = leakage_canary(
        fit_predict,
        transactions,
        transactions["is_fraud"].to_numpy().astype(int),
    )
    assert result["ratio"] < 3.0, f"possible leak: {result}"
