"""Feature correctness on hand-computed fixtures (CLAUDE.md testing priority 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nexis.features.behavioural import behavioural_features
from nexis.features.tabular import tabular_features
from nexis.features.velocity import rolling_velocity


def _tx(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "src": [r[0] for r in rows],
            "dst": [r[1] for r in rows],
            "timestamp": pd.to_datetime([r[2] for r in rows]),
            "amount": [r[3] for r in rows],
        }
    )


def test_pass_through_is_visible_from_the_sender_side():
    """B pays A 1000 at 10:00; A forwards 950 to C at 10:20."""
    df = _tx(
        [
            ("B", "A", "2026-01-01 10:00", 1000.0),
            ("A", "C", "2026-01-01 10:20", 950.0),
        ]
    )
    f = behavioural_features(df, windows=("1h", "24h"))
    a = f.iloc[1]
    assert a["src_in_cnt_1h"] == 1
    assert a["src_in_sum_1h"] == 1000.0
    assert a["src_secs_since_in"] == 20 * 60
    assert a["amt_over_src_in_24h"] == pytest.approx(950.0 / 1001.0)
    assert np.isnan(a["src_secs_since_out"]), "A never sent before"


def test_fan_in_counts_distinct_senders_strictly_before():
    df = _tx(
        [
            ("S1", "M", "2026-01-01 09:00", 100.0),
            ("S2", "M", "2026-01-01 09:10", 100.0),
            ("S1", "M", "2026-01-01 09:20", 100.0),
            ("S3", "M", "2026-01-01 09:30", 100.0),
        ]
    )
    f = behavioural_features(df, windows=("1h",))
    assert f["dst_in_cnt_1h"].tolist() == [0, 1, 2, 3]
    assert f["dst_n_counterparties"].tolist() == [0, 1, 2, 2]
    assert f["counterparty_is_new"].tolist() == [1, 1, 0, 1]


def test_window_is_closed_left():
    """An event exactly w before t is inside [t - w, t); one second older is not."""
    df = _tx(
        [
            ("A", "X", "2026-01-01 09:59:59", 1.0),
            ("A", "X", "2026-01-01 10:00:00", 2.0),
            ("A", "X", "2026-01-01 11:00:00", 4.0),
        ]
    )
    f = behavioural_features(df, windows=("1h",))
    assert f["src_out_cnt_1h"].iloc[2] == 1
    assert f["src_out_sum_1h"].iloc[2] == 2.0


def test_matches_reference_velocity_when_timestamps_are_unique():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame(
        {
            "src": rng.choice(list("ABCDEFGH"), n),
            "dst": rng.choice(list("STUVWXYZ"), n),
            "timestamp": pd.Timestamp("2026-01-01")
            + pd.to_timedelta(np.sort(rng.choice(10**6, n, replace=False)), unit="s"),
            "amount": rng.lognormal(5, 1, n).round(2),
        }
    )
    ours = behavioural_features(df, windows=("1h", "24h"))
    ref = rolling_velocity(df, windows=("1h", "24h"))
    for w in ("1h", "24h"):
        np.testing.assert_array_equal(ours[f"src_out_cnt_{w}"], ref[f"cnt_{w}"])
        np.testing.assert_allclose(ours[f"src_out_sum_{w}"], ref[f"sum_{w}"], rtol=1e-5)


def test_features_do_not_depend_on_input_row_order():
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame(
        {
            "src": rng.choice(list("ABCDE"), n),
            "dst": rng.choice(list("VWXYZ"), n),
            "timestamp": pd.Timestamp("2026-01-01")
            + pd.to_timedelta(rng.integers(0, 3000, n) * 60, unit="s"),
            "amount": rng.lognormal(5, 1, n).round(2),
        }
    )
    a = behavioural_features(df)
    shuffled = df.sample(frac=1.0, random_state=3)
    b = behavioural_features(shuffled).loc[df.index]
    pd.testing.assert_frame_equal(a, b)


def test_tabular_features_hand_computed():
    df = pd.DataFrame(
        {
            "amount": [99.0, 1500.0],
            "amount_paid": [99.0, 1500.0],
            "timestamp": pd.to_datetime(["2022-09-01 03:15", "2022-09-02 23:59"]),
            "payment_format": pd.Categorical(["ACH", "Wire"]),
            "pay_currency": pd.Categorical(["Euro", "US Dollar"]),
            "recv_currency": pd.Categorical(["Euro", "Euro"]),
            "is_cross_currency": [0, 1],
            "is_cross_bank": [1, 0],
            "is_self_transfer": [0, 0],
        }
    )
    f = tabular_features(df)
    assert f["log_amount"].tolist() == pytest.approx([np.log1p(99.0), np.log1p(1500.0)])
    assert f["hour"].tolist() == [3, 23]
    assert f["is_round_amount"].tolist() == [0, 1]
    assert "day_of_week" not in f.columns, "a 10-day dataset turns weekday into a clock"
