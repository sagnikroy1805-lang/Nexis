"""Velocity, inter-arrival and personal-baseline features.

Implements Concept Mastery Module 15.

Highest value-per-hour in the project. These features are cheap, interpretable,
and make the tabular baseline genuinely hard to beat -- which is exactly what
makes beating it a real result.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_WINDOWS: tuple[str, ...] = ("5min", "1h", "24h", "7D")


def rolling_velocity(
    df: pd.DataFrame,
    key: str = "src",
    time_col: str = "timestamp",
    amount_col: str = "amount",
    windows: Sequence[str] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Per-account rolling counts, sums and maxima over several time windows.

    LEAKAGE GUARD: closed='left' excludes the current row. Without it the 24h
    count for the transaction being scored includes that transaction, so the
    feature distribution differs between training and inference in a way that
    is invisible in offline metrics. Do not remove it.

    Returns:
        A frame aligned to df's index with one column per (statistic, window).
    """
    df = df.sort_values(time_col).copy()
    out = pd.DataFrame(index=df.index)
    indexed = df.set_index(time_col)
    grouped = indexed.groupby(key)[amount_col]

    for w in windows:
        cnt = grouped.rolling(w, closed="left").count()
        tot = grouped.rolling(w, closed="left").sum()
        mx = grouped.rolling(w, closed="left").max()
        out[f"cnt_{w}"] = cnt.reset_index(level=0, drop=True).to_numpy()
        out[f"sum_{w}"] = tot.reset_index(level=0, drop=True).to_numpy()
        out[f"max_{w}"] = mx.reset_index(level=0, drop=True).to_numpy()

    # Ratios between windows are more informative than raw counts: they are
    # self-normalising, so they compare across accounts of very different sizes.
    if "1h" in windows and "7D" in windows:
        baseline_hourly = out["cnt_7D"] / 168.0
        out["burst_1h"] = out["cnt_1h"] / (baseline_hourly + 1e-6)

    return out.fillna(0.0)


def interarrival_features(times: pd.Series) -> dict[str, float]:
    """Inter-arrival statistics for one account's event sequence.

    For a Poisson process the coefficient of variation is exactly 1. Departures
    are informative:
        CV ~ 1   memoryless, human-like
        CV >> 1  bursty -- dormant-to-burst signature
        CV ~ 0   near-constant intervals -- automation signature

    Combined with hour-of-day entropy this separates scripted operation from
    human behaviour remarkably well, using no learned model at all.
    """
    t = np.sort(pd.to_datetime(times).astype("int64").to_numpy() // 10**9)
    if len(t) < 3:
        return {
            "iat_cv": float("nan"),
            "iat_median": float("nan"),
            "iat_min": float("nan"),
            "hour_entropy": float("nan"),
            "automation_score": 0.0,
        }

    dt = np.diff(t).astype(float)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return {
            "iat_cv": 0.0,
            "iat_median": 0.0,
            "iat_min": 0.0,
            "hour_entropy": 0.0,
            "automation_score": 1.0,
        }

    cv = float(dt.std() / max(dt.mean(), 1e-9))
    hours = pd.to_datetime(t, unit="s").hour
    p = np.bincount(hours, minlength=24) / len(hours)
    p = p[p > 0]
    entropy = float(-(p * np.log(p)).sum())

    return {
        "iat_cv": cv,
        "iat_median": float(np.median(dt)),
        "iat_min": float(dt.min()),
        "hour_entropy": entropy,
        # high when intervals are regular AND activity is concentrated in
        # few hours -- both hallmarks of a script rather than a person
        "automation_score": float(np.exp(-cv) * np.exp(-entropy / 3.0)),
    }


def personal_baseline_features(
    df: pd.DataFrame,
    key: str = "src",
    counterparty: str = "dst",
    time_col: str = "timestamp",
    amount_col: str = "amount",
    min_history: int = 5,
) -> pd.DataFrame:
    """Compare each transaction to the account's OWN history, not the population.

    A Rs 8,000 transfer is p40 in the population and a 12-sigma event for a
    student account whose largest previous transfer was Rs 900. Population
    z-scores miss exactly the cases that matter.

    LEAKAGE GUARD: .shift(1) excludes the current row from its own baseline.
    A one-character omission here inflates results with no visible symptom.

    Accounts with fewer than min_history prior events yield NaN. Do not fill
    with zero -- 'no history' is itself a strong signal (young accounts are
    disproportionately mules) and deserves its own indicator column.
    """
    df = df.sort_values(time_col).copy()
    grouped = df.groupby(key)[amount_col]

    hist_mean = grouped.transform(
        lambda s: s.shift(1).expanding(min_history).mean()
    )
    hist_std = grouped.transform(lambda s: s.shift(1).expanding(min_history).std())
    hist_max = grouped.transform(lambda s: s.shift(1).expanding(min_history).max())
    hist_med = grouped.transform(
        lambda s: s.shift(1).expanding(min_history).median()
    )

    out = pd.DataFrame(index=df.index)
    out["amt_z_personal"] = (df[amount_col] - hist_mean) / (hist_std + 1e-6)
    out["amt_vs_personal_max"] = df[amount_col] / (hist_max + 1e-6)
    out["amt_log_ratio_median"] = np.log1p(df[amount_col]) - np.log1p(hist_med)
    out["has_history"] = hist_mean.notna().astype(float)

    seen: dict[str, set] = {}
    is_new = np.zeros(len(df), dtype=float)
    for i, (src, dst) in enumerate(
        zip(df[key].to_numpy(), df[counterparty].to_numpy(), strict=True)
    ):
        bucket = seen.setdefault(src, set())
        is_new[i] = float(dst not in bucket)
        bucket.add(dst)
    out["counterparty_is_new"] = is_new

    return out


def concentration_hhi(amounts_by_counterparty: pd.Series) -> float:
    """Herfindahl index of outflow concentration.

    1.0 = all money to a single counterparty (perfect pass-through, mule
    signature). Near 0 = spread evenly across many (fan-out / smurfing).
    A single scalar capturing the SHAPE of the outflow, complementing degree
    which captures only its size.
    """
    total = amounts_by_counterparty.sum()
    if total <= 0:
        return 0.0
    shares = amounts_by_counterparty / total
    return float((shares**2).sum())
