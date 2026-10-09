"""Time-aware data splitting.

Implements Concept Mastery §3.6 and §9.4.

The single most common way a fraud-detection project produces fake results is a
random train/test split on temporal data. Every function here exists to make that
mistake structurally difficult.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd

SplitOn = Literal["time", "rows"]


@dataclass(frozen=True)
class SplitResult:
    """Three chronological folds plus the boundaries used to produce them."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    embargo: pd.Timedelta

    def summary(self) -> dict:
        return {
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test),
            "train_end": str(self.train_end),
            "val_end": str(self.val_end),
            "embargo": str(self.embargo),
        }


@dataclass(frozen=True)
class SplitConfig:
    """Split settings as loaded from configs/*.yaml.

    Kept as a dataclass so an experiment manifest records exactly how the data
    was divided, and so the embargo can be checked against the feature windows
    at load time rather than discovered after a run.
    """

    train_frac: float = 0.6
    val_frac: float = 0.2
    embargo: str = "1D"
    split_on: SplitOn = "time"

    def apply(self, df: pd.DataFrame, time_col: str = "timestamp") -> SplitResult:
        return temporal_split(
            df,
            time_col=time_col,
            train_frac=self.train_frac,
            val_frac=self.val_frac,
            embargo=self.embargo,
            split_on=self.split_on,
        )


def temporal_split(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    embargo: str = "1D",
    split_on: SplitOn = "time",
) -> SplitResult:
    """Split chronologically with an embargo gap between folds.

    The embargo is not optional padding. Rolling-window features look backwards,
    so a transaction one hour after a split boundary has features computed partly
    from the previous fold's data. The gap must be at least as long as the longest
    feature window in use, or that bleed becomes leakage.

    Args:
        df: transactions, one row per event.
        time_col: name of the timestamp column.
        train_frac: fraction used for training -- of the time span, or of the
            rows, depending on split_on.
        val_frac: fraction used for validation, in the same unit.
        embargo: pandas offset string; rows inside the gap are dropped entirely.
        split_on: "time" places boundaries at fractions of the time span; "rows"
            places them at row-count quantiles, still in time order. Use "rows"
            when volume is very uneven over time (IBM AML: day 1 alone holds 22%
            of the data), where "time" yields folds of wildly different sizes.

    Returns:
        SplitResult with strictly ordered, non-overlapping folds.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1:
        raise ValueError("train_frac and val_frac must be in (0, 1)")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must leave room for a test fold")

    # Stable sort: rows sharing a timestamp keep their input order, so the split
    # is a deterministic function of the input.
    df = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    t = pd.to_datetime(df[time_col])

    if split_on == "time":
        t0, t1 = t.min(), t.max()
        span = t1 - t0
        train_end = t0 + span * train_frac
        val_end = t0 + span * (train_frac + val_frac)
    elif split_on == "rows":
        # Boundaries are still timestamps, read off at row-count quantiles. Every
        # row sharing the boundary timestamp satisfies t <= boundary and lands on
        # the earlier side, so a timestamp tie can never straddle two folds.
        n = len(df)
        train_end = t.iloc[max(int(n * train_frac) - 1, 0)]
        val_end = t.iloc[max(int(n * (train_frac + val_frac)) - 1, 0)]
    else:
        raise ValueError(f"split_on must be 'time' or 'rows', got {split_on!r}")
    gap = pd.Timedelta(embargo)

    train = df[t <= train_end]
    val = df[(t > train_end + gap) & (t <= val_end)]
    test = df[t > val_end + gap]

    if len(val) == 0 or len(test) == 0:
        raise ValueError(
            f"embargo {embargo} left an empty fold; reduce it or widen the span"
        )

    return SplitResult(train, val, test, train_end, val_end, gap)


def rolling_origin_splits(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    n_folds: int = 5,
    horizon: str = "7D",
    min_history_frac: float = 0.4,
    embargo: str = "1D",
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Walk-forward cross-validation: the temporal analogue of k-fold.

    fold k trains on everything before cut_k and tests on [cut_k, cut_k + horizon).
    The training window grows; the test window always lies strictly in the future.

    Yields:
        (fold_index, train_df, test_df)
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    t = pd.to_datetime(df[time_col])
    h = pd.Timedelta(horizon)
    gap = pd.Timedelta(embargo)
    start = t.min() + (t.max() - t.min()) * min_history_frac

    for k in range(n_folds):
        cut = start + k * h
        train = df[t < cut - gap]
        test = df[(t >= cut) & (t < cut + h)]
        if len(test) == 0 or len(train) == 0:
            continue
        yield k, train, test


def assert_temporal_integrity(
    split: SplitResult, time_col: str = "timestamp"
) -> None:
    """Assert the folds are ordered, non-overlapping and properly embargoed.

    Call this immediately after every split. It is cheap and it converts a silent
    correctness failure into a loud one.
    """
    tr, va, te = split.train, split.val, split.test
    tr_t = pd.to_datetime(tr[time_col])
    va_t = pd.to_datetime(va[time_col])
    te_t = pd.to_datetime(te[time_col])

    assert tr_t.max() < va_t.min(), "train/val overlap in time"
    assert va_t.max() < te_t.min(), "val/test overlap in time"
    assert va_t.min() - tr_t.max() >= split.embargo, "train/val embargo too small"
    assert te_t.min() - va_t.max() >= split.embargo, "val/test embargo too small"


def assert_embargo_covers_windows(embargo: str, windows: Sequence[str]) -> None:
    """Fail if any feature window is longer than the embargo between folds.

    PROJECT_RULES.md rule 1 states this as a requirement; this function enforces it. A
    24h embargo with a 7D velocity window lets the first six days of each fold
    carry features computed from the previous fold, and nothing in the metrics
    shows it.

    Raises:
        ValueError naming the offending windows.
    """
    gap = pd.Timedelta(embargo)
    too_long = [w for w in windows if pd.Timedelta(w) > gap]
    if too_long:
        raise ValueError(
            f"feature windows {too_long} exceed the embargo {embargo}; shorten "
            "the windows or widen the embargo"
        )


def assert_no_id_overlap(
    train: pd.DataFrame, test: pd.DataFrame, id_col: str = "tx_id"
) -> None:
    """Assert no record appears in both folds.

    Duplicates arise from retries, reversals and re-ingestion. They leak directly.
    """
    overlap = set(train[id_col]) & set(test[id_col])
    assert not overlap, f"{len(overlap)} ids appear in both folds, e.g. {list(overlap)[:5]}"
