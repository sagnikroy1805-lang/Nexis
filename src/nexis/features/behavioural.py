"""Vectorised behavioural features for millions of transactions.

Implements Concept Mastery Module 15 (§15.1 velocity, §15.2 inter-arrival, §15.3
rolling statistics, §15.4 the personal baseline) and the rolling-window rules of
§9.2 / §9.4.

Why a second implementation next to features/velocity.py: the reference version
applies a Python lambda per account, which does not finish on 5M rows, and its
expanding baseline uses row order, so two transactions in the same minute see
each other. Here every statistic is computed with binary search over events
sorted by (account, time).

LEAKAGE GUARD -- strict past. Every feature for a transaction at time t uses only
events with timestamp < t. Events in the same timestamp are treated as
simultaneous and are invisible to each other, the current row included. This is
the closed='left' rule applied consistently to windows AND expanding baselines.
The guard lives in EventIndex.before(); tests/test_leakage.py pins it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

_TIME_BITS = 40
# Shift every timestamp up so t - window never goes negative inside the packed
# (account, time) key. 2**33 s is ~272 years: far larger than any window.
_OFFSET = 1 << 33


def _seconds(ts: pd.Series) -> np.ndarray:
    """Seconds since the earliest timestamp, independent of datetime resolution."""
    return ((ts - ts.min()) // pd.Timedelta("1s")).to_numpy(dtype=np.int64) + _OFFSET


@dataclass
class EventIndex:
    """Events of one kind (e.g. outgoing payments) sorted by (account, time).

    Supports, for any query (account, t):
      - aggregates over [t - w, t)   (rolling windows, closed left)
      - aggregates over (-inf, t)    (expanding personal baseline, strict past)
      - the time of the latest event strictly before t
    """

    key: np.ndarray
    t: np.ndarray
    csum: np.ndarray
    csum_log: np.ndarray
    csum_log2: np.ndarray
    cummax: np.ndarray

    @classmethod
    def build(cls, acct: np.ndarray, t: np.ndarray, amount: np.ndarray) -> EventIndex:
        order = np.lexsort((t, acct))
        acct, t, amount = acct[order], t[order], amount[order].astype(np.float64)
        log_amt = np.log1p(amount)
        cummax = pd.Series(amount).groupby(acct).cummax().to_numpy()
        zero = np.zeros(1)
        return cls(
            key=(acct.astype(np.int64) << _TIME_BITS) | t,
            t=t,
            csum=np.concatenate([zero, np.cumsum(amount)]),
            csum_log=np.concatenate([zero, np.cumsum(log_amt)]),
            csum_log2=np.concatenate([zero, np.cumsum(log_amt**2)]),
            cummax=cummax,
        )

    def before(self, acct: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Index of the first event of `acct` at or after t.

        LEAKAGE GUARD: side='left' is what makes this strict. Events at exactly t
        (the current transaction, and anything in the same minute) sort at or
        after the returned index, so every aggregate over [start, before) excludes
        them. Changing it to side='right' would leak the current row.
        """
        return np.searchsorted(self.key, (acct.astype(np.int64) << _TIME_BITS) | t, "left")

    def group_start(self, acct: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.key, acct.astype(np.int64) << _TIME_BITS, "left")


def _window(
    idx: EventIndex, acct: np.ndarray, t: np.ndarray, seconds: int
) -> tuple[np.ndarray, np.ndarray]:
    """(count, sum) of an account's events in [t - seconds, t)."""
    hi = idx.before(acct, t)
    lo = idx.before(acct, t - seconds)
    return (hi - lo).astype(np.float32), (idx.csum[hi] - idx.csum[lo]).astype(np.float32)


def _history(
    idx: EventIndex, acct: np.ndarray, t: np.ndarray
) -> dict[str, np.ndarray]:
    """Expanding statistics over all of an account's events strictly before t."""
    hi = idx.before(acct, t)
    g0 = idx.group_start(acct)
    n = (hi - g0).astype(np.float64)
    has = n > 0
    safe_n = np.where(has, n, 1.0)
    mean_log = (idx.csum_log[hi] - idx.csum_log[g0]) / safe_n
    var_log = (idx.csum_log2[hi] - idx.csum_log2[g0]) / safe_n - mean_log**2
    last = np.where(has, hi - 1, 0)
    return {
        "n": n,
        "has": has,
        "mean_log": np.where(has, mean_log, np.nan),
        "std_log": np.where(n > 1, np.sqrt(np.clip(var_log, 0.0, None)), np.nan),
        "max": np.where(has, idx.cummax[last], np.nan),
        "last_t": np.where(has, idx.t[last], -1),
    }


def window_seconds(window: str) -> int:
    return int(pd.Timedelta(window) // pd.Timedelta("1s"))


def behavioural_features(
    df: pd.DataFrame,
    windows: Sequence[str] = ("5min", "1h", "24h"),
    min_history: int = 3,
) -> pd.DataFrame:
    """Sender- and receiver-side behaviour for every transaction, strict past only.

    Four event streams, because laundering shows up in how money ARRIVES as much
    as in how it leaves:
        src_out  what the sender has been sending        (velocity, bursts)
        src_in   what the sender has just received       (pass-through / mule)
        dst_in   what the receiver has been receiving    (fan-in)
        dst_out  what the receiver has been sending on   (onward layering)

    Personal baseline (§15.4): the amount is compared to the SENDER's own history
    in log space -- the account's past is the reference, not the population.

    Args:
        df: standard schema (src, dst, amount, timestamp). Need not be sorted.
        windows: rolling windows; must not exceed the split embargo (rule 1).
        min_history: below this many prior sends, z-scores are NaN, not 0 --
            'no history' is a signal of its own and has its own column.

    Returns:
        float32 frame aligned to df.index.
    """
    codes, _ = pd.factorize(pd.concat([df["src"], df["dst"]], ignore_index=True))
    src = codes[: len(df)].astype(np.int64)
    dst = codes[len(df) :].astype(np.int64)
    t = _seconds(df["timestamp"])
    amount = df["amount"].to_numpy(dtype=np.float64)
    log_amt = np.log1p(amount)

    out_idx = EventIndex.build(src, t, amount)  # events keyed by the payer
    in_idx = EventIndex.build(dst, t, amount)  # events keyed by the payee

    f: dict[str, np.ndarray] = {}
    for w in windows:
        s = window_seconds(w)
        f[f"src_out_cnt_{w}"], f[f"src_out_sum_{w}"] = _window(out_idx, src, t, s)
        f[f"src_in_cnt_{w}"], f[f"src_in_sum_{w}"] = _window(in_idx, src, t, s)
        f[f"dst_in_cnt_{w}"], f[f"dst_in_sum_{w}"] = _window(in_idx, dst, t, s)
        f[f"dst_out_cnt_{w}"], f[f"dst_out_sum_{w}"] = _window(out_idx, dst, t, s)

    longest = max(windows, key=window_seconds)
    # Share of what the sender just received that this payment sends on.
    # Near 1 with a short receive-to-send gap is the pass-through signature.
    f[f"amt_over_src_in_{longest}"] = (
        amount / (f[f"src_in_sum_{longest}"].astype(np.float64) + 1.0)
    ).astype(np.float32)

    hist = _history(out_idx, src, t)
    enough = hist["n"] >= min_history
    f["src_out_n_hist"] = hist["n"].astype(np.float32)
    f["src_has_history"] = enough.astype(np.float32)
    f["amt_z_src"] = np.where(
        enough, (log_amt - hist["mean_log"]) / (hist["std_log"] + 1e-3), np.nan
    ).astype(np.float32)
    f["amt_log_ratio_src_mean"] = np.where(
        hist["has"], log_amt - hist["mean_log"], np.nan
    ).astype(np.float32)
    f["amt_vs_src_max"] = np.where(hist["has"], amount / (hist["max"] + 1e-6), np.nan).astype(
        np.float32
    )
    f["src_secs_since_out"] = np.where(hist["has"], t - hist["last_t"], np.nan).astype(
        np.float32
    )

    src_in_hist = _history(in_idx, src, t)
    f["src_in_n_hist"] = src_in_hist["n"].astype(np.float32)
    # Receive-to-send latency: mules forward money soon after it lands.
    f["src_secs_since_in"] = np.where(
        src_in_hist["has"], t - src_in_hist["last_t"], np.nan
    ).astype(np.float32)
    f["dst_in_n_hist"] = _history(in_idx, dst, t)["n"].astype(np.float32)
    f["dst_out_n_hist"] = _history(out_idx, dst, t)["n"].astype(np.float32)

    # Counterparty novelty, strict past: a pair is "new" at t if it never
    # transacted at an earlier timestamp. Distinct-counterparty counts come from
    # the stream of each pair's first appearance.
    pair = pd.DataFrame({"src": src, "dst": dst, "t": t})
    first_t = pair.groupby(["src", "dst"])["t"].transform("min").to_numpy()
    f["counterparty_is_new"] = (first_t == t).astype(np.float32)
    firsts = pair.loc[first_t == t].drop_duplicates(["src", "dst"])
    fan_out_idx = EventIndex.build(
        firsts["src"].to_numpy(), firsts["t"].to_numpy(), np.ones(len(firsts))
    )
    fan_in_idx = EventIndex.build(
        firsts["dst"].to_numpy(), firsts["t"].to_numpy(), np.ones(len(firsts))
    )
    f["src_n_counterparties"] = _history(fan_out_idx, src, t)["n"].astype(np.float32)
    f["dst_n_counterparties"] = _history(fan_in_idx, dst, t)["n"].astype(np.float32)

    return pd.DataFrame(f, index=df.index)
