"""Streaming drift detectors over a monitored statistic (one value per window).

Implements Concept Mastery §13.4 (thresholds on PSI) and §13.5 (change points:
Page-Hinkley, the sequential mean-shift test that CUSUM belongs to, and ADWIN's
adaptive window).

A detector is characterised by two numbers, and both must be reported (§13.5):
the detection delay after a known change, and the false-alarm rate when nothing
changes. `calibrate_threshold` sets the second from stationary data before the
first is measured, so the threshold is never tuned on the drift it must detect.

Every detector has the same interface: `update(value) -> bool` (True on alarm),
`reset()` (forget the current regime, keep the alarm log), `detections` (every
alarm, as 0-based update indices), `detected_at` (the first one), `describe()`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np


class Detector(Protocol):
    detections: list[int]

    @property
    def detected_at(self) -> int | None: ...

    def update(self, value: float) -> bool: ...

    def reset(self) -> None: ...

    def describe(self) -> dict[str, Any]: ...


@dataclass
class _AlarmLog:
    """Shared bookkeeping: a global update counter and the alarm indices."""

    n_seen: int = field(default=0, init=False)
    detections: list[int] = field(default_factory=list, init=False)

    @property
    def detected_at(self) -> int | None:
        return self.detections[0] if self.detections else None

    def _tick(self, alarm: bool) -> bool:
        if alarm:
            self.detections.append(self.n_seen)
        self.n_seen += 1
        return alarm


@dataclass
class PageHinkley(_AlarmLog):
    """Page-Hinkley test for a sustained shift in the mean of a stream.

    m_t = sum_{i<=t} (x_i - mean_t - delta); alarm when m_t - min_{s<=t} m_s >
    threshold (upward shift; mirrored for downward). `delta` is the shift size
    tolerated as noise; `threshold` trades delay against false alarms. After an
    alarm the statistic restarts, so a later, second shift can also be caught.
    """

    delta: float = 0.005
    threshold: float = 50.0
    min_instances: int = 1
    direction: Literal["up", "down", "both"] = "up"
    _n: int = field(default=0, init=False, repr=False)
    _mean: float = field(default=0.0, init=False, repr=False)
    _m_up: float = field(default=0.0, init=False, repr=False)
    _min_up: float = field(default=0.0, init=False, repr=False)
    _m_down: float = field(default=0.0, init=False, repr=False)
    _max_down: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.direction not in ("up", "down", "both"):
            raise ValueError("direction must be 'up', 'down' or 'both'")

    @property
    def statistic(self) -> float:
        up = self._m_up - self._min_up
        down = self._max_down - self._m_down
        return {"up": up, "down": down, "both": max(up, down)}[self.direction]

    def update(self, value: float) -> bool:
        x = float(value)
        self._n += 1
        self._mean += (x - self._mean) / self._n
        self._m_up += x - self._mean - self.delta
        self._min_up = min(self._min_up, self._m_up)
        self._m_down += x - self._mean + self.delta
        self._max_down = max(self._max_down, self._m_down)
        alarm = self._n >= self.min_instances and self.statistic > self.threshold
        if alarm:
            self.reset()
        return self._tick(alarm)

    def reset(self) -> None:
        self._n, self._mean = 0, 0.0
        self._m_up = self._min_up = self._m_down = self._max_down = 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "detector": "PageHinkley",
            "delta": self.delta,
            "threshold": self.threshold,
            "min_instances": self.min_instances,
            "direction": self.direction,
        }


@dataclass
class ADWIN(_AlarmLog):
    """Adaptive windowing (Bifet & Gavalda, 2007), the plain O(n) version.

    Keeps a window of recent values; at each update it tests every split into
    an older part W0 and a newer part W1 and, if their means differ by more than

        eps = sqrt(2 var / m * ln(2/d)) + 2 / (3 m) * ln(2/d),
        m = 1 / (1/n0 + 1/n1),  d = delta / n,

    drops W0 and signals a change. No window length to choose, which is its
    appeal (§13.5); `min_sub_window` stops tiny halves from deciding anything.
    Reports changes in either direction.
    """

    delta: float = 0.002
    min_sub_window: int = 5
    max_window: int = 10_000
    _window: list[float] = field(default_factory=list, init=False, repr=False)

    @property
    def width(self) -> int:
        return len(self._window)

    def _cut_point(self) -> int | None:
        x = np.asarray(self._window)
        n = x.size
        k = self.min_sub_window
        if n < 2 * k:
            return None
        split = np.arange(k, n - k + 1)
        cs = np.cumsum(x)
        mu0 = cs[split - 1] / split
        mu1 = (cs[-1] - cs[split - 1]) / (n - split)
        m = 1.0 / (1.0 / split + 1.0 / (n - split))
        log_term = math.log(2.0 * n / self.delta)
        eps = np.sqrt(2.0 * x.var() * log_term / m) + 2.0 * log_term / (3.0 * m)
        hit = np.flatnonzero(np.abs(mu0 - mu1) > eps)
        return int(split[hit[0]]) if hit.size else None

    def update(self, value: float) -> bool:
        self._window.append(float(value))
        if len(self._window) > self.max_window:
            del self._window[0]
        changed = False
        while (cut := self._cut_point()) is not None:
            del self._window[:cut]
            changed = True
        return self._tick(changed)

    def reset(self) -> None:
        self._window.clear()

    def describe(self) -> dict[str, Any]:
        return {
            "detector": "ADWIN",
            "delta": self.delta,
            "min_sub_window": self.min_sub_window,
            "max_window": self.max_window,
        }


@dataclass
class ThresholdDetector(_AlarmLog):
    """Fixed levels on a statistic, PSI by default (§13.4.1).

    Alarm when the value reaches `drift_level` on `patience` consecutive
    updates. `last_level` keeps the ok / warn / drift reading for dashboards.
    The defaults are the credit-risk conventions; `calibrate_threshold` gives a
    level with a measured false-alarm rate instead.
    """

    drift_level: float = 0.25
    warn_level: float = 0.1
    patience: int = 1
    last_level: str = field(default="ok", init=False)
    _streak: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.warn_level > self.drift_level:
            raise ValueError("warn_level must not exceed drift_level")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")

    def update(self, value: float) -> bool:
        v = float(value)
        if v >= self.drift_level:
            self.last_level = "drift"
            self._streak += 1
        else:
            self.last_level = "warn" if v >= self.warn_level else "ok"
            self._streak = 0
        alarm = self._streak >= self.patience
        if alarm:
            self._streak = 0
        return self._tick(alarm)

    def reset(self) -> None:
        self._streak = 0
        self.last_level = "ok"

    def describe(self) -> dict[str, Any]:
        return {
            "detector": "ThresholdDetector",
            "drift_level": self.drift_level,
            "warn_level": self.warn_level,
            "patience": self.patience,
        }


def calibrate_threshold(
    stationary_values: Sequence[float] | np.ndarray, target_false_alarm_rate: float
) -> float:
    """Alarm level whose exceedance rate on stationary data is the target.

    `stationary_values` are the monitored statistic computed where no drift
    exists: per-window PSIs from the reference period, a no-drift synthetic run,
    or `monitors.psi_null_distribution`. The level is their (1 - rate) quantile,
    so a ThresholdDetector at this level alarms on roughly that share of
    stationary windows. Calibrate on stationary data only -- choosing the level
    on the drifted stream would tune the detector to the answer.
    """
    vals = np.asarray(stationary_values, dtype=float)
    vals = vals[~np.isnan(vals)]
    rate = float(target_false_alarm_rate)
    if not 0.0 < rate < 1.0:
        raise ValueError("target_false_alarm_rate must be in (0, 1)")
    if vals.size < math.ceil(1.0 / rate):
        raise ValueError(
            f"{vals.size} stationary values cannot resolve a {rate:g} false-alarm "
            f"rate; need at least {math.ceil(1.0 / rate)}"
        )
    return float(np.quantile(vals, 1.0 - rate))
