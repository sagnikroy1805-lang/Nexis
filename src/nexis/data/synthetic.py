"""Tier-3 synthetic transaction generator: an instrument for controlled experiments.

Implements Concept Mastery Module 14:

- §14.1 role. Validating only on data we generated is circular (we inject the
  patterns, engineer features for them, then detect them). This generator exists
  for the five knobs no public dataset exposes -- ring size, prevalence,
  camouflage, hop delay, drift onset -- and never as the sole evidence.
- §14.2 normal behaviour: per-account (personal-baseline) lognormal amounts,
  heavy-tailed Poisson activity with a daily cycle and weekend dip, a small set
  of repeat payees chosen by popularity (preferential attachment), IBM payment
  formats and currencies, cross-bank and self-transfers.
- §14.3 injection of the eight IBM AML typologies, hop by hop, so a pattern
  unfolds over time and early detection is measurable.
- §14.4 camouflage in [0, 1], which blends pattern payments into the members'
  own behaviour (amount, payment format, timing).
- §14.5 stress testing: knobs are held independent (see "Random streams").
- §14.6 reproducibility: fully determined by the config; `save_synthetic` writes
  a manifest with content hashes.

It also injects the controlled drift used by `nexis.drift.experiment` (§13.1
data drift, §13.2 concept drift): one shift of a known kind at a known instant.

Output schema. `transactions` is exactly the IBM adapter's standard schema
(`nexis.data.ibm_aml.REQUIRED_COLUMNS + EXTRA_COLUMNS`, same dtypes), so every
feature, graph and model in the repository runs on it unchanged. `patterns` has
the shape of `ibm_aml.match_patterns` output. Both are label-derived ground
truth: for evaluation only, never joined in as features.

Label semantics (CLAUDE.md rule 5): `is_fraud` = 1 means "this row was written by
the pattern injector". It is a simulator label, not a finding about any account.

Random streams. Every random draw comes from a stream keyed by
(seed, purpose, day[, pattern, aspect]) rather than one shared generator. Why:

- Drift is exactly testable. Days before `drift_onset_day` are drawn from the
  same streams whether or not drift is configured, and drift only changes
  post-onset draws (velocity, typology mix) or is a post-transform keyed on time
  (amounts). So the pre-onset data of a drift run is byte-identical to a
  no-drift run with the same seed.
- Sweeps are paired (§14.5). The normal background does not depend on any
  injection knob, and pattern j of day d has the same typology, members and
  structure at every camouflage level, so a camouflage sweep compares the same
  rings blended to different degrees, not different rings.

Usage:
    python -m nexis.data.synthetic --config configs/synthetic.yaml --out data/synthetic
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from nexis.data.ibm_aml import (
    EXTRA_COLUMNS,
    REQUIRED_COLUMNS,
    DataValidationError,
    file_sha256,
    validate,
)

GENERATOR_VERSION = "1.0.0"
TX_PREFIX = "SYN"

IBM_TYPOLOGIES: tuple[str, ...] = (
    "FAN-IN",
    "FAN-OUT",
    "CYCLE",
    "SCATTER-GATHER",
    "GATHER-SCATTER",
    "STACK",
    "BIPARTITE",
    "RANDOM",
)
DRIFT_KINDS: tuple[str, ...] = ("amount_shift", "velocity_shift", "new_typology")

# Categories are fixed by the module, not inferred from the rows, so a frame's
# dtypes do not depend on which values happened to be drawn. That is what lets a
# drift run's pre-onset rows compare equal to a no-drift run, dtypes included.
PAYMENT_FORMATS: tuple[str, ...] = (
    "ACH",
    "Bitcoin",
    "Cash",
    "Cheque",
    "Credit Card",
    "Reinvestment",
    "Wire",
)
# USD per unit, as estimated from IBM HI-Small day 1 (data/processed manifest).
# Only plausibility matters here; the generator draws amounts in USD.
USD_PER_UNIT: dict[str, float] = {
    "Bitcoin": 11875.0,
    "Euro": 1.1718,
    "Rupee": 0.013616,
    "UK Pound": 1.2917,
    "US Dollar": 1.0,
    "Yuan": 0.14931,
}
CURRENCIES: tuple[str, ...] = tuple(sorted(USD_PER_UNIT))
_DECIMALS = {c: (6 if c == "Bitcoin" else 2) for c in CURRENCIES}
# Mostly USD, so cross-currency payments stay rare (a few percent; IBM HI-Small
# has 1.4%) rather than a large, trivially-separable block.
ACCOUNT_CURRENCY_MIX: dict[str, float] = {
    "US Dollar": 0.98,
    "Euro": 0.008,
    "UK Pound": 0.005,
    "Rupee": 0.004,
    "Yuan": 0.003,
}
# Format shares of IBM HI-Small (docs/dataset_card.md). Normal payments use the
# overall mix without Reinvestment (which is reserved for self-transfers);
# uncamouflaged pattern payments use the mix of laundering-labelled rows, which
# is ~87% ACH. Camouflage moves pattern payments from the second to the first.
NORMAL_FORMAT_MIX: dict[str, float] = {
    "Cheque": 0.405,
    "Credit Card": 0.288,
    "ACH": 0.131,
    "Cash": 0.107,
    "Wire": 0.037,
    "Bitcoin": 0.032,
}
LAUNDERING_FORMAT_MIX: dict[str, float] = {
    "ACH": 0.87,
    "Cheque": 0.06,
    "Credit Card": 0.04,
    "Cash": 0.02,
    "Bitcoin": 0.01,
}

MINUTES_PER_DAY = 1440
# Stream identifiers (see module docstring). Values are arbitrary but frozen:
# changing one changes every dataset generated with the affected stream.
_POPULATION, _NORMAL, _PATTERN = 0, 1, 2
_STRUCTURE, _TIMING, _AMOUNT, _FORMAT = 0, 1, 2, 3
# Burn-in days are negative; SeedSequence entropy must be non-negative.
_DAY_KEY_OFFSET = 1_000
# Hop-to-hop multiplicative noise on uncamouflaged pattern amounts (log scale),
# so a chain is not a run of exactly equal amounts.
_HOP_AMOUNT_JITTER = 0.1
# Burn-in covers this multiple of the longest pattern's expected duration.
_BURN_IN_MARGIN = 1.5


@dataclass(frozen=True)
class GeneratorConfig:
    """configs/synthetic.yaml, loaded and checked.

    The five experimental knobs (§14.1, feasibility §3.3) are ring_size,
    prevalence, camouflage, hop_delay_minutes and the drift_* fields. The
    behaviour parameters below them make the problem hard in the ways real data
    is hard (§14.2) but are not meant to be swept.
    """

    seed: int = 0
    n_accounts: int = 5_000
    n_banks: int = 30
    n_days: int = 20
    start: pd.Timestamp = pd.Timestamp("2026-01-05")
    tx_per_account_per_day: float = 2.0

    # --- the experimental knobs ---
    ring_size: tuple[int, int] = (3, 12)
    prevalence: float = 0.002
    camouflage: float = 0.3
    hop_delay_minutes: float = 180.0
    drift_onset_day: int | None = None
    drift_kind: str = "amount_shift"
    # amount_shift: multiplier on normal amounts; velocity_shift: multiplier on
    # normal activity; new_typology: share of post-onset patterns that use it.
    drift_magnitude: float = 3.0
    drift_typology: str | None = None
    typologies: tuple[str, ...] = IBM_TYPOLOGIES

    # --- normal behaviour (§14.2) ---
    activity_sigma: float = 0.9
    amount_log_mean: float = 6.8
    amount_log_sd_between: float = 1.0
    amount_sigma_range: tuple[float, float] = (0.5, 1.2)
    hour_sd: float = 3.0
    weekend_factor: float = 0.6
    bank_zipf: float = 1.0
    payees_mean: float = 4.0
    repeat_payee_prob: float = 0.8
    same_bank_payee_prob: float = 0.3
    popularity_sigma: float = 1.0
    self_transfer_rate: float = 0.05
    primary_format_prob: float = 0.6

    # --- injection details (§14.3, §14.4) ---
    blatant_log_offset: float = 2.0
    blatant_log_sd: float = 0.5
    skim_range: tuple[float, float] = (0.02, 0.10)
    scripted_delay_shape: float = 8.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", pd.Timestamp(self.start))
        object.__setattr__(self, "ring_size", tuple(int(v) for v in self.ring_size))
        object.__setattr__(self, "typologies", tuple(str(t) for t in self.typologies))
        object.__setattr__(
            self, "amount_sigma_range", tuple(float(v) for v in self.amount_sigma_range)
        )
        object.__setattr__(self, "skim_range", tuple(float(v) for v in self.skim_range))
        self._check()

    def _check(self) -> None:
        """Raise on any invalid combination, all problems at once."""
        p: list[str] = []
        if self.seed < 0:
            p.append("seed must be >= 0")
        if self.n_accounts < max(self.ring_size[-1], 10):
            p.append("n_accounts must be >= max(ring_size[1], 10)")
        if not 1 <= self.n_banks <= 999:
            p.append("n_banks must be in [1, 999]")
        if self.n_days < 1:
            p.append("n_days must be >= 1")
        if self.tx_per_account_per_day <= 0:
            p.append("tx_per_account_per_day must be > 0")
        if len(self.ring_size) != 2 or not 3 <= self.ring_size[0] <= self.ring_size[1]:
            p.append("ring_size must be (min, max) with 3 <= min <= max")
        if not 0 < self.prevalence < 0.5:
            p.append("prevalence must be in (0, 0.5)")
        if not 0.0 <= self.camouflage <= 1.0:
            p.append("camouflage must be in [0, 1]")
        if self.hop_delay_minutes <= 0:
            p.append("hop_delay_minutes must be > 0")
        unknown = sorted(set(self.typologies) - set(IBM_TYPOLOGIES))
        if unknown or not self.typologies:
            p.append(f"typologies must be a non-empty subset of {IBM_TYPOLOGIES}")
        if self.drift_kind not in DRIFT_KINDS:
            p.append(f"drift_kind must be one of {DRIFT_KINDS}")
        if self.drift_onset_day is not None:
            if not 1 <= self.drift_onset_day <= self.n_days - 1:
                p.append("drift_onset_day must be in [1, n_days - 1] or null")
            if self.drift_kind == "new_typology":
                if not 0 < self.drift_magnitude <= 1:
                    p.append("for new_typology, drift_magnitude is a share in (0, 1]")
                if self.resolved_drift_typology is None:
                    p.append("new_typology drift needs a typology not in `typologies`")
            elif self.drift_magnitude <= 0:
                p.append("drift_magnitude must be > 0")
        if self.drift_typology is not None and (
            self.drift_typology not in IBM_TYPOLOGIES
            or self.drift_typology in self.typologies
        ):
            p.append("drift_typology must be an IBM typology not in `typologies`")
        for name in ("repeat_payee_prob", "same_bank_payee_prob", "self_transfer_rate",
                     "primary_format_prob"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                p.append(f"{name} must be in [0, 1]")
        if not self.weekend_factor > 0:
            p.append("weekend_factor must be > 0")
        if self.payees_mean < 1:
            p.append("payees_mean must be >= 1")
        if self.scripted_delay_shape < 1:
            p.append("scripted_delay_shape must be >= 1")
        lo, hi = self.skim_range
        if not 0 <= lo <= hi < 1:
            p.append("skim_range must satisfy 0 <= lo <= hi < 1")
        if p:
            raise ValueError("invalid GeneratorConfig: " + "; ".join(p))

    @property
    def resolved_drift_typology(self) -> str | None:
        """The typology that appears only after onset (new_typology drift)."""
        if self.drift_typology is not None:
            return self.drift_typology
        unused = [t for t in IBM_TYPOLOGIES if t not in self.typologies]
        return unused[0] if unused else None

    @property
    def drift_onset(self) -> pd.Timestamp | None:
        """The instant drift starts; every row before it is unaffected by drift."""
        if self.drift_onset_day is None:
            return None
        return self.start + pd.Timedelta(days=self.drift_onset_day)

    @property
    def end(self) -> pd.Timestamp:
        """Exclusive end of the generated period."""
        return self.start + pd.Timedelta(days=self.n_days)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready view for manifests (timestamps as ISO strings, tuples as lists)."""
        out = asdict(self)
        out["start"] = self.start.isoformat()
        return {k: list(v) if isinstance(v, tuple) else v for k, v in out.items()}


_FIELD_NAMES = {f.name for f in fields(GeneratorConfig)}


def load_generator_config(path: str | Path) -> GeneratorConfig:
    """Load configs/synthetic.yaml.

    Sections (generator / injection / drift / behaviour) only group keys for the
    reader; every key is a GeneratorConfig field name. Unknown keys raise, so a
    typo cannot silently fall back to a default.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    merged: dict[str, Any] = {}
    for section, values in raw.items():
        if not isinstance(values, dict):
            raise ValueError(f"section '{section}' must be a mapping")
        dup = set(merged) & set(values)
        if dup:
            raise ValueError(f"keys defined twice: {sorted(dup)}")
        merged |= values
    unknown = sorted(set(merged) - _FIELD_NAMES)
    if unknown:
        raise ValueError(f"unknown generator config keys: {unknown}")
    return GeneratorConfig(**merged)


@dataclass
class SyntheticData:
    """One generated dataset.

    transactions: standard schema, sorted by timestamp.
    patterns: one row per injected pattern payment (pattern_id, typology, detail,
        timestamp, tx_id). tx_id is null for payments that fall outside the
        generated period (a pattern still running at the end, or one that
        started during burn-in); they are kept so each pattern's true start and
        end are known, as in ibm_aml.match_patterns.
    members: (pattern_id, account) -- every account a pattern was built on.
    Label-derived; evaluation only.
    """

    transactions: pd.DataFrame
    patterns: pd.DataFrame
    members: pd.DataFrame
    config: GeneratorConfig

    @property
    def prevalence(self) -> float:
        return float(self.transactions["is_fraud"].mean())

    def rings(self) -> dict[str, dict[str, Any]]:
        """Ground truth in the shape nexis.evaluation.metrics.early_warning_times takes.

        t_start / t_end span every payment of the pattern, including any that fall
        outside the generated period, so early-warning time is measured against
        the pattern's true completion.
        """
        span = self.patterns.groupby("pattern_id")["timestamp"].agg(["min", "max"])
        members = self.members.groupby("pattern_id")["account"].agg(set)
        typ = self.patterns.groupby("pattern_id", observed=True)["typology"].first()
        return {
            f"P{int(pid):05d}": {
                "members": members.loc[pid],
                "t_start": span.loc[pid, "min"],
                "t_end": span.loc[pid, "max"],
                "typology": str(typ.loc[pid]),
            }
            for pid in span.index
        }


# --------------------------------------------------------------------------- #
# population (§14.2)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Population:
    account_id: np.ndarray
    bank: np.ndarray
    rate: np.ndarray
    amount_mu: np.ndarray
    amount_sigma: np.ndarray
    home_hour: np.ndarray
    currency: np.ndarray
    primary_format: np.ndarray
    payees: np.ndarray
    n_payees: np.ndarray
    popularity_cum: np.ndarray


def _rng(*key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(k) for k in key]))


def _cum(probs: dict[str, float], order: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """(category indices into `order`, normalised cumulative probabilities)."""
    idx = np.array([order.index(k) for k in probs])
    p = np.array(list(probs.values()), dtype=float)
    return idx, np.cumsum(p / p.sum())


def _draw(rng: np.random.Generator, cum: np.ndarray, size: int) -> np.ndarray:
    """Inverse-CDF draw from cumulative weights: one uniform per draw, always."""
    out = np.searchsorted(cum, rng.random(size) * cum[-1], side="right")
    return np.minimum(out, len(cum) - 1)


_NORMAL_FMT = _cum(NORMAL_FORMAT_MIX, PAYMENT_FORMATS)
_LAUNDER_FMT = _cum(LAUNDERING_FORMAT_MIX, PAYMENT_FORMATS)
_ACCOUNT_CUR = _cum(ACCOUNT_CURRENCY_MIX, CURRENCIES)
_REINVESTMENT = PAYMENT_FORMATS.index("Reinvestment")
_BITCOIN_FMT = PAYMENT_FORMATS.index("Bitcoin")
_BITCOIN_CUR = CURRENCIES.index("Bitcoin")


def _population(cfg: GeneratorConfig) -> _Population:
    """Accounts with their own amount scale, activity, rhythm and payees.

    Personal amount parameters make "amount vs own baseline" a different feature
    from "amount vs population" (§14.2, §15.4). Popularity-weighted payee choice
    gives a heavy-tailed in-degree, so degree features face the same difficulty
    as on real data rather than separating pattern members trivially.
    """
    rng = _rng(cfg.seed, _POPULATION)
    n = cfg.n_accounts

    zipf = 1.0 / np.arange(1, cfg.n_banks + 1) ** cfg.bank_zipf
    bank = _draw(rng, np.cumsum(zipf / zipf.sum()), n)
    s = cfg.activity_sigma
    rate = cfg.tx_per_account_per_day * rng.lognormal(-0.5 * s * s, s, n)
    amount_mu = rng.normal(cfg.amount_log_mean, cfg.amount_log_sd_between, n)
    amount_sigma = rng.uniform(*cfg.amount_sigma_range, n)
    home_hour = rng.integers(7, 22, n).astype(float)
    cur_idx, cur_cum = _ACCOUNT_CUR
    currency = cur_idx[_draw(rng, cur_cum, n)]
    fmt_idx, fmt_cum = _NORMAL_FMT
    primary_format = fmt_idx[_draw(rng, fmt_cum, n)]

    popularity = rate * rng.lognormal(0.0, cfg.popularity_sigma, n)
    popularity_cum = np.cumsum(popularity / popularity.sum())

    k_max = max(2, int(math.ceil(4 * cfg.payees_mean)))
    n_payees = np.minimum(1 + rng.poisson(cfg.payees_mean - 1, n), k_max)
    payees = _draw(rng, popularity_cum, n * k_max).reshape(n, k_max)
    same_bank = rng.random((n, k_max)) < cfg.same_bank_payee_prob
    u = rng.random((n, k_max))
    for b in range(cfg.n_banks):
        in_bank = np.flatnonzero(bank == b)
        rows = same_bank & (bank == b)[:, None]
        if in_bank.size == 0 or not rows.any():
            continue
        cum = np.cumsum(popularity[in_bank])
        pick = np.searchsorted(cum, u[rows] * cum[-1], side="right")
        payees[rows] = in_bank[np.minimum(pick, in_bank.size - 1)]
    own = payees == np.arange(n)[:, None]
    payees[own] = (payees[own] + 1) % n

    account_id = np.array(
        [f"{b:03d}_A{i:07d}" for i, b in enumerate(bank)], dtype=object
    )
    return _Population(
        account_id=account_id,
        bank=bank,
        rate=rate,
        amount_mu=amount_mu,
        amount_sigma=amount_sigma,
        home_hour=home_hour,
        currency=currency,
        primary_format=primary_format,
        payees=payees,
        n_payees=n_payees,
        popularity_cum=popularity_cum,
    )


def _day_factor(cfg: GeneratorConfig, day: int) -> float:
    """Activity multiplier for a day: weekend dip, and velocity drift after onset."""
    factor = 1.0
    if (cfg.start + pd.Timedelta(days=day)).dayofweek >= 5:
        factor *= cfg.weekend_factor
    if (
        cfg.drift_kind == "velocity_shift"
        and cfg.drift_onset_day is not None
        and day >= cfg.drift_onset_day
    ):
        factor *= cfg.drift_magnitude
    return factor


def _account_formats(
    rng: np.random.Generator, pop: _Population, src: np.ndarray, cfg: GeneratorConfig
) -> np.ndarray:
    """An account's usual payment format, or a population draw."""
    use_primary = rng.random(src.size) < cfg.primary_format_prob
    fmt_idx, fmt_cum = _NORMAL_FMT
    other = fmt_idx[_draw(rng, fmt_cum, src.size)]
    return np.where(use_primary, pop.primary_format[src], other)


def _normal_day(cfg: GeneratorConfig, pop: _Population, day: int) -> dict[str, np.ndarray]:
    """One day of normal payments, from the day's own stream."""
    rng = _rng(cfg.seed, _NORMAL, day + _DAY_KEY_OFFSET)
    n = cfg.n_accounts
    counts = rng.poisson(pop.rate * _day_factor(cfg, day))
    src = np.repeat(np.arange(n), counts)
    m = src.size

    hour = (pop.home_hour[src] + rng.normal(0.0, cfg.hour_sd, m)) % 24.0
    minute = np.minimum((hour * 60).astype(np.int64), MINUTES_PER_DAY - 1)

    is_self = rng.random(m) < cfg.self_transfer_rate
    use_payee = rng.random(m) < cfg.repeat_payee_prob
    slot = (rng.random(m) * pop.n_payees[src]).astype(np.int64)
    new_cp = _draw(rng, pop.popularity_cum, m)
    dst = np.where(use_payee, pop.payees[src, slot], new_cp)
    dst = np.where(dst == src, (dst + 1) % n, dst)
    dst = np.where(is_self, src, dst)

    fmt = _account_formats(rng, pop, src, cfg)
    fmt = np.where(is_self, _REINVESTMENT, fmt)
    log_usd = pop.amount_mu[src] + pop.amount_sigma[src] * rng.standard_normal(m)
    return {
        "minute": day * MINUTES_PER_DAY + minute,
        "src": src,
        "dst": dst,
        "fmt": fmt,
        "usd": np.exp(log_usd),
    }


# --------------------------------------------------------------------------- #
# typologies (§14.3): (src position, dst position, depth) over members 0..s-1.
# depth = hops from where the money entered; it drives the per-hop skim.
# --------------------------------------------------------------------------- #

Edge = tuple[int, int, int]


def _fan_out(s: int, rng: np.random.Generator) -> list[Edge]:
    return [(0, i, 0) for i in range(1, s)]


def _fan_in(s: int, rng: np.random.Generator) -> list[Edge]:
    return [(i, 0, 0) for i in range(1, s)]


def _cycle(s: int, rng: np.random.Generator) -> list[Edge]:
    return [(i, (i + 1) % s, i) for i in range(s)]


def _scatter_gather(s: int, rng: np.random.Generator) -> list[Edge]:
    mids = range(1, s - 1)
    return [(0, i, 0) for i in mids] + [(i, s - 1, 1) for i in mids]


def _gather_scatter(s: int, rng: np.random.Generator) -> list[Edge]:
    k = (s - 1) // 2
    sources, targets = range(1, 1 + k), range(1 + k, s)
    return [(i, 0, 0) for i in sources] + [(0, j, 1) for j in targets]


def _layers(s: int, n_layers: int) -> list[list[int]]:
    bounds = np.linspace(0, s, n_layers + 1).round().astype(int)
    return [list(range(bounds[i], bounds[i + 1])) for i in range(n_layers)]


def _stack(s: int, rng: np.random.Generator) -> list[Edge]:
    layers = _layers(s, 3)
    return [
        (a, b, depth)
        for depth in range(2)
        for a in layers[depth]
        for b in layers[depth + 1]
    ]


def _bipartite(s: int, rng: np.random.Generator) -> list[Edge]:
    left, right = _layers(s, 2)
    return [(a, b, 0) for a in left for b in right]


def _random(s: int, rng: np.random.Generator) -> list[Edge]:
    """Random tree: member i receives from a uniformly chosen earlier member."""
    depth = [0] * s
    edges: list[Edge] = []
    for i in range(1, s):
        parent = int(rng.integers(i))
        depth[i] = depth[parent] + 1
        edges.append((parent, i, depth[parent]))
    return edges


_BUILDERS: dict[str, Callable[[int, np.random.Generator], list[Edge]]] = {
    "FAN-IN": _fan_in,
    "FAN-OUT": _fan_out,
    "CYCLE": _cycle,
    "SCATTER-GATHER": _scatter_gather,
    "GATHER-SCATTER": _gather_scatter,
    "STACK": _stack,
    "BIPARTITE": _bipartite,
    "RANDOM": _random,
}


def _max_pattern_tx(s: int) -> int:
    """Largest payment count any typology produces with s members."""
    probe = np.random.default_rng(0)
    return max(len(build(s, probe)) for build in _BUILDERS.values())


@dataclass
class _Pattern:
    day: int
    typology: str
    members: np.ndarray
    minute: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    fmt: np.ndarray
    usd: np.ndarray


def _make_pattern(cfg: GeneratorConfig, pop: _Population, day: int, j: int) -> _Pattern:
    """Pattern j started on `day`. Each aspect has its own stream (paired sweeps)."""
    key = (cfg.seed, _PATTERN, day + _DAY_KEY_OFFSET, j)
    c = cfg.camouflage

    rs = _rng(*key, _STRUCTURE)
    u_new, base_pick = rs.random(), int(rs.integers(len(cfg.typologies)))
    new_active = (
        cfg.drift_kind == "new_typology"
        and cfg.drift_onset_day is not None
        and day >= cfg.drift_onset_day
    )
    typology = cfg.typologies[base_pick]
    if new_active and u_new < cfg.drift_magnitude:
        typology = str(cfg.resolved_drift_typology)
    size = int(rs.integers(cfg.ring_size[0], cfg.ring_size[1] + 1))
    members = rs.choice(cfg.n_accounts, size, replace=False)
    edges = np.array(_BUILDERS[typology](size, rs), dtype=np.int64)
    src, dst, depth = members[edges[:, 0]], members[edges[:, 1]], edges[:, 2]
    n_tx = len(edges)

    # Timing (§14.4 mechanism 3). The mean hop delay is the delay knob at every
    # camouflage level; camouflage changes only the shape: scripted, regular
    # hops starting at any hour (c=0) versus Poisson-like gaps starting in the
    # first member's usual hours (c=1).
    rt = _rng(*key, _TIMING)
    use_cycle = rt.random() < c
    any_minute = rt.random() * MINUTES_PER_DAY
    cycle_minute = ((pop.home_hour[members[0]] + rt.normal(0.0, cfg.hour_sd)) % 24.0) * 60
    start_minute = cycle_minute if use_cycle else any_minute
    shape = 1.0 + (1.0 - c) * (cfg.scripted_delay_shape - 1.0)
    delays = rt.gamma(shape, cfg.hop_delay_minutes / shape, n_tx - 1)
    offsets = np.concatenate([[0.0], np.cumsum(delays)])
    minute = np.floor(day * MINUTES_PER_DAY + start_minute + offsets).astype(np.int64)

    # Amounts (§14.4 mechanism 1): blend, in log space, a blatant chain amount
    # (large, skimmed per hop) with a draw from the sender's own distribution.
    # At c=1 the amount IS a normal draw for that sender.
    ra = _rng(*key, _AMOUNT)
    log_a0 = ra.normal(cfg.amount_log_mean + cfg.blatant_log_offset, cfg.blatant_log_sd)
    skim = ra.uniform(*cfg.skim_range)
    blatant = log_a0 + depth * math.log1p(-skim) + ra.normal(0.0, _HOP_AMOUNT_JITTER, n_tx)
    personal = pop.amount_mu[src] + pop.amount_sigma[src] * ra.standard_normal(n_tx)
    usd = np.exp((1.0 - c) * blatant + c * personal)

    # Payment format: the laundering-labelled mix, or the sender's own habit.
    rf = _rng(*key, _FORMAT)
    use_own = rf.random(n_tx) < c
    own_fmt = _account_formats(rf, pop, src, cfg)
    l_idx, l_cum = _LAUNDER_FMT
    launder_fmt = l_idx[_draw(rf, l_cum, n_tx)]
    fmt = np.where(use_own, own_fmt, launder_fmt)

    return _Pattern(day, typology, members, minute, src, dst, fmt, usd)


def _burn_in_days(cfg: GeneratorConfig) -> int:
    """Days of pattern starts simulated before day 0.

    Without burn-in the first days would hold only freshly started patterns,
    so prevalence would ramp up over the first few days instead of being
    stationary -- and a ramp is itself a drift that would confound the drift
    experiment.
    """
    longest = _max_pattern_tx(cfg.ring_size[1]) * cfg.hop_delay_minutes
    return int(math.ceil(_BURN_IN_MARGIN * longest / MINUTES_PER_DAY)) + 1


def _inject(cfg: GeneratorConfig, pop: _Population) -> list[_Pattern]:
    """Start patterns day by day until each day's payment budget is spent.

    Budget for day d = prevalence/(1-prevalence) x expected normal volume of d,
    so laundering-labelled payments make up `prevalence` of all rows in the
    stationary regime. Over- or under-shoot (patterns are discrete) carries to
    the next day; it only ever flows forward in time.
    """
    odds = cfg.prevalence / (1.0 - cfg.prevalence)
    base_volume = float(pop.rate.sum())
    patterns: list[_Pattern] = []
    carry = 0.0
    for day in range(-_burn_in_days(cfg), cfg.n_days):
        budget = odds * base_volume * _day_factor(cfg, day) + carry
        spent, j = 0, 0
        while spent < budget:
            pattern = _make_pattern(cfg, pop, day, j)
            patterns.append(pattern)
            spent += pattern.minute.size
            j += 1
        carry = budget - spent
    return patterns


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #


def _round_amount(values: np.ndarray, currency: np.ndarray) -> np.ndarray:
    out = np.empty_like(values)
    for ci, name in enumerate(CURRENCIES):
        m = currency == ci
        if m.any():
            d = _DECIMALS[name]
            out[m] = np.maximum(np.round(values[m], d), 10.0**-d)
    return out


def _categorical(codes: np.ndarray, categories: tuple[str, ...] | list[str]) -> pd.Categorical:
    return pd.Categorical.from_codes(codes, categories=pd.Index(list(categories), dtype="str"))


def generate(cfg: GeneratorConfig) -> SyntheticData:
    """Generate a dataset. Fully determined by `cfg` (including `cfg.seed`)."""
    pop = _population(cfg)
    horizon = cfg.n_days * MINUTES_PER_DAY

    normal_days = [_normal_day(cfg, pop, d) for d in range(cfg.n_days)]
    normal = {k: np.concatenate([nd[k] for nd in normal_days]) for k in normal_days[0]}
    n_normal = normal["minute"].size

    # Patterns that never touch the generated period (burn-in only) are dropped.
    patterns = [
        p for p in _inject(cfg, pop) if ((p.minute >= 0) & (p.minute < horizon)).any()
    ]
    pat_minute = np.concatenate([p.minute for p in patterns]) if patterns else np.zeros(0, np.int64)
    pat_id = np.concatenate(
        [np.full(p.minute.size, i, dtype=np.int64) for i, p in enumerate(patterns)]
    ) if patterns else np.zeros(0, np.int64)
    inside = (pat_minute >= 0) & (pat_minute < horizon)

    def cat(key: str) -> np.ndarray:
        pat_vals = [getattr(p, key) for p in patterns]
        pat = np.concatenate(pat_vals)[inside] if pat_vals else np.zeros(0)
        return np.concatenate([normal[key], pat.astype(normal[key].dtype)])

    minute = cat("minute")
    src, dst, fmt, usd = cat("src"), cat("dst"), cat("fmt"), cat("usd")
    is_pattern = np.r_[np.zeros(n_normal, bool), np.ones(int(inside.sum()), bool)]

    # Amount drift is a post-transform keyed on time, so every row before onset
    # is untouched by construction. Pattern payments follow the shift only in
    # their camouflaged share (log-blend weight c => factor magnitude**c): they
    # mimic the members' *current* normal behaviour, the blatant part does not
    # move, and so P(label | amount) changes too.
    if cfg.drift_kind == "amount_shift" and cfg.drift_onset_day is not None:
        after = minute >= cfg.drift_onset_day * MINUTES_PER_DAY
        factor = np.where(is_pattern, cfg.drift_magnitude**cfg.camouflage, cfg.drift_magnitude)
        usd = np.where(after, usd * factor, usd)

    bitcoin = fmt == _BITCOIN_FMT
    pay_cur = np.where(bitcoin, _BITCOIN_CUR, pop.currency[src])
    recv_cur = np.where(bitcoin, _BITCOIN_CUR, pop.currency[dst])
    rates = np.array([USD_PER_UNIT[c] for c in CURRENCIES])
    amount_paid = _round_amount(usd / rates[pay_cur], pay_cur)
    amount_received = _round_amount(usd / rates[recv_cur], recv_cur)

    # Stable sort: same-minute rows keep construction order (normal rows day by
    # day, then pattern rows by pattern and hop), so the order -- and tx_id, the
    # sorted position -- is a deterministic function of the config.
    order = np.argsort(minute, kind="stable")
    position = np.empty_like(order)
    position[order] = np.arange(order.size)
    width = max(7, len(str(max(order.size - 1, 0))))
    tx_ids = np.array([f"{TX_PREFIX}:{i:0{width}d}" for i in range(order.size)], dtype=object)

    start_us = cfg.start.as_unit("us").to_datetime64()
    ts = start_us + minute[order].astype("timedelta64[m]")
    s, d = src[order], dst[order]
    bank_codes = [f"{b:03d}" for b in range(cfg.n_banks)]
    df = pd.DataFrame(
        {
            "tx_id": pd.Series(tx_ids, dtype="str"),
            "src": pd.Series(pop.account_id[s], dtype="str"),
            "dst": pd.Series(pop.account_id[d], dtype="str"),
            "amount": amount_paid[order] * rates[pay_cur[order]],
            "timestamp": pd.Series(ts, dtype="datetime64[us]"),
            "is_fraud": is_pattern[order].astype("int8"),
            "src_bank": _categorical(pop.bank[s], bank_codes),
            "dst_bank": _categorical(pop.bank[d], bank_codes),
            "amount_paid": amount_paid[order],
            "pay_currency": _categorical(pay_cur[order], CURRENCIES),
            "amount_received": amount_received[order],
            "recv_currency": _categorical(recv_cur[order], CURRENCIES),
            "payment_format": _categorical(fmt[order], PAYMENT_FORMATS),
            "is_self_transfer": (s == d).astype("int8"),
            "is_cross_bank": (pop.bank[s] != pop.bank[d]).astype("int8"),
            "is_cross_currency": (pay_cur[order] != recv_cur[order]).astype("int8"),
        }
    )[list(REQUIRED_COLUMNS + EXTRA_COLUMNS)]

    # Pattern table: every pattern payment, inside the period or not.
    pat_tx = np.full(pat_minute.size, None, dtype=object)
    pat_tx[inside] = tx_ids[position[n_normal:]]
    typ = np.concatenate([np.full(p.minute.size, p.typology, dtype=object) for p in patterns]) if patterns else np.zeros(0, object)
    sizes = np.concatenate([np.full(p.minute.size, p.members.size) for p in patterns]) if patterns else np.zeros(0, int)
    mapping = pd.DataFrame(
        {
            "pattern_id": pat_id.astype("int32"),
            "typology": pd.Categorical(typ, categories=pd.Index(sorted(IBM_TYPOLOGIES), dtype="str")),
            "detail": pd.Series([f"{n} accounts" for n in sizes], dtype="str"),
            "timestamp": pd.Series(
                start_us + pat_minute.astype("timedelta64[m]"), dtype="datetime64[us]"
            ),
            "tx_id": pd.Series(pat_tx, dtype="str"),
        }
    )
    mapping = mapping.sort_values(["pattern_id", "timestamp"], kind="stable").reset_index(drop=True)

    members = pd.DataFrame(
        {
            "pattern_id": np.concatenate(
                [np.full(p.members.size, i) for i, p in enumerate(patterns)]
            ).astype("int32") if patterns else np.zeros(0, "int32"),
            "account": pd.Series(
                np.concatenate([pop.account_id[p.members] for p in patterns]) if patterns else [],
                dtype="str",
            ),
        }
    )

    validate(df)
    return SyntheticData(transactions=df, patterns=mapping, members=members, config=cfg)


# --------------------------------------------------------------------------- #
# persistence (§14.6)
# --------------------------------------------------------------------------- #

_FILES = {
    "transactions": "transactions.parquet",
    "patterns": "patterns.parquet",
    "members": "members.parquet",
}


def save_synthetic(
    data: SyntheticData, out_dir: str | Path, record_code_state: bool = True
) -> Path:
    """Write parquet files plus a manifest with config, counts and content hashes.

    The hashes are what make a synthetic result citable: an experiment records
    the manifest it ran on, and `load_synthetic` refuses files whose bytes no
    longer match it. Returns the manifest path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    code_state: dict[str, Any] = {}
    if record_code_state:
        # Captured before writing, so this build's own outputs cannot make the
        # tree look dirty (same convention as ibm_aml.build_processed).
        from nexis.evaluation.harness import _git_state

        code_state = _git_state()

    tx = data.transactions
    for attr, name in _FILES.items():
        getattr(data, attr).to_parquet(out / name, index=False)

    by_typ = data.patterns.groupby("typology", observed=True)["pattern_id"].nunique()
    manifest: dict[str, Any] = {
        "dataset": "synthetic",
        "generator_version": GENERATOR_VERSION,
        "config": data.config.to_dict(),
        "n_rows": len(tx),
        "n_pos": int(tx["is_fraud"].sum()),
        "prevalence": data.prevalence,
        "target_prevalence": data.config.prevalence,
        "n_accounts_active": int(pd.concat([tx["src"], tx["dst"]]).nunique()),
        "n_patterns": int(data.patterns["pattern_id"].nunique()),
        "n_pattern_rows": len(data.patterns),
        "n_pattern_rows_outside_period": int(data.patterns["tx_id"].isna().sum()),
        "patterns_by_typology": {str(k): int(v) for k, v in by_typ.items()},
        "t_min": str(tx["timestamp"].min()),
        "t_max": str(tx["timestamp"].max()),
        "drift_onset": None if data.config.drift_onset is None else data.config.drift_onset.isoformat(),
        "files": {
            attr: {"file": name, "sha256": file_sha256(out / name)}
            for attr, name in _FILES.items()
        },
        **code_state,
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load_synthetic(out_dir: str | Path) -> SyntheticData:
    """Load a saved dataset, verifying every file against its manifest hash."""
    out = Path(out_dir)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    frames: dict[str, pd.DataFrame] = {}
    for attr, entry in manifest["files"].items():
        path = out / entry["file"]
        actual = file_sha256(path)
        if actual != entry["sha256"]:
            raise DataValidationError(
                f"{path.name} hash {actual[:12]} does not match manifest "
                f"{entry['sha256'][:12]}; regenerate the dataset"
            )
        frames[attr] = pd.read_parquet(path)
    validate(frames["transactions"])
    return SyntheticData(config=GeneratorConfig(**manifest["config"]), **frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args()

    cfg = load_generator_config(args.config)
    t0 = time.perf_counter()
    data = generate(cfg)
    elapsed = time.perf_counter() - t0
    manifest_path = save_synthetic(data, args.out)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"generated in {elapsed:.1f}s; wrote {args.out}")
    for key in ("n_rows", "n_pos", "prevalence", "target_prevalence", "n_patterns",
                "n_pattern_rows_outside_period", "t_min", "t_max", "drift_onset"):
        print(f"  {key:<32} {manifest[key]}")
    for typ, count in manifest["patterns_by_typology"].items():
        print(f"  patterns.{typ:<23} {count}")


if __name__ == "__main__":
    main()
