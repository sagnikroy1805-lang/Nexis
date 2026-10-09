"""Tier-3 synthetic generator tests (Concept Mastery Module 14).

What can fail silently here: the schema drifting away from the IBM adapter's
(every downstream module would still run, on subtly different types), labels
that do not match the injected patterns, knobs that do not do what they claim,
and drift that leaks into the pre-onset period (which would make every
detection-latency number meaningless).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from nexis.data.ibm_aml import (
    EXTRA_COLUMNS,
    REQUIRED_COLUMNS,
    file_sha256,
    read_raw,
    to_standard_schema,
    validate,
)
from nexis.data.synthetic import (
    IBM_TYPOLOGIES,
    GeneratorConfig,
    generate,
    load_generator_config,
    load_synthetic,
    save_synthetic,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ibm_aml_sample.csv"
CONFIG = Path(__file__).parent.parent / "configs" / "synthetic.yaml"

# Small enough to generate in well under a second, with enough positives
# (prevalence 1%) for distributional checks.
SMALL = GeneratorConfig(seed=7, n_accounts=1500, n_days=12, prevalence=0.01)


@pytest.fixture(scope="module")
def data():
    return generate(SMALL)


def test_same_seed_gives_identical_data(data):
    again = generate(SMALL)
    pd.testing.assert_frame_equal(data.transactions, again.transactions)
    pd.testing.assert_frame_equal(data.patterns, again.patterns)
    pd.testing.assert_frame_equal(data.members, again.members)


def test_different_seed_gives_different_data(data):
    other = generate(dataclasses.replace(SMALL, seed=8))
    assert not data.transactions[["src", "dst", "amount"]].equals(
        other.transactions[["src", "dst", "amount"]]
    )


def test_output_passes_standard_schema_validation(data):
    validate(data.transactions)
    assert data.transactions["tx_id"].str.match(r"^SYN:\d{7}$").all()


def test_schema_matches_ibm_adapter_exactly(data):
    """Same columns, order and dtypes as the IBM adapter, so every module runs on both."""
    ibm, _ = to_standard_schema(
        read_raw(FIXTURE), "TEST", pd.Timestamp("2022-09-11"), pd.Timestamp("2022-09-02")
    )
    syn = data.transactions
    assert list(syn.columns) == list(REQUIRED_COLUMNS + EXTRA_COLUMNS) == list(ibm.columns)
    for col in ibm.columns:
        want, got = ibm[col].dtype, syn[col].dtype
        if isinstance(want, pd.CategoricalDtype):
            assert isinstance(got, pd.CategoricalDtype), col
            assert got.categories.dtype == want.categories.dtype, col
        else:
            assert got == want, f"{col}: {got} != {want}"


def test_pattern_table_has_ibm_mapping_shape(data):
    assert list(data.patterns.columns) == ["pattern_id", "typology", "detail", "timestamp", "tx_id"]
    assert data.patterns["pattern_id"].dtype == "int32"
    assert isinstance(data.patterns["typology"].dtype, pd.CategoricalDtype)
    assert data.patterns["timestamp"].dtype == "datetime64[us]"


def test_prevalence_close_to_target():
    """Burn-in makes prevalence stationary, so the realised share tracks the target."""
    for target in (0.002, 0.01):
        d = generate(dataclasses.replace(SMALL, n_accounts=3000, n_days=20, prevalence=target))
        assert abs(d.prevalence - target) / target < 0.25, (target, d.prevalence)


def test_labels_and_patterns_agree(data):
    """Every pattern tx exists and is positive; every positive belongs to exactly one pattern."""
    tx = data.transactions.set_index("tx_id")
    kept = data.patterns["tx_id"].dropna()
    assert kept.is_unique
    assert kept.isin(tx.index).all()
    assert (tx.loc[kept, "is_fraud"] == 1).all()
    positives = set(tx.index[tx["is_fraud"] == 1])
    assert positives == set(kept)


def test_pattern_sizes_within_ring_size_and_edges_within_members(data):
    lo, hi = SMALL.ring_size
    sizes = data.members.groupby("pattern_id")["account"].nunique()
    assert sizes.between(lo, hi).all()
    assert set(sizes.index) == set(data.patterns["pattern_id"])

    tx = data.transactions.set_index("tx_id")
    rows = data.patterns.dropna(subset=["tx_id"])
    members = data.members.groupby("pattern_id")["account"].agg(set)
    for pid, grp in rows.groupby("pattern_id"):
        accounts = set(tx.loc[grp["tx_id"], "src"]) | set(tx.loc[grp["tx_id"], "dst"])
        assert accounts <= members.loc[pid]


def test_hop_delays_match_the_knob():
    """Mean gap between consecutive payments of a pattern ~= hop_delay_minutes."""
    for delay in (60.0, 240.0):
        d = generate(dataclasses.replace(SMALL, hop_delay_minutes=delay))
        gaps = d.patterns.groupby("pattern_id")["timestamp"].diff().dropna()
        mean = gaps.dt.total_seconds().mean() / 60.0
        assert abs(mean - delay) / delay < 0.15, (delay, mean)


def test_patterns_unfold_over_time(data):
    """Patterns are not instantaneous, so early detection is measurable."""
    span = data.patterns.groupby("pattern_id")["timestamp"].agg(lambda s: s.max() - s.min())
    assert (span > pd.Timedelta(hours=1)).mean() > 0.9


def test_camouflage_monotonically_reduces_amount_separability():
    """KS(log amount | pattern vs normal) must fall as camouflage rises (§14.4)."""
    ks = []
    for c in (0.0, 0.25, 0.5, 0.75, 1.0):
        tx = generate(dataclasses.replace(SMALL, camouflage=c)).transactions
        la = np.log(tx["amount"].to_numpy())
        pos = tx["is_fraud"].to_numpy() == 1
        ks.append(ks_2samp(la[pos], la[~pos]).statistic)
    assert all(a > b for a, b in zip(ks, ks[1:], strict=False)), ks
    assert ks[0] > 0.6 and ks[-1] < 0.15, ks


def test_camouflage_sweep_keeps_the_same_rings():
    """Paired sweep (§14.5): only the blending changes, not which rings exist."""
    def rings(cfg: GeneratorConfig) -> set[frozenset[str]]:
        m = generate(cfg).members
        return {frozenset(g) for _, g in m.groupby("pattern_id")["account"]}

    a = rings(dataclasses.replace(SMALL, camouflage=0.0))
    b = rings(dataclasses.replace(SMALL, camouflage=1.0))
    # Not exactly equal: timing changes with camouflage, so a few burn-in
    # patterns reach into the period at one level and not at the other.
    assert len(a & b) / len(a | b) > 0.9


def test_injection_knobs_leave_normal_background_unchanged():
    a = generate(dataclasses.replace(SMALL, camouflage=0.0, prevalence=0.005)).transactions
    b = generate(dataclasses.replace(SMALL, camouflage=0.9, prevalence=0.02)).transactions
    cols = ["src", "dst", "amount", "timestamp", "payment_format"]
    na = a.loc[a["is_fraud"] == 0, cols].reset_index(drop=True)
    nb = b.loc[b["is_fraud"] == 0, cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(na, nb)


NO_STACK = tuple(t for t in IBM_TYPOLOGIES if t != "STACK")
DRIFTS = {
    "amount_shift": {"drift_magnitude": 3.0},
    "velocity_shift": {"drift_magnitude": 2.0},
    "new_typology": {"drift_magnitude": 0.6, "typologies": NO_STACK},
}


@pytest.mark.parametrize("kind", sorted(DRIFTS))
def test_drift_leaves_pre_onset_data_identical(kind):
    """LEAKAGE GUARD for the drift experiment: nothing before onset may change.

    If drift leaked backwards, a detector could "detect" it before it happened
    and every latency number would be wrong.
    """
    extra = DRIFTS[kind]
    base = dataclasses.replace(SMALL, typologies=extra.get("typologies", SMALL.typologies))
    plain = generate(base)
    drifted = generate(
        dataclasses.replace(
            base, drift_onset_day=7, drift_kind=kind, drift_magnitude=extra["drift_magnitude"]
        )
    )
    onset = drifted.config.drift_onset
    assert onset == SMALL.start + pd.Timedelta(days=7)
    for attr in ("transactions", "patterns"):
        a, b = getattr(plain, attr), getattr(drifted, attr)
        pd.testing.assert_frame_equal(a[a["timestamp"] < onset], b[b["timestamp"] < onset])
    assert not plain.transactions.equals(drifted.transactions)


def _normal_after(d, onset):
    tx = d.transactions
    return tx[(tx["timestamp"] >= onset) & (tx["is_fraud"] == 0)]


def test_amount_shift_scales_normal_amounts_after_onset():
    plain = generate(SMALL)
    drifted = generate(dataclasses.replace(SMALL, drift_onset_day=7, drift_magnitude=3.0))
    onset = drifted.config.drift_onset
    ratio = _normal_after(drifted, onset)["amount"].median() / _normal_after(plain, onset)["amount"].median()
    assert ratio == pytest.approx(3.0, rel=0.02)


def test_velocity_shift_scales_normal_volume_after_onset():
    plain = generate(SMALL)
    drifted = generate(
        dataclasses.replace(SMALL, drift_onset_day=7, drift_kind="velocity_shift", drift_magnitude=2.0)
    )
    onset = drifted.config.drift_onset
    ratio = len(_normal_after(drifted, onset)) / len(_normal_after(plain, onset))
    assert ratio == pytest.approx(2.0, rel=0.05)
    assert drifted.prevalence == pytest.approx(SMALL.prevalence, rel=0.25)


def test_new_typology_appears_only_after_onset():
    cfg = dataclasses.replace(
        SMALL, typologies=NO_STACK, drift_onset_day=6, drift_kind="new_typology", drift_magnitude=0.6
    )
    assert cfg.resolved_drift_typology == "STACK"
    d = generate(cfg)
    stack = d.patterns[d.patterns["typology"] == "STACK"]
    assert len(stack) > 0
    assert (stack["timestamp"] >= cfg.drift_onset).all()


def test_rings_have_early_warning_shape(data):
    rings = data.rings()
    assert len(rings) == data.patterns["pattern_id"].nunique()
    ring = next(iter(rings.values()))
    assert {"members", "t_start", "t_end"} <= set(ring)
    assert ring["t_start"] <= ring["t_end"]


def test_save_writes_hashed_manifest_and_round_trips(data, tmp_path):
    manifest_path = save_synthetic(data, tmp_path, record_code_state=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["n_rows"] == len(data.transactions)
    assert manifest["prevalence"] == pytest.approx(data.prevalence)
    assert manifest["config"]["seed"] == SMALL.seed
    for entry in manifest["files"].values():
        assert file_sha256(tmp_path / entry["file"]) == entry["sha256"]

    loaded = load_synthetic(tmp_path)
    pd.testing.assert_frame_equal(loaded.transactions, data.transactions)
    pd.testing.assert_frame_equal(loaded.patterns, data.patterns)
    assert loaded.config == data.config


def test_load_refuses_tampered_files(data, tmp_path):
    save_synthetic(data, tmp_path, record_code_state=False)
    data.transactions.iloc[:-1].to_parquet(tmp_path / "transactions.parquet", index=False)
    with pytest.raises(ValueError, match="does not match manifest"):
        load_synthetic(tmp_path)


def test_default_config_loads():
    cfg = load_generator_config(CONFIG)
    assert cfg.n_accounts == 5000 and cfg.drift_onset_day == 14
    assert cfg.ring_size == (3, 12) and cfg.drift_kind == "amount_shift"


@pytest.mark.parametrize(
    "bad",
    [
        {"camouflage": 1.5},
        {"prevalence": 0.0},
        {"ring_size": (2, 5)},
        {"typologies": ("SMURFING",)},
        {"drift_onset_day": 0},
        {"drift_onset_day": 5, "drift_kind": "new_typology"},  # all typologies already used
    ],
)
def test_invalid_config_is_rejected(bad):
    with pytest.raises(ValueError):
        dataclasses.replace(SMALL, **bad)


def test_unknown_yaml_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("generator:\n  seed: 0\n  n_acounts: 10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_generator_config(path)
