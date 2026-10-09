"""Snapshot GNNs: they train, they score, and they cannot see the future."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from nexis.evaluation.splits import temporal_split  # noqa: E402
from nexis.features.behavioural import behavioural_features  # noqa: E402
from nexis.graphs.snapshots import build_snapshots  # noqa: E402
from nexis.models.gnn.train import GNNScorer  # noqa: E402

FORMATS = ["ACH", "Cheque", "Wire"]


def _transactions(n: int, seed: int, start: str = "2026-01-01", days: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    accts = [f"{b:03d}_{i:05d}" for b in range(3) for i in range(40)]
    src = rng.choice(accts, n)
    dst = rng.choice(accts, n)
    ts = pd.Timestamp(start) + pd.to_timedelta(np.sort(rng.integers(0, days * 1440, n)), unit="min")
    df = pd.DataFrame(
        {
            "tx_id": [f"T{seed}:{i:06d}" for i in range(n)],
            "timestamp": ts,
            "src": src,
            "dst": dst,
            "src_bank": pd.Categorical([s[:3] for s in src]),
            "dst_bank": pd.Categorical([d[:3] for d in dst]),
            "amount": rng.lognormal(6, 1, n),
            "payment_format": pd.Categorical(rng.choice(FORMATS, n)),
            "is_fraud": (rng.random(n) < 0.05).astype("int8"),
        }
    )
    # Make the label learnable: a fan-in hub receives the positives.
    hub = df["is_fraud"] == 1
    df.loc[hub, "dst"] = "000_99999"
    return df


def _frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feats = behavioural_features(df, windows=("1h", "24h"))
    feats["log_amount"] = np.log1p(df["amount"]).astype(np.float32)
    return pd.concat([df[["tx_id", "timestamp", "is_fraud"]], feats], axis=1), list(feats.columns)


@pytest.fixture(scope="module")
def setup():
    df = _transactions(3000, seed=0)
    frame, cols = _frame(df)
    split = temporal_split(frame, train_frac=0.5, val_frac=0.25, embargo="24h")
    snaps = build_snapshots(df, delta="24h")
    return df, frame, cols, split, snaps


@pytest.mark.parametrize("kind", ["homogeneous", "heterogeneous", "temporal"])
def test_gnn_trains_and_scores(setup, kind):
    _, frame, cols, split, snaps = setup
    m = GNNScorer(kind, snaps, frame, cols, seed=0, hidden=16, layers=2, epochs=3, patience=3)
    m.fit(split.train, split.val)
    s = m.score(split.test)
    assert s.shape == (len(split.test),)
    assert np.isfinite(s).all() and (s >= 0).all() and (s <= 1).all()
    assert len(m.history) >= 1


@pytest.mark.parametrize("kind", ["homogeneous", "heterogeneous", "temporal"])
def test_future_transactions_do_not_change_earlier_scores(setup, kind):
    """Graph-native leakage guard (§9.4 form 4), checked end to end.

    Train once. Then append a burst of new transactions AFTER the last scored
    period -- including brand-new accounts -- and rebuild the snapshots. With
    the same weights, every earlier transaction must get the same score.
    """
    df, frame, cols, split, snaps = setup
    m = GNNScorer(kind, snaps, frame, cols, seed=0, hidden=16, layers=2, epochs=2, patience=2)
    m.fit(split.train, split.val)
    early = split.val
    before = m.score(early)

    late = _transactions(800, seed=1, start=str(df["timestamp"].max() + pd.Timedelta("1D")), days=2)
    late["dst"] = late["dst"].str.replace("_", "_N", regex=False)  # new accounts too
    late["dst_bank"] = pd.Categorical(late["dst"].str[:3])
    df2 = pd.concat([df, late], ignore_index=True)
    frame2, _ = _frame(df2)
    m2 = copy.copy(m)
    m2.snapshots = build_snapshots(df2, delta="24h")
    m2.frame = frame2
    m2._edge_x = torch.as_tensor(m._prep.transform(frame2), device=m._edge_x.device)
    if kind != "homogeneous":
        assert set(m2.snapshots.formats) == set(snaps.formats)
    after = m2.score(early)
    np.testing.assert_allclose(before, after, rtol=1e-4, atol=1e-5)
