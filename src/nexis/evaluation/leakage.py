"""Leakage detection.

Implements Concept Mastery §9.4.

Do not rely on being careful. Assert.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def assert_no_future_edges(data, t_cutoff) -> None:
    """Assert no edge in a PyG graph carries a timestamp after the cutoff.

    Graph-native leakage (§9.4 form 4) is invisible in the label split: the labels
    can be correctly separated while message passing still aggregates over future
    edges. Run this on every graph construction, in CI.

    Requires edge stores to carry an `edge_time` attribute in epoch seconds.
    """
    cutoff = int(pd.Timestamp(t_cutoff).timestamp())
    checked = 0
    for store in data.edge_stores:
        if hasattr(store, "edge_time") and store.edge_time is not None:
            latest = int(store.edge_time.max())
            assert latest <= cutoff, (
                f"LEAK: edge at epoch {latest} exceeds cutoff {cutoff} "
                f"({pd.Timestamp(latest, unit='s')} > {pd.Timestamp(cutoff, unit='s')})"
            )
            checked += 1
    assert checked > 0, (
        "no edge store carried edge_time; the leakage guard did not actually run"
    )


def leakage_canary(
    fit_predict: Callable[[pd.DataFrame, np.ndarray], np.ndarray],
    df: pd.DataFrame,
    y: np.ndarray,
    time_col: str = "timestamp",
    train_frac: float = 0.6,
    n_shuffles: int = 3,
    seed: int = 0,
) -> dict[str, float]:
    """Shuffle training labels; a leak-free pipeline must score at chance.

    The strongest test available, and the only one that catches leaks no code
    review will. If the pipeline still predicts above chance after the training
    labels have been destroyed, information is reaching the model through a
    channel other than the labels -- a global statistic, an unmasked edge, a
    fitted scaler.

    Args:
        fit_predict: takes (df, y) and returns scores for every row of df.
        df: full dataset, chronologically sortable.
        y: true labels.

    Returns:
        {"observed": mean PR-AUC on the test period, "chance": prevalence,
         "ratio": observed / chance}. Ratio near 1.0 is healthy; above ~3 is a
        strong signal of leakage.
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    t = pd.to_datetime(df[time_col])
    cutoff = t.quantile(train_frac)
    is_train = (t <= cutoff).to_numpy()

    scores = []
    for s in range(n_shuffles):
        rng = np.random.default_rng(seed + s)
        y_shuffled = y.copy()
        y_shuffled[is_train] = rng.permutation(y_shuffled[is_train])
        preds = fit_predict(df, y_shuffled)
        scores.append(average_precision_score(y[~is_train], preds[~is_train]))

    observed = float(np.mean(scores))
    chance = float(y[~is_train].mean())
    return {
        "observed": observed,
        "chance": chance,
        "ratio": observed / max(chance, 1e-12),
    }
