"""Elliptic++ adapter on a hand-built miniature in the published file layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nexis.data.elliptic import StepSplit, load_elliptic
from nexis.data.ibm_aml import DataValidationError


def _write(tmp: Path, edges: list[tuple[int, int]] | None = None) -> Path:
    feats = pd.DataFrame(
        {
            "txId": [10, 11, 12, 20, 21, 22],
            "Time step": [1, 1, 1, 2, 2, 2],
            "Local_feature_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "Aggregate_feature_1": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            "in_txs_degree": [1, 2, 0, 1, 1, 0],
        }
    )
    feats.to_csv(tmp / "txs_features.csv", index=False)
    pd.DataFrame({"txId": [10, 11, 12, 20, 21, 22], "class": ["1", "2", "3", "2", "unknown", "1"]}).to_csv(
        tmp / "txs_classes.csv", index=False
    )
    pd.DataFrame(edges or [(10, 11), (11, 12), (20, 21)], columns=["txId1", "txId2"]).to_csv(
        tmp / "txs_edgelist.csv", index=False
    )
    return tmp


def test_labels_unknown_is_missing_not_licit(tmp_path):
    d = load_elliptic(_write(tmp_path))
    lab = d.nodes.set_index("tx_id")["label"]
    assert lab[10] == 1 and lab[11] == 0 and lab[22] == 1
    assert np.isnan(lab[12]) and np.isnan(lab[21]), "unknown must never become licit (0)"


def test_local_features_exclude_neighbour_aggregates(tmp_path):
    d = load_elliptic(_write(tmp_path))
    assert "Aggregate_feature_1" in d.feature_cols
    assert "Aggregate_feature_1" not in d.local_cols
    assert "Local_feature_1" in d.local_cols


def test_edges_across_time_steps_are_rejected(tmp_path):
    with pytest.raises(DataValidationError, match="cross time steps"):
        load_elliptic(_write(tmp_path, edges=[(10, 20)]))


def test_step_split_is_ordered():
    f = StepSplit(train_end=29, val_end=34).fold(pd.Series([1, 29, 30, 34, 35, 49]))
    assert f.tolist() == ["train", "train", "val", "val", "test", "test"]
