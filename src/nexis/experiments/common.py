"""Shared experiment setup: one place that loads data, features and the split.

Every experiment script goes through `prepare`, so all rungs of the ladder see
the same rows, the same features and the same folds -- the precondition for a
results table whose rows are comparable (Concept Mastery §22.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from nexis.data.ibm_aml import IbmAmlConfig, load_config, load_processed
from nexis.evaluation.splits import (
    SplitResult,
    assert_no_id_overlap,
    assert_temporal_integrity,
)
from nexis.features.pipeline import (
    behavioural_columns,
    feature_table,
    graph_feature_table,
)
from nexis.features.tabular import TABULAR_COLUMNS

LABEL = "is_fraud"
ID_COLS = ["tx_id", "timestamp", LABEL]


@dataclass
class Prepared:
    cfg: IbmAmlConfig
    raw: dict[str, Any]  # the YAML, for sections the dataclass does not model
    df: pd.DataFrame  # processed transactions
    frame: pd.DataFrame  # ids + label + every feature column, aligned to df
    split: SplitResult
    tabular_cols: list[str]
    behavioural_cols: list[str]  # tabular + behavioural (the rung-4 input)
    graph_cols: list[str]  # rung-5 structural columns only

    @property
    def with_graph_cols(self) -> list[str]:
        return self.behavioural_cols + self.graph_cols

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw["evaluation"]


SCORES_DIR = Path("results") / "scores"


def save_scores(
    model: str, seed: int, folds: dict[str, tuple[pd.DataFrame, Any]]
) -> Path:
    """Persist per-transaction scores so fusion, ring detection and the API can
    reuse a trained model's output instead of retraining it.

    folds: {"val": (val_df, scores), "test": (test_df, scores)}.
    """
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    parts = [
        pd.DataFrame({"tx_id": f["tx_id"].to_numpy(), "fold": name, "score": s})
        for name, (f, s) in folds.items()
    ]
    path = SCORES_DIR / f"{model}_seed{seed}.parquet"
    pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
    return path


def load_scores(model: str, seed: int) -> pd.DataFrame:
    return pd.read_parquet(SCORES_DIR / f"{model}_seed{seed}.parquet")


def prepare(config_path: str | Path, with_graph: bool = True) -> Prepared:
    cfg = load_config(config_path)
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    df = load_processed(cfg)
    feats = feature_table(cfg, df)
    parts = [df[ID_COLS], feats]
    graph_cols: list[str] = []
    if with_graph:
        gf = graph_feature_table(cfg, df, raw["graph"]["snapshot"])
        graph_cols = list(gf.columns)
        parts.append(gf)
    frame = pd.concat(parts, axis=1)
    split = cfg.split.apply(frame)
    assert_temporal_integrity(split)  # loud failure beats silent leakage
    assert_no_id_overlap(split.train, split.test)
    assert_no_id_overlap(split.train, split.val)
    tab = list(TABULAR_COLUMNS)
    return Prepared(
        cfg=cfg,
        raw=raw,
        df=df,
        frame=frame,
        split=split,
        tabular_cols=tab,
        behavioural_cols=tab + behavioural_columns(feats),
        graph_cols=graph_cols,
    )
