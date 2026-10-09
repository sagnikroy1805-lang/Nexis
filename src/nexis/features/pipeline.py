"""Feature table assembly and caching.

The full behavioural feature set takes minutes on 5M rows, so it is computed
once per (processed data, windows, feature code version) and cached. The cache
key includes the processed data's content hash: a rebuilt dataset can never be
paired with stale features.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from nexis.data.ibm_aml import IbmAmlConfig
from nexis.features.behavioural import behavioural_features
from nexis.features.tabular import TABULAR_COLUMNS, tabular_features

# Bump when feature code changes meaning; it invalidates every cache.
FEATURE_VERSION = "behavioural-v1"


def _cache_key(cfg: IbmAmlConfig) -> str:
    manifest = json.loads(cfg.manifest_path.read_text(encoding="utf-8"))
    payload = json.dumps(
        [manifest["processed_sha256"], list(cfg.feature_windows), FEATURE_VERSION]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def feature_table(cfg: IbmAmlConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Tabular + behavioural features aligned to df (the processed transactions)."""
    path = cfg.processed_path.with_name(
        f"{cfg.processed_path.stem}_features_{_cache_key(cfg)}.parquet"
    )
    if path.exists():
        feats = pd.read_parquet(path)
        if len(feats) == len(df):
            feats.index = df.index
            return feats
    feats = pd.concat(
        [tabular_features(df), behavioural_features(df, cfg.feature_windows)], axis=1
    )
    feats.to_parquet(path, index=False)
    return feats


def behavioural_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c not in TABULAR_COLUMNS]


GRAPH_VERSION = "graph-v1"


def graph_feature_table(cfg: IbmAmlConfig, df: pd.DataFrame, delta: str) -> pd.DataFrame:
    """Rung-5 structural features (snapshot graphs, t_cutoff = period start)."""
    from nexis.graphs.homogeneous import graph_features

    manifest = json.loads(cfg.manifest_path.read_text(encoding="utf-8"))
    key = hashlib.sha256(
        json.dumps([manifest["processed_sha256"], delta, GRAPH_VERSION]).encode()
    ).hexdigest()[:16]
    path = cfg.processed_path.with_name(f"{cfg.processed_path.stem}_graphfeat_{key}.parquet")
    if path.exists():
        feats = pd.read_parquet(path)
        if len(feats) == len(df):
            feats.index = df.index
            return feats
    feats = graph_features(df, delta=delta)
    feats.to_parquet(path, index=False)
    return feats
