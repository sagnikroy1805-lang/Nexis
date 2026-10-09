"""Elliptic++ (transactions component) adapter -- the tier-1 credibility anchor.

Implements Concept Mastery §3.6 / §9.4 (split by time step, never shuffled) and
the dataset handling required by the feasibility study (§3.3):
  - labels: illicit -> 1, licit -> 0, unknown -> missing. Unknown is NEVER
    treated as licit; unknown nodes are excluded from losses and metrics but stay
    in the graph (they carry structure).
  - edges in Elliptic connect transactions within the same time step, so each
    step is its own graph and a step-based split cannot pass messages across
    folds. `assert_edges_within_steps` checks that instead of trusting it.

Source: Elmougy & Liu, "Demystifying Fraudulent Transactions and Illicit Nodes in
the Bitcoin Network for Financial Forensics", KDD 2023
(github.com/git-disl/EllipticPlusPlus). Files expected in data/raw/elliptic/:
    txs_features.csv  txs_classes.csv  txs_edgelist.csv

Label semantics (rule 5): "illicit" is Elliptic's annotation of a transaction,
not a finding about any person.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nexis.data.ibm_aml import DataValidationError, file_sha256

FILES = ("txs_features.csv", "txs_classes.csv", "txs_edgelist.csv")
# Elliptic's public class coding: 1 = illicit, 2 = licit, 3 / "unknown" = unlabelled.
_LABEL_MAP = {"1": 1.0, "2": 0.0, "3": np.nan, "unknown": np.nan}


@dataclass
class EllipticData:
    nodes: pd.DataFrame  # tx_id, time_step, label (1/0/NaN), feature columns
    edges: pd.DataFrame  # src, dst as tx_id
    feature_cols: list[str]
    local_cols: list[str]  # the transaction's own features (no neighbour aggregates)


def _find(cols: list[str], *needles: str) -> str:
    for c in cols:
        lc = c.lower().replace(" ", "").replace("_", "")
        if any(n in lc for n in needles):
            return c
    raise DataValidationError(f"no column matching {needles} in {cols[:8]}...")


def load_elliptic(raw_dir: str | Path) -> EllipticData:
    """Read the three transaction files into one validated node table + edges."""
    raw_dir = Path(raw_dir)
    missing = [f for f in FILES if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"missing Elliptic++ files in {raw_dir}: {missing}")

    feats = pd.read_csv(raw_dir / "txs_features.csv")
    classes = pd.read_csv(raw_dir / "txs_classes.csv", dtype=str)
    edges = pd.read_csv(raw_dir / "txs_edgelist.csv")

    fcols = list(feats.columns)
    id_col = fcols[0]
    step_col = _find(fcols, "timestep", "time")
    cid = classes.columns[0]
    ccls = _find(list(classes.columns), "class", "label")
    classes = classes.rename(columns={cid: "tx_id", ccls: "class"})
    classes["tx_id"] = classes["tx_id"].astype(np.int64)

    nodes = feats.rename(columns={id_col: "tx_id", step_col: "time_step"})
    nodes = nodes.merge(classes[["tx_id", "class"]], on="tx_id", how="left", validate="one_to_one").copy()
    unknown_codes = set(nodes["class"].dropna().str.strip().str.lower()) - set(_LABEL_MAP)
    if unknown_codes:
        raise DataValidationError(f"unexpected class codes {sorted(unknown_codes)}")
    nodes["label"] = nodes["class"].str.strip().str.lower().map(_LABEL_MAP)
    nodes = nodes.drop(columns=["class"]).sort_values(["time_step", "tx_id"], kind="stable")
    nodes = nodes.reset_index(drop=True)

    feature_cols = [c for c in nodes.columns if c not in ("tx_id", "time_step", "label")]
    local_cols = [c for c in feature_cols if "aggregate" not in c.lower()]

    e0, e1 = edges.columns[:2]
    edges = edges.rename(columns={e0: "src", e1: "dst"})[["src", "dst"]]

    data = EllipticData(nodes, edges, feature_cols, local_cols)
    validate_elliptic(data)
    return data


def assert_edges_within_steps(data: EllipticData) -> None:
    """Every edge joins two transactions of the same time step.

    If this ever failed, a per-step graph would hide cross-step edges and a
    full graph would leak across the split; either way the protocol breaks.
    """
    step = data.nodes.set_index("tx_id")["time_step"]
    s, d = step.reindex(data.edges["src"]).to_numpy(), step.reindex(data.edges["dst"]).to_numpy()
    bad = ~np.isnan(s) & ~np.isnan(d) & (s != d)
    if bad.any():
        raise DataValidationError(f"{int(bad.sum())} edges cross time steps")


def validate_elliptic(data: EllipticData) -> None:
    n = data.nodes
    problems = []
    if n["tx_id"].duplicated().any():
        problems.append("duplicate tx_id")
    if n["time_step"].isna().any():
        problems.append("missing time_step")
    known = n["label"].notna()
    if known.sum() == 0 or n.loc[known, "label"].sum() == 0:
        problems.append("no labelled illicit transactions")
    ids = set(n["tx_id"])
    dangling = (~data.edges["src"].isin(ids) | ~data.edges["dst"].isin(ids)).sum()
    if dangling:
        problems.append(f"{int(dangling)} edges reference unknown transactions")
    if problems:
        raise DataValidationError("; ".join(problems))
    assert_edges_within_steps(data)


@dataclass(frozen=True)
class StepSplit:
    """Time-step folds. Defaults follow the common Elliptic protocol (train <= 34,
    test >= 35), with steps 30-34 held out of training as validation."""

    train_end: int = 29
    val_end: int = 34

    def fold(self, step: pd.Series) -> pd.Series:
        return pd.Series(
            np.where(step <= self.train_end, "train", np.where(step <= self.val_end, "val", "test")),
            index=step.index,
        )


def manifest(raw_dir: str | Path, data: EllipticData) -> dict:
    """Content hashes + label counts, for results/ provenance."""
    raw_dir = Path(raw_dir)
    lab = data.nodes["label"]
    return {
        "files": {f: file_sha256(raw_dir / f) for f in FILES},
        "n_nodes": int(len(data.nodes)),
        "n_edges": int(len(data.edges)),
        "n_illicit": int((lab == 1).sum()),
        "n_licit": int((lab == 0).sum()),
        "n_unknown": int(lab.isna().sum()),
        "n_steps": int(data.nodes["time_step"].nunique()),
        "n_features": len(data.feature_cols),
        "n_local_features": len(data.local_cols),
    }


def write_manifest(raw_dir: str | Path, data: EllipticData, out: str | Path) -> None:
    Path(out).write_text(json.dumps(manifest(raw_dir, data), indent=2), encoding="utf-8")
