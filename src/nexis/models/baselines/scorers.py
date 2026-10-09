"""Baseline scorers for rungs 1-4 of the modelling ladder.

Implements Concept Mastery §3.1-3.4 (logistic regression, trees, random forest,
XGBoost), §5.2 (Isolation Forest) and the "rules first" baseline of §22.1.

Every scorer satisfies nexis.evaluation.harness.Scorer: fit(train, val) and
score(df) -> risk scores. None of them computes a metric; the harness does.
Frames passed in carry the feature columns plus `is_fraud`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

LABEL = "is_fraud"


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    """Counts and sums span 10 orders of magnitude; linear models need them compressed."""
    return np.sign(x) * np.log1p(np.abs(x))


def subsample_negatives(df: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    """Keep every positive and a seeded fraction of negatives.

    Used only to make RF and logistic regression tractable on 3M training rows.
    Ranking metrics are invariant to the resulting shift in base rate; the seed
    is part of the run, so the sampling variance is inside the reported std.
    """
    rng = np.random.default_rng(seed)
    keep = (df[LABEL].to_numpy() == 1) | (rng.random(len(df)) < rate)
    return df.loc[keep]


@dataclass
class RandomScorer:
    """Rung 1: chance. PR-AUC should equal prevalence; anything else is a bug."""

    seed: int = 0

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> RandomScorer:
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return np.random.default_rng(self.seed).random(len(df))


@dataclass
class AmountScorer:
    """Rung 1: rank by amount alone -- the 'majority-class' analogue for ranking."""

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> AmountScorer:
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return df["log_amount"].to_numpy(dtype=float)


@dataclass
class RulesScorer:
    """Rung 2: the thresholds an analyst would write, counted.

    Thresholds are quantiles of the TRAINING fold, fitted without labels, so the
    rules are tuned the way a bank would tune them -- on volume, not on outcomes.
    Score = number of rules fired, tie-broken by amount.
    """

    quantile: float = 0.99
    thresholds: dict[str, float] = field(default_factory=dict)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> RulesScorer:
        q = self.quantile
        self.thresholds = {
            "large_amount": float(train["log_amount"].quantile(q)),
            "burst_1h": float(train["src_out_cnt_1h"].quantile(q)),
            "fan_in_24h": float(train["dst_in_cnt_24h"].quantile(q)),
        }
        return self

    def rule_hits(self, df: pd.DataFrame) -> pd.DataFrame:
        th = self.thresholds
        return pd.DataFrame(
            {
                "large_amount": df["log_amount"] > th["large_amount"],
                "far_above_own_history": df["amt_z_src"].fillna(0) > 3,
                "burst_1h": df["src_out_cnt_1h"] > th["burst_1h"],
                "fan_in_24h": df["dst_in_cnt_24h"] > th["fan_in_24h"],
                "pass_through_1h": (df["src_secs_since_in"] <= 3600)
                & df["amt_over_src_in_24h"].between(0.5, 1.5),
                "new_cross_bank_counterparty": (df["counterparty_is_new"] == 1)
                & (df["is_cross_bank"] == 1),
                "cross_currency": df["is_cross_currency"] == 1,
            },
            index=df.index,
        )

    def score(self, df: pd.DataFrame) -> np.ndarray:
        hits = self.rule_hits(df).sum(axis=1).to_numpy(dtype=float)
        return hits + 1e-3 * df["log_amount"].to_numpy(dtype=float) / 30.0


@dataclass
class SklearnScorer:
    """Rung 3/4 wrapper for scikit-learn classifiers on a fixed feature list."""

    kind: str
    features: Sequence[str]
    seed: int = 0
    neg_rate: float = 0.1
    params: dict[str, Any] = field(default_factory=dict)
    model: Any = None

    def _build(self) -> Any:
        if self.kind == "logistic_regression":
            # Imputer and scaler are fitted inside the pipeline on the TRAINING
            # fold only (rule 1: never fit preprocessing on val/test).
            return Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("log", FunctionTransformer(_signed_log1p)),
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=3000,
                            class_weight="balanced",
                            C=self.params.get("C", 1.0),
                            random_state=self.seed,
                        ),
                    ),
                ]
            )
        if self.kind == "random_forest":
            return Pipeline(
                [
                    ("impute", SimpleImputer(strategy="constant", fill_value=-1.0)),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=self.params.get("n_estimators", 300),
                            min_samples_leaf=self.params.get("min_samples_leaf", 5),
                            max_features="sqrt",
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=self.seed,
                        ),
                    ),
                ]
            )
        raise ValueError(f"unknown kind {self.kind}")

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> SklearnScorer:
        sub = subsample_negatives(train, self.neg_rate, self.seed)
        self.model = self._build().fit(sub[list(self.features)], sub[LABEL])
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[list(self.features)])[:, 1]


@lru_cache(maxsize=1)
def xgb_device() -> str:
    """'cuda' if this XGBoost build can train on the GPU, else 'cpu'.

    NEXIS_XGB_DEVICE=cpu forces the CPU, e.g. while a GNN run holds the GPU.
    """
    import os

    if os.environ.get("NEXIS_XGB_DEVICE"):
        return os.environ["NEXIS_XGB_DEVICE"]
    try:
        import xgboost as xgb

        xgb.XGBClassifier(device="cuda", n_estimators=1).fit(
            np.zeros((4, 1)), np.array([0, 1, 0, 1])
        )
        return "cuda"
    except Exception:
        return "cpu"


@dataclass
class XGBScorer:
    """Rung 3/4 gradient boosting. Early-stopped on the validation fold by aucpr."""

    features: Sequence[str]
    seed: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    # 300, not 100: validation PR-AUC with ~500 positives is noisy, and a
    # 100-round window let one seed stop at tree 3 (test PR-AUC 0.35 vs 0.52 for
    # the same configuration trained to convergence). Recorded in docs/results.md.
    early_stopping_rounds: int = 300
    model: Any = None
    best_iteration: int | None = None

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> XGBScorer:
        import xgboost as xgb

        y = train[LABEL].to_numpy()
        spw = float((y == 0).sum() / max((y == 1).sum(), 1))
        p = {
            "n_estimators": 5000,  # a cap, not a target: early stopping decides
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            **self.params,
        }
        self.model = xgb.XGBClassifier(
            **p,
            scale_pos_weight=spw,
            eval_metric="aucpr",  # rule 2: never auc, error or logloss
            early_stopping_rounds=self.early_stopping_rounds,
            tree_method="hist",
            device=xgb_device(),
            random_state=self.seed,
        )
        cols = list(self.features)
        self.model.fit(
            train[cols], y, eval_set=[(val[cols], val[LABEL].to_numpy())], verbose=False
        )
        self.best_iteration = int(self.model.best_iteration)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[list(self.features)])[:, 1]


@dataclass
class IsolationForestScorer:
    """Unsupervised anomaly baseline (§5.2). Fitted without labels.

    Matters because most real data is unlabelled; it shows how far pure
    unusualness gets without any supervision.
    """

    features: Sequence[str]
    seed: int = 0
    model: Any = None

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> IsolationForestScorer:
        sample = train.sample(n=min(len(train), 500_000), random_state=self.seed)
        self.model = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("log", FunctionTransformer(_signed_log1p)),
                (
                    "iso",
                    IsolationForest(
                        n_estimators=300, random_state=self.seed, n_jobs=-1
                    ),
                ),
            ]
        ).fit(sample[list(self.features)])
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(df[list(self.features)])


def tune_xgb(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: Sequence[str],
    n_trials: int,
    seed: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Random search on the validation fold, optimising PR-AUC.

    The trial count is returned with the log and recorded in the manifest: every
    later model gets a comparable search budget, or the comparison is unfair
    (§22.2).
    """
    rng = np.random.default_rng(seed)
    trials: list[dict[str, Any]] = []
    for _ in range(n_trials):
        params = {
            "max_depth": int(rng.choice([4, 6, 8, 10])),
            "learning_rate": float(rng.choice([0.03, 0.05, 0.1])),
            "min_child_weight": int(rng.choice([1, 5, 20])),
            "subsample": float(rng.choice([0.7, 0.85, 1.0])),
            "colsample_bytree": float(rng.choice([0.6, 0.8, 1.0])),
            "reg_lambda": float(rng.choice([0.5, 1.0, 5.0])),
        }
        m = XGBScorer(features, seed=seed, params=params).fit(train, val)
        ap = float(average_precision_score(val[LABEL], m.score(val)))
        trials.append({**params, "val_pr_auc": ap, "best_iteration": m.best_iteration})
    best = max(trials, key=lambda r: r["val_pr_auc"])
    best_params = {k: v for k, v in best.items() if k not in ("val_pr_auc", "best_iteration")}
    return best_params, trials
