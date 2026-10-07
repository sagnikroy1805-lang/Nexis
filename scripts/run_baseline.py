"""First runnable experiment: tabular baselines through the harness.

This is deliberately the first thing that works end to end. It proves the spine
of the project -- split, features, model, metrics, multi-seed aggregation,
persisted results -- before any graph code exists.

Usage:
    python scripts/run_baseline.py --data data/synthetic/transactions.parquet
    python scripts/run_baseline.py --demo        # generates toy data, no download needed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nexis.evaluation.harness import aggregate_over_seeds, format_table
from nexis.evaluation.metrics import evaluate
from nexis.evaluation.seeds import SEEDS, set_all_seeds
from nexis.evaluation.splits import assert_temporal_integrity, temporal_split
from nexis.features.velocity import (
    personal_baseline_features,
    rolling_velocity,
)

FEATURE_COLS = [
    "log_amount",
    "hour",
    "day_of_week",
    "cnt_5min",
    "cnt_1h",
    "cnt_24h",
    "cnt_7D",
    "sum_1h",
    "sum_24h",
    "max_24h",
    "burst_1h",
    "amt_z_personal",
    "amt_vs_personal_max",
    "amt_log_ratio_median",
    "has_history",
    "counterparty_is_new",
]


def make_demo_data(n: int = 60_000, seed: int = 0) -> pd.DataFrame:
    """Toy transactions with injected burst-style fraud.

    Only for verifying the pipeline runs. Real experiments use IBM AML,
    Elliptic++ or the project's own generator.
    """
    rng = np.random.default_rng(seed)
    accounts = [f"ACC_{i:05d}" for i in range(2_000)]
    df = pd.DataFrame(
        {
            "tx_id": [f"TX{i:07d}" for i in range(n)],
            "src": rng.choice(accounts, n),
            "dst": rng.choice(accounts, n),
            "amount": rng.lognormal(7.2, 1.1, n).round(2),
            "timestamp": pd.Timestamp("2026-01-01")
            + pd.to_timedelta(np.sort(rng.uniform(0, 90 * 24 * 3600, n)), unit="s"),
            "is_fraud": 0,
        }
    )

    # Inject bursts: a handful of accounts transacting rapidly at elevated amounts.
    for _ in range(120):
        actor = rng.choice(accounts)
        start = rng.integers(0, n - 60)
        idx = df.index[start : start + rng.integers(8, 25)]
        df.loc[idx, "src"] = actor
        df.loc[idx, "amount"] = (df.loc[idx, "amount"] * rng.uniform(2.5, 5.0)).round(2)
        df.loc[idx, "is_fraud"] = 1

    return df.sort_values("timestamp").reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the feature matrix. Every guard here is load-bearing."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    base = pd.DataFrame(index=df.index)
    base["log_amount"] = np.log1p(df["amount"])
    base["hour"] = df["timestamp"].dt.hour
    base["day_of_week"] = df["timestamp"].dt.dayofweek

    vel = rolling_velocity(df)
    per = personal_baseline_features(df)
    out = pd.concat([base, vel, per], axis=1)

    for col in FEATURE_COLS:
        if col not in out.columns:
            out[col] = 0.0
    return out[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--alert-budget", type=int, default=500)
    args = parser.parse_args()

    if args.demo or args.data is None:
        print("Generating demo data (use --data for a real dataset)...")
        df = make_demo_data()
    else:
        df = pd.read_parquet(args.data)

    print(f"{len(df):,} transactions, prevalence {df['is_fraud'].mean():.4%}")

    split = temporal_split(df, embargo="1D")
    assert_temporal_integrity(split)          # loud failure beats silent leakage
    print("split:", split.summary())

    X_tr, y_tr = build_features(split.train), split.train["is_fraud"].to_numpy()
    X_te, y_te = build_features(split.test), split.test["is_fraud"].to_numpy()

    models = {
        "logistic_regression": lambda seed: Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", random_state=seed
                    ),
                ),
            ]
        ),
        "random_forest": lambda seed: RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
    }

    try:
        import xgboost as xgb

        def make_xgb(seed: int):
            spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
            return xgb.XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=10,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=2.0,
                scale_pos_weight=spw,
                eval_metric="aucpr",     # NOT auc, NOT logloss
                tree_method="hist",
                random_state=seed,
                n_jobs=-1,
            )

        models["xgboost"] = make_xgb
    except ImportError:
        print("xgboost not installed; skipping")

    results = []
    for name, factory in models.items():
        for seed in SEEDS:
            set_all_seeds(seed, deterministic=False)
            model = factory(seed).fit(X_tr, y_tr)
            scores = model.predict_proba(X_te)[:, 1]
            results.append(
                evaluate(
                    y_te,
                    scores,
                    model=name,
                    seed=seed,
                    split="test",
                    alert_budget=args.alert_budget,
                )
            )
        print(f"  {name}: done")

    summary = aggregate_over_seeds(results)
    print("\n" + format_table(summary))
    print(
        "\nEvery number is mean +/- std over 5 seeds. A gap smaller than the "
        "std is not a result."
    )


if __name__ == "__main__":
    main()
