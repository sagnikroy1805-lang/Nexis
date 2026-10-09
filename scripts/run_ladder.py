"""Rungs 1-4 of the modelling ladder on a real dataset, through the harness.

Rung 1  trivial        random, amount-only
Rung 2  rules          analyst thresholds (fitted on training volume, not labels)
Rung 3  tabular        LR / RF / XGBoost on transaction fields only
Rung 4  behavioural    the same models + sender/receiver behaviour, Isolation Forest

Every model runs over the five project seeds; results land in results/ via
nexis.evaluation.harness.run_experiment (manifest with git state, per-seed JSON,
mean +/- std summary). The leakage canary then re-runs the best rung-4 pipeline
with shuffled training labels.

Usage:
    python scripts/run_ladder.py --config configs/ibm_aml.yaml [--rungs 1 2 3 4] [--canary]
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from nexis.data.ibm_aml import load_config, load_processed
from nexis.evaluation.harness import format_table, run_experiment
from nexis.evaluation.leakage import leakage_canary
from nexis.evaluation.metrics import EvalResult, evaluate
from nexis.evaluation.splits import assert_no_id_overlap, assert_temporal_integrity
from nexis.experiments.common import save_scores
from nexis.features.pipeline import (
    behavioural_columns,
    feature_table,
    graph_feature_table,
)
from nexis.features.tabular import TABULAR_COLUMNS
from nexis.models.baselines.scorers import (
    LABEL,
    AmountScorer,
    IsolationForestScorer,
    RandomScorer,
    RulesScorer,
    SklearnScorer,
    XGBScorer,
    tune_xgb,
    xgb_device,
)

RESULTS = Path("results")


def load_frame(
    config_path: Path, with_graph: bool = False
) -> tuple[Any, pd.DataFrame, dict[str, Any], list[str]]:
    cfg = load_config(config_path)
    df = load_processed(cfg)
    raw_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    parts = [df[["tx_id", "timestamp", LABEL]], feature_table(cfg, df)]
    graph_cols: list[str] = []
    if with_graph:
        gf = graph_feature_table(cfg, df, raw_cfg["graph"]["snapshot"])
        graph_cols = list(gf.columns)
        parts.append(gf)
    return cfg, pd.concat(parts, axis=1), raw_cfg["evaluation"], graph_cols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/ibm_aml.yaml"))
    parser.add_argument("--rungs", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    cfg, frame, ev, graph_cols = load_frame(args.config, with_graph=5 in args.rungs)
    split = cfg.split.apply(frame)
    assert_temporal_integrity(split)  # loud failure beats silent leakage
    assert_no_id_overlap(split.train, split.test)
    train, val, test = split.train, split.val, split.test
    print(f"features ready in {time.time() - t0:.0f}s; split {split.summary()}")
    print(f"xgboost device: {xgb_device()}")

    tab = list(TABULAR_COLUMNS)
    non_graph = frame.drop(columns=["tx_id", "timestamp", LABEL, *graph_cols])
    full = tab + behavioural_columns(non_graph)
    with_graph = full + graph_cols
    budget, trials, neg = ev["alert_budget"], ev["tuning_trials"], ev["neg_subsample"]
    y_test = test[LABEL].to_numpy()

    tuned: dict[str, dict[str, Any]] = {}
    family = {"tabular": (3, tab), "behavioural": (4, full), "graph": (5, with_graph)}
    for name, (rung_of, cols) in family.items():
        if rung_of in args.rungs:
            s = time.time()
            best, log = tune_xgb(train, val, cols, n_trials=trials, seed=0)
            tuned[name] = best
            RESULTS.mkdir(exist_ok=True)
            (RESULTS / f"tuning_ibm_aml_xgb_{name}.json").write_text(
                json.dumps({"budget": trials, "best": best, "trials": log}, indent=2)
            )
            print(f"tuned xgb_{name} in {time.time() - s:.0f}s: {best}")

    ladder: dict[int, list[tuple[str, Callable[[int], Any]]]] = {
        1: [("random", lambda s: RandomScorer(seed=s)), ("amount_only", lambda s: AmountScorer())],
        2: [("rules", lambda s: RulesScorer())],
        3: [
            ("lr_tabular", lambda s: SklearnScorer("logistic_regression", tab, s, neg)),
            ("rf_tabular", lambda s: SklearnScorer("random_forest", tab, s, neg)),
            ("xgb_tabular", lambda s: XGBScorer(tab, s, tuned.get("tabular", {}))),
        ],
        4: [
            ("isolation_forest", lambda s: IsolationForestScorer(full, s)),
            ("lr_behavioural", lambda s: SklearnScorer("logistic_regression", full, s, neg)),
            ("rf_behavioural", lambda s: SklearnScorer("random_forest", full, s, neg)),
            ("xgb_behavioural", lambda s: XGBScorer(full, s, tuned.get("behavioural", {}))),
        ],
        5: [("xgb_graph", lambda s: XGBScorer(with_graph, s, tuned.get("graph", {})))],
    }

    summaries = []
    for rung in sorted(args.rungs):
        for name, factory in ladder[rung]:
            s = time.time()

            def evaluate_fn(model: Any, seed: int, name: str = name, rung: int = rung) -> EvalResult:
                model.fit(train, val)
                test_scores = model.score(test)
                if rung >= 4:  # kept for fusion, ring detection and the API
                    save_scores(name, seed, {"val": (val, model.score(val)), "test": (test, test_scores)})
                return evaluate(
                    y_test,
                    test_scores,
                    model=name,
                    seed=seed,
                    split="test",
                    alert_budget=budget,
                    extra={"rung": rung},
                )

            summary = run_experiment(
                name=f"ibm_aml_r{rung}_{name}",
                config={
                    "dataset": "ibm_aml",
                    "variant": cfg.variant,
                    "rung": rung,
                    "model": name,
                    "split": cfg.split.__dict__,
                    "windows": list(cfg.feature_windows),
                    "alert_budget": budget,
                    "xgb_params": tuned.get({3: "tabular", 4: "behavioural", 5: "graph"}.get(rung, "")),
                },
                model_factory=lambda _c, seed, factory=factory: factory(seed),
                evaluate_fn=evaluate_fn,
                results_dir=RESULTS,
            )
            summary["rung"] = rung
            summaries.append(summary)
            row = summary.iloc[0]
            print(
                f"  r{rung} {name:<18} PR-AUC {row['pr_auc']:.4f} ± {row['pr_auc_std']:.4f}"
                f"  ({time.time() - s:.0f}s)"
            )

    table = pd.concat(summaries)
    print("\n" + format_table(table.sort_values("pr_auc", ascending=False)))
    print(f"\ntest prevalence {y_test.mean():.4%} ({int(y_test.sum())} positives)")

    if args.canary:
        run_canary(frame, full, tuned.get("behavioural", {}), cfg.split.train_frac)


def run_canary(
    frame: pd.DataFrame, cols: list[str], params: dict[str, Any], train_frac: float
) -> None:
    """Shuffle training labels; the real rung-4 pipeline must fall to chance."""
    import xgboost as xgb

    def fit_predict(df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        t = pd.to_datetime(df["timestamp"])
        is_train = (t <= t.quantile(train_frac)).to_numpy()
        model = xgb.XGBClassifier(
            n_estimators=300,
            **{k: v for k, v in params.items()},
            eval_metric="aucpr",
            tree_method="hist",
            device=xgb_device(),
            random_state=0,
        )
        model.fit(df.loc[is_train, cols], y[is_train])
        return model.predict_proba(df[cols])[:, 1]

    result = leakage_canary(fit_predict, frame, frame[LABEL].to_numpy().astype(int))
    print(f"\nleakage canary: {result}")
    (RESULTS / "canary_ibm_aml_xgb_behavioural.json").write_text(json.dumps(result, indent=2))
    if result["ratio"] >= 3.0:
        raise SystemExit("CANARY FAILED: information reaches the model without labels")


if __name__ == "__main__":
    main()
