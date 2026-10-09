"""Elliptic++ (tier 1): the modelling ladder on real Bitcoin labels.

Split by time step (train 1-29, val 30-34, test 35-49). Unknown-label nodes are
excluded from training targets and from every metric -- never treated as licit.

Rungs on this dataset:
  1 random                          chance (PR-AUC = illicit share of labelled test)
  3 tabular: LR / RF / XGB on the transaction's LOCAL features only
  5 graph information: XGB with Elliptic's neighbour-AGGREGATED features too
  6 GNN: GraphSAGE node classifier on the transaction graph (local features as
    input, so the graph has to be learned, not read off the aggregates)
Steps 43+ follow a dark-market shutdown, a documented regime change; per-step
PR-AUC is saved to show it.

Usage:
    python scripts/run_elliptic.py --raw data/raw/elliptic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from nexis.data.elliptic import StepSplit, load_elliptic, manifest
from nexis.evaluation.harness import format_table, run_experiment
from nexis.evaluation.metrics import evaluate
from nexis.models.baselines.scorers import (
    RandomScorer,
    SklearnScorer,
    XGBScorer,
    tune_xgb,
)

RESULTS = Path("results")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw/elliptic"))
    parser.add_argument("--trials", type=int, default=6)  # same budget as IBM AML
    args = parser.parse_args()

    data = load_elliptic(args.raw)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "elliptic_manifest.json").write_text(json.dumps(manifest(args.raw, data), indent=2))
    nodes = data.nodes.copy()
    nodes["fold"] = StepSplit().fold(nodes["time_step"])
    nodes["is_fraud"] = nodes["label"]  # scorer convention; NaN rows are filtered below
    labelled = nodes.loc[nodes["label"].notna()].copy()
    labelled["is_fraud"] = labelled["is_fraud"].astype(int)
    train, val, test = (labelled.loc[labelled["fold"] == f] for f in ("train", "val", "test"))
    y = test["is_fraud"].to_numpy()
    budget = int(round(0.01 * len(test)))
    print(f"labelled: train {len(train)}, val {len(val)}, test {len(test)} (illicit {y.mean():.2%})")

    local, full = data.local_cols, data.feature_cols
    best_local, _ = tune_xgb(train, val, local, n_trials=args.trials)
    best_full, _ = tune_xgb(train, val, full, n_trials=args.trials)

    from nexis.models.gnn.node import NodeGNNScorer

    ladder = {
        "random": (1, lambda s: RandomScorer(seed=s)),
        "lr_local": (3, lambda s: SklearnScorer("logistic_regression", local, s, 1.0)),
        "rf_local": (3, lambda s: SklearnScorer("random_forest", local, s, 1.0)),
        "xgb_local": (3, lambda s: XGBScorer(local, s, best_local)),
        "xgb_all_features": (5, lambda s: XGBScorer(full, s, best_full)),
        "gnn_sage_local": (6, lambda s: NodeGNNScorer(nodes, data.edges, local, seed=s)),
    }
    summaries, per_step = [], {}
    for name, (rung, factory) in ladder.items():
        def evaluate_fn(m, seed, name=name, rung=rung):
            m.fit(train, val)
            s = m.score(test)
            if seed == 0:
                per_step[name] = {
                    int(k): float(average_precision_score(g["is_fraud"], s[g.index.map(test.index.get_loc)]))
                    for k, g in test.groupby("time_step") if g["is_fraud"].sum() > 0
                }
            return evaluate(y, s, model=f"elliptic_{name}", seed=seed, split="test",
                            alert_budget=budget, extra={"rung": rung})

        summ = run_experiment(
            name=f"elliptic_r{rung}_{name}",
            config={"dataset": "elliptic++", "rung": rung, "model": name,
                    "split": {"train_end": 29, "val_end": 34}, "alert_budget": budget},
            model_factory=lambda _c, seed, factory=factory: factory(seed),
            evaluate_fn=evaluate_fn,
            results_dir=RESULTS,
        )
        row = summ.iloc[0]
        print(f"  r{rung} {name:<18} PR-AUC {row['pr_auc']:.4f} ± {row['pr_auc_std']:.4f}")
        summaries.append(summ)
    (RESULTS / "elliptic_per_step_pr_auc.json").write_text(json.dumps(per_step, indent=2))
    print("\n" + format_table(pd.concat(summaries).sort_values("pr_auc", ascending=False)))
    print(f"test illicit share among labelled: {np.mean(y):.4f}")


if __name__ == "__main__":
    main()
