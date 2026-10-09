"""Feature-group ablation on the rung-5 XGBoost (Concept Mastery §22.3).

Each variant drops one group of features and re-runs the five seeds with the
SAME tuned hyper-parameters, so differences come from information, not tuning.
A drop smaller than the seed std is reported as "no measurable effect".

Usage:
    python scripts/run_ablation.py --config configs/ibm_aml.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nexis.evaluation.harness import format_table, run_experiment
from nexis.evaluation.metrics import evaluate
from nexis.experiments.common import LABEL, prepare
from nexis.features.tabular import TABULAR_COLUMNS
from nexis.models.baselines.scorers import XGBScorer

RESULTS = Path("results")


def groups(beh: list[str], graph: list[str]) -> dict[str, list[str]]:
    """Feature groups, each a hypothesis about where the signal lives."""
    b = [c for c in beh if c not in TABULAR_COLUMNS]
    return {
        "graph_structure": graph,
        "behavioural_all": b,
        "sender_behaviour": [c for c in b if c.startswith("src_") or c.startswith("amt_")],
        "receiver_behaviour": [c for c in b if c.startswith("dst_")],
        "pass_through": [c for c in b if c.startswith("src_in_") or c in ("src_secs_since_in", "amt_over_src_in_24h")],
        "counterparty_novelty": ["counterparty_is_new", "src_n_counterparties", "dst_n_counterparties"],
        "graph_edge_motifs": [c for c in graph if c in ("closes_cycle", "reverse_edge_exists", "prior_pair_tx")],
        "transaction_fields": list(TABULAR_COLUMNS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/ibm_aml.yaml"))
    args = parser.parse_args()

    prep = prepare(args.config)
    train, val, test = prep.split.train, prep.split.val, prep.split.test
    full = prep.with_graph_cols
    params = json.loads((RESULTS / "tuning_ibm_aml_xgb_graph.json").read_text())["best"]
    budget = prep.evaluation["alert_budget"]
    y = test[LABEL].to_numpy()

    variants = {"full": full} | {
        f"minus_{g}": [c for c in full if c not in set(cols)]
        for g, cols in groups(prep.behavioural_cols, prep.graph_cols).items()
    }
    summaries = []
    for name, cols in variants.items():
        def evaluate_fn(m: XGBScorer, seed: int, name: str = name) -> object:
            m.fit(train, val)
            return evaluate(y, m.score(test), model=f"ablation_{name}", seed=seed, split="test",
                            alert_budget=budget, extra={"n_features": len(m.features)})

        s = run_experiment(
            name=f"ibm_aml_ablation_{name}",
            config={"dataset": "ibm_aml", "rung": "ablation", "variant": name,
                    "n_features": len(cols), "dropped": sorted(set(full) - set(cols)), "params": params},
            model_factory=lambda _c, seed, cols=cols: XGBScorer(cols, seed=seed, params=params),
            evaluate_fn=evaluate_fn,
            results_dir=RESULTS,
        )
        row = s.iloc[0]
        print(f"  {name:<32} PR-AUC {row['pr_auc']:.4f} ± {row['pr_auc_std']:.4f} ({len(cols)} features)")
        summaries.append(s)
    table = pd.concat(summaries)
    print("\n" + format_table(table.sort_values("pr_auc", ascending=False)))


if __name__ == "__main__":
    main()
