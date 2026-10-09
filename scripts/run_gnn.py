"""Rungs 6-8: homogeneous, heterogeneous and temporal heterogeneous GNNs.

Each GNN receives, per transaction, exactly the inputs of the rung-5 XGBoost
(transaction + behavioural + structural features) plus learned graph context
from leak-free snapshots. Any gain over rung 5 is therefore attributable to
graph LEARNING, not to extra information (Concept Mastery §22.2, §22.3).

Tuning uses the same trial budget as XGBoost (configs: evaluation.tuning_trials),
on the validation fold, by PR-AUC.

Usage:
    python scripts/run_gnn.py --kinds homogeneous heterogeneous temporal [--trials N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from nexis.evaluation.harness import format_table, run_experiment
from nexis.evaluation.metrics import EvalResult, evaluate
from nexis.experiments.common import LABEL, prepare, save_scores
from nexis.graphs.diagnostics import degree_summary, positive_edge_adjacency
from nexis.graphs.snapshots import build_snapshots
from nexis.models.gnn.train import GNNScorer

RESULTS = Path("results")


def sample_params(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "hidden": int(rng.choice([32, 64, 128])),
        "layers": int(rng.choice([1, 2, 3])),
        "lr": float(rng.choice([1e-3, 3e-3, 1e-2])),
        "dropout": float(rng.choice([0.1, 0.2, 0.3])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/ibm_aml.yaml"))
    parser.add_argument(
        "--kinds", nargs="+", default=["homogeneous", "heterogeneous", "temporal"]
    )
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    prep = prepare(args.config)
    delta = prep.raw["graph"]["snapshot"]
    snaps = build_snapshots(prep.df, delta=delta)
    train, val, test = prep.split.train, prep.split.val, prep.split.test
    print(f"data + {len(snaps.snapshots)} snapshots ready in {time.time() - t0:.0f}s")

    # Diagnostics before trusting or blaming a GNN (Getting Started §4.1).
    tr_pos = pd.Index(prep.frame["tx_id"]).get_indexer(train["tx_id"])
    last = snaps.snapshots[-1]
    diag = {
        "degree": degree_summary(last.edge_index, snaps.n_accounts),
        "positive_edge_adjacency_train": positive_edge_adjacency(
            snaps.src[tr_pos], snaps.dst[tr_pos], train[LABEL].to_numpy()
        ),
        "n_accounts": snaps.n_accounts,
        "n_banks": snaps.n_banks,
        "edges_last_snapshot": int(last.edge_index.shape[1]),
        "snapshot_delta": delta,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "gnn_diagnostics_ibm_aml.json").write_text(json.dumps(diag, indent=2))
    print("diagnostics:", json.dumps(diag))

    ev = prep.evaluation
    trials = args.trials if args.trials is not None else ev["tuning_trials"]
    cols = prep.with_graph_cols
    y_test = test[LABEL].to_numpy()
    rung_of = {"homogeneous": 6, "heterogeneous": 7, "temporal": 8}

    summaries = []
    for kind in args.kinds:
        rng = np.random.default_rng(0)
        log = []
        s = time.time()
        for trial in range(trials):
            params = sample_params(rng)
            t_trial = time.time()
            try:
                m = GNNScorer(
                    kind, snaps, prep.frame, cols, seed=0, epochs=args.epochs, **params
                ).fit(train, val)
                ap = float(average_precision_score(val[LABEL], m.score(val)))
                log.append({**params, "val_pr_auc": ap, "epochs_run": len(m.history),
                            "secs": round(time.time() - t_trial)})
            except torch.cuda.OutOfMemoryError:
                # Counted against the budget, reported, never silently dropped.
                log.append({**params, "val_pr_auc": None, "failed": "out of GPU memory"})
                ap = float("nan")
            finally:
                torch.cuda.empty_cache()
            print(f"  {kind} trial {trial}: {params} val PR-AUC {ap:.4f} ({time.time() - t_trial:.0f}s)", flush=True)
        ok = [r for r in log if r.get("val_pr_auc") is not None]
        best = max(ok, key=lambda r: r["val_pr_auc"]) if ok else {}
        best_params = {k: best[k] for k in ("hidden", "layers", "lr", "dropout")} if best else {}
        (RESULTS / f"tuning_ibm_aml_gnn_{kind}.json").write_text(
            json.dumps({"budget": trials, "best": best_params, "trials": log}, indent=2)
        )
        print(f"tuned {kind} in {time.time() - s:.0f}s: {best_params}")

        name = f"gnn_{kind}"

        def evaluate_fn(model: GNNScorer, seed: int, name: str = name, kind: str = kind) -> EvalResult:
            model.fit(train, val)
            test_scores = model.score(test)
            save_scores(name, seed, {"val": (val, model.score(val)), "test": (test, test_scores)})
            return evaluate(
                y_test,
                test_scores,
                model=name,
                seed=seed,
                split="test",
                alert_budget=ev["alert_budget"],
                extra={"rung": rung_of[kind], "epochs_run": len(model.history)},
            )

        s = time.time()
        summary = run_experiment(
            name=f"ibm_aml_r{rung_of[kind]}_{name}",
            config={
                "dataset": "ibm_aml",
                "variant": prep.cfg.variant,
                "rung": rung_of[kind],
                "model": name,
                "split": prep.cfg.split.__dict__,
                "snapshot": delta,
                "edge_features": cols,
                "params": best_params,
                "tuning_budget": trials,
            },
            model_factory=lambda _c, seed, kind=kind, bp=best_params: GNNScorer(
                kind, snaps, prep.frame, cols, seed=seed, epochs=args.epochs,
                verbose=args.verbose, **bp,
            ),
            evaluate_fn=evaluate_fn,
            results_dir=RESULTS,
        )
        row = summary.iloc[0]
        print(f"  r{rung_of[kind]} {name}: PR-AUC {row['pr_auc']:.4f} ± {row['pr_auc_std']:.4f} ({time.time() - s:.0f}s)")
        summaries.append(summary)

    print("\n" + format_table(pd.concat(summaries).sort_values("pr_auc", ascending=False)))
    print(f"test prevalence {y_test.mean():.4%}")


if __name__ == "__main__":
    main()
