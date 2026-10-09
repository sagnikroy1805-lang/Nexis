"""Experiment runner: multi-seed execution, aggregation, provenance.

Implements Concept Mastery §21.4 and §22.6.

This replaces MLflow. A structured results/ directory with JSON manifests gives
full reproducibility without running a tracking server.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from nexis.evaluation.metrics import EvalResult
from nexis.evaluation.seeds import SEEDS, set_all_seeds


class Scorer(Protocol):
    """Every model in the project satisfies this interface and nothing more.

    Models produce scores. The harness does splitting, metrics and logging.
    """

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> Scorer: ...

    def score(self, df: pd.DataFrame) -> np.ndarray: ...


def _git_state() -> dict[str, Any]:
    """Record the exact code that produced a result.

    working_tree_dirty matters: a result from uncommitted code is not
    reproducible, and recording that stops you trusting it six months later.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            ).decode().strip()
        )
    except Exception:
        commit, dirty = "unknown", True
    return {"code_commit": commit, "working_tree_dirty": dirty}


def run_experiment(
    name: str,
    config: dict[str, Any],
    model_factory: Callable[[dict, int], Scorer],
    evaluate_fn: Callable[[Scorer, int], EvalResult],
    seeds: Sequence[int] = SEEDS,
    results_dir: str | Path = "results",
) -> pd.DataFrame:
    """Run one configuration across seeds and persist everything.

    Writes results/<name>_<timestamp>/ containing manifest.json (config, git state,
    data hash), per_seed.json (raw results) and summary.json (mean +/- std).

    Returns:
        The aggregated summary as a DataFrame, ready for a paper table.
    """
    run_id = f"{name}_{int(time.time())}"
    out = Path(results_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    # Captured BEFORE running: the code that produces the results is the code
    # imported at start. Checking afterwards would blame edits made while a
    # long run was in progress, and miss nothing that actually ran.
    code_state = _git_state()

    results: list[EvalResult] = []
    for seed in seeds:
        set_all_seeds(seed)
        model = model_factory(config, seed)
        results.append(evaluate_fn(model, seed))

    per_seed = [asdict(r) for r in results]
    summary = aggregate_over_seeds(results)

    (out / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "name": name,
                "config": config,
                "seeds": list(seeds),
                "timestamp": time.time(),
                **code_state,
            },
            indent=2,
            default=str,
        )
    )
    (out / "per_seed.json").write_text(json.dumps(per_seed, indent=2, default=str))
    (out / "summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2, default=str)
    )
    return summary


def aggregate_over_seeds(results: Sequence[EvalResult]) -> pd.DataFrame:
    """Mean and standard deviation across seeds, grouped by model.

    This is what goes in the paper. No number is reported without its std, and a
    difference smaller than the std is not a result.
    """
    df = pd.DataFrame([asdict(r) for r in results])
    numeric = [
        c
        for c in df.select_dtypes(include=[float, int]).columns
        if c not in ("seed",)
    ]
    grouped = df.groupby("model")[numeric]
    mean = grouped.mean()
    std = grouped.std().add_suffix("_std")
    return mean.join(std).sort_values("pr_auc", ascending=False)


def format_table(summary: pd.DataFrame, cols: Sequence[str] | None = None) -> str:
    """Render 'mean ± std' strings for a results table."""
    cols = cols or ["pr_auc", "roc_auc", "recall_at_fpr_1e3", "precision_at_budget"]
    lines = []
    for model, row in summary.iterrows():
        cells = [
            f"{row[c]:.4f} ± {row.get(f'{c}_std', float('nan')):.4f}" for c in cols
        ]
        lines.append(f"{model:<28} " + "  ".join(cells))
    header = f"{'model':<28} " + "  ".join(f"{c:<18}" for c in cols)
    return "\n".join([header, "-" * len(header), *lines])
