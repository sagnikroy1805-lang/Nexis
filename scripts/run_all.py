"""Run every experiment in order, one process per step, from a committed tree.

One process per step because a CUDA driver reset ("unspecified launch failure")
poisons the process that hit it; a fresh process recovers. Each GPU step is
retried once. Logs go to results/logs/<step>.log.

The GNN passed to rung 9 is the one with the best VALIDATION PR-AUC in its
tuning log -- choosing it by test score would be selection on the test set.

Usage:
    python scripts/run_all.py [--skip ladder gnn_homogeneous ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

RESULTS = Path("results")
LOGS = RESULTS / "logs"
PY = sys.executable


def run(name: str, args: list[str], retries: int = 0) -> bool:
    LOGS.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries + 1):
        t0 = time.time()
        with (LOGS / f"{name}.log").open("a", encoding="utf-8") as log:
            log.write(f"\n=== {name} attempt {attempt + 1} {time.ctime()} ===\n")
            log.flush()
            code = subprocess.call([PY, "-u", *args], stdout=log, stderr=subprocess.STDOUT)
        print(f"{name}: exit {code} after {time.time() - t0:.0f}s (attempt {attempt + 1})", flush=True)
        if code == 0:
            return True
    return False


def best_gnn() -> str:
    best, name = -1.0, "gnn_temporal"
    for kind in ("homogeneous", "heterogeneous", "temporal"):
        path = RESULTS / f"tuning_ibm_aml_gnn_{kind}.json"
        if not path.exists():
            continue
        trials = json.loads(path.read_text())["trials"]
        vals = [t["val_pr_auc"] for t in trials if t.get("val_pr_auc") is not None]
        if vals and max(vals) > best and Path(f"results/scores/gnn_{kind}_seed0.parquet").exists():
            best, name = max(vals), f"gnn_{kind}"
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ibm_aml.yaml")
    parser.add_argument("--skip", nargs="*", default=[])
    args = parser.parse_args()
    cfg = ["--config", args.config]

    steps: list[tuple[str, list[str], int]] = [
        ("ladder", ["scripts/run_ladder.py", *cfg, "--rungs", "1", "2", "3", "4", "5", "--canary"], 1),
        ("gnn_homogeneous", ["scripts/run_gnn.py", *cfg, "--kinds", "homogeneous"], 1),
        ("gnn_heterogeneous", ["scripts/run_gnn.py", *cfg, "--kinds", "heterogeneous"], 1),
        ("gnn_temporal", ["scripts/run_gnn.py", *cfg, "--kinds", "temporal"], 1),
    ]
    for name, step, retries in steps:
        if name not in args.skip:
            run(name, step, retries)

    if "rung9" not in args.skip:
        gnn = best_gnn()
        print(f"rung 9 fuses xgb_graph with {gnn} (best validation PR-AUC)", flush=True)
        run("rung9", ["scripts/run_rung9.py", *cfg, "--base", "xgb_graph", gnn], 1)
    for name, step in (
        ("ablation", ["scripts/run_ablation.py", *cfg]),
        ("elliptic", ["scripts/run_elliptic.py", "--raw", "data/raw/elliptic"]),
        ("drift", ["scripts/run_drift.py", "--config", "configs/synthetic.yaml"]),
        ("results", ["scripts/results_table.py"]),
    ):
        if name not in args.skip:
            run(name, step, 1)


if __name__ == "__main__":
    main()
