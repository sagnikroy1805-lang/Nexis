"""Drift experiment on the tier-3 generator (Concept Mastery Module 13).

The protocol from the feasibility study (§2.3), run once per project seed:
  - one injected behavioural shift at a known time (the generator's drift onset);
  - one label-free detector: PSI on the score distribution (§13.3, §13.4);
  - one adaptation policy: retrain on the most recent labelled window (§13.6);
  - a static model scored on the identical stream;
  - latency accounting: how long after onset the alarm fired, and how much
    PR-AUC was lost in that gap.
Each seed generates its own synthetic world, so the reported std covers data
variation as well as model variation (rule 3).

Usage:
    python scripts/run_drift.py --config configs/synthetic.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from nexis.data.synthetic import generate, load_generator_config
from nexis.drift.adaptation import RetrainRecentWindow
from nexis.drift.detectors import ThresholdDetector
from nexis.drift.experiment import plot_drift_experiment, run_drift_experiment
from nexis.evaluation.seeds import SEEDS
from nexis.features.behavioural import behavioural_features
from nexis.features.tabular import tabular_features
from nexis.models.baselines.scorers import XGBScorer

RESULTS = Path("results")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument("--reference-day", type=int, default=8)
    parser.add_argument("--window", default="1D")
    parser.add_argument("--retrain-window", default="5D")
    parser.add_argument("--alert-rate", type=float, default=0.005)
    args = parser.parse_args()

    base = load_generator_config(args.config)
    runs = []
    for seed in SEEDS:
        cfg = replace(base, seed=seed)
        data = generate(cfg)
        df = data.transactions
        feats = pd.concat([tabular_features(df), behavioural_features(df, ("5min", "1h", "24h"))], axis=1)
        stream = pd.concat([df[["tx_id", "timestamp", "is_fraud"]], feats], axis=1)
        cols = list(feats.columns)
        result = run_drift_experiment(
            stream,
            feature_cols=cols,
            label_col="is_fraud",
            time_col="timestamp",
            model_factory=lambda cols=cols, seed=seed: XGBScorer(cols, seed=seed, params={"n_estimators": 400}),
            reference_end=pd.Timestamp(cfg.start) + pd.Timedelta(days=args.reference_day),
            window=args.window,
            detector=ThresholdDetector(drift_level=0.25, warn_level=0.1, patience=1),
            policy=RetrainRecentWindow(window=args.retrain_window),
            alert_budget_rate=args.alert_rate,
            drift_onset=cfg.drift_onset,
            embargo="24h",
            name=f"synthetic_{cfg.drift_kind}_seed{seed}",
        )
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"drift_synthetic_seed{seed}.json").write_text(
            json.dumps(asdict(result), indent=2, default=str)
        )
        if seed == SEEDS[0]:
            plot_drift_experiment(result, RESULTS / "drift_synthetic_seed0.png")
        runs.append(result)
        print(
            f"seed {seed}: detected {result.detected_at} (latency {result.detection_latency_hours} h), "
            f"false alarms {result.false_alarms_before_onset}, post-onset PR-AUC static "
            f"{result.mean_pr_auc_post_onset_static} vs adaptive {result.mean_pr_auc_post_onset_adaptive}"
        )

    def stat(attr: str) -> tuple[float, float]:
        v = np.array([getattr(r, attr) for r in runs if getattr(r, attr) is not None], dtype=float)
        return (float(v.mean()), float(v.std(ddof=1))) if len(v) > 1 else (float("nan"), float("nan"))

    summary = {
        "drift_kind": base.drift_kind,
        "drift_magnitude": base.drift_magnitude,
        "n_seeds": len(runs),
        "detected": sum(r.detected_at is not None for r in runs),
        "detection_latency_hours": stat("detection_latency_hours"),
        "false_alarms_before_onset": stat("false_alarms_before_onset"),
        "post_onset_pr_auc_static": stat("mean_pr_auc_post_onset_static"),
        "post_onset_pr_auc_adaptive": stat("mean_pr_auc_post_onset_adaptive"),
        "pr_auc_lost_onset_to_detection": stat("pr_auc_lost_onset_to_detection"),
    }
    (RESULTS / "drift_synthetic_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
