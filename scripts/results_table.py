"""Aggregate results/ into a paper-ready table: docs/results.md.

Implements the /results-table command: mean ± std over seeds, grouped by
model (latest run of each), sorted by PR-AUC; prevalence and seed count as
columns; pairs whose PR-AUC difference is smaller than their combined std are
flagged as not distinguishable; runs from a dirty working tree are listed.

Usage:
    python scripts/results_table.py [--out docs/results.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path("results")
COLS = ("pr_auc", "roc_auc", "recall_at_fpr_1e3", "precision_at_budget")


def collect() -> list[dict]:
    latest: dict[str, dict] = {}
    for mpath in RESULTS.glob("*/manifest.json"):
        spath = mpath.with_name("summary.json")
        if not spath.exists():
            continue
        m = json.loads(mpath.read_text(encoding="utf-8"))
        s = json.loads(spath.read_text(encoding="utf-8"))
        for model in s.get("pr_auc", {}):
            row = {
                "model": model,
                "rung": m.get("config", {}).get("rung"),
                "n_seeds": len(m.get("seeds", [])),
                "prevalence": s.get("prevalence", {}).get(model),
                "dirty": bool(m.get("working_tree_dirty")),
                "commit": str(m.get("code_commit", ""))[:8],
                "run_id": m.get("run_id"),
                "ts": float(m.get("timestamp", 0)),
                "median_ewt_minutes": s.get("median_ewt_minutes", {}).get(model),
                "ring_detection_rate": s.get("ring_detection_rate", {}).get(model),
            }
            for c in COLS:
                row[c] = s.get(c, {}).get(model)
                row[f"{c}_std"] = s.get(f"{c}_std", {}).get(model)
            if model not in latest or row["ts"] > latest[model]["ts"]:
                latest[model] = row
    return list(latest.values())


def fmt(v: float | None, sd: float | None) -> str:
    if v is None:
        return "–"
    return f"{v:.4f} ± {sd:.4f}" if sd is not None else f"{v:.4f}"


def _json(name: str) -> object | None:
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _mean_std(values: list[float]) -> str:
    import statistics as st

    if not values:
        return "–"
    sd = st.stdev(values) if len(values) > 1 else 0.0
    return f"{st.mean(values):.3f} ± {sd:.3f}"


def extra_sections(out: list[str]) -> None:
    rings = _json("rings_ibm_aml.json")
    if isinstance(rings, list) and rings:
        out.extend(["## Ring detection and early warning (IBM AML, rung 9)", "",
                    f"Fused scores; stage-2 scorer trained on validation-window candidates. "
                    f"{rings[0]['n_truth']} documented patterns are active in the test window. "
                    "Ring metrics use greedy one-to-one Jaccard matching (§11.4.3).", "",
                    "| candidates | θ | ring precision | ring recall | member precision | member recall |",
                    "|---|---|---|---|---|---|"])
        for k in rings[0]["ring_metrics"]:
            sel, theta = k.split("@")
            label = "top-K (K = #patterns)" if sel == "top_k_truth" else "all"
            cells = [_mean_std([r["ring_metrics"][k][m] for r in rings])
                     for m in ("ring_precision", "ring_recall", "member_precision", "member_recall")]
            out.append(f"| {label} | {theta} | " + " | ".join(cells) + " |")
        out.extend(["", "| pattern detection rate (alert on a member before completion) | median minutes of early warning |",
                    "|---|---|",
                    f"| {_mean_std([r['ewt']['detection_rate'] for r in rings])} | "
                    f"{_mean_std([r['ewt']['median_ewt_minutes'] for r in rings])} |", ""])
    drift = _json("drift_synthetic_summary.json")
    if isinstance(drift, dict):
        def ms(key: str) -> str:
            v = drift.get(key)
            return f"{v[0]:.3f} ± {v[1]:.3f}" if isinstance(v, list) else "–"

        out.extend(["## Drift (synthetic generator, amount shift ×3)", "",
                    "| detected | latency (h) | false alarms before onset | post-onset PR-AUC static | post-onset PR-AUC adaptive |",
                    "|---|---|---|---|---|",
                    f"| {drift['detected']}/{drift['n_seeds']} | {ms('detection_latency_hours')} | "
                    f"{ms('false_alarms_before_onset')} | {ms('post_onset_pr_auc_static')} | "
                    f"{ms('post_onset_pr_auc_adaptive')} |", ""])
    per_step = _json("elliptic_per_step_pr_auc.json")
    if isinstance(per_step, dict) and per_step:
        steps = sorted({int(s) for m in per_step.values() for s in m})
        out.extend(["## Elliptic++ per test step (seed 0) — the dark-market shutdown", "",
                    "| model | " + " | ".join(str(s) for s in steps) + " |",
                    "|---|" + "---|" * len(steps)])
        for model, vals in per_step.items():
            out.append(f"| {model} | " + " | ".join(f"{vals.get(str(s), float('nan')):.2f}" for s in steps) + " |")
        out.append("")


def render(rows: list[dict]) -> str:
    ibm = [r for r in rows if isinstance(r["rung"], int) and not r["model"].startswith("elliptic_")]
    ell = [r for r in rows if isinstance(r["rung"], int) and r["model"].startswith("elliptic_")]
    ablation = [r for r in rows if r["rung"] == "ablation"]
    out = ["# NEXIS results", "",
           "Generated by `make results` from `results/`. Test folds only. Every number is mean ± std "
           "over the seeds shown. **A difference smaller than the combined std is not a result.**", ""]

    def table(rs: list[dict], title: str) -> None:
        out.extend([f"## {title}", "",
                    "| rung | model | PR-AUC | ROC-AUC | recall@FPR=1e-3 | precision@budget | prevalence | seeds |",
                    "|---|---|---|---|---|---|---|---|"])
        for r in sorted(rs, key=lambda r: -(r["pr_auc"] or 0)):
            out.append(
                f"| {r['rung']} | {r['model']} | {fmt(r['pr_auc'], r['pr_auc_std'])} | "
                f"{fmt(r['roc_auc'], r['roc_auc_std'])} | {fmt(r['recall_at_fpr_1e3'], r['recall_at_fpr_1e3_std'])} | "
                f"{fmt(r['precision_at_budget'], r['precision_at_budget_std'])} | "
                f"{(r['prevalence'] or 0):.4%} | {r['n_seeds']} |"
            )
        out.append("")
        ranked = sorted(rs, key=lambda r: -(r["pr_auc"] or 0))
        close = [
            (a["model"], b["model"])
            for a, b in zip(ranked, ranked[1:], strict=False)
            if a["pr_auc"] is not None and b["pr_auc"] is not None
            and abs(a["pr_auc"] - b["pr_auc"]) < (a["pr_auc_std"] or 0) + (b["pr_auc_std"] or 0)
        ]
        if close:
            out.append("**Not distinguishable at this sample size** (PR-AUC gap < combined std): "
                       + "; ".join(f"{a} vs {b}" for a, b in close) + ".")
            out.append("")

    table(ibm, "IBM AML HI-Small — modelling ladder (tier 2)")
    if ablation:
        table(ablation, "IBM AML — feature-group ablation (rung-5 XGBoost, fixed hyper-parameters)")
    if ell:
        table(ell, "Elliptic++ transactions — modelling ladder (tier 1; labelled nodes only)")
    extra_sections(out)
    dirty = [r for r in rows if r["dirty"]]
    out.extend(["## Provenance", ""])
    if dirty:
        out.append("Runs from a **dirty working tree** (not reproducible from a commit): "
                   + ", ".join(f"`{r['run_id']}`" for r in dirty) + ".")
    else:
        out.append("Every run above was produced from a clean, committed working tree.")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/results.md"))
    args = parser.parse_args()
    text = render(collect())
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
