"""Day-3 dataset statistics for docs/dataset_card.md, printed as markdown.

Every number in the card's statistics sections comes from this script, so the
card can be regenerated rather than trusted.

Usage:
    python scripts/dataset_stats.py --config configs/ibm_aml.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew

from nexis.data.ibm_aml import load_config, load_patterns, load_processed

QUANTILES = (0.5, 0.9, 0.99, 0.999, 1.0)


def _table(df: pd.DataFrame, floatfmt: str = "{:,.2f}") -> str:
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join([df.index.name or "", *cols]) + " |",
        "|" + "---|" * (len(cols) + 1),
    ]
    # Format by column dtype, not by cell: iterrows() upcasts a mixed row to float.
    fmts = {
        c: (lambda v: f"{int(v):,}")
        if pd.api.types.is_integer_dtype(df[c])
        else (lambda v: floatfmt.format(v))
        for c in df.columns
    }
    for idx in df.index:
        cells = [fmts[c](df.loc[idx, c]) for c in df.columns]
        lines.append("| " + " | ".join([str(idx), *cells]) + " |")
    return "\n".join(lines)


def amount_section(df: pd.DataFrame) -> str:
    rows = {}
    for name, s in (
        ("all", df["amount"]),
        ("negatives", df.loc[df["is_fraud"] == 0, "amount"]),
        ("positives", df.loc[df["is_fraud"] == 1, "amount"]),
    ):
        q = s.quantile(QUANTILES)
        rows[name] = {
            "median": float(q[0.5]),
            "p90": float(q[0.9]),
            "p99": float(q[0.99]),
            "max": float(q[1.0]),
            "skew": float(skew(s)),
            "skew log1p": float(skew(np.log1p(s))),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "USD amount"
    return _table(out)


def account_section(df: pd.DataFrame) -> str:
    involvement = pd.concat([df["src"], df["dst"]]).value_counts()
    sent = df["src"].value_counts()
    top1 = involvement.head(max(1, len(involvement) // 100)).sum()
    q = involvement.quantile(QUANTILES)
    lines = [
        f"- Accounts: {len(involvement):,}; senders: {len(sent):,}",
        "- Transactions per account (as src or dst): "
        + ", ".join(f"p{int(k * 1000) / 10:g} = {int(v):,}" for k, v in q.items()),
        f"- Accounts seen exactly once: {int((involvement == 1).sum()):,} "
        f"({(involvement == 1).mean():.1%})",
        f"- The busiest 1% of accounts take part in {top1 / involvement.sum():.1%} "
        "of all account-transaction incidences",
    ]
    return "\n".join(lines)


def time_section(df: pd.DataFrame) -> str:
    per_minute = df.groupby("timestamp").size()
    per_hour = df.set_index("timestamp").resample("1h").size()
    return "\n".join(
        [
            f"- Distinct timestamps: {len(per_minute):,} (resolution: 1 minute)",
            f"- Rows per occupied minute: median {int(per_minute.median()):,}, "
            f"p99 {int(per_minute.quantile(0.99)):,}, max {int(per_minute.max()):,}",
            f"- Rows per hour: median {int(per_hour.median()):,}, "
            f"max {int(per_hour.max()):,}, empty hours {int((per_hour == 0).sum())}",
        ]
    )


def pattern_section(df: pd.DataFrame, patterns: pd.DataFrame, cfg) -> str:
    per = patterns.groupby("pattern_id").agg(
        typology=("typology", "first"),
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        n_tx=("timestamp", "size"),
        n_kept=("tx_id", "count"),
    )
    per["hours"] = (per["end"] - per["start"]).dt.total_seconds() / 3600
    by_typ = per.groupby("typology", observed=True).agg(
        patterns=("n_tx", "size"),
        tx=("n_tx", "sum"),
        tx_kept=("n_kept", "sum"),
        median_tx=("n_tx", "median"),
        median_hours=("hours", "median"),
        p90_hours=("hours", lambda h: h.quantile(0.9)),
        max_hours=("hours", "max"),
    )
    by_typ.loc["ALL"] = [
        len(per),
        int(per["n_tx"].sum()),
        int(per["n_kept"].sum()),
        float(per["n_tx"].median()),
        float(per["hours"].median()),
        float(per["hours"].quantile(0.9)),
        float(per["hours"].max()),
    ]
    for c in ("patterns", "tx", "tx_kept"):
        by_typ[c] = by_typ[c].astype(int)
    by_typ.index.name = "typology"

    # Which folds does each pattern touch? A pattern spanning train and test is
    # what early-warning evaluation is about -- and it also means part of the
    # pattern's labels are legitimately known when its later legs are scored.
    split = cfg.split.apply(df)
    fold_of = pd.concat(
        [
            pd.Series("train", index=split.train["tx_id"]),
            pd.Series("val", index=split.val["tx_id"]),
            pd.Series("test", index=split.test["tx_id"]),
        ]
    )
    kept = patterns.dropna(subset=["tx_id"]).copy()
    kept["fold"] = kept["tx_id"].map(fold_of).fillna("embargo")
    folds = kept.groupby("pattern_id")["fold"].agg(lambda f: "+".join(sorted(set(f))))

    unattributed = int(df["is_fraud"].sum()) - len(kept)
    return "\n\n".join(
        [
            _table(by_typ, "{:,.1f}"),
            f"Positives in kept data not in any listed pattern: {unattributed:,} "
            f"of {int(df['is_fraud'].sum()):,} ({unattributed / df['is_fraud'].sum():.1%}).",
            "Patterns by the folds their kept transactions fall in:\n\n"
            + "\n".join(f"- {k}: {v}" for k, v in folds.value_counts().items()),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/ibm_aml.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_processed(cfg)
    print(f"## Statistics ({cfg.variant}, after processing)\n")
    print(f"Rows {len(df):,}; positives {int(df['is_fraud'].sum()):,}; "
          f"prevalence {df['is_fraud'].mean():.4%}\n")
    print("### Amounts (USD)\n\n" + amount_section(df) + "\n")
    print("### Accounts\n\n" + account_section(df) + "\n")
    print("### Time\n\n" + time_section(df) + "\n")
    if cfg.patterns_path is not None:
        print("### Laundering patterns\n\n" + pattern_section(df, load_patterns(cfg), cfg))


if __name__ == "__main__":
    main()
