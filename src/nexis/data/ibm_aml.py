"""IBM AML (IT-AML) dataset adapter: raw CSV -> standard schema -> processed parquet.

Implements Concept Mastery §1.1 (the transaction as row, event and edge -- the
standard schema below is the minimum that supports all three views), §3.6 and
§9.4 (nothing derived from data later than the prediction it feeds), and §22.6
(every processed artefact is traceable to its exact input by content hash).

Source: Altman et al., "Realistic Synthetic Financial Transactions for Anti-Money
Laundering Models", NeurIPS 2023 Datasets and Benchmarks; distributed on Kaggle as
ealtman2019/ibm-transactions-for-anti-money-laundering-aml.

Label semantics (CLAUDE.md rule 5): `is_fraud` is the generator's `Is Laundering`
flag. It records that the simulator placed a transaction inside one of its
laundering patterns. It is a dataset label, not a finding about any account.

Bitemporality (§17.1.1): the dataset records only when a transaction occurred,
not when it became known. We treat the two as identical. That is an assumption
of the dataset, not a property of real payment systems, and the paper should say so.

Usage:
    python -m nexis.data.ibm_aml --config configs/ibm_aml.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import yaml

from nexis.evaluation.harness import _git_state
from nexis.evaluation.splits import SplitConfig, assert_embargo_covers_windows

# The raw header repeats "Account" for sender and receiver, so columns are named
# by position and the header is checked verbatim instead of trusted.
RAW_HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,"
    "Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering"
)
RAW_COLUMNS = (
    "Timestamp",
    "From Bank",
    "From Account",
    "To Bank",
    "To Account",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
)
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"
USD = "US Dollar"

REQUIRED_COLUMNS = ("tx_id", "src", "dst", "amount", "timestamp", "is_fraud")
EXTRA_COLUMNS = (
    "src_bank",
    "dst_bank",
    "amount_paid",
    "pay_currency",
    "amount_received",
    "recv_currency",
    "payment_format",
    "is_self_transfer",
    "is_cross_bank",
    "is_cross_currency",
)
_CATEGORICAL = ("src_bank", "dst_bank", "pay_currency", "recv_currency", "payment_format")


class DataValidationError(ValueError):
    """Raised when a dataset violates the standard-schema contract."""


@dataclass(frozen=True)
class IbmAmlConfig:
    """configs/ibm_aml.yaml, loaded and checked."""

    variant: str
    raw_path: Path
    processed_path: Path
    tail_cutoff: pd.Timestamp
    rate_fit_end: pd.Timestamp
    split: SplitConfig
    feature_windows: tuple[str, ...]

    @property
    def manifest_path(self) -> Path:
        return self.processed_path.with_suffix(".manifest.json")


@dataclass
class LoadReport:
    """What the loader kept, dropped and estimated. Written into the manifest."""

    variant: str
    n_raw: int
    n_dropped_tail: int
    n_pos_dropped_tail: int
    n_rows: int
    n_pos: int
    prevalence: float
    t_min: str
    t_max: str
    n_accounts: int
    n_self_transfers: int
    n_cross_currency: int
    n_exact_duplicate_raw_rows: int
    usd_per_unit: dict[str, float] = field(default_factory=dict)


def load_config(path: str | Path) -> IbmAmlConfig:
    """Load the dataset config and enforce rule 1 before any data is touched.

    Relative paths resolve against the repository root (the parent of configs/),
    so the config behaves the same whatever directory a script runs from.
    """
    path = Path(path).resolve()
    root = path.parent.parent
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    ds, sp, ft = raw["dataset"], raw["split"], raw["features"]

    split = SplitConfig(
        train_frac=float(sp["train_frac"]),
        val_frac=float(sp["val_frac"]),
        embargo=str(sp["embargo"]),
        split_on=sp["split_on"],
    )
    windows = tuple(str(w) for w in ft["windows"])
    # Checked here so an invalid combination cannot reach an experiment at all.
    assert_embargo_covers_windows(split.embargo, windows)

    return IbmAmlConfig(
        variant=str(ds["variant"]),
        raw_path=root / ds["raw_path"],
        processed_path=root / ds["processed_path"],
        tail_cutoff=pd.Timestamp(ds["tail_cutoff"]),
        rate_fit_end=pd.Timestamp(ds["rate_fit_end"]),
        split=split,
        feature_windows=windows,
    )


def read_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV with every identifier forced to string before parsing.

    Why not pd.read_csv(dtype=str): with the pyarrow engine, pandas applies the
    dtype *after* type inference, so bank "010" becomes 10 and then "10", and
    "001" and "01" collapse into the same bank. That silently merges distinct
    accounts. Here pyarrow is told the column types up front, so nothing is
    ever inferred as a number.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().strip()
    if header != RAW_HEADER:
        raise DataValidationError(
            f"unexpected header in {path.name}:\n  got      {header}\n"
            f"  expected {RAW_HEADER}"
        )

    string_cols = (
        "Timestamp",
        "From Bank",
        "From Account",
        "To Bank",
        "To Account",
        "Receiving Currency",
        "Payment Currency",
        "Payment Format",
    )
    column_types: dict[str, pa.DataType] = {c: pa.string() for c in string_cols}
    column_types |= {
        "Amount Received": pa.float64(),
        "Amount Paid": pa.float64(),
        "Is Laundering": pa.int8(),
    }
    table = pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(column_names=list(RAW_COLUMNS), skip_rows=1),
        convert_options=pacsv.ConvertOptions(column_types=column_types),
    )
    return table.to_pandas()


def estimate_usd_rates(df: pd.DataFrame) -> dict[str, float]:
    """USD value of one unit of each currency, from cross-currency rows with USD.

    The dataset publishes no rates, but every cross-currency row implies one:
    amount_received / amount_paid. The median over all such rows is robust to
    the rounding noise in tiny amounts (Bitcoin especially).

    LEAKAGE GUARD: callers pass only rows before rate_fit_end. Estimating from the
    full file would let test-period rows shape a training feature. The rates are
    constant in the generator, so this costs no accuracy -- but the guard is what
    makes that a checked fact rather than an assumption.
    """
    cross = df[df["pay_currency"] != df["recv_currency"]]
    log_ratio = np.log(cross["amount_received"] / cross["amount_paid"])

    to_usd = cross["recv_currency"] == USD  # paid in C, received USD: USD per C
    from_usd = cross["pay_currency"] == USD  # paid USD, received C: invert
    log_usd_per_unit = pd.concat(
        [
            pd.Series(
                log_ratio[to_usd].to_numpy(),
                index=cross.loc[to_usd, "pay_currency"].to_numpy(),
            ),
            pd.Series(
                -log_ratio[from_usd].to_numpy(),
                index=cross.loc[from_usd, "recv_currency"].to_numpy(),
            ),
        ]
    )
    medians = log_usd_per_unit.groupby(level=0).median()
    rates = {str(cur): float(np.exp(v)) for cur, v in medians.items()}
    rates[USD] = 1.0
    return dict(sorted(rates.items()))


def to_standard_schema(
    raw: pd.DataFrame,
    variant: str,
    tail_cutoff: pd.Timestamp,
    rate_fit_end: pd.Timestamp,
) -> tuple[pd.DataFrame, LoadReport]:
    """Map raw columns to the project schema, convert amounts, drop the tail.

    Accounts are identified as "<bank>_<account>": an account number is only
    unique within its bank, and dropping the bank code would merge unrelated
    accounts into one graph node.

    tx_id is the 0-based row number in the raw file, so every record traces back
    to its source line. That matters for evidence packets (rule 6), and it keeps
    the 9 exact-duplicate raw rows distinct rather than silently merged.
    """
    n_raw = len(raw)
    n_dup = int(raw.duplicated().sum())

    row = pd.Series(np.arange(n_raw), index=raw.index).astype(str).str.zfill(7)
    df = pd.DataFrame(
        {
            "tx_id": variant + ":" + row,
            "timestamp": pd.to_datetime(raw["Timestamp"], format=TIMESTAMP_FORMAT),
            "src": raw["From Bank"] + "_" + raw["From Account"],
            "dst": raw["To Bank"] + "_" + raw["To Account"],
            "is_fraud": raw["Is Laundering"].astype("int8"),
            "src_bank": raw["From Bank"],
            "dst_bank": raw["To Bank"],
            "amount_paid": raw["Amount Paid"],
            "pay_currency": raw["Payment Currency"],
            "amount_received": raw["Amount Received"],
            "recv_currency": raw["Receiving Currency"],
            "payment_format": raw["Payment Format"],
        }
    )

    rates = estimate_usd_rates(df[df["timestamp"] < rate_fit_end])
    unknown = sorted(set(df["pay_currency"]) - set(rates))
    if unknown:
        raise DataValidationError(
            f"no USD rate for {unknown}: these currencies have no cross-currency "
            f"USD row before rate_fit_end={rate_fit_end}"
        )
    df["amount"] = df["amount_paid"] * df["pay_currency"].map(rates).astype(float)
    df["is_self_transfer"] = (df["src"] == df["dst"]).astype("int8")
    df["is_cross_bank"] = (df["src_bank"] != df["dst_bank"]).astype("int8")
    df["is_cross_currency"] = (df["pay_currency"] != df["recv_currency"]).astype("int8")

    tail = df["timestamp"] >= tail_cutoff
    n_tail, n_tail_pos = int(tail.sum()), int(df.loc[tail, "is_fraud"].sum())
    df = df.loc[~tail]

    # Stable sort: rows in the same minute keep raw-file order, so the output is
    # a deterministic function of the input file.
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    for col in _CATEGORICAL:
        df[col] = df[col].astype("category")
    df = df[list(REQUIRED_COLUMNS + EXTRA_COLUMNS)]

    report = LoadReport(
        variant=variant,
        n_raw=n_raw,
        n_dropped_tail=n_tail,
        n_pos_dropped_tail=n_tail_pos,
        n_rows=len(df),
        n_pos=int(df["is_fraud"].sum()),
        prevalence=float(df["is_fraud"].mean()),
        t_min=str(df["timestamp"].min()),
        t_max=str(df["timestamp"].max()),
        n_accounts=int(pd.concat([df["src"], df["dst"]]).nunique()),
        n_self_transfers=int(df["is_self_transfer"].sum()),
        n_cross_currency=int(df["is_cross_currency"].sum()),
        n_exact_duplicate_raw_rows=n_dup,
        usd_per_unit=rates,
    )
    return df, report


def validate(df: pd.DataFrame, tail_cutoff: pd.Timestamp | None = None) -> None:
    """Check the standard-schema contract; report every violation at once.

    Raises rather than asserts: these checks guard the inputs of every
    experiment and must still run under `python -O`.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing required columns: {missing}")

    problems: list[str] = []
    nulls = df[list(REQUIRED_COLUMNS)].isna().sum()
    if nulls.any():
        problems.append(f"nulls in required columns: {nulls[nulls > 0].to_dict()}")
    if (df["amount"] <= 0).any():
        problems.append(f"{int((df['amount'] <= 0).sum())} non-positive amounts")
    if not df["timestamp"].is_monotonic_increasing:
        problems.append("timestamps are not sorted ascending")
    if df["tx_id"].duplicated().any():
        problems.append(f"{int(df['tx_id'].duplicated().sum())} duplicate tx_id values")
    bad_labels = set(pd.unique(df["is_fraud"])) - {0, 1}
    if bad_labels:
        problems.append(f"is_fraud has values other than 0/1: {sorted(bad_labels)}")
    if tail_cutoff is not None and (df["timestamp"] >= tail_cutoff).any():
        problems.append(f"rows at or after tail_cutoff {tail_cutoff}")

    if problems:
        raise DataValidationError("; ".join(problems))


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """Content hash, so a result can be tied to the exact bytes it came from."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def build_processed(cfg: IbmAmlConfig) -> LoadReport:
    """Raw CSV -> validated parquet + manifest (source hash, output hash, report)."""
    raw = read_raw(cfg.raw_path)
    df, report = to_standard_schema(raw, cfg.variant, cfg.tail_cutoff, cfg.rate_fit_end)
    validate(df, cfg.tail_cutoff)

    cfg.processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.processed_path, index=False)

    manifest: dict[str, Any] = {
        "dataset": "ibm_aml",
        "variant": cfg.variant,
        "source_file": cfg.raw_path.name,
        "source_sha256": file_sha256(cfg.raw_path),
        "processed_file": cfg.processed_path.name,
        "processed_sha256": file_sha256(cfg.processed_path),
        "tail_cutoff": str(cfg.tail_cutoff),
        "rate_fit_end": str(cfg.rate_fit_end),
        "report": asdict(report),
        **_git_state(),
    }
    cfg.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report


def load_processed(cfg: IbmAmlConfig) -> pd.DataFrame:
    """Load the processed parquet, refusing it if it differs from its manifest.

    The hash check means an experiment can never run on a file that was rebuilt
    or edited after the manifest it cites was written.
    """
    if not cfg.manifest_path.exists():
        raise FileNotFoundError(
            f"{cfg.manifest_path} not found; build it with "
            "`python -m nexis.data.ibm_aml --config configs/ibm_aml.yaml`"
        )
    manifest = json.loads(cfg.manifest_path.read_text(encoding="utf-8"))
    actual = file_sha256(cfg.processed_path)
    if actual != manifest["processed_sha256"]:
        raise DataValidationError(
            f"{cfg.processed_path.name} hash {actual[:12]} does not match manifest "
            f"{manifest['processed_sha256'][:12]}; rebuild the dataset"
        )
    df = pd.read_parquet(cfg.processed_path)
    validate(df, cfg.tail_cutoff)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=Path("configs/ibm_aml.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    report = build_processed(cfg)
    print(f"wrote {cfg.processed_path}")
    print(f"wrote {cfg.manifest_path}")
    for key, value in asdict(report).items():
        if key != "usd_per_unit":
            print(f"  {key:<28} {value}")


if __name__ == "__main__":
    main()
