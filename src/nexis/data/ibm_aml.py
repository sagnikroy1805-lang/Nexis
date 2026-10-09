"""IBM AML (IT-AML) dataset adapter: raw CSV -> standard schema -> processed parquet.

Implements Concept Mastery §1.1 (the transaction as row, event and edge -- the
standard schema below is the minimum that supports all three views), §3.6 and
§9.4 (nothing derived from data later than the prediction it feeds), and §22.6
(every processed artefact is traceable to its exact input by content hash).

Source: Altman et al., "Realistic Synthetic Financial Transactions for Anti-Money
Laundering Models", NeurIPS 2023 Datasets and Benchmarks; distributed on Kaggle as
ealtman2019/ibm-transactions-for-anti-money-laundering-aml.

Label semantics (PROJECT_RULES.md rule 5): `is_fraud` is the generator's `Is Laundering`
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
import re
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
    patterns_path: Path | None = None

    @property
    def manifest_path(self) -> Path:
        return self.processed_path.with_suffix(".manifest.json")

    @property
    def patterns_processed_path(self) -> Path:
        return self.processed_path.with_name(self.processed_path.stem + "_patterns.parquet")


@dataclass
class PatternReport:
    """How the documented laundering patterns line up with the kept transactions."""

    n_patterns: int
    n_pattern_rows: int
    n_rows_matched: int
    n_rows_in_dropped_tail: int
    n_patterns_truncated_by_tail: int
    n_patterns_entirely_in_tail: int
    n_positives_attributed: int
    share_of_positives_attributed: float
    patterns_by_typology: dict[str, int] = field(default_factory=dict)


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
    patterns: PatternReport | None = None


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
        patterns_path=root / ds["patterns_path"] if ds.get("patterns_path") else None,
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


_BEGIN = re.compile(r"^BEGIN LAUNDERING ATTEMPT - (?P<typology>[A-Z-]+):?\s*(?P<detail>.*)$")
# Every field the two files share. Matching on all of them, not a subset, means
# an ambiguous match is a data problem we see, not one we resolve by accident.
_PATTERN_KEY = (
    "timestamp",
    "src",
    "dst",
    "amount_paid",
    "pay_currency",
    "amount_received",
    "recv_currency",
    "payment_format",
)


def read_patterns(path: str | Path) -> pd.DataFrame:
    """Parse the laundering-patterns file: one row per pattern transaction.

    The file is a sequence of blocks:
        BEGIN LAUNDERING ATTEMPT - <TYPOLOGY>[:  <detail, e.g. "Max 10 hops">]
        <transaction rows, same 11 fields as the raw CSV>
        END LAUNDERING ATTEMPT - <TYPOLOGY>
    pattern_id is the block's 0-based position in the file. Fields are split as
    text, so identifiers keep their leading zeros exactly as in read_raw.
    """
    path = Path(path)
    records: list[tuple[Any, ...]] = []
    pattern_id, typology, detail = -1, None, ""
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if line.startswith("BEGIN"):
            match = _BEGIN.match(line)
            if match is None or typology is not None:
                raise DataValidationError(f"{path.name}:{lineno}: unexpected '{line}'")
            pattern_id += 1
            typology, detail = match["typology"], match["detail"].strip()
        elif line.startswith("END"):
            if typology is None:
                raise DataValidationError(f"{path.name}:{lineno}: END without BEGIN")
            typology = None
        else:
            fields = line.split(",")
            if typology is None or len(fields) != len(RAW_COLUMNS):
                raise DataValidationError(f"{path.name}:{lineno}: malformed row '{line}'")
            records.append((pattern_id, typology, detail, *fields))
    if typology is not None:
        raise DataValidationError(f"{path.name}: last pattern has no END line")

    raw = pd.DataFrame(records, columns=["pattern_id", "typology", "detail", *RAW_COLUMNS])
    return pd.DataFrame(
        {
            "pattern_id": raw["pattern_id"].astype("int32"),
            "typology": raw["typology"],
            "detail": raw["detail"],
            "timestamp": pd.to_datetime(raw["Timestamp"], format=TIMESTAMP_FORMAT),
            "src": raw["From Bank"] + "_" + raw["From Account"],
            "dst": raw["To Bank"] + "_" + raw["To Account"],
            "amount_paid": raw["Amount Paid"].astype(float),
            "pay_currency": raw["Payment Currency"],
            "amount_received": raw["Amount Received"].astype(float),
            "recv_currency": raw["Receiving Currency"],
            "payment_format": raw["Payment Format"],
            "is_fraud": raw["Is Laundering"].astype("int8"),
        }
    )


def match_patterns(
    df: pd.DataFrame, patterns: pd.DataFrame, tail_cutoff: pd.Timestamp
) -> tuple[pd.DataFrame, PatternReport]:
    """Link every pattern transaction to its tx_id, and check the two files agree.

    Returns one row per pattern transaction: pattern_id, typology, detail,
    timestamp, tx_id. tx_id is null for rows that fall in the dropped tail; they
    are kept because they record each pattern's true end, which is needed to
    say how much of a pattern was visible when it was first flagged.

    This table is label-derived. It is for evaluation (per-typology recall, ring
    detection, early-warning time) and must never be joined in as a feature.

    Raises if any pattern row is ambiguous, matches a negative, or is missing
    from the kept data without being in the tail. Each would mean the two files
    disagree, and any per-typology number built on top would be wrong.
    """
    key = list(_PATTERN_KEY)
    tx = df[[*key, "tx_id", "is_fraud"]].copy()
    pat = patterns.copy()
    for col in ("pay_currency", "recv_currency", "payment_format"):
        tx[col] = tx[col].astype(str)
        pat[col] = pat[col].astype(str)

    merged = pat.merge(tx, on=key, how="left", suffixes=("_pattern", ""))
    problems: list[str] = []
    if len(merged) != len(pat):
        problems.append(f"{len(merged) - len(pat)} pattern rows match more than one tx")
    if (pat["is_fraud"] != 1).any():
        problems.append("pattern file contains rows not labelled as laundering")
    matched = merged["tx_id"].notna()
    if (merged.loc[matched, "is_fraud"] != 1).any():
        problems.append(f"{int((merged.loc[matched, 'is_fraud'] != 1).sum())} pattern rows match a negative tx")
    stray = ~matched & (merged["timestamp"] < tail_cutoff)
    if stray.any():
        problems.append(f"{int(stray.sum())} pattern rows before tail_cutoff match no tx")
    if merged.loc[matched, "tx_id"].duplicated().any():
        problems.append("a tx belongs to more than one pattern")
    if problems:
        raise DataValidationError("; ".join(problems))

    mapping = (
        merged[["pattern_id", "typology", "detail", "timestamp", "tx_id"]]
        .sort_values(["pattern_id", "timestamp"], kind="stable")
        .reset_index(drop=True)
    )
    mapping["typology"] = mapping["typology"].astype("category")

    per_pattern = mapping.assign(kept=mapping["tx_id"].notna()).groupby("pattern_id")["kept"]
    n_pos = int(df["is_fraud"].sum())
    n_matched = int(matched.sum())
    report = PatternReport(
        n_patterns=int(mapping["pattern_id"].nunique()),
        n_pattern_rows=len(mapping),
        n_rows_matched=n_matched,
        n_rows_in_dropped_tail=int((~matched).sum()),
        n_patterns_truncated_by_tail=int((per_pattern.any() & ~per_pattern.all()).sum()),
        n_patterns_entirely_in_tail=int((~per_pattern.any()).sum()),
        n_positives_attributed=n_matched,
        share_of_positives_attributed=n_matched / n_pos if n_pos else 0.0,
        patterns_by_typology={
            str(k): int(v)
            for k, v in mapping.groupby("typology", observed=True)["pattern_id"]
            .nunique()
            .items()
        },
    )
    return mapping, report


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """Content hash, so a result can be tied to the exact bytes it came from."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def build_processed(cfg: IbmAmlConfig) -> LoadReport:
    """Raw CSV -> validated parquet + manifest (source hash, output hash, report)."""
    # Captured before any output is written, so the build's own files cannot
    # make the tree look dirty: this records the code that produced the data.
    code_state = _git_state()
    raw = read_raw(cfg.raw_path)
    df, report = to_standard_schema(raw, cfg.variant, cfg.tail_cutoff, cfg.rate_fit_end)
    validate(df, cfg.tail_cutoff)

    mapping = None
    if cfg.patterns_path is not None:
        mapping, report.patterns = match_patterns(
            df, read_patterns(cfg.patterns_path), cfg.tail_cutoff
        )

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
    }
    if mapping is not None and cfg.patterns_path is not None:
        mapping.to_parquet(cfg.patterns_processed_path, index=False)
        manifest |= {
            "patterns_source_file": cfg.patterns_path.name,
            "patterns_source_sha256": file_sha256(cfg.patterns_path),
            "patterns_file": cfg.patterns_processed_path.name,
            "patterns_sha256": file_sha256(cfg.patterns_processed_path),
        }
    manifest |= {"report": asdict(report), **code_state}
    cfg.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return report


def _read_verified(cfg: IbmAmlConfig, path: Path, hash_key: str) -> pd.DataFrame:
    """Read a parquet only if its bytes match the hash its manifest recorded.

    The check means an experiment can never run on a file that was rebuilt or
    edited after the manifest it cites was written.
    """
    if not cfg.manifest_path.exists():
        raise FileNotFoundError(
            f"{cfg.manifest_path} not found; build it with "
            "`python -m nexis.data.ibm_aml --config configs/ibm_aml.yaml`"
        )
    manifest = json.loads(cfg.manifest_path.read_text(encoding="utf-8"))
    if hash_key not in manifest:
        raise DataValidationError(f"manifest has no {hash_key}; rebuild the dataset")
    actual = file_sha256(path)
    if actual != manifest[hash_key]:
        raise DataValidationError(
            f"{path.name} hash {actual[:12]} does not match manifest "
            f"{manifest[hash_key][:12]}; rebuild the dataset"
        )
    return pd.read_parquet(path)


def load_processed(cfg: IbmAmlConfig) -> pd.DataFrame:
    """Load the processed transactions, verified against the manifest."""
    df = _read_verified(cfg, cfg.processed_path, "processed_sha256")
    validate(df, cfg.tail_cutoff)
    return df


def load_patterns(cfg: IbmAmlConfig) -> pd.DataFrame:
    """Load the pattern table (pattern_id, typology, detail, timestamp, tx_id).

    Label-derived: for evaluation only, never a feature. See match_patterns.
    """
    return _read_verified(cfg, cfg.patterns_processed_path, "patterns_sha256")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=Path("configs/ibm_aml.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    report = build_processed(cfg)
    print(f"wrote {cfg.processed_path}")
    if report.patterns is not None:
        print(f"wrote {cfg.patterns_processed_path}")
    print(f"wrote {cfg.manifest_path}")
    for key, value in asdict(report).items():
        if key == "patterns" and value is not None:
            for pkey, pvalue in value.items():
                print(f"  patterns.{pkey:<29} {pvalue}")
        elif key not in ("usd_per_unit", "patterns"):
            print(f"  {key:<38} {value}")


if __name__ == "__main__":
    main()
