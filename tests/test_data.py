"""IBM AML adapter tests against a hand-built 12-row fixture.

Fixture rows (0-based raw row number = tx_id suffix):
    0  self-transfer, bank 010
    1  bank 001 -> 010, 00:05            } same minute, file order 1 then 2;
    2  bank 01  -> 010, 00:05            } "001" and "01" are different banks
    3  pays 100 USD, receives 80 EUR     -> 1.25 USD per EUR
    4  pays 100 EUR, receives 125 USD    -> 1.25 USD per EUR
    5  all-digit account with a leading zero (012345678)
    6  pays 1000 EUR, positive           -> amount 1250 USD
    7  day 3: implies 0.1 USD per EUR     -> after rate_fit_end, must be ignored
    8  2022-09-10 23:59, positive        -> last kept row
    9  2022-09-11 00:00, positive        -> exactly at the cutoff, dropped
   10  2022-09-12, positive              -> tail, dropped
   11  exact duplicate of row 5
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nexis.data.ibm_aml import (
    DataValidationError,
    IbmAmlConfig,
    build_processed,
    load_config,
    load_processed,
    read_raw,
    to_standard_schema,
    validate,
)
from nexis.evaluation.splits import SplitConfig

FIXTURE = Path(__file__).parent / "fixtures" / "ibm_aml_sample.csv"
TAIL_CUTOFF = pd.Timestamp("2022-09-11")
RATE_FIT_END = pd.Timestamp("2022-09-02")


@pytest.fixture
def loaded() -> tuple[pd.DataFrame, object]:
    return to_standard_schema(read_raw(FIXTURE), "TEST", TAIL_CUTOFF, RATE_FIT_END)


def _row(df: pd.DataFrame, raw_row: int) -> pd.Series:
    return df.loc[df["tx_id"] == f"TEST:{raw_row:07d}"].iloc[0]


def test_bank_codes_keep_leading_zeros(loaded):
    """'001' and '01' are different banks; numeric inference would merge them."""
    df, _ = loaded
    assert _row(df, 1)["src"] == "001_8000A0001"
    assert _row(df, 2)["src"] == "01_8000A0001"
    assert _row(df, 0)["src"] == "010_8000EBD30"


def test_all_digit_account_keeps_leading_zero(loaded):
    df, _ = loaded
    assert _row(df, 5)["src"] == "040_012345678"


def test_sender_and_receiver_columns_are_not_conflated(loaded):
    """The raw header repeats 'Account'; the receiver must come from the 2nd one."""
    df, _ = loaded
    r = _row(df, 3)
    assert (r["src"], r["dst"]) == ("020_8000C0003", "030_8000D0004")
    assert _row(df, 0)["is_self_transfer"] == 1
    assert _row(df, 1)["is_self_transfer"] == 0


def test_amounts_are_converted_to_usd(loaded):
    df, report = loaded
    assert report.usd_per_unit["Euro"] == pytest.approx(1.25)
    assert _row(df, 6)["amount"] == pytest.approx(1250.0)
    assert _row(df, 1)["amount"] == pytest.approx(100.0)
    assert _row(df, 6)["is_cross_currency"] == 0
    assert _row(df, 3)["is_cross_currency"] == 1


def test_tail_is_dropped_and_reported(loaded):
    df, report = loaded
    assert report.n_raw == 12
    assert report.n_dropped_tail == 2
    assert report.n_pos_dropped_tail == 2
    assert report.n_rows == 10
    assert report.n_pos == 2
    assert df["timestamp"].max() < TAIL_CUTOFF


def test_rows_sorted_with_ties_in_file_order(loaded):
    df, _ = loaded
    assert df["timestamp"].is_monotonic_increasing
    first_three = df["tx_id"].head(3).tolist()
    assert first_three == ["TEST:0000001", "TEST:0000002", "TEST:0000000"]


def test_exact_duplicate_rows_are_kept_distinct(loaded):
    df, report = loaded
    assert report.n_exact_duplicate_raw_rows == 1
    assert df["tx_id"].is_unique
    assert {"TEST:0000005", "TEST:0000011"} <= set(df["tx_id"])


def test_validate_accepts_clean_output(loaded):
    df, _ = loaded
    validate(df, TAIL_CUTOFF)


def test_validate_reports_every_problem(loaded):
    df, _ = loaded
    bad = df.copy()
    bad.loc[0, "amount"] = 0.0
    bad.loc[1, "tx_id"] = bad.loc[2, "tx_id"]
    bad = bad.iloc[::-1].reset_index(drop=True)
    with pytest.raises(DataValidationError) as err:
        validate(bad)
    msg = str(err.value)
    assert "non-positive" in msg
    assert "duplicate tx_id" in msg
    assert "not sorted" in msg


def test_validate_rejects_nulls(loaded):
    df, _ = loaded
    bad = df.copy()
    bad.loc[3, "dst"] = None
    with pytest.raises(DataValidationError, match="nulls"):
        validate(bad)


def test_unexpected_header_is_rejected(tmp_path):
    p = tmp_path / "wrong.csv"
    p.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="unexpected header"):
        read_raw(p)


def test_currency_without_rate_is_rejected(tmp_path):
    """A currency with no USD cross row before rate_fit_end has no defensible rate."""
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    lines.append(
        "2022/09/05 12:00,060,8000F0006,060,8000F0007,"
        "900.00,Yen,900.00,Yen,Cash,0"
    )
    p = tmp_path / "with_yen.csv"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="Yen"):
        to_standard_schema(read_raw(p), "TEST", TAIL_CUTOFF, RATE_FIT_END)


def _cfg(tmp_path: Path) -> IbmAmlConfig:
    return IbmAmlConfig(
        variant="TEST",
        raw_path=FIXTURE,
        processed_path=tmp_path / "test.parquet",
        tail_cutoff=TAIL_CUTOFF,
        rate_fit_end=RATE_FIT_END,
        split=SplitConfig(),
        feature_windows=("1h",),
    )


def test_build_then_load_roundtrip(tmp_path, loaded):
    expected, _ = loaded
    cfg = _cfg(tmp_path)
    report = build_processed(cfg)
    assert cfg.manifest_path.exists()
    df = load_processed(cfg)
    assert len(df) == report.n_rows
    assert df["tx_id"].tolist() == expected["tx_id"].tolist()
    assert df["amount"].tolist() == pytest.approx(expected["amount"].tolist())


def test_load_refuses_a_file_that_changed_after_its_manifest(tmp_path):
    cfg = _cfg(tmp_path)
    build_processed(cfg)
    df = pd.read_parquet(cfg.processed_path)
    df.loc[0, "amount"] = 999.0
    df.to_parquet(cfg.processed_path, index=False)
    with pytest.raises(DataValidationError, match="does not match manifest"):
        load_processed(cfg)


def test_repo_config_loads_and_respects_rule_1():
    cfg = load_config(Path(__file__).parent.parent / "configs" / "ibm_aml.yaml")
    assert cfg.split.split_on == "rows"
    assert max(pd.Timedelta(w) for w in cfg.feature_windows) <= pd.Timedelta(
        cfg.split.embargo
    )
    assert cfg.raw_path.name == "HI-Small_Trans.csv"
