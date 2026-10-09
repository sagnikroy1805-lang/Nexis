# Dataset card — IBM AML (HI-Small)

Tier 2 dataset (see `CLAUDE.md`). Built by `make data-ibm` from
`configs/ibm_aml.yaml`; the exact bytes are pinned by
`data/processed/ibm_aml_hi_small.manifest.json`.

## Source

- Altman et al., *Realistic Synthetic Financial Transactions for Anti-Money
  Laundering Models*, NeurIPS 2023 Datasets & Benchmarks.
- Kaggle: `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`, file
  `HI-Small_Trans.csv` (475,664,283 bytes, 5,078,345 rows, 11 columns).
- **HI** = higher illicit ratio. The LI variant is the harder, lower-prevalence
  follow-up.
- **Label:** `Is Laundering`, meaning the generator placed the transaction inside one of
  its laundering patterns. It is a simulator label, not a finding (rule 5).

## Raw file, as downloaded

| | |
|---|---|
| Rows | 5,078,345 |
| Time span | 2022-09-01 00:00 → 2022-09-18 16:18 (minute resolution) |
| Positives | 5,177 (0.102%) |
| Accounts (`bank_account`) | 515,088 |
| Currencies | 15 |
| Nulls | 0 |
| Non-positive amounts | 0 |
| Exact duplicate rows | 9 |
| Self-transfers (src = dst) | 591,212, of which 11 positive |
| Cross-currency rows | 72,170 |

Positives by payment format:

| Format | Rows | Positives |
|---|---|---|
| ACH | 600,797 | 4,483 |
| Cheque | 1,864,331 | 324 |
| Credit Card | 1,323,324 | 206 |
| Cash | 490,891 | 108 |
| Bitcoin | 146,091 | 56 |
| Reinvestment | 481,056 | 0 |
| Wire | 171,855 | 0 |

## Processing decisions

### 1. Tail cut at 2022-09-11 00:00

Simulated normal activity ends on 10 Sep, but laundering chains keep completing
for eight more days with almost no background traffic:

| | Rows | Positives |
|---|---|---|
| 1–10 Sep | 5,077,237 | 4,522 |
| 11–18 Sep (dropped) | 1,108 | 655 (59%) |

Kept, these rows would form a test fold at roughly 60% prevalence and an inflated PR-AUC.
Dropping them removes 12.7% of positives; the loader's report records the exact
counts.

### 2. Amounts converted to USD, with rates estimated from day 1 only

The file publishes no exchange rates, but every cross-currency row implies one.
`amount` is `Amount Paid × USD-per-unit(payment currency)`. Each rate is the
median implied rate over cross-currency rows with USD on one side, **using only
rows before 2022-09-02**, so no later data shapes a training feature (rule 1).

The rates are constant in the generator: the daily median for USD↔EUR is
identical on all 18 days, and day 1 contains all 15 currencies. Estimated rates
are in the manifest (e.g. EUR 1.1718, GBP 1.2917, INR 0.01362, BTC 11,875).

### 3. Identifiers kept as strings

Accounts are `"<bank>_<account>"`, because an account number is only unique within its
bank. Bank codes have leading zeros (`010`, `001`, `01`). Reading them with
`pd.read_csv(engine="pyarrow", dtype=str)` silently turns `001` and `01` into the
same `"1"`. The loader declares the column types to pyarrow up front instead;
`tests/test_data.py` guards this.

### 4. Split: row-count quantiles, 24h embargo, feature windows ≤ 24h

After the cut the dataset covers only 10 days, and daily volume is very uneven
(1.11M rows on day 1, about 207k on weekends). Splitting the *time span* 60/20/20
would leave a single weekend day as the test fold.

`split_on: rows` places the boundaries at row-count quantiles, still in time
order. Rows that share the boundary timestamp all go to the earlier fold.

| Fold | Rows | Positives | Prevalence | Window |
|---|---|---|---|---|
| train | 3,046,524 | 2,297 | 0.0754% | 09-01 00:00 → 09-06 13:34 |
| val | 533,396 | 559 | 0.1048% | 09-07 13:35 → 09-08 16:09 |
| test | 416,866 | 613 | 0.1470% | 09-09 16:10 → 09-10 23:59 |
| *embargo gaps* | *1,080,451* | | | |

Rule 1 requires the embargo to be at least the longest feature window, so a 7-day
velocity window is impossible on 10 days. For this dataset the windows are
`5min, 1h, 24h`, and `load_config` refuses any config that breaks this.

## Things to state in the paper

- **Prevalence rises over time** (0.075% → 0.105% → 0.147%), so validation and test
  PR-AUC are not directly comparable. Always report prevalence next to PR-AUC (rule 2).
- **The embargo costs 21% of rows.** That is the price of leak-free 24h features on a
  10-day dataset.
- **Minute resolution creates many ties.** Time-based rolling windows with
  `closed='left'` exclude every row at the current timestamp, so same-minute rows never
  see each other. Expanding baselines that use `.shift(1)` follow row order, so they
  *do* see earlier same-minute rows, in raw-file order. Decide on this before the
  Day-4 features: group by `(src, timestamp)`, or accept it and state it.
- **Bitemporality (§17.1.1):** the data has only an occurrence time, so occurrence
  and availability are treated as identical.
- **Self-transfers** (11.6% of rows, almost all Reinvestment) are kept and flagged
  with `is_self_transfer`. Whether models see them is a modelling decision.

## Still to do (Day 3)

- Duration distribution of labelled laundering patterns. This sets the snapshot
  window Δ for Module 9. It needs `HI-Small_Patterns.txt`.
- Transactions per typology (fan-in, fan-out, cycle, scatter-gather, …). Also from the
  patterns file.
- Amount distribution (raw and `log1p`) and skew.
- Transactions per account: how heavy is the tail?
