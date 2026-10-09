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

## Statistics (after processing)

Regenerate with `make stats-ibm` (`scripts/dataset_stats.py`). Every number below
comes from that script.

### Amounts (USD)

| | median | p90 | p99 | max | skew | skew of log1p |
|---|---|---|---|---|---|---|
| all | 892.90 | 35,407 | 3,100,925 | 26,558,612,966 | 556.0 | 0.51 |
| negatives | 890.87 | 35,451 | 3,099,589 | 26,558,612,966 | 573.8 | 0.51 |
| positives | 5,074.02 | 18,384 | 19,360,301 | 16,297,045,517 | 63.4 | 1.28 |

- **Use `log1p(amount)`** for anything scale-sensitive (logistic regression, GNN
  inputs). The raw skew is 556; after the log it is 0.51.
- The median positive is **5.7×** the median negative. Amount alone is a strong
  signal, so the tabular baseline will not be weak.

### Accounts

- 515,078 accounts; 496,973 of them send at least once.
- Transactions per account (as sender or receiver): median 6, p90 51, p99 119,
  p99.9 190, **max 169,756**.
- 17,480 accounts (3.4%) appear exactly once.
- The busiest 1% of accounts take part in 12.0% of all transactions.
- **One hub account dwarfs the rest.** Full-neighbourhood message passing would
  explode on it, so GNNs need neighbour sampling (§8.3), and raw degree features will
  be dominated by a few nodes.

### Time

- 14,400 distinct timestamps, i.e. every minute of the 10 days is occupied.
- Rows per minute: median 326, p99 813, max 11,193.
- Rows per hour: median 19,515, max 344,208, no empty hours.

### Laundering patterns (`HI-Small_Patterns.txt`)

370 documented patterns, 3,209 transactions. Every pattern row matches exactly one
transaction, and every match is labelled positive. The 655 rows that match nothing
are exactly the 655 positives in the dropped tail. `match_patterns` raises if any of
this stops being true.

| typology | patterns | tx | tx kept | median tx | median hours | p90 hours | max hours |
|---|---|---|---|---|---|---|---|
| BIPARTITE | 49 | 263 | 241 | 4.0 | 24.3 | 43.1 | 47.4 |
| CYCLE | 54 | 287 | 243 | 4.0 | 71.9 | 90.3 | 95.6 |
| FAN-IN | 40 | 318 | 252 | 8.0 | 84.9 | 94.7 | 95.8 |
| FAN-OUT | 48 | 342 | 276 | 7.0 | 76.7 | 94.7 | 96.0 |
| GATHER-SCATTER | 51 | 716 | 462 | 14.0 | 150.8 | 183.8 | 202.3 |
| RANDOM | 41 | 191 | 150 | 3.0 | 46.0 | 86.6 | 91.6 |
| SCATTER-GATHER | 44 | 626 | 497 | 14.0 | 88.8 | 95.5 | 95.9 |
| STACK | 43 | 466 | 433 | 10.0 | 73.3 | 101.0 | 114.0 |
| **all** | **370** | **3,209** | **2,554** | **6.5** | **74.7** | **110.2** | **202.3** |

- **Only 56.5% of positives belong to a listed pattern** (2,554 of 4,522). The other
  1,968 are labelled laundering with no documented typology. Per-typology results
  therefore cover about half the positives; report the rest as their own
  "unattributed" group, never silently dropped.
- **102 patterns are cut short by the tail cut-off and 7 lie entirely after it.** The
  pattern table keeps their tail rows (with a null `tx_id`), so each pattern's true end
  is still known.
- **Patterns last days, not hours** (median 75h, up to 202h). This sets Δ for the
  Module 9 snapshots. With Δ = 24h, a median pattern spans 3–4 snapshots, which is
  enough for a temporal model to see it develop. With Δ = 6h it spans about 12.
- **Patterns cross the train/val/test boundaries.** By the folds their kept
  transactions fall in: 104 patterns are train only, 53 test only, 4 val only. **26 lie
  entirely in embargo gaps and are never evaluated.** The rest span two or more folds.
  For the early-warning metric (§4.7, §9.5), a pattern's start is its first transaction
  *in the full table*, even if that falls in an earlier fold.
- The pattern table is **label-derived**. It lives in a separate parquet
  (`*_patterns.parquet`, loaded with `load_patterns`), never in the transactions table,
  so no model fitted on "all columns" can read it. `tests/test_leakage.py` guards this.
