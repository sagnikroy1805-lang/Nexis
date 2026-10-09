# NEXIS — Project Rules

Read this before writing any code in this repository.

---

## What this project is

NEXIS is a **research prototype**, not a production system. It detects coordinated
financial fraud (fraud rings, mule networks, layering chains) in time-evolving
transaction graphs, and its central claim is about **early detection at a fixed
false-positive budget**.

**Research question:** Can an adaptive temporal heterogeneous graph-learning system
detect coordinated financial fraud earlier, and with fewer false positives, than
transaction-level ML baselines?

Everything in this repo either produces evidence for that question or supports
producing it. If a proposed piece of work does neither, say so before building it.

---

## Non-negotiable rules

These are correctness rules, not style preferences. Violating any of them silently
invalidates the research results. If a request would violate one, **say so and
propose the correct alternative instead of complying**.

### 1. No temporal leakage, ever

Every prediction for time `t` may use only information that existed at `t`.

- **Never** use `train_test_split` with `shuffle=True` on transaction data.
  Use `nexis.evaluation.splits.temporal_split`.
- **Never** call `.fit()` or `.fit_transform()` on validation or test data.
  Fit preprocessing on train only, then `.transform()` the rest.
- **Never** compute graph statistics (degree, PageRank, community, centrality) on
  the complete graph and use them as features for earlier timesteps. They must be
  computed within the time window being predicted.
- **Never** build a graph without an explicit `t_cutoff` argument. Every graph
  constructor in `src/nexis/graphs/` takes one and asserts on it.
- **Never** use `closed='right'` or the default on rolling-window features. Use
  `closed='left'` so the current row is excluded.
- **Never** compute an expanding personal baseline without `.shift(1)`.
- Always keep an embargo gap between splits at least as long as the longest
  feature window.

If you write any feature-engineering or graph-construction code, add a
corresponding assertion to `tests/test_leakage.py` in the same change.

### 2. Never use accuracy as a metric

Fraud prevalence is well under 1%. Accuracy is meaningless here.

- **Primary metric:** PR-AUC (`average_precision_score`), always reported
  alongside the prevalence.
- **Secondary:** ROC-AUC, precision/recall at the alert budget, recall at fixed FPR.
- **Research metric:** early-warning time and ring detection rate.
- XGBoost `eval_metric` must be `aucpr`, never `auc`, `error` or `logloss`.

### 3. Every reported number carries a standard deviation

Run everything over at least 5 seeds (`SEEDS = (0, 1, 2, 3, 4)`) and report
`mean ± std`. A difference smaller than the seed variance is not a result, and
must not be described as one.

### 4. Baselines before complexity

The modelling ladder is fixed and must be climbed in order. Do not build rung
N+1 until rung N is fully evaluated through the harness:

1. Trivial (random, majority)
2. Rules (analyst-style thresholds)
3. Tabular ML (logistic regression, random forest, XGBoost)
4. **Tabular + behavioural features** ← this is the baseline that matters
5. Graph structural features fed to XGBoost (isolates graph *information* from graph *learning*)
6. Homogeneous GNN
7. Heterogeneous GNN
8. Temporal heterogeneous GNN
9. Fusion + ring detection

### 5. Risk scores are not accusations

The system outputs `P(label | features)`, where `label` is whatever the dataset
recorded. It does not output `P(crime)`.

- Never write code, comments, log messages, API responses or UI strings that
  assert a person committed fraud, laundered money, or acted with intent.
- Use "risk score", "the model weighted", "the system observed".
- Never "because", "proves", "is guilty of", "is laundering".

### 6. No alert without evidence

Alerts and their evidence packets are written in the same database transaction.
An alert with no `source_record_ids` is a bug.

---

## Architecture

```
src/nexis/
  data/            loading, validation, dataset adapters (IBM AML, Elliptic++, synthetic)
  features/        tabular, velocity, behavioural, personal-baseline features
  graphs/          graph construction (homogeneous, heterogeneous, temporal snapshots)
  models/
    baselines/     logistic regression, random forest, XGBoost, isolation forest
    gnn/           GCN, GraphSAGE, GAT, heterogeneous models
    temporal/      snapshot models, event-time models
  explainability/  SHAP, GNNExplainer, evidence packets
  drift/           monitors, detectors, adaptation policies
  investigation/   evidence retrieval, LLM investigator, output verification
  evaluation/      splits, metrics, harness, experiment runner   <- BUILD THIS FIRST
```

**The evaluation harness is the spine.** Every model returns scores; the harness does
splitting, metric computation, seed management and result logging. Build it before
any model, and do not let model code reimplement any part of it.

---

## Datasets

| Tier | Dataset | Role |
|---|---|---|
| 1 | **Elliptic++** | Real Bitcoin labels, 49 time steps, heterogeneous. Credibility anchor. |
| 2 | **IBM AML (IT-AML)** | Large synthetic, 8 labelled laundering typologies someone else defined. Breaks circular validation. |
| 3 | **Own generator** | Controlled experiments only: ring size, prevalence, camouflage, delay, drift onset. |

Never validate on the own generator alone — that is circular. Tier 3 exists to sweep
parameters no public dataset exposes.

Raw data lives in `data/raw/` and is **gitignored**. Never commit datasets.
Every processed artefact records a content hash in its manifest.

### Dataset decisions already made (details and numbers in `docs/dataset_card.md`)

- **Load data only through the adapters** (`nexis.data.ibm_aml.load_processed`), never
  `pd.read_csv` on raw files: the pyarrow engine strips leading zeros from bank codes
  even with `dtype=str`, merging distinct accounts.
- **IBM AML HI-Small:** rows from 2022-09-11 onward are dropped (59% positive tail, no
  background traffic). Amounts are USD, using rates estimated from day 1 only.
- **IBM AML split:** `split_on: rows`, 60/20/20, 24h embargo. Feature windows must be
  ≤ 24h on this dataset (no `7D`); `load_config` enforces it.
- Each dataset's split and feature windows live in `configs/<dataset>.yaml`, not in code.
- **XGBoost: `early_stopping_rounds=300`, `n_estimators` cap 5000.** With 100 rounds one
  seed stopped at tree 3 (PR-AUC 0.35 vs 0.52); validation PR-AUC with ~500 positives
  is noisy. When a seed's std looks wrong, check `best_iteration` first.
- **Equal tuning budget** (`evaluation.tuning_trials`, currently 6) for XGBoost and every
  GNN. Never give one family more trials than the other (§22.2).
- **Anything that pools over accounts must use only accounts active before `t_cutoff`**
  (PageRank, bank nodes). Account codes span the whole dataset; future accounts leaked
  twice before tests caught it (`test_leakage.py`, `test_gnn.py`).
- **GPU (8 GB):** GNN training samples negatives per step, scores in chunks and caps its
  memory share; on Windows an over-budget run silently spills into system RAM instead
  of failing. Run experiments through `scripts/run_all.py` (one process per step).
- **Serving scores are Platt-calibrated on the validation fold** (`scripts/replay.py`);
  the alert threshold is calibrated on validation scores, never on the test stream.
- **Drift monitoring watches stationary features only.** Cumulative features (history
  counts, graph degree, PageRank) grow by construction and always look drifted.
- **Typology labels are evaluation-only.** `load_patterns` returns them from a separate
  parquet; never join `pattern_id`, `typology` or `detail` onto features. Only 56.5% of
  IBM AML positives have a typology, so report the rest as "unattributed".

---

## Environment

- Python 3.11, managed with `uv` (or venv + pip).
- PyTorch with CUDA, PyTorch Geometric for GNNs.
- Package installed in editable mode: `pip install -e ".[dev]"`.
- Imports are absolute from the package root: `from nexis.evaluation import ...`.

Run commands through the Makefile where one exists (`make test`, `make lint`,
`make baseline`). Add a target rather than documenting a long command.

---

## Code conventions

- Type hints on every public function.
- Docstrings that state **why**, not what — especially for any line that guards
  against leakage. Those lines look removable and are not.
- Dataclasses for structured returns (`EvalResult`, `EvidencePacket`), not dicts.
- Configuration in `configs/*.yaml`, loaded into dataclasses. No magic numbers
  in model code.
- `ruff` for linting and formatting, `mypy` for types, `pytest` for tests.
- Random seeds set through `nexis.evaluation.seeds.set_all_seeds`.

---

## Testing priorities

Test the things whose failure is **silent**. In an ML project that is features,
labels and leakage — not model accuracy.

Required test categories, in priority order:

1. **Leakage tests** (`tests/test_leakage.py`) — split ordering, embargo, no future
   edges in graphs, rolling windows exclude the current row, expanding baselines
   are shifted.
2. **Feature correctness** — hand-computed expected values on small fixtures.
3. **Metric correctness** — hand-computed confusion matrices.
4. **Integration** — API returns a score, alert carries evidence.

The strongest test in the suite is the **leakage canary**: shuffle labels within
the training period only; a leak-free pipeline scores at chance. If it scores
above chance, information is flowing through a channel other than the labels.

---

## Working practices

**Before writing large amounts of code**, write down the assumptions and the file
paths to be created or modified, and agree anything architectural first.

**When implementing from the study material**, the module numbers in docs/ map to
sections of the Concept Mastery textbook. Cite the section in the docstring so the
code and the theory stay linked.

**A request that would violate a rule above** is answered with the rule it breaks
and the correct approach, never implemented silently. A correction beats a fast
wrong answer.

**When a result looks too good** (PR-AUC above ~0.95 on real data, or any metric
that jumps suddenly), treat it as a suspected leak, not a success. Check the
canary before celebrating.

**Prefer small, verifiable steps.** One module, its tests, and a demonstration that
it runs, before moving on.

---

## Out of scope

Do not add, and push back if asked:

- Kafka, Redis, or any message broker (simulate streaming by replaying a sorted file)
- Neo4j (Postgres + NetworkX + PyG covers every need; Neo4j is a stretch goal)
- MLflow or any tracking server (a structured `results/` directory with JSON is enough)
- A production Docker Compose stack (one reproducible Dockerfile is the requirement)
- Real customer data, card numbers, UPI identifiers, or fine-grained geography
- Any automated action on an account (blocking, freezing, reporting)

The concept documentation lists "adding Kafka/Redis/Neo4j before the core model
works" as a result-ruining mistake. It is right.
