# NEXIS

Adaptive AI financial crime intelligence platform — research prototype.

**Research question:** Can an adaptive temporal heterogeneous graph-learning system
detect coordinated financial fraud earlier, and with fewer false positives, than
transaction-level ML baselines?

Results: [`docs/results.md`](docs/results.md) · Data: [`docs/dataset_card.md`](docs/dataset_card.md) ·
API: [`docs/api_contract.md`](docs/api_contract.md) · Rules: [`PROJECT_RULES.md`](PROJECT_RULES.md)

---

## Headline results (test folds, mean ± std over 5 seeds — full tables in `docs/results.md`)

| IBM AML HI-Small (prevalence 0.147%) | PR-AUC |
|---|---|
| Rules (rung 2) | 0.0143 |
| XGBoost, transaction fields (rung 3) | 0.0971 ± 0.0022 |
| XGBoost + strict-past behaviour (rung 4) | 0.5285 ± 0.0022 |
| XGBoost + leak-free graph structure (rung 5) | **0.5598 ± 0.0034** |
| Best GNN, homogeneous (rung 6) | 0.2858 ± 0.0124 |
| Fusion XGBoost + GNN (rung 9) | 0.5647 ± 0.0023 (not distinguishable from rung 5) |

- Leakage canary (shuffled training labels): PR-AUC / chance = **0.997**.
- Patterns flagged before they complete: **88%**, median ~43 h early.
- Elliptic++ (tier 1): XGBoost 0.7945 ± 0.0017; per-step PR-AUC collapses from ~0.9 to
  ~0.05 after the step-43 dark-market shutdown — real-world drift.
- Synthetic drift (tier 3): 4/5 shifts detected 24 h after onset, 0 false alarms.

---

## What is in the box

| Layer | Where | What it does |
|---|---|---|
| Data | `src/nexis/data/` | IBM AML adapter (tier 2), Elliptic++ adapter (tier 1), synthetic generator with drift (tier 3); hashed manifests |
| Features | `src/nexis/features/` | strict-past sender/receiver behaviour (velocity, pass-through, fan-in, personal baselines, counterparty novelty) |
| Graphs | `src/nexis/graphs/` | `t_cutoff`-asserted account graphs, structural features, leak-free snapshots typed by payment format |
| Models | `src/nexis/models/` | ladder rungs 1–9: rules, LR/RF/XGBoost, Isolation Forest, homogeneous / heterogeneous / temporal GNNs, fusion, two-stage ring detection |
| Evaluation | `src/nexis/evaluation/` | temporal splits with embargo, PR-AUC-first metrics, 5-seed harness with git provenance, leakage canary |
| Explanation | `src/nexis/explainability/` | TreeSHAP contributions, evidence packets (rule 6) |
| Drift | `src/nexis/drift/` | reference-frozen PSI, Page-Hinkley, ADWIN, adaptation policies, walk-forward experiment |
| Investigator | `src/nexis/investigation/` | retrieval-first LLM summary with mechanical claim verification; template fallback |
| Serving | `src/nexis/db/`, `src/nexis/api/`, `scripts/replay.py` | PostgreSQL schema, streaming replay, FastAPI |
| Dashboard | `frontend/` | React + Tailwind analyst UI: dashboard, alerts, explanation, graph explorer, rings, drift, models |

---

## Setup (Windows, NVIDIA GPU)

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev,api,gnn]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`make` is not installed on Windows by default. Either install it
(`winget install ezwinports.make`) or run the command each target shows in the `Makefile`.

### Data

Download into `data/raw/` (gitignored):
- IBM AML: `HI-Small_Trans.csv` and `HI-Small_Patterns.txt` from
  [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).
- Elliptic++: `txs_features.csv`, `txs_classes.csv`, `txs_edgelist.csv` from the
  [dataset's Drive folder](https://github.com/git-disl/EllipticPlusPlus) into `data/raw/elliptic/`.

### Database

```bash
make db-up        # PostgreSQL 16 in Docker on localhost:5432 (user/password/db: nexis)
copy .env.example .env
```

Without `NEXIS_DATABASE_URL` everything falls back to SQLite at `data/nexis.db`.

### Investigator (optional)

Create a key at [console.anthropic.com](https://console.anthropic.com) → Settings → API Keys,
then store it in your user environment (never in the repo):

```bash
setx ANTHROPIC_API_KEY "sk-ant-..."
```

Open a new terminal afterwards. Without a key the investigator writes a template summary.

---

## Running the whole pipeline

```bash
make data-ibm     # raw CSV -> validated parquet + manifest
make test         # leakage guards first
make ladder       # rungs 1-5, 5 seeds each, + leakage canary
make gnn          # rungs 6-8
make rung9        # fusion, ring detection, early-warning time
make ablation     # feature-group ablation
make drift        # drift experiment on the synthetic generator
python scripts/run_elliptic.py   # tier-1 ladder on Elliptic++
make results      # -> docs/results.md
make replay       # stream the test period into the database
make api          # http://localhost:8000
make frontend     # http://localhost:5173
```

---

## The rules

Full detail in `PROJECT_RULES.md`:

1. No temporal leakage — time-aware splits, embargo, `t_cutoff` on every graph.
2. Never accuracy — PR-AUC primary, always with prevalence.
3. Five seeds minimum, mean ± std, always.
4. Baselines before complexity — climb the ladder in order.
5. Risk scores, not accusations.
6. No alert without evidence.

Out of scope by design: Kafka, Redis, Neo4j, MLflow, a production Docker Compose stack,
real customer data, and any automated action on an account.
