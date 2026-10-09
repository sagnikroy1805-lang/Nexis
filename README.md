# NEXIS

Adaptive AI financial crime intelligence platform — research prototype.

**Research question:** Can an adaptive temporal heterogeneous graph-learning system
detect coordinated financial fraud earlier, and with fewer false positives, than
transaction-level ML baselines?

---

## Quick start

```bash
git clone <your-repo> nexis && cd nexis

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 1. PyTorch first, matched to your CUDA version -- check with nvidia-smi
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 2. Then the project
pip install -e ".[dev]"

# 3. Then PyG (needs torch already present)
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.7.0+cu126.html

# 4. Verify
make test
make baseline
```

`make baseline` runs the full spine — temporal split, behavioural features, three
tabular models, five seeds, aggregated results — on generated demo data. If that
prints a table, your environment is correct.

### Checking your CUDA version

```bash
nvidia-smi                                        # driver's max supported CUDA
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import torch; print(torch.cuda.is_available())"
```

The PyG wheel index URL must match your **torch** version and **CUDA** version
exactly. `torch-2.7.0+cu126` means torch 2.7.0 with CUDA 12.6. A mismatch here is
the most common setup failure; if `import torch_geometric` segfaults or complains
about undefined symbols, this is why.

PyG works without the optional extensions (`pyg_lib` and friends) — they add
faster heterogeneous operators and sparse kernels. Skip them if the wheels fight
you; add them when you reach Module 10.

---

## Layout

```
configs/          YAML experiment configs
data/raw/         downloaded datasets (gitignored)
data/synthetic/   generated data (gitignored)
docs/             study material, architecture notes
notebooks/        exploration only -- production code lives in src/
results/          experiment outputs; manifests are committed
scripts/          runnable entry points
src/nexis/
  data/           loading, validation, dataset adapters
  features/       tabular, velocity, behavioural, personal-baseline
  graphs/         graph construction (homogeneous, heterogeneous, snapshots)
  models/         baselines/, gnn/, temporal/
  explainability/ SHAP, GNNExplainer, evidence packets
  drift/          monitors, detectors, adaptation
  investigation/  evidence retrieval, LLM investigator
  evaluation/     splits, metrics, harness, leakage    <- the spine
tests/            leakage tests first, then features, metrics, integration
```

---

## What already works

| Component | Status |
|---|---|
| `data/ibm_aml.py` | IBM AML adapter: standard schema, USD amounts, tail cut, typology table, hashed manifest (`make data-ibm`) |
| `evaluation/splits.py` | Temporal split (time-span or row-count) with embargo, walk-forward CV, integrity assertions |
| `evaluation/metrics.py` | PR-AUC, alert-budget thresholds, recall@FPR, early-warning time, ring metrics |
| `evaluation/harness.py` | Multi-seed runner, git provenance, mean ± std aggregation |
| `evaluation/leakage.py` | Future-edge assertion, leakage canary |
| `features/velocity.py` | Rolling velocity, inter-arrival, personal baselines, HHI |
| `tests/` | 42 passing tests, leakage guards included |
| `scripts/run_baseline.py` | End-to-end tabular baseline experiment |
| `scripts/dataset_stats.py` | Dataset-card statistics (`make stats-ibm`) |

Everything else is yours to build. The order is fixed — see `CLAUDE.md`.

---

## Datasets

| Tier | Dataset | Purpose |
|---|---|---|
| 1 | [Elliptic++](https://github.com/git-disl/EllipticPlusPlus) | Real labels, 49 time steps, heterogeneous. Credibility anchor. |
| 2 | [IBM AML](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) | 8 labelled laundering typologies you did not design. |
| 3 | Own generator | Controlled sweeps: ring size, prevalence, camouflage, drift. |

Start with an IBM AML **Small** variant (~2–5M transactions). Do not download a
Large variant (175M+) until you have a reason.

---

## The rules

Full detail in `CLAUDE.md`. The short version:

1. No temporal leakage — time-aware splits, embargo, `t_cutoff` on every graph.
2. Never accuracy — PR-AUC primary, always with prevalence.
3. Five seeds minimum, mean ± std, always.
4. Baselines before complexity — climb the ladder in order.
5. Risk scores, not accusations.
6. No alert without evidence.

## Out of scope

Kafka, Redis, Neo4j, MLflow, production Docker Compose. Simulate streaming by
replaying a sorted file. These are deferred deliberately — adding them before the
core model works is the single most reliable way to run out of time.

---

## Commands

```bash
make test        # full suite
make test-fast   # skip slow tests
make lint        # ruff
make typecheck   # mypy
make baseline    # end-to-end baseline experiment
```
