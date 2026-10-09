"""Training loop for snapshot GNN edge scorers (rungs 6-8).

The scorer follows the harness Scorer protocol: fit(train, val) and score(df).
Graph context comes from a SnapshotSet built over the full time range, which is
legitimate because every snapshot only holds edges before its own period
(graphs/snapshots.py asserts it). Labels are used only for training targets in
the train fold, and for early stopping on the val fold -- never as features.

Preprocessing (EdgeFeaturePrep, node feature scaling) is fitted on training rows
and training snapshots only (rule 1).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import Tensor, nn

from nexis.graphs.snapshots import Snapshot, SnapshotSet
from nexis.models.gnn.models import (
    EdgeScorer,
    HeterogeneousEncoder,
    HomogeneousEncoder,
    TemporalEncoder,
)

Kind = Literal["homogeneous", "heterogeneous", "temporal"]
LABEL = "is_fraud"


@dataclass
class EdgeFeaturePrep:
    """signed-log -> impute -> standardise, with statistics from training rows only."""

    columns: Sequence[str]
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    nan_cols: list[int] = field(default_factory=list)

    @staticmethod
    def _log(x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.log1p(np.abs(x))

    def fit(self, train: pd.DataFrame) -> EdgeFeaturePrep:
        x = self._log(train[list(self.columns)].to_numpy(dtype=np.float64))
        self.nan_cols = [i for i in range(x.shape[1]) if np.isnan(x[:, i]).any()]
        self.mean = np.nanmean(x, axis=0)
        self.std = np.nanstd(x, axis=0) + 1e-6
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self.mean is not None and self.std is not None
        x = self._log(df[list(self.columns)].to_numpy(dtype=np.float64))
        miss = np.isnan(x[:, self.nan_cols]).astype(np.float32)
        x = np.where(np.isnan(x), self.mean, x)
        z = ((x - self.mean) / self.std).astype(np.float32)
        return np.concatenate([np.clip(z, -10, 10), miss], axis=1)

    @property
    def out_dim(self) -> int:
        return len(self.columns) + len(self.nan_cols)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class GNNScorer:
    """Snapshot GNN edge scorer. `frame` rows align with snapshots' row positions."""

    kind: Kind
    snapshots: SnapshotSet
    frame: pd.DataFrame
    edge_cols: Sequence[str]
    seed: int = 0
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    lr: float = 3e-3
    weight_decay: float = 1e-5
    epochs: int = 40
    patience: int = 6
    max_pos_weight: float = 100.0
    # Share of training negatives used per snapshot step (re-drawn every epoch).
    # Every positive is kept. Ranking metrics are unaffected by the base-rate
    # change; it cuts the edge-scorer's activation memory ~1/neg_rate times.
    neg_rate: float = 0.2
    # Cap on this process's share of GPU memory. Past it PyTorch raises an
    # out-of-memory error; without it Windows silently spills into system RAM
    # and training slows by an order of magnitude.
    gpu_memory_fraction: float = 0.9
    score_chunk: int = 200_000
    verbose: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)

    # populated by fit
    _prep: EdgeFeaturePrep | None = None
    _node_mean: Tensor | None = None
    _node_std: Tensor | None = None
    _encoder: nn.Module | None = None
    _scorer: EdgeScorer | None = None
    _edge_x: Tensor | None = None

    # ---- graph tensors ------------------------------------------------------
    def _graph(self, snap: Snapshot, dev: torch.device, recent: bool = False) -> dict:
        ss = self.snapshots
        if recent:
            ei_np, w_np = snap.recent_edges, snap.recent_weight
            typed = {f: (snap.recent_typed_edges[f], snap.recent_typed_weight[f]) for f in ss.formats}
        else:
            ei_np, w_np = snap.edge_index, snap.edge_weight
            typed = {f: (snap.typed_edges[f], snap.typed_weight[f]) for f in ss.formats}
        assert ei_np is not None and w_np is not None
        return {
            "edge_index": torch.as_tensor(ei_np, device=dev),
            "edge_weight": torch.as_tensor(w_np, device=dev),
            "typed": {
                k: (torch.as_tensor(e, device=dev), torch.as_tensor(w, device=dev))
                for k, (e, w) in typed.items()
            },
            "account_bank": torch.as_tensor(ss.account_bank, device=dev),
            "n_banks": ss.n_banks,
            "active": torch.as_tensor(snap.active, device=dev, dtype=torch.float32),
        }

    def _node_x(self, snap: Snapshot, dev: torch.device) -> Tensor:
        x = torch.as_tensor(snap.node_x, device=dev)
        assert self._node_mean is not None and self._node_std is not None
        return (x - self._node_mean) / self._node_std

    def _positions(self, df: pd.DataFrame) -> np.ndarray:
        pos = pd.Index(self.frame["tx_id"]).get_indexer(df["tx_id"])
        assert (pos >= 0).all(), "rows not in the scorer's frame"
        return pos

    # ---- model --------------------------------------------------------------
    def _build(self, node_in: int, edge_in: int) -> None:
        rel = list(self.snapshots.formats)
        if self.kind == "homogeneous":
            self._encoder = HomogeneousEncoder(node_in, self.hidden, self.layers, self.dropout)
        elif self.kind == "heterogeneous":
            self._encoder = HeterogeneousEncoder(node_in, rel, self.hidden, self.layers, self.dropout)
        else:
            self._encoder = TemporalEncoder(node_in, rel, self.hidden, self.layers, self.dropout)
        self._scorer = EdgeScorer(self.hidden, edge_in, self.dropout)

    def _embed_all(
        self, dev: torch.device, train: bool, grad_snaps: set[int] | None = None
    ):
        """Yield (snapshot index, embeddings) in time order.

        Static encoders embed each snapshot independently. The temporal encoder
        threads its state forward; gradients flow only within a snapshot
        (truncated BPTT of length 1) to bound memory on 500k accounts.
        """
        assert self._encoder is not None
        state = torch.zeros(self.snapshots.n_accounts, self.hidden, device=dev)
        for i, snap in enumerate(self.snapshots.snapshots):
            need_grad = train and (grad_snaps is None or i in grad_snaps)
            with torch.set_grad_enabled(need_grad):
                x = self._node_x(snap, dev)
                if self.kind == "temporal":
                    g = self._graph(snap, dev, recent=True)
                    state = self._encoder(x, g, state.detach())
                    yield i, state
                else:
                    if grad_snaps is not None and not need_grad and train:
                        continue
                    yield i, self._encoder(x, self._graph(snap, dev))

    # ---- protocol -----------------------------------------------------------
    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> GNNScorer:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)  # noqa: NPY002 - seeds torch_geometric internals too
        dev = _device()
        if dev.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)
        rng = np.random.default_rng(self.seed)
        self._prep = EdgeFeaturePrep(self.edge_cols).fit(train)
        self._edge_x = torch.as_tensor(self._prep.transform(self.frame), device=dev)

        tr_pos, va_pos = self._positions(train), self._positions(val)
        y = torch.as_tensor(self.frame[LABEL].to_numpy(dtype=np.float32), device=dev)
        is_tr = np.zeros(len(self.frame), bool)
        is_tr[tr_pos] = True
        is_va = np.zeros(len(self.frame), bool)
        is_va[va_pos] = True

        snaps = self.snapshots.snapshots
        train_snaps = {i for i, s in enumerate(snaps) if is_tr[s.target_rows].any()}
        # Node-feature scaling from training snapshots only.
        stacked = np.concatenate([snaps[i].node_x for i in sorted(train_snaps)])
        self._node_mean = torch.as_tensor(stacked.mean(0), device=dev)
        self._node_std = torch.as_tensor(stacked.std(0) + 1e-6, device=dev)

        self._build(snaps[0].node_x.shape[1], self._prep.out_dim)
        assert self._encoder is not None and self._scorer is not None
        self._encoder.to(dev)
        self._scorer.to(dev)
        params = list(self._encoder.parameters()) + list(self._scorer.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

        labels_np = self.frame[LABEL].to_numpy()
        n_pos = float(labels_np[tr_pos].sum())
        n_neg_used = (len(tr_pos) - n_pos) * self.neg_rate
        pos_weight = min(n_neg_used / max(n_pos, 1.0), self.max_pos_weight)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dev))
        src = torch.as_tensor(self.snapshots.src, device=dev)
        dst = torch.as_tensor(self.snapshots.dst, device=dev)

        best, best_state, bad = -1.0, None, 0
        for epoch in range(self.epochs):
            t0 = time.time()
            self._encoder.train()
            self._scorer.train()
            total = 0.0
            for i, h in self._embed_all(dev, train=True, grad_snaps=train_snaps):
                rows = snaps[i].target_rows[is_tr[snaps[i].target_rows]]
                keep = (labels_np[rows] == 1) | (rng.random(len(rows)) < self.neg_rate)
                rows = rows[keep]
                if len(rows) == 0:
                    continue
                r = torch.as_tensor(rows, device=dev)
                logits = self._scorer(h, src[r], dst[r], self._edge_x[r])
                loss = loss_fn(logits, y[r])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                total += float(loss.detach())
            val_scores = self._score_positions(va_pos, dev)
            ap = float(average_precision_score(self.frame[LABEL].to_numpy()[va_pos], val_scores))
            self.history.append({"epoch": epoch, "loss": total, "val_pr_auc": ap, "secs": time.time() - t0})
            if self.verbose:
                print(f"    epoch {epoch:2d} loss {total:.3f} val PR-AUC {ap:.4f} ({time.time() - t0:.0f}s)")
            if ap > best:
                best, bad = ap, 0
                best_state = (
                    {k: v.detach().clone() for k, v in self._encoder.state_dict().items()},
                    {k: v.detach().clone() for k, v in self._scorer.state_dict().items()},
                )
            else:
                bad += 1
                if bad >= self.patience:
                    break
        assert best_state is not None
        self._encoder.load_state_dict(best_state[0])
        self._scorer.load_state_dict(best_state[1])
        return self

    @torch.no_grad()
    def _score_positions(self, positions: np.ndarray, dev: torch.device) -> np.ndarray:
        assert self._encoder is not None and self._scorer is not None and self._edge_x is not None
        self._encoder.eval()
        self._scorer.eval()
        want = np.zeros(len(self.frame), bool)
        want[positions] = True
        out = np.full(len(self.frame), np.nan, np.float32)
        src = torch.as_tensor(self.snapshots.src, device=dev)
        dst = torch.as_tensor(self.snapshots.dst, device=dev)
        for i, h in self._embed_all(dev, train=False):
            rows = self.snapshots.snapshots[i].target_rows
            rows = rows[want[rows]]
            for c0 in range(0, len(rows), self.score_chunk):
                chunk = rows[c0 : c0 + self.score_chunk]
                r = torch.as_tensor(chunk, device=dev)
                out[chunk] = torch.sigmoid(self._scorer(h, src[r], dst[r], self._edge_x[r])).cpu().numpy()
        scores = out[positions]
        assert not np.isnan(scores).any(), "some rows fell outside every snapshot"
        return scores

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self._score_positions(self._positions(df), _device())

    def embeddings_at(self, period_start: pd.Timestamp) -> np.ndarray:
        """Account embeddings of the snapshot starting at period_start (for explanation)."""
        dev = _device()
        for i, h in self._embed_all(dev, train=False):
            if self.snapshots.snapshots[i].period_start == period_start:
                return h.cpu().numpy()
        raise KeyError(period_start)
