"""Node-classification GNN for Elliptic++ (Concept Mastery §8.3, §8.5, §8.7).

Elliptic's edges join transactions of the same time step only, so the union of
all steps is a disjoint union of per-step graphs: one full-graph pass cannot
carry information across the time-step split. load_elliptic asserts this.

Labels: training uses labelled nodes of training steps only; unknown nodes stay
in the graph as context but never enter the loss or the metrics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch_geometric.nn import SAGEConv


class NodeSAGE(nn.Module):
    """GraphSAGE with separate root/neighbour weights and sum aggregation.

    Separate weights keep the model usable under heterophily (§8.7); sum keeps
    the neighbourhood size, which mean would discard (§8.1.1).
    """

    def __init__(self, d_in: int, dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, dim)
        self.convs = nn.ModuleList(SAGEConv(dim, dim, aggr="sum") for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(dim + d_in, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.proj(x))
        for conv, norm in zip(self.convs, self.norms, strict=True):
            h = h + self.drop(torch.relu(norm(conv(h, ei) + conv(h, ei.flip(0)))))
        return self.head(torch.cat([h, x], dim=1)).squeeze(-1)


@dataclass
class NodeGNNScorer:
    nodes: pd.DataFrame  # tx_id, time_step, label, features
    edges: pd.DataFrame  # src, dst (tx ids)
    feature_cols: Sequence[str]
    seed: int = 0
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    lr: float = 3e-3
    epochs: int = 200
    patience: int = 20
    history: list[dict[str, Any]] = field(default_factory=list)
    _model: NodeSAGE | None = None
    _x: torch.Tensor | None = None
    _ei: torch.Tensor | None = None

    def _setup(self, train_mask: np.ndarray, dev: torch.device) -> None:
        x = self.nodes[list(self.feature_cols)].to_numpy(dtype=np.float64)
        # Standardise with training-step statistics only (rule 1).
        mu = np.nanmean(x[train_mask], axis=0)
        sd = np.nanstd(x[train_mask], axis=0) + 1e-6
        x = np.nan_to_num((x - mu) / sd).clip(-10, 10)
        self._x = torch.as_tensor(x, dtype=torch.float32, device=dev)
        pos = pd.Series(np.arange(len(self.nodes)), index=self.nodes["tx_id"].to_numpy())
        ei = np.vstack([pos.reindex(self.edges["src"]).to_numpy(), pos.reindex(self.edges["dst"]).to_numpy()])
        ei = ei[:, ~np.isnan(ei).any(axis=0)].astype(np.int64)
        self._ei = torch.as_tensor(ei, device=dev)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> NodeGNNScorer:
        torch.manual_seed(self.seed)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        idx = pd.Index(self.nodes["tx_id"])
        tr = idx.get_indexer(train["tx_id"])
        va = idx.get_indexer(val["tx_id"])
        train_steps = np.isin(self.nodes["time_step"], train["time_step"].unique())
        self._setup(train_steps, dev)
        y = torch.as_tensor(self.nodes["label"].fillna(-1).to_numpy(), dtype=torch.float32, device=dev)
        tr_l = tr[self.nodes["label"].to_numpy()[tr] >= 0]
        va_l = va[self.nodes["label"].to_numpy()[va] >= 0]
        pw = float((y[tr_l] == 0).sum() / max(float((y[tr_l] == 1).sum()), 1.0))
        assert self._x is not None and self._ei is not None
        self._model = NodeSAGE(self._x.shape[1], self.hidden, self.layers, self.dropout).to(dev)
        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=dev))
        tr_t = torch.as_tensor(tr_l, device=dev)
        best, best_state, bad = -1.0, None, 0
        for epoch in range(self.epochs):
            self._model.train()
            out = self._model(self._x, self._ei)
            loss = loss_fn(out[tr_t], y[tr_t])
            opt.zero_grad()
            loss.backward()
            opt.step()
            self._model.eval()
            with torch.no_grad():
                p = torch.sigmoid(self._model(self._x, self._ei)).cpu().numpy()
            ap = float(average_precision_score(self.nodes["label"].to_numpy()[va_l], p[va_l]))
            self.history.append({"epoch": epoch, "loss": float(loss.detach()), "val_pr_auc": ap})
            if ap > best:
                best, bad = ap, 0
                best_state = {k: v.detach().clone() for k, v in self._model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        assert best_state is not None
        self._model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def score(self, df: pd.DataFrame) -> np.ndarray:
        assert self._model is not None and self._x is not None and self._ei is not None
        self._model.eval()
        p = torch.sigmoid(self._model(self._x, self._ei)).cpu().numpy()
        return p[pd.Index(self.nodes["tx_id"]).get_indexer(df["tx_id"])]
