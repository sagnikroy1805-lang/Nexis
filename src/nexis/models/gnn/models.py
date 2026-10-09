"""GNN encoders and the edge scorer (rungs 6-8).

Implements Concept Mastery §8.1 (message passing; sum aggregation keeps the
fan-in cardinality that mean discards, §8.1.1), §8.3 (GraphSAGE-style root +
neighbour weights), §8.6 (over-smoothing: two layers, LayerNorm, residuals),
§8.7 (heterophily: root and neighbour messages keep separate weights, so a node
is not forced to resemble its neighbours), §10.1-10.3 (relation-wise
heterogeneous aggregation) and §9.1 (snapshot models with recurrent state).

Every encoder returns account embeddings. EdgeScorer turns (sender embedding,
receiver embedding, transaction features) into a risk logit, so a GNN receives
exactly the transaction features XGBoost receives, plus graph context. The
comparison then measures the graph, not a feature gap (Getting Started §4.1).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import GraphConv


class DirectedLayer(nn.Module):
    """One round of message passing in both directions of money flow.

    Who paid an account and whom it paid carry different evidence (fan-in vs
    fan-out), so incoming and outgoing neighbours get separate weights.
    """

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.inc = GraphConv(dim, dim, aggr="add")
        self.out = GraphConv(dim, dim, aggr="add", bias=False)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: Tensor, edge_index: Tensor, weight: Tensor) -> Tensor:
        rev = edge_index.flip(0)
        m = self.inc(h, edge_index, weight) + self.out(h, rev, weight)
        return h + self.drop(torch.relu(self.norm(m)))  # residual: limits over-smoothing


class HomogeneousEncoder(nn.Module):
    """Rung 6: one account node type, one relation (any payment)."""

    def __init__(self, node_in: int, dim: int = 64, layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.proj = nn.Linear(node_in, dim)
        self.layers = nn.ModuleList(DirectedLayer(dim, dropout) for _ in range(layers))

    def forward(self, x: Tensor, graph: dict) -> Tensor:
        h = torch.relu(self.proj(x))
        for layer in self.layers:
            h = layer(h, graph["edge_index"], graph["edge_weight"])
        return h


class HeteroLayer(nn.Module):
    """Relation-wise aggregation (§10.1): h_i' = W0 h_i + Σ_r Σ_{j∈N_r(i)} W_r h_j.

    Relations: one per payment format, in both directions, plus the account-bank
    relation (accounts -> bank node by sum, bank node -> its accounts). An ACH
    edge and a cheque edge no longer share weights, and accounts at the same
    bank exchange information through the bank node type.
    """

    def __init__(self, dim: int, relations: list[str], dropout: float) -> None:
        super().__init__()
        self.root = nn.Linear(dim, dim)
        self.inc = nn.ModuleDict({r: GraphConv(dim, dim, aggr="add", bias=False) for r in relations})
        self.out = nn.ModuleDict({r: GraphConv(dim, dim, aggr="add", bias=False) for r in relations})
        self.bank_in = nn.Linear(dim, dim, bias=False)
        self.bank_out = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: Tensor, graph: dict) -> Tensor:
        m = self.root(h)
        for r, (ei, w) in graph["typed"].items():
            if ei.shape[1] == 0:
                continue
            # Only GraphConv's neighbour part, lin_rel(sum_j w_ji h_j), is used per
            # relation; the single shared root term is self.root above.
            inc: GraphConv = self.inc[r]
            out: GraphConv = self.out[r]
            m = m + inc.lin_rel(inc.propagate(ei, x=(h, h), edge_weight=w))
            m = m + out.lin_rel(out.propagate(ei.flip(0), x=(h, h), edge_weight=w))
        bank = graph["account_bank"]
        # LEAKAGE GUARD: only accounts active before the cutoff feed their bank
        # node. Inactive accounts exist only in the future; pooling over them
        # would leak how many customers a bank WILL have.
        msg = self.bank_in(h) * graph["active"].unsqueeze(1)
        bank_sum = torch.zeros(graph["n_banks"], h.shape[1], device=h.device).index_add_(0, bank, msg)
        m = m + self.bank_out(bank_sum[bank])
        return h + self.drop(torch.relu(self.norm(m)))


class HeterogeneousEncoder(nn.Module):
    """Rung 7: account and bank node types, one relation per payment format."""

    def __init__(
        self, node_in: int, relations: list[str], dim: int = 64, layers: int = 2, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(node_in, dim)
        self.layers = nn.ModuleList(HeteroLayer(dim, relations, dropout) for _ in range(layers))

    def forward(self, x: Tensor, graph: dict) -> Tensor:
        h = torch.relu(self.proj(x))
        for layer in self.layers:
            h = layer(h, graph)
        return h


class TemporalEncoder(nn.Module):
    """Rung 8: heterogeneous encoder + per-account recurrent state over snapshots.

    At snapshot t the encoder reads the structural features of the graph < p_t
    together with each account's state s_{t-1}; message passing runs over the
    typed edges of the most recent period [p_t - Δ, p_t) (what changed); a GRU
    cell folds the result into s_t. The state carries an account's history
    without ever touching an edge at or after p_t.
    """

    def __init__(
        self, node_in: int, relations: list[str], dim: int = 64, layers: int = 2, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.dim = dim
        self.spatial = HeterogeneousEncoder(node_in + dim, relations, dim, layers, dropout)
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, x: Tensor, graph: dict, state: Tensor) -> Tensor:
        m = self.spatial(torch.cat([x, state], dim=1), graph)
        return self.gru(m, state)


class EdgeScorer(nn.Module):
    """Risk logit for a transaction from both endpoints and its own features."""

    def __init__(self, dim: int, edge_in: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim + edge_in, 2 * dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, h: Tensor, src: Tensor, dst: Tensor, edge_x: Tensor) -> Tensor:
        return self.mlp(torch.cat([h[src], h[dst], edge_x], dim=1)).squeeze(-1)
