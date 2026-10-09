"""Human-readable feature names for explanations shown to analysts.

Rule 5: these describe what the system OBSERVED about a transaction. None of
them asserts intent or wrongdoing.
"""

from __future__ import annotations

_FIXED = {
    "log_amount": "Amount (log USD)",
    "hour": "Hour of day",
    "payment_format": "Payment format",
    "pay_currency": "Payment currency",
    "recv_currency": "Receiving currency",
    "is_cross_currency": "Paid and received in different currencies",
    "is_cross_bank": "Sender and receiver at different banks",
    "is_self_transfer": "Transfer between the same account",
    "is_round_amount": "Whole-hundred amount",
    "amt_over_src_in_24h": "Amount relative to funds the sender received in the last 24h",
    "src_out_n_hist": "Sender's number of earlier payments",
    "src_has_history": "Sender has at least 3 earlier payments",
    "amt_z_src": "Amount vs the sender's own history (z-score, log scale)",
    "amt_log_ratio_src_mean": "Amount vs the sender's typical amount (log ratio)",
    "amt_vs_src_max": "Amount vs the sender's largest earlier payment",
    "src_secs_since_out": "Seconds since the sender's previous payment",
    "src_in_n_hist": "Sender's number of earlier receipts",
    "src_secs_since_in": "Seconds since the sender last received funds",
    "dst_in_n_hist": "Receiver's number of earlier receipts",
    "dst_out_n_hist": "Receiver's number of earlier payments",
    "counterparty_is_new": "First payment to this counterparty",
    "src_n_counterparties": "Sender's distinct earlier payees",
    "dst_n_counterparties": "Receiver's distinct earlier payers",
    "closes_cycle": "Payment closes a loop in the earlier payment graph",
    "reverse_edge_exists": "Receiver has paid the sender before",
    "prior_pair_tx": "Earlier payments between this pair",
}

_STREAM = {
    "src_out": "Sender's payments",
    "src_in": "Sender's receipts",
    "dst_in": "Receiver's receipts",
    "dst_out": "Receiver's payments",
}

_NODE = {
    "in_degree": "distinct payers",
    "out_degree": "distinct payees",
    "in_tx": "payments received",
    "out_tx": "payments sent",
    "log_in_weight": "volume received (log)",
    "log_out_weight": "volume sent (log)",
    "degree_asymmetry": "payee/payer asymmetry",
    "flow_through": "pass-through ratio",
    "reciprocal_partners": "two-way counterparties",
    "pagerank": "PageRank",
    "log_scc_size": "size of strongly connected group (log)",
    "nbr_out_degree_mean": "payees' mean number of payees",
    "nbr_in_degree_mean": "payers' mean number of payers",
}


def feature_label(name: str) -> str:
    """Analyst-facing description of a feature column."""
    if name in _FIXED:
        return _FIXED[name]
    for prefix, text in _STREAM.items():
        for stat, word in (("cnt", "count"), ("sum", "total USD")):
            head = f"{prefix}_{stat}_"
            if name.startswith(head):
                return f"{text}: {word}, last {name[len(head):]}"
    for side, who in (("src_", "Sender"), ("dst_", "Receiver")):
        if name.startswith(side) and name[len(side):] in _NODE:
            return f"{who}: {_NODE[name[len(side):]]} (earlier graph)"
    return name.replace("_", " ")
