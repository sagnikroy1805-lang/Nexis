"""Transaction-level (row-view) features: rung 3 of the modelling ladder.

Implements the "row" view of Concept Mastery §1.1: what can be said about one
transaction from its own fields, with no history and no graph.

Deliberately excluded:
  - day of week / date: the dataset spans ten days, so these act as a time index
    and let a model learn the prevalence trend across folds rather than behaviour.
  - account and bank identifiers as categories: the model would memorise
    accounts, which does not transfer to unseen accounts (§1.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TABULAR_COLUMNS: tuple[str, ...] = (
    "log_amount",
    "hour",
    "payment_format",
    "pay_currency",
    "recv_currency",
    "is_cross_currency",
    "is_cross_bank",
    "is_self_transfer",
    "is_round_amount",
)


def tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-transaction features. Categoricals become stable integer codes.

    Codes come from the categorical dtype's full vocabulary (the set of possible
    values), not from fold statistics, so they are identical in every fold and
    carry no information about labels or about later data.
    """
    out = pd.DataFrame(index=df.index)
    out["log_amount"] = np.log1p(df["amount"]).astype(np.float32)
    out["hour"] = df["timestamp"].dt.hour.astype(np.float32)
    for col in ("payment_format", "pay_currency", "recv_currency"):
        cat = df[col].astype("category")
        out[col] = cat.cat.codes.astype(np.float32)
    for col in ("is_cross_currency", "is_cross_bank", "is_self_transfer"):
        out[col] = df[col].astype(np.float32)
    # Whole-hundred amounts in the payment currency: a structuring tell.
    out["is_round_amount"] = (np.mod(df["amount_paid"], 100.0) == 0).astype(np.float32)
    return out[list(TABULAR_COLUMNS)]
