"""SHAP explanations for the tree models (Concept Mastery §12.1-12.2).

TreeSHAP gives exact Shapley values for tree ensembles in polynomial time, so
each alert's explanation is the model's actual attribution, not an
approximation. XGBoost computes it natively (pred_contribs), which avoids an
extra dependency in the serving path; the `shap` package is used for global
summaries when installed.

§12.5: a SHAP value says what the MODEL weighted for this row. It is not a
cause, and the UI and packets present it as "the model weighted", never
"because".
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def tree_shap(model: object, x: pd.DataFrame) -> np.ndarray:
    """Per-row SHAP values (n_rows, n_features) for an XGBClassifier, log-odds units."""
    import xgboost as xgb

    booster = model.get_booster()  # type: ignore[attr-defined]
    contribs = booster.predict(xgb.DMatrix(x), pred_contribs=True)
    return contribs[:, :-1]  # last column is the bias term


def global_importance(shap_values: np.ndarray, columns: Sequence[str]) -> pd.DataFrame:
    """Mean |SHAP| per feature: the global view (§12.2), sorted descending."""
    imp = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": list(columns), "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
