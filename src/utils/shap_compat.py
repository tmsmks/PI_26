"""Compatibilité SHAP TreeExplainer (LightGBM / sklearn binaire)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

_SHAP_LGBM_WARN = (
    "LightGBM binary classifier with TreeExplainer shap values output "
    "has changed to a list of ndarray"
)


def normalize_shap_matrix(sv) -> np.ndarray:
    """Matrice (n_samples, n_features) pour la classe positive."""
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[-1] > 1 else sv[:, :, 0]
    return sv


def tree_expected_value(explainer) -> float:
    exp = np.asarray(explainer.expected_value).ravel()
    return float(exp[1] if exp.size > 1 else exp[0])


def tree_shap_values(explainer, X: pd.DataFrame) -> np.ndarray:
    """SHAP TreeExplainer sans warning LightGBM (liste de ndarray)."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_SHAP_LGBM_WARN,
            category=UserWarning,
            module=r"shap\.explainers\._tree",
        )
        sv = explainer.shap_values(X)
    return normalize_shap_matrix(sv)
