"""Tests de normalisation SHAP (sans charger LightGBM)."""

import numpy as np

from src.utils.shap_compat import normalize_shap_matrix, tree_expected_value


def test_normalize_shap_list_positive_class():
    sv = [np.zeros((2, 3)), np.ones((2, 3))]
    out = normalize_shap_matrix(sv)
    assert out.shape == (2, 3)
    assert np.allclose(out, 1.0)


def test_normalize_shap_3d_positive_class():
    sv = np.stack([np.zeros((2, 3)), np.full((2, 3), 2.0)], axis=-1)
    out = normalize_shap_matrix(sv)
    assert np.allclose(out, 2.0)


def test_tree_expected_value_binary():
    class Expl:
        expected_value = [0.1, 0.9]

    assert tree_expected_value(Expl()) == 0.9
