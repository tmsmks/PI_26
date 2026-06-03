"""
Tests du modèle de durée des coupures (src/models/train_duration.py).

Couvre les transformations DÉTERMINISTES (aucun réseau) :
  - reconstruction des épisodes + durées à partir d'une série is_outage ;
  - alignement X (features au déclenchement) / y (durée) / horodatage ;
  - heuristique historique de repli.

Lancer : python -m pytest tests/test_duration.py -q
"""

import numpy as np
import pandas as pd

from src.models.train_duration import (
    _legacy_heuristic_duration,
    build_duration_dataset,
)


def _toy_lacor(seed: int = 0) -> pd.DataFrame:
    """Série horaire Lacor synthétique avec des épisodes de coupure connus."""
    n = 400
    dt = pd.date_range("2022-01-01", periods=n, freq="h")
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "datetime": dt,
        "hospital": "lacor_uganda",
        "hour": dt.hour,
        "day_of_week": dt.dayofweek,
        "month": dt.month,
        "total_load_kw": rng.uniform(80, 200, n),
        "temperature_2m": rng.uniform(18, 34, n),
        "is_outage": 0,
    })
    # Trois épisodes de durées 3, 1 et 5 heures.
    for start, length in [(50, 3), (120, 1), (300, 5)]:
        df.loc[start:start + length - 1, "is_outage"] = 1
    return df


def test_build_duration_dataset_recovers_episode_durations():
    df = _toy_lacor()
    X, y, onset_dt = build_duration_dataset(df)

    # 3 épisodes → 3 échantillons, durées exactes.
    assert len(X) == len(y) == len(onset_dt) == 3
    assert sorted(y.tolist()) == [1.0, 3.0, 5.0]
    # Les déclenchements sont triés chronologiquement (pour le split temporel).
    assert list(onset_dt) == sorted(onset_dt)
    # X ne contient que des colonnes numériques de features (pas la cible).
    assert "is_outage" not in X.columns
    assert not X.isna().any().any()


def test_onset_features_align_with_start_hour():
    df = _toy_lacor()
    X, y, onset_dt = build_duration_dataset(df)
    # Le 1er déclenchement est à l'index 50 → son horodatage doit correspondre.
    assert onset_dt.iloc[0] == df.loc[50, "datetime"]


def test_legacy_heuristic_matches_app_formula():
    p = np.array([0.2, 0.5, 0.6, 0.9])
    d = _legacy_heuristic_duration(p)
    # p ≤ 0.5 → 0.5 h ; sinon 1 + 4p.
    assert d[0] == 0.5
    assert d[1] == 0.5
    assert np.isclose(d[2], 1.0 + 4.0 * 0.6)
    assert np.isclose(d[3], 1.0 + 4.0 * 0.9)
