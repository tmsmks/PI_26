"""
Modèle de durée dédié : combien de temps dure une coupure une fois commencée.

Pourquoi
--------
L'app estimait la durée d'une coupure par une **heuristique aveugle** liée à la
seule probabilité : `durée = 1.0 + p × 4.0` (plafonnée à 5 h, ignore la météo,
la charge, l'heure…). Or les coupures réelles de Lacor durent de 1 à 16 h
(médiane 2 h, moyenne 3 h) selon les conditions. On entraîne donc une
**régression dédiée** sur les épisodes réels, conditionnée aux features au
**début** de l'épisode (mêmes features que le nowcast ⇒ l'app passe le même X).

Données
-------
Reconstruction des épisodes : sur la série horaire `is_outage` de Lacor, un
épisode est une suite maximale d'heures à 1. Chaque épisode fournit :
  - X = features à l'heure de **déclenchement** (1ʳᵉ heure de l'épisode) ;
  - y = durée de l'épisode en heures.

Sortie
------
  models/duration_model.joblib   bundle {model, features, target, duration_max…}
  models/duration_summary.json   métriques (MAE/RMSE) vs baselines + top features

À lancer après `train_baseline`. Intégré à `run_pipeline.py` (étape 5 bis).
"""

from __future__ import annotations

import json
import logging
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.train_baseline import prepare_data
from src.utils.config import FEATURES_DIR, MODELS_DIR, RANDOM_SEED
from src.utils.io import load_table, setup_logging

logger = logging.getLogger(__name__)

LACOR = "lacor_uganda"
TEST_FRACTION = 0.2
DURATION_MIN_H = 0.5
DURATION_MAX_H = 24.0
MODEL_PATH = MODELS_DIR / "duration_model.joblib"
SUMMARY_PATH = MODELS_DIR / "duration_summary.json"


def build_duration_dataset(lac: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """À partir d'un DataFrame Lacor horaire trié, renvoie (X_onset, y, onset_dt).

    X_onset : features (mêmes colonnes que le nowcast) à l'heure de déclenchement
              de chaque épisode de coupure.
    y       : durée de l'épisode en heures.
    onset_dt: horodatage du déclenchement (pour un split chronologique).
    """
    lac = lac.sort_values("datetime").reset_index(drop=True)
    X_all, _ = prepare_data(lac)
    X_all = X_all.apply(pd.to_numeric, errors="coerce").fillna(0.0).reset_index(drop=True)

    s = lac["is_outage"].astype(int).to_numpy()
    prev = np.r_[0, s[:-1]]
    onset_mask = (s == 1) & (prev == 0)
    onset_pos = np.flatnonzero(onset_mask)

    # Durée = longueur de la suite de 1 démarrant à chaque onset.
    episode_id = np.where(s == 1, onset_mask.cumsum(), -1)
    durations = (
        pd.Series(s).groupby(episode_id).sum().loc[lambda d: d.index >= 0].to_numpy()
    )

    X_onset = X_all.iloc[onset_pos].reset_index(drop=True)
    onset_dt = lac["datetime"].iloc[onset_pos].reset_index(drop=True)
    y = durations.astype(float)
    assert len(X_onset) == len(y) == len(onset_dt)
    return X_onset, y, onset_dt


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "medae": round(float(np.median(np.abs(y_true - y_pred))), 3),
    }


def _legacy_heuristic_duration(proba: np.ndarray) -> np.ndarray:
    """Heuristique historique : durée = 1 + 4p si p>0.5, sinon 0.5."""
    return np.where(proba > 0.5, 1.0 + 4.0 * proba, 0.5)


def run(scope: str = "real") -> dict:
    """Régression durée d'épisode sur **Lacor seul**.

    ``scope`` est traceabilité pipeline uniquement ; les lignes utilisées
    sont toujours ``df[df['hospital'] == LACOR]``.
    """
    if scope == "all":
        logger.warning(
            "train_duration : scope='all' ignoré — entraînement toujours sur Lacor."
        )
    t0 = perf_counter()
    df = load_table(FEATURES_DIR / "features_dataset.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    lac = df[df["hospital"] == LACOR].sort_values("datetime").reset_index(drop=True)
    if lac.empty or "is_outage" not in lac.columns:
        raise ValueError("Pas de données Lacor avec is_outage dans features_dataset.")

    X, y, onset_dt = build_duration_dataset(lac)
    feature_cols = list(X.columns)
    n = len(y)
    if n < 20:
        raise ValueError(f"Trop peu d'épisodes ({n}) pour entraîner un modèle de durée.")

    # Split chronologique (les onsets sont déjà triés par datetime).
    cut = int(n * (1 - TEST_FRACTION))
    X_tr, X_te = X.iloc[:cut], X.iloc[cut:]
    y_tr, y_te = y[:cut], y[cut:]

    model = LGBMRegressor(
        objective="regression_l1",  # MAE : robuste à la longue traîne des durées
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=4,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_tr, y_tr)
    pred_te = np.clip(model.predict(X_te), DURATION_MIN_H, DURATION_MAX_H)

    # ── Baselines honnêtes ──
    median_train = float(np.median(y_tr))
    base_median = _metrics(y_te, np.full_like(y_te, median_train))
    model_metrics = _metrics(y_te, pred_te)

    # Heuristique historique : nécessite la proba du nowcast au déclenchement.
    heuristic_metrics = None
    try:
        served = joblib.load(MODELS_DIR / "calibrated_model.joblib")
        feats_served = list(getattr(served, "feature_names_in_", feature_cols))
        proba_te = served.predict_proba(X_te.reindex(columns=feats_served, fill_value=0.0))[:, 1]
        heuristic_metrics = _metrics(y_te, _legacy_heuristic_duration(proba_te))
    except Exception as exc:  # modèle servi absent ⇒ on saute la comparaison
        logger.warning("Comparaison heuristique impossible (%s).", exc)

    # Réentraînement sur tout le jeu pour le modèle servi.
    final_model = LGBMRegressor(**model.get_params())
    final_model.fit(X, y)

    importances = (
        pd.Series(final_model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
        .head(15)
    )

    bundle = {
        "model": final_model,
        "features": feature_cols,
        "target": "episode_duration_h",
        "duration_min_h": DURATION_MIN_H,
        "duration_max_h": DURATION_MAX_H,
        "site": LACOR,
        "scope": scope,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    summary = {
        "type": "outage_duration_regression",
        "site": LACOR,
        "scope": scope,
        "n_episodes": int(n),
        "n_train": int(cut),
        "n_test": int(n - cut),
        "n_features": len(feature_cols),
        "target_stats_h": {
            "mean": round(float(np.mean(y)), 2),
            "median": round(float(np.median(y)), 2),
            "min": round(float(np.min(y)), 2),
            "max": round(float(np.max(y)), 2),
            "p90": round(float(np.quantile(y, 0.90)), 2),
        },
        "holdout_metrics_h": {
            "duration_model": model_metrics,
            "baseline_median": base_median,
            "legacy_heuristic": heuristic_metrics,
        },
        "top_features": {k: round(float(v), 1) for k, v in importances.items()},
        "note": (
            "Régression dédiée à la DURÉE d'un épisode de coupure (heures), "
            "entraînée sur les épisodes réels de Lacor, conditionnée aux features "
            "au déclenchement. Remplace l'heuristique `1 + 4p` dans l'app (repli "
            "sur l'heuristique si ce modèle est absent). MAE comparée à deux "
            "baselines : médiane constante et heuristique historique."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _h = heuristic_metrics["mae"] if heuristic_metrics else float("nan")
    logger.info(
        "Modèle de durée : MAE=%.2f h (médiane=%.2f, heuristique=%.2f) sur %d épisodes "
        "test — %s | %.1fs",
        model_metrics["mae"], base_median["mae"], _h, n - cut, MODEL_PATH.name,
        perf_counter() - t0,
    )
    return summary


if __name__ == "__main__":
    setup_logging()
    run()
