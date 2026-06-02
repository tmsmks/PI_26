"""
Entraînement des modèles « coupure dans les 1 / 3 / 6 prochaines heures ».

Même pipeline que le nowcast Lacor :
  - mêmes features (historique de coupure, conso, météo, charge, temps) ;
  - cible = coupure dans la fenêtre future (pas `is_outage` à l'heure t).

Sorties : `models/nowcast_horizons/horizon_{1,3,6}h/horizon_model.joblib`
          `models/nowcast_horizons/horizons_summary.json`

À lancer après `train_baseline` (lit le gagnant dans training_summary.json).
Intégré dans `run_pipeline.py` étape 5.
"""

from __future__ import annotations

import json
import logging
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.models.horizon_targets import build_horizon_target
from src.models.train_baseline import prepare_data
from src.utils.config import FEATURES_DIR, MODELS_DIR, RANDOM_SEED
from src.utils.io import load_table, setup_logging

logger = logging.getLogger(__name__)

HORIZONS = (1, 3, 6)
MIN_TRAIN_MONTHS = 6
DECISION_THRESHOLD = 0.5
CALIB_FRACTION = 0.2
LACOR = "lacor_uganda"
OUT_DIR = MODELS_DIR / "nowcast_horizons"


def _walk_forward(X: pd.DataFrame, y: pd.Series, month: np.ndarray, estimator) -> dict:
    rows = []
    for m in range(MIN_TRAIN_MONTHS + 1, 13):
        tr, te = month < m, month == m
        if te.sum() == 0 or y[tr].nunique() < 2 or y[te].nunique() < 2:
            continue
        mdl = _clone_estimator(estimator)
        mdl.fit(X[tr], y[tr])
        proba = mdl.predict_proba(X[te])[:, 1]
        pred = (proba >= DECISION_THRESHOLD).astype(int)
        rows.append({
            "f1": f1_score(y[te], pred, zero_division=0),
            "precision": precision_score(y[te], pred, zero_division=0),
            "recall": recall_score(y[te], pred, zero_division=0),
            "roc_auc": roc_auc_score(y[te], proba) if y[te].nunique() > 1 else float("nan"),
            "brier": brier_score_loss(y[te], proba),
        })
    by = pd.DataFrame(rows)
    if by.empty:
        return {"f1": 0.0, "f1_std": 0.0, "roc_auc": 0.0, "recall": 0.0,
                "precision": 0.0, "brier": 1.0, "n_folds": 0}
    return {
        **{k: round(float(by[k].mean()), 4) for k in by.columns},
        "f1_std": round(float(by["f1"].std(ddof=0)), 4),
        "n_folds": len(by),
    }


def _clone_estimator(template):
    """Copie fraîche du même type d'estimateur (hyperparamètres du gagnant)."""
    return template.__class__(**template.get_params())


def _estimator_from_summary(summary: dict, outage_ratio: float, fast_mode: bool) -> object:
    winner = summary.get("winner", "LightGBM")
    params = dict(summary.get("winner_params") or {})
    spw = params.pop("scale_pos_weight", None)
    if spw is None:
        spw = max(2, round(1.0 / max(outage_ratio, 1e-3)))

    if winner == "RandomForest":
        cw = params.pop("class_weight", {0: 1, 1: spw})
        return RandomForestClassifier(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight=cw,
            **params,
        )
    if winner == "XGBoost":
        return XGBClassifier(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            eval_metric="logloss",
            scale_pos_weight=spw,
            **params,
        )
    # LightGBM (défaut)
    n_est = 200 if fast_mode else params.pop("n_estimators", 400)
    return LGBMClassifier(
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
        scale_pos_weight=spw,
        n_estimators=n_est,
        **params,
    )


def _fit_calibrated(X: pd.DataFrame, y: pd.Series, template) -> tuple:
    n = len(X)
    cut = int(n * (1 - CALIB_FRACTION))
    cal_model = _clone_estimator(template)
    cal_model.fit(X.iloc[:cut], y.iloc[:cut])
    proba_hold = cal_model.predict_proba(X.iloc[cut:])[:, 1]
    y_hold = y.iloc[cut:]
    if y_hold.nunique() > 1:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(proba_hold, y_hold.to_numpy())
    else:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit([0.0, 1.0], [0.0, 1.0])
    final_model = _clone_estimator(template)
    final_model.fit(X, y)
    return final_model, calibrator


def run(scope: str = "real", fast_mode: bool = False) -> dict:
    t0 = perf_counter()
    summary_path = MODELS_DIR / "training_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            "training_summary.json absent — lance d'abord train_baseline / run_pipeline."
        )
    with open(summary_path) as f:
        base_summary = json.load(f)

    df = load_table(FEATURES_DIR / "features_dataset.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    lac = (
        df[df["hospital"] == LACOR]
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    if lac.empty:
        raise ValueError("Pas de données Lacor dans features_dataset.")

    X_all, _ = prepare_data(lac)
    feature_cols = list(X_all.columns)
    X_all = X_all.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    month = lac["datetime"].dt.month.to_numpy()
    outage_ratio = float(lac["is_outage"].mean()) if "is_outage" in lac.columns else 0.1
    template = _estimator_from_summary(base_summary, outage_ratio, fast_mode)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    horizons_summary = {}

    for h in HORIZONS:
        y_raw = build_horizon_target(lac["is_outage"], h)
        valid = y_raw.notna()
        X_h = X_all.loc[valid].reset_index(drop=True)
        y = y_raw.loc[valid].astype(int).reset_index(drop=True)
        month_h = month[valid.to_numpy()]

        metrics = _walk_forward(X_h, y, month_h, template)
        model, calibrator = _fit_calibrated(X_h, y, template)

        bundle = {
            "model": model,
            "calibrator": calibrator,
            "features": feature_cols,
            "horizon_hours": h,
            "target": f"outage_within_{h}h",
            "base_winner": base_summary.get("winner"),
        }
        hdir = OUT_DIR / f"horizon_{h}h"
        hdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, hdir / "horizon_model.joblib")

        horizons_summary[f"{h}h"] = {
            **metrics,
            "positive_rate": round(float(y.mean()), 4),
            "n_rows": int(len(y)),
        }
        logger.info(
            "Horizon %dh (nowcast features) : F1=%.3f±%.3f ROC=%.3f rec=%.3f "
            "prec=%.3f brier=%.3f (%d folds, taux+=%.1f%%)",
            h, metrics["f1"], metrics["f1_std"], metrics["roc_auc"],
            metrics["recall"], metrics["precision"], metrics["brier"],
            metrics["n_folds"], 100 * y.mean(),
        )

    out = {
        "type": "nowcast_horizons",
        "site": LACOR,
        "scope": scope,
        "n_features": len(feature_cols),
        "features": feature_cols,
        "inputs": "conso+charge+meteo+historique coupure (mêmes features que le nowcast)",
        "base_model": base_summary.get("winner"),
        "horizons": horizons_summary,
    }
    with open(OUT_DIR / "horizons_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info(
        "Modèles horizons (%d features) → %s | %.1fs",
        len(feature_cols), OUT_DIR, perf_counter() - t0,
    )
    return out


if __name__ == "__main__":
    setup_logging()
    run()
