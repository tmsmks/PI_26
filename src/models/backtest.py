"""
Validation temporelle honnête (généralisation dans le temps) sur Lacor.

Le hold-out 80/20 de l'entraînement est déjà chronologique, mais il ne teste
qu'UNE fenêtre future. Ce module fournit une évaluation « futur réel » bien
plus robuste, indispensable vu qu'on n'a qu'un site × une année :

  1. Hold-out chronologique explicite : train jan–sept, test oct–déc.
  2. Backtest walk-forward (origine glissante / fenêtre expansive) : pour
     chaque mois M (à partir d'un minimum d'historique), on entraîne sur
     tous les mois < M et on teste sur M, puis on agrège. C'est le standard
     pour mesurer la stabilité d'un prédicteur temporel.

On réutilise les hyperparamètres du modèle gagnant (training_summary.json) et
le MÊME jeu de features que la prod (prepare_data → 49 features, signaux
externes exclus). Aucune fuite : les features rolling/historique sont causales
(décalées), donc chaque ligne ne dépend que de son passé.

Lancer : python -m src.models.backtest
Sorties : models/backtest_by_month.csv + models/backtest_summary.json
"""

from __future__ import annotations

import json
import logging
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

from src.utils.config import FEATURES_DIR, MODELS_DIR, RANDOM_SEED
from src.utils.io import load_table, setup_logging
from src.models.train_baseline import (
    REAL_DATA_HOSPITALS,
    TARGET,
    compute_metrics,
    prepare_data,
)

logger = logging.getLogger(__name__)

MIN_TRAIN_MONTHS = 6      # historique minimal avant de commencer à tester
DECISION_THRESHOLD = 0.5


def _build_winner_estimator():
    """Reconstruit un estimateur non-entraîné avec les hyperparamètres du
    gagnant (training_summary.json). Fallback RandomForest raisonnable."""
    summary_path = MODELS_DIR / "training_summary.json"
    winner, params = "RandomForest", {}
    if summary_path.exists():
        try:
            s = json.load(open(summary_path))
            winner = s.get("winner", winner)
            params = dict(s.get("winner_params", {}))
        except Exception as exc:
            logger.warning("Lecture training_summary impossible (%s) — RF par défaut.", exc)

    if winner == "RandomForest":
        cw = params.get("class_weight")
        if isinstance(cw, dict):  # clés JSON "0"/"1" → int pour sklearn
            params["class_weight"] = {int(k): v for k, v in cw.items()}
        return winner, RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1, **params)
    if winner == "LightGBM":
        from lightgbm import LGBMClassifier
        return winner, LGBMClassifier(random_state=RANDOM_SEED, n_jobs=-1, verbose=-1, **params)
    if winner == "XGBoost":
        from xgboost import XGBClassifier
        return winner, XGBClassifier(
            random_state=RANDOM_SEED, n_jobs=-1,
            eval_metric="logloss", tree_method="hist", **params,
        )
    logger.warning("Gagnant inconnu '%s' — RandomForest par défaut.", winner)
    return "RandomForest", RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)


def _fit_eval(estimator, X_tr, y_tr, X_te, y_te) -> dict | None:
    """Entraîne une copie fraîche et évalue sur le test. None si test dégénéré."""
    if len(X_te) == 0 or y_tr.nunique() < 2:
        return None
    est = clone(estimator)
    est.fit(X_tr, y_tr)
    proba = est.predict_proba(X_te)[:, 1]
    pred = (proba >= DECISION_THRESHOLD).astype(int)
    return compute_metrics(y_te, pred, proba)


def run() -> dict:
    t0 = perf_counter()
    df = load_table(FEATURES_DIR / "features_dataset.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = (
        df[df["hospital"].isin(REAL_DATA_HOSPITALS)]
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError("Aucune donnée réelle (Lacor) pour le backtest.")

    X, y = prepare_data(df)
    month = df["datetime"].dt.month.to_numpy()
    winner_name, estimator = _build_winner_estimator()
    logger.info("Backtest temporel — modèle : %s | %d features | %d lignes",
                winner_name, X.shape[1], len(X))

    # ── 1. Hold-out chronologique explicite : train mois 1–9, test mois 10–12 ──
    tr = month <= 9
    te = month >= 10
    headline = _fit_eval(estimator, X[tr], y[tr], X[te], y[te])
    logger.info("═══ Hold-out chronologique : train mois 1–9 (%d) → test mois 10–12 (%d, coupures=%.1f%%) ═══",
                int(tr.sum()), int(te.sum()), 100 * float(y[te].mean()))
    for k, v in headline.items():
        logger.info("  %-10s : %.4f", k, v)

    # ── 2. Walk-forward : train mois < M, test mois M (M ≥ MIN_TRAIN_MONTHS+1) ──
    logger.info("═══ Backtest walk-forward (origine glissante, fenêtre expansive) ═══")
    rows = []
    for m in range(MIN_TRAIN_MONTHS + 1, 13):
        tr_m = month < m
        te_m = month == m
        metrics = _fit_eval(estimator, X[tr_m], y[tr_m], X[te_m], y[te_m])
        if metrics is None:
            continue
        metrics = {"test_month": m, "n_train": int(tr_m.sum()), "n_test": int(te_m.sum()),
                   "outage_rate": round(float(y[te_m].mean()), 4), **metrics}
        rows.append(metrics)
        logger.info("  mois %2d | train=%5d test=%4d (coup.=%4.1f%%) | F1=%.3f recall=%.3f ROC=%.3f Brier=%.4f",
                    m, metrics["n_train"], metrics["n_test"], 100 * metrics["outage_rate"],
                    metrics["f1"], metrics["recall"], metrics["roc_auc"], metrics["brier"])

    by_month = pd.DataFrame(rows)
    agg_keys = ["f1", "precision", "recall", "roc_auc", "brier", "accuracy"]
    aggregate = {
        k: {"mean": round(float(by_month[k].mean()), 4),
            "std": round(float(by_month[k].std(ddof=0)), 4),
            "min": round(float(by_month[k].min()), 4),
            "max": round(float(by_month[k].max()), 4)}
        for k in agg_keys
    } if not by_month.empty else {}

    logger.info("── Walk-forward agrégé (%d folds mensuels) ──", len(by_month))
    for k in agg_keys:
        if k in aggregate:
            a = aggregate[k]
            logger.info("  %-10s : %.3f ± %.3f  [%.3f–%.3f]", k, a["mean"], a["std"], a["min"], a["max"])

    by_month.to_csv(MODELS_DIR / "backtest_by_month.csv", index=False)
    summary = {
        "model": winner_name,
        "n_features": int(X.shape[1]),
        "decision_threshold": DECISION_THRESHOLD,
        "holdout_chrono_train1_9_test10_12": headline,
        "walk_forward_aggregate": aggregate,
        "walk_forward_n_folds": int(len(by_month)),
        "note": (
            "Évaluation sur Lacor uniquement (seul site à coupures réelles). "
            "Le walk-forward mesure la stabilité dans le temps ; il ne valide PAS "
            "la généralisation à d'autres sites (nécessiterait des données réelles "
            "multi-sites)."
        ),
    }
    with open(MODELS_DIR / "backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Résumé → models/backtest_summary.json + backtest_by_month.csv")
    logger.info("Temps total backtest : %.2fs", perf_counter() - t0)
    return summary


if __name__ == "__main__":
    setup_logging()
    run()
