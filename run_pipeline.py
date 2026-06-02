"""
Script principal : exécute le pipeline complet de bout en bout.

Étapes :
  1. Ingestion (Lacor, ERIC NHS, NYC LL84, météo archive + prévisions, Electricity Maps)
  2. Preprocessing (rééchantillonnage, fusion multi-hôpitaux)
  3. Feature engineering
  4. Entraînement nowcast (RF / XGBoost / LightGBM + SHAP)
  5. Entraînement horizons 1/3/6 h (mêmes features, cible coupure future)
"""

import logging
import sys
from datetime import date, timedelta
import argparse
from time import perf_counter

from src.utils.io import setup_logging

logger = logging.getLogger(__name__)


def _run_timed(label: str, fn, *args, **kwargs):
    start = perf_counter()
    fn(*args, **kwargs)
    elapsed = perf_counter() - start
    logger.info("  ✓ %s terminé en %.2fs", label, elapsed)
    return elapsed


def main(
    mode: str = "train",
    window_days: int = 30,
    fast_mode: bool = False,
    grid_scale: str = "full",
    cv_folds: int | None = None,
    shap_sample_size: int | None = None,
    save_full_artifacts: bool = True,
    scope: str = "real",
    calibration_method: str = "auto",
):
    setup_logging()
    logger.info("=" * 60)
    logger.info("  PIPELINE — Prédiction de coupures d'électricité")
    if mode == "live":
        logger.info("  Mode LIVE : fenêtre glissante récente (%d jours)", window_days)
    else:
        logger.info("  Données réelles : Lacor Hospital (Ouganda, 2022)")
    logger.info("=" * 60)

    # ── Étape 1 : Ingestion ─────────────────────────────────────────
    logger.info("\n▶ ÉTAPE 1 : Ingestion des données")

    stage_start = perf_counter()
    logger.info("  → Chargement des datasets de consommation…")
    from src.data.ingest_consumption import run as ingest_consumption
    _run_timed("Ingestion consommation", ingest_consumption)

    logger.info("  → Récupération de la météo historique 2022…")
    try:
        from src.data.ingest_meteo import run as ingest_meteo
        if mode == "live":
            end_d = date.today()
            start_d = end_d - timedelta(days=window_days)
            _run_timed(
                "Ingestion météo",
                ingest_meteo,
                start_date=start_d.isoformat(),
                end_date=end_d.isoformat(),
            )
        else:
            _run_timed("Ingestion météo", ingest_meteo)
    except Exception as e:
        logger.warning("  ⚠ Ingestion météo échouée : %s", e)

    logger.info("  → Ingestion données ERIC NHS…")
    try:
        from src.data.ingest_eric import run as ingest_eric
        _run_timed("Ingestion ERIC", ingest_eric)
    except Exception as e:
        logger.warning("  ⚠ Ingestion ERIC échouée : %s", e)

    logger.info("  → Ingestion données NYC LL84…")
    try:
        from src.data.ingest_nyc_ll84 import run as ingest_nyc
        _run_timed("Ingestion NYC LL84", ingest_nyc)
    except Exception as e:
        logger.warning("  ⚠ Ingestion NYC LL84 échouée : %s", e)

    logger.info("  → Ingestion prévisions météo (Open-Meteo Forecast)…")
    try:
        from src.data.ingest_openmeteo_forecast import run as ingest_forecast
        _run_timed("Ingestion Forecast", ingest_forecast)
    except Exception as e:
        logger.warning("  ⚠ Ingestion prévisions échouée : %s", e)

    logger.info("  → Ingestion réseau local Electricity Maps (charge + mix)…")
    try:
        from src.data.ingest_electricitymaps import run as ingest_em_train, run_live as ingest_em_live
        if mode == "live":
            _run_timed("Ingestion Electricity Maps", ingest_em_live, window_hours=window_days * 24)
        else:
            _run_timed("Ingestion Electricity Maps", ingest_em_train)
    except Exception as e:
        logger.warning("  ⚠ Ingestion Electricity Maps échouée : %s", e)

    logger.info("  ✓ Étape ingestion terminée en %.2fs", perf_counter() - stage_start)

    # ── Étape 2 : Preprocessing ─────────────────────────────────────
    logger.info("\n▶ ÉTAPE 2 : Preprocessing (rééchantillonnage + fusion)")
    from src.data.preprocessing import run as preprocess
    _run_timed("Preprocessing", preprocess)

    # ── Étape 3 : Feature engineering ───────────────────────────────
    logger.info("\n▶ ÉTAPE 3 : Feature engineering")
    from src.features.build_features import run as build_features
    _run_timed("Feature engineering", build_features)

    # ── Étape 4 : Entraînement baseline ─────────────────────────────
    logger.info("\n▶ ÉTAPE 4 : Entraînement du modèle baseline")
    from src.models.train_baseline import run as train_baseline
    _run_timed(
        "Entraînement",
        train_baseline,
        fast_mode=fast_mode,
        grid_scale=grid_scale,
        cv_folds=cv_folds,
        shap_sample_size=shap_sample_size,
        save_full_artifacts=save_full_artifacts,
        scope=scope,
        calibration_method=calibration_method,
    )

    logger.info("\n▶ ÉTAPE 5 : Modèles horizons (coupure dans 1 / 3 / 6 h)")
    from src.models.train_horizons import run as train_horizons
    try:
        _run_timed("Horizons 1/3/6 h", train_horizons, scope=scope, fast_mode=fast_mode)
    except Exception as e:
        logger.warning("  ⚠ Entraînement horizons échoué : %s", e)

    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE TERMINÉ AVEC SUCCÈS")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline prédiction coupures")
    parser.add_argument(
        "--mode",
        choices=["train", "live"],
        default="train",
        help="train = pipeline historique 2022, live = fenêtre glissante récente",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Taille de la fenêtre glissante en jours (mode live).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Active un mode rapide (CV réduit, grilles compactes, SHAP échantillonné).",
    )
    parser.add_argument(
        "--grid-scale",
        choices=["compact", "full"],
        default="full",
        help="Taille de grille d'hyperparamètres.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Nombre de folds TimeSeriesSplit (override).",
    )
    parser.add_argument(
        "--shap-sample-size",
        type=int,
        default=None,
        help="Taille d'échantillon max pour SHAP.",
    )
    parser.add_argument(
        "--no-full-artifacts",
        action="store_true",
        help="Ne sauvegarde pas les artefacts lourds (explainer/joblib SHAP).",
    )
    parser.add_argument(
        "--scope",
        choices=["real", "all"],
        default="real",
        help=(
            "Portée d'entraînement. real (défaut) = uniquement les hôpitaux à "
            "coupures réellement observées (Lacor) → métriques honnêtes. "
            "all = multi-hôpitaux complet, inclut des coupures synthétiques "
            "(ERIC/NYC) qui biaisent le F1 global."
        ),
    )
    parser.add_argument(
        "--calibration",
        choices=["auto", "none", "isotonic", "sigmoid"],
        default="auto",
        help=(
            "Méthode de calibration. auto (défaut) = choisit none/isotonic/"
            "sigmoid selon le Brier sur une validation interne (un GBM est "
            "souvent déjà bien calibré → 'none' gagne sur petit jeu). "
            "none/isotonic/sigmoid = forcer."
        ),
    )
    args = parser.parse_args()
    main(
        mode=args.mode,
        window_days=args.window_days,
        fast_mode=args.fast,
        grid_scale=args.grid_scale,
        cv_folds=args.cv_folds,
        shap_sample_size=args.shap_sample_size,
        save_full_artifacts=not args.no_full_artifacts,
        scope=args.scope,
        calibration_method=args.calibration,
    )
