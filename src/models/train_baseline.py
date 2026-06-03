"""
Pipeline d'entraînement multi-modèles avec tuning, validation croisée,
calibration et explications SHAP.

Modèles comparés :
  - Random Forest (scikit-learn)
  - XGBoost (xgboost)
  - LightGBM (lightgbm)

Pipeline :
  1. Split temporel 80/20 (train / hold-out test)
  2. Pour chaque modèle : GridSearchCV + TimeSeriesSplit (5 folds)
  3. Tableau comparatif → sélection du meilleur (F1 sur CV)
  4. Entraînement final du meilleur sur tout le train set
  5. Calibration isotonique
  6. Évaluation finale sur le hold-out
  7. SHAP : TreeExplainer sur le test set + sauvegarde
"""

import json
import logging
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# `xgboost`, `lightgbm`, `shap` sont des imports lourds (>200 MB total
# de C extensions). On les charge en lazy : un consommateur qui veut
# seulement `compute_metrics_by_source` ou les constantes n'a pas
# besoin de payer ce coût.

from src.utils.config import (
    COLS_TO_DROP,
    CV_FOLDS,
    FAST_MODE,
    FEATURES_DIR,
    GRID_SCALE,
    MODELS_DIR,
    RANDOM_SEED,
    SHAP_SAMPLE_SIZE,
    TEST_SIZE,
    drop_external_signal_columns,
)
from src.utils.io import load_table, setup_logging

logger = logging.getLogger(__name__)

TARGET = "is_outage"

# ── Grilles d'hyperparamètres par modèle ─────────────────────────────

# Valeur par défaut (~9.7 % de coupures Lacor). Recalculée dynamiquement
# dans `run()` à partir du y_train réel pour rester correcte quand on
# change la composition du multi-hôpitaux.
DEFAULT_OUTAGE_RATIO = 0.097


def build_model_configs(
    grid_scale: str = "full",
    outage_ratio: float = DEFAULT_OUTAGE_RATIO,
) -> dict:
    # Lazy imports : voir docstring du module — ces librairies pèsent
    # plusieurs centaines de Mo et ne sont nécessaires qu'à l'entraînement.
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    compact = grid_scale == "compact"
    # `scale_pos_weight` cible 1/ratio pour rééquilibrer la classe positive.
    # On garde un plancher de 2 pour éviter les valeurs aberrantes si le
    # taux remonte (ex. multi-hôpitaux ≥ 30%).
    base_weight = max(2, round(1.0 / max(outage_ratio, 1e-3)))
    aggressive_weight = max(base_weight + 2, round(1.5 / max(outage_ratio, 1e-3)))
    # `class_weight` RF : poids identique pour la classe positive.
    rf_weight_base = {0: 1, 1: base_weight}
    rf_weight_high = {0: 1, 1: aggressive_weight}

    return {
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=1),
            "param_grid": {
                "n_estimators": [200] if compact else [200, 300],
                "max_depth": [12, 18] if compact else [12, 18, 25],
                "min_samples_leaf": [4] if compact else [4, 8],
                "class_weight": [rf_weight_base] if compact else [rf_weight_base, rf_weight_high],
            },
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                random_state=RANDOM_SEED,
                n_jobs=1,
                eval_metric="logloss",
                tree_method="hist",
            ),
            "param_grid": {
                "n_estimators": [200] if compact else [200, 300],
                "max_depth": [5, 8] if compact else [5, 8, 12],
                "learning_rate": [0.1] if compact else [0.05, 0.1],
                "scale_pos_weight": [base_weight] if compact else [base_weight, aggressive_weight],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
            },
        },
        "LightGBM": {
            "estimator": LGBMClassifier(
                random_state=RANDOM_SEED,
                n_jobs=1,
                verbose=-1,
            ),
            "param_grid": {
                "n_estimators": [200] if compact else [200, 300],
                "max_depth": [8, -1] if compact else [8, 15, -1],
                "learning_rate": [0.1] if compact else [0.05, 0.1],
                "scale_pos_weight": [base_weight] if compact else [base_weight, aggressive_weight],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
            },
        },
    }


# ── Fonctions utilitaires ────────────────────────────────────────────

def prepare_data(df: pd.DataFrame) -> tuple:
    drop = [c for c in COLS_TO_DROP if c in df.columns]
    X = df.drop(columns=drop).select_dtypes(include=[np.number])
    # Exclure les signaux externes (gdelt/gdacs/eq/air/em/noaa/storm) : non
    # servables de façon cohérente hors Lacor (train-serve skew) + proxy
    # temporel spurieux. Cf. config.EXTERNAL_SIGNAL_PREFIXES et #3.
    X = X[drop_external_signal_columns(X.columns)]
    y = df[TARGET].astype(int)
    return X, y


def temporal_split(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
):
    """Split temporel par hôpital.

    - Si la colonne `hospital` existe : on découpe chaque hôpital
      chronologiquement (80/20 par défaut), puis on concatène.
    - Sinon : fallback au split temporel global historique.
    """
    if "hospital" not in df.columns:
        split_idx = int(len(X) * (1 - test_size))
        return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

    ordered = df.sort_values(["hospital", "datetime"]).copy()
    train_indices: list[int] = []
    test_indices: list[int] = []

    for hospital_name, grp in ordered.groupby("hospital", sort=False):
        idx = grp.index.to_list()
        if len(idx) <= 1:
            logger.warning(
                "Hôpital %s : %d ligne(s) seulement → tout en train, "
                "test vide pour ce site.",
                hospital_name, len(idx),
            )
            train_indices.extend(idx)
            continue
        split_idx = int(len(idx) * (1 - test_size))
        split_idx = max(1, min(split_idx, len(idx) - 1))
        train_indices.extend(idx[:split_idx])
        test_indices.extend(idx[split_idx:])

    return (
        X.loc[train_indices],
        X.loc[test_indices],
        y.loc[train_indices],
        y.loc[test_indices],
    )


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4) if len(set(y_true)) > 1 else float("nan"),
        "brier": round(brier_score_loss(y_true, y_proba), 4),
    }


# Sources de données : "lacor_uganda" est le seul site avec coupures
# *réellement observées* (relevés terrain 2022). Les autres profils
# horaires (ERIC, NYC) ont leurs coupures **simulées** par une formule
# stochastique connue — un modèle bien entraîné peut les retrouver,
# ce qui gonfle artificiellement F1. On rapporte donc séparément
# pour ne pas masquer la performance "réelle".
REAL_DATA_HOSPITALS = {"lacor_uganda"}


def compute_metrics_by_source(
    y_true: pd.Series,
    y_pred,
    y_proba,
    hospitals: pd.Series,
) -> dict:
    """Métriques ventilées entre hôpitaux à coupures *réelles* (Lacor)
    et hôpitaux à coupures *synthétiques* (ERIC, NYC).
    """
    real_mask = hospitals.isin(REAL_DATA_HOSPITALS).to_numpy()
    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.to_numpy()
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    out: dict = {}
    if real_mask.any():
        out["real"] = compute_metrics(
            y_true[real_mask], y_pred[real_mask], y_proba[real_mask]
        )
        out["real_n"] = int(real_mask.sum())
        out["real_outage_rate"] = round(float(y_true[real_mask].mean()), 4)
    synth_mask = ~real_mask
    if synth_mask.any():
        out["synthetic"] = compute_metrics(
            y_true[synth_mask], y_pred[synth_mask], y_proba[synth_mask]
        )
        out["synthetic_n"] = int(synth_mask.sum())
        out["synthetic_outage_rate"] = round(float(y_true[synth_mask].mean()), 4)
    return out


def log_hospital_split_stats(df: pd.DataFrame, train_idx: pd.Index, test_idx: pd.Index) -> None:
    """Affiche les volumes train/test par hôpital quand la colonne existe."""
    if "hospital" not in df.columns:
        return

    train_stats = (
        df.loc[train_idx, ["hospital", "is_outage"]]
        .groupby("hospital")
        .agg(train_rows=("is_outage", "size"), train_outages=("is_outage", "sum"))
    )
    test_stats = (
        df.loc[test_idx, ["hospital", "is_outage"]]
        .groupby("hospital")
        .agg(test_rows=("is_outage", "size"), test_outages=("is_outage", "sum"))
    )
    stats = train_stats.join(test_stats, how="outer").fillna(0)

    logger.info("═══ Répartition train/test par hôpital ═══")
    for hospital, row in stats.sort_index().iterrows():
        train_rows = int(row["train_rows"])
        test_rows = int(row["test_rows"])
        train_outages = int(row["train_outages"])
        test_outages = int(row["test_outages"])
        train_rate = 100 * train_outages / max(train_rows, 1)
        test_rate = 100 * test_outages / max(test_rows, 1)
        logger.info(
            "  %-24s train=%6d (coupures=%4d, %.2f%%) | test=%6d (coupures=%4d, %.2f%%)",
            hospital,
            train_rows,
            train_outages,
            train_rate,
            test_rows,
            test_outages,
            test_rate,
        )


def _log_metrics(metrics: dict, prefix: str = "") -> None:
    tag = f"[{prefix}] " if prefix else ""
    for k, v in metrics.items():
        logger.info("%s%-10s : %.4f", tag, k.capitalize(), v)


# ── Grid Search multi-modèles ────────────────────────────────────────

def run_model_comparison(X_train, y_train, model_configs: dict, cv_folds: int) -> dict:
    """
    Pour chaque modèle, exécute un GridSearchCV avec TimeSeriesSplit.
    Retourne un dict {nom: {best_params, best_f1, best_estimator}}.
    """
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    results = {}

    for name, cfg in model_configs.items():
        logger.info("═══ Grid Search : %s ═══", name)
        grid = GridSearchCV(
            estimator=cfg["estimator"],
            param_grid=cfg["param_grid"],
            cv=tscv,
            scoring="f1",
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        grid.fit(X_train, y_train)

        logger.info("  Meilleur F1 (CV) : %.4f", grid.best_score_)
        for k, v in grid.best_params_.items():
            logger.info("    %-25s : %s", k, v)

        results[name] = {
            "best_params": grid.best_params_,
            "best_f1_cv": round(grid.best_score_, 4),
            "best_estimator": grid.best_estimator_,
        }

    return results


def print_comparison_table(comparison: dict, X_test, y_test) -> pd.DataFrame:
    """Évalue chaque meilleur modèle sur le test set et affiche un tableau comparatif."""
    rows = []
    for name, info in comparison.items():
        est = info["best_estimator"]
        y_pred = est.predict(X_test)
        y_proba = est.predict_proba(X_test)[:, 1]
        m = compute_metrics(y_test, y_pred, y_proba)
        m["model"] = name
        m["f1_cv"] = info["best_f1_cv"]
        rows.append(m)

    table = pd.DataFrame(rows).set_index("model")
    table = table[["f1_cv", "accuracy", "precision", "recall", "f1", "roc_auc", "brier"]]
    # Sélection sur la CV (f1_cv) pour éviter d'utiliser le hold-out à la fois
    # pour choisir le gagnant ET pour reporter sa performance (leakage indirect).
    table = table.sort_values("f1_cv", ascending=False)

    logger.info("═══ Comparaison des modèles (tri par F1 CV) ═══")
    logger.info("\n%s", table.to_string())
    return table


# ── Calibration ──────────────────────────────────────────────────────

def calibrate_model(model, X_train, y_train, method: str = "auto"):
    """Calibre les probabilités du modèle.

    method="auto" (défaut) : compare `none` (modèle brut), `isotonic` et
    `sigmoid` sur une validation temporelle interne au train (derniers 20 %)
    et conserve la meilleure au sens du Brier. Points clés :
      - `none` est un vrai candidat : un GBM entraîné en log-loss est souvent
        DÉJÀ bien calibré ; sur un petit jeu, toute recalibration post-hoc
        dégrade alors le Brier. Sans ce candidat, on calibrait à perte.
      - l'isotonique (non-paramétrique) sur-apprend sa courbe quand les
        coupures sont rares ; la sigmoïde (Platt, 2 paramètres) est plus
        robuste mais pas toujours utile.
    Le choix piloté par les données évite un seuil arbitraire et s'adapte au
    `scope`.

    method="none" / "isotonic" / "sigmoid" : force le comportement.

    Retourne (modèle_servi, méthode_retenue). Si la méthode retenue est
    "none", le modèle brut est renvoyé tel quel (aucun wrapper).
    """
    from sklearn.base import clone

    tscv_calib = TimeSeriesSplit(n_splits=3)

    def _fit_calibrated(meth: str, X, y):
        cal = CalibratedClassifierCV(estimator=model, method=meth, cv=tscv_calib)
        cal.fit(X, y)
        return cal

    if method == "auto":
        n = len(X_train)
        split = max(1, int(n * 0.8))
        X_fit, X_val = X_train.iloc[:split], X_train.iloc[split:]
        y_fit, y_val = y_train.iloc[:split], y_train.iloc[split:]
        scores: dict[str, float] = {}
        # Comparaison possible seulement si la validation contient les 2 classes.
        if len(X_val) > 0 and y_val.nunique() > 1:
            # Candidat "none" : modèle brut réajusté sur X_fit (comparaison juste).
            try:
                raw = clone(model)
                raw.fit(X_fit, y_fit)
                scores["none"] = brier_score_loss(y_val, raw.predict_proba(X_val)[:, 1])
            except Exception as exc:
                logger.warning("Évaluation 'none' impossible : %s", exc)
            for meth in ("isotonic", "sigmoid"):
                try:
                    cal_m = _fit_calibrated(meth, X_fit, y_fit)
                    scores[meth] = brier_score_loss(y_val, cal_m.predict_proba(X_val)[:, 1])
                except Exception as exc:
                    logger.warning("Calibration %s impossible : %s", meth, exc)
            # "none" (modèle brut) est la référence : un GBM entraîné en
            # log-loss est souvent déjà bien calibré. On n'adopte une
            # recalibration QUE si elle bat 'none' d'une marge NETTE (>5 %
            # relatif sur le Brier). Sinon l'écart n'est que du bruit
            # d'échantillonnage (le hold-out montre alors que 'none'
            # généralise mieux). Principe de parcimonie.
            margin = 0.05
            none_score = scores.get("none", float("inf"))
            others = {m: s for m, s in scores.items() if m != "none"}
            best_other = min(others, key=others.get) if others else None
            if best_other is not None and others[best_other] < none_score * (1 - margin):
                chosen = best_other
            else:
                chosen = "none" if "none" in scores else (
                    min(scores, key=scores.get) if scores else "none"
                )
            logger.info(
                "Calibration auto → %s  (Brier validation : %s ; marge %d%% vs 'none')",
                chosen, {k: round(v, 4) for k, v in scores.items()}, int(margin * 100),
            )
        else:
            chosen = "none"
            logger.info("Calibration auto → none (validation insuffisante pour comparer).")
    else:
        chosen = method
        logger.info("Calibration forcée → %s", chosen)

    if chosen == "none":
        logger.info("═══ Pas de recalibration (modèle brut conservé : déjà le mieux calibré) ═══")
        return model, "none"

    logger.info("═══ Calibration finale (%s) ═══", chosen)
    calibrated = _fit_calibrated(chosen, X_train, y_train)
    return calibrated, chosen


def evaluate_calibration(y_true, y_proba_raw, y_proba_cal) -> None:
    brier_raw = brier_score_loss(y_true, y_proba_raw)
    brier_cal = brier_score_loss(y_true, y_proba_cal)
    rel = (brier_raw - brier_cal) / brier_raw * 100 if brier_raw > 0 else 0.0
    sens = "mieux" if rel >= 0 else "moins bien"
    logger.info("Brier (brut) : %.4f → Brier (calibré) : %.4f  (%+.1f%% → %s)",
                brier_raw, brier_cal, rel, sens)

    for label, proba in [("brut", y_proba_raw), ("calibré", y_proba_cal)]:
        try:
            frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=5, strategy="quantile")
            logger.info("Calibration (%s) :", label)
            for mp, fp in zip(mean_pred, frac_pos):
                logger.info("  Prédit: %.2f → Observé: %.2f", mp, fp)
        except ValueError:
            pass


# ── Feature importance ───────────────────────────────────────────────

def extract_feature_importances(model, feature_names: list) -> pd.DataFrame:
    imp = None
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "calibrated_classifiers_"):
        base = model.calibrated_classifiers_[0].estimator
        if hasattr(base, "feature_importances_"):
            imp = base.feature_importances_

    if imp is None:
        return pd.DataFrame({"feature": feature_names, "importance": 0.0})

    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    df = df.sort_values("importance", ascending=False)

    logger.info("── Top 15 features (MDI) ──")
    for _, row in df.head(15).iterrows():
        logger.info("  %-40s %.4f", row["feature"], row["importance"])
    return df


# ── SHAP ─────────────────────────────────────────────────────────────

def compute_and_save_shap(
    model,
    X_test: pd.DataFrame,
    feature_names: list,
    sample_size: int | None = None,
    save_full_artifacts: bool = True,
) -> None:
    """
    Calcule les SHAP values via TreeExplainer et sauvegarde :
      - models/shap_values.npz   (matrice SHAP + expected value)
      - models/shap_feature_importance.csv  (|SHAP| moyen global)
    """
    logger.info("═══ Calcul des SHAP values (TreeExplainer) ═══")
    # Lazy import : `shap` charge ~150 MB de C extensions ; éviter ce coût
    # quand on n'appelle que les utilitaires de métriques.
    import shap

    # Si le modèle est calibré, on calcule les SHAP sur la MOYENNE des K
    # classifieurs internes (avant : on prenait seulement le premier, ce
    # qui sous-représentait la calibration).
    calibrated_estimators = (
        [c.estimator for c in model.calibrated_classifiers_]
        if hasattr(model, "calibrated_classifiers_") else [model]
    )

    if sample_size is not None and len(X_test) > sample_size:
        X_shap = X_test.sample(n=sample_size, random_state=RANDOM_SEED)
        logger.info("SHAP échantillonné : %d/%d lignes", len(X_shap), len(X_test))
    else:
        X_shap = X_test

    sv_list = []
    exp_list = []
    explainer = None  # explainer du dernier estimator, utilisé pour la persistance
    from src.utils.shap_compat import tree_expected_value, tree_shap_values

    for est in calibrated_estimators:
        explainer = shap.TreeExplainer(est)
        sv_k = tree_shap_values(explainer, X_shap)
        exp_k = tree_expected_value(explainer)
        sv_list.append(sv_k)
        exp_list.append(exp_k)

    sv = np.mean(sv_list, axis=0)
    expected = float(np.mean(exp_list))
    if len(calibrated_estimators) > 1:
        logger.info("SHAP moyenné sur %d classifieurs calibrés.", len(calibrated_estimators))

    np.savez_compressed(
        MODELS_DIR / "shap_values.npz",
        shap_values=sv,
        expected_value=np.array([expected]),
        feature_names=np.array(feature_names),
    )
    logger.info("SHAP values sauvegardées → models/shap_values.npz  (%d lignes, %d features)",
                sv.shape[0], sv.shape[1])

    mean_abs = np.abs(sv).mean(axis=0)
    shap_imp = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)
    shap_imp.to_csv(MODELS_DIR / "shap_feature_importance.csv", index=False)

    logger.info("── Top 15 features (SHAP |mean|) ──")
    for _, row in shap_imp.head(15).iterrows():
        logger.info("  %-40s %.4f", row["feature"], row["mean_abs_shap"])

    if save_full_artifacts:
        joblib.dump(explainer, MODELS_DIR / "shap_explainer.joblib")
        logger.info("Explainer SHAP sauvegardé → models/shap_explainer.joblib")
    else:
        logger.info("Explainer SHAP non sauvegardé (mode artefacts légers).")


# ── Sauvegarde ───────────────────────────────────────────────────────

def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Modèle sauvegardé → %s", path)


# ── Pipeline principal ───────────────────────────────────────────────

def run(
    fast_mode: bool = FAST_MODE,
    grid_scale: str = GRID_SCALE,
    cv_folds: int | None = None,
    shap_sample_size: int | None = SHAP_SAMPLE_SIZE,
    save_full_artifacts: bool = True,
    scope: str = "real",
    calibration_method: str = "auto",
) -> None:
    t0 = perf_counter()
    df = load_table(FEATURES_DIR / "features_dataset.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])

    # ── Portée d'entraînement ────────────────────────────────────────
    # scope="real" (défaut) : on n'entraîne et n'évalue QUE sur les hôpitaux
    # dont les coupures sont *réellement observées* (Lacor). Les sites ERIC/NYC
    # ont des coupures générées par une formule stochastique connue : les
    # garder revient à entraîner sur ~94 % de bruit et gonfle artificiellement
    # le F1 global. scope="all" rétablit le comportement multi-hôpitaux complet
    # (à n'utiliser qu'en connaissance de cause — métriques globales biaisées).
    if scope not in ("real", "all"):
        raise ValueError(f"scope invalide : {scope!r} (attendu 'real' ou 'all').")
    if scope == "real" and "hospital" in df.columns:
        before = len(df)
        df = df[df["hospital"].isin(REAL_DATA_HOSPITALS)].reset_index(drop=True)
        logger.info(
            "Scope=real : %d → %d lignes conservées (cible réelle : %s)",
            before, len(df), sorted(REAL_DATA_HOSPITALS),
        )
        if df.empty:
            raise ValueError(
                "scope='real' mais aucune ligne pour les hôpitaux à cible réelle "
                f"({sorted(REAL_DATA_HOSPITALS)}). Vérifiez features_dataset / la "
                "colonne 'hospital'."
            )
    elif scope == "all":
        logger.warning(
            "Scope=all : entraînement multi-hôpitaux incluant des coupures "
            "SYNTHÉTIQUES (ERIC/NYC). Les métriques globales sont biaisées — "
            "ne se fier qu'aux métriques 'real'. Voir docs/."
        )

    effective_grid_scale = "compact" if fast_mode else grid_scale
    effective_cv_folds = cv_folds if cv_folds is not None else (3 if fast_mode else CV_FOLDS)
    effective_shap_sample_size = shap_sample_size if shap_sample_size is not None else SHAP_SAMPLE_SIZE

    X, y = prepare_data(df)
    logger.info("Features : %d colonnes, %d lignes", X.shape[1], X.shape[0])
    logger.info("Cible (is_outage) : %d coupures / %d total (%.1f%%)",
                y.sum(), len(y), 100 * y.mean())

    X_train, X_test, y_train, y_test = temporal_split(df, X, y)
    logger.info("Train : %d | Test : %d", len(X_train), len(X_test))

    # Réordonner le train chronologiquement (datetime global). `temporal_split`
    # empile les hôpitaux les uns après les autres ; sans ce tri, les folds
    # `TimeSeriesSplit` (GridSearch + calibration) découpent une séquence
    # empilée par site et n'ont aucune sémantique temporelle réelle. Tri stable
    # (mergesort) → à timestamp égal, l'ordre par hôpital est conservé, donc
    # déterministe. X_test n'a pas besoin d'être réordonné (pas de CV dessus).
    if "datetime" in df.columns:
        train_order = (
            df.loc[X_train.index, "datetime"].sort_values(kind="mergesort").index
        )
        X_train = X_train.loc[train_order]
        y_train = y_train.loc[train_order]

    log_hospital_split_stats(df, X_train.index, X_test.index)
    hospitals_test = df.loc[X_test.index, "hospital"] if "hospital" in df.columns else pd.Series(["unknown"] * len(X_test))

    # `OUTAGE_RATIO` dynamique sur le train (et plus une constante codée
    # en dur) : nourrit `scale_pos_weight` et `class_weight` pour s'adapter
    # à la composition multi-hôpitaux courante.
    outage_ratio_train = float(y_train.mean()) if len(y_train) > 0 else DEFAULT_OUTAGE_RATIO
    if outage_ratio_train <= 0:
        outage_ratio_train = DEFAULT_OUTAGE_RATIO
    logger.info("Taux de coupure (train) : %.3f%% → scale_pos_weight cible ~%d",
                100 * outage_ratio_train, max(2, round(1 / outage_ratio_train)))
    model_configs = build_model_configs(effective_grid_scale, outage_ratio=outage_ratio_train)

    # ── 1. Comparaison multi-modèles ──────────────────────────────
    step = perf_counter()
    comparison = run_model_comparison(
        X_train,
        y_train,
        model_configs=model_configs,
        cv_folds=effective_cv_folds,
    )
    logger.info("Timing comparaison modèles : %.2fs", perf_counter() - step)
    comp_table = print_comparison_table(comparison, X_test, y_test)
    comp_table.to_csv(MODELS_DIR / "model_comparison.csv")

    winner_name = comp_table.index[0]
    winner_model = comparison[winner_name]["best_estimator"]
    winner_params = comparison[winner_name]["best_params"]
    logger.info("═══ Meilleur modèle : %s ═══", winner_name)

    # ── 2. Évaluation du gagnant (brut) ──────────────────────────
    y_pred_raw = winner_model.predict(X_test)
    y_proba_raw = winner_model.predict_proba(X_test)[:, 1]
    raw_metrics = compute_metrics(y_test, y_pred_raw, y_proba_raw)
    logger.info("── %s (brut) ──", winner_name)
    _log_metrics(raw_metrics, prefix=winner_name)
    logger.info("\n%s", classification_report(y_test, y_pred_raw, zero_division=0))
    raw_by_source = compute_metrics_by_source(y_test, y_pred_raw, y_proba_raw, hospitals_test)
    if "real" in raw_by_source:
        logger.info("── %s (brut, Lacor RÉEL, n=%d, taux=%.2f%%) ──",
                    winner_name, raw_by_source["real_n"], 100 * raw_by_source["real_outage_rate"])
        _log_metrics(raw_by_source["real"], prefix=f"{winner_name} réel")
    if "synthetic" in raw_by_source:
        logger.info("── %s (brut, ERIC/NYC SYNTHÉTIQUE, n=%d, taux=%.2f%%) ──",
                    winner_name, raw_by_source["synthetic_n"], 100 * raw_by_source["synthetic_outage_rate"])
        _log_metrics(raw_by_source["synthetic"], prefix=f"{winner_name} synth.")
        logger.warning(
            "⚠ Les métriques 'synthetic' reflètent partiellement la formule "
            "qui a généré les coupures ERIC/NYC — ce n'est PAS un signal de "
            "généralisation à des données terrain. Voir docs/."
        )

    # ── 3. Calibration ────────────────────────────────────────────
    step = perf_counter()
    calibrated, calibration_method_used = calibrate_model(
        winner_model, X_train, y_train, method=calibration_method
    )
    y_pred_cal = calibrated.predict(X_test)
    y_proba_cal = calibrated.predict_proba(X_test)[:, 1]
    cal_metrics = compute_metrics(y_test, y_pred_cal, y_proba_cal)
    logger.info("── %s (calibré) ──", winner_name)
    _log_metrics(cal_metrics, prefix=f"{winner_name} cal.")
    logger.info("\n%s", classification_report(y_test, y_pred_cal, zero_division=0))
    cal_by_source = compute_metrics_by_source(y_test, y_pred_cal, y_proba_cal, hospitals_test)
    if "real" in cal_by_source:
        logger.info("── %s (calibré, Lacor RÉEL) ──", winner_name)
        _log_metrics(cal_by_source["real"], prefix=f"{winner_name} cal. réel")
    if "synthetic" in cal_by_source:
        logger.info("── %s (calibré, ERIC/NYC SYNTHÉTIQUE) ──", winner_name)
        _log_metrics(cal_by_source["synthetic"], prefix=f"{winner_name} cal. synth.")
    evaluate_calibration(y_test, y_proba_raw, y_proba_cal)
    logger.info("Timing calibration + évaluation : %.2fs", perf_counter() - step)

    # ── 4. Feature importance (MDI) ──────────────────────────────
    step = perf_counter()
    importances = extract_feature_importances(winner_model, list(X.columns))
    importances.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    logger.info("Timing feature importance : %.2fs", perf_counter() - step)

    # ── 5. SHAP values ───────────────────────────────────────────
    step = perf_counter()
    compute_and_save_shap(
        calibrated,
        X_test,
        list(X.columns),
        sample_size=effective_shap_sample_size,
        save_full_artifacts=save_full_artifacts,
    )
    logger.info("Timing SHAP : %.2fs", perf_counter() - step)

    # ── 6. Sauvegarder ───────────────────────────────────────────
    # Noms neutres : le gagnant peut être RF, XGBoost ou LightGBM (cf. winner).
    # Les anciens noms `*_rf.joblib` étaient trompeurs.
    save_model(winner_model, MODELS_DIR / "baseline_model.joblib")
    save_model(calibrated, MODELS_DIR / "calibrated_model.joblib")

    summary = {
        "winner": winner_name,
        "winner_params": winner_params,
        "winner_selection_metric": "f1_cv",
        "scope": scope,
        "trained_hospitals": sorted(df["hospital"].unique()) if "hospital" in df.columns else ["unknown"],
        "calibration_method": calibration_method_used,
        "n_cv_folds": effective_cv_folds,
        "grid_scale": effective_grid_scale,
        "fast_mode": fast_mode,
        "shap_sample_size": effective_shap_sample_size,
        "models_compared": list(model_configs.keys()),
        "test_metrics_raw": raw_metrics,
        "test_metrics_calibrated": cal_metrics,
        "test_metrics_by_source_raw": raw_by_source,
        "test_metrics_by_source_calibrated": cal_by_source,
        "real_data_hospitals": sorted(REAL_DATA_HOSPITALS),
        "comparison": {
            name: {
                "best_params": info["best_params"],
                "best_f1_cv": info["best_f1_cv"],
            }
            for name, info in comparison.items()
        },
    }
    with open(MODELS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Résumé → models/training_summary.json")
    logger.info("Temps total entraînement : %.2fs", perf_counter() - t0)


if __name__ == "__main__":
    setup_logging()
    run()
