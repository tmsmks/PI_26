"""
Interface Streamlit — Prédiction de coupures d'électricité en hôpitaux.
Deux modes : Analyse historique + Simulation manuelle.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# `shap` est chargé à la demande (~150 MB de C extensions). Il est utilisé
# uniquement par `compute_shap_local` et `load_shap_explainer` : déférer
# l'import accélère le cold start Streamlit.

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils.config import (
    COLS_TO_DROP,
    MODELS_DIR,
    drop_external_signal_columns,
)
from src.app_data import (
    HOSPITAL_DISPLAY,
    _features_file_mtime,
    _forecast_file_mtime,
    _model_file_mtime,
    detect_hospital_data_sources,
    electricitymaps_snapshot_path,
    load_electricitymaps_snapshot,
    load_global_shap_importance,
    load_hospital_data,
    load_lacor_features,
    load_loadshedding,
    load_meteo_forecast,
    load_model,
    load_horizon_models,
    load_realtime_forecast,
    _horizon_models_mtime,
    load_shap_explainer,
)
from src.loadshedding import is_supported as loadshedding_supported

# ── Configuration ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Prédiction de coupures",
    page_icon="⚡",
    layout="wide",
)

# HOSPITAL_DISPLAY (catalogue filtré des hôpitaux visibles) est désormais
# défini dans src/app_data.py et importé ci-dessus.

# N'afficher que les hôpitaux avec données de consommation réelles
# (pas de profil estimé/cloné).
REAL_DATA_SOURCES = {"eric", "nyc_ll84"}
# Hôpitaux à RÉSEAU temps réel (Electricity Maps live) : la consommation y est
# estimée, mais le signal réseau (charge, mix, carbone) est une vraie donnée
# live → on les rend accessibles même en mode « données réelles » car ils
# alimentent la prévision temps réel de l'onglet « Prochaine coupure ».
REALTIME_GRID_SOURCES = {"africa_grid"}
REAL_HOSPITAL_KEYS = [
    k for k, v in HOSPITAL_DISPLAY.items()
    if k == "lacor_uganda"
    or v.get("data_source") in REAL_DATA_SOURCES
    or v.get("data_source") in REALTIME_GRID_SOURCES
]
ALL_HOSPITAL_KEYS = list(HOSPITAL_DISPLAY.keys())

from src.nowcast_horizons import predict_horizons
from src.ui_content import (
    DATA_SOURCES,
    REMOVED_DATA_SOURCES_NOTE,
    source_role_context_app,
    source_role_model_pipeline,
    FEATURE_CATEGORIES,
    FEATURE_LABELS,
)
from src.ui_components import (
    show_factors,
    show_risk_result,
    show_shap_waterfall,
    ui_step,
)


# ── Chargement ───────────────────────────────────────────────────────


# Noms de fichiers modèle : nouveaux noms neutres (le gagnant peut être RF,
# XGBoost ou LightGBM), avec repli sur les anciens `*_rf.joblib` pour la
# rétro-compatibilité tant que le pipeline n'a pas été ré-exécuté.


def _match_similar_historical_rows_bulk(
    hist_df: pd.DataFrame,
    target_hours: np.ndarray,
    target_months: np.ndarray,
    target_dows: np.ndarray,
    target_temps: np.ndarray,
) -> np.ndarray:
    """Pour chaque triplet (hour, month, dow, temp) de prévision, retourne
    l'index hist_df de la ligne la plus proche. Vectorisé (broadcasting
    numpy) plutôt qu'une boucle Python qui copiait `hist_df` 168 fois.
    """
    h = hist_df["hour"].to_numpy()
    m = hist_df["month"].to_numpy()
    d = hist_df["day_of_week"].to_numpy()
    t = hist_df["temperature_2m"].to_numpy()
    hist_index = hist_df.index.to_numpy()

    # Matrice (N_targets × N_hist) — pour 168 × 8760 ≈ 1.5M flottants, OK en RAM.
    score = (
        3.0 * np.abs(h[None, :] - target_hours[:, None])
        + 2.0 * np.abs(m[None, :] - target_months[:, None])
        + np.abs(d[None, :] - target_dows[:, None])
        + 0.1 * np.abs(t[None, :] - target_temps[:, None])
    )
    best = score.argmin(axis=1)
    return hist_index[best]


def build_forecast_predictions(
    hist_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    feature_cols: list[str],
    hospital_key: str,
) -> pd.DataFrame:
    """Pour chaque heure future du CSV prévisions, construit une ligne de
    features (consommation empruntée à l'heure historique similaire, météo
    remplacée par les prévisions) et prédit la probabilité de coupure.

    Optimisé : matching vectorisé + 1 seul appel `predict_proba` batché
    (avant : 168 itérations Python × `df.copy()` × `predict_proba` unitaire).
    """
    if forecast_df.empty:
        return pd.DataFrame()

    meteo_cols_forecast = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "wind_speed_10m", "wind_gusts_10m", "precipitation",
        "surface_pressure", "shortwave_radiation", "cloud_cover",
        "visibility", "et0_fao_evapotranspiration", "cape", "weathercode",
    ]

    fc = forecast_df.copy()
    fc["datetime"] = pd.to_datetime(fc["datetime"])
    if "temperature_2m" not in fc.columns:
        fc["temperature_2m"] = 25.0
    hours_t = fc["datetime"].dt.hour.to_numpy()
    months_t = fc["datetime"].dt.month.to_numpy()
    dows_t = fc["datetime"].dt.dayofweek.to_numpy()
    temps_t = fc["temperature_2m"].fillna(25.0).to_numpy(dtype=np.float64)

    # 1) Trouver l'historique le plus proche pour chaque heure prévue.
    best_idx = _match_similar_historical_rows_bulk(
        hist_df, hours_t, months_t, dows_t, temps_t,
    )
    feat_batch = hist_df.loc[best_idx, feature_cols].reset_index(drop=True).copy()

    # 2) Écraser les variables météo par les prévisions.
    for mcol in meteo_cols_forecast:
        if mcol in fc.columns and mcol in feat_batch.columns:
            feat_batch[mcol] = pd.to_numeric(fc[mcol].values, errors="coerce")

    # 3) Recalculer les variables dérivées cohérentes avec la météo prévue.
    if {"temperature_2m", "relative_humidity_2m"}.issubset(feat_batch.columns):
        feat_batch["temp_humidity_interaction"] = (
            feat_batch["temperature_2m"] * feat_batch["relative_humidity_2m"] / 100
        )
    if {"wind_speed_10m", "precipitation"}.issubset(feat_batch.columns):
        feat_batch["wind_precipitation_interaction"] = (
            feat_batch["wind_speed_10m"] * feat_batch["precipitation"]
        )
        if "rain_intensity" in feat_batch.columns:
            feat_batch["rain_intensity"] = (
                feat_batch["precipitation"] * feat_batch["wind_speed_10m"]
            )
    if "shortwave_radiation" in feat_batch.columns and "solar_available" in feat_batch.columns:
        feat_batch["solar_available"] = (feat_batch["shortwave_radiation"] > 50).astype(int)
    if "temperature_2m" in feat_batch.columns and "heat_stress" in feat_batch.columns:
        feat_batch["heat_stress"] = (feat_batch["temperature_2m"] > 30).astype(int)
    if "cloud_cover" in feat_batch.columns and "cloud_cover_pct" in feat_batch.columns:
        feat_batch["cloud_cover_pct"] = feat_batch["cloud_cover"]
    if "visibility" in feat_batch.columns and "visibility_m" in feat_batch.columns:
        feat_batch["visibility_m"] = feat_batch["visibility"]
    if "et0_fao_evapotranspiration" in feat_batch.columns and "evapotranspiration" in feat_batch.columns:
        feat_batch["evapotranspiration"] = feat_batch["et0_fao_evapotranspiration"]

    # 4) Mettre à jour les variables temporelles cycliques pour les heures futures.
    feat_batch["hour"] = hours_t
    feat_batch["month"] = months_t
    feat_batch["day_of_week"] = dows_t
    feat_batch["is_weekend"] = (dows_t >= 5).astype(int)
    feat_batch["hour_sin"] = np.sin(2 * np.pi * hours_t / 24)
    feat_batch["hour_cos"] = np.cos(2 * np.pi * hours_t / 24)
    feat_batch["month_sin"] = np.sin(2 * np.pi * months_t / 12)
    feat_batch["month_cos"] = np.cos(2 * np.pi * months_t / 12)

    # 5) Prédiction batchée via le modèle hôpital (entraîné sur Lacor).
    #    Pour un site ≠ Lacor : score illustratif (cf. site_profile_notes).
    proba_adj = site_predict_proba(feat_batch)

    return pd.DataFrame({
        "datetime": fc["datetime"].values,
        "outage_probability": proba_adj,
        "temperature_2m": fc.get("temperature_2m", pd.Series(0.0, index=fc.index)).astype(float).values,
        "precipitation": fc.get("precipitation", pd.Series(0.0, index=fc.index)).astype(float).values,
        "wind_speed_10m": fc.get("wind_speed_10m", pd.Series(0.0, index=fc.index)).astype(float).values,
        "shortwave_radiation": fc.get("shortwave_radiation", pd.Series(0.0, index=fc.index)).astype(float).values,
    })


# ── Détection des sources de données disponibles par hôpital ───────
# Permet d'afficher dans l'UI exactement de quelles sources chaque hôpital
# bénéficie. Les fichiers sont regardés sur disque, donc ça reflète l'état
# réel du projet.


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    drop = [c for c in COLS_TO_DROP if c in df.columns]
    cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]
    # Exclure les signaux externes du jeu de features du modèle (cf. #3).
    # (En pratique, feature_cols est ensuite réaligné sur model.feature_names_in_,
    # mais on filtre ici aussi pour la cohérence et le cas sans feature_names_in_.)
    return drop_external_signal_columns(cols)


# ── Fonctions utilitaires ────────────────────────────────────────────

def ensure_numeric_feature_frame(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Aligne et convertit les features en numérique pour l'inférence."""
    out = frame.copy()
    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0.0
        series = out[col]
        if not pd.api.types.is_numeric_dtype(series):
            as_num = pd.to_numeric(series, errors="coerce")
            if as_num.notna().any():
                out[col] = as_num
            else:
                out[col] = pd.factorize(series.fillna("NA").astype(str))[0].astype(float)
        out[col] = (
            pd.to_numeric(out[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
    return out[feature_cols]


def _extract_feature_importances(model) -> np.ndarray | None:
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "estimators_"):
        return np.mean([e.feature_importances_ for e in model.estimators_], axis=0)
    if hasattr(model, "calibrated_classifiers_"):
        base = model.calibrated_classifiers_[0].estimator
        if hasattr(base, "feature_importances_"):
            return base.feature_importances_
    fi_path = MODELS_DIR / "feature_importance.csv"
    if fi_path.exists():
        fi_df = pd.read_csv(fi_path)
        return fi_df["importance"].values
    return None


def get_top_factors(model, feature_cols: list[str], values: pd.Series, top_n: int = 5):
    imp_arr = _extract_feature_importances(model)
    if imp_arr is not None and len(imp_arr) == len(feature_cols):
        importances = pd.Series(imp_arr, index=feature_cols)
    else:
        fi_path = MODELS_DIR / "feature_importance.csv"
        if fi_path.exists():
            fi_df = pd.read_csv(fi_path)
            importances = pd.Series(fi_df["importance"].values, index=fi_df["feature"].values)
            importances = importances.reindex(feature_cols, fill_value=0.0)
        else:
            importances = pd.Series(1.0 / len(feature_cols), index=feature_cols)

    importances = importances.sort_values(ascending=False).head(top_n)

    factors = []
    for feat, imp in importances.items():
        factors.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "importance": imp,
            "value": values[feat] if feat in values.index else 0,
        })
    return factors


def show_top_factors_panel(top_n: int = 12) -> None:
    """Top N facteurs globaux du modèle Lacor (importance SHAP)."""
    shap_df = load_global_shap_importance()
    if shap_df is None or shap_df.empty:
        st.info("Aucune importance SHAP globale disponible (relancez l'entraînement).")
        return

    top = shap_df.head(top_n)
    max_val = float(top["mean_abs_shap"].max())

    fig = go.Figure()
    for _, row in top.iloc[::-1].iterrows():
        cat = FEATURE_CATEGORIES.get(row["category"], FEATURE_CATEGORIES["other"])
        fig.add_trace(go.Bar(
            x=[row["mean_abs_shap"]],
            y=[f"{cat['emoji']}  {row['label']}"],
            orientation="h",
            marker_color=cat["color"],
            hovertemplate=(
                f"<b>{row['label']}</b><br>"
                f"Catégorie : {cat['label']}<br>"
                f"SHAP |moyen| : {row['mean_abs_shap']:.4f}<extra></extra>"
            ),
            showlegend=False,
            text=[f"{row['mean_abs_shap']:.3f}"],
            textposition="outside",
        ))
    chart_title = f"Top {top_n} facteurs — modèle Lacor (importance SHAP)"
    fig.update_layout(
        title=dict(text=chart_title, font=dict(size=14)),
        xaxis=dict(title="Impact moyen sur la prédiction", range=[0, max_val * 1.15]),
        height=max(360, top_n * 30),
        margin=dict(l=260, r=60, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")


def show_category_breakdown() -> None:
    """Importance SHAP cumulée par catégorie de feature (modèle Lacor)."""
    shap_df = load_global_shap_importance()
    if shap_df is None or shap_df.empty:
        return
    by_cat = shap_df.groupby("category")["mean_abs_shap"].sum().sort_values(ascending=True)
    by_cat = by_cat[by_cat > 0]
    labels = [
        f"{FEATURE_CATEGORIES[c]['emoji']}  {FEATURE_CATEGORIES[c]['label']}"
        for c in by_cat.index
    ]
    colors = [FEATURE_CATEGORIES[c]["color"] for c in by_cat.index]
    fig = go.Figure(go.Bar(
        x=by_cat.values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}" for v in by_cat.values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>SHAP cumulé : %{x:.3f}<extra></extra>",
    ))
    cat_title = "Importance par catégorie — modèle Lacor"
    fig.update_layout(
        title=dict(text=cat_title, font=dict(size=14)),
        xaxis=dict(title="Somme des |SHAP| par catégorie"),
        height=320,
        margin=dict(l=220, r=60, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")


def _source_card_html(src: dict) -> str:
    """Carte HTML pour une source (pipeline ou contexte app)."""
    star = " ⭐" if src.get("key") else ""
    if source_role_model_pipeline(src):
        use_lbl, use_col, border = "✓ utilisé par le modèle Lacor", "#2ecc71", "#2ecc71"
    else:
        use_lbl, use_col, border = "contexte app — hors modèle", "#e67e22", "#e67e22"
    return (
        f"<div style='border:1px solid #e0e0e0;border-left:4px solid {border};"
        f"border-radius:8px;padding:10px 14px;margin-bottom:8px;background:#fafafa'>"
        f"<div style='display:flex;justify-content:space-between;"
        f"align-items:center;gap:8px'>"
        f"<b style='font-size:14px'>{src['icon']}  {src['name']}{star}</b>"
        f"<span style='background:#34495e22;color:#34495e;"
        f"padding:2px 8px;border-radius:10px;font-size:10px;"
        f"font-weight:600'>{src['type']}</span>"
        f"</div>"
        f"<div style='color:#666;font-size:12px;margin-top:4px'>"
        f"{src['desc']}</div>"
        f"<div style='margin-top:4px;font-size:10px;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:0.5px;color:{use_col}'>"
        f"● {use_lbl}</div>"
        f"</div>"
    )


def show_data_sources_panel() -> None:
    """Affiche les sources actives du pipeline, groupées modèle vs contexte."""
    model_sources = [s for s in DATA_SOURCES if source_role_model_pipeline(s)]
    context_sources = [s for s in DATA_SOURCES if source_role_context_app(s)]

    st.markdown("##### Alimentent le modèle (conso, météo, historique coupures)")
    cols = st.columns(2)
    for i, src in enumerate(model_sources):
        with cols[i % 2]:
            st.markdown(_source_card_html(src), unsafe_allow_html=True)

    st.markdown("##### Contexte application uniquement (temps réel, hors modèle)")
    cols2 = st.columns(2)
    for i, src in enumerate(context_sources):
        with cols2[i % 2]:
            st.markdown(_source_card_html(src), unsafe_allow_html=True)

    st.caption(REMOVED_DATA_SOURCES_NOTE)


def compute_shap_local(explainer, row_df: pd.DataFrame, feature_cols: list[str]):
    """Calcule les SHAP values pour une seule ligne et retourne (shap_values_1d, expected)."""
    if explainer is None:
        return None, None
    try:
        sv = explainer.shap_values(row_df[feature_cols])
        # Normaliser la sortie SHAP vers la classe positive (cf. train_baseline) :
        # liste [c0, c1] | ndarray 3D (n, features, classes) | 2D (n, features).
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1] if sv.shape[-1] > 1 else sv[:, :, 0]
        expected = np.asarray(explainer.expected_value).ravel()
        expected = float(expected[1] if expected.size > 1 else expected[0])
        # Une seule ligne → vecteur 1D de longueur n_features.
        sv_row = sv[0] if sv.ndim == 2 else np.ravel(sv)
        return sv_row, expected
    except Exception:
        return None, None


def apply_extrapolation_stress(
    proba_model: float,
    params: dict,
    df: pd.DataFrame,
) -> tuple[float, list[str]]:
    """
    Le Random Forest ne sait pas extrapoler au-delà des données d'entraînement.
    Cette fonction détecte les paramètres qui dépassent les bornes connues
    et applique un bonus de risque proportionnel au dépassement.

    Retourne (probabilité_ajustée, liste_des_facteurs_de_stress).
    """
    stress = 0.0
    details = []

    bounds = {
        "total_load_kw": ("Consommation", df["total_load_kw"].max(), df["total_load_kw"].quantile(0.95)),
        "temperature_2m": ("Température", df["temperature_2m"].max(), df["temperature_2m"].quantile(0.95)),
        "wind_speed_10m": ("Vent", df["wind_speed_10m"].max(), df["wind_speed_10m"].quantile(0.95)),
        "precipitation": ("Précipitations", df["precipitation"].max(), df["precipitation"].quantile(0.95)),
    }

    param_map = {
        "total_load_kw": params["total_load_kw"],
        "temperature_2m": params["temperature_2m"],
        "wind_speed_10m": params["wind_speed"],
        "precipitation": params["precipitation"],
    }

    for key, (label, data_max, p95) in bounds.items():
        val = param_map[key]
        if val > data_max:
            overshoot = (val - data_max) / max(data_max - p95, 1)
            bonus = min(0.25, overshoot * 0.10)
            stress += bonus
            details.append(f"{label} ({val:.0f}) dépasse le max observé ({data_max:.0f})")
        elif val > p95:
            overshoot = (val - p95) / max(data_max - p95, 1)
            bonus = min(0.10, overshoot * 0.05)
            stress += bonus
            details.append(f"{label} ({val:.0f}) au-dessus du 95e percentile ({p95:.0f})")

    # Synergie : si plusieurs facteurs sont en stress simultanément, le risque est amplifié
    if len(details) >= 2:
        stress *= 1.0 + 0.3 * (len(details) - 1)

    proba_adjusted = min(0.99, proba_model + stress)
    return proba_adjusted, details


def build_simulation_row(params: dict, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Construit une ligne de features à partir des paramètres utilisateur.

    Stratégie : on cherche dans les données réelles la ligne la plus similaire
    aux conditions demandées (même heure, même mois, consommation proche).
    On part de cette ligne RÉELLE (qui a des features rolling cohérentes)
    et on ne remplace que les paramètres que l'utilisateur a modifiés.
    """
    hour = params["hour"]
    month = params["month"]
    day_of_week = params["day_of_week"]
    load = params["total_load_kw"]
    solar = params["solar_pv_kw"]
    base = params["base_load_kw"]
    steril = params["sterilization_kw"]

    candidates = df.copy()
    candidates["_hour_dist"] = abs(candidates["hour"] - hour)
    candidates["_month_dist"] = abs(candidates["month"] - month)
    candidates["_load_dist"] = abs(candidates["total_load_kw"] - load)
    candidates["_score"] = (
        candidates["_hour_dist"] * 3
        + candidates["_month_dist"]
        + candidates["_load_dist"] / 30
    )
    best_idx = candidates["_score"].idxmin()
    ref = df.loc[best_idx, feature_cols].copy()

    ref["total_load_kw"] = load
    ref["solar_pv_kw"] = solar
    ref["base_load_kw"] = base
    ref["sterilization_kw"] = steril
    ref["temperature_2m"] = params["temperature_2m"]
    ref["relative_humidity_2m"] = params["humidity"]
    ref["wind_speed_10m"] = params["wind_speed"]
    ref["precipitation"] = params["precipitation"]
    ref["surface_pressure"] = params["pressure"]
    ref["shortwave_radiation"] = params["radiation"]

    ref["hour"] = hour
    ref["month"] = month
    ref["day_of_week"] = day_of_week
    ref["is_weekend"] = 1 if day_of_week >= 5 else 0
    ref["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    ref["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    ref["month_sin"] = np.sin(2 * np.pi * month / 12)
    ref["month_cos"] = np.cos(2 * np.pi * month / 12)
    ref["is_public_holiday"] = 0

    total = max(load, 1.0)
    ref["solar_ratio"] = solar / total
    ref["base_load_ratio"] = base / total
    ref["peak_ratio"] = load / max(ref.get("load_rolling_24h", load), 1.0)

    ref["temp_humidity_interaction"] = params["temperature_2m"] * params["humidity"] / 100
    ref["wind_precipitation_interaction"] = params["wind_speed"] * params["precipitation"]
    ref["solar_available"] = 1 if params["radiation"] > 50 else 0
    ref["heat_stress"] = 1 if params["temperature_2m"] > 30 else 0

    # Météo avancée : recalcul cohérent avec les paramètres utilisateur
    t = params["temperature_2m"]
    rh = params["humidity"]
    a, b = 17.27, 237.7
    gamma = (a * t / (b + t)) + np.log(rh / 100 + 1e-10)
    ref["dew_point_2m"] = b * gamma / (a - gamma)
    ref["rain_intensity"] = params["precipitation"] * params["wind_speed"]
    ref["humidity_change_3h"] = 0.0
    ref["pressure_change_3h"] = 0.0

    row_df = pd.DataFrame([ref])
    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = 0
    return row_df[feature_cols]


# ── Hero / en-tête ──────────────────────────────────────────────────

_mtime = _model_file_mtime()
model = load_model(_mtime)
shap_explainer = load_shap_explainer(_mtime)
lacor_df = load_lacor_features(_features_file_mtime())
feature_cols = get_feature_columns(lacor_df)

HORIZON_MODELS = load_horizon_models(_horizon_models_mtime())

# ── Garde-fou : cohérence des features entre entraînement et inférence ──
# Si quelqu'un régénère le dataset sans réentraîner (ou inversement), les
# colonnes peuvent diverger silencieusement et produire des prédictions
# faussées. On force ici l'ordre à correspondre à celui du modèle.
def _model_feature_names(_model):
    fn = getattr(_model, "feature_names_in_", None)
    if fn is None and hasattr(_model, "calibrated_classifiers_"):
        fn = getattr(_model.calibrated_classifiers_[0].estimator,
                      "feature_names_in_", None)
    return list(fn) if fn is not None else None


_model_feats = _model_feature_names(model)
if _model_feats is not None:
    missing_in_data = [c for c in _model_feats if c not in feature_cols]
    extra_in_data = [c for c in feature_cols if c not in _model_feats]
    if missing_in_data:
        st.warning(
            "**Désynchronisation features ↔ modèle** — colonnes attendues par le "
            "modèle absentes du dataset. Re-lancez `python run_pipeline.py`.\n\n"
            f"- Manquantes : {missing_in_data[:8]}{' …' if len(missing_in_data) > 8 else ''}"
        )
    elif extra_in_data:
        st.caption(
            f"ℹ️ Colonnes ignorées (absentes du modèle) : "
            f"{', '.join(extra_in_data[:6])}{' …' if len(extra_in_data) > 6 else ''}. "
            "Régénérez les features avec `python run_pipeline.py` pour nettoyer le dataset."
        )
    feature_cols = _model_feats

TRAINED_HOSPITAL = "lacor_uganda"  # seul site avec un modèle propre (données réelles)


def site_predict_proba(frame: pd.DataFrame) -> np.ndarray:
    """Probabilité de coupure via le modèle Lacor (score illustratif hors Lacor)."""
    X = ensure_numeric_feature_frame(frame, feature_cols)
    return model.predict_proba(X)[:, 1].astype(float)


def forecast_next_outage(
    full_df: pd.DataFrame,
    ref_ts: pd.Timestamp,
) -> dict[int, float]:
    """P(coupure dans les 1/3/6 h) : modèles horizons si entraînés, sinon repli
    nowcast horaire."""
    if full_df is None or full_df.empty:
        return {}

    def _predict(frame: pd.DataFrame) -> np.ndarray:
        X = ensure_numeric_feature_frame(frame, feature_cols)
        return site_predict_proba(X)

    return predict_horizons(
        full_df,
        ref_ts,
        feature_cols,
        _predict,
        horizon_models=HORIZON_MODELS or None,
    )


def site_profile_notes(hkey: str, hinfo: dict) -> list[str]:
    """Notes contextuelles honnêtes selon le site."""
    notes: list[str] = []
    if hkey == TRAINED_HOSPITAL:
        notes.append(
            "Probabilité **calibrée** du modèle hôpital complet (consommation + "
            "historique + météo) — site d'entraînement Lacor."
        )
    else:
        notes.append(
            "⚠️ **Score illustratif** : modèle entraîné sur **Lacor** appliqué à un "
            "autre site (profil de consommation emprunté), **non validé** pour ce "
            "site — la généralisation inter-sites n'est pas démontrée (cf. README)."
        )
    rel = hinfo.get("who_reliability")
    stab = hinfo.get("grid_stability", "moyen")
    if rel is not None:
        if rel < 30:
            notes.append(f"Réseau {stab} — fiabilité OMS très basse ({rel:.0f}%)")
        elif rel < 55:
            notes.append(f"Réseau {stab} — fiabilité OMS basse ({rel:.0f}%)")
        elif rel > 90:
            notes.append(f"Réseau {stab} — fiabilité OMS élevée ({rel:.0f}%)")
    if not hinfo.get("has_solar"):
        notes.append("Pas de panneaux solaires — dépendance totale au réseau")
    if not hinfo.get("has_generator"):
        notes.append("Pas de générateur de secours")
    return notes


_summary_path = MODELS_DIR / "training_summary.json"
_winner_name = "RandomForest"
_n_features_train = len(feature_cols)
_train_scope = None
if _summary_path.exists():
    try:
        with open(_summary_path) as _f:
            _summary = json.load(_f)
        _winner_name = _summary.get("winner", _winner_name)
        _train_scope = _summary.get("scope")
    except Exception:
        pass

# Indique la portée d'entraînement : "real" = modèle entraîné uniquement sur
# des coupures réellement observées (métriques honnêtes) ; "all" = inclut des
# coupures synthétiques (métriques globales biaisées).
if _train_scope == "real":
    st.sidebar.caption("🎯 Entraîné sur **données réelles** (coupures observées)")
elif _train_scope == "all":
    st.sidebar.caption("⚠️ Entraîné sur **tous** les sites (coupures synthétiques incluses)")

st.markdown(
    f"""
    <div style='background:linear-gradient(135deg,#1a2530,#2c3e50);
                color:white;padding:24px 32px;border-radius:14px;
                margin-bottom:20px'>
        <div style='display:flex;justify-content:space-between;
                    align-items:center;flex-wrap:wrap;gap:20px'>
            <div>
                <div style='font-size:28px;font-weight:800;line-height:1.1'>
                    ⚡ Prédiction de coupures d'électricité
                </div>
                <div style='font-size:14px;color:#bdc3c7;margin-top:4px'>
                    Hôpitaux · Données réelles 2022 · Modèle expliqué par SHAP
                </div>
            </div>
            <div style='display:flex;gap:28px;flex-wrap:wrap'>
                <div>
                    <div style='font-size:11px;color:#95a5a6;
                                text-transform:uppercase;letter-spacing:1.2px'>
                        Modèle
                    </div>
                    <div style='font-size:22px;font-weight:700;color:#3498db'>
                        {_winner_name}
                    </div>
                </div>
                <div>
                    <div style='font-size:11px;color:#95a5a6;
                                text-transform:uppercase;letter-spacing:1.2px'>
                        Features
                    </div>
                    <div style='font-size:22px;font-weight:700;color:#2ecc71'>
                        {_n_features_train}
                    </div>
                </div>
                <div>
                    <div style='font-size:11px;color:#95a5a6;
                                text-transform:uppercase;letter-spacing:1.2px'>
                        Sources
                    </div>
                    <div style='font-size:22px;font-weight:700;color:#f39c12'>
                        {len(DATA_SOURCES)}
                    </div>
                </div>
                <div>
                    <div style='font-size:11px;color:#95a5a6;
                                text-transform:uppercase;letter-spacing:1.2px'>
                        Hôpitaux
                    </div>
                    <div style='font-size:22px;font-weight:700;color:#e84393'>
                        {len(ALL_HOSPITAL_KEYS)}
                    </div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sélection de l'hôpital ─────────────────────────────────────────

col_select, col_info = st.columns([1, 2])

with col_select:
    hospital_options = REAL_HOSPITAL_KEYS
    hospital_key = st.selectbox(
        "Hôpital",
        options=hospital_options,
        format_func=lambda k: f"{HOSPITAL_DISPLAY[k]['flag']}  {HOSPITAL_DISPLAY[k]['name']} — {HOSPITAL_DISPLAY[k]['location']}",
    )

hospital = HOSPITAL_DISPLAY[hospital_key]

with col_info:
    reliability = hospital.get("who_reliability", 50)
    if reliability < 30:
        rel_color = "#e74c3c"
    elif reliability < 55:
        rel_color = "#f39c12"
    elif reliability < 80:
        rel_color = "#3498db"
    else:
        rel_color = "#2ecc71"

    solar_html = (
        "<span style='color:#f39c12;font-weight:600'>☀️ Solaire</span>"
        if hospital.get("has_solar")
        else "<span style='color:#999'>✕ Pas de solaire</span>"
    )
    gen_html = (
        "<span style='color:#e67e22;font-weight:600'>⚙️ Générateur</span>"
        if hospital.get("has_generator")
        else "<span style='color:#999'>✕ Pas de générateur</span>"
    )

    eric_line = ""
    if hospital.get("data_source") == "eric":
        area = hospital.get("floor_area_m2", 0)
        annual = hospital.get("annual_electricity_kwh", 0)
        eric_line = (
            f"<div style='margin-top:6px;font-size:12px;color:#666'>"
            f"📊 <b>Données ERIC NHS</b> · {area:,} m² · "
            f"{annual / 1e6:.0f} GWh/an · {annual / area:.0f} kWh/m²"
            f"</div>"
        )
    elif hospital.get("data_source") == "nyc_ll84":
        area = hospital.get("floor_area_m2", 0)
        annual = hospital.get("annual_electricity_kwh", 0)
        eric_line = (
            f"<div style='margin-top:6px;font-size:12px;color:#666'>"
            f"📊 <b>NYC Local Law 84</b> · {area:,} m² · "
            f"{annual / 1e6:.0f} GWh/an · {annual / area:.0f} kWh/m²"
            f"</div>"
        )

    st.markdown(
        f"""
        <div style='border:1px solid #e0e0e0;border-radius:10px;
                    padding:14px 18px;background:#fafafa'>
            <div style='font-size:18px;font-weight:700'>
                {hospital['flag']} {hospital['name']}
            </div>
            <div style='font-size:13px;color:#666'>
                {hospital['location']} · {hospital['type']}
            </div>
            <div style='display:flex;gap:24px;margin-top:10px;flex-wrap:wrap;
                        font-size:13px'>
                <div><b>{hospital['beds']}</b> lits</div>
                <div><b>{hospital.get('avg_load_kw', '?'):,}</b> –
                     <b>{hospital.get('max_load_kw', '?'):,}</b> kW</div>
                <div>{solar_html}</div>
                <div>{gen_html}</div>
                <div>Réseau : <b>{hospital.get('grid_stability', '?')}</b></div>
                <div>Fiabilité OMS :
                     <span style='color:{rel_color};font-weight:700'>
                     {reliability:.0f}%</span></div>
            </div>
            {eric_line}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Sources de données spécifiques à cet hôpital ───────────────────
_hospital_sources = detect_hospital_data_sources(hospital_key, hospital)

_sources_html = "".join(
    f"<li style='margin:4px 0'>{s['emoji']}  {s['label']}</li>"
    for s in _hospital_sources
)
st.markdown(
    f"""
    <div style='border:1px solid #e0e0e0;border-radius:10px;
                padding:12px 18px;background:#fafafa;margin-top:10px'>
        <div style='font-size:13px;font-weight:700;color:#2c3e50;
                    margin-bottom:6px'>
            🗂️  Sources utilisées pour {hospital['name']}
            <span style='font-weight:400;color:#888'>
                ({len(_hospital_sources)} sources)
            </span>
        </div>
        <ul style='margin:0;padding-left:20px;font-size:13px;color:#444;
                   columns:2;column-gap:24px'>
            {_sources_html}
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── État réseau local (Electricity Maps, 24 h glissantes) ─────────
_em_df = load_electricitymaps_snapshot(hospital_key)
if _em_df is not None and not _em_df.empty:
    em_24h = _em_df.tail(24).copy()
    _em_last = em_24h.iloc[-1]
    _em_zone = _em_last.get("em_zone", "N/A")
    _em_load = pd.to_numeric(_em_last.get("em_total_load_mw"), errors="coerce")
    _em_carbon = pd.to_numeric(_em_last.get("em_carbon_intensity_gco2_kwh"), errors="coerce")
    _em_ren = pd.to_numeric(_em_last.get("em_renewable_pct"), errors="coerce")
    _em_fossil = pd.to_numeric(_em_last.get("em_fossil_pct"), errors="coerce")
    _em_ts = pd.to_datetime(_em_last.get("datetime"), errors="coerce")

    if "em_total_load_mw" in em_24h.columns:
        em_24h["em_total_load_mw"] = pd.to_numeric(
            em_24h["em_total_load_mw"], errors="coerce",
        )
    em_avg_24h = em_24h["em_total_load_mw"].mean() if "em_total_load_mw" in em_24h else float("nan")
    stress_ratio = (_em_load / em_avg_24h) if (em_avg_24h and not pd.isna(em_avg_24h) and em_avg_24h > 0) else float("nan")

    avg_load_kw_h = float(hospital.get("avg_load_kw", 0) or 0)
    if avg_load_kw_h > 0 and not pd.isna(em_avg_24h) and em_avg_24h > 0:
        em_24h["hospital_load_kw_est"] = avg_load_kw_h * (
            em_24h["em_total_load_mw"] / em_avg_24h
        )
        hospital_now_kw = float(em_24h["hospital_load_kw_est"].iloc[-1])
    else:
        em_24h["hospital_load_kw_est"] = pd.NA
        hospital_now_kw = float("nan")

    st.markdown(
        f"#### ⚡ Réseau local temps réel — Electricity Maps "
        f"<span style='color:#888;font-size:13px;font-weight:400'>"
        f"(zone {_em_zone}, 24 h glissantes)</span>",
        unsafe_allow_html=True,
    )
    em_c1, em_c2, em_c3, em_c4, em_c5 = st.columns(5)
    em_c1.metric("Charge réseau", "N/A" if pd.isna(_em_load) else f"{_em_load:,.0f} MW")
    em_c2.metric(
        "Stress vs moy. 24 h",
        "N/A" if pd.isna(stress_ratio) else f"× {stress_ratio:.2f}",
        delta=None if pd.isna(stress_ratio) else f"{(stress_ratio - 1) * 100:+.1f} %",
    )
    em_c3.metric(
        "Conso hôpital estimée",
        "N/A" if pd.isna(hospital_now_kw) else f"{hospital_now_kw:,.0f} kW",
    )
    em_c4.metric(
        "Intensité carbone",
        "N/A" if pd.isna(_em_carbon) else f"{_em_carbon:,.0f} gCO₂/kWh",
    )
    if not pd.isna(_em_ren) and not pd.isna(_em_fossil):
        em_c5.metric("Mix", f"{_em_ren:.0f}% ren. / {_em_fossil:.0f}% foss.")
    else:
        em_c5.metric("Mix", "N/A")

    if "em_total_load_mw" in em_24h.columns and em_24h["em_total_load_mw"].notna().any():
        em_chart_l, em_chart_r = st.columns(2)
        with em_chart_l:
            fig_grid = go.Figure()
            fig_grid.add_trace(go.Scatter(
                x=em_24h["datetime"], y=em_24h["em_total_load_mw"],
                mode="lines+markers", name="Charge réseau (MW)",
                line=dict(color="#f1c40f", width=2),
            ))
            fig_grid.update_layout(
                title="Charge réseau zone (24 h)",
                xaxis_title="Heure", yaxis_title="MW",
                height=260, margin=dict(l=40, r=20, t=40, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_grid, width="stretch")
        with em_chart_r:
            if em_24h["hospital_load_kw_est"].notna().any():
                fig_hosp = go.Figure()
                fig_hosp.add_trace(go.Scatter(
                    x=em_24h["datetime"], y=em_24h["hospital_load_kw_est"],
                    mode="lines+markers",
                    name="Conso hôpital estimée (kW)",
                    line=dict(color="#e84393", width=2),
                ))
                fig_hosp.update_layout(
                    title=f"Conso {hospital['name']} estimée (24 h)",
                    xaxis_title="Heure", yaxis_title="kW",
                    height=260, margin=dict(l=40, r=20, t=40, b=40),
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_hosp, width="stretch")
            else:
                st.caption(
                    "Estimation indisponible (avg_load_kw inconnu pour cet hôpital)."
                )

    if not pd.isna(_em_ts):
        st.caption(
            f"Dernière mesure Electricity Maps : {_em_ts.strftime('%Y-%m-%d %H:%M UTC')}"
            " — estimation = avg_load_kw × (charge_réseau_now / charge_réseau_moy_24h)."
        )
else:
    _em_path = electricitymaps_snapshot_path(hospital_key)
    _ds = hospital.get("data_source", "")
    if _ds in ("eric", "nyc_ll84"):
        st.info(
            "Contexte réseau **Electricity Maps** : aucun snapshot local pour cet "
            "hôpital (données ERIC / NYC LL84 = consommation + météo historiques). "
            "Pour afficher la charge du réseau régional (NY-ISO ou National Grid), "
            "définis `ELECTRICITY_MAPS_TOKEN` puis lance "
            "`python -m src.data.ingest_electricitymaps`.",
            icon="⚡",
        )
    elif not _em_path.exists():
        st.info(
            "Electricity Maps : fichier snapshot absent. "
            "Avec un token API (`ELECTRICITY_MAPS_TOKEN`), lance "
            "`python -m src.data.ingest_electricitymaps`.",
            icon="⚡",
        )
    else:
        st.info(
            "Electricity Maps : fichier présent mais vide ou illisible.",
            icon="⚡",
        )

# ── Bandeau « Sources & facteurs du modèle » ───────────────────────

with st.expander("📊  Sources de données & facteurs du modèle", expanded=False):
    n_model = sum(source_role_model_pipeline(s) for s in DATA_SOURCES)
    n_context = sum(source_role_context_app(s) for s in DATA_SOURCES)
    st.markdown(
        f"**Modèle principal** : **{_winner_name}** sur **Lacor 2022** "
        f"(**{_n_features_train} features**) — conso, météo, charge, **historique "
        "des coupures**. Pour « Prochaine coupure », des modèles complémentaires "
        "1/3/6 h (`nowcast_horizons/`) apprennent explicitement "
        "**coupure dans les H prochaines heures** avec **les mêmes features**.\n\n"
        f"**Pipeline :** {len(DATA_SOURCES)} sources actives "
        f"({n_model} alimentent le modèle, {n_context} en contexte temps réel seulement)."
    )

    st.markdown("#### Facteurs les plus influents (SHAP global, modèle Lacor)")
    panel_left, panel_right = st.columns([3, 2])
    with panel_left:
        show_top_factors_panel(top_n=12)
    with panel_right:
        show_category_breakdown()

    st.markdown("#### Sources de données du pipeline")
    show_data_sources_panel()

st.divider()

# ── Avertissement si les signaux externes sont neutralisés ────────
if hospital.get("data_source") == "eric":
    st.info(
        f"**Hôpital NHS — données réelles ERIC 2022-23** · "
        f"La consommation horaire de {hospital['name']} est désagrégée à "
        "partir des relevés annuels officiels NHS Estates Returns "
        "Information Collection. La météo locale est récupérée via "
        "Open-Meteo (latitude/longitude réelles de l'hôpital).",
        icon="🇬🇧",
    )
elif hospital.get("data_source") == "nyc_ll84":
    st.info(
        f"**Hôpital NYC — données réelles Local Law 84** · "
        f"La consommation annuelle de {hospital['name']} est issue du "
        "registre obligatoire NYC LL84 (data.cityofnewyork.us, dataset "
        "5zyy-y8am, ~120 hôpitaux NYC publiés). Désagrégation horaire avec "
        "pic estival (climatisation Con Edison) et météo Open-Meteo "
        "locale.",
        icon="🇺🇸",
    )
elif hospital.get("data_source") == "africa_grid":
    st.info(
        f"**Hôpital Afrique — profil de consommation estimé (temps réel)** · "
        f"La charge de {hospital['name']} est estimée à partir d'un profil "
        "hospitalier de référence (Lacor) mis à l'échelle selon la taille "
        "du site, puis pilotée par la météo locale et le signal réseau "
        "Electricity Maps (zone locale, charge réseau, mix, carbone). "
        "Ce ne sont pas des compteurs internes publiés par l'hôpital.",
        icon="🌍",
    )

# ── Chargement des données spécifiques à l'hôpital ────────────────────
try:
    df = load_hospital_data(hospital_key, hospital)
except Exception as e:
    st.error(
        f"**Impossible de charger les données pour {hospital['name']}** : {e}\n\n"
        "Vérifiez que le pipeline a été exécuté et que les fichiers de données existent."
    )
    st.stop()

if df is None or df.empty:
    st.error(
        f"**Aucune donnée disponible pour {hospital['name']}.**\n\n"
        "Exécutez le pipeline pour générer les données :\n"
        "```bash\npython run_pipeline.py\n```"
    )
    st.stop()

for col in feature_cols:
    if col not in df.columns:
        df[col] = 0.0

# ── Onglets ──────────────────────────────────────────────────────────

tab_next, tab_predict, tab_forecast, tab_simulate = st.tabs([
    "🚨  Prochaine coupure (24 h)",
    "🔍  Analyse historique (par période)",
    "🔮  Prévisions J+7",
    "🎛️  Simulation manuelle",
])


# ═══════════════════════════════════════════════════════════════════
# ONGLET 0 : PROCHAINE COUPURE (même modèle nowcast Lacor)
# ═══════════════════════════════════════════════════════════════════

with tab_next:
    st.markdown(
        "<p style='color:#888'>À partir des <b>dernières 24 h</b> de "
        "<b>consommation, charge, météo</b> et de l'<b>historique des coupures</b>, "
        "le <b>même modèle Lacor</b> que l'analyse historique estime la probabilité "
        "d'une coupure dans les <b>1 / 3 / 6 prochaines heures</b> (modèles horizons "
        "entraînés sur cette cible ; repli = nowcast horaire si absents).</p>",
        unsafe_allow_html=True,
    )

    if model is None:
        st.warning(
            "**Modèle absent.** Lance d'abord "
            "`python run_pipeline.py`."
        )
    elif not HORIZON_MODELS:
        st.info(
            "Modèles **1/3/6 h** non trouvés — repli sur le nowcast horaire. "
            "Pour les entraîner : `python run_pipeline.py` (étape horizons) ou "
            "`python -m src.models.train_horizons`."
        )

    if model is not None:
        # Défaut « temps réel » pour les hôpitaux à réseau live (africa_grid) :
        # leur consommation 2022 est un clone, le signal pertinent est le
        # réseau Electricity Maps en direct. Clé par hôpital pour que le défaut
        # s'applique à chaque changement de site.
        _is_grid_live = hospital.get("data_source") in REALTIME_GRID_SOURCES
        # Le mode « Temps réel » n'a de sens que pour les hôpitaux raccordés à un
        # réseau interrogeable en direct (Electricity Maps) : africa_grid + Lacor.
        # Pour NHS/NYC (réseaux très stables, pas de zone EM exploitée), on ne
        # propose QUE l'historique — inutile d'offrir une option « indisponible ».
        _can_realtime = _is_grid_live or hospital_key == TRAINED_HOSPITAL
        if _can_realtime:
            src_mode = st.radio(
                "Source des dernières 24 h",
                ["🕘 Historique 2022", "📡 Temps réel (Electricity Maps)"],
                index=1 if _is_grid_live else 0,
                horizontal=True,
                key=f"next_src_mode_{hospital_key}",
            )
            realtime = src_mode.startswith("📡")
        else:
            realtime = False

        # ── Mode TEMPS RÉEL : Electricity Maps + météo récente ─────────
        if realtime:
            st.caption(
                "Charge du **réseau régional** (Electricity Maps, live) → consommation "
                "estimée (normalisée à l'échelle Lacor) + **météo récente** Open-Meteo. "
                "Résultat = **risque régional indicatif** (modèle entraîné sur Lacor)."
            )
            # Bouton « Rafraîchir » : force un nouvel appel aux APIs avant
            # expiration du cache (Electricity Maps + délestage EskomSePush).
            _refresh = st.button("🔄 Rafraîchir", key="btn_rt_refresh", help="Force un nouvel appel Electricity Maps + EskomSePush + météo")
            if _refresh:
                load_realtime_forecast.clear()
                load_loadshedding.clear()

            # ── Délestage programmé (EskomSePush) — sites sud-africains ──
            # Cause DIRECTE des coupures en Afrique du Sud. Affiché comme
            # CONTEXTE temps réel (non testable sur Lacor → hors modèle).
            if loadshedding_supported(hospital_key):
                ls = load_loadshedding(hospital_key)
                if ls is None:
                    st.info(
                        "⚡ **Délestage programmé (EskomSePush)** — cause directe des "
                        "coupures en Afrique du Sud. Pour l'activer, définis la variable "
                        "d'environnement `ESKOM_SEPUSH_TOKEN` "
                        "([token gratuit](https://eskomsepush.gumroad.com/l/api))."
                    )
                else:
                    sev_icon = {"none": "🟢", "low": "🟠", "high": "🔴"}.get(ls["severity"], "⚪")
                    cstage, cnext = st.columns([1, 2])
                    cstage.metric(
                        f"{sev_icon} Délestage · {ls['name']}",
                        ls["stage_label"],
                    )
                    if ls["next"]:
                        lignes = "\n".join(
                            f"- Stade **{n['stage']}** à partir de "
                            f"`{(n.get('start') or '?')[:16].replace('T', ' ')}`"
                            for n in ls["next"][:3]
                        )
                        cnext.markdown("**Prochains changements programmés**\n" + lignes)
                    else:
                        cnext.caption("Pas de changement de stade programmé annoncé.")
                    st.caption(
                        "Source : EskomSePush (délestage Eskom/municipal). **Contexte "
                        "causal temps réel** — pas une entrée du modèle (entraîné sur "
                        "Lacor, Ouganda)."
                    )
                    st.divider()

            # Récupération AUTOMATIQUE à l'ouverture (le cache 15 min de
            # load_realtime_forecast protège l'API ; le bouton ci-dessus force
            # une mise à jour avant expiration du cache).
            with st.spinner("Récupération Electricity Maps + météo…"):
                rt = load_realtime_forecast(
                    hospital_key,
                    tuple(feature_cols),
                    _mtime,
                    _horizon_models_mtime(),
                )

            if rt is None:
                st.warning(
                    "**Données réseau temps réel indisponibles** pour cette zone "
                    "(token Electricity Maps absent, zone hors plan, ou charge réseau "
                    "non fournie par l'API). Bascule sur « Historique 2022 »."
                )
            else:
                probs = rt["probs"]
                ui_step("Étape 2", f"Risque de coupure à venir — zone réseau {rt.get('zone', '?')}")
                horizon_labels = {1: "≤ 1 h", 3: "≤ 3 h", 6: "≤ 6 h"}
                cols = st.columns(len(probs))
                for col, (h, p) in zip(cols, probs.items()):
                    tone = "🔴" if p > 0.5 else ("🟠" if p > 0.3 else "🟢")
                    col.metric(f"{tone} Coupure {horizon_labels.get(h, f'{h}h')}", f"{p:.0%}")
                peak_h = max(probs, key=probs.get)
                peak_p = probs[peak_h]
                show_risk_result(peak_p, float(peak_h), round(1.0 + peak_p * 4.0, 1) if peak_p > 0.5 else 0.5)
                st.info(
                    f"📡 **Temps réel** — zone **{rt.get('zone', '?')}** (Electricity Maps). "
                    "Conso estimée depuis la charge réseau + météo ; horizons 1/3/6 h "
                    "via le **modèle Lacor** et la météo prévue Open-Meteo. "
                    "**Indicatif** hors Lacor."
                )
                st.divider()
                ui_step("Étape 3", "Fenêtre 24 h temps réel (charge réseau → conso estimée + météo)")
                wv = rt["window"].copy()
                wv["datetime"] = pd.to_datetime(wv["datetime"])
                fig_w = go.Figure()
                fig_w.add_trace(go.Scatter(
                    x=wv["datetime"], y=wv["total_load_kw"],
                    mode="lines", name="Conso estimée (kW, éch. Lacor)",
                    line=dict(color="#3498db", width=2), yaxis="y1",
                ))
                if "temperature_2m" in wv.columns and wv["temperature_2m"].notna().any():
                    fig_w.add_trace(go.Scatter(
                        x=wv["datetime"], y=wv["temperature_2m"],
                        mode="lines", name="Température (°C)",
                        line=dict(color="#e67e22", width=2, dash="dot"), yaxis="y2",
                    ))
                fig_w.update_layout(
                    height=300, margin=dict(l=40, r=40, t=20, b=40),
                    yaxis=dict(title="Conso estimée (kW)", side="left"),
                    yaxis2=dict(title="Température (°C)", side="right", overlaying="y", showgrid=False),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_w, width="stretch")

                # ── Contexte orage / convection (Open-Meteo, fenêtre live) ──
                # Affiché comme CONTEXTE seulement : le test walk-forward sur
                # Lacor montre que pluie/rafales DÉGRADENT le modèle (F1 −0.01
                # à −0.05). On informe sans biaiser la prévision.
                _w = rt["window"]
                _prec = pd.to_numeric(_w.get("precipitation"), errors="coerce") if "precipitation" in _w.columns else None
                _gust = pd.to_numeric(_w.get("wind_gusts_10m"), errors="coerce") if "wind_gusts_10m" in _w.columns else None
                _cape = pd.to_numeric(_w.get("cape"), errors="coerce") if "cape" in _w.columns else None
                if _prec is not None or _gust is not None:
                    with st.expander("🌩️ Contexte orage / convection (météo)", expanded=False):
                        sc = st.columns(3)
                        if _prec is not None:
                            sc[0].metric("Pluie max (fenêtre)", f"{_prec.max():.1f} mm/h")
                        if _gust is not None:
                            sc[1].metric("Rafales max", f"{_gust.max():.0f} km/h")
                        if _cape is not None and _cape.max() > 0:
                            cape_max = _cape.max()
                            conv = "élevé" if cape_max > 1000 else ("modéré" if cape_max > 300 else "faible")
                            sc[2].metric("CAPE max (convection)", f"{cape_max:.0f} J/kg", help=f"Potentiel orageux {conv}")
                        else:
                            sc[2].metric("CAPE max (convection)", "n/d", help="Non fourni par l'API pour cette zone")
                        st.caption(
                            "ℹ️ **Contexte uniquement.** Testé en walk-forward sur Lacor : "
                            "ajouter pluie/rafales **dégrade** la prévision (F1 −0.01 à −0.05). "
                            "L'orage n'est pas un précurseur fiable des coupures de Lacor "
                            "(CAPE indisponible en archive, aucun code orage en 2022)."
                        )

        # ── Mode HISTORIQUE 2022 (rejoue les données du site) ──────────
        else:
            nx_dt = pd.to_datetime(df["datetime"])
            nx_min, nx_max = nx_dt.min(), nx_dt.max()

            ui_step("Étape 1", "Choisir le moment d'analyse (« maintenant »)")
            st.caption(
                f"Données disponibles : **{nx_min.strftime('%Y-%m-%d %Hh')}** → "
                f"**{nx_max.strftime('%Y-%m-%d %Hh')}**. Le modèle lit les 24 h qui "
                "précèdent ce point."
            )
            ref_day = st.date_input(
                "Date de référence",
                value=nx_max.date(),
                min_value=nx_min.date(),
                max_value=nx_max.date(),
                key="next_ref_day",
            )
            ref_hour = st.slider("Heure de référence", 0, 23, int(nx_max.hour), key="next_ref_hour")
            ref_ts = pd.Timestamp(ref_day) + pd.Timedelta(hours=ref_hour)

            window = df[(nx_dt > ref_ts - pd.Timedelta(hours=24)) & (nx_dt <= ref_ts)].copy()
            if window.empty:
                window = df[nx_dt <= ref_ts].tail(24).copy()

            if len(window) < 2:
                st.info("Pas assez de données avant ce point pour constituer une fenêtre.")
            else:
                ref_ts_eff = pd.to_datetime(window["datetime"]).max()
                probs = forecast_next_outage(df, ref_ts_eff)
                ui_step("Étape 2", "Risque de coupure à venir")

                if not probs:
                    st.info(
                        "Pas d'heures **après** ce point dans les données — choisis une "
                        "date/heure plus tôt (il faut au moins 1 h future pour estimer le risque)."
                    )
                else:
                    horizon_labels = {1: "≤ 1 h", 3: "≤ 3 h", 6: "≤ 6 h"}
                    cols = st.columns(len(probs))
                    for col, (h, p) in zip(cols, probs.items()):
                        if p > 0.5:
                            tone = "🔴"
                        elif p > 0.3:
                            tone = "🟠"
                        else:
                            tone = "🟢"
                        col.metric(f"{tone} Coupure {horizon_labels.get(h, f'{h}h')}", f"{p:.0%}")

                    peak_h = max(probs, key=probs.get)
                    peak_p = probs[peak_h]
                    show_risk_result(
                        peak_p, float(peak_h),
                        round(1.0 + peak_p * 4.0, 1) if peak_p > 0.5 else 0.5,
                    )

                    st.info(
                        "**Score illustratif** : modèle Lacor appliqué à un autre site."
                        if hospital_key != TRAINED_HOSPITAL else
                        "**Modèle Lacor** — horizons 1/3/6 h entraînés sur "
                        "« coupure dans les H prochaines heures » (mêmes features)."
                        if HORIZON_MODELS else
                        "**Modèle Lacor** (repli nowcast horaire — lance "
                        "`python -m src.models.train_horizons`)."
                    )

                st.divider()
                ui_step("Étape 3", "Fenêtre 24 h analysée (conso + météo)")
                wv = window.copy()
                wv["datetime"] = pd.to_datetime(wv["datetime"])
                fig_w = go.Figure()
                fig_w.add_trace(go.Scatter(
                    x=wv["datetime"], y=wv["total_load_kw"],
                    mode="lines", name="Charge totale (kW)",
                    line=dict(color="#3498db", width=2), yaxis="y1",
                ))
                if "temperature_2m" in wv.columns:
                    fig_w.add_trace(go.Scatter(
                        x=wv["datetime"], y=wv["temperature_2m"],
                        mode="lines", name="Température (°C)",
                        line=dict(color="#e67e22", width=2, dash="dot"), yaxis="y2",
                    ))
                fig_w.update_layout(
                    height=300, margin=dict(l=40, r=40, t=20, b=40),
                    yaxis=dict(title="Charge (kW)", side="left"),
                    yaxis2=dict(title="Température (°C)", side="right", overlaying="y", showgrid=False),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_w, width="stretch")


# ═══════════════════════════════════════════════════════════════════
# ONGLET 1 : PRÉDICTION HISTORIQUE
# ═══════════════════════════════════════════════════════════════════

with tab_predict:
    if hospital.get("data_source") == "eric":
        data_label = "données ERIC NHS (historique)"
    elif hospital.get("data_source") == "nyc_ll84":
        data_label = "données NYC LL84 (historique)"
    elif hospital.get("data_source") == "africa_grid":
        data_label = "profil estimé (quasi temps réel)"
    else:
        data_label = "données historiques"

    df_dt = pd.to_datetime(df["datetime"])
    df_min_d = df_dt.min().date()
    df_max_d = df_dt.max().date()
    default_end_d = df_max_d
    default_start_d = max(df_min_d, default_end_d - timedelta(days=2))

    st.markdown(
        f"<p style='color:#888'>Estime le risque de coupure pour <b>{hospital['name']}</b> "
        f"sur la <b>période de votre choix</b> ({data_label}). "
        f"Données disponibles : <b>{df_min_d.isoformat()}</b> → <b>{df_max_d.isoformat()}</b>. "
        f"Seuil d'alerte principal : <b>50%</b>.</p>",
        unsafe_allow_html=True,
    )
    ui_step("Étape 1", "Choisir la période d'analyse")

    PRESETS = {
        "Personnalisé": None,
        "Dernières 72 h disponibles": (
            max(df_min_d, df_max_d - timedelta(days=2)), df_max_d,
        ),
        "Janvier 2022": (
            max(df_min_d, date(2022, 1, 1)), min(df_max_d, date(2022, 1, 31)),
        ),
        "Saison sèche (déc-fév)": (
            max(df_min_d, date(2022, 1, 1)), min(df_max_d, date(2022, 2, 28)),
        ),
        "Saison des pluies (mars-mai)": (
            max(df_min_d, date(2022, 3, 1)), min(df_max_d, date(2022, 5, 31)),
        ),
        "Été (juin-août)": (
            max(df_min_d, date(2022, 6, 1)), min(df_max_d, date(2022, 8, 31)),
        ),
        "Automne (sept-nov)": (
            max(df_min_d, date(2022, 9, 1)), min(df_max_d, date(2022, 11, 30)),
        ),
        "Toute l'année 2022": (
            max(df_min_d, date(2022, 1, 1)), min(df_max_d, date(2022, 12, 31)),
        ),
    }

    col_preset, col_dates = st.columns([2, 3])
    with col_preset:
        preset_name = st.selectbox(
            "Période prédéfinie",
            options=list(PRESETS.keys()),
            index=0,
            key="predict_preset",
            help="Choisis un raccourci ou « Personnalisé » pour fixer toi-même les dates.",
        )

    if preset_name != "Personnalisé" and PRESETS[preset_name] is not None:
        preset_start, preset_end = PRESETS[preset_name]
    else:
        preset_start, preset_end = default_start_d, default_end_d

    with col_dates:
        date_range = st.date_input(
            "Période d'analyse",
            value=(preset_start, preset_end),
            min_value=df_min_d,
            max_value=df_max_d,
            key=f"predict_date_range_{preset_name}",
        )

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date_pick, end_date_pick = date_range
    elif isinstance(date_range, (list, tuple)) and len(date_range) == 1:
        start_date_pick = end_date_pick = date_range[0]
    else:
        start_date_pick = end_date_pick = date_range

    if start_date_pick > end_date_pick:
        start_date_pick, end_date_pick = end_date_pick, start_date_pick

    start_dt_sel = pd.Timestamp(start_date_pick)
    end_dt_sel = pd.Timestamp(end_date_pick) + pd.Timedelta(hours=23, minutes=59)

    selected_df = df[(df_dt >= start_dt_sel) & (df_dt <= end_dt_sel)].copy()
    n_hours = len(selected_df)

    if n_hours <= 1:
        window_label = f"{n_hours} h"
    elif n_hours <= 168:
        window_label = f"{n_hours} h"
    else:
        window_label = f"{n_hours / 24:.0f} j"

    st.caption(
        f"📅 Fenêtre : **{start_date_pick.isoformat()}** → "
        f"**{end_date_pick.isoformat()}** · **{n_hours} heures** de données "
        f"({window_label})."
    )

    analysis_id = f"{hospital_key}:{start_date_pick.isoformat()}:{end_date_pick.isoformat()}"
    analysis_state_key = "predict_analysis_result"
    run_predict = st.button(
        f"Analyser le risque sur la période ({window_label})",
        type="primary",
        width="stretch",
        key="btn_predict",
        disabled=n_hours < 2,
    )
    saved_predict = st.session_state.get(analysis_state_key)
    has_saved_predict = (
        isinstance(saved_predict, dict)
        and saved_predict.get("analysis_id") == analysis_id
    )

    if run_predict or has_saved_predict:
        if run_predict:
            try:
                with st.spinner(f"Analyse de {n_hours} heures en cours…"):
                    recent = selected_df.copy()
                    if len(recent) < 2:
                        st.warning("Pas assez de données pour l'analyse (minimum 2 heures requises).")
                        st.stop()
                    # Prédiction via le modèle hôpital (entraîné sur Lacor).
                    # Pour un site ≠ Lacor : score illustratif (cf. site_profile_notes).
                    recent["outage_probability"] = site_predict_proba(recent)

                    high_risk = recent[recent["outage_probability"] > 0.5]
                    if high_risk.empty:
                        max_idx = recent["outage_probability"].idxmax()
                        max_proba = recent.loc[max_idx, "outage_probability"]
                        hours_away = abs((recent.loc[max_idx, "datetime"] - recent["datetime"].iloc[-1]).total_seconds() / 3600)
                    else:
                        max_proba = high_risk.iloc[0]["outage_probability"]
                        hours_away = max(0, (high_risk.iloc[0]["datetime"] - recent["datetime"].iloc[-1]).total_seconds() / 3600)

                    h_notes = site_profile_notes(hospital_key, hospital)
                    duration = round(1.0 + max_proba * 4.0, 1) if max_proba > 0.5 else 0.5
                    last_row = ensure_numeric_feature_frame(recent.tail(1), feature_cols).iloc[-1]
                    factors = get_top_factors(model, feature_cols, last_row)
                    last_row_df = pd.DataFrame([last_row])
                    shap_sv, shap_ev = compute_shap_local(shap_explainer, last_row_df, feature_cols)

                    st.session_state[analysis_state_key] = {
                        "analysis_id": analysis_id,
                        "recent": recent,
                        "max_proba": max_proba,
                        "hours_away": hours_away,
                        "duration": duration,
                        "h_notes": h_notes,
                        "factors": factors,
                        "shap_sv": shap_sv,
                        "shap_ev": shap_ev,
                    }
            except Exception as e:
                st.error(f"**Erreur lors de l'analyse** : {e}")
                st.stop()
        else:
            recent = saved_predict["recent"]
            max_proba = saved_predict["max_proba"]
            hours_away = saved_predict["hours_away"]
            duration = saved_predict["duration"]
            h_notes = saved_predict["h_notes"]
            factors = saved_predict["factors"]
            shap_sv = saved_predict["shap_sv"]
            shap_ev = saved_predict["shap_ev"]

        ui_step("Étape 2", "Résumé du risque estimé")
        show_risk_result(max_proba, hours_away, duration)
        if h_notes:
            st.info("**Profil de l'hôpital** :\n" + "\n".join(f"- {n}" for n in h_notes))
        st.divider()

        ui_step("Étape 3", "Pourquoi ce niveau de risque ?")
        col_factors, col_chart = st.columns([2, 3])

        with col_factors:
            st.subheader("Facteurs explicatifs")
            if shap_sv is not None:
                show_shap_waterfall(shap_sv, shap_ev, feature_cols, title="Facteurs explicatifs (SHAP)")
            else:
                show_factors(factors)

        with col_chart:
            st.subheader(f"Évolution du risque ({window_label})")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=recent["datetime"], y=recent["outage_probability"],
                mode="lines", fill="tozeroy",
                line=dict(color="#e74c3c", width=2),
                fillcolor="rgba(231, 76, 60, 0.15)",
                name="Probabilité",
            ))
            fig.add_hline(y=0.5, line_dash="dash", line_color="#f39c12",
                          annotation_text="Seuil d'alerte (50%)", annotation_position="top left")
            fig.update_layout(
                yaxis=dict(title="Probabilité", range=[0, 1], tickformat=".0%"),
                xaxis=dict(title=""), height=350,
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, width="stretch")

        st.divider()

        ui_step("Étape 4", "Contexte de consommation et statistiques")
        st.subheader(f"Consommation observée — {hospital['name']} ({window_label})")
        consumption_view_mode = st.radio(
            "Affichage du signal",
            options=["Auto", "Horaire", "Journalier", "Hebdomadaire"],
            horizontal=True,
            key=f"consumption_view_mode_{start_date_pick}_{end_date_pick}",
            help="Auto : vue horaire si ≤ 2 semaines, journalière jusqu’à ~11 semaines, "
            "puis hebdomadaire au-delà (pour éviter les courbes illisibles).",
        )

        if consumption_view_mode == "Auto":
            # Seuils resserrés : au-delà de ~2 semaines l’horaire devient trop dense.
            if n_hours > 24 * 80:
                resolved_view = "Hebdomadaire"
            elif n_hours > 24 * 14:
                resolved_view = "Journalier"
            else:
                resolved_view = "Horaire"
        else:
            resolved_view = consumption_view_mode

        plot_df = recent.copy()
        plot_df["datetime"] = pd.to_datetime(plot_df["datetime"])
        if resolved_view in {"Journalier", "Hebdomadaire"}:
            rule = "D" if resolved_view == "Journalier" else "W-SUN"
            agg_spec = {"total_load_kw": "mean", "is_outage": "sum"}
            if "solar_pv_kw" in plot_df.columns:
                agg_spec["solar_pv_kw"] = "mean"
            if "generators_kw" in plot_df.columns:
                agg_spec["generators_kw"] = "mean"
            plot_df = (
                plot_df
                .set_index("datetime")
                .resample(rule)
                .agg(agg_spec)
                .reset_index()
            )

        outage_col = "is_outage"
        outage_mask = plot_df[outage_col] > 0 if outage_col in plot_df.columns else pd.Series(False, index=plot_df.index)
        st.caption(
            f"Vue utilisée : **{resolved_view}** ({len(plot_df)} points affichés). "
            "Astuce : choisis **Journalier** ou **Hebdomadaire** si la courbe bleue "
            "forme encore un « ruban » illisible."
        )

        show_detail_traces = st.checkbox(
            "Afficher solaire & générateur (détail)",
            value=(resolved_view == "Horaire"),
            key=f"consumption_detail_{start_date_pick}_{end_date_pick}",
            help="Sur les longues périodes, masquer ces courbes met en avant la charge totale.",
        )

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=plot_df["datetime"], y=plot_df["total_load_kw"],
            mode="lines", name="Charge totale", line=dict(color="#3498db", width=2),
        ))
        if show_detail_traces and "solar_pv_kw" in plot_df.columns:
            fig2.add_trace(go.Scatter(
                x=plot_df["datetime"], y=plot_df["solar_pv_kw"],
                mode="lines", name="Solaire PV", line=dict(color="#f1c40f", width=2),
            ))
        if show_detail_traces and "generators_kw" in plot_df.columns:
            fig2.add_trace(go.Scatter(
                x=plot_df["datetime"], y=plot_df["generators_kw"],
                mode="lines", name="Générateur", line=dict(color="#e67e22", width=2),
            ))
        outages = plot_df[outage_mask]
        if not outages.empty:
            fig2.add_trace(go.Scatter(
                x=outages["datetime"], y=outages["total_load_kw"],
                mode="markers", marker=dict(color="#e74c3c", size=10, symbol="x"),
                name="Coupures",
            ))
        fig2.update_layout(
            yaxis=dict(title="Puissance (kW)"),
            xaxis=dict(title="", rangeslider=dict(visible=True)),
            height=300, margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig2, width="stretch")

        if "is_outage" in recent.columns and n_hours >= 48:
            wo = (
                recent.assign(datetime=pd.to_datetime(recent["datetime"]))
                .set_index("datetime")["is_outage"]
                .resample("W-SUN")
                .sum()
                .reset_index()
            )
            wo = wo[wo["is_outage"] > 0]
            if not wo.empty:
                st.markdown("**Résumé des coupures (heures / semaine)**")
                fig_out = go.Figure(
                    go.Bar(
                        x=wo["datetime"],
                        y=wo["is_outage"],
                        marker_color="#e74c3c",
                        name="Heures en coupure",
                    )
                )
                fig_out.update_layout(
                    yaxis=dict(title="Heures"),
                    xaxis=dict(title=""),
                    height=240,
                    margin=dict(l=40, r=20, t=20, b=40),
                    showlegend=False,
                )
                st.plotly_chart(fig_out, width="stretch")

        st.subheader(f"Statistiques clés — {hospital['name']} (période analysée)")
        s1, s2, s3, s4 = st.columns(4)
        n_outages = int(recent["is_outage"].sum()) if "is_outage" in recent.columns else 0
        pct_outage = (
            100 * recent["is_outage"].mean()
            if "is_outage" in recent.columns and len(recent) > 0
            else 0
        )
        outage_label = "Coupures (2022)" if hospital.get("data_source") != "africa_grid" else "Coupures estimées (fenêtre affichée)"
        s1.metric(outage_label, f"{n_outages}")
        s2.metric("Taux de coupure", f"{pct_outage:.2f}%")
        s3.metric("Charge moyenne", f"{recent['total_load_kw'].mean():.0f} kW")
        s4.metric("Charge max", f"{recent['total_load_kw'].max():.0f} kW")


# ═══════════════════════════════════════════════════════════════════
# ONGLET 2 : PRÉVISIONS J+7 (à partir d'Open-Meteo Forecast)
# ═══════════════════════════════════════════════════════════════════

with tab_forecast:
    st.markdown(
        "<p style='color:#888'>Projette le risque de coupure sur les <b>7 prochains jours</b> "
        "à partir des prévisions météo et du profil énergétique de l'hôpital. "
        "Seuil d'alerte : <b>50%</b>, seuil critique : <b>70%</b>.</p>",
        unsafe_allow_html=True,
    )
    ui_step("Étape 1", "Vérifier les prévisions météo disponibles")

    forecast_df = load_meteo_forecast(hospital_key, _forecast_file_mtime(hospital_key))

    if forecast_df is None or forecast_df.empty:
        st.warning(
            "**Pas de prévisions météo disponibles pour cet hôpital.**\n\n"
            "Exécutez la récupération des prévisions :\n"
            "```bash\npython -m src.data.ingest_openmeteo_forecast\n```"
        )
    else:
        fetched_at = forecast_df.get("fetched_at", pd.Series([None])).iloc[0]
        info_line = f"Prévisions reçues : **{fetched_at}**  " if fetched_at else ""
        horizon = f"{(forecast_df['datetime'].max() - forecast_df['datetime'].min()).total_seconds() / 3600:.0f} h"
        st.caption(f"{info_line}· Horizon : **{horizon}** · Source : Open-Meteo Forecast API")

        if st.button("Projeter le risque (J+7)", type="primary", width="stretch", key="btn_forecast"):
            try:
                with st.spinner("Projection horaire du risque sur 7 jours…"):
                    preds = build_forecast_predictions(
                        hist_df=df,
                        forecast_df=forecast_df,
                        feature_cols=feature_cols,
                        hospital_key=hospital_key,
                    )
            except Exception as e:
                st.error(f"**Erreur lors de la prévision** : {e}")
                st.stop()

            if preds.empty:
                st.warning("Aucune prédiction n'a pu être générée.")
                st.stop()

            max_idx = preds["outage_probability"].idxmax()
            max_proba = float(preds.loc[max_idx, "outage_probability"])
            max_time = preds.loc[max_idx, "datetime"]
            hours_away = max(0.0, (max_time - pd.Timestamp.now(tz=max_time.tz)).total_seconds() / 3600) \
                if max_time.tz is not None else \
                max(0.0, (max_time - pd.Timestamp.now()).total_seconds() / 3600)

            duration = round(1.0 + max_proba * 4.0, 1) if max_proba > 0.5 else 0.5

            ui_step("Étape 2", "Résumé du pic de risque prévisionnel")
            show_risk_result(max_proba, hours_away, duration)

            # ── Bandeau : horaire du pic ─────────────────────────
            max_time_display = pd.to_datetime(max_time).strftime("%a %d %b %Y · %Hh")
            st.info(
                f"**Pic de risque prévu** : {max_time_display}  "
                f"· Dans **{hours_away:.0f} h** · Probabilité **{max_proba:.0%}**"
            )

            st.divider()

            # ── Timeline principale ──────────────────────────────
            ui_step("Étape 3", "Lecture de la trajectoire horaire et du contexte météo")
            st.subheader("Trajectoire du risque — 7 jours")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=preds["datetime"], y=preds["outage_probability"],
                mode="lines", fill="tozeroy",
                line=dict(color="#e74c3c", width=2),
                fillcolor="rgba(231, 76, 60, 0.15)",
                name="Probabilité",
                hovertemplate="%{x|%a %d %b %Hh}<br>Risque : %{y:.0%}<extra></extra>",
            ))
            fig.add_hline(y=0.5, line_dash="dash", line_color="#f39c12",
                          annotation_text="Seuil d'alerte (50%)", annotation_position="top left")
            fig.add_hline(y=0.7, line_dash="dot", line_color="#e74c3c",
                          annotation_text="Seuil critique (70%)", annotation_position="top left")
            fig.update_layout(
                yaxis=dict(title="Probabilité", range=[0, 1], tickformat=".0%"),
                xaxis=dict(title=""), height=350,
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, width="stretch")

            # ── Météo prévue ─────────────────────────────────────
            st.subheader("Contexte météo prévu")
            fig_meteo = go.Figure()
            fig_meteo.add_trace(go.Scatter(
                x=preds["datetime"], y=preds["temperature_2m"],
                mode="lines", name="Température (°C)",
                line=dict(color="#e67e22", width=2), yaxis="y1",
            ))
            fig_meteo.add_trace(go.Bar(
                x=preds["datetime"], y=preds["precipitation"],
                name="Précipitations (mm)", marker_color="#3498db",
                yaxis="y2", opacity=0.6,
            ))
            fig_meteo.update_layout(
                height=280,
                margin=dict(l=40, r=40, t=20, b=40),
                yaxis=dict(title="Température (°C)", side="left"),
                yaxis2=dict(title="Pluie (mm)", side="right", overlaying="y", showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_meteo, width="stretch")

            # ── Résumé quotidien ─────────────────────────────────
            st.subheader("Synthèse par jour")
            daily = preds.copy()
            daily["date"] = daily["datetime"].dt.date
            summary = daily.groupby("date").agg(
                proba_max=("outage_probability", "max"),
                proba_mean=("outage_probability", "mean"),
                heures_risque=("outage_probability", lambda x: int((x > 0.5).sum())),
                temp_max=("temperature_2m", "max"),
                pluie_mm=("precipitation", "sum"),
            ).reset_index()

            def _risk_label(p: float) -> str:
                if p > 0.7:
                    return "🔴 Élevé"
                if p > 0.4:
                    return "🟠 Moyen"
                return "🟢 Faible"

            summary["Niveau"] = summary["proba_max"].map(_risk_label)
            summary = summary.sort_values(["proba_max", "proba_mean"], ascending=[False, False]).reset_index(drop=True)
            summary_display = pd.DataFrame({
                "Jour": pd.to_datetime(summary["date"]).dt.strftime("%a %d %b"),
                "Niveau": summary["Niveau"],
                "Risque max": summary["proba_max"].map(lambda p: f"{p:.0%}"),
                "Risque moyen": summary["proba_mean"].map(lambda p: f"{p:.0%}"),
                "Heures à risque (>50%)": summary["heures_risque"],
                "Temp. max (°C)": summary["temp_max"].round(1),
                "Pluie (mm)": summary["pluie_mm"].round(1),
            })
            st.dataframe(summary_display, hide_index=True, width="stretch")

            # ── Top 5 heures critiques ───────────────────────────
            st.subheader("Top 5 heures les plus à risque")
            top5 = preds.nlargest(5, "outage_probability")[[
                "datetime", "outage_probability", "temperature_2m",
                "precipitation", "wind_speed_10m",
            ]].copy()
            top5_display = pd.DataFrame({
                "Date & heure": top5["datetime"].dt.strftime("%a %d %b %Hh"),
                "Probabilité": top5["outage_probability"].map(lambda p: f"{p:.0%}"),
                "Temp. (°C)": top5["temperature_2m"].round(1),
                "Pluie (mm)": top5["precipitation"].round(1),
                "Vent (km/h)": top5["wind_speed_10m"].round(1),
            })
            st.dataframe(top5_display, hide_index=True, width="stretch")


# ═══════════════════════════════════════════════════════════════════
# ONGLET 3 : SIMULATION MANUELLE
# ═══════════════════════════════════════════════════════════════════

with tab_simulate:
    st.markdown(
        "<p style='color:#888'>Ajustez les paramètres ci-dessous pour simuler "
        "un scénario et voir la probabilité de coupure correspondante.</p>",
        unsafe_allow_html=True,
    )

    # ── Paramètres de simulation ─────────────────────────────────

    st.subheader("Paramètres de la simulation")

    col_time, col_energy, col_meteo = st.columns(3)

    with col_time:
        st.markdown("**⏰ Temporel**")
        sim_hour = st.slider("Heure", 0, 23, 14, key="sim_hour")
        sim_month = st.slider("Mois", 1, 12, 6, key="sim_month")
        day_names = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
        sim_dow = st.selectbox("Jour de la semaine", options=range(7),
                               format_func=lambda x: day_names[x], index=2, key="sim_dow")

    h_avg = hospital.get("avg_load_kw", 133)
    h_max = hospital.get("max_load_kw", 235)
    h_solar = hospital.get("has_solar", True)

    with col_energy:
        st.markdown("**🔌 Énergie**")
        sim_load = st.slider(
            "Consommation totale (kW)",
            min_value=10.0, max_value=float(h_max * 1.5),
            value=float(h_avg),
            step=5.0, key="sim_load",
        )
        if h_solar:
            sim_solar = st.slider(
                "Production solaire PV (kW)",
                min_value=0.0, max_value=float(h_max * 0.7),
                value=float(h_avg * 0.3),
                step=5.0, key="sim_solar",
            )
        else:
            st.slider("Production solaire PV (kW)", min_value=0.0, max_value=1.0,
                       value=0.0, disabled=True, key="sim_solar_disabled")
            sim_solar = 0.0
        sim_base = st.slider(
            "Charge de base (kW)",
            min_value=10.0, max_value=float(h_max),
            value=float(h_avg * 0.85),
            step=5.0, key="sim_base",
        )
        sim_steril = st.slider(
            "Stérilisation (kW)",
            min_value=0.0, max_value=float(h_max * 0.3),
            value=float(h_avg * 0.06),
            step=1.0, key="sim_steril",
        )

    with col_meteo:
        st.markdown("**🌡️ Météo**")
        sim_temp = st.slider("Température (°C)", -10.0, 50.0, 25.0, step=0.5, key="sim_temp")
        sim_hum = st.slider("Humidité (%)", 0, 100, 70, key="sim_hum")
        sim_wind = st.slider("Vent (km/h)", 0.0, 100.0, 10.0, step=1.0, key="sim_wind")
        sim_precip = st.slider("Précipitations (mm)", 0.0, 50.0, 0.0, step=0.5, key="sim_precip")
        sim_pressure = st.slider("Pression (hPa)", 900.0, 1050.0, 1013.0, step=1.0, key="sim_pres")
        sim_rad = st.slider("Rayonnement solaire (W/m²)", 0.0, 1000.0, 200.0, step=10.0, key="sim_rad")

    st.divider()

    # ── Lancer la simulation ─────────────────────────────────────

    if st.button("🎯  Simuler", type="primary", width="stretch", key="btn_simulate"):

        params = {
            "hour": sim_hour,
            "month": sim_month,
            "day_of_week": sim_dow,
            "total_load_kw": sim_load,
            "solar_pv_kw": sim_solar,
            "base_load_kw": sim_base,
            "sterilization_kw": sim_steril,
            "temperature_2m": sim_temp,
            "humidity": sim_hum,
            "wind_speed": sim_wind,
            "precipitation": sim_precip,
            "pressure": sim_pressure,
            "radiation": sim_rad,
        }

        try:
            with st.spinner("Simulation en cours…"):
                sim_row = build_simulation_row(params, df, feature_cols)
                sim_row = ensure_numeric_feature_frame(sim_row, feature_cols)
                # Modèle hôpital (Lacor). Pour un site ≠ Lacor, le score est
                # illustratif (cf. site_profile_notes).
                proba_raw = model.predict_proba(sim_row)[0][1]
                proba, stress_details = apply_extrapolation_stress(proba_raw, params, df)
                factors = get_top_factors(model, feature_cols, sim_row.iloc[0])
                sim_shap_sv, sim_shap_ev = compute_shap_local(shap_explainer, sim_row, feature_cols)
                hospital_notes = site_profile_notes(hospital_key, hospital)
                duration = round(1.0 + proba * 4.0, 1) if proba > 0.5 else 0.5
                hours_away = max(1, round((1 - proba) * 24))
        except Exception as e:
            st.error(f"**Erreur lors de la simulation** : {e}")
            st.stop()

        show_risk_result(proba, hours_away, duration)

        if hospital_notes:
            st.info(
                f"**Profil de l'hôpital** :\n"
                + "\n".join(f"- {n}" for n in hospital_notes)
            )

        if stress_details:
            st.warning(
                "**Conditions extrêmes détectées** (hors des données d'entraînement) :\n"
                + "\n".join(f"- {d}" for d in stress_details)
                + f"\n\nProbabilité du modèle seul : {proba_raw:.0%} → ajustée à **{proba:.0%}**"
            )

        st.divider()

        col_gauge, col_explain = st.columns([1, 1])

        with col_gauge:
            st.subheader("Jauge de risque")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#e74c3c" if proba > 0.5 else "#2ecc71"},
                    "steps": [
                        {"range": [0, 40], "color": "rgba(46, 204, 113, 0.2)"},
                        {"range": [40, 70], "color": "rgba(243, 156, 18, 0.2)"},
                        {"range": [70, 100], "color": "rgba(231, 76, 60, 0.2)"},
                    ],
                    "threshold": {
                        "line": {"color": "#f39c12", "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=30, r=30, t=40, b=20))
            st.plotly_chart(fig_gauge, width="stretch")

        with col_explain:
            st.subheader("Facteurs explicatifs (SHAP)")
            if sim_shap_sv is not None:
                show_shap_waterfall(sim_shap_sv, sim_shap_ev, feature_cols,
                                    title="Pourquoi ce risque ?")
            else:
                show_factors(factors)

        st.divider()

        # ── Résumé du scénario simulé ────────────────────────────
        st.subheader("Résumé du scénario")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown("**⏰ Temporel**")
            st.markdown(
                f"- Heure : **{sim_hour}h**\n"
                f"- Mois : **{sim_month}**\n"
                f"- Jour : **{day_names[sim_dow]}**\n"
                f"- Week-end : **{'Oui' if sim_dow >= 5 else 'Non'}**"
            )
        with r2:
            st.markdown("**🔌 Énergie**")
            st.markdown(
                f"- Consommation : **{sim_load} kW**\n"
                f"- Solaire PV : **{sim_solar} kW**\n"
                f"- Charge de base : **{sim_base} kW**\n"
                f"- Stérilisation : **{sim_steril} kW**"
            )
        with r3:
            st.markdown("**🌡️ Météo**")
            st.markdown(
                f"- Température : **{sim_temp}°C**\n"
                f"- Humidité : **{sim_hum}%**\n"
                f"- Vent : **{sim_wind} km/h**\n"
                f"- Précipitations : **{sim_precip} mm**\n"
                f"- Pression : **{sim_pressure} hPa**\n"
                f"- Rayonnement : **{sim_rad} W/m²**"
            )

        # ── Comparaison avec la médiane ──────────────────────────
        st.divider()
        st.subheader("Comparaison avec les conditions moyennes")

        median_row = build_simulation_row({
            "hour": 12, "month": 6, "day_of_week": 2,
            "total_load_kw": float(h_avg),
            "solar_pv_kw": float(h_avg * 0.3) if h_solar else 0.0,
            "base_load_kw": float(h_avg * 0.85),
            "sterilization_kw": float(h_avg * 0.06),
            "temperature_2m": 25.0, "humidity": 70, "wind_speed": 10.0,
            "precipitation": 0.0, "pressure": 1013.0, "radiation": 200.0,
        }, df, feature_cols)
        median_row = ensure_numeric_feature_frame(median_row, feature_cols)
        median_proba = float(site_predict_proba(median_row)[0])

        delta = proba - median_proba
        delta_str = f"{delta:+.0%}"

        c1, c2, c3 = st.columns(3)
        c1.metric("Votre scénario", f"{proba:.0%}")
        c2.metric("Conditions moyennes", f"{median_proba:.0%}")
        c3.metric("Différence", delta_str, delta=f"{delta:+.0%}",
                   delta_color="inverse")
