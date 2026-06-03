"""
Couche d'accès aux données pour l'app Streamlit (app.py).

Chargements cachés (@st.cache_*) : modèle + SHAP, features Lacor, profils
ERIC/NYC/africa_grid, détection des sources par hôpital et catalogue
``HOSPITAL_DISPLAY`` (hors ``ui_hidden``). Ne dépend QUE de
src.* (aucun import d'app.py) -> pas de cycle. Extrait d'app.py (#10, palier 3).
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.utils.config import (
    ELECTRICITYMAPS_TOKEN_ENV,
    EXTERNAL_SIGNAL_PREFIXES,
    FEATURES_DIR,
    HOSPITAL_LOCATIONS,
    MODELS_DIR,
    ROOT_DIR,
)
from src.utils.hospitals import (
    HOSPITAL_DISPLAY as _HOSPITAL_DISPLAY_FULL,
    TARGET_SOURCE_META,
    get_target_source,
)
from src.utils.local_signals import eaglei_path
# ui_content est pur (zéro Streamlit, n'importe ni app ni app_data) -> pas de cycle.
# load_table / apply_feature_engineering_single sont importés localement dans les
# fonctions concernées.
from src.ui_content import feature_label, get_feature_category

ROOT = ROOT_DIR


HOSPITAL_DISPLAY = {
    k: v for k, v in _HOSPITAL_DISPLAY_FULL.items()
    if not v.get("ui_hidden")
}


ERIC_DIR = ROOT / "data" / "raw" / "eric"


_CALIBRATED_MODEL_NAMES = ("calibrated_model.joblib", "calibrated_rf.joblib")


_BASELINE_MODEL_NAMES = ("baseline_model.joblib", "baseline_rf.joblib")


def _resolve_model_path(names: tuple[str, ...]) -> Path | None:
    """Renvoie le premier fichier modèle existant parmi `names`, ou None."""
    for name in names:
        p = MODELS_DIR / name
        if p.exists():
            return p
    return None


def _model_file_mtime() -> float:
    """Retourne le mtime du modèle pour invalider le cache quand le fichier change."""
    for names in (_CALIBRATED_MODEL_NAMES, _BASELINE_MODEL_NAMES):
        p = _resolve_model_path(names)
        if p is not None:
            return p.stat().st_mtime
    return 0.0


@st.cache_resource
def load_model(_mtime: float = 0.0):
    calibrated_path = _resolve_model_path(_CALIBRATED_MODEL_NAMES)
    baseline_path = _resolve_model_path(_BASELINE_MODEL_NAMES)
    summary_path = MODELS_DIR / "training_summary.json"

    winner_name = "?"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                winner_name = json.load(f).get("winner", "?")
        except Exception:
            pass

    if calibrated_path is not None:
        try:
            return joblib.load(calibrated_path)
        except Exception as e:
            st.sidebar.warning(f"Échec du modèle calibré : {e} — fallback sur le brut")

    if baseline_path is not None:
        try:
            return joblib.load(baseline_path)
        except Exception as e:
            st.error(f"**Erreur au chargement du modèle** : {e}")
            st.stop()

    st.error(
        "**Aucun modèle trouvé.**\n\n"
        "Exécutez d'abord le pipeline d'entraînement :\n"
        "```bash\npython run_pipeline.py\n```"
    )
    st.stop()


_HORIZON_DIR = MODELS_DIR / "nowcast_horizons"
_HORIZON_HOURS = (1, 3, 6)


def _horizon_models_mtime() -> float:
    """mtime max des modèles horizons (invalide le cache au ré-entraînement)."""
    mtimes = []
    for h in _HORIZON_HOURS:
        p = _HORIZON_DIR / f"horizon_{h}h" / "horizon_model.joblib"
        if p.exists():
            mtimes.append(p.stat().st_mtime)
    summary = _HORIZON_DIR / "horizons_summary.json"
    if summary.exists():
        mtimes.append(summary.stat().st_mtime)
    return max(mtimes) if mtimes else 0.0


@st.cache_resource
def load_horizon_models(_mtime: float = 0.0) -> dict[int, dict]:
    """Modèles « coupure dans les H h » (mêmes features que le nowcast Lacor).

    Retourne {1: bundle, 3: bundle, 6: bundle} ou {} si non entraînés
    (`python -m src.models.train_horizons` ou `run_pipeline.py`)."""
    out: dict[int, dict] = {}
    for h in _HORIZON_HOURS:
        p = _HORIZON_DIR / f"horizon_{h}h" / "horizon_model.joblib"
        if not p.exists():
            continue
        try:
            out[h] = joblib.load(p)
        except Exception as e:  # noqa: BLE001
            st.sidebar.warning(f"Modèle horizon {h}h illisible : {e}")
    return out


@st.cache_resource
def load_duration_model(_mtime: float = 0.0) -> dict | None:
    """Modèle de durée des coupures (régression dédiée, cf. train_duration.py).

    Retourne le bundle {model, features, duration_min_h, duration_max_h, …}
    ou None si non entraîné (l'app retombe alors sur l'heuristique `1 + 4p`).
    """
    path = MODELS_DIR / "duration_model.joblib"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as e:  # noqa: BLE001
        st.sidebar.warning(f"Modèle de durée illisible : {e}")
        return None


@st.cache_resource
def load_shap_explainer(_mtime: float = 0.0):
    explainer_path = MODELS_DIR / "shap_explainer.joblib"
    if not explainer_path.exists():
        return None
    try:
        return joblib.load(explainer_path)
    except Exception:
        return None


def _features_file_mtime() -> float:
    # On regarde parquet ET CSV : le plus récent suffit pour invalider
    # le cache Streamlit dès que l'un est régénéré.
    candidates = [
        FEATURES_DIR / "features_dataset.parquet",
        FEATURES_DIR / "features_dataset.csv",
    ]
    mtimes = [p.stat().st_mtime for p in candidates if p.exists()]
    return max(mtimes) if mtimes else 0.0


@st.cache_data
def load_lacor_features(_mtime: float = 0.0):
    from src.utils.io import load_table
    csv_path = FEATURES_DIR / "features_dataset.csv"
    parquet_path = FEATURES_DIR / "features_dataset.parquet"
    if not csv_path.exists() and not parquet_path.exists():
        st.error(
            f"**Données Lacor introuvables** : `{csv_path}`\n\n"
            "Exécutez d'abord le pipeline de preprocessing :\n"
            "```bash\npython run_pipeline.py\n```"
        )
        st.stop()
    try:
        # `load_table` privilégie parquet (~10× plus rapide qu'un CSV de
        # 100 colonnes × 100k lignes), fallback CSV transparent.
        df = load_table(csv_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        # Le dataset de features contient désormais plusieurs hôpitaux
        # (expérience multi-sites archivée). L'app pilote est MONO-SITE Lacor :
        # on ne garde que ses lignes, sinon les vues (charge, fenêtre 24 h) et
        # les prédictions mélangent 16 hôpitaux (charges jusqu'à 8000 kW).
        if "hospital" in df.columns and (df["hospital"] == "lacor_uganda").any():
            df = df[df["hospital"] == "lacor_uganda"].reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"**Erreur au chargement des données Lacor** : {e}")
        st.stop()


def _apply_feature_engineering(df: pd.DataFrame, hospital_key: str) -> pd.DataFrame:
    """Feature engineering aligné pipeline ; fériés = calendrier du pays du site."""
    from src.features.build_features import apply_feature_engineering_single
    return apply_feature_engineering_single(df, hospital_key=hospital_key)


@st.cache_data
def load_eric_features(eric_code: str, hospital_info: dict) -> pd.DataFrame | None:
    csv_path = ERIC_DIR / f"eric_{eric_code}_hourly.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.warning(f"Impossible de lire les données ERIC `{eric_code}` : {e}")
        return None

    # Récupérer la météo : on cherche d'abord un fichier propre à
    # l'hôpital (vraie météo locale Open-Meteo), sinon fallback sur Lacor
    # avec offset de température lié à la latitude.
    hospital_key = next(
        (k for k, v in HOSPITAL_DISPLAY.items()
         if v.get("eric_code") == eric_code),
        None,
    )
    local_meteo = (
        ROOT / "data" / "raw" / f"meteo_{hospital_key}.csv"
        if hospital_key else None
    )

    df["datetime"] = pd.to_datetime(df["datetime"])

    if local_meteo and local_meteo.exists():
        meteo = pd.read_csv(local_meteo)
        meteo["datetime"] = pd.to_datetime(meteo["datetime"])
        meteo_cols = [c for c in meteo.columns if c not in ("datetime", "hospital")]
        meteo = (
            meteo.sort_values("datetime")
            .drop_duplicates(subset=["datetime"], keep="last")
        )
        # Alignement robuste par timestamp (évite les erreurs si longueurs différentes,
        # ex. météo partielle 744 lignes vs ERIC 8760 lignes).
        df = df.merge(meteo[["datetime", *meteo_cols]], on="datetime", how="left")
    else:
        lacor_meteo = ROOT / "data" / "raw" / "meteo_lacor_uganda.csv"
        if lacor_meteo.exists():
            meteo = pd.read_csv(lacor_meteo)
            meteo["datetime"] = pd.to_datetime(meteo["datetime"])
            lat = hospital_info.get("lat", 51.5)
            temp_offset = (51.5 - lat) * 0.15
            meteo["temperature_2m"] = meteo["temperature_2m"] - temp_offset
            meteo_cols = [c for c in meteo.columns if c not in ("datetime", "hospital")]
            meteo = (
                meteo.sort_values("datetime")
                .drop_duplicates(subset=["datetime"], keep="last")
            )
            df = df.merge(meteo[["datetime", *meteo_cols]], on="datetime", how="left")

    df = _apply_feature_engineering(df, hospital_key or "__eric__")
    return df


@st.cache_data
def load_africa_grid_features(hospital_key: str, hospital_info: dict) -> pd.DataFrame | None:
    """Profil `africa_grid` : Lacor cloné + météo locale au point (lat/lon).

    Pas de compteur public ni de coupure étiquetée pour le site. Les colonnes
    ``em_*`` éventuelles héritées de Lacor sont mises à 0 : Electricity Maps
    décrit une **zone réseau entière**, pas l'hôpital — on ne fusionne pas
    ``electricitymaps_<key>.csv`` ici (cf. panneau sources = contexte documenté
    seulement).
    """
    base = load_lacor_features(_features_file_mtime())
    if base is None or base.empty:
        return None
    df = base.copy()
    # NOTE : on conserve les timestamps 2022 d'origine pour que la météo
    # locale 2022, l'historique Lacor 2022 et les features cycliques
    # (month, hour_sin/cos, is_public_holiday) restent cohérents entre
    # eux. L'ancien rebase vers `now()` désynchronisait la météo 2022
    # avec le mois affiché (cf. analyse P1.2).
    target_avg = float(hospital_info.get("avg_load_kw", 133))
    lacor_avg = 133.0
    scale = target_avg / lacor_avg if lacor_avg > 0 else 1.0

    consumption_cols = [
        "total_load_kw", "solar_pv_kw", "base_load_kw",
        "generators_kw", "sterilization_kw",
    ]
    for col in consumption_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") * scale

    if not hospital_info.get("has_solar") and "solar_pv_kw" in df.columns:
        df["solar_pv_kw"] = 0.0
    if not hospital_info.get("has_generator"):
        for col in ("generators_kw",):
            if col in df.columns:
                df[col] = 0.0

    cols_to_zero = [
        c for c in df.columns
        if any(c.startswith(p) for p in EXTERNAL_SIGNAL_PREFIXES)
    ]
    if cols_to_zero:
        df.loc[:, cols_to_zero] = 0

    local_meteo = ROOT / "data" / "raw" / f"meteo_{hospital_key}.csv"
    if local_meteo.exists():
        try:
            meteo = pd.read_csv(local_meteo)
            meteo["datetime"] = pd.to_datetime(meteo["datetime"])
            meteo_cols = [c for c in meteo.columns if c not in ("datetime", "hospital")]
            df["datetime"] = pd.to_datetime(df["datetime"])
            # Merge robuste par datetime (et plus par position) : la météo
            # locale a la même résolution horaire que le profil Lacor
            # mais ses lignes peuvent être décalées d'une heure ou avoir
            # des trous.
            df = df.drop(columns=[c for c in meteo_cols if c in df.columns])
            df = pd.merge_asof(
                df.sort_values("datetime"),
                meteo[["datetime", *meteo_cols]].sort_values("datetime"),
                on="datetime",
                direction="nearest",
                tolerance=pd.Timedelta("1h"),
            )
        except Exception as e:
            st.warning(f"Météo locale {hospital_key} illisible : {e}")

    # Pas de fusion Electricity Maps : signal zone entière, pas local au site.

    df = _apply_feature_engineering(df, hospital_key)
    return df


@st.cache_data
def load_nyc_features(nyc_code: str, hospital_info: dict) -> pd.DataFrame | None:
    """Charge les profils horaires NYC LL84 + météo locale Open-Meteo."""
    nyc_dir = ROOT / "data" / "raw" / "nyc_ll84"
    csv_path = nyc_dir / f"{nyc_code}_hourly.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.warning(f"Impossible de lire les données NYC LL84 `{nyc_code}` : {e}")
        return None

    hospital_key = next(
        (k for k, v in HOSPITAL_DISPLAY.items()
         if v.get("nyc_code") == nyc_code),
        None,
    )
    local_meteo = (
        ROOT / "data" / "raw" / f"meteo_{hospital_key}.csv"
        if hospital_key else None
    )
    df["datetime"] = pd.to_datetime(df["datetime"])

    if local_meteo and local_meteo.exists():
        meteo = pd.read_csv(local_meteo)
        meteo["datetime"] = pd.to_datetime(meteo["datetime"])
        meteo_cols = [c for c in meteo.columns if c not in ("datetime", "hospital")]
        meteo = (
            meteo.sort_values("datetime")
            .drop_duplicates(subset=["datetime"], keep="last")
        )
        df = df.merge(meteo[["datetime", *meteo_cols]], on="datetime", how="left")

    df = _apply_feature_engineering(df, hospital_key or "__nyc__")
    return df


_RAW_DIR = ROOT / "data" / "raw"


def detect_hospital_data_sources(hospital_key: str, hospital_info: dict) -> list[dict]:
    """Renvoie la liste des sources de données réellement disponibles pour
    cet hôpital, sous forme de dicts {label, emoji, color, status, detail}.
    `status` ∈ {"primary", "available", "synthetic", "missing"}.
    """
    sources: list[dict] = []

    # ── 0. Cible coupures (is_outage) — provenance de l'étiquette ──
    # Affiché en tête : c'est l'information d'honnêteté la plus importante
    # (un score n'a pas le même statut selon que la coupure est réelle,
    # synthétique ou absente). Source de vérité : get_target_source().
    _tsrc = get_target_source(hospital_key, hospital_info)
    _tmeta = TARGET_SOURCE_META[_tsrc]
    sources.append({
        "label": f"Cible coupures — {_tmeta['label']}",
        "emoji": _tmeta["emoji"], "color": _tmeta["color"], "status": _tmeta["status"],
        "detail": _tmeta["detail"],
    })

    # ── 1. Consommation électrique ─────────────────────────────────
    if hospital_info.get("data_source") == "eric":
        eric_code = hospital_info.get("eric_code", "")
        eric_path = ROOT / "data" / "raw" / "eric" / f"eric_{eric_code}_hourly.csv"
        if eric_path.exists():
            sources.append({
                "label": "Consommation NHS ERIC",
                "emoji": "📊", "color": "#2ecc71", "status": "primary",
                "detail": f"Données réelles · {eric_path.name}",
            })
        else:
            sources.append({
                "label": "Consommation NHS ERIC",
                "emoji": "📊", "color": "#e74c3c", "status": "missing",
                "detail": "Fichier introuvable",
            })
    elif hospital_info.get("data_source") == "nyc_ll84":
        nyc_code = hospital_info.get("nyc_code", "")
        nyc_path = ROOT / "data" / "raw" / "nyc_ll84" / f"{nyc_code}_hourly.csv"
        if nyc_path.exists():
            sources.append({
                "label": "Consommation NYC Local Law 84",
                "emoji": "📊", "color": "#2ecc71", "status": "primary",
                "detail": f"data.cityofnewyork.us · {nyc_path.name}",
            })
        else:
            sources.append({
                "label": "Consommation NYC Local Law 84",
                "emoji": "📊", "color": "#e74c3c", "status": "missing",
                "detail": "Fichier introuvable",
            })
    elif hospital_key == "lacor_uganda":
        if (_RAW_DIR / "lacor_clean.csv").exists():
            sources.append({
                "label": "Consommation Lacor (terrain)",
                "emoji": "📊", "color": "#2ecc71", "status": "primary",
                "detail": "Relevés horaires Hôpital Lacor 2022",
            })
    else:
        sources.append({
            "label": "Consommation (profil cloné Lacor + scaling)",
            "emoji": "📊", "color": "#f39c12", "status": "synthetic",
            "detail": f"Profil Lacor re-mis à l'échelle ({hospital_info.get('avg_load_kw', '?')} kW)",
        })

    # ── 2. Météo ──────────────────────────────────────────────────
    meteo_path = _RAW_DIR / f"meteo_{hospital_key}.csv"
    if meteo_path.exists():
        sources.append({
            "label": "Météo Open-Meteo (historique)",
            "emoji": "🌤️", "color": "#2ecc71", "status": "primary",
            "detail": "Historique horaire 2022 (lat/lon hôpital)",
        })
    else:
        sources.append({
            "label": "Météo extrapolée Lacor (offset latitude)",
            "emoji": "🌤️", "color": "#f39c12", "status": "synthetic",
            "detail": "Compromis : météo Lacor avec correction de température",
        })
    # ── 3. Electricity Maps (zone au point GPS) ───────────────────
    from src.utils.em_zone import load_zone_cache, resolve_zone_precise

    _em_cache = load_zone_cache().get(hospital_key, {})
    _zone = _em_cache.get("zone")
    _em_csv = _RAW_DIR / f"electricitymaps_{hospital_key}.csv"
    if _zone or _em_csv.exists():
        _src = _em_cache.get("source", "csv")
        sources.append({
            "label": "Electricity Maps (zone réseau au point hôpital)",
            "emoji": "⚡",
            "color": "#3498db",
            "status": "available",
            "detail": (
                f"Zone `{_zone or '?'}` (résolution {_src} via lat/lon) — "
                "risque contextuel dans le panneau dédié ; hors modèle ML"
            ),
        })
    elif os.environ.get(ELECTRICITYMAPS_TOKEN_ENV):
        _res = resolve_zone_precise(hospital_key)
        if _res.get("zone"):
            sources.append({
                "label": "Electricity Maps (zone réseau au point hôpital)",
                "emoji": "⚡",
                "color": "#95a5a6",
                "status": "context",
                "detail": f"Zone `{_res['zone']}` ({_res.get('source')}) — rafraîchir le panneau risque live",
            })
        else:
            sources.append({
                "label": "Electricity Maps",
                "emoji": "⚡",
                "color": "#bdc3c7",
                "status": "missing",
                "detail": "Zone non résolue pour ce point — vérifier token ou couverture",
            })
    else:
        sources.append({
            "label": "Electricity Maps",
            "emoji": "⚡",
            "color": "#bdc3c7",
            "status": "missing",
            "detail": f"Définir `{ELECTRICITYMAPS_TOKEN_ENV}` pour activer l'analyse réseau live",
        })

    # ── 4. Coupures réelles comté (EAGLE-I, USA) ─────────────────
    _eaglei = eaglei_path(hospital_key, hospital_info)
    if _eaglei is not None:
        if _eaglei.exists():
            sources.append({
                "label": "Coupures réelles comté (EAGLE-I)",
                "emoji": "🗽", "color": "#2ecc71", "status": "primary",
                "detail": (
                    f"Clients sans courant dans le comté — {_eaglei.name} "
                    f"(plus fin qu'une zone NY-ISO entière)"
                ),
            })
        else:
            sources.append({
                "label": "Coupures réelles comté (EAGLE-I)",
                "emoji": "🗽", "color": "#e67e22", "status": "missing",
                "detail": (
                    f"Comté `{hospital_info.get('eaglei_county_key')}` — lancer "
                    "`python -m src.data.ingest_eaglei` (bruts figshare requis)"
                ),
            })

    return sources


@st.cache_data
def load_eaglei_local(hospital_key: str, hospital_info: dict) -> pd.DataFrame | None:
    """Coupures réelles EAGLE-I au comté de l'hôpital (USA), si rattaché."""
    path = eaglei_path(hospital_key, hospital_info)
    if path is None or not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception:
        return None


def _neutralize_external_signals(df: pd.DataFrame, hospital_key: str) -> pd.DataFrame:
    """Hors Lacor : met à 0 les colonnes `em_*` (réseau Electricity Maps)."""
    if hospital_key == "lacor_uganda":
        return df
    df = df.copy()
    cols_to_zero = [
        c for c in df.columns
        if any(c.startswith(p) for p in EXTERNAL_SIGNAL_PREFIXES)
    ]
    if cols_to_zero:
        df.loc[:, cols_to_zero] = 0
    return df


@st.cache_data
def load_hospital_data(hospital_key: str, hospital_info: dict) -> pd.DataFrame:
    """Charge les données de l'hôpital sélectionné.

    Sources supportées :
      - lacor_uganda : relevés terrain horaires 2022
      - *_nhs        : données NHS ERIC désagrégées en horaire
      - nyc_*        : données NYC LL84 désagrégées en horaire
      - africa_grid  : profil estimé à partir d'un profil de référence
    """
    if hospital_info.get("data_source") == "eric":
        eric_code = hospital_info["eric_code"]
        eric_df = load_eric_features(eric_code, hospital_info)
        if eric_df is not None:
            return _neutralize_external_signals(eric_df, hospital_key)
        st.error(
            f"**Données ERIC introuvables** pour `{eric_code}`. "
            f"Vérifiez `data/raw/eric/eric_{eric_code}_hourly.csv`."
        )
        st.stop()

    if hospital_info.get("data_source") == "nyc_ll84":
        nyc_code = hospital_info["nyc_code"]
        nyc_df = load_nyc_features(nyc_code, hospital_info)
        if nyc_df is not None:
            return _neutralize_external_signals(nyc_df, hospital_key)
        st.error(
            f"**Données NYC LL84 introuvables** pour `{nyc_code}`. "
            f"Vérifiez `data/raw/nyc_ll84/{nyc_code}_hourly.csv`."
        )
        st.stop()

    if hospital_info.get("data_source") == "africa_grid":
        africa_df = load_africa_grid_features(hospital_key, hospital_info)
        if africa_df is not None:
            return africa_df
        st.error(
            f"**Profil africain introuvable** pour `{hospital_key}`. "
            "Vérifiez que `data/features/features_dataset.csv` existe."
        )
        st.stop()

    if hospital_key == "lacor_uganda":
        return load_lacor_features(_features_file_mtime())

    st.error(
        f"Hôpital `{hospital_key}` non supporté : aucune source de "
        "consommation réelle disponible."
    )
    st.stop()


@st.cache_data
def load_global_shap_importance() -> pd.DataFrame | None:
    """Charge l'importance SHAP moyenne par feature (entraînement)."""
    p = MODELS_DIR / "shap_feature_importance.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if "feature" not in df.columns:
            df.columns = ["feature", "mean_abs_shap"]
        else:
            value_col = [c for c in df.columns if c != "feature"][0]
            df = df.rename(columns={value_col: "mean_abs_shap"})
        df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        df["category"] = df["feature"].apply(get_feature_category)
        df["label"] = df["feature"].apply(feature_label)
        return df
    except Exception:
        return None


def _duration_summary_mtime() -> float:
    p = MODELS_DIR / "duration_summary.json"
    return p.stat().st_mtime if p.exists() else 0.0


def _multisite_summary_mtime() -> float:
    p = MODELS_DIR / "multisite_summary.json"
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data
def load_duration_summary(_mtime: float = 0.0) -> dict | None:
    """Résumé du modèle de durée (MAE vs baselines, top features, stats).

    Lu depuis models/duration_summary.json (écrit par train_duration.py).
    Renvoie None si le modèle de durée n'a pas été entraîné.
    """
    p = MODELS_DIR / "duration_summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def load_live_outage_risk(hospital_key: str, _bust: float = 0.0) -> dict | None:
    """Risque contextuel réseau (EM zone GPS) + météo point hôpital. Cache 15 min."""
    from src.grid_outage_risk import live_outage_risk

    try:
        return live_outage_risk(hospital_key)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Analyse Electricity Maps / risque live indisponible : {e}")
        return None


@st.cache_data
def load_multisite_summary(_mtime: float = 0.0) -> dict | None:
    """Résumé de la validation de généralisation inter-sites (LOSO).

    Lu depuis models/multisite_summary.json (écrit par multisite_experiment.py).
    Renvoie None si l'expérience multi-sites (EAGLE-I) n'a pas été lancée.
    """
    p = MODELS_DIR / "multisite_summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


