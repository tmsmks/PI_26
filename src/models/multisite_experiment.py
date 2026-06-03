"""
Expérience multi-sites : la généralisation inter-sites se MESURE, pas se suppose.

Question
--------
Le modèle hôpital complet (Lacor seul) atteint ROC-AUC ≈ 0.99, mais il est
entraîné et testé sur **un unique site** à coupures réelles. Tient-il sur
d'autres réseaux réels qu'il n'a jamais vus ?

Protocole — Leave-One-Site-Out (LOSO)
-------------------------------------
On rassemble tous les sites à **coupures réelles** disponibles :
  • Lacor (Ouganda, hôpital, `is_outage` réel 2022) ;
  • les grands comtés urbains américains présents (coupures réelles EAGLE-I, cf.
    `src/data/ingest_eaglei.py`), cible binarisée au quantile p90 par site.
    Avec le brut national 2023 ingéré, cela fait Lacor + 11 comtés = 12 sites.
Pour chaque site, on entraîne un LightGBM sur tous les AUTRES sites et on prédit
le site exclu. ROC-AUC moyen > 0.5 ⇒ le signal **exogène** (météo + temporel)
généralise (partiellement) entre réseaux réels. Les coupures EAGLE-I (UTC) sont
réalignées sur l'heure locale du comté avant jointure avec la météo Open-Meteo.

Modèle exogène volontairement
-----------------------------
On n'utilise QUE des features disponibles partout : météo Open-Meteo +
calendrier. Pas de consommation kW ni d'auto-régression des coupures (qui
dominent sur Lacor mais ne transfèrent pas). Ce modèle ne REMPLACE donc pas
le modèle hôpital complet : il quantifie ce que la météo seule annonce.

Sorties (régénère les artefacts existants)
------------------------------------------
  models/multisite_summary.json       résumé + ROC LOSO par agrégat
  models/multisite_loso_by_site.csv    métriques détaillées par site

Lancer :  python -m src.models.multisite_experiment
Prérequis : data/raw/eaglei_<site>.csv (sinon le script l'explique et s'arrête).
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.ingest_eaglei import EAGLEI_COUNTIES, output_path
from src.data.ingest_meteo import fetch_meteo_archive
from src.utils.config import (
    EAGLEI_YEAR,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DIR,
)
from src.utils.io import load_csv, load_table, save_csv

logger = logging.getLogger(__name__)

LACOR_SITE = "Lacor/Uganda (hôpital)"
LACOR_LAT, LACOR_LON, LACOR_YEAR = 2.77, 32.30, 2022

# ── Jeu de features EXOGÈNES (météo + temporel) — exactement 29 ──────
# Disponibles sur n'importe quel site ⇒ transférables. Aucune conso, aucune
# auto-régression de coupure (qui ne généralisent pas hors du site source).
TEMPORAL_FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
]
WEATHER_RAW_FEATURES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "wind_speed_10m",
    "wind_gusts_10m", "precipitation", "surface_pressure", "shortwave_radiation",
    "cloud_cover", "visibility", "et0_fao_evapotranspiration", "cape", "weathercode",
]
WEATHER_DERIVED_FEATURES = [
    "temp_humidity_interaction", "wind_precipitation_interaction",
    "rain_intensity", "heat_stress", "thermal_amplitude_24h", "pressure_change_3h",
]
EXOG_FEATURES = TEMPORAL_FEATURES + WEATHER_RAW_FEATURES + WEATHER_DERIVED_FEATURES
assert len(EXOG_FEATURES) == 29, f"attendu 29 features exogènes, obtenu {len(EXOG_FEATURES)}"


def build_exog_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construit les 29 features exogènes à partir de `datetime` + météo brute.

    `df` doit contenir `datetime` et les 13 colonnes météo Open-Meteo (les
    manquantes sont créées à 0). Renvoie une copie avec toutes les colonnes
    de `EXOG_FEATURES` garanties présentes.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Temporel
    dt = df["datetime"].dt
    df["hour"] = dt.hour
    df["day_of_week"] = dt.dayofweek
    df["month"] = dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    doy = dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)

    # Météo brute : garantir la présence des 13 colonnes.
    for col in WEATHER_RAW_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Météo dérivée (mêmes formules que build_features, sans groupby hôpital
    # car chaque site est traité séparément en amont).
    df["temp_humidity_interaction"] = df["temperature_2m"] * df["relative_humidity_2m"] / 100
    df["wind_precipitation_interaction"] = df["wind_speed_10m"] * df["precipitation"]
    df["rain_intensity"] = df["precipitation"] * df["wind_speed_10m"]
    df["heat_stress"] = (df["temperature_2m"] > 30).astype(int)
    df["thermal_amplitude_24h"] = (
        df["temperature_2m"].rolling(24, min_periods=1).max()
        - df["temperature_2m"].rolling(24, min_periods=1).min()
    )
    df["pressure_change_3h"] = df["surface_pressure"].diff(3).fillna(0)

    df[EXOG_FEATURES] = df[EXOG_FEATURES].fillna(0.0)
    return df


def _cached_weather(key: str, lat: float, lon: float, year: int) -> pd.DataFrame:
    """Météo horaire annuelle pour un site (cache CSV pour éviter de re-fetch)."""
    cache = RAW_DIR / f"eaglei_meteo_{key}.csv"
    if cache.exists():
        df = load_csv(cache)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df
    df = fetch_meteo_archive(lat, lon, f"{year}-01-01", f"{year}-12-31")
    save_csv(df, cache)
    return df


def load_site_frame(
    site: str, lat: float, lon: float, year: int, outages: pd.DataFrame,
    tz: str | None = None,
) -> pd.DataFrame:
    """Assemble un site : cible `is_outage` + 29 features exogènes alignées.

    `outages` : DataFrame horaire (`datetime`, `is_outage`) du site.
    Jointure interne sur l'heure calendaire avec la météo du même site/année.

    Alignement temporel : Open-Meteo renvoie l'heure **locale** (`timezone=auto`).
    EAGLE-I horodate en **UTC** → si `tz` est fourni (fuseau IANA du comté), on
    convertit la grille des coupures UTC → heure locale avant la jointure, pour
    apparier chaque coupure à la météo de la **même heure locale** (gère l'heure
    d'été). Quelques heures en bord d'année tombent alors hors jointure (perte
    négligeable). `tz=None` (ex. Lacor, déjà en heure locale) → aucun décalage.
    """
    outages = outages.copy()
    outages["datetime"] = pd.to_datetime(outages["datetime"])
    if tz is not None:
        from zoneinfo import ZoneInfo
        local = outages["datetime"].dt.tz_localize("UTC").dt.tz_convert(ZoneInfo(tz))
        outages["datetime"] = local.dt.tz_localize(None)
    weather = _cached_weather_key(site, lat, lon, year)
    merged = outages.merge(weather, on="datetime", how="inner")
    feats = build_exog_features(merged)
    feats["site"] = site
    keep = ["site", "datetime", "is_outage", *EXOG_FEATURES]
    return feats[keep]


def _cached_weather_key(site: str, lat: float, lon: float, year: int) -> pd.DataFrame:
    # `site` libellé → clé de fichier sûre.
    key = site.split("/")[0].lower().replace(" ", "_").replace("(", "").replace(")", "")
    return _cached_weather(key, lat, lon, year)


def _load_lacor_outages() -> pd.DataFrame:
    df = load_table(PROCESSED_DIR / "hospital_merged.csv")
    lac = df[df["hospital"] == "lacor_uganda"][["datetime", "is_outage"]].copy()
    lac["datetime"] = pd.to_datetime(lac["datetime"])
    return lac.reset_index(drop=True)


def assemble_dataset() -> pd.DataFrame:
    """Empile Lacor (réel) + comtés EAGLE-I disponibles en un seul DataFrame."""
    frames = [load_site_frame(LACOR_SITE, LACOR_LAT, LACOR_LON, LACOR_YEAR, _load_lacor_outages())]

    for county in EAGLEI_COUNTIES:
        path = output_path(county["key"])
        if not path.exists():
            logger.warning("Comté %s absent (%s) — ignoré.", county["site"], path.name)
            continue
        out = load_csv(path)[["datetime", "is_outage"]]
        out["datetime"] = pd.to_datetime(out["datetime"])
        frames.append(load_site_frame(
            county["site"], county["lat"], county["lon"], EAGLEI_YEAR, out,
            tz=county.get("tz"),
        ))

    return pd.concat(frames, ignore_index=True)


def _make_model(pos_rate: float):
    """LightGBM exogène, `scale_pos_weight` ajusté au déséquilibre du train."""
    from lightgbm import LGBMClassifier

    spw = float(np.clip((1 - pos_rate) / max(pos_rate, 1e-3), 1, 50))
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        scale_pos_weight=spw,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=1,
        verbose=-1,
    )


def _site_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4) if len(set(y_true)) > 1 else float("nan"),
        "brier": round(brier_score_loss(y_true, proba), 4),
    }


def run_loso(data: pd.DataFrame) -> list[dict]:
    """Validation leave-one-site-out : chaque site prédit par les autres."""
    sites = list(dict.fromkeys(data["site"]))
    rows: list[dict] = []
    for held in sites:
        train = data[data["site"] != held]
        test = data[data["site"] == held]
        X_tr, y_tr = train[EXOG_FEATURES], train["is_outage"].to_numpy()
        X_te, y_te = test[EXOG_FEATURES], test["is_outage"].to_numpy()

        model = _make_model(y_tr.mean())
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]

        metrics = _site_metrics(y_te, proba)
        rows.append({
            "site": held,
            "n_test": int(len(test)),
            "outage_rate": round(float(y_te.mean()), 4),
            **metrics,
        })
        logger.info(
            "  LOSO %-28s ROC-AUC=%.3f  F1=%.3f  (n=%d, taux=%.1f%%)",
            held, metrics["roc_auc"], metrics["f1"], len(test), 100 * y_te.mean(),
        )
    return rows


def _single_site_baseline(data: pd.DataFrame) -> float:
    """ROC d'un modèle entraîné sur Lacor SEUL, testé sur un comté US.

    Sanity-check : sans pooling, un modèle mono-site ne transfère quasiment
    pas (ROC proche de 0.5) — d'où l'intérêt du pooling multi-sites.
    """
    target_site = "Maricopa/Phoenix AZ"
    if target_site not in set(data["site"]):
        return float("nan")
    train = data[data["site"] == LACOR_SITE]
    test = data[data["site"] == target_site]
    model = _make_model(train["is_outage"].mean())
    model.fit(train[EXOG_FEATURES], train["is_outage"].to_numpy())
    proba = model.predict_proba(test[EXOG_FEATURES])[:, 1]
    y = test["is_outage"].to_numpy()
    return round(roc_auc_score(y, proba), 4) if len(set(y)) > 1 else float("nan")


def run() -> None:
    data = assemble_dataset()
    n_sites = data["site"].nunique()
    if n_sites < 2:
        logger.warning(
            "Expérience multi-sites : %d site(s) seulement (EAGLE-I manquant ?). "
            "Lancer d'abord `python -m src.data.ingest_eaglei`. Abandon.",
            n_sites,
        )
        return

    logger.info("Expérience multi-sites : %d sites, %d lignes, %d features exogènes.",
                n_sites, len(data), len(EXOG_FEATURES))

    rows = run_loso(data)
    by_site = pd.DataFrame(rows)
    save_csv(by_site, MODELS_DIR / "multisite_loso_by_site.csv")

    roc = by_site["roc_auc"].dropna()
    summary = {
        "approach": "pool Lacor + comtés EAGLE-I, features exogènes (météo+temporel)",
        "n_sites": int(n_sites),
        "n_features": len(EXOG_FEATURES),
        "n_rows": int(len(data)),
        "eaglei_year": EAGLEI_YEAR,
        "loso_roc_auc": {
            "mean": round(float(roc.mean()), 4),
            "std": round(float(roc.std()), 4),
            "min": round(float(roc.min()), 4),
            "max": round(float(roc.max()), 4),
        },
        "loso_by_site": rows,
        "single_site_baseline_lacor_to_maricopa_roc": _single_site_baseline(data),
        "note": (
            "Validation leave-one-site-out : chaque site est prédit par un modèle "
            "entraîné UNIQUEMENT sur les autres sites. ROC-AUC moyen > 0.5 ⇒ le "
            "signal météo généralise (partiellement) entre sites réels. Cible binaire "
            "~10 %/site (quantile clients coupés pour les comtés ; is_outage réel pour "
            "Lacor). Modèle exogène (sans consommation) → ne remplace PAS le modèle "
            "hôpital complet de Lacor (qui exploite charge + auto-régression). "
            "Régénéré par src/models/multisite_experiment.py."
        ),
    }
    out = MODELS_DIR / "multisite_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Résumé écrit : %s (ROC-AUC LOSO moyen=%.3f)", out.name, summary["loso_roc_auc"]["mean"])


if __name__ == "__main__":
    from src.utils.io import setup_logging
    setup_logging()
    run()
