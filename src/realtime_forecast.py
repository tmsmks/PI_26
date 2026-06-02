"""
Prévision de coupure EN TEMPS RÉEL pilotée par Electricity Maps.

Idée
────
On ne dispose pas de la consommation interne live d'un hôpital, mais on a,
via Electricity Maps, la **charge du réseau électrique régional** (MW) auquel
il est raccordé, en temps réel. La courbe de charge réseau des dernières 24 h
sert de **proxy de forme** pour estimer la consommation de l'hôpital
(mise à l'échelle par son `avg_load_kw`), puis on y ajoute la **météo récente**
locale (Open-Meteo, sans clé). On reconstruit les features nowcast (comme le pipeline Lacor) et on applique
le **même modèle calibré** que l'onglet Analyse historique.

⚠️ Honnêteté : la consommation est ESTIMÉE depuis la charge réseau (pas un
compteur interne) et le modèle est entraîné sur Lacor. Le résultat est un
**risque régional indicatif**, pas une mesure validée pour le site.

Pré-requis : variable d'env `ELECTRICITY_MAPS_TOKEN`.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import requests

from src.data.ingest_electricitymaps import (
    EP_TOTAL_LOAD_HISTORY,
    _call,
    resolve_zone,
)
from src.features.build_features import apply_feature_engineering_single
from src.utils.config import (
    ELECTRICITYMAPS_TOKEN_ENV,
    METEO_FORECAST_BASE,
    METEO_HOURLY_VARS,
)
from src.utils.hospitals import HOSPITAL_DISPLAY

logger = logging.getLogger(__name__)


def fetch_grid_load_24h(zone: str, token: str) -> pd.DataFrame:
    """24 h glissantes de charge réseau (MW). Gère le schéma API courant où la
    valeur est sous la clé `value` (et non `totalLoad`)."""
    payload = _call(EP_TOTAL_LOAD_HISTORY, token, {"zone": zone})
    if not payload:
        return pd.DataFrame()
    items = payload.get("history") or payload.get("data") or []
    rows = []
    for it in items:
        ts = it.get("datetime")
        val = it.get("value", it.get("totalLoad", it.get("total_load")))
        if ts is None or val is None:
            continue
        rows.append({"datetime": ts, "em_total_load_mw": float(val)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    return df.sort_values("datetime").reset_index(drop=True)


def fetch_recent_weather(lat: float, lon: float) -> pd.DataFrame:
    """Météo horaire récente (24-48 h passées + jour courant) via Open-Meteo
    Forecast (sans clé). Renvoie un DataFrame [datetime, <vars météo>]."""
    try:
        resp = requests.get(
            METEO_FORECAST_BASE,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": ",".join(METEO_HOURLY_VARS),
                "past_days": 2,
                "forecast_days": 1,
                "timezone": "UTC",
            },
            timeout=30,
        )
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Météo temps réel indisponible (%s)", exc)
        return pd.DataFrame()
    hourly = data.get("hourly", {})
    if not hourly.get("time"):
        return pd.DataFrame()
    df = pd.DataFrame(hourly).rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def build_realtime_window(hospital_key: str, token: str | None = None) -> dict | None:
    """Construit la fenêtre 24 h temps réel (conso estimée + météo) et ses
    features. Renvoie {zone, features, raw} ou None si la charge réseau est
    indisponible (impossible d'estimer la consommation)."""
    info = HOSPITAL_DISPLAY.get(hospital_key) or {}
    lat, lon = info.get("lat"), info.get("lon")
    token = token or os.environ.get(ELECTRICITYMAPS_TOKEN_ENV)

    zone = None
    grid = pd.DataFrame()
    if token and lat is not None and lon is not None:
        zone = resolve_zone(hospital_key, token, lat, lon)
        if zone:
            grid = fetch_grid_load_24h(zone, token)
    if grid.empty:
        return None

    df = grid.copy()
    wx = fetch_recent_weather(lat, lon) if lat is not None else pd.DataFrame()
    if not wx.empty:
        df = pd.merge_asof(
            df.sort_values("datetime"),
            wx.sort_values("datetime"),
            on="datetime",
            direction="nearest",
            tolerance=pd.Timedelta("1h"),
        )

    # Consommation ESTIMÉE depuis la FORME de la charge réseau, normalisée à
    # l'échelle de Lacor (≈133 kW). Le modèle est entraîné sur l'échelle
    # absolue de Lacor ; mettre la conso d'un grand hôpital (1000-5000 kW)
    # ferait saturer les arbres au maximum. Comme la conso n'est ici qu'un
    # proxy de FORME (charge réseau / sa moyenne 24 h), on la recentre sur la
    # distribution d'entraînement → la prédiction reflète la dynamique réseau
    # + la météo, pas la taille absolue du site.
    LACOR_REF_AVG = 133.0
    mean_load = float(df["em_total_load_mw"].mean()) or 1.0
    ratio = df["em_total_load_mw"] / mean_load
    df["total_load_kw"] = LACOR_REF_AVG * ratio
    avg = LACOR_REF_AVG
    df["base_load_kw"] = df["total_load_kw"] * 0.85
    df["sterilization_kw"] = df["total_load_kw"] * 0.06
    if info.get("has_solar") and "shortwave_radiation" in df.columns:
        df["solar_pv_kw"] = (df["shortwave_radiation"].clip(lower=0) / 1000.0) * avg * 0.5
    else:
        df["solar_pv_kw"] = 0.0
    df["generators_kw"] = 0.0

    feats = apply_feature_engineering_single(
        df.drop(columns=["em_total_load_mw"], errors="ignore")
    )
    return {"zone": zone, "features": feats, "raw": df}


def realtime_forecast(
    hospital_key: str,
    feature_cols: list[str],
    predict_proba,
    horizon_models: dict | None = None,
    token: str | None = None,
) -> dict | None:
    """Prévision temps réel avec les modèles Lacor (horizons ou repli nowcast).

    Renvoie {zone, probs:{h:p}, window, features} ou None si charge réseau
    indisponible."""
    from src.nowcast_horizons import predict_horizons_realtime

    built = build_realtime_window(hospital_key, token)
    if built is None:
        return None
    probs = predict_horizons_realtime(
        built["raw"],
        hospital_key,
        feature_cols,
        predict_proba,
        horizon_models=horizon_models,
    )
    if not probs:
        last = built["features"].tail(1)
        X = last.reindex(columns=feature_cols).fillna(0.0)
        p_now = float(np.clip(predict_proba(X)[0], 0.0, 1.0))
        probs = {h: p_now for h in (1, 3, 6)}
    return {
        "zone": built["zone"],
        "probs": dict(sorted(probs.items())),
        "window": built["raw"],
        "features": built["features"],
    }
