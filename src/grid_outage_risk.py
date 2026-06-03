"""
Risque de coupure indicatif : zone Electricity Maps (lat/lon hôpital) + météo récente.

⚠️ Ce n'est pas la probabilité du modèle ML Lacor : score heuristique 0–100 %
basé sur le stress du réseau local (charge, mix, carbone) et des facteurs météo
(chaleur, intempéries). La zone EM est la plus fine disponible via `/v4/zone`
au point GPS de l'hôpital — pas le bâtiment lui-même.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.data.ingest_electricitymaps import (
    _merge_signals,
    fetch_carbon_intensity_history,
    fetch_electricity_mix_history,
    fetch_total_load_history,
)
from src.realtime_forecast import fetch_recent_weather
from src.utils.config import ELECTRICITYMAPS_TOKEN_ENV, RAW_DIR
from src.utils.em_zone import resolve_zone_precise
from src.utils.hospitals import HOSPITAL_DISPLAY

logger = logging.getLogger(__name__)

def _load_meteo_tail(hospital_key: str, hours: int = 48) -> pd.DataFrame:
    path = RAW_DIR / f"meteo_{hospital_key}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.sort_values("datetime").tail(hours).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def fetch_grid_weather_context(
    hospital_key: str,
    token: str | None = None,
) -> dict[str, Any] | None:
    """Charge 24 h réseau EM + météo récente au point hôpital."""
    info = HOSPITAL_DISPLAY.get(hospital_key) or {}
    lat, lon = info.get("lat"), info.get("lon")
    if lat is None or lon is None:
        return None

    token = token or os.environ.get(ELECTRICITYMAPS_TOKEN_ENV)
    zone_meta = resolve_zone_precise(hospital_key, token=token)
    zone = zone_meta.get("zone")
    if not token or not zone:
        return None

    load_df = fetch_total_load_history(zone, token)
    carbon_df = fetch_carbon_intensity_history(zone, token)
    mix_df = fetch_electricity_mix_history(zone, token)
    grid = _merge_signals(load_df, carbon_df, mix_df)
    if grid.empty:
        return None

    grid["em_zone"] = zone
    grid["hospital"] = hospital_key

    wx = fetch_recent_weather(float(lat), float(lon))
    if wx.empty:
        wx = _load_meteo_tail(hospital_key, hours=48)

    return {
        "zone_meta": zone_meta,
        "grid": grid,
        "weather": wx,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _grid_stress_score(grid: pd.DataFrame) -> tuple[float, str]:
    if grid.empty or "em_total_load_mw" not in grid.columns:
        return 0.0, "Charge réseau indisponible"
    s = pd.to_numeric(grid["em_total_load_mw"], errors="coerce").dropna()
    if len(s) < 2:
        return 0.0, "Historique charge trop court"
    last = float(s.iloc[-1])
    p90 = float(s.quantile(0.9))
    mean = float(s.mean()) or 1.0
    ratio = last / max(p90, mean * 0.5)
    score = _clip01((ratio - 0.85) / 0.35)
    return score, f"Dernière charge {last:.0f} MW vs P90 {p90:.0f} MW (zone)"


def _carbon_stress_score(grid: pd.DataFrame) -> tuple[float, str]:
    col = "em_carbon_intensity_gco2_kwh"
    if col not in grid.columns:
        return 0.0, "Carbone indisponible"
    s = pd.to_numeric(grid[col], errors="coerce").dropna()
    if len(s) < 2:
        return 0.0, "Historique carbone trop court"
    last = float(s.iloc[-1])
    mean = float(s.mean())
    std = float(s.std()) or 1.0
    z = (last - mean) / std
    score = _clip01(z / 2.0)
    return score, f"Intensité carbone {last:.0f} gCO₂/kWh (z={z:+.1f})"


def _fossil_stress_score(grid: pd.DataFrame) -> tuple[float, str]:
    if "em_fossil_pct" in grid.columns:
        last = pd.to_numeric(grid["em_fossil_pct"], errors="coerce").dropna()
        if not last.empty:
            v = float(last.iloc[-1])
            return _clip01(v / 100.0), f"Part fossile réseau ≈ {v:.0f} %"
    if "em_renewable_pct" in grid.columns:
        last = pd.to_numeric(grid["em_renewable_pct"], errors="coerce").dropna()
        if not last.empty:
            v = float(last.iloc[-1])
            return _clip01(1.0 - v / 100.0), f"Part renouvelable réseau ≈ {v:.0f} %"
    return 0.0, "Mix production indisponible"


def _weather_stress_score(wx: pd.DataFrame, *, lat: float | None) -> tuple[float, str]:
    if wx.empty:
        return 0.0, "Météo récente indisponible"
    row = wx.sort_values("datetime").iloc[-1]
    parts: list[str] = []
    scores: list[float] = []

    temp = row.get("temperature_2m")
    if pd.notna(temp):
        t = float(temp)
        heat = _clip01((t - 28.0) / 10.0) if lat is None or lat > 0 else _clip01((t - 32.0) / 8.0)
        if heat > 0.2:
            scores.append(heat * 0.5)
            parts.append(f"T={t:.0f}°C")
        cold = _clip01((18.0 - t) / 15.0) if t < 18 else 0.0
        if cold > 0.3:
            scores.append(cold * 0.2)

    precip = float(row.get("precipitation", 0) or 0)
    wind = float(row.get("wind_speed_10m", 0) or 0)
    storm = _clip01((precip / 15.0) * min(wind / 25.0, 1.0))
    if storm > 0.15:
        scores.append(storm)
        parts.append(f"pluie {precip:.1f} mm, vent {wind:.0f} km/h")

    cape = row.get("cape")
    if pd.notna(cape) and float(cape) > 500:
        scores.append(_clip01(float(cape) / 3000.0) * 0.4)
        parts.append(f"CAPE {float(cape):.0f}")

    if not scores:
        return 0.0, "Conditions météo modérées au point hôpital"
    return _clip01(max(scores)), " · ".join(parts) if parts else "Stress météo"


def assess_outage_risk(
    context: dict[str, Any],
    hospital_key: str | None = None,
) -> dict[str, Any]:
    """Score composite 0–1 + facteurs explicites."""
    info = HOSPITAL_DISPLAY.get(hospital_key or "", {})
    grid = context.get("grid", pd.DataFrame())
    wx = context.get("weather", pd.DataFrame())
    zone_meta = context.get("zone_meta", {})

    g_load, d_load = _grid_stress_score(grid)
    g_carbon, d_carbon = _carbon_stress_score(grid)
    g_fossil, d_fossil = _fossil_stress_score(grid)
    g_wx, d_wx = _weather_stress_score(wx, lat=info.get("lat"))

    weights = {
        "grid_load": 0.35,
        "carbon": 0.15,
        "fossil": 0.15,
        "weather": 0.35,
    }
    score = (
        weights["grid_load"] * g_load
        + weights["carbon"] * g_carbon
        + weights["fossil"] * g_fossil
        + weights["weather"] * g_wx
    )
    score = _clip01(score)

    if score >= 0.65:
        level = "ÉLEVÉ"
    elif score >= 0.40:
        level = "MOYEN"
    else:
        level = "FAIBLE"

    return {
        "score": score,
        "score_pct": round(100 * score),
        "level": level,
        "zone": zone_meta.get("zone"),
        "zone_source": zone_meta.get("source"),
        "zone_country": zone_meta.get("country_name"),
        "factors": [
            {"key": "grid_load", "label": "Stress charge réseau (zone)", "weight": weights["grid_load"], "contrib": g_load, "detail": d_load},
            {"key": "carbon", "label": "Intensité carbone réseau", "weight": weights["carbon"], "contrib": g_carbon, "detail": d_carbon},
            {"key": "fossil", "label": "Dépendance fossile (mix)", "weight": weights["fossil"], "contrib": g_fossil, "detail": d_fossil},
            {"key": "weather", "label": "Stress météo (point hôpital)", "weight": weights["weather"], "contrib": g_wx, "detail": d_wx},
        ],
        "fetched_at": context.get("fetched_at"),
        "disclaimer": (
            "Indicateur contextuel (réseau Electricity Maps au point GPS + météo). "
            "Ce n'est pas la sortie du modèle Lacor ni une coupure observée à l'hôpital."
        ),
    }


def live_outage_risk(hospital_key: str, token: str | None = None) -> dict[str, Any] | None:
    """Contexte + évaluation pour l'app Streamlit."""
    ctx = fetch_grid_weather_context(hospital_key, token=token)
    if ctx is None:
        return None
    assessment = assess_outage_risk(ctx, hospital_key=hospital_key)
    return {**assessment, "grid": ctx["grid"], "weather": ctx["weather"], "zone_meta": ctx["zone_meta"]}
