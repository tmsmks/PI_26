"""
Ingestion des prévisions météo horaires via Open-Meteo Forecast API.

Contrairement à `ingest_meteo.py` qui récupère l'historique 2022 pour
l'entraînement, ce module fournit des **prévisions futures** (jusqu'à
16 jours) utilisées par l'application Streamlit pour prédire la
probabilité de coupure à J+1, J+2, etc.

Les variables demandées sont strictement identiques à celles de
l'historique (voir METEO_HOURLY_VARS), afin de garantir que le modèle
entraîné reçoive exactement les mêmes colonnes au moment de l'inférence.

Sortie : `data/raw/meteo_forecast_<hospital>.csv`
    datetime, temperature_2m, …, hospital, fetched_at
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
import requests

from src.utils.config import (
    HOSPITAL_LOCATIONS,
    METEO_FORECAST_BASE,
    METEO_FORECAST_DAYS,
    METEO_HOURLY_VARS,
    RAW_DIR,
)
from src.utils.http import http_get
from src.utils.io import save_csv

logger = logging.getLogger(__name__)


def fetch_forecast(
    lat: float,
    lon: float,
    forecast_days: int = METEO_FORECAST_DAYS,
) -> pd.DataFrame:
    """Récupère les prévisions horaires Open-Meteo pour un site."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(METEO_HOURLY_VARS),
        "forecast_days": forecast_days,
        "timezone": "auto",
    }
    logger.info(
        "Open-Meteo Forecast lat=%.2f lon=%.2f horizon=%d jours",
        lat, lon, forecast_days,
    )
    resp = http_get(METEO_FORECAST_BASE, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    if not hourly:
        raise RuntimeError("Open-Meteo : réponse sans section hourly")

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"time": "datetime"})
    return df


def run(forecast_days: int = METEO_FORECAST_DAYS) -> None:
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for name, coords in HOSPITAL_LOCATIONS.items():
        try:
            df = fetch_forecast(
                lat=coords["lat"],
                lon=coords["lon"],
                forecast_days=forecast_days,
            )
        except Exception as exc:
            logger.warning("Forecast %s : échec (%s)", name, exc)
            continue

        df["hospital"] = name
        df["fetched_at"] = fetched_at
        save_csv(df, RAW_DIR / f"meteo_forecast_{name}.csv")

    logger.info("Ingestion prévisions météo terminée (%d jours).", forecast_days)


if __name__ == "__main__":
    from src.utils.io import setup_logging
    setup_logging()
    run()
