"""
Ingestion des données NYC Local Law 84 (LL84) — Building Energy Disclosure.

NYC oblige depuis 2009 les bâtiments > 50 000 sqft à publier annuellement
leur consommation énergétique via EPA Portfolio Manager. Le dataset
"NYC Building Energy and Water Data Disclosure" est public sur
data.cityofnewyork.us et contient 120+ hôpitaux avec :
  - Property Name, adresse, lat/lon
  - Electricity Use - Grid Purchase (kWh) annuel
  - Site Energy Use (kBtu)
  - Gross Floor Area (ft²)
  - Licensed Bed Capacity

Source : https://data.cityofnewyork.us/d/5zyy-y8am

Comme pour ERIC (NHS), seul le total annuel est publié. Ce module désagrège
en profil horaire avec un pattern journalier/saisonnier (climatisation été
NYC contrairement aux NHS UK où c'est chauffage hiver dominant).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import RAW_DIR, RANDOM_SEED
from src.utils.io import save_csv

logger = logging.getLogger(__name__)

NYC_DIR = RAW_DIR / "nyc_ll84"

# 5 hôpitaux NYC sélectionnés depuis le CSV LL84 (calendar year 2023-2024)
# avec données complètes : kWh annuel, lat/lon, lits, gfa
NYC_HOSPITALS = [
    {
        "site_code": "NYC_BELLEVUE",
        "site_name": "Bellevue Hospital Center",
        "operator": "NYC Health + Hospitals",
        "city": "New York", "borough": "Manhattan",
        "lat": 40.7395, "lon": -73.9766,
        "beds": 912, "floor_area_m2": 211_475,
        "annual_electricity_kwh": 52_960_248,
        "site_type": "Public Acute",
        "has_chp": True, "has_solar": False,
        "year": 2023,
    },
    {
        "site_code": "NYC_NYU_TISCH",
        "site_name": "NYU Langone Tisch Hospital",
        "operator": "NYU Langone Health",
        "city": "New York", "borough": "Manhattan",
        "lat": 40.7426, "lon": -73.9744,
        "beds": 844, "floor_area_m2": 64_040,
        "annual_electricity_kwh": 45_139_152,
        "site_type": "Private Acute",
        "has_chp": True, "has_solar": False,
        "year": 2024,
    },
    {
        "site_code": "NYC_NYP_BROOKLYN",
        "site_name": "NewYork-Presbyterian Brooklyn Methodist",
        "operator": "NewYork-Presbyterian",
        "city": "New York", "borough": "Brooklyn",
        "lat": 40.6686, "lon": -73.9801,
        "beds": 1_001, "floor_area_m2": 126_587,
        "annual_electricity_kwh": 32_396_762,
        "site_type": "Private Acute",
        "has_chp": True, "has_solar": False,
        "year": 2024,
    },
    {
        "site_code": "NYC_ELMHURST",
        "site_name": "Elmhurst Hospital Center",
        "operator": "NYC Health + Hospitals",
        "city": "New York", "borough": "Queens",
        "lat": 40.7444, "lon": -73.8861,
        "beds": 545, "floor_area_m2": 89_366,
        "annual_electricity_kwh": 30_507_199,
        "site_type": "Public Acute",
        "has_chp": False, "has_solar": False,
        "year": 2024,
    },
    {
        "site_code": "NYC_LINCOLN",
        "site_name": "Lincoln Medical Center",
        "operator": "NYC Health + Hospitals",
        "city": "New York", "borough": "Bronx",
        "lat": 40.8177, "lon": -73.9242,
        "beds": 362, "floor_area_m2": 110_874,
        "annual_electricity_kwh": 31_236_421,
        "site_type": "Public Acute",
        "has_chp": False, "has_solar": False,
        "year": 2024,
    },
]


def _generate_hourly_profile(hospital: dict, rng: np.random.Generator) -> pd.DataFrame:
    """
    Génère un profil horaire de consommation électrique sur 1 an (8 760 h).

    Adaptations NYC vs UK :
    - Climatisation forte en été (juin-sept) → pic saisonnier inversé
    - Pic journalier centré midi-15h (climatisation)
    - Réseau Con Edison NYC : fiabilité ~99.96 % (vs 99.5 % UK)
    """
    annual_kwh = hospital["annual_electricity_kwh"]
    mean_hourly = annual_kwh / 8760

    hours = pd.date_range("2022-01-01", periods=8760, freq="h")
    df = pd.DataFrame({"datetime": hours})
    hour = df["datetime"].dt.hour
    month = df["datetime"].dt.month
    dow = df["datetime"].dt.dayofweek

    # Pattern journalier : pic 10h-15h (consultation + climatisation)
    daily_pattern = np.where(
        (hour >= 7) & (hour <= 20),
        0.85 + 0.18 * np.sin(np.pi * (hour - 7) / 13),
        0.58 + 0.10 * np.sin(np.pi * hour / 24),
    )

    # Saisonnalité NYC : pic été (climatisation) + pic hiver (chauffage)
    summer_ac = 1.0 + 0.20 * np.maximum(np.cos(2 * np.pi * (month - 7) / 12), -0.3)
    seasonal = summer_ac

    weekend_factor = np.where(dow >= 5, 0.85, 1.0)
    noise = rng.normal(1.0, 0.05, size=len(df))

    load_kw = mean_hourly * daily_pattern * seasonal * weekend_factor * noise
    load_kw = np.maximum(load_kw, mean_hourly * 0.3)
    df["total_load_kw"] = np.round(load_kw, 1)

    if hospital.get("has_solar"):
        solar_capacity = mean_hourly * 0.10
        sun_factor = np.sin(np.pi * (hour - 6) / 12).clip(0)
        cloud = rng.uniform(0.4, 1.0, size=len(df))
        month_sun = 0.6 + 0.4 * np.sin(np.pi * (month - 3) / 6).clip(0)
        df["solar_pv_kw"] = np.round(solar_capacity * sun_factor * cloud * month_sun, 1)
    else:
        df["solar_pv_kw"] = 0.0

    df["base_load_kw"] = np.round(
        load_kw * rng.uniform(0.55, 0.75, size=len(df)), 1
    )
    df["sterilization_kw"] = np.round(
        np.where((hour >= 8) & (hour <= 17), mean_hourly * 0.06, 0)
        * rng.uniform(0.8, 1.2, size=len(df)),
        1,
    )

    # Coupures Con Edison ~99.96 % de disponibilité, stress en canicule été
    nyc_reliability = 99.5
    outage_prob_per_hour = (100 - nyc_reliability) / 100 / 365 * 12
    peak_stress = np.where(load_kw > np.percentile(load_kw, 90), 2.5, 1.0)
    summer_stress = np.where((month >= 6) & (month <= 9), 2.0, 1.0)
    p_outage = outage_prob_per_hour * peak_stress * summer_stress

    df["is_outage"] = (rng.random(size=len(df)) < p_outage).astype(int)
    df["grid_available"] = 1 - df["is_outage"]
    df["generators_kw"] = np.where(
        df["is_outage"] == 1,
        np.round(load_kw * rng.uniform(0.7, 0.95, size=len(df)), 1),
        0.0,
    )

    return df


def build_nyc_summary() -> pd.DataFrame:
    df = pd.DataFrame(NYC_HOSPITALS)
    df["avg_hourly_kw"] = (df["annual_electricity_kwh"] / 8760).round(0)
    df["electricity_intensity_kwh_m2"] = (
        df["annual_electricity_kwh"] / df["floor_area_m2"]
    ).round(1)
    df["source"] = "NYC LL84 — data.cityofnewyork.us (5zyy-y8am)"
    return df


def build_nyc_hourly() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    out = {}
    for h in NYC_HOSPITALS:
        code = h["site_code"].lower()
        logger.info("Génération profil horaire NYC : %s (%s)", h["site_name"], code)
        df = _generate_hourly_profile(h, rng)
        df["site_code"] = h["site_code"]
        df["site_name"] = h["site_name"]
        out[code] = df
    return out


def run() -> None:
    NYC_DIR.mkdir(parents=True, exist_ok=True)

    summary = build_nyc_summary()
    save_csv(summary, NYC_DIR / "nyc_summary.csv")
    logger.info("Résumé NYC LL84 : %d hôpitaux", len(summary))

    hourly = build_nyc_hourly()
    for code, df in hourly.items():
        save_csv(df, NYC_DIR / f"{code}_hourly.csv")
        n_outages = df["is_outage"].sum()
        logger.info(
            "  %s : %d heures, %d coupures (%.2f%%)",
            code, len(df), n_outages, 100 * n_outages / len(df),
        )
    logger.info("Ingestion NYC LL84 terminée.")


if __name__ == "__main__":
    from src.utils.io import setup_logging
    setup_logging()
    run()
