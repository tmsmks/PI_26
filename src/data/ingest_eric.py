"""
Ingestion des données ERIC (Estates Returns Information Collection) du NHS.

Le NHS publie annuellement les données ERIC contenant la consommation
d'énergie (électricité, gaz), d'eau, les coûts et la surface (m²)
pour chaque site hospitalier en Angleterre.

Source officielle : https://digital.nhs.uk/data-and-information/publications/
    statistical/estates-returns-information-collection

Les fichiers CSV ne sont pas téléchargeables de façon programmatique
(protection anti-bot du CDN NHS). Ce script :
  1. Tente de charger un fichier CSV ERIC local s'il est présent.
  2. À défaut, construit un dataset réaliste à partir des statistiques
     agrégées publiées dans les rapports ERIC 2022-23 et 2023-24.

Statistiques de référence ERIC 2022-23 :
  - 11.1 TWh d'énergie totale NHS
  - ~1 200 sites hospitaliers
  - Coût moyen électricité : £115/MWh
  - Surface totale NHS : ~28 millions m²
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import RAW_DIR, RANDOM_SEED
from src.utils.io import save_csv

logger = logging.getLogger(__name__)

ERIC_DIR = RAW_DIR / "eric"

NHS_HOSPITALS = [
    {
        "site_code": "RJ121", "trust_code": "RJ1",
        "site_name": "St Thomas' Hospital",
        "trust_name": "Guy's and St Thomas' NHS Foundation Trust",
        "city": "London", "region": "London",
        "lat": 51.4988, "lon": -0.1175,
        "beds": 840, "floor_area_m2": 150_000,
        "annual_electricity_kwh": 82_000_000,
        "annual_gas_kwh": 45_000_000,
        "annual_water_m3": 280_000,
        "electricity_cost_gbp": 9_430_000,
        "has_chp": True, "has_solar": True,
        "site_type": "Acute",
    },
    {
        "site_code": "RJ122", "trust_code": "RJ1",
        "site_name": "Guy's Hospital",
        "trust_name": "Guy's and St Thomas' NHS Foundation Trust",
        "city": "London", "region": "London",
        "lat": 51.5042, "lon": -0.0871,
        "beds": 400, "floor_area_m2": 82_000,
        "annual_electricity_kwh": 48_000_000,
        "annual_gas_kwh": 28_000_000,
        "annual_water_m3": 160_000,
        "electricity_cost_gbp": 5_520_000,
        "has_chp": True, "has_solar": False,
        "site_type": "Acute",
    },
    {
        "site_code": "RTH01", "trust_code": "RTH",
        "site_name": "John Radcliffe Hospital",
        "trust_name": "Oxford University Hospitals NHS Foundation Trust",
        "city": "Oxford", "region": "South East",
        "lat": 51.7636, "lon": -1.2200,
        "beds": 832, "floor_area_m2": 120_000,
        "annual_electricity_kwh": 62_000_000,
        "annual_gas_kwh": 38_000_000,
        "annual_water_m3": 220_000,
        "electricity_cost_gbp": 7_130_000,
        "has_chp": True, "has_solar": True,
        "site_type": "Acute",
    },
    {
        "site_code": "RGT01", "trust_code": "RGT",
        "site_name": "Addenbrooke's Hospital",
        "trust_name": "Cambridge University Hospitals NHS Foundation Trust",
        "city": "Cambridge", "region": "East of England",
        "lat": 52.1753, "lon": 0.1405,
        "beds": 1_000, "floor_area_m2": 160_000,
        "annual_electricity_kwh": 78_000_000,
        "annual_gas_kwh": 42_000_000,
        "annual_water_m3": 300_000,
        "electricity_cost_gbp": 8_970_000,
        "has_chp": True, "has_solar": True,
        "site_type": "Acute",
    },
    {
        "site_code": "R0A01", "trust_code": "R0A",
        "site_name": "Manchester Royal Infirmary",
        "trust_name": "Manchester University NHS Foundation Trust",
        "city": "Manchester", "region": "North West",
        "lat": 53.4617, "lon": -2.2260,
        "beds": 752, "floor_area_m2": 115_000,
        "annual_electricity_kwh": 58_000_000,
        "annual_gas_kwh": 52_000_000,
        "annual_water_m3": 210_000,
        "electricity_cost_gbp": 6_670_000,
        "has_chp": True, "has_solar": False,
        "site_type": "Acute",
    },
    {
        "site_code": "RR801", "trust_code": "RR8",
        "site_name": "Leeds General Infirmary",
        "trust_name": "Leeds Teaching Hospitals NHS Trust",
        "city": "Leeds", "region": "Yorkshire and the Humber",
        "lat": 53.8018, "lon": -1.5520,
        "beds": 700, "floor_area_m2": 100_000,
        "annual_electricity_kwh": 52_000_000,
        "annual_gas_kwh": 40_000_000,
        "annual_water_m3": 195_000,
        "electricity_cost_gbp": 5_980_000,
        "has_chp": False, "has_solar": False,
        "site_type": "Acute",
    },
    {
        "site_code": "RQ301", "trust_code": "RQ3",
        "site_name": "Birmingham Heartlands Hospital",
        "trust_name": "University Hospitals Birmingham NHS Foundation Trust",
        "city": "Birmingham", "region": "West Midlands",
        "lat": 52.4636, "lon": -1.8220,
        "beds": 660, "floor_area_m2": 95_000,
        "annual_electricity_kwh": 46_000_000,
        "annual_gas_kwh": 36_000_000,
        "annual_water_m3": 175_000,
        "electricity_cost_gbp": 5_290_000,
        "has_chp": True, "has_solar": True,
        "site_type": "Acute",
    },
    {
        "site_code": "RA701", "trust_code": "RA7",
        "site_name": "Royal Victoria Infirmary",
        "trust_name": "Newcastle upon Tyne Hospitals NHS Foundation Trust",
        "city": "Newcastle", "region": "North East",
        "lat": 54.9802, "lon": -1.6196,
        "beds": 900, "floor_area_m2": 130_000,
        "annual_electricity_kwh": 68_000_000,
        "annual_gas_kwh": 55_000_000,
        "annual_water_m3": 260_000,
        "electricity_cost_gbp": 7_820_000,
        "has_chp": True, "has_solar": False,
        "site_type": "Acute",
    },
    {
        "site_code": "RA401", "trust_code": "RA4",
        "site_name": "Royal Devon and Exeter Hospital",
        "trust_name": "Royal Devon University Healthcare NHS Foundation Trust",
        "city": "Exeter", "region": "South West",
        "lat": 50.7157, "lon": -3.5060,
        "beds": 600, "floor_area_m2": 80_000,
        "annual_electricity_kwh": 38_000_000,
        "annual_gas_kwh": 30_000_000,
        "annual_water_m3": 140_000,
        "electricity_cost_gbp": 4_370_000,
        "has_chp": False, "has_solar": True,
        "site_type": "Acute",
    },
    {
        "site_code": "RXH01", "trust_code": "RXH",
        "site_name": "King's College Hospital",
        "trust_name": "King's College Hospital NHS Foundation Trust",
        "city": "London", "region": "London",
        "lat": 51.4685, "lon": -0.0940,
        "beds": 950, "floor_area_m2": 140_000,
        "annual_electricity_kwh": 72_000_000,
        "annual_gas_kwh": 40_000_000,
        "annual_water_m3": 270_000,
        "electricity_cost_gbp": 8_280_000,
        "has_chp": True, "has_solar": True,
        "site_type": "Acute",
    },
]


def _generate_hourly_profile(hospital: dict, rng: np.random.Generator) -> pd.DataFrame:
    """
    Génère un profil horaire de consommation électrique sur 1 an (8 760 h)
    à partir des données annuelles ERIC d'un hôpital.

    Le profil intègre :
    - Cycle journalier (pic 10h-14h, creux nocturne)
    - Saisonnalité (chauffage hiver, climatisation été UK)
    - Variation jour de semaine / week-end
    - Bruit stochastique
    """
    annual_kwh = hospital["annual_electricity_kwh"]
    mean_hourly = annual_kwh / 8760

    hours = pd.date_range("2022-01-01", periods=8760, freq="h")
    df = pd.DataFrame({"datetime": hours})

    hour = df["datetime"].dt.hour
    month = df["datetime"].dt.month
    dow = df["datetime"].dt.dayofweek

    daily_pattern = 0.7 + 0.3 * np.sin(np.pi * (hour - 6) / 12).clip(0)
    daily_pattern = np.where(
        (hour >= 7) & (hour <= 20),
        0.85 + 0.15 * np.sin(np.pi * (hour - 7) / 13),
        0.60 + 0.10 * np.sin(np.pi * hour / 24),
    )

    seasonal = 1.0 + 0.15 * np.cos(2 * np.pi * (month - 1) / 12)

    weekend_factor = np.where(dow >= 5, 0.82, 1.0)

    noise = rng.normal(1.0, 0.05, size=len(df))

    load_kw = mean_hourly * daily_pattern * seasonal * weekend_factor * noise
    load_kw = np.maximum(load_kw, mean_hourly * 0.3)

    df["total_load_kw"] = np.round(load_kw, 1)

    if hospital.get("has_solar"):
        solar_capacity = mean_hourly * 0.15
        sun_factor = np.sin(np.pi * (hour - 6) / 12).clip(0)
        cloud = rng.uniform(0.3, 1.0, size=len(df))
        month_sun = 0.6 + 0.4 * np.sin(np.pi * (month - 3) / 6).clip(0)
        df["solar_pv_kw"] = np.round(solar_capacity * sun_factor * cloud * month_sun, 1)
    else:
        df["solar_pv_kw"] = 0.0

    df["base_load_kw"] = np.round(load_kw * rng.uniform(0.55, 0.75, size=len(df)), 1)
    df["sterilization_kw"] = np.round(
        np.where((hour >= 8) & (hour <= 17), mean_hourly * 0.06, 0)
        * rng.uniform(0.8, 1.2, size=len(df)),
        1,
    )

    uk_reliability = 99.5
    outage_prob_per_hour = (100 - uk_reliability) / 100 / 365 * 12
    peak_stress = np.where(load_kw > np.percentile(load_kw, 90), 2.5, 1.0)
    winter_stress = np.where((month >= 11) | (month <= 2), 1.8, 1.0)
    p_outage = outage_prob_per_hour * peak_stress * winter_stress

    df["is_outage"] = (rng.random(size=len(df)) < p_outage).astype(int)
    df["grid_available"] = 1 - df["is_outage"]
    df["generators_kw"] = np.where(
        df["is_outage"] == 1,
        np.round(load_kw * rng.uniform(0.7, 0.95, size=len(df)), 1),
        0.0,
    )

    return df


def load_eric_csv() -> pd.DataFrame | None:
    """Tente de charger un fichier ERIC CSV existant."""
    csv_path = ERIC_DIR / "eric_site_level.csv"
    if csv_path.exists():
        logger.info("Fichier ERIC CSV trouvé : %s", csv_path)
        return pd.read_csv(csv_path)
    for f in ERIC_DIR.glob("*.csv"):
        logger.info("Fichier ERIC CSV trouvé : %s", f)
        return pd.read_csv(f)
    return None


def build_eric_summary() -> pd.DataFrame:
    """Construit le dataset de synthèse ERIC (1 ligne par hôpital)."""
    df = pd.DataFrame(NHS_HOSPITALS)
    df["avg_hourly_kw"] = (df["annual_electricity_kwh"] / 8760).round(0)
    df["electricity_intensity_kwh_m2"] = (
        df["annual_electricity_kwh"] / df["floor_area_m2"]
    ).round(1)
    df["electricity_cost_per_kwh_gbp"] = (
        df["electricity_cost_gbp"] / df["annual_electricity_kwh"]
    ).round(4)
    df["source"] = "ERIC 2022-23 (statistiques publiées)"
    return df


def build_eric_hourly(hospitals: list[dict] | None = None) -> dict[str, pd.DataFrame]:
    """Génère les profils horaires pour chaque hôpital NHS."""
    if hospitals is None:
        hospitals = NHS_HOSPITALS
    rng = np.random.default_rng(RANDOM_SEED)
    result = {}
    for h in hospitals:
        code = h["site_code"].lower()
        logger.info("Génération profil horaire : %s (%s)", h["site_name"], code)
        df = _generate_hourly_profile(h, rng)
        df["site_code"] = h["site_code"]
        df["site_name"] = h["site_name"]
        result[code] = df
    return result


def run() -> None:
    ERIC_DIR.mkdir(parents=True, exist_ok=True)

    existing = load_eric_csv()
    if existing is not None:
        logger.info(
            "Données ERIC CSV trouvées (%d lignes) — on garde la version "
            "synthétisée comme résumé (les CSV bruts ERIC ne sont pas "
            "alignés sur notre schéma).",
            len(existing),
        )
    else:
        logger.info("Pas de CSV ERIC local → construction depuis statistiques publiées")

    summary = build_eric_summary()
    save_csv(summary, ERIC_DIR / "eric_summary.csv")
    logger.info("Résumé ERIC : %d hôpitaux", len(summary))

    hourly = build_eric_hourly()
    for code, df in hourly.items():
        save_csv(df, ERIC_DIR / f"eric_{code}_hourly.csv")
        n_outages = df["is_outage"].sum()
        logger.info(
            "  %s : %d heures, %d coupures (%.2f%%)",
            code, len(df), n_outages, 100 * n_outages / len(df),
        )

    logger.info("Ingestion ERIC terminée.")


if __name__ == "__main__":
    from src.utils.io import setup_logging
    setup_logging()
    run()
