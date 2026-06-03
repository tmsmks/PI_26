"""
Application des coupures EAGLE-I (comté US) sur des profils horaires hospitaliers.

Les coupures sont celles du **réseau du comté** (clients sans courant), pas du
compteur de l'hôpital. Utilisé par l'ingestion NYC LL84 quand les CSV
``data/raw/eaglei_<county_key>.csv`` existent.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ingest_eaglei import output_path
from src.utils.config import EAGLEI_YEAR, RANDOM_SEED

logger = logging.getLogger(__name__)

OUTAGE_SOURCE_EAGLEI = "eaglei_county"
OUTAGE_SOURCE_SYNTHETIC = "synthetic_stochastic"
PROFILE_YEAR_DEFAULT = 2022


def align_eaglei_calendar(
    eaglei: pd.DataFrame,
    profile_year: int,
) -> pd.DataFrame:
    """Aligne l'année EAGLE-I sur le calendrier du profil (ex. 2023 → 2022)."""
    eaglei = eaglei.copy()
    eaglei["datetime"] = pd.to_datetime(eaglei["datetime"])
    eaglei_year = int(eaglei["datetime"].dt.year.mode().iloc[0])
    if eaglei_year != profile_year:
        logger.info(
            "EAGLE-I : réalignement calendrier %d → %d pour jointure avec le profil horaire.",
            eaglei_year,
            profile_year,
        )
        eaglei["datetime"] = eaglei["datetime"].apply(
            lambda t: t.replace(year=profile_year)
        )
    return eaglei.sort_values("datetime").reset_index(drop=True)


def load_county_hourly(county_key: str) -> pd.DataFrame | None:
    path = output_path(county_key)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception as exc:
        logger.warning("EAGLE-I %s illisible : %s", path.name, exc)
        return None


def apply_eaglei_outages(
    hourly: pd.DataFrame,
    county_key: str,
    *,
    profile_year: int = PROFILE_YEAR_DEFAULT,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, str]:
    """Remplace ``is_outage`` par la cible comté EAGLE-I si le fichier existe.

    Renvoie ``(dataframe, outage_source)`` avec ``outage_source`` ∈
    {``eaglei_county``, ``synthetic_stochastic``}.
    """
    hourly = hourly.copy()
    hourly["datetime"] = pd.to_datetime(hourly["datetime"])

    eaglei = load_county_hourly(county_key)
    if eaglei is None or eaglei.empty or "is_outage" not in eaglei.columns:
        hourly["outage_source"] = OUTAGE_SOURCE_SYNTHETIC
        hourly["outage_label_detail"] = "Coupures simulées (formule stochastique LL84)"
        return hourly, OUTAGE_SOURCE_SYNTHETIC

    eaglei = align_eaglei_calendar(eaglei, profile_year)
    cols = ["datetime", "is_outage"]
    if "customers_out_frac" in eaglei.columns:
        cols.append("customers_out_frac")
    if "customers_out" in eaglei.columns:
        cols.append("customers_out")

    merged = hourly.merge(
        eaglei[cols].rename(columns={
            "is_outage": "is_outage_eaglei",
            "customers_out_frac": "county_customers_out_frac",
            "customers_out": "county_customers_out",
        }),
        on="datetime",
        how="left",
    )

    n_match = int(merged["is_outage_eaglei"].notna().sum())
    if n_match == 0:
        logger.warning(
            "EAGLE-I %s : aucune heure commune avec le profil (%d–%d) — repli synthétique.",
            county_key,
            profile_year,
            EAGLEI_YEAR,
        )
        hourly["outage_source"] = OUTAGE_SOURCE_SYNTHETIC
        hourly["outage_label_detail"] = "Coupures simulées (EAGLE-I non aligné)"
        return hourly, OUTAGE_SOURCE_SYNTHETIC

    merged["is_outage"] = merged["is_outage_eaglei"].fillna(0).astype(int)
    merged["grid_available"] = 1 - merged["is_outage"]

    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    load = merged["total_load_kw"].to_numpy(dtype=float)
    merged["generators_kw"] = np.where(
        merged["is_outage"] == 1,
        np.round(load * rng.uniform(0.7, 0.95, size=len(merged)), 1),
        0.0,
    )

    merged["outage_source"] = OUTAGE_SOURCE_EAGLEI
    merged["outage_label_detail"] = (
        f"EAGLE-I comté `{county_key}` — clients sans courant (réseau comté, "
        f"pas compteur hôpital ; calendrier EAGLE-I {EAGLEI_YEAR} → {profile_year})"
    )
    merged = merged.drop(columns=["is_outage_eaglei"], errors="ignore")

    rate = 100 * float(merged["is_outage"].mean())
    logger.info(
        "  EAGLE-I %s : %d/%d h jointes, taux coupure comté %.2f%%",
        county_key, n_match, len(merged), rate,
    )
    return merged, OUTAGE_SOURCE_EAGLEI
