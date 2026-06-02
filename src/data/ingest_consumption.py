"""
Ingestion du dataset principal Lacor Hospital (Ouganda).

  - Source  : Zenodo (doi:10.5281/zenodo.7466652)
  - Format  : 35 040 lignes × 7 colonnes, toutes les 15 min, année 2022
  - Colonnes clés :
      * Total load kW    → consommation totale
      * Grid avail       → 1 = réseau OK, 0 = coupure  ← VARIABLE CIBLE
      * Solar PV kW      → production solaire
      * Generators kW    → usage du générateur de secours
      * Base load kW     → charge de base
"""

import logging

import pandas as pd

from src.utils.config import LACOR_FILE, RAW_DIR
from src.utils.io import save_csv

logger = logging.getLogger(__name__)


def load_lacor() -> pd.DataFrame:
    """Charge et nettoie le dataset Lacor Hospital."""
    logger.info("Chargement Lacor Hospital : %s", LACOR_FILE)
    df = pd.read_excel(LACOR_FILE, sheet_name="Sheet1")

    df = df.rename(columns={
        "Unnamed: 0": "datetime",
        "Solar PV kW": "solar_pv_kw",
        "Total load kW": "total_load_kw",
        "Generators kW": "generators_kw",
        "Sterilization kW": "sterilization_kw",
        "Base load kW": "base_load_kw",
        "Grid avail": "grid_available",
    })
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Robustesse : `grid_available` est documenté comme binaire mais
    # certaines lignes Excel exportées peuvent contenir 0.5 (lecture
    # partielle au pas 15 min). On binarise explicitement (<0.5 = coupure).
    df["is_outage"] = (pd.to_numeric(df["grid_available"], errors="coerce") < 0.5).astype(int)

    logger.info(
        "Lacor : %d lignes, plage %s → %s, coupures=%d (%.1f%%)",
        len(df),
        df["datetime"].min().date(),
        df["datetime"].max().date(),
        df["is_outage"].sum(),
        100 * df["is_outage"].mean(),
    )
    return df


def run() -> None:
    lacor = load_lacor()
    save_csv(lacor, RAW_DIR / "lacor_clean.csv")
    logger.info("Ingestion consommation terminée.")


if __name__ == "__main__":
    from src.utils.io import setup_logging
    setup_logging()
    run()
