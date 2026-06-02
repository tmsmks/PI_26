"""
Preprocessing : nettoyage, rééchantillonnage, fusion des datasets.

Stratégie :
  1. Dataset principal = Lacor Hospital (15 min, 2022)
     → rééchantillonné à l'heure pour aligner avec la météo
  2. Météo historique 2022 (Open-Meteo Archive, horaire) → jointure temporelle
  3. Contexte réseau Electricity Maps (`em_*`) → fusion par hôpital sur
     l'horodatage le plus proche (affiché comme contexte, exclu du modèle)

Le résultat est un DataFrame horaire unique prêt pour le feature engineering.
"""

import logging
from time import perf_counter

import numpy as np
import pandas as pd

from src.utils.config import RAW_DIR, PROCESSED_DIR
from src.utils.io import load_csv, save_csv, save_table

logger = logging.getLogger(__name__)

ERIC_CODE_TO_HOSPITAL = {
    "rj121": "st_thomas_nhs",
    "rj122": "guys_nhs",
    "rth01": "john_radcliffe_nhs",
    "rgt01": "addenbrookes_nhs",
    "r0a01": "manchester_nhs",
    "rr801": "leeds_general_nhs",
    "rq301": "birmingham_heartlands_nhs",
    "ra701": "newcastle_rvi_nhs",
    "ra401": "royal_devon_nhs",
    "rxh01": "kings_college_nhs",
}

NYC_CODE_TO_HOSPITAL = {
    "nyc_bellevue": "nyc_bellevue",
    "nyc_nyu_tisch": "nyc_nyu_tisch",
    "nyc_nyp_brooklyn": "nyc_nyp_brooklyn",
    "nyc_elmhurst": "nyc_elmhurst",
    "nyc_lincoln": "nyc_lincoln",
}


def resample_lacor_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rééchantillonne les données 15 min de Lacor en données horaires.
    Pour les variables continues → moyenne horaire.
    Pour is_outage → 1 si au moins une coupure dans l'heure.
    Pour grid_available → fraction de disponibilité dans l'heure.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")

    agg_rules = {
        "solar_pv_kw": "mean",
        "total_load_kw": "mean",
        "generators_kw": "mean",
        "sterilization_kw": "mean",
        "base_load_kw": "mean",
        "grid_available": "mean",
        "is_outage": "max",
    }

    hourly = df.resample("1h").agg(agg_rules)
    hourly = hourly.reset_index()

    # grid_available devient la fraction de l'heure avec réseau (0.0 à 1.0)
    hourly = hourly.rename(columns={"grid_available": "grid_availability_ratio"})

    logger.info(
        "Rééchantillonnage Lacor : %d → %d lignes (15 min → horaire)",
        len(df), len(hourly),
    )
    return hourly


def merge_with_meteo(consumption: pd.DataFrame, meteo: pd.DataFrame) -> pd.DataFrame:
    """Jointure temporelle consommation + météo sur l'heure la plus proche."""
    meteo = meteo.copy()
    meteo["datetime"] = pd.to_datetime(meteo["datetime"])
    meteo_cols = [c for c in meteo.columns if c not in ("datetime", "hospital")]
    meteo_subset = meteo[["datetime"] + meteo_cols].copy()

    # Convertir en numérique les colonnes qui le peuvent
    for col in meteo_cols:
        meteo_subset[col] = pd.to_numeric(meteo_subset[col], errors="coerce")

    merged = pd.merge_asof(
        consumption,
        meteo_subset.sort_values("datetime"),
        on="datetime",
        direction="nearest",
        tolerance=pd.Timedelta("1h"),
    )

    n_missing = merged[meteo_cols].isnull().sum().sum()
    if n_missing > 0:
        logger.warning("%d cellules météo sans correspondance → ffill causal", n_missing)
        # ffill (report de la dernière valeur connue) plutôt qu'interpolate/bfill
        # qui utilisent des valeurs FUTURES → fuite temporelle (#5). Les NaN de
        # début de série (aucune valeur passée) sont mis à 0, neutre et cohérent
        # avec le fillna(0) final de build_features.
        merged[meteo_cols] = merged[meteo_cols].ffill().fillna(0.0)

    logger.info("Fusion météo : %d colonnes ajoutées (%s)", len(meteo_cols), meteo_cols)
    return merged


def _merge_external_signal(
    df: pd.DataFrame,
    path: "pd.PathLike",
    column_prefix: str,
    label: str,
) -> pd.DataFrame:
    """Fusion générique d'une série externe horaire identifiée par préfixe.

    Utilisé pour Electricity Maps (`em_*`) : un CSV horaire dont les colonnes
    utiles commencent par `column_prefix`. Fusion sur `datetime` (tolérance
    1 h), manquants → 0.
    """
    if not path.exists():
        logger.info("Pas de données %s pour ce site — ignoré.", label)
        return df

    src = load_csv(path)
    src["datetime"] = pd.to_datetime(src["datetime"])
    cols = [c for c in src.columns if c.startswith(column_prefix)]
    if not cols:
        logger.warning("Fichier %s sans colonne %s* — ignoré.", path.name, column_prefix)
        return df

    merged = pd.merge_asof(
        df,
        src[["datetime"] + cols].sort_values("datetime"),
        on="datetime",
        direction="nearest",
        tolerance=pd.Timedelta("1h"),
    )
    merged[cols] = merged[cols].fillna(0)
    logger.info("Signal %-15s fusionné : %d colonnes ajoutées", label, len(cols))
    return merged


def add_electricitymaps_signal(df: pd.DataFrame, hospital: str) -> pd.DataFrame:
    """Fusionne les indicateurs Electricity Maps (préfixe `em_*`).

    Apporte le contexte « état réseau électrique local » :
      - charge totale (em_total_load_mw)        → stress du réseau
      - intensité carbone (em_carbon_intensity_gco2_kwh) → composition
      - mix renouvelable / fossile / bas-carbone → stabilité

    Ces variables sont la cause racine principale des coupures externes :
    quand le réseau de zone sature, le délestage tombe rapidement sur les
    hôpitaux (sauf priorisation explicite, rarement complète).
    """
    return _merge_external_signal(
        df,
        path=RAW_DIR / f"electricitymaps_{hospital}.csv",
        column_prefix="em_",
        label="ElectricityMaps",
    )


def load_hourly_hospital_bases() -> list[pd.DataFrame]:
    """Charge les séries horaires de base (avec cible `is_outage`) pour tous les hôpitaux."""
    frames: list[pd.DataFrame] = []

    # Lacor : 15 min -> horaire
    lacor = load_csv(RAW_DIR / "lacor_clean.csv")
    lacor_hourly = resample_lacor_hourly(lacor)
    lacor_hourly["hospital"] = "lacor_uganda"
    frames.append(lacor_hourly)

    # NHS ERIC : déjà horaire
    eric_dir = RAW_DIR / "eric"
    for code, hospital in ERIC_CODE_TO_HOSPITAL.items():
        path = eric_dir / f"eric_{code}_hourly.csv"
        if not path.exists():
            logger.info("Base ERIC absente pour %s (%s) — ignoré.", hospital, path.name)
            continue
        df = load_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["hospital"] = hospital
        frames.append(df)

    # NYC LL84 : déjà horaire
    nyc_dir = RAW_DIR / "nyc_ll84"
    for code, hospital in NYC_CODE_TO_HOSPITAL.items():
        path = nyc_dir / f"{code}_hourly.csv"
        if not path.exists():
            logger.info("Base NYC absente pour %s (%s) — ignoré.", hospital, path.name)
            continue
        df = load_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["hospital"] = hospital
        frames.append(df)

    return frames


def enrich_one_hospital(base_df: pd.DataFrame, hospital: str) -> pd.DataFrame:
    """Applique toutes les jointures de contexte externe pour un hôpital."""
    out = base_df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out = out.sort_values("datetime").reset_index(drop=True)

    meteo_path = RAW_DIR / f"meteo_{hospital}.csv"
    if meteo_path.exists():
        meteo = load_csv(meteo_path)
        out = merge_with_meteo(out, meteo)
    else:
        logger.info("Pas de météo pour %s — ignoré.", hospital)

    out = add_electricitymaps_signal(out, hospital=hospital)

    out["hour"] = out["datetime"].dt.hour
    out["day_of_week"] = out["datetime"].dt.dayofweek
    out["month"] = out["datetime"].dt.month
    out["hospital"] = hospital
    return out


def run() -> None:
    t0 = perf_counter()
    base_frames = load_hourly_hospital_bases()
    if not base_frames:
        raise FileNotFoundError("Aucune base horaire hospitalière trouvée.")

    enriched_frames: list[pd.DataFrame] = []
    for base in base_frames:
        loop_start = perf_counter()
        hospital = str(base["hospital"].iloc[0])
        logger.info("── Enrichissement hôpital : %s ──", hospital)
        enriched = enrich_one_hospital(base, hospital)

        numeric_cols = enriched.select_dtypes(include=[np.number]).columns
        # Ne JAMAIS imputer la cible : un NaN sur is_outage est un problème de
        # données à laisser visible, pas à combler (le combler fuiterait la
        # cible). On l'exclut donc de l'imputation.
        impute_cols = [c for c in numeric_cols if c != "is_outage"]
        n_missing = enriched[impute_cols].isnull().sum().sum()
        if n_missing > 0:
            logger.warning("%d valeurs manquantes pour %s → ffill causal", n_missing, hospital)
            # ffill causal au lieu d'interpolate/bfill (qui regardent le futur,
            # #5). NaN de début de série → 0 (cohérent avec build_features).
            enriched[impute_cols] = enriched[impute_cols].ffill().fillna(0.0)

        enriched_frames.append(enriched)
        logger.info("  ✓ %s enrichi en %.2fs", hospital, perf_counter() - loop_start)

    merged_all = pd.concat(enriched_frames, ignore_index=True)
    merged_all["datetime"] = pd.to_datetime(merged_all["datetime"])
    merged_all = merged_all.sort_values(["datetime", "hospital"]).reset_index(drop=True)

    save_table(merged_all, PROCESSED_DIR / "hospital_merged.csv")
    logger.info(
        "Preprocessing multi-hôpitaux terminé : %d lignes, %d colonnes, %d hôpitaux, coupures=%.2f%%",
        len(merged_all),
        len(merged_all.columns),
        merged_all["hospital"].nunique(),
        100 * merged_all["is_outage"].mean(),
    )
    logger.info("Temps total preprocessing : %.2fs", perf_counter() - t0)


if __name__ == "__main__":
    from src.utils.io import setup_logging
    setup_logging()
    run()
