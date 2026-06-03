"""
Ingestion des coupures réelles EAGLE-I (Oak Ridge National Laboratory).

EAGLE-I agrège les coupures d'électricité **réellement subies** aux USA :
pour chaque comté, le nombre de clients sans courant (`customers_out`) est
relevé toutes les 15 minutes. Un fichier `MCC.csv` (Max Customer Count)
donne le nombre maximal de clients par comté, ce qui permet de normaliser.

Pourquoi cette source
---------------------
Le modèle pilote n'a qu'**un seul site à vérité terrain** (Lacor, Ouganda) :
les coupures des autres hôpitaux du projet (NHS/NYC) sont synthétiques. EAGLE-I
apporte des coupures **réelles et indépendantes** sur 8 grands comtés urbains
américains (ceux où se trouvent des hôpitaux du catalogue). On les utilise pour
valider la **généralisation inter-sites** en leave-one-site-out
(cf. `src/models/multisite_experiment.py`), donc pour mesurer — au lieu de
supposer — la sur-spécialisation au contexte ougandais.

Honnêteté / reproductibilité
-----------------------------
Les fichiers EAGLE-I bruts (CSV annuels nationaux, ~centaines de Mo) ne sont
PAS versionnés dans le repo. Ce module est un *transformateur déterministe* :
si les bruts sont présents dans ``EAGLEI_SOURCE_DIR`` il produit
``data/raw/eaglei_<site>.csv`` ; sinon il journalise clairement comment les
obtenir (DOI figshare) et s'arrête proprement sans planter le pipeline — exactement
le même contrat que les ingestions réseau (Electricity Maps, EskomSePush).

Sortie par site (``data/raw/eaglei_<site>.csv``)
-----------------------------------------------
    datetime              horaire, année EAGLEI_YEAR complète
    customers_out         max de clients coupés sur l'heure (pire cas)
    customers_out_frac    customers_out / clients_max_du_comté ∈ [0, 1]
    is_outage             1 si customers_out > quantile p90 du site, sinon 0
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.config import (
    EAGLEI_DOWNLOAD_DOI,
    EAGLEI_OUTAGE_QUANTILE,
    EAGLEI_SOURCE_DIR,
    EAGLEI_YEAR,
    RAW_DIR,
)
from src.utils.io import save_csv

logger = logging.getLogger(__name__)

# ── Catalogue des comtés EAGLE-I (hôpitaux urbains américains) ───────
# `site` reprend EXACTEMENT le libellé attendu par l'expérience multi-sites
# (models/multisite_loso_by_site.csv). lat/lon servent à la météo Open-Meteo.
# fips est le code comté à 5 chiffres (filtre sur la colonne `fips_code`).
EAGLEI_COUNTIES: list[dict] = [
    {"key": "maricopa_az",  "site": "Maricopa/Phoenix AZ",      "fips": "04013", "lat": 33.45, "lon": -112.07},
    {"key": "kings_ny",     "site": "Kings/Brooklyn NY",        "fips": "36047", "lat": 40.65, "lon": -73.95},
    {"key": "harris_tx",    "site": "Harris/Houston TX",        "fips": "48201", "lat": 29.76, "lon": -95.37},
    {"key": "cook_il",      "site": "Cook/Chicago IL",          "fips": "17031", "lat": 41.88, "lon": -87.63},
    {"key": "miamidade_fl", "site": "Miami-Dade FL",            "fips": "12086", "lat": 25.76, "lon": -80.19},
    {"key": "losangeles_ca","site": "Los Angeles CA",           "fips": "06037", "lat": 34.05, "lon": -118.24},
    {"key": "king_wa",      "site": "King/Seattle WA",          "fips": "53033", "lat": 47.61, "lon": -122.33},
    {"key": "orleans_la",   "site": "Orleans/New Orleans LA",   "fips": "22071", "lat": 29.95, "lon": -90.07},
]


def output_path(county_key: str) -> Path:
    """Chemin du CSV horaire produit pour un comté."""
    return RAW_DIR / f"eaglei_{county_key}.csv"


def _normalize_fips(series: pd.Series) -> pd.Series:
    """Force un code FIPS comté sur 5 caractères ('4013' → '04013')."""
    return (
        series.astype("string")
        .str.replace(r"\.0$", "", regex=True)  # 4013.0 (lu en float) → 4013
        .str.strip()
        .str.zfill(5)
    )


def _find_source_file() -> Path | None:
    """Localise le CSV EAGLE-I de l'année cible dans EAGLEI_SOURCE_DIR."""
    candidates = [
        EAGLEI_SOURCE_DIR / f"eaglei_outages_{EAGLEI_YEAR}.csv",
        EAGLEI_SOURCE_DIR / f"eaglei_outages_{EAGLEI_YEAR}.csv.gz",
    ]
    for path in candidates:
        if path.exists():
            return path
    # Repli : tout CSV contenant l'année dans son nom.
    if EAGLEI_SOURCE_DIR.exists():
        for path in sorted(EAGLEI_SOURCE_DIR.glob(f"*{EAGLEI_YEAR}*.csv*")):
            return path
    return None


def _resolve_eaglei_columns(columns: pd.Index) -> tuple[str, str]:
    """Repère les colonnes temps / clients coupés selon le millésime EAGLE-I.

    - Anciens exports : ``customers_out`` + ``run_start_time``
    - 2023+ (figshare) : ``sum`` (= clients sans courant) + ``run_start_time``
    """
    cols = list(columns)
    time_col = next((c for c in cols if "time" in c.lower()), None)
    out_col = next((c for c in cols if "customers_out" in c.lower()), None)
    if out_col is None:
        out_col = next((c for c in cols if c.lower() == "sum"), None)
    if time_col is None or out_col is None:
        raise ValueError(
            f"Colonnes EAGLE-I attendues introuvables "
            f"(time={time_col}, clients_coupés={out_col}, colonnes={cols})"
        )
    return time_col, out_col


def _load_mcc() -> pd.Series | None:
    """Charge le Max Customer Count par comté (FIPS → nb clients max)."""
    mcc_path = EAGLEI_SOURCE_DIR / "MCC.csv"
    if not mcc_path.exists():
        return None
    mcc = pd.read_csv(mcc_path)
    # Tolérance sur les noms de colonnes selon les millésimes EAGLE-I.
    fips_col = next((c for c in mcc.columns if "fips" in c.lower()), None)
    cust_col = next(
        (c for c in mcc.columns if c.lower() in ("customers", "max_customers", "mcc")),
        None,
    )
    if fips_col is None or cust_col is None:
        logger.warning("MCC.csv : colonnes FIPS/clients introuvables — normalisation ignorée.")
        return None
    mcc[fips_col] = _normalize_fips(mcc[fips_col])
    return mcc.groupby(fips_col)[cust_col].max()


def to_hourly(
    raw: pd.DataFrame,
    *,
    fips: str,
    year: int,
    max_customers: float | None,
    quantile: float = EAGLEI_OUTAGE_QUANTILE,
) -> pd.DataFrame:
    """Transforme les snapshots 15 min d'un comté en série horaire + cible.

    Étapes déterministes :
      1. parse `run_start_time` ;
      2. agrège à l'heure par **max** de `customers_out` (pire cas de l'heure) ;
      3. réindexe sur l'année complète (heures sans relevé ⇒ 0 coupure) ;
      4. normalise par le max clients du comté (si MCC dispo) ;
      5. binarise : `is_outage = customers_out > quantile p90 du site`.
    """
    time_col, out_col = _resolve_eaglei_columns(raw.columns)

    df = raw[[time_col, out_col]].copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=False)
    df[out_col] = pd.to_numeric(df[out_col], errors="coerce").fillna(0)
    df = df.dropna(subset=[time_col])

    hourly = (
        df.set_index(time_col)[out_col]
        .resample("h")
        .max()
        .fillna(0)
    )

    full_index = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h")
    hourly = hourly.reindex(full_index, fill_value=0.0)

    result = pd.DataFrame({"datetime": full_index, "customers_out": hourly.to_numpy()})
    if max_customers and max_customers > 0:
        result["customers_out_frac"] = (result["customers_out"] / max_customers).clip(0, 1)
    else:
        # Pas de MCC : normalisation min-max sur la série (reste dans [0, 1]).
        peak = result["customers_out"].max()
        result["customers_out_frac"] = result["customers_out"] / peak if peak > 0 else 0.0

    threshold = result["customers_out"].quantile(quantile)
    # Si la distribution est trop plate (beaucoup de zéros), le quantile peut être 0 :
    # on exige alors un dépassement STRICT de 0 pour ne pas tout marquer en coupure.
    result["is_outage"] = (result["customers_out"] > max(threshold, 0)).astype(int)
    result["fips"] = fips
    return result


def run(quantile: float = EAGLEI_OUTAGE_QUANTILE) -> list[Path]:
    """Produit ``data/raw/eaglei_<site>.csv`` pour chaque comté du catalogue.

    Renvoie la liste des fichiers écrits (vide si les bruts EAGLE-I sont absents).
    """
    source = _find_source_file()
    if source is None:
        logger.warning(
            "EAGLE-I : aucun brut trouvé dans %s.\n"
            "  → Télécharger les coupures réelles (figshare, libre) : %s\n"
            "  → Déposer eaglei_outages_%d.csv (+ MCC.csv) dans ce dossier, puis relancer.\n"
            "  Étape multi-sites ignorée (le reste du pipeline n'est pas affecté).",
            EAGLEI_SOURCE_DIR, EAGLEI_DOWNLOAD_DOI, EAGLEI_YEAR,
        )
        return []

    logger.info("EAGLE-I : lecture de %s", source.name)
    fips_wanted = {c["fips"] for c in EAGLEI_COUNTIES}

    # Lecture par morceaux : le CSV national annuel peut dépasser la RAM.
    fips_col_guess = None
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(source, chunksize=1_000_000, low_memory=False):
        if fips_col_guess is None:
            fips_col_guess = next((c for c in chunk.columns if "fips" in c.lower()), None)
            if fips_col_guess is None:
                raise ValueError("Colonne FIPS introuvable dans le CSV EAGLE-I.")
        chunk = chunk.copy()
        chunk[fips_col_guess] = _normalize_fips(chunk[fips_col_guess])
        chunks.append(chunk[chunk[fips_col_guess].isin(fips_wanted)])

    filtered = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if filtered.empty:
        logger.warning("EAGLE-I : aucun des comtés ciblés n'est présent dans %s.", source.name)
        return []

    mcc = _load_mcc()
    written: list[Path] = []
    for county in EAGLEI_COUNTIES:
        sub = filtered[filtered[fips_col_guess] == county["fips"]]
        if sub.empty:
            logger.warning("EAGLE-I : comté %s (FIPS %s) absent — ignoré.", county["site"], county["fips"])
            continue
        max_customers = float(mcc.get(county["fips"])) if mcc is not None and county["fips"] in mcc.index else None
        hourly = to_hourly(
            sub, fips=county["fips"], year=EAGLEI_YEAR, max_customers=max_customers, quantile=quantile,
        )
        path = output_path(county["key"])
        save_csv(hourly, path)
        rate = hourly["is_outage"].mean()
        logger.info(
            "  %s : %d h, taux coupure %.2f%% (seuil p%d, MCC=%s)",
            county["site"], len(hourly), 100 * rate,
            int(quantile * 100), int(max_customers) if max_customers else "n/a",
        )
        written.append(path)

    logger.info("Ingestion EAGLE-I terminée : %d/%d comtés écrits.", len(written), len(EAGLEI_COUNTIES))
    return written


if __name__ == "__main__":
    from src.utils.io import setup_logging
    setup_logging()
    run()
