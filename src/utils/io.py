"""
Fonctions utilitaires d'entrée/sortie (lecture, sauvegarde, logs).
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    # force=True : si Streamlit / pytest / un autre module a déjà appelé
    # basicConfig, sans force=True notre format/level sont silencieusement
    # ignorés et les logs disparaissent.
    logging.basicConfig(
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        force=True,
    )


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    logger.info("Sauvegardé %s  (%d lignes, %d colonnes)", path.name, len(df), len(df.columns))


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, **kwargs)
    logger.info("Chargé %s  (%d lignes, %d colonnes)", path.name, len(df), len(df.columns))
    return df


# ── Tables volumineuses : parquet + CSV ─────────────────────────────
# Pour les artefacts intermédiaires lourds (features_dataset,
# hospital_merged) on écrit en parquet (10-30× plus rapide à lire,
# 5× plus compact) ET on conserve un CSV de rétrocompatibilité /
# inspection humaine.
#
# La lecture (`load_table`) tente parquet d'abord, fallback CSV si
# pyarrow indisponible ou si le parquet n'existe pas.

def _parquet_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".parquet")


def _has_parquet_engine() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401
            return True
        except ImportError:
            return False


def save_table(df: pd.DataFrame, csv_path: Path, index: bool = False) -> None:
    """Sauvegarde une table en parquet (si dispo) + CSV.

    Le parquet est le format primaire (chargement rapide) ; le CSV reste
    pour `head`, `diff`, partage, et compatibilité ascendante avec les
    consommateurs qui font encore `pd.read_csv`.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=index)
    if _has_parquet_engine():
        pq = _parquet_path(csv_path)
        try:
            df.to_parquet(pq, index=index)
            logger.info(
                "Sauvegardé %s + %s (%d lignes, %d colonnes)",
                csv_path.name, pq.name, len(df), len(df.columns),
            )
            return
        except Exception as exc:
            logger.warning("Échec écriture parquet %s : %s", pq.name, exc)
    logger.info(
        "Sauvegardé %s (CSV seul, parquet indisponible) (%d lignes, %d colonnes)",
        csv_path.name, len(df), len(df.columns),
    )


def load_table(csv_path: Path, **csv_kwargs) -> pd.DataFrame:
    """Charge une table en privilégiant le parquet (10-30× plus rapide).

    Si un parquet à côté du CSV existe ET que pyarrow est dispo, on l'utilise.
    Sinon fallback `read_csv`. `csv_kwargs` ne s'applique qu'au fallback CSV.
    """
    pq = _parquet_path(csv_path)
    if pq.exists() and _has_parquet_engine():
        try:
            df = pd.read_parquet(pq)
            logger.info(
                "Chargé %s [parquet] (%d lignes, %d colonnes)",
                pq.name, len(df), len(df.columns),
            )
            return df
        except Exception as exc:
            logger.warning("Échec lecture parquet %s : %s — fallback CSV", pq.name, exc)
    return load_csv(csv_path, **csv_kwargs)
