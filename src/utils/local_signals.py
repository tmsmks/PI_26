"""
Granularité « locale » des signaux par hôpital.

Electricity Maps = zone réseau (ISO/pays), pas le site.
Ce module décrit ce qui est réellement local et où le charger.
"""

from __future__ import annotations

from pathlib import Path

from src.utils.config import RAW_DIR
from src.utils.hospitals import HOSPITAL_DISPLAY, get_target_source

# Niveaux de localité (du plus fin au plus grossier)
LOCALITY = {
    "building": "Bâtiment / compteur (profil horaire site)",
    "county": "Comté US (coupures EAGLE-I, clients sans courant)",
    "point_meteo": "Point météo (lat/lon Open-Meteo)",
    "grid_zone": "Zone EM au point GPS (panneau risque live)",
    "none": "Non disponible (profil cloné)",
}


def eaglei_county_key(hospital_key: str, info: dict | None = None) -> str | None:
    """Clé `eaglei_<key>.csv` si l'hôpital est rattaché à un comté EAGLE-I."""
    if info is None:
        info = HOSPITAL_DISPLAY.get(hospital_key, {})
    return info.get("eaglei_county_key")


def eaglei_path(hospital_key: str, info: dict | None = None) -> Path | None:
    key = eaglei_county_key(hospital_key, info)
    if not key:
        return None
    return RAW_DIR / f"eaglei_{key}.csv"


def local_signal_profile(hospital_key: str, info: dict | None = None) -> dict:
    """Résumé honnête des signaux locaux exploitables pour un site."""
    if info is None:
        info = HOSPITAL_DISPLAY.get(hospital_key, {})

    ds = info.get("data_source")
    tsrc = get_target_source(hospital_key, info)

    consumption = LOCALITY["none"]
    outage = LOCALITY["none"]
    if hospital_key == "lacor_uganda":
        consumption = LOCALITY["building"]
        outage = LOCALITY["building"]
    elif ds == "eric":
        consumption = LOCALITY["building"]
        outage = "Étiquettes simulées (pas terrain)"
    elif ds == "nyc_ll84":
        consumption = LOCALITY["building"]
        outage = LOCALITY["county"]
        if info.get("target_source") == "county_network":
            outage = (
                f"{LOCALITY['county']} (EAGLE-I si ingéré, sinon repli synthétique)"
            )
    elif ds == "africa_grid":
        consumption = "Profil cloné Lacor (pas de compteur public)"
        outage = LOCALITY["none"]

    return {
        "consumption": consumption,
        "outage": outage,
        "meteo": LOCALITY["point_meteo"],
        "grid_context": LOCALITY["grid_zone"],
        "target_source": tsrc,
        "eaglei_county_key": eaglei_county_key(hospital_key, info),
    }
