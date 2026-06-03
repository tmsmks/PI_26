"""
Catalogue centralisé des hôpitaux couverts par le projet.

`HOSPITAL_DISPLAY` est la SOURCE DE VÉRITÉ unique pour :
  - les métadonnées UI (nom, drapeau, type, lits, fiabilité OMS…)
  - les coordonnées géographiques (lat/lon/country)
  - la stratégie d'accès aux données (`data_source` ∈ {None, "eric",
    "nyc_ll84", "africa_grid"})

`HOSPITAL_LOCATIONS` (dans `config.py`) est désormais DÉRIVÉ de ce dict
pour les sites qui doivent être ingérés par les pipelines géolocalisés
(Open-Meteo, Electricity Maps, etc.).

Conventions :
  - country : code ISO3
  - data_source absent ⇒ site de référence (Lacor) ou ingéré avec un
    pipeline dédié
"""

from __future__ import annotations

HOSPITAL_DISPLAY: dict[str, dict] = {
    "lacor_uganda": {
        "name": "Lacor Hospital",
        "location": "Gulu, Ouganda",
        "flag": "🇺🇬",
        "beds": 482,
        "type": "Hôpital général (PNL)",
        "who_reliability": 50.0,
        "lat": 2.77, "lon": 32.30, "country": "UGA",
        "avg_load_kw": 133, "max_load_kw": 235,
        "has_solar": True, "has_generator": True,
        "grid_stability": "faible",
        "ingest_geo": True,
    },
    # ── Hôpitaux africains (réseau temps réel via Electricity Maps) ──
    # data_source = africa_grid : profil Lacor mis à l'échelle par
    # avg_load_kw et météo Open-Meteo locale (pas de compteur public).
    "kenyatta_kenya": {
        "name": "Kenyatta National Hospital",
        "location": "Nairobi, Kenya",
        "flag": "🇰🇪",
        "beds": 1800,
        "type": "Hôpital de référence national",
        "who_reliability": 65.0,
        "lat": -1.30, "lon": 36.81, "country": "KEN",
        "avg_load_kw": 1900, "max_load_kw": 2700,
        "has_solar": True, "has_generator": True,
        "grid_stability": "moyen",
        "data_source": "africa_grid",
        "ingest_geo": True,
    },
    "tikur_ethiopia": {
        "name": "Tikur Anbessa Specialized Hospital",
        "location": "Addis-Abeba, Éthiopie",
        "flag": "🇪🇹",
        "beds": 800,
        "type": "Hôpital universitaire",
        "who_reliability": 45.0,
        "lat": 9.01, "lon": 38.75, "country": "ETH",
        "avg_load_kw": 950, "max_load_kw": 1500,
        "has_solar": False, "has_generator": True,
        "grid_stability": "faible",
        "data_source": "africa_grid",
        "ingest_geo": True,
    },
    "groote_schuur_sa": {
        "name": "Groote Schuur Hospital",
        "location": "Le Cap, Afrique du Sud",
        "flag": "🇿🇦",
        "beds": 893,
        "type": "Hôpital universitaire (UCT)",
        "who_reliability": 88.0,
        "lat": -33.94, "lon": 18.46, "country": "ZAF",
        "avg_load_kw": 2400, "max_load_kw": 3300,
        "has_solar": True, "has_generator": True,
        "grid_stability": "instable (Eskom)",
        "data_source": "africa_grid",
        "ingest_geo": True,
    },
    "dhaka_bangladesh": {
        "name": "Dhaka Medical College Hospital",
        "location": "Dhaka, Bangladesh",
        "flag": "🇧🇩",
        "beds": 2600,
        "type": "Hôpital universitaire public",
        "who_reliability": 55.0,
        "lat": 23.73, "lon": 90.40, "country": "BGD",
        "avg_load_kw": 2200, "max_load_kw": 3200,
        "has_solar": False, "has_generator": True,
        "grid_stability": "moyen",
        "data_source": "africa_grid",
        "ingest_geo": True,
        # Préserve le comportement historique : ingestion oui (ancien
        # HOSPITAL_LOCATIONS l'incluait) mais site non affiché dans
        # l'UI Streamlit (ancien HOSPITAL_DISPLAY ne l'avait pas).
        "ui_hidden": True,
    },
    "fann_senegal": {
        "name": "CHU de Fann",
        "location": "Dakar, Sénégal",
        "flag": "🇸🇳",
        "beds": 600,
        "type": "Centre hospitalier universitaire",
        "who_reliability": 60.0,
        "lat": 14.69, "lon": -17.46, "country": "SEN",
        "avg_load_kw": 800, "max_load_kw": 1200,
        "has_solar": True, "has_generator": True,
        "grid_stability": "moyen",
        "data_source": "africa_grid",
        "ingest_geo": True,
    },
    "parirenyatwa_zimbabwe": {
        "name": "Parirenyatwa Group of Hospitals",
        "location": "Harare, Zimbabwe",
        "flag": "🇿🇼",
        "beds": 1800,
        "type": "Hôpital universitaire de référence",
        "who_reliability": 35.0,
        "lat": -17.79, "lon": 31.05, "country": "ZWE",
        "avg_load_kw": 1600, "max_load_kw": 2400,
        "has_solar": False, "has_generator": True,
        "grid_stability": "très faible",
        "data_source": "africa_grid",
        "ingest_geo": True,
    },
    "muhimbili_tanzania": {
        "name": "Muhimbili National Hospital",
        "location": "Dar es Salaam, Tanzanie",
        "flag": "🇹🇿",
        "beds": 1500,
        "type": "Hôpital national de référence",
        "who_reliability": 58.0,
        "lat": -6.80, "lon": 39.27, "country": "TZA",
        "avg_load_kw": 1700, "max_load_kw": 2500,
        "has_solar": True, "has_generator": True,
        "grid_stability": "moyen",
        "data_source": "africa_grid",
        "ingest_geo": True,
    },
    "luth_nigeria": {
        "name": "Lagos University Teaching Hospital (LUTH)",
        "location": "Lagos, Nigeria",
        "flag": "🇳🇬",
        "beds": 760,
        "type": "Hôpital universitaire",
        "who_reliability": 30.0,
        "lat": 6.515, "lon": 3.358, "country": "NGA",
        "avg_load_kw": 1400, "max_load_kw": 2200,
        "has_solar": True, "has_generator": True,
        "grid_stability": "très faible",
        "data_source": "africa_grid",
        "ingest_geo": True,
    },
    "korle_bu_ghana": {
        "name": "Korle Bu Teaching Hospital",
        "location": "Accra, Ghana",
        "flag": "🇬🇭",
        "beds": 2000,
        "type": "Hôpital universitaire",
        "who_reliability": 70.0,
        "lat": 5.535, "lon": -0.224, "country": "GHA",
        "avg_load_kw": 1800, "max_load_kw": 2700,
        "has_solar": False, "has_generator": True,
        "grid_stability": "faible",
        "data_source": "africa_grid",
        "ingest_geo": True,
    },
    "ibn_sina_morocco": {
        "name": "CHU Ibn Sina",
        "location": "Rabat, Maroc",
        "flag": "🇲🇦",
        "beds": 1100,
        "type": "Centre hospitalier universitaire",
        "who_reliability": 92.0,
        "lat": 34.005, "lon": -6.834, "country": "MAR",
        "avg_load_kw": 1500, "max_load_kw": 2200,
        "has_solar": True, "has_generator": True,
        "grid_stability": "stable",
        "data_source": "africa_grid",
        "ingest_geo": True,
    },
    "kasr_alainy_egypt": {
        "name": "Kasr Al Ainy Hospital (Cairo Univ.)",
        "location": "Le Caire, Égypte",
        "flag": "🇪🇬",
        "beds": 5500,
        "type": "Hôpital universitaire (Cairo Univ.)",
        "who_reliability": 88.0,
        "lat": 30.029, "lon": 31.213, "country": "EGY",
        "avg_load_kw": 4500, "max_load_kw": 6500,
        "has_solar": False, "has_generator": True,
        "grid_stability": "moyen",
        "data_source": "africa_grid",
        "ingest_geo": True,
    },
    "chuk_rwanda": {
        "name": "CHU de Kigali (CHUK)",
        "location": "Kigali, Rwanda",
        "flag": "🇷🇼",
        "beds": 519,
        "type": "Centre hospitalier universitaire",
        "who_reliability": 75.0,
        "lat": -1.954, "lon": 30.057, "country": "RWA",
        "avg_load_kw": 700, "max_load_kw": 1100,
        "has_solar": True, "has_generator": True,
        "grid_stability": "moyen",
        "data_source": "africa_grid",
        "ingest_geo": True,
    },
    # ── Hôpitaux NHS (source : ERIC 2022-23) ────────────────────────
    "st_thomas_nhs": {
        "name": "St Thomas' Hospital",
        "location": "London, Angleterre",
        "flag": "🇬🇧",
        "beds": 840,
        "type": "Acute NHS Trust (ERIC)",
        "who_reliability": 99.5,
        "lat": 51.4988, "lon": -0.1175, "country": "GBR",
        "avg_load_kw": 9361, "max_load_kw": 11863,
        "has_solar": True, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "eric", "eric_code": "rj121",
        "floor_area_m2": 150_000,
        "annual_electricity_kwh": 82_000_000,
        "ingest_geo": True,
    },
    "addenbrookes_nhs": {
        "name": "Addenbrooke's Hospital",
        "location": "Cambridge, Angleterre",
        "flag": "🇬🇧",
        "beds": 1000,
        "type": "Acute NHS Trust (ERIC)",
        "who_reliability": 99.5,
        "lat": 52.1753, "lon": 0.1405, "country": "GBR",
        "avg_load_kw": 8904, "max_load_kw": 11500,
        "has_solar": True, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "eric", "eric_code": "rgt01",
        "floor_area_m2": 160_000,
        "annual_electricity_kwh": 78_000_000,
        "ingest_geo": True,
    },
    "manchester_nhs": {
        "name": "Manchester Royal Infirmary",
        "location": "Manchester, Angleterre",
        "flag": "🇬🇧",
        "beds": 752,
        "type": "Acute NHS Trust (ERIC)",
        "who_reliability": 99.5,
        "lat": 53.4617, "lon": -2.2260, "country": "GBR",
        "avg_load_kw": 6621, "max_load_kw": 8500,
        "has_solar": False, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "eric", "eric_code": "r0a01",
        "floor_area_m2": 115_000,
        "annual_electricity_kwh": 58_000_000,
        "ingest_geo": True,
    },
    "kings_college_nhs": {
        "name": "King's College Hospital",
        "location": "London, Angleterre",
        "flag": "🇬🇧",
        "beds": 950,
        "type": "Acute NHS Trust (ERIC)",
        "who_reliability": 99.5,
        "lat": 51.4685, "lon": -0.0940, "country": "GBR",
        "avg_load_kw": 8219, "max_load_kw": 10500,
        "has_solar": True, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "eric", "eric_code": "rxh01",
        "floor_area_m2": 140_000,
        "annual_electricity_kwh": 72_000_000,
        "ingest_geo": True,
    },
    "john_radcliffe_nhs": {
        "name": "John Radcliffe Hospital",
        "location": "Oxford, Angleterre",
        "flag": "🇬🇧",
        "beds": 832,
        "type": "Acute NHS Trust (ERIC)",
        "who_reliability": 99.5,
        "lat": 51.7636, "lon": -1.2200, "country": "GBR",
        "avg_load_kw": 7078, "max_load_kw": 9000,
        "has_solar": True, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "eric", "eric_code": "rth01",
        "floor_area_m2": 120_000,
        "annual_electricity_kwh": 62_000_000,
        "ingest_geo": True,
    },
    "guys_nhs": {
        "name": "Guy's Hospital",
        "location": "London, Angleterre",
        "flag": "🇬🇧",
        "beds": 400,
        "type": "Acute NHS Trust (ERIC)",
        "who_reliability": 99.5,
        "lat": 51.5042, "lon": -0.0871, "country": "GBR",
        "avg_load_kw": 5479, "max_load_kw": 7000,
        "has_solar": False, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "eric", "eric_code": "rj122",
        "floor_area_m2": 82_000,
        "annual_electricity_kwh": 48_000_000,
        "ingest_geo": True,
    },
    "leeds_general_nhs": {
        "name": "Leeds General Infirmary",
        "location": "Leeds, Angleterre",
        "flag": "🇬🇧",
        "beds": 700,
        "type": "Acute NHS Trust (ERIC)",
        "who_reliability": 99.5,
        "lat": 53.8018, "lon": -1.5520, "country": "GBR",
        "avg_load_kw": 5936, "max_load_kw": 7600,
        "has_solar": False, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "eric", "eric_code": "rr801",
        "floor_area_m2": 100_000,
        "annual_electricity_kwh": 52_000_000,
        "ingest_geo": True,
    },
    "birmingham_heartlands_nhs": {
        "name": "Birmingham Heartlands Hospital",
        "location": "Birmingham, Angleterre",
        "flag": "🇬🇧",
        "beds": 660,
        "type": "Acute NHS Trust (ERIC)",
        "who_reliability": 99.5,
        "lat": 52.4636, "lon": -1.8220, "country": "GBR",
        "avg_load_kw": 5251, "max_load_kw": 6700,
        "has_solar": True, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "eric", "eric_code": "rq301",
        "floor_area_m2": 95_000,
        "annual_electricity_kwh": 46_000_000,
        "ingest_geo": True,
    },
    "newcastle_rvi_nhs": {
        "name": "Royal Victoria Infirmary",
        "location": "Newcastle, Angleterre",
        "flag": "🇬🇧",
        "beds": 900,
        "type": "Acute NHS Trust (ERIC)",
        "who_reliability": 99.5,
        "lat": 54.9802, "lon": -1.6196, "country": "GBR",
        "avg_load_kw": 7763, "max_load_kw": 9900,
        "has_solar": False, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "eric", "eric_code": "ra701",
        "floor_area_m2": 130_000,
        "annual_electricity_kwh": 68_000_000,
        "ingest_geo": True,
    },
    "royal_devon_nhs": {
        "name": "Royal Devon and Exeter Hospital",
        "location": "Exeter, Angleterre",
        "flag": "🇬🇧",
        "beds": 600,
        "type": "Acute NHS Trust (ERIC)",
        "who_reliability": 99.5,
        "lat": 50.7157, "lon": -3.5060, "country": "GBR",
        "avg_load_kw": 4338, "max_load_kw": 5500,
        "has_solar": True, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "eric", "eric_code": "ra401",
        "floor_area_m2": 80_000,
        "annual_electricity_kwh": 38_000_000,
        "ingest_geo": True,
    },
    "nyc_bellevue": {
        "name": "Bellevue Hospital Center",
        "location": "Manhattan, New York",
        "flag": "🇺🇸",
        "beds": 912,
        "type": "Public Acute (NYC H+H)",
        "who_reliability": 99.96,
        "lat": 40.7395, "lon": -73.9766, "country": "USA",
        # Coupures réelles comté (EAGLE-I) — plus local qu'une zone NY-ISO entière.
        "eaglei_county_key": "new_york_ny",
        "avg_load_kw": 6046, "max_load_kw": 7800,
        "has_solar": False, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "nyc_ll84", "nyc_code": "nyc_bellevue",
        "target_source": "county_network",
        "floor_area_m2": 211_475,
        "annual_electricity_kwh": 52_960_248,
        "ingest_geo": True,
    },
    "nyc_nyu_tisch": {
        "name": "NYU Langone Tisch Hospital",
        "location": "Manhattan, New York",
        "flag": "🇺🇸",
        "beds": 844,
        "type": "Private Acute (NYU Langone)",
        "who_reliability": 99.96,
        "lat": 40.7426, "lon": -73.9744, "country": "USA",
        "eaglei_county_key": "new_york_ny",
        "avg_load_kw": 5153, "max_load_kw": 6700,
        "has_solar": False, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "nyc_ll84", "nyc_code": "nyc_nyu_tisch",
        "target_source": "county_network",
        "floor_area_m2": 64_040,
        "annual_electricity_kwh": 45_139_152,
        "ingest_geo": True,
    },
    "nyc_nyp_brooklyn": {
        "name": "NewYork-Presbyterian Brooklyn Methodist",
        "location": "Brooklyn, New York",
        "flag": "🇺🇸",
        "beds": 1_001,
        "type": "Private Acute (NYP)",
        "who_reliability": 99.96,
        "lat": 40.6686, "lon": -73.9801, "country": "USA",
        "eaglei_county_key": "kings_ny",
        "avg_load_kw": 3698, "max_load_kw": 4800,
        "has_solar": False, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "nyc_ll84", "nyc_code": "nyc_nyp_brooklyn",
        "target_source": "county_network",
        "floor_area_m2": 126_587,
        "annual_electricity_kwh": 32_396_762,
        "ingest_geo": True,
    },
    "nyc_elmhurst": {
        "name": "Elmhurst Hospital Center",
        "location": "Queens, New York",
        "flag": "🇺🇸",
        "beds": 545,
        "type": "Public Acute (NYC H+H)",
        "who_reliability": 99.96,
        "lat": 40.7444, "lon": -73.8861, "country": "USA",
        "eaglei_county_key": "queens_ny",
        "avg_load_kw": 3483, "max_load_kw": 4500,
        "has_solar": False, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "nyc_ll84", "nyc_code": "nyc_elmhurst",
        "target_source": "county_network",
        "floor_area_m2": 89_366,
        "annual_electricity_kwh": 30_507_199,
        "ingest_geo": True,
    },
    "nyc_lincoln": {
        "name": "Lincoln Medical Center",
        "location": "Bronx, New York",
        "flag": "🇺🇸",
        "beds": 362,
        "type": "Public Acute (NYC H+H)",
        "who_reliability": 99.96,
        "lat": 40.8177, "lon": -73.9242, "country": "USA",
        "eaglei_county_key": "bronx_ny",
        "avg_load_kw": 3566, "max_load_kw": 4600,
        "has_solar": False, "has_generator": True,
        "grid_stability": "très stable",
        "data_source": "nyc_ll84", "nyc_code": "nyc_lincoln",
        "target_source": "county_network",
        "floor_area_m2": 110_874,
        "annual_electricity_kwh": 31_236_421,
        "ingest_geo": True,
    },
}


# ── Provenance de la cible `is_outage` par site ─────────────────────
# Honnêteté UX : distinguer EXPLICITEMENT d'où vient (ou non) l'étiquette de
# coupure d'un site, pour ne jamais présenter un score synthétique comme du
# terrain. Source de vérité unique, consommée par l'app (notes par site +
# panneau « sources »).
#   real      → coupures réellement observées (relevés terrain). Seul : Lacor.
#   synthetic → étiquettes `is_outage` SYNTHÉTIQUES générées à l'ingestion
#               (NHS ERIC, NYC LL84) : profils de charge réels mais coupures
#               simulées ⇒ scores indicatifs, pas des métriques terrain.
#   cloned    → aucune étiquette propre : le modèle Lacor est appliqué à un
#               profil de consommation cloné/redimensionné (hôpitaux
#               `africa_grid`) ⇒ score purement illustratif.
TARGET_SOURCE_META: dict[str, dict] = {
    "real": {
        "label": "Coupures réelles observées (terrain)",
        "emoji": "🎯", "color": "#2ecc71", "status": "primary",
        "detail": "is_outage = relevés horaires Lacor 2022",
    },
    "synthetic": {
        "label": "Charge réelle — coupures simulées",
        "emoji": "🧪", "color": "#f39c12", "status": "synthetic",
        "detail": "Profil de charge réel (NHS/NYC) ; coupures générées à l'ingestion",
    },
    "cloned": {
        "label": "Aucune coupure étiquetée (profil cloné Lacor)",
        "emoji": "♻️", "color": "#f39c12", "status": "synthetic",
        "detail": "Modèle Lacor appliqué à un profil de consommation cloné",
    },
    "county_network": {
        "label": "Coupures réseau comté (EAGLE-I, pas l'hôpital)",
        "emoji": "🗽", "color": "#3498db", "status": "primary",
        "detail": "Clients sans courant dans le comté/borough (LL84 = conso bâtiment)",
    },
}


def hospital_label(info: dict, *, with_location: bool = False) -> str:
    """Nom affiché avec le drapeau pays (emoji du catalogue)."""
    flag = (info.get("flag") or "").strip()
    name = info.get("name", "?")
    if with_location:
        loc = info.get("location", "")
        text = f"{name} — {loc}" if loc else name
    else:
        text = name
    return f"{flag} {text}".strip() if flag else text


def get_target_source(key: str, info: dict | None = None) -> str:
    """Provenance de la cible `is_outage` d'un site : real | synthetic | cloned.

    Source de vérité unique. Un override explicite `target_source` posé dans
    `HOSPITAL_DISPLAY` est respecté ; sinon la valeur est dérivée du
    `data_source` (lacor → real, eric → synthetic, nyc_ll84 → county_network,
    africa_grid → cloned). Défaut prudent : `synthetic`.
    """
    if info is None:
        info = HOSPITAL_DISPLAY.get(key, {})
    explicit = info.get("target_source")
    if explicit in TARGET_SOURCE_META:
        return explicit
    if key == "lacor_uganda":
        return "real"
    ds = info.get("data_source")
    if ds == "nyc_ll84":
        return "county_network"
    if ds == "eric":
        return "synthetic"
    if ds == "africa_grid":
        return "cloned"
    return "synthetic"


def build_hospital_locations() -> dict[str, dict]:
    """Sous-ensemble géolocalisé : sites pour lesquels on lance les
    pipelines d'ingestion géo (Open-Meteo archive ; Forecast/EM optionnels).

    Marqués par `ingest_geo: True` dans `HOSPITAL_DISPLAY`. Reproduit
    fidèlement l'ancien dict statique de `config.py`, mais maintenant
    dérivé pour rester en sync avec le catalogue UI.
    """
    return {
        key: {
            "lat": info["lat"],
            "lon": info["lon"],
            "country": info.get("country", ""),
        }
        for key, info in HOSPITAL_DISPLAY.items()
        if info.get("ingest_geo")
    }


# Pour la résolution Electricity Maps zone réseau : conservé statique
# car certaines zones ne suivent pas le code ISO2 du pays (US-SW-AZPS,
# US-NY-NYIS…). Le mapping reste dans `config.py`.
